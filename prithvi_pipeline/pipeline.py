from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tornado_prithvi_mpl")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import LineString, shape

import run_rf_path_workflow as legacy
from .data import (
    PATCH_SIZE,
    CaseData,
    EncodedPatch,
    auxiliary_features,
    change_magnitude,
    create_nws_training_label,
    discover_cases,
    encode_training_patches,
    validate_prithvi_bands,
    window_batches,
)
from .model import PrithviSegmentationHead, encode_prithvi, load_prithvi_encoder, normalize_prithvi


PALETTE = {
    "red": "#FF1F1F",
    "cyan": "#00D9FF",
    "orange": "#FF9F1C",
    "ink": "#16221D",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    binary = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    binary = (binary * valid).sum() / valid.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits) * valid
    target_valid = target * valid
    intersection = (probability * target_valid).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target_valid.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return binary + dice


def train_head(
    patches: list[EncodedPatch],
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[PrithviSegmentationHead, list[dict[str, float]]]:
    if not patches:
        raise ValueError("No encoded training patches were supplied.")
    set_seed(seed)
    head = PrithviSegmentationHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    positive = sum(float((patch.target * patch.valid).sum()) for patch in patches)
    valid_count = sum(float(patch.valid.sum()) for patch in patches)
    ratio = np.clip((valid_count - positive) / max(positive, 1.0), 1.0, 8.0)
    pos_weight = torch.tensor([ratio], dtype=torch.float32, device=device).view(1, 1, 1, 1)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        head.train()
        order = rng.permutation(len(patches))
        losses = []
        for start in range(0, len(order), batch_size):
            selected = [patches[index] for index in order[start : start + batch_size]]
            features = torch.stack([item.features for item in selected]).float()
            auxiliary = torch.stack([item.auxiliary for item in selected]).float()
            target = torch.stack([item.target for item in selected]).float()
            valid = torch.stack([item.valid for item in selected]).float()
            if rng.random() < 0.5:
                features = torch.flip(features, dims=(-1,))
                auxiliary = torch.flip(auxiliary, dims=(-1,))
                target = torch.flip(target, dims=(-1,))
                valid = torch.flip(valid, dims=(-1,))
            if rng.random() < 0.5:
                features = torch.flip(features, dims=(-2,))
                auxiliary = torch.flip(auxiliary, dims=(-2,))
                target = torch.flip(target, dims=(-2,))
                valid = torch.flip(valid, dims=(-2,))
            features = features.to(device)
            auxiliary = auxiliary.to(device)
            target = target.to(device)
            valid = valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features, auxiliary, valid)
            loss = _loss(logits, target, valid, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    head.eval()
    return head, history


def infer_probability(
    encoder: torch.nn.Module,
    head: PrithviSegmentationHead,
    case: CaseData,
    device: torch.device,
    batch_size: int,
    stride: int,
) -> np.ndarray:
    with rasterio.open(case.before) as src:
        height, width = src.height, src.width
    accumulated = np.zeros((height, width), dtype="float32")
    weights = np.zeros((height, width), dtype="float32")
    axis_weight = np.clip(np.hanning(PATCH_SIZE).astype("float32"), 0.08, 1.0)
    blend = np.outer(axis_weight, axis_weight).astype("float32")
    encoder.to(device).eval()
    head.to(device).eval()
    for locations, images, auxiliary, valids in window_batches(case, batch_size, stride):
        image_tensor = normalize_prithvi(torch.from_numpy(images).to(device))
        auxiliary_tensor = torch.from_numpy(auxiliary).to(device)
        valid_tensor = torch.from_numpy(valids.astype("float32")).unsqueeze(1).to(device)
        with torch.inference_mode():
            features = encode_prithvi(encoder, image_tensor)
            probabilities = torch.sigmoid(head(features, auxiliary_tensor, valid_tensor)).cpu().numpy()[:, 0]
        for (top, left), probability, valid in zip(locations, probabilities, valids):
            bottom = min(top + PATCH_SIZE, height)
            right = min(left + PATCH_SIZE, width)
            patch_height = bottom - top
            patch_width = right - left
            local_weight = blend[:patch_height, :patch_width] * valid[:patch_height, :patch_width]
            accumulated[top:bottom, left:right] += probability[:patch_height, :patch_width] * local_weight
            weights[top:bottom, left:right] += local_weight
    probability = np.divide(accumulated, weights, out=np.zeros_like(accumulated), where=weights > 0)
    with rasterio.open(case.valid_mask) as src:
        valid = src.read(1) > 0
    probability[~valid] = 0.0
    encoder.to("cpu")
    head.to("cpu")
    return probability


def binary_metrics(probability: np.ndarray, target: np.ndarray, valid: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold) & valid
    truth = target & valid
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(dice),
        "false_negative_rate": float(fn / max(tp + fn, 1)),
    }


def _component_properties(points: np.ndarray) -> dict[str, float]:
    if len(points) < 2:
        return {"elongation": 0.0, "length_scale": 0.0}
    if len(points) > 100000:
        points = points[np.linspace(0, len(points) - 1, 100000).astype(int)]
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    return {
        "elongation": float(np.sqrt((eigenvalues[-1] + 1.0) / (eigenvalues[0] + 1.0))),
        "length_scale": float(np.sqrt(eigenvalues[-1] + 1.0)),
    }


def select_corridor(
    probability: np.ndarray,
    valid: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = valid & (probability >= threshold)
    raw = ndi.binary_opening(raw, structure=np.ones((2, 2), dtype=bool))
    raw = ndi.binary_closing(raw, structure=np.ones((7, 7), dtype=bool))
    labels, count = ndi.label(raw, structure=np.ones((3, 3), dtype="uint8"))
    if count == 0:
        return np.zeros_like(raw), {"reason": "no connected components", "candidates": []}
    magnitude_values = magnitude[valid]
    magnitude_scale = float(np.nanpercentile(magnitude_values, 90)) if magnitude_values.size else 1.0
    minimum = max(100, int(valid.sum() * 0.00005))
    maximum = max(minimum + 1, int(valid.sum() * 0.12))
    candidates = []
    for label_id, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None:
            continue
        local = labels[bounds] == label_id
        area = int(local.sum())
        if area < minimum or area > maximum:
            continue
        points = np.argwhere(local) + np.asarray([axis.start for axis in bounds])
        props = _component_properties(points)
        if props["elongation"] < 1.8 or props["length_scale"] < 12:
            continue
        local_probability = float(np.mean(probability[bounds][local]))
        local_change = float(np.mean(magnitude[bounds][local]) / max(magnitude_scale, 1e-6))
        score = (
            local_probability
            * math.log1p(area)
            * min(props["elongation"], 20.0)
            * math.sqrt(props["length_scale"])
            * (0.65 + 0.35 * min(local_change, 2.0))
        )
        candidates.append(
            {
                "label": label_id,
                "area": area,
                "mean_probability": local_probability,
                "mean_change_ratio": local_change,
                "elongation": props["elongation"],
                "length_scale": props["length_scale"],
                "score": float(score),
            }
        )
    if not candidates:
        return np.zeros_like(raw), {"reason": "no plausible elongated component", "candidates": []}
    selected = max(candidates, key=lambda item: item["score"])
    return labels == selected["label"], {"reason": "selected highest-scoring imagery component", "selected": selected, "candidates": candidates}


def _line_pixel_length(line: LineString | None, transform) -> float | None:
    if line is None:
        return None
    pixel_size = max(abs(transform.a), abs(transform.e))
    return float(line.length / max(pixel_size, 1e-12))


def path_quality(
    corridor: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    line: LineString | None,
    transform,
    threshold: float,
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    points = np.argwhere(corridor)
    props = _component_properties(points)
    area_fraction = float(corridor.sum() / max(valid.sum(), 1))
    mean_probability = float(np.mean(probability[corridor])) if corridor.any() else 0.0
    tortuosity = legacy.line_tortuosity(line)
    pixel_length = _line_pixel_length(line, transform)
    if line is None:
        reasons.append("no continuous skeleton centerline")
    if area_fraction <= 0 or area_fraction > 0.12:
        reasons.append("corridor area is implausible")
    if props["elongation"] < 2.2:
        reasons.append("candidate is not sufficiently elongated")
    if mean_probability < threshold + 0.02:
        reasons.append("mean model confidence is too close to threshold")
    if tortuosity is not None and tortuosity > 2.5:
        reasons.append("centerline is excessively tortuous")
    if pixel_length is None or pixel_length < 30:
        reasons.append("centerline is too short")
    metrics = {
        "area_fraction": area_fraction,
        "mean_probability": mean_probability,
        "elongation": props["elongation"],
        "path_tortuosity": tortuosity,
        "path_length_pixels": pixel_length,
    }
    return not reasons, reasons, metrics


def nws_comparison(
    corridor: np.ndarray,
    line: LineString | None,
    label_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if label_info is None:
        return {"nws_available": False, "nws_path_overlap_percent": None, "mean_nws_distance_pixels": None}
    centerline = label_info["centerline"]
    if not corridor.any() or line is None or not centerline.any():
        return {"nws_available": True, "nws_path_overlap_percent": 0.0, "mean_nws_distance_pixels": None}
    expanded = ndi.binary_dilation(corridor, iterations=3)
    overlap = float(np.sum(centerline & expanded) / max(centerline.sum(), 1))
    distance = ndi.distance_transform_edt(~corridor)
    mean_distance = float(np.mean(distance[centerline]))
    return {
        "nws_available": True,
        "nws_path_overlap_percent": 100.0 * overlap,
        "mean_nws_distance_pixels": mean_distance,
    }


def _write_raster(path: Path, array: np.ndarray, profile: dict[str, Any], dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = profile.copy()
    updated.update(count=1, dtype=dtype, nodata=0, compress="deflate", BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **updated) as dst:
        dst.write(array.astype(dtype), 1)


def _write_geometry_products(case_dir: Path, corridor: np.ndarray, line: LineString | None, profile: dict[str, Any]) -> None:
    polygons = [
        shape(geometry)
        for geometry, value in shapes(corridor.astype("uint8"), mask=corridor, transform=profile["transform"])
        if value == 1
    ]
    gpd.GeoDataFrame({"class": ["prithvi_damage_candidate"] * len(polygons)}, geometry=polygons, crs=profile["crs"]).to_file(
        case_dir / "predicted_damage_corridor.geojson", driver="GeoJSON"
    )
    if line is None:
        frame = gpd.GeoDataFrame({"class": []}, geometry=[], crs=profile["crs"])
    else:
        frame = gpd.GeoDataFrame({"class": ["prithvi_candidate_centerline"]}, geometry=[line], crs=profile["crs"])
    frame.to_file(case_dir / "predicted_path_centerline.geojson", driver="GeoJSON")


def _plot_line(axis, line: LineString, transform, step: int, color: str, label: str, linestyle: str = "-") -> None:
    xs, ys = line.xy
    rows, columns = rasterio.transform.rowcol(transform, xs, ys)
    artist = axis.plot(np.asarray(columns) / step, np.asarray(rows) / step, color=color, lw=4.0, linestyle=linestyle, label=label)[0]
    artist.set_path_effects([path_effects.Stroke(linewidth=7.0, foreground="white"), path_effects.Normal()])


def create_case_figures(
    case: CaseData,
    probability: np.ndarray,
    corridor: np.ndarray,
    published_line: LineString | None,
    candidate_line: LineString | None,
    label_info: dict[str, Any] | None,
    status: str,
    case_dir: Path,
) -> None:
    after = legacy.rgb(case.after, [3, 2, 1])
    with rasterio.open(case.after) as src:
        transform, crs = src.transform, src.crs
    step = legacy.display_scale(after.shape[:2], 1800)
    after_show = legacy.decimate(after, step)
    probability_show = legacy.decimate(probability, step)
    corridor_show = legacy.decimate(corridor, step)
    figure_height = float(np.clip(16.0 * after_show.shape[0] / max(after_show.shape[1], 1) + 1.0, 5.5, 10.0))

    fig, axis = plt.subplots(figsize=(16, figure_height), constrained_layout=True)
    axis.imshow(after_show)
    heat = axis.imshow(np.ma.masked_where(probability_show <= 0.05, probability_show), cmap="magma", alpha=0.72, vmin=0, vmax=1)
    axis.set_title(f"{case.case_id}: Prithvi tornado-damage probability", fontsize=18, fontweight="bold")
    axis.set_axis_off()
    fig.colorbar(heat, ax=axis, fraction=0.025, pad=0.012, label="P(damage)")
    fig.savefig(case_dir / "prithvi_probability_map.png", dpi=190, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(16, figure_height), constrained_layout=True)
    axis.imshow(after_show)
    if corridor.any():
        color = PALETTE["red"] if published_line is not None else PALETTE["orange"]
        axis.imshow(np.ma.masked_where(~corridor_show, corridor_show), cmap=matplotlib.colors.ListedColormap([color]), alpha=0.30)
    if label_info is not None and case.nws_path is not None:
        legacy.plot_nws(axis, case.nws_path, crs, transform, step)
    if published_line is not None:
        _plot_line(axis, published_line, transform, step, PALETTE["red"], "Prithvi predicted centerline")
    elif candidate_line is not None:
        _plot_line(axis, candidate_line, transform, step, PALETTE["orange"], "Rejected model candidate", "--")
    axis.text(
        0.015,
        0.975,
        status,
        transform=axis.transAxes,
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#9B1C1C" if status.startswith("Rejected") else PALETTE["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#C9D2CC", "alpha": 0.93},
    )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="lower left", fontsize=10, framealpha=0.93)
    axis.set_axis_off()
    fig.savefig(case_dir / "final_path_map.png", dpi=210, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    if label_info is not None:
        ground_truth = legacy.decimate(label_info["label"], step)
        fig, axes = plt.subplots(1, 2, figsize=(16, figure_height), constrained_layout=True)
        axes[0].imshow(after_show)
        axes[0].imshow(np.ma.masked_where(~ground_truth, ground_truth), cmap=matplotlib.colors.ListedColormap([PALETTE["cyan"]]), alpha=0.35)
        axes[0].set_title("Official NWS reference corridor", fontsize=15, fontweight="bold")
        axes[1].imshow(after_show)
        axes[1].imshow(np.ma.masked_where(~corridor_show, corridor_show), cmap=matplotlib.colors.ListedColormap([PALETTE["red"]]), alpha=0.35)
        axes[1].set_title("Imagery-only Prithvi prediction", fontsize=15, fontweight="bold")
        for axis in axes:
            axis.set_axis_off()
        fig.savefig(case_dir / "prediction_vs_nws.png", dpi=190, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)


def process_prediction(
    case: CaseData,
    probability: np.ndarray,
    threshold: float,
    label_info: dict[str, Any] | None,
    output_dir: Path,
    evaluation_role: str,
    model_name: str = "Prithvi-EO-2.0-tiny-TL frozen encoder + trained segmentation head",
) -> dict[str, Any]:
    case_dir = output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(case.before) as src:
        profile = src.profile.copy()
    with rasterio.open(case.valid_mask) as src:
        valid = src.read(1) > 0
    magnitude = change_magnitude(case)
    corridor, selection = select_corridor(probability, valid, magnitude, threshold)
    candidate_line = legacy.centerline_from_mask(corridor, profile["transform"]) if corridor.any() else None
    passes, reasons, quality_metrics = path_quality(corridor, probability, valid, candidate_line, profile["transform"], threshold)
    comparison = nws_comparison(corridor, candidate_line, label_info)
    if label_info is not None and passes:
        overlap = comparison["nws_path_overlap_percent"] or 0.0
        distance = comparison["mean_nws_distance_pixels"]
        if overlap < 35.0 or distance is None or distance > 20.0:
            passes = False
            reasons.append("imagery prediction does not agree sufficiently with the independent NWS reference")
    if passes and label_info is not None:
        status = "NWS-consistent training-reference result"
    elif passes:
        status = "Unverified imagery-only candidate"
    else:
        status = "Rejected: " + ("; ".join(reasons) if reasons else "failed quality control")
    published_line = candidate_line if passes else None

    _write_raster(case_dir / "predicted_probability.tif", probability, profile, "float32")
    _write_raster(case_dir / "predicted_damage_mask.tif", corridor.astype("uint8"), profile, "uint8")
    _write_geometry_products(case_dir, corridor, published_line, profile)
    create_case_figures(case, probability, corridor, published_line, candidate_line, label_info, status, case_dir)

    metrics = {
        "case_id": case.case_id,
        "model": model_name,
        "evaluation_role": evaluation_role,
        "threshold": threshold,
        "status": status,
        "published_centerline": published_line is not None,
        "quality_reasons": reasons,
        "selection": selection,
        **quality_metrics,
        **comparison,
    }
    if label_info is not None:
        metrics.update(binary_metrics(probability, label_info["label"], valid, threshold))
    (case_dir / "case_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _save_head(head: PrithviSegmentationHead, model_dir: Path, metadata: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), model_dir / "prithvi_segmentation_head.pt")
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))


def run_prithvi_pipeline(
    source: Path,
    shapefiles: Path,
    aligned_root: Path,
    output: Path,
    vendor_dir: Path,
    epochs: int = 18,
    patches_per_class: int = 36,
    batch_size: int = 8,
    stride: int = 160,
    seed: int = 42,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    set_seed(seed)

    cases = discover_cases(source, aligned_root, output, shapefiles)
    inventory_rows = []
    usable_cases = []
    for case in cases:
        validation = validate_prithvi_bands(case.before, case.geospatial_metadata.get("band_descriptions"))
        inventory_rows.append(
            {
                "case_id": case.case_id,
                "before": str(case.before),
                "after": str(case.after),
                "nws_path": str(case.nws_path) if case.nws_path else "",
                "valid_fraction": case.geospatial_metadata.get("valid_fraction"),
                **validation,
            }
        )
        if validation["usable"]:
            usable_cases.append(case)
    pd.DataFrame(inventory_rows).to_csv(reports / "dataset_inventory.csv", index=False)
    if not usable_cases:
        raise RuntimeError("No cases have the six Landsat bands required by Prithvi.")

    encoder, encoder_metadata = load_prithvi_encoder(vendor_dir, num_frames=2)
    labels: dict[str, dict[str, Any]] = {}
    label_rows = []
    encoded_patches: list[EncodedPatch] = []
    for case in usable_cases:
        label = create_nws_training_label(case, output)
        if label is None:
            label_rows.append({"case_id": case.case_id, "label_available": False, "quality": "Rejected", "reason": "no safe event-specific NWS geometry"})
            continue
        labels[case.case_id] = label
        label_rows.append({key: value for key, value in label.items() if not isinstance(value, np.ndarray)} | {"label_available": True})
        encoded_patches.extend(
            encode_training_patches(encoder, case, label, patches_per_class, batch_size, seed, device)
        )
    pd.DataFrame(label_rows).to_csv(reports / "label_quality_report.csv", index=False)
    labelled_ids = sorted(labels, key=lambda value: int(re.sub(r"\D", "", value)))
    if len(labelled_ids) < 2:
        raise RuntimeError("Prithvi training requires at least two independently labelled tornado cases.")

    fold_probabilities: dict[str, np.ndarray] = {}
    fold_histories = []
    for fold_index, held_out in enumerate(labelled_ids):
        train_patches = [patch for patch in encoded_patches if patch.case_id != held_out]
        head, history = train_head(train_patches, epochs, batch_size, seed + fold_index, device)
        case = next(item for item in usable_cases if item.case_id == held_out)
        probability = infer_probability(encoder, head, case, device, batch_size, stride)
        fold_probabilities[held_out] = probability
        evaluation_dir = output / "evaluation" / held_out
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(case.before) as src:
            profile = src.profile.copy()
        _write_raster(evaluation_dir / "leave_one_case_out_probability.tif", probability, profile, "float32")
        for row in history:
            fold_histories.append({"held_out_case": held_out, **row})
    pd.DataFrame(fold_histories).to_csv(reports / "training_history.csv", index=False)

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    threshold_rows = []
    for threshold in thresholds:
        for case_id, probability in fold_probabilities.items():
            case = next(item for item in usable_cases if item.case_id == case_id)
            with rasterio.open(case.valid_mask) as src:
                valid = src.read(1) > 0
            threshold_rows.append({"case_id": case_id, **binary_metrics(probability, labels[case_id]["label"], valid, threshold)})
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(reports / "threshold_analysis.csv", index=False)
    macro = threshold_frame.groupby("threshold")[["dice", "iou", "precision", "recall"]].mean().reset_index()
    selected_threshold = float(macro.sort_values(["dice", "iou"], ascending=False).iloc[0]["threshold"])
    macro.to_csv(reports / "threshold_macro_summary.csv", index=False)

    cv_rows = []
    for case_id, probability in fold_probabilities.items():
        case = next(item for item in usable_cases if item.case_id == case_id)
        with rasterio.open(case.valid_mask) as src:
            valid = src.read(1) > 0
        row = {
            "case_id": case_id,
            "evaluation_role": "leave-one-tornado-out",
            "training_cases": ";".join(value for value in labelled_ids if value != case_id),
            **binary_metrics(probability, labels[case_id]["label"], valid, selected_threshold),
        }
        cv_rows.append(row)
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(reports / "cross_validation_results.csv", index=False)
    for case_id, probability in fold_probabilities.items():
        case = next(item for item in usable_cases if item.case_id == case_id)
        process_prediction(
            case,
            probability,
            selected_threshold,
            labels[case_id],
            output / "leave_one_out",
            "leave-one-tornado-out",
        )

    final_head, final_history = train_head(encoded_patches, epochs, batch_size, seed + 100, device)
    model_metadata = {
        **encoder_metadata,
        "segmentation_head": "convolutional decoder with fine-scale spectral-change fusion",
        "encoder_frozen": True,
        "training_cases": labelled_ids,
        "unlabelled_cases": [case.case_id for case in usable_cases if case.case_id not in labels],
        "validation": "leave-one-tornado-out; no independent final test set",
        "selected_threshold": selected_threshold,
        "patch_size": PATCH_SIZE,
        "stride": stride,
        "training_patches": len(encoded_patches),
        "epochs": epochs,
        "random_seed": seed,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "known_limitations": [
            "Only four event-specific NWS-labelled cases are available.",
            "TOR10, TOR12, and TOR13 share the 2011-04-27 outbreak, limiting event diversity.",
            "NWS widths are maximum-width corridor proxies, not dense pixel-perfect damage masks.",
            "Unlabelled-case centerlines are candidates and require independent review.",
        ],
    }
    _save_head(final_head, output / "models" / "final_model", model_metadata)
    pd.DataFrame(final_history).to_csv(output / "models" / "final_model" / "training_history.csv", index=False)

    final_rows = []
    for case in usable_cases:
        probability = infer_probability(encoder, final_head, case, device, batch_size, stride)
        role = "training-reference comparison (not independent)" if case.case_id in labels else "unlabelled inference; independent validation required"
        final_rows.append(process_prediction(case, probability, selected_threshold, labels.get(case.case_id), output, role))
    pd.DataFrame(final_rows).to_csv(reports / "current_dataset_results.csv", index=False)

    summary = {
        "cases_discovered": len(cases),
        "cases_processed": len(usable_cases),
        "training_cases": labelled_ids,
        "selected_threshold": selected_threshold,
        "cross_validation_macro": cv_frame[["dice", "iou", "precision", "recall", "f1"]].mean().to_dict(),
        "output": str(output),
    }
    (reports / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
