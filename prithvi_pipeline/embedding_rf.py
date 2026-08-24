from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.ensemble import ExtraTreesClassifier

from .data import (
    PATCH_SIZE,
    CaseData,
    EncodedPatch,
    create_nws_training_label,
    discover_cases,
    encode_training_patches,
    validate_prithvi_bands,
    window_batches,
)
from .model import encode_prithvi, load_prithvi_encoder, normalize_prithvi
from .pipeline import _write_raster, binary_metrics, process_prediction, set_seed


def token_dataset(patches: list[EncodedPatch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    groups = []
    for patch in patches:
        auxiliary = torch.nn.functional.avg_pool2d(patch.auxiliary.float().unsqueeze(0), 16).squeeze(0)
        target = torch.nn.functional.avg_pool2d(patch.target.float().unsqueeze(0), 16).squeeze(0).squeeze(0)
        valid = torch.nn.functional.avg_pool2d(patch.valid.float().unsqueeze(0), 16).squeeze(0).squeeze(0)
        stacked = torch.cat([patch.features.float(), auxiliary], dim=0).permute(1, 2, 0).reshape(-1, 392)
        keep = valid.reshape(-1) >= 0.80
        features.append(stacked[keep].numpy())
        targets.append((target.reshape(-1)[keep] >= 0.12).numpy().astype("uint8"))
        groups.append(np.full(int(keep.sum()), patch.case_id, dtype=object))
    return np.concatenate(features), np.concatenate(targets), np.concatenate(groups)


def train_embedding_forest(patches: list[EncodedPatch], seed: int) -> ExtraTreesClassifier:
    x, y, _ = token_dataset(patches)
    if len(np.unique(y)) < 2:
        raise ValueError("Both damage and non-damage tokens are required.")
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def infer_embedding_forest(
    encoder: torch.nn.Module,
    model: ExtraTreesClassifier,
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
    for locations, images, auxiliary, valids in window_batches(case, batch_size, stride):
        image_tensor = normalize_prithvi(torch.from_numpy(images).to(device))
        with torch.inference_mode():
            encoded = encode_prithvi(encoder, image_tensor).cpu().float()
        auxiliary_tensor = torch.from_numpy(auxiliary).float()
        pooled = torch.nn.functional.avg_pool2d(auxiliary_tensor, 16)
        x = torch.cat([encoded, pooled], dim=1).permute(0, 2, 3, 1).reshape(-1, 392).numpy()
        token_probability = model.predict_proba(x)[:, 1].reshape(len(locations), 1, 14, 14)
        probability_patches = torch.nn.functional.interpolate(
            torch.from_numpy(token_probability).float(),
            size=(PATCH_SIZE, PATCH_SIZE),
            mode="bilinear",
            align_corners=False,
        ).numpy()[:, 0]
        for (top, left), probability, valid in zip(locations, probability_patches, valids):
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
    return probability


def run_embedding_forest_pipeline(
    source: Path,
    shapefiles: Path,
    aligned_root: Path,
    output: Path,
    vendor_dir: Path,
    patches_per_class: int = 36,
    batch_size: int = 8,
    stride: int = 160,
    seed: int = 42,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cpu")
    set_seed(seed)
    cases = discover_cases(source, aligned_root, output, shapefiles)
    cases = [
        case
        for case in cases
        if validate_prithvi_bands(case.before, case.geospatial_metadata.get("band_descriptions"))["usable"]
    ]
    encoder, encoder_metadata = load_prithvi_encoder(vendor_dir, num_frames=2)
    labels: dict[str, dict[str, Any]] = {}
    patches: list[EncodedPatch] = []
    for case in cases:
        label = create_nws_training_label(case, output)
        if label is None:
            continue
        labels[case.case_id] = label
        patches.extend(encode_training_patches(encoder, case, label, patches_per_class, batch_size, seed, device))
    labelled_ids = sorted(labels, key=lambda value: int(re.sub(r"\D", "", value)))
    if len(labelled_ids) < 2:
        raise RuntimeError("At least two labelled cases are required.")

    fold_probabilities: dict[str, np.ndarray] = {}
    for fold_index, held_out in enumerate(labelled_ids):
        model = train_embedding_forest([patch for patch in patches if patch.case_id != held_out], seed + fold_index)
        case = next(item for item in cases if item.case_id == held_out)
        probability = infer_embedding_forest(encoder, model, case, device, batch_size, stride)
        fold_probabilities[held_out] = probability
        with rasterio.open(case.before) as src:
            profile = src.profile.copy()
        _write_raster(output / "evaluation" / held_out / "leave_one_case_out_probability.tif", probability, profile, "float32")

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    threshold_rows = []
    for threshold in thresholds:
        for case_id, probability in fold_probabilities.items():
            case = next(item for item in cases if item.case_id == case_id)
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
        case = next(item for item in cases if item.case_id == case_id)
        with rasterio.open(case.valid_mask) as src:
            valid = src.read(1) > 0
        cv_rows.append(
            {
                "case_id": case_id,
                "evaluation_role": "leave-one-tornado-out",
                "training_cases": ";".join(value for value in labelled_ids if value != case_id),
                **binary_metrics(probability, labels[case_id]["label"], valid, selected_threshold),
            }
        )
        process_prediction(
            case,
            probability,
            selected_threshold,
            labels[case_id],
            output / "leave_one_out",
            "leave-one-tornado-out",
            "Prithvi frozen embeddings + balanced Extra Trees token classifier",
        )
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(reports / "cross_validation_results.csv", index=False)

    final_model = train_embedding_forest(patches, seed + 100)
    model_dir = output / "models" / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / "prithvi_embedding_extra_trees.joblib", compress=3)
    metadata = {
        **encoder_metadata,
        "model": "Prithvi frozen embeddings + balanced Extra Trees token classifier",
        "training_cases": labelled_ids,
        "selected_threshold": selected_threshold,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "validation": "leave-one-tornado-out; no independent final test set",
        "known_limitations": [
            "Only four event-specific NWS-labelled cases are available.",
            "Three labelled cases share one outbreak and similar geography.",
            "Unlabelled outputs are review candidates, not verified tornado paths.",
        ],
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    final_rows = []
    for case in cases:
        probability = infer_embedding_forest(encoder, final_model, case, device, batch_size, stride)
        role = "training-reference comparison (not independent)" if case.case_id in labels else "unlabelled inference; independent validation required"
        final_rows.append(
            process_prediction(
                case,
                probability,
                selected_threshold,
                labels.get(case.case_id),
                output,
                role,
                "Prithvi frozen embeddings + balanced Extra Trees token classifier",
            )
        )
    pd.DataFrame(final_rows).to_csv(reports / "current_dataset_results.csv", index=False)
    summary = {
        "cases_processed": len(cases),
        "training_cases": labelled_ids,
        "selected_threshold": selected_threshold,
        "cross_validation_macro": cv_frame[["dice", "iou", "precision", "recall", "f1"]].mean().to_dict(),
        "output": str(output),
    }
    (reports / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
