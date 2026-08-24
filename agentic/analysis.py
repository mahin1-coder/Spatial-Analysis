from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tornado_agentic_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy import ndimage as ndi
from shapely.geometry import LineString, shape
from sklearn.cluster import KMeans

import run_rf_path_workflow as wf


def resolve_band_mapping(descriptions: tuple[str | None, ...]) -> dict[str, int] | None:
    normalized = {str(name).upper(): index for index, name in enumerate(descriptions) if name}
    if "SR_B3" in normalized and "SR_B4" in normalized:
        return {
            "blue": normalized.get("SR_B1", normalized.get("SR_B2", 0)),
            "green": normalized.get("SR_B2", 1),
            "red": normalized["SR_B3"],
            "nir": normalized["SR_B4"],
            "swir1": normalized.get("SR_B5", normalized["SR_B4"]),
            "swir2": normalized.get("SR_B7", normalized.get("SR_B5", normalized["SR_B4"])),
        }
    return None


def _preview_shape(height: int, width: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def load_preview(
    before_path: Path,
    after_path: Path,
    max_side: int = 1100,
    descriptions_override: tuple[str | None, ...] | None = None,
) -> dict[str, Any]:
    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        height, width = _preview_shape(before.height, before.width, max_side)
        out_shape = (before.count, height, width)
        before_arr = before.read(out_shape=out_shape, masked=True, resampling=Resampling.average).astype("float32")
        after_arr = after.read(out_shape=out_shape, masked=True, resampling=Resampling.average).astype("float32")
        b = np.ma.filled(before_arr, np.nan)
        a = np.ma.filled(after_arr, np.nan)
        valid = np.all(np.isfinite(b) & np.isfinite(a), axis=0)
        transform = before.transform * Affine.scale(before.width / width, before.height / height)
        descriptions = before.descriptions
        if descriptions_override and not any(descriptions):
            descriptions = tuple(descriptions_override[: before.count])
        return {
            "before": b,
            "after": a,
            "valid": valid,
            "transform": transform,
            "crs": before.crs,
            "profile": before.profile.copy(),
            "descriptions": descriptions,
            "full_shape": (before.height, before.width),
        }


def robust_norm(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(stack, dtype="float32")
    for band in range(stack.shape[0]):
        values = stack[band][valid & np.isfinite(stack[band])]
        if values.size < 10:
            continue
        low, high = np.nanpercentile(values, [2, 98])
        result[band] = np.clip((stack[band] - low) / (high - low + 1e-8), 0, 1)
    return np.nan_to_num(result)


def rgb_image(stack: np.ndarray, valid: np.ndarray, mapping: dict[str, int] | None) -> np.ndarray:
    normalized = robust_norm(stack, valid)
    if mapping:
        indices = [mapping["red"], mapping["green"], mapping["blue"]]
    else:
        indices = list(range(min(3, stack.shape[0])))
    while len(indices) < 3:
        indices.append(indices[-1])
    image = np.moveaxis(normalized[indices[:3]], 0, -1)
    image[~valid] = 1.0
    return image


def false_color_image(stack: np.ndarray, valid: np.ndarray, indices: list[int]) -> np.ndarray:
    normalized = robust_norm(stack, valid)
    image = np.moveaxis(normalized[indices], 0, -1)
    image[~valid] = 1.0
    return image


def ndvi(stack: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    red = stack[mapping["red"]]
    nir = stack[mapping["nir"]]
    result = (nir - red) / (nir + red + 1e-8)
    return np.clip(result, -1, 1).astype("float32")


def stream_band_statistics(
    before_path: Path,
    after_path: Path,
    descriptions_override: tuple[str | None, ...] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        for band in range(1, min(before.count, after.count) + 1):
            accumulators = {
                "before": {"count": 0, "sum": 0.0, "sum2": 0.0, "min": math.inf, "max": -math.inf, "invalid": 0},
                "after": {"count": 0, "sum": 0.0, "sum2": 0.0, "min": math.inf, "max": -math.inf, "invalid": 0},
            }
            total = 0
            for _, window in before.block_windows(band):
                arrays = {
                    "before": before.read(band, window=window, masked=True),
                    "after": after.read(band, window=window, masked=True),
                }
                total += int(window.width * window.height)
                for period, array in arrays.items():
                    values = array.compressed().astype("float64")
                    values = values[np.isfinite(values)]
                    acc = accumulators[period]
                    acc["invalid"] += int(array.size - values.size)
                    if not values.size:
                        continue
                    acc["count"] += int(values.size)
                    acc["sum"] += float(values.sum())
                    acc["sum2"] += float(np.square(values).sum())
                    acc["min"] = min(acc["min"], float(values.min()))
                    acc["max"] = max(acc["max"], float(values.max()))
            fallback = descriptions_override[band - 1] if descriptions_override and band <= len(descriptions_override) else None
            description = before.descriptions[band - 1] or fallback or f"Band {band}"
            row: dict[str, Any] = {"band": band, "description": description}
            for period, acc in accumulators.items():
                count = max(int(acc["count"]), 1)
                mean = float(acc["sum"] / count)
                variance = max(0.0, float(acc["sum2"] / count - mean * mean))
                row.update(
                    {
                        f"{period}_mean": mean,
                        f"{period}_variance": variance,
                        f"{period}_min": None if not np.isfinite(acc["min"]) else float(acc["min"]),
                        f"{period}_max": None if not np.isfinite(acc["max"]) else float(acc["max"]),
                        f"{period}_invalid_percent": 100.0 * float(acc["invalid"]) / max(total, 1),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def create_eda_figures(
    case_id: str,
    before_path: Path,
    after_path: Path,
    case_dir: Path,
    descriptions_override: tuple[str | None, ...] | None = None,
) -> dict[str, Any]:
    preview = load_preview(before_path, after_path, descriptions_override=descriptions_override)
    before = preview["before"]
    after = preview["after"]
    valid = preview["valid"]
    descriptions = tuple(name or f"Band {i + 1}" for i, name in enumerate(preview["descriptions"]))
    mapping = resolve_band_mapping(preview["descriptions"])
    difference = after - before
    absolute = np.abs(difference)
    bands = before.shape[0]
    columns = 3
    rows = int(math.ceil(bands / columns))

    before_rgb = rgb_image(before, valid, mapping)
    after_rgb = rgb_image(after, valid, mapping)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(before_rgb)
    axes[0].set_title(f"{case_id} BEFORE", fontsize=17, fontweight="bold")
    axes[1].imshow(after_rgb)
    axes[1].set_title(f"{case_id} AFTER", fontsize=17, fontweight="bold")
    for axis in axes:
        axis.set_axis_off()
    before_after_path = case_dir / "before_after_full.png"
    _save_figure(fig, before_after_path)

    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.6 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for band, axis in enumerate(axes):
        if band >= bands:
            axis.set_axis_off()
            continue
        values = difference[band][valid]
        limit = float(np.nanpercentile(np.abs(values), 98)) if values.size else 1.0
        image = axis.imshow(difference[band], cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"{descriptions[band]}: AFTER - BEFORE", fontsize=12, fontweight="bold")
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    difference_path = case_dir / "band_difference_maps.png"
    _save_figure(fig, difference_path)

    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.6 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for band, axis in enumerate(axes):
        if band >= bands:
            axis.set_axis_off()
            continue
        values = absolute[band][valid]
        limit = float(np.nanpercentile(values, 98)) if values.size else 1.0
        image = axis.imshow(absolute[band], cmap="magma", vmin=0, vmax=limit)
        axis.set_title(f"{descriptions[band]}: absolute change", fontsize=12, fontweight="bold")
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    absolute_path = case_dir / "absolute_difference_maps.png"
    _save_figure(fig, absolute_path)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.2 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for band, axis in enumerate(axes):
        if band >= bands:
            axis.set_axis_off()
            continue
        before_values = before[band][valid]
        after_values = after[band][valid]
        if before_values.size > 50000:
            before_values = rng.choice(before_values, 50000, replace=False)
        if after_values.size > 50000:
            after_values = rng.choice(after_values, 50000, replace=False)
        combined = np.concatenate([before_values, after_values])
        low, high = np.nanpercentile(combined, [1, 99])
        bins = np.linspace(low, high, 70)
        axis.hist(before_values, bins=bins, density=True, alpha=0.55, color="#007C78", label="BEFORE")
        axis.hist(after_values, bins=bins, density=True, alpha=0.55, color="#D1495B", label="AFTER")
        axis.set_title(descriptions[band], fontsize=12, fontweight="bold")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    distribution_path = case_dir / "distribution_histograms.png"
    _save_figure(fig, distribution_path)

    statistics = stream_band_statistics(before_path, after_path, descriptions_override)
    statistics_path = case_dir / "channel_statistics.csv"
    statistics.to_csv(statistics_path, index=False)
    positions = np.arange(len(statistics))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    axes[0].bar(positions - width / 2, statistics["before_mean"], width, label="BEFORE", color="#007C78")
    axes[0].bar(positions + width / 2, statistics["after_mean"], width, label="AFTER", color="#D1495B")
    axes[0].set_ylabel("Mean")
    axes[0].legend()
    axes[1].bar(positions - width / 2, statistics["before_variance"], width, label="BEFORE", color="#007C78")
    axes[1].bar(positions + width / 2, statistics["after_variance"], width, label="AFTER", color="#D1495B")
    axes[1].set_ylabel("Variance")
    axes[1].set_xticks(positions, descriptions, rotation=20)
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    statistics_figure_path = case_dir / "channel_statistics.png"
    _save_figure(fig, statistics_figure_path)

    sample = valid.reshape(-1)
    before_flat = before.reshape(bands, -1)[:, sample]
    after_flat = after.reshape(bands, -1)[:, sample]
    if before_flat.shape[1] > 100000:
        indices = rng.choice(before_flat.shape[1], 100000, replace=False)
        before_flat = before_flat[:, indices]
        after_flat = after_flat[:, indices]
    before_corr = np.corrcoef(before_flat)
    after_corr = np.corrcoef(after_flat)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    for matrix, axis, title in [(before_corr, axes[0], "BEFORE correlation"), (after_corr, axes[1], "AFTER correlation")]:
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_xticks(range(bands), descriptions, rotation=35, ha="right")
        axis.set_yticks(range(bands), descriptions)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    correlation_path = case_dir / "correlation_matrices.png"
    _save_figure(fig, correlation_path)

    composites: list[tuple[str, list[int]]] = []
    if mapping:
        composites = [
            ("Natural color", [mapping["red"], mapping["green"], mapping["blue"]]),
            ("Color infrared", [mapping["nir"], mapping["red"], mapping["green"]]),
            ("SWIR / NIR / Red", [mapping["swir1"], mapping["nir"], mapping["red"]]),
        ]
    else:
        composites = [("Channels 1 / 2 / 3", [0, min(1, bands - 1), min(2, bands - 1)])]
    fig, axes = plt.subplots(2, len(composites), figsize=(5.4 * len(composites), 9), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, -1)
    for column, (name, indices) in enumerate(composites):
        axes[0, column].imshow(false_color_image(before, valid, indices))
        axes[0, column].set_title(f"BEFORE: {name}", fontsize=12, fontweight="bold")
        axes[1, column].imshow(false_color_image(after, valid, indices))
        axes[1, column].set_title(f"AFTER: {name}", fontsize=12, fontweight="bold")
        axes[0, column].set_axis_off()
        axes[1, column].set_axis_off()
    composites_path = case_dir / "band_combinations.png"
    _save_figure(fig, composites_path)

    ndvi_path: Path | None = None
    ndvi_loss_small: np.ndarray | None = None
    if mapping:
        before_ndvi = ndvi(before, mapping)
        after_ndvi = ndvi(after, mapping)
        ndvi_loss_small = before_ndvi - after_ndvi
        values = ndvi_loss_small[valid]
        limit = float(np.nanpercentile(np.abs(values), 98)) if values.size else 1.0
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
        axes[0].imshow(before_rgb)
        axes[0].set_title("BEFORE")
        axes[1].imshow(after_rgb)
        axes[1].set_title("AFTER")
        image = axes[2].imshow(ndvi_loss_small, cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[2].set_title("NDVI loss: red = vegetation decrease")
        fig.colorbar(image, ax=axes[2], fraction=0.04, pad=0.02)
        for axis in axes:
            axis.set_axis_off()
        ndvi_path = case_dir / "ndvi_change_map.png"
        _save_figure(fig, ndvi_path)

        with rasterio.open(before_path) as reference:
            full_ndvi_loss = np.zeros((reference.height, reference.width), dtype="float32")
            reproject(
                source=ndvi_loss_small.astype("float32"),
                destination=full_ndvi_loss,
                src_transform=preview["transform"],
                src_crs=reference.crs,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                resampling=Resampling.bilinear,
            )
            wf.write_raster(case_dir / "ndvi_loss.tif", full_ndvi_loss, reference.profile, "float32", -9999.0)

    return {
        "band_mapping": mapping,
        "descriptions": list(descriptions),
        "figures": {
            "before_after": str(before_after_path),
            "signed_difference": str(difference_path),
            "absolute_difference": str(absolute_path),
            "distributions": str(distribution_path),
            "statistics": str(statistics_figure_path),
            "correlations": str(correlation_path),
            "composites": str(composites_path),
            "ndvi": str(ndvi_path) if ndvi_path else "",
        },
        "statistics_csv": str(statistics_path),
        "preview_shape": list(valid.shape),
    }


def create_kmeans_products(
    case_id: str,
    before_path: Path,
    after_path: Path,
    case_dir: Path,
    random_seed: int = 42,
    descriptions_override: tuple[str | None, ...] | None = None,
) -> dict[str, Any]:
    preview = load_preview(before_path, after_path, descriptions_override=descriptions_override)
    before = preview["before"]
    after = preview["after"]
    valid = preview["valid"]
    mapping = resolve_band_mapping(preview["descriptions"])
    normalized_before = robust_norm(before, valid)
    normalized_after = robust_norm(after, valid)
    signed = normalized_after - normalized_before
    magnitude = np.sqrt(np.sum(np.square(signed), axis=0))
    features = [magnitude]
    if mapping:
        ndvi_loss = ndvi(before, mapping) - ndvi(after, mapping)
        features.append(ndvi_loss)
    features.extend([np.abs(signed[index]) for index in range(signed.shape[0])])
    stack = np.stack(features, axis=-1)
    values = stack[valid]
    if values.shape[0] > 120000:
        rng = np.random.default_rng(random_seed)
        values_for_fit = values[rng.choice(values.shape[0], 120000, replace=False)]
    else:
        values_for_fit = values
    means = np.nanmean(values_for_fit, axis=0)
    stds = np.nanstd(values_for_fit, axis=0) + 1e-6
    model = KMeans(n_clusters=4, random_state=random_seed, n_init=10)
    model.fit((values_for_fit - means) / stds)
    labels = np.full(valid.shape, -1, dtype="int16")
    labels[valid] = model.predict((values - means) / stds)
    cluster_rows = []
    for cluster in range(model.n_clusters):
        cluster_mask = labels == cluster
        cluster_rows.append(
            {
                "cluster": cluster,
                "pixels": int(cluster_mask.sum()),
                "mean_change_magnitude": float(np.nanmean(magnitude[cluster_mask])),
                "mean_ndvi_loss": float(np.nanmean(features[1][cluster_mask])) if mapping else None,
            }
        )
    cluster_table = pd.DataFrame(cluster_rows)
    mag_scale = cluster_table["mean_change_magnitude"].std() + 1e-8
    scores = cluster_table["mean_change_magnitude"] / mag_scale
    if mapping:
        ndvi_scale = cluster_table["mean_ndvi_loss"].std() + 1e-8
        scores = scores + np.maximum(cluster_table["mean_ndvi_loss"], 0) / ndvi_scale
    selected_cluster = int(cluster_table.loc[scores.idxmax(), "cluster"])
    selected_small = labels == selected_cluster
    selected_small = ndi.binary_closing(selected_small, structure=np.ones((3, 3)))
    selected_small = ndi.binary_opening(selected_small, structure=np.ones((2, 2)))

    with rasterio.open(before_path) as reference:
        full_mask = np.zeros((reference.height, reference.width), dtype="uint8")
        reproject(
            source=selected_small.astype("uint8"),
            destination=full_mask,
            src_transform=preview["transform"],
            src_crs=reference.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            resampling=Resampling.nearest,
        )
        mask_path = case_dir / "kmeans_damage_mask.tif"
        wf.write_raster(mask_path, full_mask, reference.profile, "uint8", 0)

    after_rgb = rgb_image(after, valid, mapping)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(after_rgb)
    axes[0, 0].set_title("AFTER image", fontweight="bold")
    magnitude_limit = float(np.nanpercentile(magnitude[valid], 98))
    mag_image = axes[0, 1].imshow(magnitude, cmap="magma", vmin=0, vmax=magnitude_limit)
    axes[0, 1].set_title("Multiband change magnitude", fontweight="bold")
    fig.colorbar(mag_image, ax=axes[0, 1], fraction=0.035, pad=0.02)
    axes[1, 0].imshow(np.ma.masked_where(labels < 0, labels), cmap="tab10", vmin=0, vmax=9)
    axes[1, 0].set_title("K-Means change clusters", fontweight="bold")
    axes[1, 1].imshow(after_rgb)
    axes[1, 1].imshow(np.ma.masked_where(~selected_small, selected_small), cmap=ListedColormap(["#FF2A2A"]), alpha=0.45)
    axes[1, 1].set_title(f"Candidate damage cluster: {selected_cluster}", fontweight="bold")
    for axis in axes.reshape(-1):
        axis.set_axis_off()
    figure_path = case_dir / "kmeans_change_analysis.png"
    _save_figure(fig, figure_path)
    cluster_csv = case_dir / "kmeans_cluster_statistics.csv"
    cluster_table.assign(selected=lambda frame: frame["cluster"] == selected_cluster).to_csv(cluster_csv, index=False)
    metadata = {
        "selected_cluster": selected_cluster,
        "feature_count": int(stack.shape[-1]),
        "fit_samples": int(values_for_fit.shape[0]),
        "important_rule": "K-Means is an unsupervised candidate generator, not final proof of tornado damage.",
    }
    (case_dir / "kmeans_metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"mask": str(mask_path), "figure": str(figure_path), "statistics_csv": str(cluster_csv), **metadata}


def _component_properties(mask: np.ndarray) -> dict[str, float]:
    points = np.argwhere(mask)
    if len(points) < 2:
        return {"pixels": float(len(points)), "elongation": 0.0, "length_scale": 0.0}
    centered = points - points.mean(axis=0)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(np.cov(centered.T)), 0.0))
    return {
        "pixels": float(len(points)),
        "elongation": float(np.sqrt((eigenvalues[-1] + 1.0) / (eigenvalues[0] + 1.0))),
        "length_scale": float(np.sqrt(eigenvalues[-1] + 1.0)),
    }


def select_damage_candidate(
    probability: np.ndarray,
    valid: np.ndarray,
    kmeans_mask: np.ndarray,
    ndvi_loss: np.ndarray | None,
    threshold: float,
    min_pixels: int,
) -> tuple[np.ndarray, str, list[dict[str, Any]]]:
    model_raw = wf.clean_prediction(probability, valid, threshold, min_pixels)
    candidates: dict[str, np.ndarray] = {"hybrid_probability": model_raw}
    kmeans_support = ndi.binary_dilation(kmeans_mask & valid, iterations=3)
    candidates["probability_and_kmeans"] = model_raw & kmeans_support
    if ndvi_loss is not None:
        ndvi_values = ndvi_loss[valid & np.isfinite(ndvi_loss)]
        if ndvi_values.size:
            ndvi_support = valid & (ndvi_loss >= np.nanpercentile(ndvi_values, 94))
            ndvi_support = ndi.binary_dilation(ndvi_support, iterations=2)
            candidates["probability_and_ndvi"] = model_raw & ndvi_support
            candidates["kmeans_and_ndvi"] = kmeans_support & ndvi_support

    records: list[dict[str, Any]] = []
    selected_name = ""
    selected_mask = np.zeros_like(valid, dtype=bool)
    selected_score = -math.inf
    valid_count = max(int(valid.sum()), 1)
    scene_diagonal = float(np.hypot(*valid.shape))
    baseline_component = wf.select_path_component(wf.clean_prediction(model_raw.astype("float32"), valid, 0.5, min_pixels))
    baseline_length_scale = _component_properties(baseline_component)["length_scale"]
    for name, candidate in candidates.items():
        clean = wf.clean_prediction(candidate.astype("float32"), valid, 0.5, min_pixels)
        component = wf.select_path_component(clean)
        props = _component_properties(component)
        fraction = float(component.sum() / valid_count)
        estimated_length_fraction = float(3.5 * props["length_scale"] / max(scene_diagonal, 1.0))
        mean_probability = float(np.nanmean(probability[component])) if component.any() else 0.0
        retains_model_extent = name == "hybrid_probability" or props["length_scale"] >= 0.25 * baseline_length_scale
        plausible = bool(
            component.any()
            and fraction <= 0.30
            and props["elongation"] >= 2.0
            and estimated_length_fraction >= 0.06
            and retains_model_extent
        )
        score = (
            2.2 * mean_probability
            + 0.7 * math.log1p(props["elongation"])
            + 0.8 * math.log1p(props["length_scale"])
            - 3.0 * fraction
        )
        if name == "hybrid_probability":
            score += 0.2
        record = {
            "candidate": name,
            "score": float(score),
            "pixels": int(component.sum()),
            "fraction": fraction,
            "elongation": props["elongation"],
            "estimated_length_fraction": estimated_length_fraction,
            "retains_model_extent": retains_model_extent,
            "mean_probability": mean_probability,
            "plausible": plausible,
        }
        records.append(record)
        if plausible and score > selected_score:
            selected_name = name
            selected_mask = component
            selected_score = score
    return selected_mask, selected_name or "rejected", records


def fit_modis_curve(mask: np.ndarray, transform: Affine) -> tuple[LineString | None, dict[str, Any]]:
    """Modernized portable adaptation of DiLiu2023/Tornado_Modis curve fitting."""
    pixels = np.argwhere(mask)
    if len(pixels) < 20:
        return None, {"selected_model": "none", "reason": "fewer than 20 corridor pixels"}
    if len(pixels) > 60000:
        rng = np.random.default_rng(42)
        pixels = pixels[rng.choice(len(pixels), 60000, replace=False)]
    center = pixels.mean(axis=0)
    centered = pixels - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ axes.T
    along = rotated[:, 0]
    across = rotated[:, 1]
    low, high = np.percentile(along, [1, 99])
    keep = (along >= low) & (along <= high)
    along = along[keep]
    across = across[keep]
    fits: list[tuple[str, np.ndarray, float, float]] = []
    for name, degree in [("linear", 1), ("quadratic", 2)]:
        coefficients = np.polyfit(along, across, degree)
        prediction = np.polyval(coefficients, along)
        mse = float(np.mean(np.square(across - prediction)))
        bic = float(len(along) * np.log(mse + 1e-8) + (degree + 1) * np.log(len(along)))
        fits.append((name, coefficients, mse, bic))
    selected = min(fits, key=lambda item: item[3])
    grid = np.linspace(low, high, 240)
    fitted_across = np.polyval(selected[1], grid)
    rotated_line = np.column_stack([grid, fitted_across])
    row_col = rotated_line @ axes + center
    coords = [rasterio.transform.xy(transform, float(row), float(col)) for row, col in row_col]
    line = LineString([(float(x), float(y)) for x, y in coords])
    metadata = {
        "selected_model": selected[0],
        "selected_mse": selected[2],
        "selected_bic": selected[3],
        "candidate_models": [
            {"model": name, "mse": mse, "bic": bic, "coefficients": coefficients.tolist()}
            for name, coefficients, mse, bic in fits
        ],
        "source_attribution": "Portable adaptation of DiLiu2023/Tornado_Modis, commit 7151aec9153dd9a655e3ac480f3f180e04861b43 (MIT).",
    }
    return line, metadata


def write_corridor_products(
    mask: np.ndarray,
    line: LineString | list[LineString] | None,
    profile: dict[str, Any],
    case_dir: Path,
) -> None:
    wf.write_raster(case_dir / "predicted_damage_mask.tif", mask.astype("uint8"), profile, "uint8", 0)
    polygons = [shape(geometry) for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=profile["transform"]) if value == 1]
    import geopandas as gpd

    gpd.GeoDataFrame({"class": ["agentic_prediction"] * len(polygons)}, geometry=polygons, crs=profile["crs"]).to_file(
        case_dir / "predicted_damage_corridor.geojson", driver="GeoJSON"
    )
    lines = line if isinstance(line, list) else ([line] if line is not None else [])
    if not lines:
        gpd.GeoDataFrame({"class": []}, geometry=[], crs=profile["crs"]).to_file(case_dir / "predicted_path_centerline.geojson", driver="GeoJSON")
    else:
        gpd.GeoDataFrame(
            {
                "class": ["agentic_centerline"] * len(lines),
                "path_id": list(range(1, len(lines) + 1)),
            },
            geometry=lines,
            crs=profile["crs"],
        ).to_file(
            case_dir / "predicted_path_centerline.geojson", driver="GeoJSON"
        )


def create_candidate_figure(
    case_id: str,
    after_path: Path,
    probability: np.ndarray,
    kmeans_mask: np.ndarray,
    selected_mask: np.ndarray,
    candidate_records: list[dict[str, Any]],
    output_path: Path,
    descriptions_override: tuple[str | None, ...] | None = None,
) -> None:
    preview = load_preview(after_path, after_path, descriptions_override=descriptions_override)
    after = preview["after"]
    valid = preview["valid"]
    mapping = resolve_band_mapping(preview["descriptions"])
    rgb = rgb_image(after, valid, mapping)
    height, width = valid.shape
    with rasterio.open(after_path) as source:
        probability_small = source.read(1, out_shape=(height, width), resampling=Resampling.average) if source.count == 1 else None
    if probability_small is None or probability_small.shape != valid.shape:
        probability_small = ndi.zoom(probability, (height / probability.shape[0], width / probability.shape[1]), order=1)
    kmeans_small = ndi.zoom(kmeans_mask.astype("uint8"), (height / kmeans_mask.shape[0], width / kmeans_mask.shape[1]), order=0).astype(bool)
    selected_small = ndi.zoom(selected_mask.astype("uint8"), (height / selected_mask.shape[0], width / selected_mask.shape[1]), order=0).astype(bool)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("AFTER image", fontweight="bold")
    probability_image = axes[0, 1].imshow(probability_small, cmap="inferno", vmin=0, vmax=1)
    axes[0, 1].set_title("Hybrid damage probability", fontweight="bold")
    fig.colorbar(probability_image, ax=axes[0, 1], fraction=0.035, pad=0.02)
    axes[1, 0].imshow(rgb)
    axes[1, 0].imshow(np.ma.masked_where(~kmeans_small, kmeans_small), cmap=ListedColormap(["#FFC857"]), alpha=0.5)
    axes[1, 0].set_title("K-Means candidate", fontweight="bold")
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(np.ma.masked_where(~selected_small, selected_small), cmap=ListedColormap(["#FF2A2A"]), alpha=0.42)
    chosen = next((row["candidate"] for row in candidate_records if row["plausible"] and row["score"] == max([r["score"] for r in candidate_records if r["plausible"]], default=-math.inf)), "rejected")
    axes[1, 1].set_title(f"Selected corridor: {chosen}", fontweight="bold")
    for axis in axes.reshape(-1):
        axis.set_axis_off()
    _save_figure(fig, output_path)


def create_curve_figure(
    after_path: Path,
    mask: np.ndarray,
    skeleton_line: LineString | None,
    curve_line: LineString | None,
    output_path: Path,
    descriptions_override: tuple[str | None, ...] | None = None,
) -> None:
    preview = load_preview(after_path, after_path, descriptions_override=descriptions_override)
    after = preview["after"]
    valid = preview["valid"]
    mapping = resolve_band_mapping(preview["descriptions"])
    rgb = rgb_image(after, valid, mapping)
    full_height, full_width = mask.shape
    height, width = valid.shape
    mask_small = ndi.zoom(mask.astype("uint8"), (height / full_height, width / full_width), order=0).astype(bool)
    with rasterio.open(after_path) as source:
        transform = source.transform
    row_scale = height / full_height
    col_scale = width / full_width
    fig, ax = plt.subplots(figsize=(16, 8.2), constrained_layout=True)
    ax.imshow(rgb)
    ax.imshow(np.ma.masked_where(~mask_small, mask_small), cmap=ListedColormap(["#FF2A2A"]), alpha=0.30)
    for line, color, width_px, label, style in [
        (curve_line, "#FFD400", 3.0, "MODIS-style fitted curve candidate", "--"),
        (skeleton_line, "#FF0000", 4.5, "Skeleton centerline", "-"),
    ]:
        if line is None:
            continue
        xs, ys = line.xy
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        ax.plot(np.asarray(cols) * col_scale, np.asarray(rows) * row_scale, color=color, lw=width_px, linestyle=style, label=label)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left", fontsize=11, framealpha=0.92)
    ax.set_axis_off()
    _save_figure(fig, output_path)
