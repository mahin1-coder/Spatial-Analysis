#!/usr/bin/env python3
"""Calibrate one reusable U-Net/change-model ensemble and extract curved paths."""
from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from scipy import ndimage as ndi
from skimage.draw import line as raster_line
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.transform import resize


PROJECT = Path(__file__).resolve().parents[1]
UNET_ROOT = PROJECT / "outputs_unet_v6"
CASE_ROOT = PROJECT / "outputs_all_cases" / "cases"
LABEL_ROOT = PROJECT / "data" / "manual_labels" / "screenshot_verified"
OUTPUT = PROJECT / "outputs_hybrid_v7"
NWS_ROOT = PROJECT / "data" / "reference" / "noaa_dat_all"


def robust(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = array[valid & np.isfinite(array)]
    if not len(values):
        return np.zeros_like(array, dtype="float32")
    low, high = np.percentile(values, [2, 98])
    return np.clip((array - low) / max(float(high - low), 1e-6), 0, 1).astype("float32")


def graph_diameter(component: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(component)
    points = np.argwhere(skeleton)
    output = np.zeros_like(component)
    if len(points) < 10:
        return output
    index = np.full(component.shape, -1, dtype="int32")
    index[points[:, 0], points[:, 1]] = np.arange(len(points))

    def neighbors(node: int):
        row, column = points[node]
        local = index[max(0, row - 1):row + 2, max(0, column - 1):column + 2]
        return [int(value) for value in local.ravel() if value >= 0 and value != node]

    def farthest(start: int, return_parent: bool = False):
        distance = np.full(len(points), -1, dtype="int32")
        parent = np.full(len(points), -1, dtype="int32")
        distance[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in neighbors(current):
                if distance[nxt] >= 0:
                    continue
                distance[nxt] = distance[current] + 1
                parent[nxt] = current
                queue.append(nxt)
        target = int(np.argmax(distance))
        return (target, parent) if return_parent else target

    first = farthest(0)
    second, parent = farthest(first, return_parent=True)
    current = second
    while current >= 0:
        row, column = points[current]
        output[row, column] = True
        if current == first:
            break
        current = int(parent[current])
    return output


def extract(probability: np.ndarray, valid: np.ndarray, water: np.ndarray, percentile: float, ratio: float):
    threshold = float(np.percentile(probability[valid], percentile))
    mask = valid & (probability >= threshold)
    mask &= ~ndi.binary_dilation(water, structure=disk(2))
    mask = ndi.binary_closing(mask, structure=disk(3))
    mask = remove_small_objects(mask, min_size=max(15, int(valid.sum() * 0.000025)))
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    candidates = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        line = graph_diameter(component)
        points = np.argwhere(line)
        if len(points) < 14:
            continue
        centered = points - points.mean(0)
        values = np.linalg.eigvalsh(np.cov(centered.T))
        elongation = np.sqrt(max(values[-1], 1e-6) / max(values[0], 1e-6))
        if elongation < 2.0:
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        projection = points @ direction
        endpoints = (points[int(np.argmin(projection))], points[int(np.argmax(projection))])
        score = len(points) * min(elongation, 18) * float(probability[component].mean())
        candidates.append({"score": score, "component": component, "line": line, "direction": direction, "endpoints": endpoints})
    if not candidates:
        return np.zeros_like(mask), np.zeros_like(mask), threshold, 0

    parent = list(range(len(candidates)))
    bridges = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> bool:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root
            return True
        return False

    max_gap = 0.16 * float(np.hypot(*valid.shape))
    cos_orientation = float(np.cos(np.deg2rad(28)))
    cos_bridge = float(np.cos(np.deg2rad(38)))
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            a, b = candidates[first], candidates[second]
            if abs(float(np.dot(a["direction"], b["direction"]))) < cos_orientation:
                continue
            choices = []
            for endpoint_a in a["endpoints"]:
                for endpoint_b in b["endpoints"]:
                    delta = endpoint_b.astype("float64") - endpoint_a.astype("float64")
                    gap = float(np.linalg.norm(delta))
                    if gap <= 1e-6:
                        continue
                    bridge_direction = delta / gap
                    alignment = min(abs(float(np.dot(bridge_direction, a["direction"]))), abs(float(np.dot(bridge_direction, b["direction"]))))
                    choices.append((gap, alignment, endpoint_a, endpoint_b))
            if not choices:
                continue
            gap, alignment, endpoint_a, endpoint_b = min(choices, key=lambda item: item[0])
            if gap <= max_gap and alignment >= cos_bridge:
                if union(first, second):
                    bridges.append((first, second, endpoint_a, endpoint_b))

    groups = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(find(index), []).append((index, candidate))
    tracks = []
    for root, members in groups.items():
        member_ids = {index for index, _ in members}
        track_line = np.logical_or.reduce([candidate["line"] for _, candidate in members])
        track_corridor = np.logical_or.reduce([candidate["component"] for _, candidate in members])
        for first, second, endpoint_a, endpoint_b in bridges:
            if first not in member_ids or second not in member_ids:
                continue
            rows, columns = raster_line(int(endpoint_a[0]), int(endpoint_a[1]), int(endpoint_b[0]), int(endpoint_b[1]))
            inside = (rows >= 0) & (rows < track_line.shape[0]) & (columns >= 0) & (columns < track_line.shape[1])
            track_line[rows[inside], columns[inside]] = True
        track_corridor |= ndi.binary_dilation(track_line, structure=disk(2)) & valid
        simplified = graph_diameter(track_corridor)
        if simplified.any():
            track_line = simplified
        points = np.argwhere(track_line)
        centered = points - points.mean(0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        track_direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        track_direction /= max(float(np.linalg.norm(track_direction)), 1e-6)
        tracks.append((sum(candidate["score"] for _, candidate in members), track_corridor, track_line, track_direction))

    tracks.sort(key=lambda item: item[0], reverse=True)
    dominant_direction = tracks[0][3]
    direction_limit = float(np.cos(np.deg2rad(35)))
    oriented_tracks = [
        item for item in tracks
        if abs(float(np.dot(item[3], dominant_direction))) >= direction_limit
    ]
    selected = [item for item in oriented_tracks if item[0] >= oriented_tracks[0][0] * ratio][:4]
    corridor = np.logical_or.reduce([item[1] for item in selected])
    centerline = np.logical_or.reduce([item[2] for item in selected])
    return corridor, centerline, threshold, len(selected)


def load_case(case_id: str):
    deployment_path = UNET_ROOT / "cases" / case_id / "deployment_probability.npz"
    last_error = None
    for attempt in range(4):
        try:
            deployment = np.load(deployment_path)
            break
        except (OSError, TimeoutError) as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Could not read {deployment_path}: {last_error}")
    unet = deployment["probability"].astype("float32")
    valid = deployment["valid"].astype(bool)
    water = deployment["water"].astype(bool)
    cache = np.load(UNET_ROOT / "cache" / f"{case_id}.npz")
    channels = cache["x"].astype("float32")
    band_count = (channels.shape[0] - 4) // 4
    before = channels[:band_count]
    after = channels[band_count:2 * band_count]
    # Landsat 7 SLC-off gaps and fill pixels may be finite zeroes rather than
    # declared NoData. Exclude them from both inference postprocessing and maps.
    fill = (np.max(before, axis=0) <= 0.005) | (np.max(after, axis=0) <= 0.005)
    valid &= ~fill
    with rasterio.open(CASE_ROOT / case_id / "model_probability.tif") as source:
        baseline = source.read(1).astype("float32")
    if baseline.shape != unet.shape:
        baseline = resize(baseline, unet.shape, order=1, preserve_range=True, anti_aliasing=True).astype("float32")
    baseline = robust(baseline, valid)
    label_path = LABEL_ROOT / case_id / "manual_damage_corridor_mask.tif"
    label = None
    if label_path.exists():
        with rasterio.open(label_path) as source:
            label = source.read(1) > 0
        if label.shape != unet.shape:
            label = resize(label.astype("float32"), unet.shape, order=0, preserve_range=True, anti_aliasing=False) > 0.5
        label &= valid
    return unet, baseline, valid, water, label, after


def score(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    intersection = int((prediction & truth).sum())
    return {
        "dice": 2 * intersection / max(int(prediction.sum() + truth.sum()), 1),
        "precision": intersection / max(int(prediction.sum()), 1),
        "recall": intersection / max(int(truth.sum()), 1),
    }


def after_image(after: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # The feature cache and prediction share this exact grid. Using it avoids
    # the spatial shift caused by resizing onto a pre-rendered comparison PNG.
    band_count = after.shape[0]
    rgb_indices = (3, 2, 1) if band_count >= 4 else (2, 1, 0)
    rgb = np.stack([after[index] for index in rgb_indices], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rgb[~valid] = 0
    return rgb


def reference_mask(case_id: str, filename: str, shape: tuple[int, int]) -> np.ndarray:
    path = NWS_ROOT / case_id.lower() / filename
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    collection = json.loads(path.read_text())
    geometries = [feature.get("geometry") for feature in collection.get("features", []) if feature.get("geometry")]
    if not geometries:
        return np.zeros(shape, dtype=bool)
    with rasterio.open(CASE_ROOT / case_id / "model_probability.tif") as source:
        transform = source.transform * source.transform.scale(source.width / shape[1], source.height / shape[0])
        projected = [transform_geom("EPSG:4326", source.crs, geometry, precision=6) for geometry in geometries]
    return rasterize(
        [(geometry, 1) for geometry in projected],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def reference_agreement(predicted: np.ndarray, official: np.ndarray, case_id: str) -> dict[str, float | str]:
    if not predicted.any() or not official.any():
        return {
            "agreement_within_150m_pct": "",
            "official_coverage_within_150m_pct": "",
            "median_centerline_distance_m": "",
        }
    with rasterio.open(CASE_ROOT / case_id / "model_probability.tif") as source:
        scale_x = source.width / predicted.shape[1]
        scale_y = source.height / predicted.shape[0]
        pixel_x = abs(source.transform.a) * scale_x
        pixel_y = abs(source.transform.e) * scale_y
        if source.crs and source.crs.is_geographic:
            latitude = 0.5 * (source.bounds.bottom + source.bounds.top)
            pixel_x *= 111_320.0 * np.cos(np.deg2rad(latitude))
            pixel_y *= 110_574.0
        pixel_size_m = float(np.mean([pixel_x, pixel_y]))
    tolerance_pixels = max(1.0, 150.0 / max(pixel_size_m, 1e-6))
    distance_to_official = ndi.distance_transform_edt(~official)
    distance_to_predicted = ndi.distance_transform_edt(~predicted)
    predicted_distances = distance_to_official[predicted]
    official_distances = distance_to_predicted[official]
    return {
        "agreement_within_150m_pct": 100.0 * float(np.mean(predicted_distances <= tolerance_pixels)),
        "official_coverage_within_150m_pct": 100.0 * float(np.mean(official_distances <= tolerance_pixels)),
        "median_centerline_distance_m": float(np.median(predicted_distances) * pixel_size_m),
    }


def main() -> None:
    case_ids = sorted(
        [path.name for path in (UNET_ROOT / "cases").glob("TOR*") if (path / "deployment_probability.npz").exists()],
        key=lambda value: int(value[3:]),
    )
    requested = {item.strip() for item in os.getenv("HYBRID_CASES", "").split(",") if item.strip()}
    if requested:
        case_ids = [case_id for case_id in case_ids if case_id in requested]
    cases = {case_id: load_case(case_id) for case_id in case_ids}
    labeled = [case_id for case_id in case_ids if cases[case_id][4] is not None]
    (OUTPUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "models").mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT / "models" / "hybrid_config.json"
    if os.getenv("HYBRID_REUSE_CONFIG") == "1" and config_path.exists():
        selected = json.loads(config_path.read_text())
    else:
        trials = []
        for alpha in (0.35, 0.50, 0.65, 0.80, 1.0):
            for percentile in (95.5, 96.0, 96.5, 97.0, 97.5, 98.0):
                for ratio in (0.12, 0.18, 0.25, 0.32):
                    metrics = []
                    for case_id in labeled:
                        unet, baseline, valid, water, label, _ = cases[case_id]
                        probability = alpha * unet + (1 - alpha) * baseline
                        corridor, _, _, _ = extract(probability, valid, water, percentile, ratio)
                        metrics.append(score(corridor, label))
                    trials.append(
                        {
                            "unet_weight": alpha,
                            "percentile": percentile,
                            "component_ratio": ratio,
                            "mean_dice": float(np.mean([item["dice"] for item in metrics])),
                            "mean_precision": float(np.mean([item["precision"] for item in metrics])),
                            "mean_recall": float(np.mean([item["recall"] for item in metrics])),
                        }
                    )
        trials.sort(key=lambda row: (row["mean_dice"], row["mean_recall"]), reverse=True)
        selected = trials[0]
        with (OUTPUT / "reports" / "training_calibration_grid.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
    print(json.dumps(selected, indent=2), flush=True)
    (OUTPUT / "models" / "hybrid_config.json").write_text(json.dumps(selected, indent=2))

    result_rows = []
    for case_id in case_ids:
        unet, baseline, valid, water, label, after_bands = cases[case_id]
        probability = selected["unet_weight"] * unet + (1 - selected["unet_weight"]) * baseline
        corridor, centerline, threshold, path_count = extract(
            probability, valid, water, selected["percentile"], selected["component_ratio"]
        )
        case_dir = OUTPUT / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "prediction.npz",
            probability=probability,
            corridor=corridor,
            centerline=centerline,
            valid=valid,
            water=water,
            threshold=threshold,
        )
        after = after_image(after_bands, valid)
        display = centerline
        official_path = reference_mask(case_id, "nws_dat_damage_paths.geojson", valid.shape)
        official_polygon = reference_mask(case_id, "nws_dat_damage_polys.geojson", valid.shape)
        height, width = after.shape[:2]
        fig = plt.figure(figsize=(12, max(4, 12 * height / max(width, 1))), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])
        axis.imshow(after)
        if display.any():
            axis.contour(ndi.binary_dilation(display, iterations=2), [0.5], colors=["#10231D"], linewidths=4)
            axis.contour(display, [0.5], colors=["#FFEA00"], linewidths=2.5)
        axis.set_axis_off()
        fig.savefig(case_dir / "model_final_path.png", dpi=180, facecolor="white", pad_inches=0)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        axes[0].imshow(after); axes[0].set_title("Full AFTER image", fontweight="bold")
        image = axes[1].imshow(probability, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("Hybrid damage probability", fontweight="bold")
        fig.colorbar(image, ax=axes[1], fraction=0.04)
        axes[2].imshow(after)
        if display.any():
            axes[2].contour(display, [0.5], colors=["#FFEA00"], linewidths=2.5)
        axes[2].set_title(f"{path_count} model path(s)", fontweight="bold")
        for axis in axes:
            axis.set_axis_off()
        fig.savefig(case_dir / "model_prediction_panel.png", dpi=170, facecolor="white")
        plt.close(fig)

        fig = plt.figure(figsize=(12, max(4, 12 * height / max(width, 1))), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])
        axis.imshow(after)
        if official_polygon.any():
            axis.contourf(official_polygon, levels=[0.5, 1.5], colors=["#249DE3"], alpha=0.24)
            axis.contour(official_polygon, [0.5], colors=["#249DE3"], linewidths=1.5)
        if official_path.any():
            axis.contour(ndi.binary_dilation(official_path, iterations=1), [0.5], colors=["#00E5FF"], linewidths=3.0)
        if not official_path.any() and not official_polygon.any():
            axis.text(
                0.5, 0.5, "No NOAA/NWS DAT reference available",
                transform=axis.transAxes, ha="center", va="center", fontsize=20, fontweight="bold",
                color="white", bbox={"facecolor": "#10231D", "alpha": 0.86, "pad": 12},
            )
        axis.set_axis_off()
        fig.savefig(case_dir / "official_dat_reference.png", dpi=180, facecolor="white", pad_inches=0)
        plt.close(fig)

        agreement = reference_agreement(display, official_path, case_id)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        axes[0].imshow(after)
        if display.any():
            axes[0].contour(ndi.binary_dilation(display, iterations=2), [0.5], colors=["#10231D"], linewidths=4)
            axes[0].contour(display, [0.5], colors=["#FFEA00"], linewidths=2.5)
        axes[0].set_title(f"Model prediction: {path_count} path(s)", fontweight="bold")
        axes[1].imshow(after)
        if official_polygon.any():
            axes[1].contourf(official_polygon, levels=[0.5, 1.5], colors=["#249DE3"], alpha=0.24)
        if official_path.any():
            axes[1].contour(ndi.binary_dilation(official_path, iterations=1), [0.5], colors=["#00E5FF"], linewidths=3.0)
        if official_path.any():
            metric_text = (
                f"Predicted line within 150 m: {agreement['agreement_within_150m_pct']:.1f}%\n"
                f"DAT line covered within 150 m: {agreement['official_coverage_within_150m_pct']:.1f}%\n"
                f"Median line distance: {agreement['median_centerline_distance_m']:.0f} m"
            )
        else:
            metric_text = "No official DAT path available for quantitative comparison"
        axes[1].text(
            0.02, 0.03, metric_text, transform=axes[1].transAxes, va="bottom", fontsize=12,
            color="white", bbox={"facecolor": "#10231D", "alpha": 0.86, "pad": 8},
        )
        axes[1].set_title("Official NOAA/NWS Damage Assessment Toolkit (DAT)", fontweight="bold")
        for axis in axes:
            axis.set_axis_off()
        fig.savefig(case_dir / "model_vs_dat_comparison.png", dpi=180, facecolor="white")
        plt.close(fig)

        row = {
            "case_id": case_id,
            "path_count": path_count,
            "threshold": threshold,
            "evaluation_type": "training-set result" if label is not None else "unlabeled deployment inference",
            "nws_dat_reference_available": bool(official_path.any() or official_polygon.any()),
            "dat_path_available": bool(official_path.any()),
            "dat_polygon_available": bool(official_polygon.any()),
            "official_reference_source": "NOAA/NWS Damage Assessment Toolkit (DAT)" if (official_path.any() or official_polygon.any()) else "",
        }
        row.update(agreement)
        if label is not None:
            row.update(score(corridor, label))
        result_rows.append(row)

    with (OUTPUT / "reports" / "deployment_results.csv").open("w", newline="") as handle:
        fields = sorted({key for row in result_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)


if __name__ == "__main__":
    main()
