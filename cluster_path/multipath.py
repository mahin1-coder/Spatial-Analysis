from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString
from skimage.draw import line as raster_line
from skimage.morphology import disk, remove_small_objects

from .core import AnalysisData, _component_properties, centerline_from_corridor


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (np.abs(a) + np.abs(b) + 1e-6)


def stable_water_mask(data: AnalysisData, mndwi_threshold: float = 0.05) -> np.ndarray:
    """Detect persistent water directly on the aligned Landsat analysis grid."""
    before_mndwi = _normalized_difference(data.before[1], data.before[4])
    after_mndwi = _normalized_difference(data.after[1], data.after[4])
    before_ndvi = _normalized_difference(data.before[3], data.before[2])
    after_ndvi = _normalized_difference(data.after[3], data.after[2])
    before_water = (before_mndwi >= mndwi_threshold) & (before_ndvi <= 0.25)
    after_water = (after_mndwi >= mndwi_threshold) & (after_ndvi <= 0.25)
    water = before_water & after_water & data.valid
    water = ndi.binary_closing(water, structure=disk(2))
    water = remove_small_objects(water, min_size=12)
    return ndi.binary_dilation(water, structure=disk(1)) & data.valid


def _principal_axis(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    if len(points) < 3:
        return np.asarray([0.0, 1.0])
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axis = vectors[0].astype("float64")
    return axis / max(float(np.linalg.norm(axis)), 1e-9)


def _nearest_boundary_points(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    first_boundary = np.argwhere(first & ~ndi.binary_erosion(first))
    second_boundary = np.argwhere(second & ~ndi.binary_erosion(second))
    if not len(first_boundary) or not len(second_boundary):
        return np.zeros(2), np.zeros(2), float("inf")
    if len(first_boundary) > len(second_boundary):
        source, target, swapped = second_boundary, first_boundary, True
    else:
        source, target, swapped = first_boundary, second_boundary, False
    distances, indexes = cKDTree(target).query(source, k=1)
    index = int(np.argmin(distances))
    one, two = source[index], target[int(indexes[index])]
    return (two, one, float(distances[index])) if swapped else (one, two, float(distances[index]))


def bridge_path_gaps(
    mask: np.ndarray,
    water: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    *,
    max_gap_pixels: int = 45,
    min_water_fraction: float = 0.30,
    max_angle_degrees: float = 50.0,
    max_feature_follow_fraction: float = 0.65,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Bridge only short, directionally consistent gaps supported by water or probability."""
    if max_gap_pixels < 2:
        return mask.copy(), []
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    components = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < 30:
            continue
        points = np.argwhere(component)
        properties = _component_properties(points)
        if properties["major"] < 6.0 or properties["elongation"] < 1.4:
            continue
        bounds = (
            int(points[:, 0].min()),
            int(points[:, 0].max()),
            int(points[:, 1].min()),
            int(points[:, 1].max()),
        )
        components.append((label_id, component, area, _principal_axis(component), bounds))
    components.sort(key=lambda item: item[2], reverse=True)
    components = components[:120]
    bridges: list[dict[str, Any]] = []
    result = mask.copy()
    cosine_limit = math.cos(math.radians(max_angle_degrees))
    support_threshold = float(np.percentile(probability[valid], 70)) if valid.any() else 1.0
    for index, (first_id, first, _, first_axis, first_bounds) in enumerate(components):
        for second_id, second, _, second_axis, second_bounds in components[index + 1 :]:
            row_gap = max(first_bounds[0] - second_bounds[1], second_bounds[0] - first_bounds[1], 0)
            column_gap = max(first_bounds[2] - second_bounds[3], second_bounds[2] - first_bounds[3], 0)
            if math.hypot(row_gap, column_gap) > max_gap_pixels:
                continue
            start, end, distance = _nearest_boundary_points(first, second)
            if not np.isfinite(distance) or distance < 2 or distance > max_gap_pixels:
                continue
            direction = end.astype("float64") - start.astype("float64")
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            first_alignment = abs(float(np.dot(first_axis, direction)))
            second_alignment = abs(float(np.dot(second_axis, direction)))
            if min(first_alignment, second_alignment) < cosine_limit:
                continue
            rows, columns = raster_line(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
            in_bounds = (
                (rows >= 0) & (rows < mask.shape[0]) & (columns >= 0) & (columns < mask.shape[1])
            )
            rows, columns = rows[in_bounds], columns[in_bounds]
            if not len(rows):
                continue
            water_fraction = float(np.mean(water[rows, columns]))
            probability_support = float(np.mean(probability[rows, columns] >= support_threshold))
            if water_fraction < min_water_fraction and probability_support < 0.45:
                continue
            # A bridge may cross a mapped linear feature, but it must not run
            # along that feature. A parallel offset remains on a broad feature
            # during a crossing; it leaves a narrow feature when following it.
            perpendicular = np.asarray([-direction[1], direction[0]])
            # Move far enough to leave a typical rasterized linear feature.
            # A true crossing remains on the feature when shifted along it;
            # a feature-following line leaves it when shifted across it.
            offset_distance = max(3.0, min(12.0, distance * 0.40))
            offset = np.rint(perpendicular * offset_distance).astype(int)
            side_rows = np.clip(rows + offset[0], 0, mask.shape[0] - 1)
            side_columns = np.clip(columns + offset[1], 0, mask.shape[1] - 1)
            side_fraction = float(np.mean(water[side_rows, side_columns]))
            feature_follow_fraction = water_fraction * (1.0 - side_fraction)
            if feature_follow_fraction > max_feature_follow_fraction:
                continue
            bridge = np.zeros_like(mask)
            bridge[rows, columns] = True
            bridge = ndi.binary_dilation(bridge, structure=disk(2)) & valid
            result |= bridge
            bridges.append(
                {
                    "from_component": first_id,
                    "to_component": second_id,
                    "gap_pixels": float(distance),
                    "water_fraction": water_fraction,
                    "probability_support": probability_support,
                    "alignment": float(min(first_alignment, second_alignment)),
                    "feature_follow_fraction": feature_follow_fraction,
                }
            )
    return result, bridges


def extract_multiple_corridors(
    probability: np.ndarray,
    valid: np.ndarray,
    percentile: float,
    *,
    water_mask: np.ndarray | None = None,
    max_paths: int = 6,
    min_area_fraction: float = 0.00005,
    max_gap_pixels: int = 45,
    min_water_fraction: float = 0.30,
    max_bridge_angle_degrees: float = 50.0,
    min_relative_path_score: float = 0.35,
    water_exclusion_buffer_pixels: int = 6,
    axis_filter_half_width_pixels: float | None = None,
    exclusion_mask: np.ndarray | None = None,
    crossing_mask: np.ndarray | None = None,
    max_feature_follow_fraction: float = 0.65,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = probability[valid]
    if values.size < 100:
        return np.zeros_like(valid), {"reason": "too few valid probability pixels", "paths": []}
    threshold = float(np.percentile(values, percentile))
    mask = valid & (probability > threshold)
    if int(mask.sum()) < 30:
        mask = valid & (probability >= threshold)
    mask = ndi.binary_closing(mask, structure=disk(3))
    mask = ndi.binary_opening(mask, structure=disk(1))
    mask &= valid
    minimum = max(30, int(valid.sum() * min_area_fraction))
    mask = remove_small_objects(mask, min_size=minimum)
    water = np.zeros_like(valid) if water_mask is None else water_mask & valid
    exclusion = water.copy() if exclusion_mask is None else (exclusion_mask | water) & valid
    crossing = exclusion if crossing_mask is None else crossing_mask & valid
    # Persistent water is not tornado damage. Remove it before component
    # selection; the gap-bridging stage may add only short, directional links
    # across water when both corridor sides support the continuation.
    water_exclusion = (
        ndi.binary_dilation(exclusion, structure=disk(water_exclusion_buffer_pixels))
        if water_exclusion_buffer_pixels > 0
        else water
    )
    mask &= ~water_exclusion
    mask = remove_small_objects(mask, min_size=minimum)
    bridged, bridges = bridge_path_gaps(
        mask,
        crossing,
        probability,
        valid,
        max_gap_pixels=max_gap_pixels,
        min_water_fraction=min_water_fraction,
        max_angle_degrees=max_bridge_angle_degrees,
        max_feature_follow_fraction=max_feature_follow_fraction,
    )
    labels, count = ndi.label(bridged, structure=np.ones((3, 3), dtype="uint8"))
    candidates: list[dict[str, Any]] = []
    for label_id, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None:
            continue
        local = labels[bounds] == label_id
        area = int(local.sum())
        if area < minimum:
            continue
        points = np.argwhere(local) + np.asarray([axis.start for axis in bounds])
        properties = _component_properties(points)
        mean_probability = float(np.mean(probability[bounds][local]))
        if properties["major"] < 8.0:
            continue
        if properties["elongation"] < 1.8 and area < int(valid.sum() * 0.002):
            continue
        score = (
            math.log1p(area)
            * max(min(properties["elongation"], 12.0), 1.0)
            * math.sqrt(properties["major"])
            * max(mean_probability, 0.01)
        )
        candidates.append(
            {
                "label": label_id,
                "area_pixels": area,
                "area_fraction": area / max(int(valid.sum()), 1),
                "elongation": properties["elongation"],
                "major_scale_pixels": properties["major"],
                "mean_probability": mean_probability,
                "selection_score": float(score),
            }
        )
    if not candidates:
        return np.zeros_like(valid), {
            "reason": "no plausible path components",
            "threshold": threshold,
            "paths": [],
            "bridges": bridges,
        }
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    best_score = candidates[0]["selection_score"]
    selected = [
        row
        for row in candidates
        if row["selection_score"] >= best_score * min_relative_path_score
        and row["mean_probability"] >= threshold * 0.85
    ][:max_paths]
    combined = np.isin(labels, [int(row["label"]) for row in selected])
    # Preserve the connectivity of the already-cleaned selected components. A
    # second erosion-based closing can split thin, valid corridors into several
    # artificial paths.
    combined &= valid
    if axis_filter_half_width_pixels is not None and combined.any():
        points = np.argwhere(combined)
        center = points.mean(axis=0)
        axis = _principal_axis(combined)
        perpendicular = np.asarray([-axis[1], axis[0]])
        rows, columns = np.indices(combined.shape)
        offsets = np.stack((rows - center[0], columns - center[1]), axis=-1)
        distance = np.abs(np.einsum("...i,i->...", offsets, perpendicular))
        combined &= distance <= float(axis_filter_half_width_pixels)
        combined = remove_small_objects(combined, min_size=minimum)
        combined, axis_bridges = bridge_path_gaps(
            combined,
            crossing,
            probability,
            valid,
            max_gap_pixels=max_gap_pixels,
            min_water_fraction=min_water_fraction,
            max_angle_degrees=max_bridge_angle_degrees,
            max_feature_follow_fraction=max_feature_follow_fraction,
        )
        bridges.extend(axis_bridges)
    return combined, {
        "reason": "multiple independent imagery corridors selected",
        "threshold": threshold,
        "path_count": len(selected),
        "paths": selected,
        "candidates": candidates,
        "bridges": bridges,
        "water_pixels": int(water.sum()),
        "water_exclusion_pixels": int(water_exclusion.sum()),
        "context_exclusion_pixels": int(exclusion.sum()),
        "axis_filter_half_width_pixels": axis_filter_half_width_pixels,
        "minimum_relative_path_score": float(min_relative_path_score),
    }


def centerlines_from_corridor(
    corridor: np.ndarray,
    transform: Affine,
    *,
    min_pixels: int = 30,
    straighten: bool = False,
) -> list[LineString]:
    labels, count = ndi.label(corridor, structure=np.ones((3, 3), dtype="uint8"))
    lines: list[LineString] = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        if int(component.sum()) < min_pixels:
            continue
        if straighten:
            points = np.argwhere(component)
            center = points.mean(axis=0)
            axis = _principal_axis(component)
            projections = (points - center) @ axis
            endpoints = [center + projections.min() * axis, center + projections.max() * axis]
            coordinates = [
                transform * (float(point[1]) + 0.5, float(point[0]) + 0.5)
                for point in endpoints
            ]
            centerline = LineString(coordinates)
        else:
            centerline = centerline_from_corridor(component, transform)
        if centerline is not None and centerline.length > 0:
            lines.append(centerline)
    return sorted(lines, key=lambda line: line.length, reverse=True)


def _sample_line(line: LineString, count: int = 100) -> list:
    if line.length <= 0:
        return []
    return [line.interpolate(index / max(count - 1, 1), normalized=True) for index in range(count)]


def _symmetric_mean_distance(first: LineString, second: LineString) -> float:
    first_distances = [point.distance(second) for point in _sample_line(first)]
    second_distances = [point.distance(first) for point in _sample_line(second)]
    return float((np.mean(first_distances) + np.mean(second_distances)) / 2.0)


def _flatten_lines(geometries) -> list[LineString]:
    lines: list[LineString] = []
    for geometry in geometries:
        if isinstance(geometry, LineString):
            lines.append(geometry)
        elif isinstance(geometry, MultiLineString):
            lines.extend(part for part in geometry.geoms if part.length > 0)
    return lines


def evaluate_path_set(
    predicted_lines: list[LineString],
    reference_frame: gpd.GeoDataFrame | None,
    analysis_crs: Any,
    *,
    reference_is_exhaustive: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reference_frame is None or reference_frame.empty:
        return {
            "reference_path_count": 0,
            "predicted_path_count": len(predicted_lines),
            "matched_path_count": 0,
            "reference_completeness": "not available",
            "path_count_precision": None,
            "path_count_recall": None,
            "path_count_f1": None,
        }, []
    reference = reference_frame.to_crs(analysis_crs)
    reference_lines = _flatten_lines(reference.geometry)
    if not reference_lines or not predicted_lines:
        return {
            "reference_path_count": len(reference_lines),
            "predicted_path_count": len(predicted_lines),
            "matched_path_count": 0,
            "reference_completeness": "exhaustive" if reference_is_exhaustive else "partial/unknown",
            "unverified_predicted_path_count": len(predicted_lines),
            "path_count_precision": 0.0 if predicted_lines and reference_is_exhaustive else None,
            "path_count_recall": 0.0 if reference_lines else None,
            "path_count_f1": 0.0 if reference_is_exhaustive else None,
        }, []
    combined = gpd.GeoSeries([*predicted_lines, *reference_lines], crs=analysis_crs)
    projected_crs = combined.estimate_utm_crs()
    projected = combined.to_crs(projected_crs)
    predicted_projected = list(projected.iloc[: len(predicted_lines)])
    reference_projected = list(projected.iloc[len(predicted_lines) :])
    costs = np.asarray(
        [
            [_symmetric_mean_distance(prediction, reference_line) for reference_line in reference_projected]
            for prediction in predicted_projected
        ],
        dtype="float64",
    )
    prediction_indexes, reference_indexes = linear_sum_assignment(costs)
    matches: list[dict[str, Any]] = []
    matched = 0
    for prediction_index, reference_index in zip(prediction_indexes, reference_indexes):
        prediction = predicted_projected[int(prediction_index)]
        reference_line = reference_projected[int(reference_index)]
        mean_distance = float(costs[prediction_index, reference_index])
        tolerance = float(np.clip(reference_line.length * 0.08, 1000.0, 5000.0))
        accepted = mean_distance <= tolerance
        matched += int(accepted)
        matches.append(
            {
                "predicted_path_id": int(prediction_index + 1),
                "reference_path_id": int(reference_index + 1),
                "matched": bool(accepted),
                "mean_symmetric_distance_m": mean_distance,
                "hausdorff_distance_m": float(prediction.hausdorff_distance(reference_line)),
                "predicted_length_m": float(prediction.length),
                "reference_length_m": float(reference_line.length),
                "length_error_percent": float(
                    100.0 * abs(prediction.length - reference_line.length) / max(reference_line.length, 1e-9)
                ),
                "reference_overlap_percent_at_1km": float(
                    100.0
                    * reference_line.intersection(prediction.buffer(1000.0)).length
                    / max(reference_line.length, 1e-9)
                ),
                "matching_tolerance_m": tolerance,
            }
        )
    precision = matched / max(len(predicted_lines), 1)
    recall = matched / max(len(reference_lines), 1)
    return {
        "reference_path_count": len(reference_lines),
        "predicted_path_count": len(predicted_lines),
        "matched_path_count": matched,
        "reference_completeness": "exhaustive" if reference_is_exhaustive else "partial/unknown",
        "unverified_predicted_path_count": len(predicted_lines) - matched,
        "path_count_precision": float(precision) if reference_is_exhaustive else None,
        "path_count_recall": float(recall),
        "path_count_f1": (
            float(2 * precision * recall / max(precision + recall, 1e-9))
            if reference_is_exhaustive
            else None
        ),
        "documented_path_recovery": float(recall),
        "mean_matched_distance_m": float(
            np.mean([row["mean_symmetric_distance_m"] for row in matches if row["matched"]])
        ) if matched else None,
    }, matches
