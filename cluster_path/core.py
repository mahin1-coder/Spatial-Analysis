from __future__ import annotations

import json
import heapq
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tornado_cluster_path_mpl")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from scipy import ndimage as ndi
from shapely.geometry import LineString, box, shape
from skimage.morphology import disk, remove_small_objects, skeletonize
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler


CASE_RE = re.compile(r"TOR\s*0*(\d+)", re.IGNORECASE)
NODATA = -9999.0
BAND_NAMES = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")


@dataclass(frozen=True)
class CasePair:
    case_id: str
    before: Path
    after: Path


@dataclass
class AnalysisData:
    before: np.ndarray
    after: np.ndarray
    valid: np.ndarray
    transform: Affine
    crs: Any
    bounds: tuple[float, float, float, float]
    profile: dict[str, Any]


@dataclass
class CaseResult:
    case_id: str
    status: str
    confidence: str
    output_dir: str
    nws_available: bool
    imagery_path_found: bool
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]


def _case_id(path: Path) -> str | None:
    match = CASE_RE.search(path.name)
    return f"TOR{int(match.group(1))}" if match else None


def discover_pairs(source: Path) -> list[CasePair]:
    grouped: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(source.glob("*.tif")):
        case_id = _case_id(path)
        if not case_id:
            continue
        role = "before" if any(token in path.name.upper() for token in ("BEFORE", "_PRE")) else None
        role = "after" if any(token in path.name.upper() for token in ("AFTER", "_POST")) else role
        if role:
            grouped.setdefault(case_id, {}).setdefault(role, []).append(path)

    pairs: list[CasePair] = []
    errors: list[str] = []
    for case_id, roles in grouped.items():
        before = roles.get("before", [])
        after = roles.get("after", [])
        if len(before) != 1 or len(after) != 1:
            errors.append(f"{case_id}: expected one BEFORE and one AFTER; found {len(before)} and {len(after)}")
            continue
        pairs.append(CasePair(case_id, before[0], after[0]))
    if errors:
        raise ValueError("Ambiguous raster pairing:\n" + "\n".join(errors))
    return sorted(pairs, key=lambda item: int(item.case_id[3:]))


def find_nws_path(case_id: str, shapefile_root: Path) -> Path | None:
    case_dir = shapefile_root / case_id.lower()
    for name in ("nws_dat_damage_paths.geojson", "nws_dat_damage_paths.shp"):
        candidate = case_dir / name
        if not candidate.exists():
            continue
        try:
            frame = gpd.read_file(candidate)
        except Exception:
            continue
        valid = frame.geometry.notna() & ~frame.geometry.is_empty
        if bool(valid.any()) and frame.crs is not None:
            return candidate
    return None


def load_analysis_data(pair: CasePair, max_dimension: int = 1800) -> AnalysisData:
    """Read AFTER on the BEFORE geospatial grid and create a memory-safe analysis grid."""

    with rasterio.open(pair.before) as before_src, rasterio.open(pair.after) as after_src:
        if before_src.crs is None or after_src.crs is None:
            raise ValueError("Both rasters must contain a CRS.")
        if before_src.count < 6 or after_src.count < 6:
            raise ValueError("Six Landsat reflectance bands are required.")

        scale = max(1.0, max(before_src.height, before_src.width) / float(max_dimension))
        height = max(1, int(round(before_src.height / scale)))
        width = max(1, int(round(before_src.width / scale)))
        transform = before_src.transform * Affine.scale(before_src.width / width, before_src.height / height)
        indexes = list(range(1, 7))

        before = before_src.read(
            indexes,
            out_shape=(6, height, width),
            masked=True,
            resampling=Resampling.bilinear,
        ).filled(np.nan).astype("float32")
        before_mask = before_src.dataset_mask(
            out_shape=(height, width),
            resampling=Resampling.nearest,
        ) > 0

        same_grid = (
            before_src.crs == after_src.crs
            and before_src.transform == after_src.transform
            and before_src.width == after_src.width
            and before_src.height == after_src.height
        )
        if same_grid:
            after_reader = after_src
        else:
            after_reader = WarpedVRT(
                after_src,
                crs=before_src.crs,
                transform=before_src.transform,
                width=before_src.width,
                height=before_src.height,
                resampling=Resampling.bilinear,
            )
        after = after_reader.read(
            indexes,
            out_shape=(6, height, width),
            masked=True,
            resampling=Resampling.bilinear,
        ).filled(np.nan).astype("float32")
        after_mask = after_reader.dataset_mask(
            out_shape=(height, width),
            resampling=Resampling.nearest,
        ) > 0
        if not same_grid:
            after_reader.close()

        valid = before_mask & after_mask & np.all(np.isfinite(before) & np.isfinite(after), axis=0)
        profile = before_src.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=transform,
            count=1,
            crs=before_src.crs,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        return AnalysisData(
            before=before,
            after=after,
            valid=valid,
            transform=transform,
            crs=before_src.crs,
            bounds=tuple(before_src.bounds),
            profile=profile,
        )


def _robust_image_normalize(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(stack, dtype="float32")
    for index in range(stack.shape[0]):
        values = stack[index][valid]
        if values.size < 100:
            continue
        low, high = np.nanpercentile(values, (2.0, 98.0))
        result[index] = np.clip((stack[index] - low) / max(high - low, 1e-6), 0.0, 1.0)
    result[:, ~valid] = 0.0
    return result


def _index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-6)


def _robust_z(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    sample = values[valid]
    if sample.size < 100:
        return np.zeros_like(values, dtype="float32")
    median = float(np.nanmedian(sample))
    q25, q75 = np.nanpercentile(sample, (25.0, 75.0))
    result = (values - median) / max(float(q75 - q25), 1e-6)
    result[~valid] = 0.0
    return np.clip(result, -6.0, 6.0).astype("float32")


def build_features(data: AnalysisData) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    before = _robust_image_normalize(data.before, data.valid)
    after = _robust_image_normalize(data.after, data.valid)
    signed = after - before
    absolute = np.abs(signed)
    magnitude = np.sqrt(np.mean(np.square(signed), axis=0))

    ndvi_before = _index(before[3], before[2])
    ndvi_after = _index(after[3], after[2])
    nbr_before = _index(before[3], before[5])
    nbr_after = _index(after[3], after[5])
    ndmi_before = _index(before[3], before[4])
    ndmi_after = _index(after[3], after[4])
    ndvi_loss = ndvi_before - ndvi_after
    nbr_loss = nbr_before - nbr_after
    ndmi_loss = ndmi_before - ndmi_after

    local_mean = ndi.gaussian_filter(magnitude, sigma=2.0)
    local_square = ndi.gaussian_filter(np.square(magnitude), sigma=2.0)
    local_std = np.sqrt(np.maximum(local_square - np.square(local_mean), 0.0))
    before_gray = np.mean(before[[2, 3, 4]], axis=0)
    after_gray = np.mean(after[[2, 3, 4]], axis=0)
    before_edge = ndi.gaussian_gradient_magnitude(before_gray, sigma=1.0)
    after_edge = ndi.gaussian_gradient_magnitude(after_gray, sigma=1.0)
    edge_change = np.abs(after_edge - before_edge)

    arrays = [*signed, *absolute, magnitude, ndvi_loss, nbr_loss, ndmi_loss, local_mean, local_std, edge_change]
    names = [
        *(f"signed_{name.lower()}" for name in BAND_NAMES),
        *(f"absolute_{name.lower()}" for name in BAND_NAMES),
        "change_magnitude",
        "ndvi_loss",
        "nbr_loss",
        "ndmi_loss",
        "local_change_mean",
        "local_change_std",
        "edge_change",
    ]
    features = np.stack(arrays, axis=-1).astype("float32")
    features[~data.valid] = 0.0
    diagnostics = {
        "before_normalized": before,
        "after_normalized": after,
        "signed": signed,
        "absolute": absolute,
        "magnitude": magnitude,
        "ndvi_loss": ndvi_loss,
        "nbr_loss": nbr_loss,
        "ndmi_loss": ndmi_loss,
        "edge_change": edge_change,
    }
    return features, names, diagnostics


def cluster_change(
    features: np.ndarray,
    names: list[str],
    valid: np.ndarray,
    clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    values = features[valid]
    rng = np.random.default_rng(seed)
    fit_values = values
    if len(values) > 160_000:
        fit_values = values[rng.choice(len(values), 160_000, replace=False)]

    scaler = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    fit_scaled = scaler.fit_transform(fit_values)
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=4096,
        n_init=10,
        max_iter=250,
    )
    model.fit(fit_scaled)
    labels = np.full(valid.shape, -1, dtype="int16")
    labels[valid] = model.predict(scaler.transform(values))

    index = {name: names.index(name) for name in names}
    rows = []
    for cluster_id in range(clusters):
        mask = labels == cluster_id
        count = int(mask.sum())
        rows.append(
            {
                "cluster": cluster_id,
                "pixels": count,
                "fraction": count / max(int(valid.sum()), 1),
                "change_magnitude": float(np.mean(features[..., index["change_magnitude"]][mask])),
                "ndvi_loss": float(np.mean(features[..., index["ndvi_loss"]][mask])),
                "nbr_loss": float(np.mean(features[..., index["nbr_loss"]][mask])),
                "local_change_mean": float(np.mean(features[..., index["local_change_mean"]][mask])),
                "edge_change": float(np.mean(features[..., index["edge_change"]][mask])),
            }
        )
    table = pd.DataFrame(rows)
    score = np.zeros(valid.shape, dtype="float32")
    score += 0.42 * np.maximum(_robust_z(features[..., index["change_magnitude"]], valid), 0.0)
    score += 0.22 * np.maximum(_robust_z(features[..., index["local_change_mean"]], valid), 0.0)
    score += 0.14 * np.maximum(_robust_z(features[..., index["ndvi_loss"]], valid), 0.0)
    score += 0.12 * np.maximum(_robust_z(features[..., index["nbr_loss"]], valid), 0.0)
    score += 0.10 * np.maximum(_robust_z(features[..., index["edge_change"]], valid), 0.0)
    score = ndi.gaussian_filter(score, sigma=1.2)
    score[~valid] = 0.0

    cluster_scores = []
    for row in rows:
        mask = labels == int(row["cluster"])
        prevalence_penalty = max(float(row["fraction"]) - 0.20, 0.0) * 3.0
        cluster_scores.append(float(np.mean(score[mask])) - prevalence_penalty)
    table["damage_score"] = cluster_scores
    table["selected"] = False
    eligible = table[table["fraction"].between(0.002, 0.35)]
    if eligible.empty:
        eligible = table
    selected_ids = eligible.nlargest(min(2, len(eligible)), "damage_score")["cluster"].astype(int).tolist()
    table.loc[table["cluster"].isin(selected_ids), "selected"] = True
    selected = np.isin(labels, selected_ids)
    return labels, score, table


def _component_properties(points: np.ndarray) -> dict[str, float]:
    if len(points) < 3:
        return {"elongation": 0.0, "major": 0.0, "minor": 0.0}
    centered = points - points.mean(axis=0)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(np.cov(centered.T)), 0.0))
    return {
        "elongation": float(np.sqrt((eigenvalues[-1] + 1.0) / (eigenvalues[0] + 1.0))),
        "major": float(np.sqrt(eigenvalues[-1] + 1.0)),
        "minor": float(np.sqrt(eigenvalues[0] + 1.0)),
    }


def select_corridor(
    selected_clusters: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    percentile: float,
    min_area_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = score[selected_clusters & valid]
    if values.size < 100:
        return np.zeros_like(valid), {"reason": "too few candidate pixels", "candidates": []}
    threshold = float(np.percentile(values, percentile))
    mask = selected_clusters & valid & (score >= threshold)
    mask = ndi.binary_closing(mask, structure=disk(4))
    mask = ndi.binary_opening(mask, structure=disk(1))
    minimum = max(30, int(valid.sum() * min_area_fraction))
    mask = remove_small_objects(mask, min_size=max(minimum, 2))
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))

    candidates: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    for label_id, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None:
            continue
        local = labels[bounds] == label_id
        area = int(local.sum())
        points = np.argwhere(local) + np.asarray([axis.start for axis in bounds])
        props = _component_properties(points)
        if area < minimum:
            continue
        mean_score = float(np.mean(score[bounds][local]))
        component = {
            "label": label_id,
            "area_pixels": area,
            "elongation": props["elongation"],
            "major_scale_pixels": props["major"],
            "minor_scale_pixels": props["minor"],
            "mean_damage_score": mean_score,
        }
        components.append(component)
        if props["elongation"] < 2.0 or props["major"] < 8.0:
            continue
        component_score = (
            math.log1p(area)
            * min(props["elongation"], 20.0)
            * math.sqrt(props["major"])
            * max(mean_score, 0.01)
        )
        candidates.append({**component, "selection_score": float(component_score)})
    if not candidates:
        return np.zeros_like(valid), {
            "reason": "no elongated high-change component",
            "threshold": threshold,
            "candidates": [],
        }
    chosen = max(candidates, key=lambda item: item["selection_score"])
    reason = "highest-scoring elongated imagery component"
    by_area = sorted(components, key=lambda item: item["area_pixels"], reverse=True)
    if len(by_area) >= 2:
        dominant = by_area[0]
        dominance_ratio = dominant["area_pixels"] / max(by_area[1]["area_pixels"], 1)
        area_fraction = dominant["area_pixels"] / max(int(valid.sum()), 1)
        if (
            dominance_ratio >= 8.0
            and 0.002 <= area_fraction <= 0.12
            and dominant["major_scale_pixels"] >= 30.0
            # Closing fills narrow gaps with nearby pixels, so the post-morphology
            # component mean can sit just below the original probability cutoff.
            and dominant["mean_damage_score"] >= threshold * 0.95
        ):
            chosen = {
                **dominant,
                "selection_score": float(dominance_ratio),
                "dominance_ratio": float(dominance_ratio),
                "area_fraction": float(area_fraction),
            }
            reason = "dominant coherent high-probability imagery component"
    corridor = labels == int(chosen["label"])
    corridor = ndi.binary_closing(corridor, structure=disk(3))
    return corridor, {
        "reason": reason,
        "threshold": threshold,
        "selected": chosen,
        "candidates": candidates,
    }


def _longest_skeleton_path(mask: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(mask)
    points = [tuple(point) for point in np.argwhere(skeleton)]
    if len(points) < 2:
        return np.empty((0, 2), dtype="float64")
    point_set = set(points)
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    def farthest(start: tuple[int, int]) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
        queue = [start]
        distance = {start: 0.0}
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        for current in queue:
            for dr, dc in offsets:
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor not in point_set or neighbor in distance:
                    continue
                distance[neighbor] = distance[current] + math.hypot(dr, dc)
                previous[neighbor] = current
                queue.append(neighbor)
        end = max(distance, key=distance.get)
        return end, previous

    end_a, _ = farthest(points[0])
    end_b, previous = farthest(end_a)
    ordered = [end_b]
    while ordered[-1] != end_a:
        parent = previous.get(ordered[-1])
        if parent is None:
            return np.empty((0, 2), dtype="float64")
        ordered.append(parent)
    ordered.reverse()
    coordinates = np.asarray(ordered, dtype="float64")
    if len(coordinates) > 8:
        coordinates[:, 0] = ndi.gaussian_filter1d(coordinates[:, 0], sigma=2.0)
        coordinates[:, 1] = ndi.gaussian_filter1d(coordinates[:, 1], sigma=2.0)
    if len(coordinates) > 400:
        coordinates = coordinates[np.linspace(0, len(coordinates) - 1, 400).astype(int)]
    return coordinates


def _direction_consistent_skeleton_path(mask: np.ndarray) -> np.ndarray:
    """Choose a long skeleton route while penalizing branch-induced detours."""
    skeleton = skeletonize(mask)
    points = [tuple(point) for point in np.argwhere(skeleton)]
    if len(points) < 2:
        return np.empty((0, 2), dtype="float64")
    point_set = set(points)
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    degree = {
        point: sum((point[0] + dr, point[1] + dc) in point_set for dr, dc in offsets)
        for point in points
    }
    endpoints = [point for point in points if degree[point] <= 1]
    if len(endpoints) < 2:
        return _longest_skeleton_path(mask)
    if len(endpoints) > 60:
        center = np.mean(np.asarray(points, dtype="float64"), axis=0)
        endpoints = sorted(
            endpoints,
            key=lambda point: float(np.linalg.norm(np.asarray(point) - center)),
            reverse=True,
        )[:60]

    branch_mask = np.zeros_like(skeleton, dtype=bool)
    for point, count in degree.items():
        if count >= 3:
            branch_mask[point] = True
    branch_labels, branch_count = ndi.label(branch_mask, structure=np.ones((3, 3), dtype="uint8"))
    junctions: list[tuple[int, int]] = []
    junction_sizes: list[int] = []
    for label_id in range(1, branch_count + 1):
        cluster = np.argwhere(branch_labels == label_id)
        if cluster.size == 0:
            continue
        center = cluster.mean(axis=0)
        representative = cluster[np.argmin(np.sum((cluster - center) ** 2, axis=1))]
        junctions.append((int(representative[0]), int(representative[1])))
        junction_sizes.append(len(cluster))
    if len(junctions) > 60:
        order = np.argsort(junction_sizes)[::-1][:60]
        junctions = [junctions[int(index)] for index in order]
    terminals = list(dict.fromkeys(endpoints + junctions))

    best_score = -1.0
    best_path: list[tuple[int, int]] = []
    for start_index, start in enumerate(terminals):
        distances = {start: 0.0}
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        queue = [(0.0, start)]
        while queue:
            distance, current = heapq.heappop(queue)
            if distance != distances.get(current):
                continue
            for dr, dc in offsets:
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor not in point_set:
                    continue
                candidate = distance + math.hypot(dr, dc)
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        for end in terminals[start_index + 1:]:
            geodesic = distances.get(end)
            if geodesic is None or geodesic < 40.0:
                continue
            displacement = math.hypot(end[0] - start[0], end[1] - start[1])
            straightness = displacement / max(geodesic, 1e-6)
            score = displacement * straightness ** 6 * math.log1p(geodesic)
            if score <= best_score:
                continue
            route = [end]
            while route[-1] != start:
                parent = previous.get(route[-1])
                if parent is None:
                    route = []
                    break
                route.append(parent)
            if route:
                route.reverse()
                best_score = score
                best_path = route
    if not best_path:
        return _longest_skeleton_path(mask)
    coordinates = np.asarray(best_path, dtype="float64")
    if len(coordinates) > 8:
        coordinates[:, 0] = ndi.gaussian_filter1d(coordinates[:, 0], sigma=2.0)
        coordinates[:, 1] = ndi.gaussian_filter1d(coordinates[:, 1], sigma=2.0)
    if len(coordinates) > 400:
        coordinates = coordinates[np.linspace(0, len(coordinates) - 1, 400).astype(int)]
    return coordinates


def centerline_from_corridor(mask: np.ndarray, transform: Affine) -> LineString | None:
    properties = _component_properties(np.argwhere(mask))
    pixels = (
        _direction_consistent_skeleton_path(mask)
        if properties["elongation"] < 2.0
        else _longest_skeleton_path(mask)
    )
    if len(pixels) < 2:
        return None
    coordinates = [
        rasterio.transform.xy(transform, float(row), float(column), offset="center")
        for row, column in pixels
    ]
    return LineString([(float(x), float(y)) for x, y in coordinates])


def _write_raster(path: Path, array: np.ndarray, profile: dict[str, Any], dtype: str, nodata: int | float) -> None:
    output_profile = profile.copy()
    output_profile.update(dtype=dtype, nodata=nodata, count=1)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype(dtype), 1)


def _write_geometry(path: Path, geometry: Any, crs: Any, kind: str) -> None:
    if geometry is None or geometry.is_empty:
        return
    gpd.GeoDataFrame({"type": [kind]}, geometry=[geometry], crs=crs).to_file(path, driver="GeoJSON")


def _corridor_geometry(mask: np.ndarray, transform: Affine) -> Any | None:
    geometries = [
        shape(geometry)
        for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform)
        if int(value) == 1
    ]
    if not geometries:
        return None
    return max(geometries, key=lambda geometry: geometry.area)


def _rgb(normalized: np.ndarray) -> np.ndarray:
    return np.moveaxis(normalized[[2, 1, 0]], 0, -1)


def _extent(data: AnalysisData) -> tuple[float, float, float, float]:
    left, bottom, right, top = data.bounds
    return left, right, bottom, top


def _save_figures(
    case_id: str,
    output_dir: Path,
    data: AnalysisData,
    diagnostics: dict[str, np.ndarray],
    labels: np.ndarray,
    score: np.ndarray,
    cluster_table: pd.DataFrame,
    corridor: np.ndarray,
    centerline: LineString | None,
    nws_frame: gpd.GeoDataFrame | None,
) -> None:
    extent = _extent(data)
    before_rgb = _rgb(diagnostics["before_normalized"])
    after_rgb = _rgb(diagnostics["after_normalized"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(before_rgb, extent=extent, origin="upper")
    axes[0].set_title(f"{case_id} BEFORE", fontweight="bold")
    axes[1].imshow(after_rgb, extent=extent, origin="upper")
    axes[1].set_title(f"{case_id} AFTER", fontweight="bold")
    for axis in axes:
        axis.set_axis_off()
    fig.savefig(output_dir / "before_after.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for period, stack in (("before", diagnostics["before_normalized"]), ("after", diagnostics["after_normalized"])):
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
        for index, axis in enumerate(axes.flat):
            axis.imshow(stack[index], cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"Band {index + 1}: {BAND_NAMES[index]}", fontweight="bold")
            axis.set_axis_off()
        fig.suptitle(f"{case_id}: {period.upper()} Six-Channel Reflectance", fontsize=18, fontweight="bold")
        fig.savefig(output_dir / f"{period}_six_channels.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        values = diagnostics["signed"][index]
        limit = max(float(np.percentile(np.abs(values[data.valid]), 98)), 1e-4)
        image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"Band {index + 1}: {BAND_NAMES[index]}", fontweight="bold")
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.suptitle(f"{case_id}: Signed AFTER - BEFORE Differences", fontsize=18, fontweight="bold")
    fig.savefig(output_dir / "signed_band_differences.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        values = diagnostics["absolute"][index]
        limit = max(float(np.percentile(values[data.valid], 98)), 1e-4)
        image = axis.imshow(values, cmap="magma", vmin=0, vmax=limit)
        axis.set_title(f"Band {index + 1}: {BAND_NAMES[index]}", fontweight="bold")
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.suptitle(f"{case_id}: Absolute AFTER - BEFORE Differences", fontsize=18, fontweight="bold")
    fig.savefig(output_dir / "absolute_band_differences.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    signed_mean = np.mean(diagnostics["signed"], axis=0)
    signed_limit = max(float(np.percentile(np.abs(signed_mean[data.valid]), 98)), 1e-4)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    first = axes[0].imshow(
        signed_mean,
        cmap="coolwarm",
        vmin=-signed_limit,
        vmax=signed_limit,
    )
    axes[0].set_title("Red/blue mean spectral-change filter", fontweight="bold")
    fig.colorbar(first, ax=axes[0], fraction=0.035, pad=0.02)
    magnitude_limit = max(float(np.percentile(diagnostics["magnitude"][data.valid], 98)), 1e-4)
    second = axes[1].imshow(diagnostics["magnitude"], cmap="magma", vmin=0, vmax=magnitude_limit)
    axes[1].set_title("Multiband change magnitude", fontweight="bold")
    fig.colorbar(second, ax=axes[1], fraction=0.035, pad=0.02)
    for axis in axes:
        axis.set_axis_off()
    fig.suptitle(f"{case_id}: Change Filters", fontsize=18, fontweight="bold")
    fig.savefig(output_dir / "red_blue_filter.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    axes[0, 0].imshow(after_rgb)
    axes[0, 0].set_title("AFTER image", fontweight="bold")
    magnitude_limit = max(float(np.percentile(diagnostics["magnitude"][data.valid], 98)), 1e-4)
    image = axes[0, 1].imshow(diagnostics["magnitude"], cmap="magma", vmin=0, vmax=magnitude_limit)
    axes[0, 1].set_title("Multiband change magnitude", fontweight="bold")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.035, pad=0.02)
    axes[1, 0].imshow(np.ma.masked_where(labels < 0, labels), cmap="tab10")
    axes[1, 0].set_title("K-means change clusters", fontweight="bold")
    axes[1, 1].imshow(after_rgb)
    axes[1, 1].imshow(np.ma.masked_where(~corridor, corridor), cmap=ListedColormap(["#FF8C00"]), alpha=0.5)
    axes[1, 1].set_title("Selected imagery-derived corridor", fontweight="bold")
    for axis in axes.flat:
        axis.set_axis_off()
    fig.savefig(output_dir / "clustering_analysis.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(15, 8), constrained_layout=True)
    axis.imshow(after_rgb, extent=extent, origin="upper")
    axis.imshow(
        np.ma.masked_where(~corridor, corridor),
        extent=extent,
        origin="upper",
        cmap=ListedColormap(["#FF8C00"]),
        alpha=0.35,
    )
    if centerline is not None:
        x, y = centerline.xy
        axis.plot(x, y, color="#FFD400", linewidth=3.2, label="Predicted centerline")
    if nws_frame is not None and not nws_frame.empty:
        nws_frame.plot(ax=axis, color="#00D9FF", linewidth=2.4, label="Official NWS path")
    axis.set_title(f"{case_id}: Final Tornado-Path Validation Map", fontsize=18, fontweight="bold")
    axis.set_axis_off()
    handles, labels_text = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels_text, loc="lower left", framealpha=0.92)
    fig.savefig(output_dir / "final_path_map.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cluster_table.to_csv(output_dir / "kmeans_cluster_statistics.csv", index=False)


def validate_against_nws(
    nws_path: Path | None,
    data: AnalysisData,
    corridor: np.ndarray,
    centerline: LineString | None,
) -> tuple[dict[str, Any], gpd.GeoDataFrame | None]:
    if nws_path is None:
        return {"available": False, "status": "not available"}, None
    frame = gpd.read_file(nws_path)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty or frame.crs is None:
        return {"available": False, "status": "empty or missing CRS"}, None
    frame = frame.to_crs(data.crs)
    footprint = gpd.GeoSeries([box(*data.bounds)], crs=data.crs)
    frame = frame[frame.geometry.intersects(footprint.iloc[0])].copy()
    if frame.empty:
        return {"available": False, "status": "does not overlap raster"}, None
    frame.geometry = frame.geometry.intersection(footprint.iloc[0])
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()

    projected_crs = frame.estimate_utm_crs()
    projected = frame.to_crs(projected_crs)
    widths = []
    buffers = []
    for _, row in projected.iterrows():
        raw_width = row.get("width", np.nan)
        width_yards = float(raw_width) if raw_width is not None and np.isfinite(raw_width) and raw_width > 0 else 660.0
        radius_m = float(np.clip(width_yards * 0.9144 / 2.0, 180.0, 2500.0))
        widths.append(radius_m)
        buffers.append(row.geometry.buffer(radius_m))
    buffered = gpd.GeoSeries(buffers, crs=projected_crs).to_crs(data.crs)
    truth = rasterize(
        [(geometry, 1) for geometry in buffered],
        out_shape=corridor.shape,
        transform=data.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    intersection = int(np.sum(corridor & truth))
    predicted = int(corridor.sum())
    actual = int(truth.sum())
    dice = 2.0 * intersection / max(predicted + actual, 1)
    iou = intersection / max(int(np.sum(corridor | truth)), 1)

    distance_m = None
    if centerline is not None:
        predicted_line = gpd.GeoSeries([centerline], crs=data.crs).to_crs(projected_crs).iloc[0]
        official = projected.geometry.union_all()
        distance_m = float(predicted_line.hausdorff_distance(official))
    metrics = {
        "available": True,
        "status": "validated after imagery prediction",
        "dice": float(dice),
        "iou": float(iou),
        "nws_path_coverage": float(intersection / max(actual, 1)),
        "prediction_precision_against_nws_buffer": float(intersection / max(predicted, 1)),
        "centerline_hausdorff_distance_m": distance_m,
        "nws_buffer_radius_m_min": float(min(widths)),
        "nws_buffer_radius_m_max": float(max(widths)),
    }
    return metrics, frame


def _confidence(
    valid_fraction: float,
    corridor: np.ndarray,
    diagnostics: dict[str, Any],
    nws_metrics: dict[str, Any],
) -> tuple[str, str]:
    if valid_fraction < 0.75:
        return "Rejected", "insufficient valid geographic overlap"
    if not corridor.any():
        return "Rejected", diagnostics.get("reason", "no plausible corridor")
    selected = diagnostics.get("selected", {})
    elongation = float(selected.get("elongation", 0.0))
    area_fraction = float(corridor.mean())
    if elongation < 2.5 or area_fraction > 0.08:
        return "Low", "imagery candidate is weak or implausibly broad"
    if nws_metrics.get("available"):
        dice = float(nws_metrics.get("dice", 0.0))
        distance = nws_metrics.get("centerline_hausdorff_distance_m")
        if dice >= 0.45 and distance is not None and distance <= 3000:
            return "High", "imagery path has strong NWS agreement"
        if dice >= 0.20:
            return "Moderate", "imagery path has partial NWS agreement"
        return "Rejected", "imagery path does not agree with the available NWS reference"
    return "Moderate", "plausible imagery-only candidate; no NWS reference is available"


def process_case(
    pair: CasePair,
    output_root: Path,
    shapefile_root: Path,
    *,
    max_dimension: int = 1800,
    clusters: int = 6,
    percentile: float = 86.0,
    min_area_fraction: float = 0.00008,
    seed: int = 42,
) -> CaseResult:
    case_dir = output_root / "cases" / pair.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(pair, max_dimension=max_dimension)
    valid_fraction = float(data.valid.mean())
    features, names, diagnostics = build_features(data)
    labels, score, cluster_table = cluster_change(features, names, data.valid, clusters, seed + int(pair.case_id[3:]))
    selected_clusters = np.isin(labels, cluster_table.loc[cluster_table["selected"], "cluster"].astype(int))
    corridor, corridor_diagnostics = select_corridor(
        selected_clusters,
        score,
        data.valid,
        percentile,
        min_area_fraction,
    )
    centerline = centerline_from_corridor(corridor, data.transform)
    corridor_geometry = _corridor_geometry(corridor, data.transform)

    nws_path = find_nws_path(pair.case_id, shapefile_root)
    nws_metrics, nws_frame = validate_against_nws(nws_path, data, corridor, centerline)
    confidence, reason = _confidence(valid_fraction, corridor, corridor_diagnostics, nws_metrics)
    status = "accepted" if confidence in {"High", "Moderate"} else "rejected"

    _write_raster(case_dir / "valid_overlap_mask.tif", data.valid, data.profile, "uint8", 0)
    _write_raster(case_dir / "change_score.tif", score, data.profile, "float32", NODATA)
    _write_raster(case_dir / "predicted_damage_mask.tif", corridor, data.profile, "uint8", 0)
    _write_geometry(case_dir / "predicted_damage_corridor.geojson", corridor_geometry, data.crs, "predicted_corridor")
    _write_geometry(case_dir / "predicted_path_centerline.geojson", centerline, data.crs, "predicted_centerline")

    _save_figures(
        pair.case_id,
        case_dir,
        data,
        diagnostics,
        labels,
        score,
        cluster_table,
        corridor,
        centerline,
        nws_frame,
    )

    band_rows = []
    for index, name in enumerate(BAND_NAMES):
        for period, values in (("before", data.before[index]), ("after", data.after[index])):
            sample = values[data.valid]
            band_rows.append(
                {
                    "band": index + 1,
                    "band_name": name,
                    "period": period,
                    "mean": float(np.mean(sample)),
                    "variance": float(np.var(sample)),
                    "median": float(np.median(sample)),
                }
            )
    pd.DataFrame(band_rows).to_csv(case_dir / "channel_statistics.csv", index=False)

    metadata = {
        "case_id": pair.case_id,
        "before": str(pair.before),
        "after": str(pair.after),
        "analysis_crs": str(data.crs),
        "analysis_shape": list(data.valid.shape),
        "analysis_transform": list(data.transform),
        "valid_overlap_fraction": valid_fraction,
        "features": names,
        "clustering": {"algorithm": "MiniBatchKMeans", "clusters": clusters, "seed": seed},
        "nws_used_for_prediction": False,
        "nws_validation": nws_metrics,
        "corridor_diagnostics": corridor_diagnostics,
        "confidence": confidence,
        "confidence_reason": reason,
        "status": status,
    }
    (case_dir / "case_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return CaseResult(
        case_id=pair.case_id,
        status=status,
        confidence=confidence,
        output_dir=str(case_dir),
        nws_available=bool(nws_metrics.get("available")),
        imagery_path_found=centerline is not None,
        metrics=nws_metrics,
        diagnostics={
            "valid_overlap_fraction": valid_fraction,
            "confidence_reason": reason,
            **corridor_diagnostics,
        },
    )


def result_row(result: CaseResult) -> dict[str, Any]:
    row = {
        "case_id": result.case_id,
        "status": result.status,
        "confidence": result.confidence,
        "imagery_path_found": result.imagery_path_found,
        "nws_available": result.nws_available,
        "output_dir": result.output_dir,
        "confidence_reason": result.diagnostics.get("confidence_reason"),
        "valid_overlap_fraction": result.diagnostics.get("valid_overlap_fraction"),
    }
    for key, value in result.metrics.items():
        if key not in {"available", "status"}:
            row[f"nws_{key}"] = value
    return row
