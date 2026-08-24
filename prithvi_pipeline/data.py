from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage as ndi
from shapely.geometry import box

import run_rf_path_workflow as legacy
from .model import encode_prithvi, normalize_prithvi


PATCH_SIZE = 224
NODATA = -9999.0


@dataclass(frozen=True)
class CaseData:
    case_id: str
    before: Path
    after: Path
    valid_mask: Path
    nws_path: Path | None
    geospatial_metadata: dict[str, Any]


@dataclass
class EncodedPatch:
    case_id: str
    features: torch.Tensor
    auxiliary: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor


def discover_cases(source: Path, aligned_root: Path, output: Path, shapefiles: Path) -> list[CaseData]:
    cases: list[CaseData] = []
    for pair in legacy.discover_pairs(source):
        aligned_case = aligned_root / pair.case_id
        expected = [
            aligned_case / "aligned_before.tif",
            aligned_case / "aligned_after.tif",
            aligned_case / "valid_overlap_mask.tif",
        ]
        if all(path.exists() for path in expected):
            before, after, valid = expected
            with rasterio.open(before) as src:
                descriptions = list(src.descriptions)
                if not any(descriptions) and src.count == 6:
                    descriptions = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
                metadata = {
                    "case_id": pair.case_id,
                    "alignment_source": "reused validated analysis grid",
                    "crs": src.crs.to_string() if src.crs else None,
                    "shape": [src.height, src.width],
                    "transform": list(src.transform),
                    "band_descriptions": descriptions,
                }
            with rasterio.open(valid) as src:
                metadata["valid_fraction"] = float(np.mean(src.read(1) > 0))
        else:
            before, after, valid, metadata = legacy.align_pair(pair, output)
            with rasterio.open(pair.before) as raw_source:
                metadata["band_descriptions"] = list(raw_source.descriptions)
        cases.append(
            CaseData(
                pair.case_id,
                before.resolve(),
                after.resolve(),
                valid.resolve(),
                legacy.find_nws_path(pair.case_id, shapefiles),
                metadata,
            )
        )
    return cases


def validate_prithvi_bands(path: Path, descriptions_override: list[str | None] | None = None) -> dict[str, Any]:
    expected = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7")
    with rasterio.open(path) as src:
        descriptions = tuple(str(value or "").upper() for value in src.descriptions[:6])
        if not any(descriptions) and descriptions_override:
            descriptions = tuple(str(value or "").upper() for value in descriptions_override[:6])
        valid = src.count >= 6 and descriptions == expected
        return {
            "usable": valid,
            "count": src.count,
            "descriptions": list(descriptions),
            "expected_semantics": ["blue", "green", "red", "nir", "swir1", "swir2"],
            "reason": "" if valid else "Prithvi requires Landsat SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B7 in semantic order.",
        }


def _valid_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()


def create_nws_training_label(case: CaseData, output_dir: Path) -> dict[str, Any] | None:
    """Rasterize an official path buffered by its reported damage width."""

    if case.nws_path is None:
        return None
    with rasterio.open(case.before) as reference:
        if reference.crs is None:
            return None
        profile = reference.profile.copy()
        out_shape = (reference.height, reference.width)
        transform = reference.transform
        footprint = gpd.GeoSeries([box(*reference.bounds)], crs=reference.crs)
        pixel_size = max(abs(transform.a), abs(transform.e))

    paths = _valid_geometries(gpd.read_file(case.nws_path))
    if paths.empty or paths.crs is None:
        return None
    paths = paths.to_crs(footprint.crs)
    paths = paths[paths.geometry.intersects(footprint.iloc[0])].copy()
    if paths.empty:
        return None

    projected_crs = paths.estimate_utm_crs()
    if projected_crs is None:
        return None
    projected = paths.to_crs(projected_crs)
    buffered = []
    radii_m = []
    for _, row in projected.iterrows():
        raw_width = row.get("width", np.nan)
        width_yards = float(raw_width) if raw_width is not None and np.isfinite(raw_width) and float(raw_width) > 0 else 660.0
        radius_m = float(np.clip(width_yards * 0.9144 / 2.0, 180.0, 2500.0))
        radii_m.append(radius_m)
        buffered.append(row.geometry.buffer(radius_m, cap_style="round", join_style="round"))

    corridor = gpd.GeoSeries(buffered, crs=projected_crs).to_crs(footprint.crs)
    line_geometries = list(paths.geometry)
    label = rasterize(
        [(geometry, 1) for geometry in corridor],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    centerline = rasterize(
        [(geometry.buffer(pixel_size * 1.5), 1) for geometry in line_geometries],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    expanded = gpd.GeoSeries(
        [geometry.buffer(600.0) for geometry in buffered],
        crs=projected_crs,
    ).to_crs(footprint.crs)
    exclusion = rasterize(
        [(geometry, 1) for geometry in expanded],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    case_dir = output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate", BIGTIFF="IF_SAFER")
    for filename, array in [
        ("ground_truth_damage_mask.tif", label),
        ("ground_truth_centerline_mask.tif", centerline),
    ]:
        with rasterio.open(case_dir / filename, "w", **profile) as dst:
            dst.write(array.astype("uint8"), 1)
    paths.to_file(case_dir / "official_nws_path.geojson", driver="GeoJSON")
    return {
        "case_id": case.case_id,
        "label": label,
        "centerline": centerline,
        "exclusion": exclusion,
        "quality": "Silver",
        "label_mode": "official NWS centerline buffered by reported maximum damage width",
        "source": str(case.nws_path),
        "feature_count": len(paths),
        "buffer_radius_m_min": float(min(radii_m)),
        "buffer_radius_m_max": float(max(radii_m)),
        "positive_pixels": int(label.sum()),
    }


def auxiliary_features(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> np.ndarray:
    difference = np.clip((after - before) / 0.05, -4.0, 4.0)
    difference = np.tanh(difference).astype("float32")
    magnitude = np.sqrt(np.mean(np.square(after - before), axis=0)) / 0.08
    magnitude = np.clip(magnitude, 0.0, 4.0).astype("float32")
    red_before, nir_before = before[2], before[3]
    red_after, nir_after = after[2], after[3]
    ndvi_before = (nir_before - red_before) / (nir_before + red_before + 1e-6)
    ndvi_after = (nir_after - red_after) / (nir_after + red_after + 1e-6)
    ndvi_loss = np.clip(ndvi_before - ndvi_after, -2.0, 2.0).astype("float32")
    result = np.concatenate([difference, magnitude[None], ndvi_loss[None]], axis=0)
    result[:, ~valid] = 0.0
    return result


def _extract_array_patch(array: np.ndarray, row: int, column: int, size: int = PATCH_SIZE) -> np.ndarray:
    top = row - size // 2
    left = column - size // 2
    result = np.zeros((size, size), dtype=array.dtype)
    src_top = max(0, top)
    src_left = max(0, left)
    src_bottom = min(array.shape[0], top + size)
    src_right = min(array.shape[1], left + size)
    if src_bottom <= src_top or src_right <= src_left:
        return result
    dst_top = src_top - top
    dst_left = src_left - left
    result[dst_top : dst_top + src_bottom - src_top, dst_left : dst_left + src_right - src_left] = array[
        src_top:src_bottom,
        src_left:src_right,
    ]
    return result


def _read_image_patch(before_src, after_src, row: int, column: int, size: int = PATCH_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = Window(column - size // 2, row - size // 2, size, size)
    indexes = list(range(1, 7))
    before = np.ma.filled(
        before_src.read(indexes, window=window, boundless=True, fill_value=NODATA, masked=True).astype("float32"),
        np.nan,
    )
    after = np.ma.filled(
        after_src.read(indexes, window=window, boundless=True, fill_value=NODATA, masked=True).astype("float32"),
        np.nan,
    )
    valid = np.all(np.isfinite(before) & np.isfinite(after) & (before != NODATA) & (after != NODATA), axis=0)
    before = np.nan_to_num(before, nan=0.0, posinf=0.0, neginf=0.0)
    after = np.nan_to_num(after, nan=0.0, posinf=0.0, neginf=0.0)
    before[:, ~valid] = 0.0
    after[:, ~valid] = 0.0
    return before, after, valid


def sample_patch_centers(
    label: np.ndarray,
    centerline: np.ndarray,
    exclusion: np.ndarray,
    valid: np.ndarray,
    count_per_class: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    positive_pool = np.argwhere(centerline & valid)
    if len(positive_pool) < count_per_class:
        positive_pool = np.argwhere(label & valid)
    near_negative = np.argwhere(exclusion & ~label & valid)
    far_negative = np.argwhere(~exclusion & valid)
    if not len(positive_pool) or not len(far_negative):
        return []

    def choose(pool: np.ndarray, count: int) -> np.ndarray:
        return pool[rng.choice(len(pool), count, replace=len(pool) < count)]

    positives = choose(positive_pool, count_per_class)
    near_count = count_per_class // 2 if len(near_negative) else 0
    negatives = np.concatenate(
        [
            choose(near_negative, near_count) if near_count else np.empty((0, 2), dtype=int),
            choose(far_negative, count_per_class - near_count),
        ],
        axis=0,
    )
    centers = np.concatenate([positives, negatives], axis=0)
    rng.shuffle(centers)
    return [(int(row), int(column)) for row, column in centers]


def encode_training_patches(
    encoder: torch.nn.Module,
    case: CaseData,
    label_info: dict[str, Any],
    count_per_class: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[EncodedPatch]:
    with rasterio.open(case.valid_mask) as src:
        valid_full = src.read(1) > 0
    centers = sample_patch_centers(
        label_info["label"],
        label_info["centerline"],
        label_info["exclusion"],
        valid_full,
        count_per_class,
        seed + int(re.sub(r"\D", "", case.case_id)),
    )
    encoded: list[EncodedPatch] = []
    encoder.to(device)
    with rasterio.open(case.before) as before_src, rasterio.open(case.after) as after_src:
        for start in range(0, len(centers), batch_size):
            batch_centers = centers[start : start + batch_size]
            images = []
            auxiliary = []
            targets = []
            valids = []
            for row, column in batch_centers:
                before, after, valid = _read_image_patch(before_src, after_src, row, column)
                images.append(np.stack([before, after], axis=1))
                auxiliary.append(auxiliary_features(before, after, valid))
                targets.append(_extract_array_patch(label_info["label"], row, column))
                valids.append(valid)
            image_tensor = normalize_prithvi(torch.from_numpy(np.stack(images)).to(device))
            with torch.inference_mode():
                features = encode_prithvi(encoder, image_tensor).cpu().half()
            for index in range(len(batch_centers)):
                encoded.append(
                    EncodedPatch(
                        case.case_id,
                        features[index],
                        torch.from_numpy(auxiliary[index]).half(),
                        torch.from_numpy(targets[index].astype("float32")).unsqueeze(0),
                        torch.from_numpy(valids[index].astype("float32")).unsqueeze(0),
                    )
                )
    encoder.to("cpu")
    return encoded


def sliding_positions(length: int, patch_size: int = PATCH_SIZE, stride: int = 160) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def window_batches(case: CaseData, batch_size: int, stride: int) -> Iterable[tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]]:
    with rasterio.open(case.before) as before_src, rasterio.open(case.after) as after_src:
        rows = sliding_positions(before_src.height, PATCH_SIZE, stride)
        columns = sliding_positions(before_src.width, PATCH_SIZE, stride)
        locations = [(row, column) for row in rows for column in columns]
        for start in range(0, len(locations), batch_size):
            batch_locations = locations[start : start + batch_size]
            images = []
            auxiliary = []
            valids = []
            for top, left in batch_locations:
                row = top + PATCH_SIZE // 2
                column = left + PATCH_SIZE // 2
                before, after, valid = _read_image_patch(before_src, after_src, row, column)
                images.append(np.stack([before, after], axis=1))
                auxiliary.append(auxiliary_features(before, after, valid))
                valids.append(valid)
            yield batch_locations, np.stack(images), np.stack(auxiliary), np.stack(valids)


def change_magnitude(case: CaseData) -> np.ndarray:
    with rasterio.open(case.before) as before_src, rasterio.open(case.after) as after_src:
        before = before_src.read(range(1, 7)).astype("float32")
        after = after_src.read(range(1, 7)).astype("float32")
    valid = np.all(np.isfinite(before) & np.isfinite(after) & (before != NODATA) & (after != NODATA), axis=0)
    magnitude = np.sqrt(np.mean(np.square(after - before), axis=0)).astype("float32")
    magnitude[~valid] = 0.0
    return magnitude
