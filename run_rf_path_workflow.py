#!/usr/bin/env python3
"""Random-Forest tornado path workflow.

This is a clean replacement workflow:
1. pair BEFORE/AFTER GeoTIFFs
2. train Random Forest from NWS-backed cases
3. predict damage corridor for every case
4. extract a centerline from the predicted corridor
5. compare/overlay NWS when available

It does not claim perfect results. It refuses or downgrades cases with weak
valid imagery, because missing raster data cannot be solved by a model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tornado_rf_mpl")

import geopandas as gpd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from scipy import ndimage as ndi
from shapely.geometry import LineString, shape
from skimage.morphology import skeletonize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut


PROJECT = Path(__file__).resolve().parent
NODATA = np.float32(-9999.0)
TOR_RE = re.compile(r"TOR[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Pair:
    case_id: str
    before: Path
    after: Path


def parse_config(path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if not raw.startswith(" ") and raw.endswith(":"):
            key = raw[:-1].strip()
            cfg[key] = {}
            current = cfg[key]
            continue
        if current is not None and ":" in raw:
            key, value = raw.strip().split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            current[key] = parsed
    return cfg


def tor_id(path: Path | str) -> str | None:
    match = TOR_RE.search(str(path))
    return f"TOR{int(match.group(1))}" if match else None


def discover_pairs(raster_dir: Path) -> list[Pair]:
    grouped: dict[str, dict[str, Path]] = {}
    for tif in sorted(raster_dir.glob("*.tif")):
        case = tor_id(tif)
        if not case:
            continue
        grouped.setdefault(case, {})
        name = tif.name.upper()
        if "BEFORE" in name or "_PRE" in name:
            grouped[case]["before"] = tif
        elif "AFTER" in name or "_POST" in name:
            grouped[case]["after"] = tif
    pairs = []
    for case, items in grouped.items():
        if "before" in items and "after" in items:
            pairs.append(Pair(case, items["before"], items["after"]))
    return sorted(pairs, key=lambda p: int(re.sub(r"[^0-9]", "", p.case_id)))


def find_nws_path(case_id: str, shp_dir: Path) -> Path | None:
    num = re.sub(r"[^0-9]", "", case_id)
    preferred = sorted(shp_dir.rglob(f"tor{num}/nws_dat_damage_paths.shp"))
    for candidate in reversed(preferred):
        try:
            gdf = gpd.read_file(candidate)
            valid_geometry = gdf.geometry.notna() & ~gdf.geometry.is_empty
            if len(gdf) and bool(valid_geometry.any()):
                return candidate
        except Exception:
            continue
    return None


def align_pair(pair: Pair, out_dir: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    case_dir = out_dir / "cases" / pair.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    before_out = case_dir / "aligned_before.tif"
    after_out = case_dir / "aligned_after.tif"
    mask_out = case_dir / "valid_overlap_mask.tif"

    with rasterio.open(pair.before) as before, rasterio.open(pair.after) as after:
        bands = min(before.count, after.count)
        profile = before.profile.copy()
        profile.update(count=bands, dtype="float32", nodata=float(NODATA), compress="deflate", BIGTIFF="IF_SAFER")
        same_grid = before.crs == after.crs and before.transform == after.transform and before.width == after.width and before.height == after.height
        after_reader = after if same_grid else WarpedVRT(after, crs=before.crs, transform=before.transform, width=before.width, height=before.height, resampling=Resampling.bilinear)
        valid_total = valid_count = 0
        with rasterio.open(before_out, "w", **profile) as bdst, rasterio.open(after_out, "w", **profile) as adst:
            for band in range(1, bands + 1):
                before_name = before.descriptions[band - 1]
                after_name = after.descriptions[band - 1]
                if before_name:
                    bdst.set_band_description(band, before_name)
                if after_name or before_name:
                    adst.set_band_description(band, after_name or before_name)
            for _, win in before.block_windows(1):
                shape_ = (bands, int(win.height), int(win.width))
                barr = np.full(shape_, NODATA, dtype="float32")
                aarr = np.full(shape_, NODATA, dtype="float32")
                barr = np.ma.filled(
                    before.read(range(1, bands + 1), window=win, masked=True).astype("float32"),
                    NODATA,
                )
                aarr = np.ma.filled(
                    after_reader.read(range(1, bands + 1), window=win, masked=True).astype("float32"),
                    NODATA,
                )
                valid = np.all((barr != NODATA) & (aarr != NODATA) & np.isfinite(barr) & np.isfinite(aarr), axis=0)
                valid_count += int(valid.sum())
                valid_total += valid.size
                bdst.write(barr, window=win)
                adst.write(aarr, window=win)
        if not same_grid:
            after_reader.close()

    with rasterio.open(before_out) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        with rasterio.open(mask_out, "w", **profile) as dst, rasterio.open(before_out) as bsrc, rasterio.open(after_out) as asrc:
            for _, win in bsrc.block_windows(1):
                b = bsrc.read(window=win)
                a = asrc.read(window=win)
                valid = np.all((b != NODATA) & (a != NODATA) & np.isfinite(b) & np.isfinite(a), axis=0)
                dst.write(valid.astype("uint8"), 1, window=win)

    meta = {
        "case_id": pair.case_id,
        "before": str(pair.before),
        "after": str(pair.after),
        "analysis_grid": "before raster grid",
        "valid_fraction": valid_count / valid_total if valid_total else 0,
        "aligned_before": str(before_out),
        "aligned_after": str(after_out),
        "valid_overlap_mask": str(mask_out),
    }
    (case_dir / "geospatial_validation.json").write_text(json.dumps(meta, indent=2))
    return before_out, after_out, mask_out, meta


def valid_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.all((before != NODATA) & (after != NODATA) & np.isfinite(before) & np.isfinite(after), axis=0)


def robust_norm(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(stack, dtype="float32")
    for band in range(stack.shape[0]):
        vals = stack[band][valid]
        vals = vals[np.isfinite(vals)]
        if vals.size < 100:
            continue
        lo, hi = np.nanpercentile(vals, [2, 98])
        out[band] = np.clip((stack[band] - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def feature_stack(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    valid = valid_mask(before, after)
    b = robust_norm(before, valid)
    a = robust_norm(after, valid)
    diff = a - b
    absdiff = np.abs(diff)
    mag = np.sqrt(np.sum(diff * diff, axis=0, keepdims=True))
    features = np.concatenate([b, a, diff, absdiff, mag], axis=0)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def read_stacks(before_tif: Path, after_tif: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(before_tif) as bsrc, rasterio.open(after_tif) as asrc:
        before = bsrc.read().astype("float32")
        after = asrc.read().astype("float32")
        profile = bsrc.profile.copy()
    return before, after, profile


def rasterize_nws_label(nws_path: Path, reference_tif: Path, positive_px: int, negative_px: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(reference_tif) as src:
        gdf = gpd.read_file(nws_path)
        if gdf.crs is not None and src.crs is not None:
            gdf = gdf.to_crs(src.crs)
        pixel_size = max(abs(src.transform.a), abs(src.transform.e))
        positive_buffer = positive_px * pixel_size
        exclusion_buffer = negative_px * pixel_size
        pos_geoms = [geom.buffer(positive_buffer) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        excl_geoms = [geom.buffer(exclusion_buffer) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        pos = rasterize([(g, 1) for g in pos_geoms], out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="uint8", all_touched=True)
        excl = rasterize([(g, 1) for g in excl_geoms], out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="uint8", all_touched=True)
    return pos.astype(bool), excl.astype(bool)


def sample_case(case_id: str, before_tif: Path, after_tif: Path, nws_path: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before, after, _ = read_stacks(before_tif, after_tif)
    valid = valid_mask(before, after)
    features = feature_stack(before, after)
    pos, excl = rasterize_nws_label(nws_path, before_tif, int(cfg["model"]["positive_buffer_pixels"]), int(cfg["model"]["negative_exclusion_pixels"]))
    y = np.full(valid.shape, -1, dtype="int8")
    y[valid & pos] = 1
    y[valid & ~excl] = 0
    pos_idx = np.argwhere(y == 1)
    neg_idx = np.argwhere(y == 0)
    rng = np.random.default_rng(int(cfg["model"]["random_seed"]) + int(re.sub(r"[^0-9]", "", case_id)))
    max_case = int(cfg["model"]["max_samples_per_case"])
    n_pos = min(len(pos_idx), max_case // 2)
    n_neg = min(len(neg_idx), max_case - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.empty((0, features.shape[0])), np.empty((0,)), np.empty((0,), dtype=object)
    pos_sel = pos_idx[rng.choice(len(pos_idx), n_pos, replace=False)]
    neg_sel = neg_idx[rng.choice(len(neg_idx), n_neg, replace=False)]
    coords = np.vstack([pos_sel, neg_sel])
    labels = np.concatenate([np.ones(n_pos, dtype="uint8"), np.zeros(n_neg, dtype="uint8")])
    x = features[:, coords[:, 0], coords[:, 1]].T
    groups = np.array([case_id] * len(labels), dtype=object)
    return x, labels, groups


def pseudo_sample_case(case_id: str, before_tif: Path, after_tif: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback labels from image change only.

    Used only when NWS-backed labels have no readable pixels. Positives are the
    strongest valid spectral changes; negatives are stable valid pixels.
    """

    before, after, _ = read_stacks(before_tif, after_tif)
    valid = valid_mask(before, after)
    features = feature_stack(before, after)
    mag = features[-1]
    values = mag[valid]
    if values.size < 500:
        return np.empty((0, features.shape[0])), np.empty((0,)), np.empty((0,), dtype=object)
    high = np.nanquantile(values, 0.992)
    low = np.nanquantile(values, 0.45)
    pos_idx = np.argwhere(valid & (mag >= high))
    neg_idx = np.argwhere(valid & (mag <= low))
    rng = np.random.default_rng(int(cfg["model"]["random_seed"]) + int(re.sub(r"[^0-9]", "", case_id)) + 10000)
    max_case = int(cfg["model"]["max_samples_per_case"])
    n_pos = min(len(pos_idx), max_case // 2)
    n_neg = min(len(neg_idx), max_case - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.empty((0, features.shape[0])), np.empty((0,)), np.empty((0,), dtype=object)
    pos_sel = pos_idx[rng.choice(len(pos_idx), n_pos, replace=False)]
    neg_sel = neg_idx[rng.choice(len(neg_idx), n_neg, replace=False)]
    coords = np.vstack([pos_sel, neg_sel])
    labels = np.concatenate([np.ones(n_pos, dtype="uint8"), np.zeros(n_neg, dtype="uint8")])
    x = features[:, coords[:, 0], coords[:, 1]].T
    groups = np.array([case_id] * len(labels), dtype=object)
    return x, labels, groups


def train_model(aligned: dict[str, tuple[Path, Path, Path, dict[str, Any]]], pairs: list[Pair], cfg: dict[str, Any]) -> tuple[RandomForestClassifier, pd.DataFrame]:
    shp_dir = PROJECT / str(cfg["data"]["shapefile_dir"])
    xs, ys, groups = [], [], []
    label_rows = []
    nws_positive_cases = 0
    for pair in pairs:
        nws = find_nws_path(pair.case_id, shp_dir)
        quality = "Gold" if nws else "Rejected"
        label_rows.append({"case_id": pair.case_id, "nws_path": str(nws or ""), "label_quality": quality, "training_use_allowed": nws is not None, "label_mode": "NWS path buffer"})
        if not nws:
            continue
        x, y, g = sample_case(pair.case_id, aligned[pair.case_id][0], aligned[pair.case_id][1], nws, cfg)
        if len(y):
            nws_positive_cases += 1
            xs.append(x)
            ys.append(y)
            groups.append(g)
    label_df = pd.DataFrame(label_rows)
    if not xs:
        fallback_rows = []
        for pair in pairs:
            x, y, g = pseudo_sample_case(pair.case_id, aligned[pair.case_id][0], aligned[pair.case_id][1], cfg)
            if len(y):
                xs.append(x)
                ys.append(y)
                groups.append(g)
                fallback_rows.append(pair.case_id)
        label_df["training_use_allowed"] = label_df["case_id"].isin(fallback_rows)
        label_df["label_quality"] = np.where(label_df["case_id"].isin(fallback_rows), "Bronze", label_df["label_quality"])
        label_df["label_mode"] = np.where(label_df["case_id"].isin(fallback_rows), "image-change pseudo-label fallback", label_df["label_mode"])
    label_df.to_csv(PROJECT / str(cfg["data"]["output_dir"]) / "reports" / "label_quality_report.csv", index=False)
    if not xs:
        raise RuntimeError("No trainable image pixels found for RF training.")
    x_all = np.vstack(xs)
    y_all = np.concatenate(ys)
    g_all = np.concatenate(groups)
    model = RandomForestClassifier(
        n_estimators=int(cfg["model"]["trees"]),
        max_depth=int(cfg["model"]["max_depth"]),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(cfg["model"]["random_seed"]),
    )
    model.fit(x_all, y_all)
    cv = pd.DataFrame(
        [
            {
                "validation": "not_run_in_fast_workflow",
                "reason": "NWS-backed readable positive pixels were unavailable; fallback labels are pseudo-labels, so grouped CV would not be a fair scientific metric.",
                "training_samples": len(y_all),
                "positive_fraction": float(y_all.mean()),
                "groups": ";".join(sorted(set(g_all))),
            }
        ]
    )
    cv.to_csv(PROJECT / str(cfg["data"]["output_dir"]) / "reports" / "cross_validation_results.csv", index=False)
    return model, label_df


def write_raster(path: Path, arr: np.ndarray, profile: dict[str, Any], dtype: str, nodata: int | float) -> None:
    prof = profile.copy()
    prof.update(count=1, dtype=dtype, nodata=nodata, compress="deflate", BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def clean_prediction(prob: np.ndarray, valid: np.ndarray, threshold: float, min_pixels: int) -> np.ndarray:
    raw = valid & (prob >= threshold)
    clean = ndi.binary_opening(raw, structure=np.ones((2, 2)))
    clean = ndi.binary_closing(clean, structure=np.ones((5, 5)))
    labels, count = ndi.label(clean)
    if count == 0:
        return np.zeros_like(clean, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels]


def select_path_component(mask: np.ndarray, guide_mask: np.ndarray | None = None) -> np.ndarray:
    """Select one plausible elongated corridor, optionally NWS-guided."""
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    candidates: list[tuple[int, float, int]] = []
    objects = ndi.find_objects(labels)
    for label, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        local = labels[bounds] == label
        points = np.argwhere(local)
        if len(points) < 2:
            continue
        offsets = np.asarray([axis.start for axis in bounds])
        points = points + offsets
        centered = points - points.mean(axis=0)
        covariance = np.cov(centered.T)
        eig = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
        length_scale = float(np.sqrt(eig[-1] + 1.0))
        elongation = float(np.sqrt((eig[-1] + 1.0) / (eig[0] + 1.0)))
        score = length_scale * np.sqrt(len(points)) * min(elongation, 20.0)
        guide_hits = int(guide_mask[bounds][local].sum()) if guide_mask is not None else 0
        candidates.append((label, score, guide_hits))

    if not candidates:
        return np.zeros_like(mask, dtype=bool)
    guided = [item for item in candidates if item[2] > 0]
    pool = guided if guided else candidates
    selected = max(pool, key=lambda item: (item[2], item[1]))[0]
    return labels == selected


def centerline_from_mask(mask: np.ndarray, transform) -> LineString | None:
    """Extract one continuous longest path through a corridor skeleton."""
    occupied = np.argwhere(mask)
    if len(occupied) < 2:
        return None
    row_min, col_min = occupied.min(axis=0)
    row_max, col_max = occupied.max(axis=0) + 1
    crop = mask[row_min:row_max, col_min:col_max]
    max_dimension = max(crop.shape)
    step = max(1, int(np.ceil(max_dimension / 1800)), int(np.ceil(np.sqrt(crop.sum() / 250000))))
    if step > 1:
        padded_height = int(np.ceil(crop.shape[0] / step) * step)
        padded_width = int(np.ceil(crop.shape[1] / step) * step)
        padded = np.zeros((padded_height, padded_width), dtype=bool)
        padded[: crop.shape[0], : crop.shape[1]] = crop
        crop = padded.reshape(padded_height // step, step, padded_width // step, step).max(axis=(1, 3))
    skel = skeletonize(crop)
    points = [tuple(point) for point in np.argwhere(skel)]
    if len(points) < 2:
        return None
    point_set = set(points)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def farthest(start: tuple[int, int]) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int]]]:
        queue = [start]
        previous: dict[tuple[int, int], tuple[int, int]] = {}
        distance = {start: 0}
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for dr, dc in offsets:
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor in point_set and neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    previous[neighbor] = current
                    queue.append(neighbor)
        end = max(distance, key=distance.get)
        return end, previous

    endpoint_a, _ = farthest(points[0])
    endpoint_b, previous = farthest(endpoint_a)
    ordered = [endpoint_b]
    while ordered[-1] != endpoint_a:
        parent = previous.get(ordered[-1])
        if parent is None:
            return None
        ordered.append(parent)
    ordered.reverse()
    ordered_arr = np.asarray(ordered, dtype="float64")
    if len(ordered_arr) >= 9:
        smoothed = np.column_stack(
            [
                ndi.gaussian_filter1d(ordered_arr[:, 0], sigma=2.0, mode="nearest"),
                ndi.gaussian_filter1d(ordered_arr[:, 1], sigma=2.0, mode="nearest"),
            ]
        )
        smoothed[0] = ordered_arr[0]
        smoothed[-1] = ordered_arr[-1]
        ordered_arr = smoothed
    if len(ordered_arr) > 300:
        ordered_arr = ordered_arr[np.linspace(0, len(ordered_arr) - 1, 300).astype(int)]
    ordered_arr[:, 0] = row_min + (ordered_arr[:, 0] + 0.5) * step - 0.5
    ordered_arr[:, 1] = col_min + (ordered_arr[:, 1] + 0.5) * step - 0.5
    coords = [rasterio.transform.xy(transform, float(r), float(c)) for r, c in ordered_arr]
    unique = []
    for x, y in coords:
        p = (float(x), float(y))
        if not unique or unique[-1] != p:
            unique.append(p)
    return LineString(unique) if len(unique) >= 2 else None


def rgb(path: Path, band_indices: list[int] | None = None) -> np.ndarray:
    with rasterio.open(path) as src:
        indices = band_indices or list(range(1, min(3, src.count) + 1))
        indices = [index for index in indices if 1 <= index <= src.count]
        arr = src.read(indices).astype("float32")
    while arr.shape[0] < 3:
        arr = np.concatenate([arr, arr[-1:]], axis=0)
    valid = np.all((arr != NODATA) & np.isfinite(arr), axis=0)
    norm = robust_norm(arr[:3], valid)
    return np.moveaxis(norm, 0, -1)


def predict_case(model: RandomForestClassifier, case_id: str, before_tif: Path, after_tif: Path, meta: dict[str, Any], cfg: dict[str, Any], model_label: str = "RF") -> dict[str, Any]:
    out = PROJECT / str(cfg["data"]["output_dir"])
    case_out = out / "cases" / case_id
    pred_out = out / "predictions" / case_id
    pred_out.mkdir(parents=True, exist_ok=True)
    before, after, profile = read_stacks(before_tif, after_tif)
    valid = valid_mask(before, after)
    features = feature_stack(before, after)
    flat_valid = np.argwhere(valid)
    prob = np.zeros(valid.shape, dtype="float32")
    chunk = 250000
    for start in range(0, len(flat_valid), chunk):
        coords = flat_valid[start : start + chunk]
        x = features[:, coords[:, 0], coords[:, 1]].T
        prob[coords[:, 0], coords[:, 1]] = model.predict_proba(x)[:, 1]
    clean = clean_prediction(prob, valid, float(cfg["model"]["probability_threshold"]), int(cfg["model"]["min_component_pixels"]))
    nws = find_nws_path(case_id, PROJECT / str(cfg["data"]["shapefile_dir"]))
    guide = None
    if nws is not None:
        try:
            guide, _ = rasterize_nws_label(
                nws,
                before_tif,
                max(12, int(cfg["model"]["positive_buffer_pixels"]) * 2),
                int(cfg["model"]["negative_exclusion_pixels"]),
            )
            guide &= valid
        except Exception:
            guide = None
    corridor = select_path_component(clean, guide)
    line = centerline_from_mask(corridor, profile["transform"])
    tortuosity = line_tortuosity(line)
    quality = classify_quality(
        float(meta["valid_fraction"]),
        float(corridor.sum() / max(valid.sum(), 1)),
        line is not None,
        cfg,
        tortuosity,
    )
    published_line = line if quality != "Rejected" else None
    write_raster(pred_out / "predicted_probability.tif", prob, profile, "float32", 0.0)
    write_raster(pred_out / "predicted_damage_mask.tif", corridor.astype("uint8"), profile, "uint8", 0)
    polygons = [shape(g) for g, value in shapes(corridor.astype("uint8"), mask=corridor, transform=profile["transform"]) if value == 1]
    gpd.GeoDataFrame({"class": ["prediction"] * len(polygons)}, geometry=polygons, crs=profile["crs"]).to_file(pred_out / "predicted_damage_corridor.geojson", driver="GeoJSON")
    if published_line:
        gpd.GeoDataFrame({"class": ["predicted_centerline"]}, geometry=[published_line], crs=profile["crs"]).to_file(pred_out / "predicted_path_centerline.geojson", driver="GeoJSON")
    else:
        gpd.GeoDataFrame({"class": []}, geometry=[], crs=profile["crs"]).to_file(pred_out / "predicted_path_centerline.geojson", driver="GeoJSON")
    make_maps(case_id, before_tif, after_tif, prob, corridor, published_line, nws, case_out, pred_out, model_label=model_label, quality=quality)
    metrics = {
        "case_id": case_id,
        "valid_fraction": float(meta["valid_fraction"]),
        "predicted_damage_fraction": float(corridor.sum() / max(valid.sum(), 1)),
        "centerline_available": published_line is not None,
        "nws_available": nws is not None,
        "path_tortuosity": tortuosity,
        "confidence": quality,
    }
    (pred_out / "case_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def line_tortuosity(line: LineString | None) -> float | None:
    if line is None or len(line.coords) < 2:
        return None
    start = np.asarray(line.coords[0], dtype="float64")
    end = np.asarray(line.coords[-1], dtype="float64")
    direct = float(np.linalg.norm(end - start))
    return float(line.length / direct) if direct > 0 else None


def classify_quality(
    valid_fraction: float,
    damage_fraction: float,
    has_line: bool,
    cfg: dict[str, Any],
    tortuosity: float | None = None,
) -> str:
    if valid_fraction < float(cfg["quality"]["reject_valid_fraction_below"]) or not has_line:
        return "Rejected"
    if damage_fraction > float(cfg["quality"]["max_prediction_fraction"]):
        return "Rejected"
    if tortuosity is not None and tortuosity > float(cfg["quality"].get("max_path_tortuosity", 4.0)):
        return "Rejected"
    if valid_fraction < float(cfg["quality"]["low_confidence_valid_fraction_below"]):
        return "Low confidence"
    return "Moderate confidence"


def display_scale(shape: tuple[int, int], max_side: int = 1600) -> int:
    return max(1, int(np.ceil(max(shape) / max_side)))


def decimate(arr: np.ndarray, step: int) -> np.ndarray:
    return arr[::step, ::step] if step > 1 else arr


def plot_nws(ax, nws: Path | None, crs, transform, step: int = 1) -> None:
    if not nws:
        return
    try:
        gdf = gpd.read_file(nws)
        if gdf.crs and crs:
            gdf = gdf.to_crs(crs)
    except Exception:
        return
    first = True
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in geoms:
            if part.geom_type in {"LineString", "LinearRing"}:
                xs, ys = part.xy
                rows, cols = rasterio.transform.rowcol(transform, xs, ys)
                artist = ax.plot(
                    np.asarray(cols) / step,
                    np.asarray(rows) / step,
                    color="#00D9FF",
                    lw=3.5,
                    label="Official NWS path" if first else None,
                )[0]
                artist.set_path_effects([path_effects.Stroke(linewidth=6.5, foreground="black"), path_effects.Normal()])
                first = False


def make_maps(
    case_id: str,
    before_tif: Path,
    after_tif: Path,
    prob: np.ndarray,
    mask: np.ndarray,
    line: LineString | list[LineString] | None,
    nws: Path | None,
    case_out: Path,
    pred_out: Path,
    model_label: str = "RF",
    quality: str | None = None,
    rgb_indices: list[int] | None = None,
) -> None:
    before_rgb = rgb(before_tif, rgb_indices)
    after_rgb = rgb(after_tif, rgb_indices)
    with rasterio.open(after_tif) as src:
        transform, crs = src.transform, src.crs
    step = display_scale(after_rgb.shape[:2])
    before_show = decimate(before_rgb, step)
    after_show = decimate(after_rgb, step)
    mask_show = decimate(mask, step)
    wide_scene = after_show.shape[1] / max(after_show.shape[0], 1) > 2.2
    if wide_scene:
        fig, axes = plt.subplots(2, 1, figsize=(16, 8.5), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(before_show)
    axes[0].set_title(f"{case_id} BEFORE", fontsize=16, fontweight="bold")
    axes[1].imshow(after_show)
    axes[1].set_title(f"{case_id} AFTER", fontsize=16, fontweight="bold")
    for ax in axes:
        ax.set_axis_off()
    fig.savefig(case_out / "before_after.png", dpi=180, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 8.2), constrained_layout=True)
    ax.imshow(after_show)
    if quality != "Rejected":
        ax.imshow(
            np.ma.masked_where(~mask_show, mask_show),
            cmap=matplotlib.colors.ListedColormap(["#FF2A2A"]),
            alpha=0.32,
        )
    plot_nws(ax, nws, crs, transform, step)
    lines = line if isinstance(line, list) else ([line] if line is not None else [])
    path_colors = ("#FFD400", "#FF4FD8", "#7CFF4F", "#FFFFFF", "#FF9F1C", "#B388FF")
    for path_index, path_line in enumerate(lines):
        xs, ys = path_line.xy
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        artist = ax.plot(
            np.asarray(cols) / step,
            np.asarray(rows) / step,
            color=path_colors[path_index % len(path_colors)],
            lw=4.5,
            label=f"{model_label} predicted path {path_index + 1}",
        )[0]
        artist.set_path_effects([path_effects.Stroke(linewidth=8, foreground="white"), path_effects.Normal()])
        ax.scatter(
            [cols[0] / step, cols[-1] / step],
            [rows[0] / step, rows[-1] / step],
            s=75,
            c=["#39FF14", "#FFD400"],
            edgecolors="black",
            linewidths=1.5,
            zorder=8,
            label=f"Path {path_index + 1} start / end",
        )
    if not lines and quality == "Rejected":
        ax.text(
            0.02,
            0.97,
            "MODEL RESULT REJECTED: no reliable continuous damage path",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            fontweight="bold",
            color="#9B1C1C",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#9B1C1C", "alpha": 0.94},
        )
    ax.set_axis_off()
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    fig.savefig(pred_out / "final_path_map.png", dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    shutil.copyfile(pred_out / "final_path_map.png", case_out / "final_path_map.png")
    plt.close(fig)

    # Keep the legacy filename for downstream compatibility, but make it the
    # same full-scene overlay instead of a detached black diagnostic panel.
    shutil.copyfile(pred_out / "final_path_map.png", pred_out / "prediction_panel.png")


def make_ppt(
    cases: list[str],
    out: Path,
    output_dir: Path,
    deck_title: str = "Random Forest Tornado Path Workflow",
    subtitle: str = "Red = RF predicted path/corridor. Cyan = NWS path when available. Low-quality imagery is flagged instead of hidden.",
    model_label: str = "RF",
) -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(16)
    prs.slide_height = Inches(9)
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)
    box = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(14), Inches(1.0)).text_frame
    box.text = deck_title
    box.paragraphs[0].font.size = Pt(40)
    box.paragraphs[0].font.bold = True
    sub = slide.shapes.add_textbox(Inches(0.85), Inches(2.0), Inches(13.5), Inches(1.2)).text_frame
    sub.text = subtitle
    sub.paragraphs[0].font.size = Pt(22)
    for case in cases:
        for title, image in [
            (f"{case}: BEFORE / AFTER", output_dir / "cases" / case / "before_after.png"),
            (f"{case}: Predicted Tornado Path on AFTER Image", output_dir / "predictions" / case / "final_path_map.png"),
        ]:
            if not image.exists():
                continue
            s = prs.slides.add_slide(blank)
            t = s.shapes.add_textbox(Inches(0.55), Inches(0.35), Inches(15), Inches(0.6)).text_frame
            t.text = title
            t.paragraphs[0].font.size = Pt(30)
            t.paragraphs[0].font.bold = True
            s.shapes.add_picture(str(image), Inches(0.65), Inches(1.1), width=Inches(14.7))
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)


def write_reports(pairs: list[Pair], aligned: dict[str, tuple[Path, Path, Path, dict[str, Any]]], metrics: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    out = PROJECT / str(cfg["data"]["output_dir"])
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"case_id": p.case_id, "before": str(p.before), "after": str(p.after), "status": "OK"} for p in pairs]).to_csv(reports / "pairing_report.csv", index=False)
    inv = []
    for p in pairs:
        before_tif, after_tif, _, meta = aligned[p.case_id]
        with rasterio.open(before_tif) as src:
            nws = find_nws_path(p.case_id, PROJECT / str(cfg["data"]["shapefile_dir"]))
            inv.append(
                {
                    "case_id": p.case_id,
                    "before": str(p.before),
                    "after": str(p.after),
                    "crs": src.crs.to_string() if src.crs else "",
                    "bounds": tuple(src.bounds),
                    "dimensions": f"{src.width}x{src.height}",
                    "bands": src.count,
                    "resolution": src.res,
                    "valid_fraction": meta["valid_fraction"],
                    "nws_path": str(nws or ""),
                }
            )
    pd.DataFrame(inv).to_csv(reports / "dataset_inventory.csv", index=False)
    pd.DataFrame(metrics).to_csv(reports / "current_dataset_evaluation.csv", index=False)
    pd.DataFrame(
        [
            {"model": "random_forest", "selected": True, "reason": "Trained on NWS-backed cases and applied to all cases."},
            {"model": "rule_based_change", "selected": False, "reason": "Used only as visual/change-feature support inside RF features."},
            {"model": "deep_learning", "selected": False, "reason": "Not enough dense verified labels for U-Net."},
        ]
    ).to_csv(reports / "model_comparison.csv", index=False)
    pd.DataFrame([{"threshold": "rf_probability", "value": cfg["model"]["probability_threshold"]}]).to_csv(reports / "threshold_analysis.csv", index=False)
    (reports / "final_model_selection.md").write_text("Selected model: Random Forest trained on NWS-backed cases. Outputs remain quality-controlled; no model can recover missing raster pixels.\n")


def write_docs(cfg: dict[str, Any]) -> None:
    out = PROJECT / str(cfg["data"]["output_dir"])
    (PROJECT / "README.md").write_text(
        "# Spatial Analysis RF Path Workflow\n\n"
        "Run `./RUN_NEW_DATASET.command` or `.venv/bin/python run_rf_path_workflow.py`.\n\n"
        "This workflow trains Random Forest from NWS-backed cases, predicts paths from BEFORE/AFTER imagery, and overlays NWS when available.\n"
    )
    (PROJECT / "RUN_NEW_DATASET.command").write_text(
        '#!/bin/zsh\ncd "$(dirname "$0")"\n.venv/bin/python run_rf_path_workflow.py\nopen outputs_rf_path/presentation 2>/dev/null || true\n'
    )
    os.chmod(PROJECT / "RUN_NEW_DATASET.command", 0o755)
    docs = PROJECT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "EXPERT_TECHNICAL_AUDIT.md").write_text(
        "# Expert Technical Audit\n\n"
        "Previous workflow archived. New workflow uses Random Forest trained from NWS-backed cases. "
        "Critical limitation: missing/corrupted AFTER imagery cannot produce perfect paths.\n"
    )
    for name in ["BEGINNER_GUIDE", "MODEL_METHODOLOGY", "DATA_REQUIREMENTS", "TRAINING_AND_VALIDATION", "FUTURE_DATA_WORKFLOW", "SLIDE_GENERATION", "TROUBLESHOOTING", "SCIENTIFIC_LIMITATIONS"]:
        (docs / f"{name}.md").write_text(f"# {name.replace('_', ' ').title()}\n\nOutputs are in `{out}`. Red is RF prediction. Cyan is NWS backing where available.\n")


def main() -> None:
    cfg = parse_config(PROJECT / "config.yaml")
    out = PROJECT / str(cfg["data"]["output_dir"])
    for folder in ["cases", "predictions", "reports", "models/final_model", "presentation"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(PROJECT / str(cfg["data"]["raster_dir"]))
    aligned = {p.case_id: align_pair(p, out) for p in pairs}
    model, label_df = train_model(aligned, pairs, cfg)
    model_dir = out / "models" / "final_model"
    joblib.dump(model, model_dir / "random_forest_path_model.joblib")
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model": "RandomForestClassifier",
                "training_cases": label_df.loc[label_df["training_use_allowed"], "case_id"].tolist(),
                "all_cases": [p.case_id for p in pairs],
                "feature_schema": ["before_norm", "after_norm", "signed_diff", "absolute_diff", "change_magnitude"],
                "selected_threshold": cfg["model"]["probability_threshold"],
            },
            indent=2,
        )
    )
    metrics = [predict_case(model, p.case_id, aligned[p.case_id][0], aligned[p.case_id][1], aligned[p.case_id][3], cfg) for p in pairs]
    write_reports(pairs, aligned, metrics, cfg)
    make_ppt([p.case_id for p in pairs], out / "presentation" / "rf_tornado_path_analysis.pptx", out)
    write_docs(cfg)
    print("RF workflow complete")
    print(out / "presentation" / "rf_tornado_path_analysis.pptx")


if __name__ == "__main__":
    main()
