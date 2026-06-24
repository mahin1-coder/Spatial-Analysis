"""Classical baseline training for tornado damage-path detection.

This module is deliberately conservative: it only trains on raster windows that
can be read successfully from both BEFORE and AFTER rasters, and it only fits a
model when the extracted samples contain both damage and background classes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from .config import ProjectConfig
from .preprocessing import pair_registered_rasters

LOGGER = logging.getLogger(__name__)


def _load_label_geometries(config: ProjectConfig) -> list[dict[str, object]]:
    geometries: list[dict[str, object]] = []
    for shp in sorted((config.raw_dir / "shapefiles").rglob("*.shp")):
        role = "polygon" if "poly" in shp.name.lower() else "path" if "path" in shp.name.lower() else "other"
        if role == "other":
            continue
        try:
            gdf = gpd.read_file(shp)
            if gdf.empty:
                continue
            if gdf.crs is None:
                LOGGER.warning("Skipping shapefile without CRS: %s", shp)
                continue
            gdf = gdf.to_crs("EPSG:4326")
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                if role == "path":
                    width_m = row.get("width", 250) if "width" in row else 250
                    try:
                        width_m = float(width_m)
                    except Exception:
                        width_m = 250
                    if width_m <= 0 or not np.isfinite(width_m):
                        width_m = 250
                    # Approximate conversion for WGS84 labels. Later phases should use
                    # local projected CRS buffering, but this is enough for baseline triage.
                    geom = geom.buffer(max(width_m / 2.0, 30.0) / 111_320.0)
                geometries.append({"geometry": geom, "source": str(shp), "role": role})
        except Exception as exc:
            LOGGER.warning("Could not read label shapefile %s: %s", shp, exc)
    return geometries


def _window_geometries(label_geoms: list[dict[str, object]], bounds) -> list[object]:
    left, bottom, right, top = bounds
    footprint = box(left, bottom, right, top)
    return [item["geometry"] for item in label_geoms if item["geometry"].intersects(footprint)]


def _sample_window(
    tor_id: str,
    before_src,
    after_src,
    window,
    label_geoms: list[dict[str, object]],
    rng: np.random.Generator,
    max_pos: int,
    max_neg: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    stats = {
        "tornado_id": tor_id,
        "window": str(window),
        "read_status": "OK",
        "positive_pixels": 0,
        "negative_pixels": 0,
        "sampled_positive": 0,
        "sampled_negative": 0,
        "error": "",
    }
    try:
        before = np.ma.filled(before_src.read(window=window, masked=True).astype("float32"), np.nan)
        after = np.ma.filled(after_src.read(window=window, masked=True).astype("float32"), np.nan)
    except Exception as exc:
        stats["read_status"] = "READ_ERROR"
        stats["error"] = str(exc)
        return np.empty((0, 0), dtype="float32"), np.empty((0,), dtype="uint8"), stats

    if before.shape != after.shape or before.shape[0] != after.shape[0]:
        stats["read_status"] = "SHAPE_MISMATCH"
        stats["error"] = f"before {before.shape}, after {after.shape}"
        return np.empty((0, 0), dtype="float32"), np.empty((0,), dtype="uint8"), stats

    bounds = before_src.window_bounds(window)
    geoms = _window_geometries(label_geoms, bounds)
    if not geoms:
        mask = np.zeros((int(window.height), int(window.width)), dtype="uint8")
    else:
        mask = rasterize(
            [(geom, 1) for geom in geoms],
            out_shape=(int(window.height), int(window.width)),
            transform=window_transform(window, before_src.transform),
            fill=0,
            dtype="uint8",
        )

    valid = np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0)
    pos = np.flatnonzero((mask == 1) & valid)
    neg = np.flatnonzero((mask == 0) & valid)
    stats["positive_pixels"] = int(pos.size)
    stats["negative_pixels"] = int(neg.size)

    if pos.size:
        pos = rng.choice(pos, min(max_pos, pos.size), replace=False)
    if neg.size:
        neg = rng.choice(neg, min(max_neg, neg.size), replace=False)
    selected = np.concatenate([pos, neg])
    if selected.size == 0:
        return np.empty((0, 0), dtype="float32"), np.empty((0,), dtype="uint8"), stats

    before_flat = before.reshape(before.shape[0], -1).T[selected]
    after_flat = after.reshape(after.shape[0], -1).T[selected]
    diff_flat = after_flat - before_flat
    X = np.hstack([before_flat, after_flat, diff_flat]).astype("float32")
    y = mask.reshape(-1)[selected].astype("uint8")
    stats["sampled_positive"] = int((y == 1).sum())
    stats["sampled_negative"] = int((y == 0).sum())
    return X, y, stats


def train_random_forest_baseline(config: ProjectConfig) -> pd.DataFrame:
    """Train a Random Forest baseline using readable windows and NWS labels."""

    config.ensure_phase1_dirs()
    out_dir = config.outputs_dir / "models" / "random_forest_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_geoms = _load_label_geometries(config)
    pairs = pair_registered_rasters(config)
    pairs.to_csv(config.inventory_dir / "registered_raster_pairs.csv", index=False)
    rng = np.random.default_rng(42)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    sample_cases: list[str] = []
    window_rows: list[dict[str, object]] = []

    for pair in pairs.to_dict("records"):
        if pair["pair_status"] != "OK":
            continue
        tor_id = pair["tornado_id"]
        try:
            with rasterio.open(pair["before_path"]) as before_src, rasterio.open(pair["after_path"]) as after_src:
                if before_src.crs != after_src.crs or before_src.transform != after_src.transform or before_src.shape != after_src.shape:
                    LOGGER.warning("Skipping unaligned pair for baseline: %s", tor_id)
                    continue
                for _, window in after_src.block_windows(1):
                    X, y, stats = _sample_window(tor_id, before_src, after_src, window, label_geoms, rng, 1200, 1200)
                    window_rows.append(stats)
                    if y.size:
                        X_parts.append(X)
                        y_parts.append(y)
                        sample_cases.extend([tor_id] * y.size)
        except Exception as exc:
            window_rows.append({"tornado_id": tor_id, "read_status": "OPEN_ERROR", "error": str(exc)})

    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(out_dir / "training_window_report.csv", index=False)

    report: dict[str, object] = {
        "label_geometry_count": len(label_geoms),
        "sample_count": 0,
        "positive_samples": 0,
        "negative_samples": 0,
        "trained": False,
        "reason": "",
    }
    if not X_parts:
        report["reason"] = "No readable labeled samples could be extracted from the provided rasters."
        (out_dir / "training_report.json").write_text(json.dumps(report, indent=2))
        return pd.DataFrame([report])

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    cases = np.array(sample_cases)
    report.update(
        {
            "sample_count": int(y_all.size),
            "positive_samples": int((y_all == 1).sum()),
            "negative_samples": int((y_all == 0).sum()),
        }
    )
    if len(np.unique(y_all)) < 2:
        report["reason"] = "Extracted samples contain only one class; cannot train a supervised damage model."
        (out_dir / "training_report.json").write_text(json.dumps(report, indent=2))
        pd.DataFrame({"case": cases, "label": y_all}).to_csv(out_dir / "sample_manifest.csv", index=False)
        return pd.DataFrame([report])

    unique_cases = sorted(np.unique(cases), key=lambda v: int(v[3:]))
    test_cases = set(unique_cases[-max(1, len(unique_cases) // 4) :])
    train_mask = np.array([case not in test_cases for case in cases])
    test_mask = ~train_mask
    if len(np.unique(y_all[train_mask])) < 2 or len(np.unique(y_all[test_mask])) < 2:
        # Fall back to deterministic sample split if spatial split is class-degenerate.
        idx = rng.permutation(len(y_all))
        test_count = max(1, int(len(idx) * 0.25))
        test_idx = idx[:test_count]
        train_idx = idx[test_count:]
    else:
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_all[train_idx], y_all[train_idx])
    pred = model.predict(X_all[test_idx])

    metrics = {
        "precision": float(precision_score(y_all[test_idx], pred, zero_division=0)),
        "recall": float(recall_score(y_all[test_idx], pred, zero_division=0)),
        "dice_f1": float(f1_score(y_all[test_idx], pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_all[test_idx], pred).tolist(),
        "classification_report": classification_report(y_all[test_idx], pred, zero_division=0, output_dict=True),
    }
    report.update({"trained": True, "reason": "", "metrics": metrics, "test_cases": sorted(test_cases)})
    joblib.dump(model, out_dir / "random_forest_damage_baseline.joblib")
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({"case": cases, "label": y_all}).to_csv(out_dir / "sample_manifest.csv", index=False)
    return pd.DataFrame([report])
