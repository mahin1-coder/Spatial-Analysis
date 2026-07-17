"""Run trained baseline model predictions on readable raster windows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import transform as window_transform
from sklearn.metrics import f1_score, precision_score, recall_score

from .baseline_model import _load_label_geometries, _window_geometries
from .config import ProjectConfig
from .postprocessing import write_vector_products
from .preprocessing import pair_registered_rasters

LOGGER = logging.getLogger(__name__)
NODATA = 255


def _window_features(before_src, after_src, window) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        before = np.ma.filled(before_src.read(window=window, masked=True).astype("float32"), np.nan)
        after = np.ma.filled(after_src.read(window=window, masked=True).astype("float32"), np.nan)
    except Exception as exc:
        return np.empty((0, 0), dtype="float32"), np.empty((0,), dtype=bool), str(exc)

    if before.shape != after.shape:
        return np.empty((0, 0), dtype="float32"), np.empty((0,), dtype=bool), f"shape mismatch: {before.shape} vs {after.shape}"

    valid = np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0)
    if not valid.any():
        return np.empty((0, 0), dtype="float32"), valid.reshape(-1), "no finite pixels"

    before_flat = before.reshape(before.shape[0], -1).T
    after_flat = after.reshape(after.shape[0], -1).T
    diff_flat = after_flat - before_flat
    X = np.hstack([before_flat, after_flat, diff_flat]).astype("float32")
    return X[valid.reshape(-1)], valid.reshape(-1), ""


def _window_label(before_src, window, label_geoms: list[dict[str, object]]) -> np.ndarray:
    geoms = _window_geometries(label_geoms, before_src.window_bounds(window), before_src.crs)
    shape = (int(window.height), int(window.width))
    if not geoms:
        return np.zeros(shape, dtype="uint8")
    return rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=shape,
        transform=window_transform(window, before_src.transform),
        fill=0,
        dtype="uint8",
    )


def _save_prediction_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display = np.ma.masked_where(mask == NODATA, mask)
    plt.figure(figsize=(8, 5))
    plt.imshow(display, cmap="magma", vmin=0, vmax=1)
    plt.title("Predicted Tornado Damage Mask")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def predict_on_pair(
    model,
    before_path: Path,
    after_path: Path,
    out_dir: Path,
    label_geoms: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Run the trained model on one BEFORE/AFTER raster pair and write outputs to out_dir.

    This is the shared core used both by the curated TOR## dataset pipeline
    (run_baseline_predictions) and by ad-hoc analysis of a new, arbitrary
    BEFORE/AFTER pair that isn't part of that dataset.
    """

    label_geoms = label_geoms or []
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_pixels = 0
    predicted_damage = 0
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    window_rows: list[dict[str, object]] = []

    with rasterio.open(before_path) as before_src, rasterio.open(after_path) as after_src:
        profile = before_src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=NODATA, compress="deflate", BIGTIFF="IF_SAFER")
        probability_profile = before_src.profile.copy()
        probability_profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", BIGTIFF="IF_SAFER")
        pred_full = np.full((before_src.height, before_src.width), NODATA, dtype="uint8")
        probability_full = np.full((before_src.height, before_src.width), np.nan, dtype="float32")

        for _, window in after_src.block_windows(1):
            X, valid_flat, error = _window_features(before_src, after_src, window)
            if X.size == 0:
                window_rows.append({"window": str(window), "status": "SKIPPED", "valid_pixels": 0, "error": error})
                continue
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:, 1].astype("float32")
                pred = (proba >= 0.5).astype("uint8")
            else:
                pred = model.predict(X).astype("uint8")
                proba = pred.astype("float32")
            window_pred_flat = np.full(valid_flat.shape, NODATA, dtype="uint8")
            window_proba_flat = np.full(valid_flat.shape, np.nan, dtype="float32")
            window_pred_flat[valid_flat] = pred
            window_proba_flat[valid_flat] = proba
            window_pred = window_pred_flat.reshape((int(window.height), int(window.width)))
            window_proba = window_proba_flat.reshape((int(window.height), int(window.width)))
            row_slice, col_slice = window.toslices()
            pred_full[row_slice, col_slice] = window_pred
            probability_full[row_slice, col_slice] = window_proba

            label = _window_label(before_src, window, label_geoms)
            valid_label = label.reshape(-1)[valid_flat]
            y_true_parts.append(valid_label)
            y_pred_parts.append(pred)
            valid_pixels += int(pred.size)
            predicted_damage += int((pred == 1).sum())
            window_rows.append(
                {
                    "window": str(window),
                    "status": "OK",
                    "valid_pixels": int(pred.size),
                    "predicted_damage_pixels": int((pred == 1).sum()),
                    "label_damage_pixels": int((valid_label == 1).sum()),
                    "error": "",
                }
            )

        pred_tif = out_dir / "prediction_mask.tif"
        with rasterio.open(pred_tif, "w", **profile) as dst:
            dst.write(pred_full, 1)
        proba_tif = out_dir / "predicted_probability.tif"
        with rasterio.open(proba_tif, "w", **probability_profile) as dst:
            dst.write(probability_full, 1)
        pred_png = out_dir / "prediction_mask.png"
        _save_prediction_png(pred_full, pred_png)
        vector_products = write_vector_products(pred_tif, out_dir)

    if y_true_parts:
        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        dice_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        label_damage = int((y_true == 1).sum())
    else:
        precision = recall = dice_f1 = np.nan
        label_damage = 0

    pd.DataFrame(window_rows).to_csv(out_dir / "prediction_window_report.csv", index=False)
    return {
        "status": "OK" if valid_pixels else "NO_VALID_PIXELS",
        "valid_pixels": valid_pixels,
        "predicted_damage_pixels": predicted_damage,
        "label_damage_pixels": label_damage,
        "precision": precision,
        "recall": recall,
        "dice_f1": dice_f1,
        "prediction_tif": str(pred_tif),
        "predicted_probability": str(proba_tif),
        "prediction_png": str(pred_png),
        **vector_products,
        "error": "",
    }


def run_baseline_predictions(config: ProjectConfig) -> pd.DataFrame:
    """Apply the trained Random Forest baseline to all readable raster windows."""

    model_path = config.outputs_dir / "models" / "random_forest_baseline" / "random_forest_damage_baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Train the baseline first: missing {model_path}")

    model = joblib.load(model_path)
    pairs = pair_registered_rasters(config)
    label_geoms = _load_label_geometries(config)
    out_root = config.outputs_dir / "predictions" / "random_forest_baseline"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for pair in pairs.to_dict("records"):
        tor_id = pair["tornado_id"]
        if pair["pair_status"] != "OK":
            rows.append({"tornado_id": tor_id, "status": "SKIPPED", "error": pair.get("warnings", "")})
            continue

        out_dir = out_root / tor_id
        try:
            result = predict_on_pair(model, Path(pair["before_path"]), Path(pair["after_path"]), out_dir, label_geoms)
            rows.append({"tornado_id": tor_id, **result})
        except Exception as exc:
            LOGGER.warning("Prediction failed for %s: %s", tor_id, exc)
            rows.append({"tornado_id": tor_id, "status": "FAILED", "error": str(exc)})

    summary = pd.DataFrame(rows)
    summary_path = out_root / "prediction_summary.csv"
    summary.to_csv(summary_path, index=False)
    (out_root / "prediction_summary.json").write_text(json.dumps(rows, indent=2, default=str))
    return summary
