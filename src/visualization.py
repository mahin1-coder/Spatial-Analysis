"""Slide-quality visualizations for tornado damage-path model outputs.

Design intent (per project requirements):
  - Background: true-color satellite composite where the source imagery is
    readable. Built from the surface-reflectance bands (Blue/Green/Red) of
    the AFTER raster.
  - Red:  model-predicted damage.
  - Cyan: official NWS damage path/polygon, drawn as vector geometry (not a
    rasterized blob) for a crisp "research slide" look.
  - Gray: pixels the model could not score, because the BEFORE/AFTER tiles
    were corrupted/truncated. This is taken directly from the prediction
    raster's NODATA flag, so the figure never implies a result where none
    exists.
  - Each map is cropped tightly around the NWS path (falling back to the
    predicted-damage footprint, then to the readable-data footprint) so the
    output is a legible close-up instead of a mostly-empty wide shot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from .baseline_model import _load_label_geometries, _window_geometries
from .config import ProjectConfig
from .predict import NODATA
from .preprocessing import pair_registered_rasters

LOGGER = logging.getLogger(__name__)

# Surface-reflectance band order confirmed against the source rasters:
# band 1=Blue, 2=Green, 3=Red, 4=NIR, 5=SWIR1, 6=SWIR2 (Landsat-style stack).
_TRUE_COLOR_BAND_INDEX = (2, 1, 0)  # (Red, Green, Blue) -> 0-based band index

_RED = "#FF3B30"
_CYAN = "#00D9FF"
_UNREADABLE_GRAY = "#C9C9C9"


def _read_after_composite(after_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the AFTER raster and build a stretched true-color (or grayscale) composite.

    Returns (rgb float32 in [0, 1] shape (H, W, 3), readable_mask shape (H, W)).
    Unreadable tiles/pixels are left as NaN in rgb and False in readable_mask -
    they are never filled with a plausible-looking value.
    """

    with rasterio.open(after_path) as src:
        height, width = src.height, src.width
        use_color = src.count >= 3
        band_idx = list(_TRUE_COLOR_BAND_INDEX) if use_color else [0]
        n_channels = len(band_idx)
        channels = np.full((n_channels, height, width), np.nan, dtype="float32")
        readable = np.zeros((height, width), dtype=bool)

        for _, window in src.block_windows(1):
            try:
                raw = src.read(window=window, masked=True).astype("float32")
                raw = np.ma.filled(raw, np.nan)
            except Exception:
                continue  # leave NaN/unreadable for this block
            row_slice, col_slice = window.toslices()
            pixel_valid = np.isfinite(raw[band_idx]).all(axis=0)
            for ch, b in enumerate(band_idx):
                channels[ch][row_slice, col_slice] = raw[b]
            readable[row_slice, col_slice] = pixel_valid

        def stretch(band: np.ndarray) -> np.ndarray:
            finite = band[np.isfinite(band) & readable]
            if finite.size == 0:
                return np.zeros_like(band)
            low, high = np.nanpercentile(finite, [2, 98])
            if high <= low:
                return np.zeros_like(band)
            out = np.clip((band - low) / (high - low), 0, 1)
            return np.nan_to_num(out, nan=0.0)

        stretched = [stretch(channels[ch]) for ch in range(n_channels)]
        if use_color:
            rgb = np.stack(stretched, axis=-1)
        else:
            rgb = np.repeat(stretched[0][:, :, None], 3, axis=2)
        rgb = np.clip(rgb ** 0.85, 0, 1)  # mild gamma lift, satellite reflectance reads dark linearly
        rgb[~readable] = np.nan
        return rgb.astype("float32"), readable


def _label_geometries_for_bounds(label_geoms: list[dict[str, object]], bounds) -> list[object]:
    return _window_geometries(label_geoms, bounds)


def _geometry_pixel_parts(geom, transform) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Convert a shapely geometry to a list of (cols, rows, is_polygon) pixel-space rings."""

    parts: list[tuple[np.ndarray, np.ndarray, bool]] = []
    geom_type = geom.geom_type
    if geom_type in {"LineString", "LinearRing"}:
        xs, ys = zip(*geom.coords)
        rows, cols = rowcol(transform, xs, ys)
        parts.append((np.array(cols), np.array(rows), False))
    elif geom_type == "MultiLineString":
        for part in geom.geoms:
            parts.extend(_geometry_pixel_parts(part, transform))
    elif geom_type == "Polygon":
        xs, ys = zip(*geom.exterior.coords)
        rows, cols = rowcol(transform, xs, ys)
        parts.append((np.array(cols), np.array(rows), True))
    elif geom_type == "MultiPolygon":
        for part in geom.geoms:
            parts.extend(_geometry_pixel_parts(part, transform))
    return parts


def _geometries_pixel_bbox(geoms: list[object], transform, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    height, width = shape
    rows_all: list[float] = []
    cols_all: list[float] = []
    for geom in geoms:
        for cols, rows, _ in _geometry_pixel_parts(geom, transform):
            rows_all.extend(rows.tolist())
            cols_all.extend(cols.tolist())
    if not rows_all:
        return None
    r0 = max(int(min(rows_all)), 0)
    r1 = min(int(max(rows_all)) + 1, height)
    c0 = max(int(min(cols_all)), 0)
    c1 = min(int(max(cols_all)) + 1, width)
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, r1, c0, c1


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _pad_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], margin_frac: float = 0.25, min_margin: int = 40) -> tuple[slice, slice]:
    r0, r1, c0, c1 = bbox
    height, width = shape
    margin_r = max(int((r1 - r0) * margin_frac), min_margin)
    margin_c = max(int((c1 - c0) * margin_frac), min_margin)
    r0 = max(r0 - margin_r, 0)
    r1 = min(r1 + margin_r, height)
    c0 = max(c0 - margin_c, 0)
    c1 = min(c1 + margin_c, width)
    return slice(r0, r1), slice(c0, c1)


def _intersect_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    r1 = min(a[1], b[1])
    c0 = max(a[2], b[2])
    c1 = min(a[3], b[3])
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, r1, c0, c1


def _choose_crop(
    label_geoms_here: list[object],
    transform,
    pred: np.ndarray,
    shape: tuple[int, int],
) -> tuple[slice, slice, str]:
    """Pick the tightest informative crop: NWS path (clipped to readable data) > predicted damage > readable extent.

    A tornado path geometry can run for many kilometers while a raster chip only
    covers a small window of it, so the path's own bounding box is clipped against
    the readable/predicted-data extent to avoid a frame that's mostly empty gray.
    """

    readable_bbox = _mask_bbox(pred != NODATA)
    label_bbox = _geometries_pixel_bbox(label_geoms_here, transform, shape)

    if label_bbox is not None and readable_bbox is not None:
        clipped = _intersect_bbox(label_bbox, readable_bbox)
        if clipped is not None:
            return *_pad_bbox(clipped, shape), "cropped to NWS path/polygon extent (clipped to readable imagery)"
        # The path crosses this raster's footprint but not its readable pixels -
        # prefer the legible readable extent over a mostly-empty path bounding box.
        return (
            *_pad_bbox(readable_bbox, shape),
            "cropped to readable-data extent (NWS path crosses this scene but outside readable imagery)",
        )
    if label_bbox is not None:
        return *_pad_bbox(label_bbox, shape), "cropped to NWS path/polygon extent"

    bbox = _mask_bbox(pred == 1)
    if bbox is not None:
        return *_pad_bbox(bbox, shape), "cropped to predicted-damage extent (no NWS geometry in scene)"

    if readable_bbox is not None:
        return *_pad_bbox(readable_bbox, shape), "cropped to readable-data extent (no prediction or NWS geometry in scene)"

    return slice(0, shape[0]), slice(0, shape[1]), "no crop available; showing full extent"


def _save_showcase(
    tor_id: str,
    rgb: np.ndarray,
    readable: np.ndarray,
    pred: np.ndarray,
    crop: tuple[slice, slice],
    crop_note: str,
    label_geoms_here: list[object],
    transform,
    metrics: dict[str, object],
    has_geometry: bool,
    out_path: Path,
) -> None:
    r_slice, c_slice = crop
    rgb_crop = rgb[r_slice, c_slice]
    readable_crop = readable[r_slice, c_slice]
    pred_crop = pred[r_slice, c_slice]

    display = np.where(np.isfinite(rgb_crop), rgb_crop, 0.0)
    unreadable_for_display = (~readable_crop) | (pred_crop == NODATA)

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="white")
    ax.imshow(display)

    gray_overlay = np.ma.masked_where(~unreadable_for_display, unreadable_for_display.astype("uint8"))
    ax.imshow(gray_overlay, cmap=ListedColormap([_UNREADABLE_GRAY]), alpha=1.0, interpolation="nearest")

    pred_overlay = np.ma.masked_where(pred_crop != 1, pred_crop)
    ax.imshow(pred_overlay, cmap=ListedColormap([_RED]), alpha=0.55, interpolation="nearest")
    if (pred_crop == 1).any():
        ax.contour(pred_crop == 1, levels=[0.5], colors=_RED, linewidths=0.7)

    for geom in label_geoms_here:
        for cols, rows, is_polygon in _geometry_pixel_parts(geom, transform):
            cols_c = cols - c_slice.start
            rows_c = rows - r_slice.start
            if is_polygon:
                ax.fill(cols_c, rows_c, facecolor=_CYAN, edgecolor=_CYAN, alpha=0.18, linewidth=2.0)
            else:
                ax.plot(cols_c, rows_c, color=_CYAN, linewidth=2.4, solid_capstyle="round")

    ax.set_xlim(0, c_slice.stop - c_slice.start)
    ax.set_ylim(r_slice.stop - r_slice.start, 0)
    ax.set_title(f"{tor_id} — Tornado Damage Path Prediction", fontsize=18, fontweight="bold", pad=12)
    ax.set_axis_off()

    readable_pixels = int(metrics.get("valid_pixels", 0) or 0)
    predicted_damage = int(metrics.get("predicted_damage_pixels", 0) or 0)
    label_damage = int(metrics.get("label_damage_pixels", 0) or 0)
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    dice = metrics.get("dice_f1")
    total_crop_pixels = (r_slice.stop - r_slice.start) * (c_slice.stop - c_slice.start)
    readable_frac_crop = float((~unreadable_for_display).sum()) / total_crop_pixels if total_crop_pixels else 0.0

    def fmt(value: object) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "n/a"

    subtitle_lines = [
        f"Readable pixels (full scene): {readable_pixels:,}    "
        f"Predicted damage: {predicted_damage:,}    "
        f"NWS-labeled damage: {label_damage:,}",
        f"Precision: {fmt(precision)}    Recall: {fmt(recall)}    Dice/F1: {fmt(dice)}    "
        f"Readable in view: {readable_frac_crop:.0%}",
        f"Note: {crop_note}. Random Forest baseline — trained only on readable BEFORE/AFTER windows.",
    ]
    if not has_geometry:
        subtitle_lines.append("No NWS path/polygon geometry intersects this raster's extent.")
    ax.text(
        0.0,
        -0.03,
        "\n".join(subtitle_lines),
        transform=ax.transAxes,
        fontsize=10,
        color="#222222",
        va="top",
    )

    legend_handles = [
        Patch(facecolor=_RED, edgecolor="none", label="Model predicted damage", alpha=0.85),
        Patch(facecolor=_UNREADABLE_GRAY, edgecolor="none", label="Unreadable / corrupted imagery"),
    ]
    if has_geometry:
        legend_handles.insert(1, Patch(facecolor=_CYAN, edgecolor="none", label="NWS official damage path", alpha=0.6))
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _build_contact_sheet(entries: list[tuple[str, Path, float]], out_path: Path) -> None:
    """entries: list of (tornado_id, showcase_png_path, dice_f1)."""

    if not entries:
        return
    cols = 3
    rows = (len(entries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.4), facecolor="white")
    axes = np.atleast_2d(axes)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.set_axis_off()
        if idx >= len(entries):
            continue
        tor_id, png_path, dice = entries[idx]
        img = plt.imread(png_path)
        ax.imshow(img)
        dice_text = f"{dice:.3f}" if np.isfinite(dice) else "n/a"
        ax.set_title(f"{tor_id}  (Dice/F1: {dice_text})", fontsize=11, fontweight="bold")
    fig.suptitle("Tornado Damage Path Detection — Random Forest Baseline Showcase", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_showcase_map(
    case_id: str,
    pred_tif: Path,
    after_path: Path,
    label_geoms_all: list[object],
    metrics: dict[str, object],
    out_path: Path,
) -> dict[str, object]:
    """Render one slide-quality showcase map for a prediction raster + its AFTER imagery.

    Shared by the curated TOR## dataset showcase loop and by ad-hoc analysis of
    a new BEFORE/AFTER pair that isn't part of that dataset.
    """

    with rasterio.open(pred_tif) as pred_src:
        pred = pred_src.read(1)
        transform = pred_src.transform
        bounds = pred_src.bounds
        shape = (pred_src.height, pred_src.width)

    rgb, readable = _read_after_composite(after_path)
    label_geoms_here = _label_geometries_for_bounds([{"geometry": g} for g in label_geoms_all], bounds)
    crop_r, crop_c, crop_note = _choose_crop(label_geoms_here, transform, pred, shape)

    _save_showcase(
        case_id,
        rgb,
        readable,
        pred,
        (crop_r, crop_c),
        crop_note,
        label_geoms_here,
        transform,
        metrics,
        has_geometry=bool(label_geoms_here),
        out_path=out_path,
    )
    return {"has_nws_geometry": bool(label_geoms_here), "crop_note": crop_note}


def create_prediction_showcase(config: ProjectConfig) -> pd.DataFrame:
    """Create slide-quality prediction maps from existing model outputs."""

    pred_root = config.outputs_dir / "predictions" / "random_forest_baseline"
    summary_path = pred_root / "prediction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run --mode predict before creating showcase figures.")

    summary = pd.read_csv(summary_path)
    pairs = pair_registered_rasters(config)
    pair_by_id = {row["tornado_id"]: row for row in pairs.to_dict("records")}
    label_geoms_all = [item["geometry"] for item in _load_label_geometries(config)]
    showcase_root = pred_root / "showcase_maps"

    rows: list[dict[str, object]] = []
    contact_entries: list[tuple[str, Path, float]] = []

    for item in summary.to_dict("records"):
        tor_id = item["tornado_id"]
        pred_path = Path(item.get("prediction_tif", ""))
        pair = pair_by_id.get(tor_id)
        if not pair or not pred_path.exists() or item.get("status") != "OK":
            rows.append({"tornado_id": tor_id, "showcase_path": "", "status": "SKIPPED"})
            continue
        try:
            out_path = showcase_root / tor_id / "showcase_prediction_map.png"
            render_info = render_showcase_map(
                tor_id, pred_path, Path(pair["after_path"]), label_geoms_all, item, out_path
            )
            rows.append({"tornado_id": tor_id, "showcase_path": str(out_path), "status": "OK", **render_info})
            dice_value = item.get("dice_f1")
            dice_float = float(dice_value) if dice_value is not None and not pd.isna(dice_value) else np.nan
            contact_entries.append((tor_id, out_path, dice_float))
        except Exception as exc:
            LOGGER.warning("Showcase visualization failed for %s: %s", tor_id, exc)
            rows.append({"tornado_id": tor_id, "showcase_path": "", "status": "FAILED", "error": str(exc)})

    contact_entries.sort(key=lambda e: int(e[0][3:]))
    _build_contact_sheet(contact_entries, showcase_root / "showcase_contact_sheet.png")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(showcase_root / "showcase_summary.csv", index=False)
    return out_df
