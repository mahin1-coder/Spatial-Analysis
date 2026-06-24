"""NWS shapefile overlay and validation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import plotting_extent
from shapely.geometry import box

from .config import ProjectConfig
from .inventory import infer_shapefile_role
from .utils import detect_tornado_id

LOGGER = logging.getLogger(__name__)


def _candidate_shapefiles(config: ProjectConfig) -> list[Path]:
    roots = [config.raw_dir / "shapefiles", config.raw_dir / "source_mirror"]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend([p for p in root.rglob("*.shp")])
    return sorted(set(paths))


def _load_readable_shapes(config: ProjectConfig) -> list[dict[str, object]]:
    shapes = []
    for path in _candidate_shapefiles(config):
        role = infer_shapefile_role(path)
        if role not in {"tornado path", "tornado polygon"}:
            continue
        try:
            gdf = gpd.read_file(path, engine="fiona")
            if gdf.empty:
                LOGGER.info("Skipping empty shapefile: %s", path)
                continue
            shapes.append({"path": path, "role": role, "tornado_id": detect_tornado_id(path), "gdf": gdf})
        except Exception as exc:
            LOGGER.warning("Skipping unreadable shapefile %s: %s", path, exc)
    return shapes


def _raster_preview(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float], object]:
    with rasterio.open(path) as src:
        band = src.read(1, masked=True).astype("float32")
        arr = np.ma.filled(band, np.nan)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            low, high = np.nanpercentile(finite, [2, 98])
            if high > low:
                arr = np.clip((arr - low) / (high - low), 0, 1)
        return arr, plotting_extent(src), src.crs


def _raster_extent_only(path: Path) -> tuple[tuple[float, float, float, float], object]:
    with rasterio.open(path) as src:
        return plotting_extent(src), src.crs


def _plot_shape_overlay(
    tor_id: str,
    label: str,
    raster_path: Path,
    shapes: list[dict[str, object]],
    out_dir: Path,
    allow_preview: bool,
) -> dict[str, object]:
    matched = [s for s in shapes if s["tornado_id"] in {None, tor_id}]
    preview_status = "raster_preview"
    try:
        if allow_preview:
            preview, extent, raster_crs = _raster_preview(raster_path)
        else:
            raise RuntimeError("using metadata-only overlay")
    except Exception:
        preview = None
        extent, raster_crs = _raster_extent_only(raster_path)
        preview_status = "metadata_extent_only"

    plt.figure(figsize=(9, 6))
    if preview is not None:
        plt.imshow(preview, extent=extent, cmap="gray")
    bounds_poly = box(extent[0], extent[2], extent[1], extent[3])
    bounds_gdf = gpd.GeoDataFrame(geometry=[bounds_poly], crs=raster_crs)
    bounds_gdf.boundary.plot(ax=plt.gca(), color="yellow", linewidth=2)

    overlay_count = 0
    overlap_count = 0
    for shape in matched:
        gdf = shape["gdf"]
        if raster_crs and gdf.crs and gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        has_overlap = bool(gdf.intersects(bounds_poly).any())
        overlap_count += int(has_overlap)
        style = (
            {"color": "cyan", "linewidth": 1.5}
            if shape["role"] == "tornado path"
            else {"edgecolor": "red", "facecolor": "none", "linewidth": 1.0}
        )
        gdf.plot(ax=plt.gca(), **style)
        overlay_count += 1

    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.title(f"{tor_id} {label.upper()} NWS Overlay")
    plt.axis("off")
    plt.tight_layout()
    out_path = out_dir / f"{label}_overlay.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    return {
        "tornado_id": tor_id,
        "raster": label,
        "overlay_status": "OK",
        "overlay_path": str(out_path),
        "candidate_shapes": len(matched),
        "overlapping_shapes": overlap_count,
        "preview_status": preview_status,
        "error": "" if matched else "no candidate NWS path/polygon shapefiles found",
    }


def run_overlays(config: ProjectConfig) -> pd.DataFrame:
    """Overlay NWS path/polygon shapefiles on aligned BEFORE/AFTER/difference rasters."""

    preprocessing_path = config.reports_dir / "preprocessing_summary.csv"
    if not preprocessing_path.exists():
        raise FileNotFoundError("Run preprocessing before overlays.")

    preprocessed = pd.read_csv(preprocessing_path)
    shapes = _load_readable_shapes(config)
    rows: list[dict[str, object]] = []

    for item in preprocessed.to_dict("records"):
        tor_id = item["tornado_id"]
        out_dir = config.outputs_dir / "overlays" / tor_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if item.get("status") == "OK":
                raster_targets = [
                    ("before", Path(item["before_aligned_path"])),
                    ("after", Path(item["after_aligned_path"])),
                    ("difference", config.outputs_dir / "plots" / tor_id / "difference_stack.tif"),
                ]
                allow_preview = True
            else:
                raster_targets = [
                    ("before_extent", Path(item["before_path"])),
                    ("after_extent", Path(item["after_path"])),
                ]
                allow_preview = False
            for label, raster_path in raster_targets:
                if not raster_path.exists():
                    continue
                rows.append(_plot_shape_overlay(tor_id, label, raster_path, shapes, out_dir, allow_preview))
        except Exception as exc:
            LOGGER.warning("Overlay failed for %s: %s", tor_id, exc)
            rows.append({"tornado_id": tor_id, "overlay_status": "FAILED", "error": str(exc)})

    overlay_summary = pd.DataFrame(rows)
    overlay_summary.to_csv(config.reports_dir / "overlay_summary.csv", index=False)
    return overlay_summary
