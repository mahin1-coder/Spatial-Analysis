"""Tornado path atlas matching the reference figure style.

The reference figure (Tornado_path.jpg, bundled with the original data
package) draws each NWS damage-path segment as a colored line directly on
satellite imagery, color-coded by EF (Enhanced Fujita) rating, with a
caption showing EF scale, storm date, and start coordinates. This module
reproduces that exact look using the real efscale/stormdate/startlat/
startlon fields in the NWS path shapefiles - one combined multi-panel
figure across all tornado cases.

Cases whose path geometry doesn't actually cross any readable pixels get a
plain satellite panel with an honest "no path visible" caption instead of
an invented line.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import box

from .config import ProjectConfig
from .preprocessing import pair_registered_rasters
from .visualization import _read_after_composite

LOGGER = logging.getLogger(__name__)

EF_COLORS = {
    "EF0": "#FFD400",  # bright yellow, not black - black is invisible against dark satellite imagery
    "EF1": "#1F77B4",
    "EF2": "#8B00FF",
    "EF3": "#2CA02C",
    "EF4": "#E03131",
    "EF5": "#7A0C0C",
    "EFU": "#888888",
}


def _load_shapefiles(config: ProjectConfig, filename: str) -> gpd.GeoDataFrame:
    shapefiles = sorted((config.raw_dir / "shapefiles").rglob(filename))
    frames = []
    for shp in shapefiles:
        try:
            gdf = gpd.read_file(shp)
            if gdf.empty or gdf.crs is None:
                continue
            frames.append(gdf.to_crs("EPSG:4326"))
        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", shp, exc)
    if not frames:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))


def _load_path_segments(config: ProjectConfig) -> gpd.GeoDataFrame:
    """LineString damage paths (preferred - matches the reference figure exactly)."""
    return _load_shapefiles(config, "nws_dat_damage_paths.shp")


def _load_damage_polygons(config: ProjectConfig) -> gpd.GeoDataFrame:
    """Polygon damage-area fallback for cases where no path line exists."""
    return _load_shapefiles(config, "nws_dat_damage_polys.shp")


def _features_for_bounds(features: gpd.GeoDataFrame, bounds) -> gpd.GeoDataFrame:
    if features.empty:
        return features
    footprint = box(*bounds)
    return features[features.geometry.intersects(footprint)]


def _geom_pixel_parts(geom, transform) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert a geometry (possibly Multi*) to one or more (cols, rows) pixel-space arrays."""

    if geom.geom_type in ("MultiLineString", "MultiPolygon", "GeometryCollection"):
        parts = []
        for part in geom.geoms:
            parts.extend(_geom_pixel_parts(part, transform))
        return parts
    if geom.geom_type == "Polygon":
        xs, ys = zip(*geom.exterior.coords)
    elif geom.geom_type == "LineString":
        xs, ys = zip(*geom.coords)
    else:
        return []
    rows, cols = rowcol(transform, xs, ys)
    return [(np.array(cols), np.array(rows))]


def create_tornado_path_atlas(config: ProjectConfig) -> Path:
    """Build one combined figure: every tornado case, EF-colored NWS path on satellite imagery."""

    pairs = pair_registered_rasters(config)
    path_segments = _load_path_segments(config)
    damage_polygons = _load_damage_polygons(config)
    out_path = (
        config.outputs_dir / "predictions" / "random_forest_baseline" / "showcase_maps" / "tornado_path_atlas.png"
    )

    def _in_view_features(features: gpd.GeoDataFrame, bounds, transform, shape, readable):
        # NWS path lines/polygons can be large, irregular shapes spanning multiple
        # counties - a bounding-box overlap with the readable area does NOT mean the
        # actual shape passes through it (the shape could bow away from that corner
        # of its own bbox). Rasterize each candidate onto the raster's real pixel
        # grid - the same technique baseline_model.py uses for training labels - and
        # check the rasterized mask against the readable mask directly. This is the
        # only check that's actually faithful to the shape's geometry.
        case_features = _features_for_bounds(features, bounds)
        hits: list[dict[str, object]] = []
        for _, row in case_features.iterrows():
            mask = rasterize([(row.geometry, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")
            overlap = mask.astype(bool) & readable
            if not overlap.any():
                continue
            overlap_rows, overlap_cols = np.where(overlap)
            hits.append(
                {
                    "row": row,
                    "draw_parts": _geom_pixel_parts(row.geometry, transform),
                    "crop_bbox": (
                        int(overlap_rows.min()),
                        int(overlap_rows.max()),
                        int(overlap_cols.min()),
                        int(overlap_cols.max()),
                    ),
                }
            )
        return hits

    cases: list[dict[str, object]] = []
    for pair in pairs.to_dict("records"):
        tor_id = pair["tornado_id"]
        if pair["pair_status"] != "OK":
            continue
        after_path = Path(pair["after_path"])
        with rasterio.open(after_path) as src:
            transform = src.transform
            bounds = src.bounds
            shape = (src.height, src.width)
        rgb, readable = _read_after_composite(after_path)

        in_view = _in_view_features(path_segments, bounds, transform, shape, readable)
        geom_kind = "line"
        if not in_view:
            in_view = _in_view_features(damage_polygons, bounds, transform, shape, readable)
            geom_kind = "polygon"

        cases.append({"tor_id": tor_id, "rgb": rgb, "shape": shape, "segments": in_view, "geom_kind": geom_kind})

    n = len(cases)
    cols_n = 2
    rows_n = (n + cols_n - 1) // cols_n
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 6.2, rows_n * 5.0), facecolor="white")
    axes = np.atleast_2d(axes)

    for idx in range(rows_n * cols_n):
        ax = axes[idx // cols_n][idx % cols_n]
        ax.set_axis_off()
        if idx >= n:
            continue
        case = cases[idx]
        display = np.where(np.isfinite(case["rgb"]), case["rgb"], 0.0)

        hits = case["segments"]
        if hits:
            height, width = case["shape"]
            all_bboxes = [h["crop_bbox"] for h in hits]
            r0 = max(min(b[0] for b in all_bboxes) - 60, 0)
            r1 = min(max(b[1] for b in all_bboxes) + 60, height)
            c0 = max(min(b[2] for b in all_bboxes) - 60, 0)
            c1 = min(max(b[3] for b in all_bboxes) + 60, width)
            ax.imshow(display[r0:r1, c0:c1])
            for hit in hits:
                ef = str(hit["row"].get("efscale", "") or "").strip().upper()
                color = EF_COLORS.get(ef, "#FFA500")
                for cols_, rows_ in hit["draw_parts"]:
                    if case["geom_kind"] == "polygon":
                        ax.fill(cols_ - c0, rows_ - r0, facecolor=color, alpha=0.25, edgecolor="none")
                        ax.plot(cols_ - c0, rows_ - r0, color=color, linewidth=2.6, solid_capstyle="round")
                    else:
                        ax.plot(cols_ - c0, rows_ - r0, color=color, linewidth=2.8, solid_capstyle="round")
            ax.set_xlim(0, c1 - c0)
            ax.set_ylim(r1 - r0, 0)
            first_row = hits[0]["row"]
            ef_label = first_row.get("efscale", "?")
            date_label = str(first_row.get("stormdate", "?"))[:10]
            if case["geom_kind"] == "polygon":
                caption = f"{case['tor_id']}  {ef_label}  {date_label}\n(damage-area polygon; no path line in this scene)"
            else:
                lat = first_row.get("startlat")
                lon = first_row.get("startlon")
                coord_label = f"({lat:.2f}, {lon:.2f})" if lat is not None and lon is not None else "(unknown)"
                caption = f"{case['tor_id']}  {ef_label}  {date_label}\nStart Point {coord_label}"
        else:
            ax.imshow(display)
            caption = f"{case['tor_id']}\nNo NWS path visible in readable imagery"

        ax.set_title(caption, fontsize=10.5)

    legend_handles = [Patch(facecolor=c, edgecolor=c, label=ef) for ef, c in EF_COLORS.items() if ef != "EFU"]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Tornado Damage Path Atlas — NWS Official Paths by EF Rating", fontsize=16, fontweight="bold", y=1.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Tornado path atlas written to %s", out_path)
    return out_path
