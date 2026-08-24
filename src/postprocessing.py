"""Post-process predicted damage masks into vector path products."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import LineString, shape

NODATA = 255


def clean_damage_mask(mask: np.ndarray, min_component_size: int = 64) -> np.ndarray:
    """Remove small speckle and fill small gaps from a binary damage mask."""

    binary = mask == 1
    binary = ndi.binary_opening(binary, structure=np.ones((3, 3)))
    binary = ndi.binary_closing(binary, structure=np.ones((5, 5)))
    labels, count = ndi.label(binary)
    clean = np.zeros(binary.shape, dtype=bool)
    for label in range(1, count + 1):
        component = labels == label
        if int(component.sum()) >= min_component_size:
            clean |= component
    return clean


def _component_line(component: np.ndarray, transform) -> LineString | None:
    rows, cols = np.where(component)
    if rows.size < 2:
        return None
    coords = np.column_stack([cols, rows]).astype("float64")
    coords -= coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    axis = vt[0]
    projections = coords @ axis
    p0 = coords[np.argmin(projections)] + np.column_stack([cols, rows]).mean(axis=0)
    p1 = coords[np.argmax(projections)] + np.column_stack([cols, rows]).mean(axis=0)
    x0, y0 = rasterio.transform.xy(transform, p0[1], p0[0])
    x1, y1 = rasterio.transform.xy(transform, p1[1], p1[0])
    return LineString([(x0, y0), (x1, y1)])


def write_vector_products(prediction_tif: Path, out_dir: Path, min_component_size: int = 64) -> dict[str, object]:
    """Write cleaned mask, damage polygons, and candidate centerlines."""

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(prediction_tif) as src:
        mask = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    clean = clean_damage_mask(mask, min_component_size=min_component_size)
    clean_uint8 = np.where(mask == NODATA, NODATA, clean.astype("uint8"))
    profile.update(dtype="uint8", count=1, nodata=NODATA, compress="deflate", BIGTIFF="IF_SAFER")
    clean_tif = out_dir / "predicted_damage_mask.tif"
    with rasterio.open(clean_tif, "w", **profile) as dst:
        dst.write(clean_uint8, 1)

    polygon_geoms = [shape(geom) for geom, value in shapes(clean.astype("uint8"), mask=clean, transform=transform) if value == 1]
    polygon_path = out_dir / "predicted_damage_polygon.geojson"
    gpd.GeoDataFrame({"class": ["predicted_damage"] * len(polygon_geoms)}, geometry=polygon_geoms, crs=crs).to_file(
        polygon_path, driver="GeoJSON"
    )

    labels, count = ndi.label(clean)
    lines = []
    for label in range(1, count + 1):
        line = _component_line(labels == label, transform)
        if line is not None:
            lines.append(line)
    centerline_path = out_dir / "predicted_path_centerline.geojson"
    gpd.GeoDataFrame({"class": ["candidate_centerline"] * len(lines)}, geometry=lines, crs=crs).to_file(
        centerline_path, driver="GeoJSON"
    )

    return {
        "clean_damage_mask": str(clean_tif),
        "predicted_damage_polygon": str(polygon_path),
        "predicted_path_centerline": str(centerline_path),
        "component_count": int(count),
    }
