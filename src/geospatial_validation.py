"""Geospatial validation for paired rasters."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import rasterio
import rasterio.warp
from rasterio.coords import BoundingBox
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from shapely.geometry import box


@dataclass(frozen=True)
class RasterMetadata:
    path: str
    readable: bool
    error: str
    crs: str
    width: int
    height: int
    count: int
    dtypes: str
    nodata: str
    resolution_x: float
    resolution_y: float
    transform: str
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class PairValidation:
    case_id: str
    before_path: str
    after_path: str
    status: str
    crs_match: bool
    band_count_match: bool
    dtype_match: bool
    resolution_match: bool
    transform_match: bool
    shape_match: bool
    overlap_fraction_before: float
    overlap_fraction_after: float
    before_readable: bool
    after_readable: bool
    warnings: str
    error: str


def read_raster_metadata(path: Path) -> RasterMetadata:
    """Read enough metadata and one small sample to catch common corruption."""

    try:
        with rasterio.open(path) as src:
            sample_w = min(src.width, 32)
            sample_h = min(src.height, 32)
            src.read(1, window=Window(0, 0, sample_w, sample_h), masked=True)
            return RasterMetadata(
                path=str(path),
                readable=True,
                error="",
                crs=src.crs.to_string() if src.crs else "",
                width=src.width,
                height=src.height,
                count=src.count,
                dtypes=";".join(src.dtypes),
                nodata=str(src.nodata),
                resolution_x=float(src.res[0]),
                resolution_y=float(src.res[1]),
                transform=str(tuple(round(v, 12) for v in src.transform)),
                bounds=(float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)),
            )
    except (RasterioIOError, Exception) as exc:
        return RasterMetadata(
            path=str(path),
            readable=False,
            error=str(exc),
            crs="",
            width=0,
            height=0,
            count=0,
            dtypes="",
            nodata="",
            resolution_x=0.0,
            resolution_y=0.0,
            transform="",
            bounds=(0.0, 0.0, 0.0, 0.0),
        )


def _area(bounds: BoundingBox | tuple[float, float, float, float]) -> float:
    left, bottom, right, top = tuple(bounds)
    return max(0.0, float(right) - float(left)) * max(0.0, float(top) - float(bottom))


def _overlap_fractions(before: Path, after: Path) -> tuple[float, float]:
    with rasterio.open(before) as bsrc, rasterio.open(after) as asrc:
        if not bsrc.crs or not asrc.crs:
            return 0.0, 0.0
        before_geom = box(*bsrc.bounds)
        after_geom = box(*asrc.bounds)
        if bsrc.crs != asrc.crs:
            after_bounds = rasterio.warp.transform_bounds(asrc.crs, bsrc.crs, *asrc.bounds, densify_pts=21)
            after_geom = box(*after_bounds)
        inter = before_geom.intersection(after_geom)
        inter_area = inter.area
        return (
            inter_area / before_geom.area if before_geom.area else 0.0,
            inter_area / after_geom.area if after_geom.area else 0.0,
        )


def validate_pair(case_id: str, before_path: Path, after_path: Path, min_overlap: float = 0.01) -> PairValidation:
    """Validate CRS, metadata compatibility, readability, and geographic overlap."""

    before = read_raster_metadata(before_path)
    after = read_raster_metadata(after_path)
    warnings: list[str] = []
    error = ""

    if not before.readable:
        error = f"BEFORE unreadable: {before.error}"
    if not after.readable:
        error = f"{error}; AFTER unreadable: {after.error}".strip("; ")

    crs_match = bool(before.crs and after.crs and before.crs == after.crs)
    band_count_match = before.count == after.count and before.count > 0
    dtype_match = before.dtypes == after.dtypes and bool(before.dtypes)
    resolution_match = before.resolution_x == after.resolution_x and before.resolution_y == after.resolution_y
    transform_match = before.transform == after.transform and bool(before.transform)
    shape_match = before.width == after.width and before.height == after.height and before.width > 0

    overlap_before = overlap_after = 0.0
    if before.readable and after.readable:
        try:
            overlap_before, overlap_after = _overlap_fractions(before_path, after_path)
        except Exception as exc:
            error = f"{error}; overlap calculation failed: {exc}".strip("; ")

    if not crs_match:
        warnings.append("CRS differs or is missing; reprojection required")
    if not transform_match:
        warnings.append("affine transform differs; pixels are not already aligned")
    if not shape_match:
        warnings.append("shape differs; do not compare arrays directly")
    if not band_count_match:
        warnings.append("band count differs; common bands only can be compared")
    if overlap_before < min_overlap or overlap_after < min_overlap:
        error = f"{error}; raster geographic overlap below {min_overlap:.0%}".strip("; ")

    status = "OK" if not error and before.readable and after.readable and band_count_match else "FAILED"
    return PairValidation(
        case_id=case_id,
        before_path=str(before_path),
        after_path=str(after_path),
        status=status,
        crs_match=crs_match,
        band_count_match=band_count_match,
        dtype_match=dtype_match,
        resolution_match=resolution_match,
        transform_match=transform_match,
        shape_match=shape_match,
        overlap_fraction_before=overlap_before,
        overlap_fraction_after=overlap_after,
        before_readable=before.readable,
        after_readable=after.readable,
        warnings="; ".join(warnings),
        error=error,
    )


def write_validation_report(rows: list[PairValidation], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame([asdict(row) for row in rows])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
