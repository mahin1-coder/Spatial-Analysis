from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.geospatial_validation import validate_pair


def _write_raster(path: Path, west: float, north: float, crs: str = "EPSG:4326") -> None:
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 2,
        "dtype": "float32",
        "crs": crs,
        "transform": from_origin(west, north, 0.01, 0.01),
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ones((2, 8, 8), dtype="float32"))


def test_validate_pair_detects_overlap_and_alignment(tmp_path):
    before = tmp_path / "TOR1_before.tif"
    after = tmp_path / "TOR1_after.tif"
    _write_raster(before, -95, 35)
    _write_raster(after, -95, 35)

    result = validate_pair("TOR1", before, after)

    assert result.status == "OK"
    assert result.crs_match
    assert result.transform_match
    assert result.overlap_fraction_before == 1.0


def test_validate_pair_rejects_non_overlapping_rasters(tmp_path):
    before = tmp_path / "TOR1_before.tif"
    after = tmp_path / "TOR1_after.tif"
    _write_raster(before, -95, 35)
    _write_raster(after, -70, 10)

    result = validate_pair("TOR1", before, after)

    assert result.status == "FAILED"
    assert "overlap" in result.error
