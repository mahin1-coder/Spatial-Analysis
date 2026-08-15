from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import run_rf_path_workflow as wf
import run_hybrid_path_workflow as hybrid


def write_tif(path: Path, value: float) -> None:
    arr = np.full((3, 40, 40), value, dtype="float32")
    arr[:, 10:30, 12:28] += 2
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=40,
        height=40,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-90, 35, 0.01, 0.01),
        nodata=-9999.0,
    ) as dst:
        dst.write(arr)


def test_discover_pairs(tmp_path: Path):
    write_tif(tmp_path / "TOR99_BEST_BEFORE.tif", 1)
    write_tif(tmp_path / "TOR99_BEST_AFTER.tif", 2)
    pairs = wf.discover_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].case_id == "TOR99"


def test_feature_stack_shape():
    before = np.ones((2, 8, 8), dtype="float32")
    after = before + 1
    feats = wf.feature_stack(before, after)
    assert feats.shape == (9, 8, 8)


def test_centerline_from_mask():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 18:22] = True
    line = wf.centerline_from_mask(mask, from_origin(0, 40, 1, 1))
    assert line is not None
    assert len(line.coords) >= 2
    assert wf.line_tortuosity(line) is not None
    assert wf.line_tortuosity(line) < 1.2


def test_centerline_from_large_corridor_uses_geospatial_scale():
    mask = np.zeros((2200, 2600), dtype=bool)
    for column in range(200, 2400):
        row = 400 + column // 3
        mask[row - 10 : row + 11, column] = True
    transform = from_origin(1000, 8000, 30, 30)
    line = wf.centerline_from_mask(mask, transform)
    assert line is not None
    assert line.length > 50000
    assert wf.line_tortuosity(line) < 1.2


def test_component_cleanup_keeps_only_large_regions():
    probability = np.zeros((100, 100), dtype="float32")
    probability[10:12, 10:12] = 0.9
    probability[40:65, 30:55] = 0.9
    clean = wf.clean_prediction(probability, np.ones_like(probability, dtype=bool), 0.5, 100)
    assert not clean[10:12, 10:12].any()
    assert clean[40:65, 30:55].any()


def test_quality_rejects_implausibly_broad_corridor():
    cfg = {
        "quality": {
            "reject_valid_fraction_below": 0.02,
            "low_confidence_valid_fraction_below": 0.10,
            "max_prediction_fraction": 0.30,
            "max_path_tortuosity": 4.0,
        }
    }
    assert wf.classify_quality(0.99, 0.40, True, cfg, 1.1) == "Rejected"


def test_tiny_unet_forward_shape():
    model = hybrid.TinyUNet(9)
    x = hybrid.torch.zeros((1, 9, 64, 64))
    y = model(x)
    assert y.shape == (1, 1, 64, 64)
