from pathlib import Path

import numpy as np
import pytest
import geopandas as gpd
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely.geometry import LineString

from cluster_path.core import (
    AnalysisData,
    CasePair,
    build_features,
    centerline_from_corridor,
    discover_pairs,
)
from cluster_path.model import mask_metrics
from cluster_path.multipath import (
    bridge_path_gaps,
    centerlines_from_corridor,
    evaluate_path_set,
    extract_multiple_corridors,
    stable_water_mask,
)


def test_pair_discovery_is_unambiguous(tmp_path: Path) -> None:
    (tmp_path / "TOR1_before.tif").touch()
    (tmp_path / "TOR1_after.tif").touch()
    pair = discover_pairs(tmp_path)[0]
    assert pair == CasePair("TOR1", tmp_path / "TOR1_before.tif", tmp_path / "TOR1_after.tif")


def test_duplicate_pair_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "TOR1_before.tif").touch()
    (tmp_path / "TOR1_pre.tif").touch()
    (tmp_path / "TOR1_after.tif").touch()
    with pytest.raises(ValueError, match="Ambiguous raster pairing"):
        discover_pairs(tmp_path)


def test_feature_stack_contains_six_band_differences() -> None:
    before = np.full((6, 30, 40), 0.3, dtype="float32")
    after = before.copy()
    after[:, 10:20, 5:35] = 0.6
    valid = np.ones((30, 40), dtype=bool)
    data = AnalysisData(
        before,
        after,
        valid,
        from_origin(-90.0, 35.0, 0.001, 0.001),
        "EPSG:4326",
        (-90.0, 34.97, -89.96, 35.0),
        {},
    )
    features, names, diagnostics = build_features(data)
    assert features.shape == (30, 40, 19)
    assert names[:6] == [
        "signed_blue",
        "signed_green",
        "signed_red",
        "signed_nir",
        "signed_swir1",
        "signed_swir2",
    ]
    assert diagnostics["magnitude"][15, 20] > diagnostics["magnitude"][0, 0]


def test_centerline_is_curved_geometry() -> None:
    mask = np.zeros((100, 120), dtype=bool)
    for column in range(10, 110):
        row = int(45 + 12 * np.sin(column / 18))
        mask[row - 2 : row + 3, column] = True
    line = centerline_from_corridor(mask, from_origin(0, 100, 1, 1))
    assert line is not None
    assert len(line.coords) > 10
    assert line.length > 80


def test_mask_metrics() -> None:
    truth = np.zeros((10, 10), dtype=bool)
    truth[2:6, 2:6] = True
    prediction = truth.copy()
    metrics = mask_metrics(prediction, truth, np.ones_like(truth))
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0


def test_stable_water_mask_uses_aligned_multispectral_bands() -> None:
    before = np.full((6, 80, 120), 0.25, dtype="float32")
    after = before.copy()
    before[1, :, 54:66] = 0.35
    before[4, :, 54:66] = 0.05
    before[3, :, 54:66] = 0.05
    before[2, :, 54:66] = 0.10
    after[:, :, 54:66] = before[:, :, 54:66]
    valid = np.ones((80, 120), dtype=bool)
    data = AnalysisData(
        before,
        after,
        valid,
        from_origin(0, 80, 1, 1),
        "EPSG:3857",
        (0, 0, 120, 80),
        {},
    )
    water = stable_water_mask(data)
    assert water[:, 58:62].mean() > 0.95
    assert water[:, :40].mean() < 0.05


def test_water_gap_bridge_connects_aligned_fragments() -> None:
    mask = np.zeros((100, 160), dtype=bool)
    mask[47:53, 15:70] = True
    mask[47:53, 90:145] = True
    water = np.zeros_like(mask)
    water[:, 68:92] = True
    probability = np.full(mask.shape, 0.1, dtype="float32")
    probability[mask] = 0.95
    valid = np.ones_like(mask)
    bridged, records = bridge_path_gaps(mask, water, probability, valid, max_gap_pixels=30)
    assert len(records) == 1
    assert records[0]["water_fraction"] >= 0.30
    assert ndi.label(bridged)[1] == 1


def test_multiple_independent_corridors_are_preserved() -> None:
    probability = np.full((180, 220), 0.05, dtype="float32")
    for row in (35, 90, 145):
        probability[row - 3 : row + 4, 20:200] = 0.98
    valid = np.ones(probability.shape, dtype=bool)
    corridor, diagnostics = extract_multiple_corridors(
        probability,
        valid,
        85,
        water_mask=np.zeros_like(valid),
        max_paths=6,
    )
    lines = centerlines_from_corridor(corridor, from_origin(0, 180, 1, 1))
    assert diagnostics["path_count"] == 3
    assert len(lines) == 3


def test_path_set_metrics_match_all_reference_paths() -> None:
    predicted = [
        LineString([(0, 0), (10_000, 0)]),
        LineString([(0, 5_000), (10_000, 5_000)]),
        LineString([(0, 10_000), (10_000, 10_000)]),
    ]
    reference = gpd.GeoDataFrame(
        {"event": [1, 2, 3]},
        geometry=[
            LineString([(0, 100), (10_000, 100)]),
            LineString([(0, 5_100), (10_000, 5_100)]),
            LineString([(0, 10_100), (10_000, 10_100)]),
        ],
        crs="EPSG:3857",
    )
    metrics, matches = evaluate_path_set(
        predicted,
        reference,
        "EPSG:3857",
        reference_is_exhaustive=True,
    )
    assert metrics["predicted_path_count"] == 3
    assert metrics["reference_path_count"] == 3
    assert metrics["matched_path_count"] == 3
    assert metrics["path_count_f1"] == 1.0
    assert len(matches) == 3
