from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from agentic.analysis import fit_modis_curve, resolve_band_mapping, select_damage_candidate
from agentic.workflow import build_graph


def test_landsat_band_mapping_from_descriptions():
    mapping = resolve_band_mapping(("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"))
    assert mapping is not None
    assert mapping["red"] == 2
    assert mapping["nir"] == 3
    assert mapping["swir2"] == 5


def test_modis_curve_fit_returns_geospatial_line():
    mask = np.zeros((120, 180), dtype=bool)
    columns = np.arange(20, 160)
    rows = (48 + 0.0018 * np.square(columns - 85)).astype(int)
    for row, column in zip(rows, columns):
        mask[max(0, row - 2) : min(mask.shape[0], row + 3), column] = True
    line, metadata = fit_modis_curve(mask, from_origin(-100, 40, 0.001, 0.001))
    assert line is not None
    assert line.length > 0
    assert metadata["selected_model"] in {"linear", "quadratic"}
    assert "DiLiu2023/Tornado_Modis" in metadata["source_attribution"]


def test_candidate_selection_prefers_elongated_supported_region():
    shape = (100, 160)
    valid = np.ones(shape, dtype=bool)
    probability = np.zeros(shape, dtype="float32")
    probability[45:51, 20:140] = 0.92
    probability[10:35, 10:35] = 0.70
    kmeans = np.zeros(shape, dtype=bool)
    kmeans[43:53, 18:142] = True
    corridor, method, records = select_damage_candidate(probability, valid, kmeans, None, 0.45, 40)
    assert corridor.any()
    assert method in {"hybrid_probability", "probability_and_kmeans"}
    assert any(record["plausible"] for record in records)


def test_langgraph_compiles():
    graph = build_graph()
    assert graph is not None
