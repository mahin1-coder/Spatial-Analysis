from __future__ import annotations

import unittest

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely.geometry import LineString

from cluster_path.core import AnalysisData
from cluster_path.multipath import (
    bridge_path_gaps,
    centerlines_from_corridor,
    evaluate_path_set,
    extract_multiple_corridors,
    stable_water_mask,
)


class MultiPathTests(unittest.TestCase):
    def test_stable_water_mask(self) -> None:
        before = np.full((6, 80, 120), 0.25, dtype="float32")
        after = before.copy()
        before[1, :, 54:66] = 0.35
        before[4, :, 54:66] = 0.05
        before[3, :, 54:66] = 0.05
        before[2, :, 54:66] = 0.10
        after[:, :, 54:66] = before[:, :, 54:66]
        data = AnalysisData(
            before,
            after,
            np.ones((80, 120), dtype=bool),
            from_origin(0, 80, 1, 1),
            "EPSG:3857",
            (0, 0, 120, 80),
            {},
        )
        water = stable_water_mask(data)
        self.assertGreater(water[:, 58:62].mean(), 0.95)
        self.assertLess(water[:, :40].mean(), 0.05)

    def test_water_gap_bridge(self) -> None:
        mask = np.zeros((100, 160), dtype=bool)
        mask[47:53, 15:70] = True
        mask[47:53, 90:145] = True
        water = np.zeros_like(mask)
        water[:, 68:92] = True
        probability = np.full(mask.shape, 0.1, dtype="float32")
        probability[mask] = 0.95
        valid = np.ones_like(mask)
        bridged, records = bridge_path_gaps(mask, water, probability, valid, max_gap_pixels=30)
        self.assertEqual(len(records), 1)
        self.assertEqual(ndi.label(bridged)[1], 1)

    def test_rejects_bridge_that_follows_a_river(self) -> None:
        mask = np.zeros((100, 160), dtype=bool)
        mask[10:40, 77:83] = True
        mask[60:90, 77:83] = True
        water = np.zeros_like(mask)
        water[:, 76:84] = True
        probability = np.full(mask.shape, 0.1, dtype="float32")
        probability[mask] = 0.95
        valid = np.ones_like(mask)
        bridged, records = bridge_path_gaps(mask, water, probability, valid, max_gap_pixels=30)
        self.assertEqual(records, [])
        self.assertGreater(ndi.label(bridged)[1], 1)

    def test_three_corridors_remain_three(self) -> None:
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
        self.assertEqual(diagnostics["path_count"], 3)
        self.assertEqual(len(lines), 3)

    def test_one_to_one_path_matching(self) -> None:
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
        self.assertEqual(metrics["predicted_path_count"], 3)
        self.assertEqual(metrics["reference_path_count"], 3)
        self.assertEqual(metrics["matched_path_count"], 3)
        self.assertEqual(metrics["path_count_f1"], 1.0)
        self.assertEqual(len(matches), 3)


if __name__ == "__main__":
    unittest.main()
