import numpy as np

from src.evaluation import binary_mask_metrics, connected_component_stats
from src.features import build_change_feature_stack


def test_change_feature_stack_preserves_invalid_mask():
    before = np.ones((2, 4, 4), dtype="float32")
    after = before.copy()
    after[:, 1, 1] = 3
    before[:, 0, 0] = np.nan

    features, valid = build_change_feature_stack(before, after)

    assert features.shape[1:] == (4, 4)
    assert not valid[0, 0]
    assert valid[1, 1]
    assert np.isfinite(features[:, 1, 1]).all()


def test_binary_metrics_for_imbalanced_masks():
    truth = np.array([[0, 0, 1], [0, 1, 1]], dtype="uint8")
    pred = np.array([[0, 0, 0], [0, 1, 1]], dtype="uint8")

    metrics = binary_mask_metrics(truth, pred)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 2 / 3
    assert metrics["iou"] == 2 / 3


def test_connected_component_stats_counts_fragments():
    mask = np.zeros((8, 8), dtype="uint8")
    mask[1:3, 1:3] = 1
    mask[6, 6] = 1

    stats = connected_component_stats(mask)

    assert stats["component_count"] == 2
    assert stats["largest_component_pixels"] == 4
