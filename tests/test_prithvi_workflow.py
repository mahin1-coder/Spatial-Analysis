import numpy as np
import torch

from prithvi_pipeline.data import auxiliary_features, sliding_positions
from prithvi_pipeline.model import PRITHVI_MEAN, PrithviSegmentationHead, normalize_prithvi
from prithvi_pipeline.pipeline import binary_metrics, select_corridor


def test_prithvi_normalization_uses_hls_reflectance_scale():
    values = (PRITHVI_MEAN / 10000.0).view(1, 6, 1, 1, 1).expand(1, 6, 2, 4, 4)
    normalized = normalize_prithvi(values)
    assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-5)


def test_segmentation_head_returns_full_patch_logits():
    head = PrithviSegmentationHead()
    logits = head(
        torch.zeros(2, 384, 14, 14),
        torch.zeros(2, 8, 224, 224),
        torch.ones(2, 1, 224, 224),
    )
    assert logits.shape == (2, 1, 224, 224)


def test_sliding_positions_include_last_edge():
    assert sliding_positions(100) == [0]
    positions = sliding_positions(1000, 224, 160)
    assert positions[0] == 0
    assert positions[-1] == 776


def test_auxiliary_features_are_finite_and_masked():
    before = np.full((6, 10, 12), 0.1, dtype="float32")
    after = before.copy()
    after[3] += 0.04
    valid = np.ones((10, 12), dtype=bool)
    valid[0, 0] = False
    features = auxiliary_features(before, after, valid)
    assert features.shape == (8, 10, 12)
    assert np.isfinite(features).all()
    assert np.all(features[:, 0, 0] == 0)


def test_corridor_selector_chooses_elongated_component():
    probability = np.zeros((140, 220), dtype="float32")
    probability[62:71, 25:195] = 0.88
    probability[15:45, 15:45] = 0.80
    magnitude = np.full_like(probability, 0.02)
    magnitude[62:71, 25:195] = 0.12
    corridor, metadata = select_corridor(probability, np.ones_like(probability, dtype=bool), magnitude, 0.5)
    assert corridor[66, 100]
    assert not corridor[25, 25]
    assert metadata["selected"]["elongation"] > 2


def test_binary_metrics_known_values():
    probability = np.array([[0.9, 0.8], [0.2, 0.1]], dtype="float32")
    target = np.array([[1, 0], [1, 0]], dtype=bool)
    metrics = binary_metrics(probability, target, np.ones((2, 2), dtype=bool), 0.5)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["dice"] == 0.5
