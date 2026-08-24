"""Feature generation and multispectral preprocessing utilities."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def robust_normalize(stack: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    """Percentile-normalize each band while preserving invalid pixels as NaN."""

    out = np.full(stack.shape, np.nan, dtype="float32")
    for idx, band in enumerate(stack):
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = np.nanpercentile(finite, [lower, upper])
        if hi <= lo:
            out[idx] = 0.0
            out[idx][~np.isfinite(band)] = np.nan
        else:
            out[idx] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def build_change_feature_stack(before: np.ndarray, after: np.ndarray, include_local: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Build before/after/difference features from aligned arrays.

    Returns a feature cube shaped (features, height, width) and a valid mask.
    Vegetation indices are intentionally not inferred here because band identity
    is usually unknown in this dataset.
    """

    if before.shape != after.shape:
        raise ValueError(f"Aligned stacks must have identical shape: {before.shape} vs {after.shape}")

    before_n = robust_normalize(before)
    after_n = robust_normalize(after)
    diff = after_n - before_n
    abs_diff = np.abs(diff)
    rel_diff = diff / (np.abs(before_n) + 1e-6)
    spectral_mag = np.sqrt(np.nansum(diff * diff, axis=0, keepdims=True))
    features = [before_n, after_n, diff, abs_diff, rel_diff, spectral_mag]

    if include_local:
        local_mean = ndi.uniform_filter(np.nan_to_num(abs_diff, nan=0.0), size=(1, 5, 5))
        local_sq = ndi.uniform_filter(np.nan_to_num(abs_diff * abs_diff, nan=0.0), size=(1, 5, 5))
        local_var = np.maximum(local_sq - local_mean * local_mean, 0.0)
        grad = np.sqrt(
            ndi.sobel(np.nan_to_num(spectral_mag[0], nan=0.0), axis=0) ** 2
            + ndi.sobel(np.nan_to_num(spectral_mag[0], nan=0.0), axis=1) ** 2
        )[None, :, :]
        features.extend([local_mean, local_var, grad])

    stack = np.concatenate(features, axis=0).astype("float32")
    valid = np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0)
    stack[:, ~valid] = np.nan
    return stack, valid
