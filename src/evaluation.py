"""Spatially meaningful evaluation metrics for damage-mask predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_mask_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    """Calculate metrics that are useful for imbalanced damage segmentation."""

    truth = np.asarray(y_true).astype(bool).reshape(-1)
    pred = np.asarray(y_pred).astype(bool).reshape(-1)
    valid = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[valid]
    pred = pred[valid]
    if truth.size == 0:
        return {name: float("nan") for name in ["iou", "dice", "precision", "recall", "f1", "false_negative_rate"]}

    tp = float(np.logical_and(truth, pred).sum())
    fp = float(np.logical_and(~truth, pred).sum())
    fn = float(np.logical_and(truth, ~pred).sum())
    union = float(np.logical_or(truth, pred).sum())
    metrics = {
        "iou": tp / union if union else 0.0,
        "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "matthews_corrcoef": float(matthews_corrcoef(truth, pred)) if len(np.unique(truth)) > 1 else float("nan"),
    }
    if y_score is not None:
        score = np.asarray(y_score).reshape(-1)[valid]
        if len(np.unique(truth)) > 1:
            metrics["average_precision"] = float(average_precision_score(truth, score))
            metrics["roc_auc"] = float(roc_auc_score(truth, score))
        else:
            metrics["average_precision"] = float("nan")
            metrics["roc_auc"] = float("nan")
    return metrics


def connected_component_stats(mask: np.ndarray) -> dict[str, float]:
    """Summarize predicted-mask fragmentation."""

    labels, count = ndi.label(np.asarray(mask).astype(bool))
    sizes = ndi.sum(mask.astype(bool), labels, index=range(1, count + 1)) if count else []
    sizes = np.asarray(sizes, dtype="float64")
    return {
        "component_count": float(count),
        "largest_component_pixels": float(sizes.max()) if sizes.size else 0.0,
        "median_component_pixels": float(np.median(sizes)) if sizes.size else 0.0,
    }


def threshold_analysis(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Create precision/recall/F1 rows for candidate probability thresholds."""

    truth = np.asarray(y_true).astype(bool).reshape(-1)
    score = np.asarray(y_score).astype("float32").reshape(-1)
    valid = np.isfinite(score)
    truth = truth[valid]
    score = score[valid]
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = score >= threshold
        row = binary_mask_metrics(truth, pred, score)
        row["threshold"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def precision_recall_rows(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    truth = np.asarray(y_true).astype(bool).reshape(-1)
    score = np.asarray(y_score).astype("float32").reshape(-1)
    valid = np.isfinite(score)
    precision, recall, thresholds = precision_recall_curve(truth[valid], score[valid])
    padded_thresholds = np.append(thresholds, np.nan)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": padded_thresholds})
