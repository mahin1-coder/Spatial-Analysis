#!/usr/bin/env python3
from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage as ndi
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from skimage.filters import frangi, sobel
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.transform import resize

MANUAL_CASES = ["TOR10", "TOR70", "TOR77", "TOR78", "TOR91", "TOR95", "TOR101", "TOR111", "TOR112", "TOR114", "TOR115", "TOR123"]
OUTPUT = PROJECT / "outputs_structured_v5"
SEED = 42


def case_ids() -> list[str]:
    return sorted((p.name for p in (PROJECT / "outputs_all_cases" / "cases").iterdir() if p.is_dir()), key=lambda v: int(v[3:]))


def read_scaled(path: Path, max_dimension: int, nearest: bool = False) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        array = src.read(1).astype("float32")
        profile = src.profile.copy()
    scale = min(1.0, max_dimension / max(array.shape))
    shape = (max(1, int(round(array.shape[0] * scale))), max(1, int(round(array.shape[1] * scale))))
    if shape != array.shape:
        array = resize(array, shape, order=0 if nearest else 1, preserve_range=True, anti_aliasing=not nearest).astype("float32")
    return array, profile


def robust(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = array[valid & np.isfinite(array)]
    if not len(values):
        return np.zeros_like(array, dtype="float32")
    low, high = np.percentile(values, [2, 98])
    return np.clip((array - low) / max(high - low, 1e-6), 0, 1).astype("float32")


def inputs(case_id: str, max_dimension: int):
    root = PROJECT / "outputs_all_cases" / "cases" / case_id
    probability, _ = read_scaled(root / "model_probability.tif", max_dimension)
    change, profile = read_scaled(root / "change_score.tif", max_dimension)
    water, _ = read_scaled(root / "stable_water_mask.tif", max_dimension, nearest=True)
    valid = np.isfinite(probability) & np.isfinite(change)
    p = robust(probability, valid)
    c = robust(change, valid)
    water = water > 0.5
    features = [p, c, water.astype("float32")]
    for sigma in (1, 3, 7, 15):
        gp = ndi.gaussian_filter(p, sigma=sigma)
        gc = ndi.gaussian_filter(c, sigma=sigma)
        features.extend([gp, gc, np.maximum(p - gp, 0), np.maximum(c - gc, 0)])
    for width in (5, 15, 31):
        mean = ndi.uniform_filter(c, size=width)
        variance = np.maximum(ndi.uniform_filter(c * c, size=width) - mean * mean, 0)
        features.extend([mean, np.sqrt(variance)])
    features.extend([
        sobel(p).astype("float32"),
        sobel(c).astype("float32"),
        frangi(p, sigmas=(1, 2, 4, 8), black_ridges=False).astype("float32"),
        frangi(c, sigmas=(1, 2, 4, 8), black_ridges=False).astype("float32"),
    ])
    stack = np.stack(features, axis=-1).astype("float32")
    stack[~valid] = 0
    return stack, valid, water, profile


def label(case_id: str, shape: tuple[int, int]) -> np.ndarray:
    path = PROJECT / "data" / "manual_labels" / "screenshot_verified" / case_id / "manual_damage_corridor_mask.tif"
    with rasterio.open(path) as src:
        array = src.read(1).astype("float32")
    if array.shape != shape:
        array = resize(array, shape, order=0, preserve_range=True, anti_aliasing=False)
    return array > 0.5


def sample_case(case_id: str, max_dimension: int, seed: int):
    feature, valid, water, _ = inputs(case_id, max_dimension)
    truth = label(case_id, valid.shape) & valid
    exclusion = ndi.binary_dilation(truth, structure=disk(6))
    positive = np.argwhere(truth)
    hard = np.argwhere(valid & ~exclusion & (feature[..., 0] >= np.percentile(feature[..., 0][valid], 65)))
    stable = np.argwhere(valid & ~exclusion & ~water & (feature[..., 1] <= np.percentile(feature[..., 1][valid], 45)))
    rng = np.random.default_rng(seed + int(case_id[3:]))
    positive = positive[rng.choice(len(positive), min(2200, len(positive)), replace=False)]
    negative_pool = np.vstack([hard, stable])
    negative = negative_pool[rng.choice(len(negative_pool), min(5200, len(negative_pool)), replace=False)]
    points = np.vstack([positive, negative])
    x = feature[points[:, 0], points[:, 1]]
    y = np.r_[np.ones(len(positive), dtype="uint8"), np.zeros(len(negative), dtype="uint8")]
    return x, y


def fit(x, y, seed):
    model = ExtraTreesClassifier(
        n_estimators=180,
        max_depth=24,
        min_samples_leaf=2,
        max_features=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x, y)
    return model


def infer(model, feature, valid):
    result = np.zeros(valid.shape, dtype="float32")
    for top in range(0, valid.shape[0], 160):
        local = valid[top : top + 160]
        if local.any():
            result[top : top + 160][local] = model.predict_proba(feature[top : top + 160][local])[:, 1]
    return result


def metrics(prediction, truth, valid):
    prediction &= valid
    truth &= valid
    intersection = int((prediction & truth).sum())
    return {
        "dice": 2 * intersection / max(int(prediction.sum() + truth.sum()), 1),
        "iou": intersection / max(int((prediction | truth).sum()), 1),
        "precision": intersection / max(int(prediction.sum()), 1),
        "recall": intersection / max(int(truth.sum()), 1),
    }


def select_corridors(probability, valid, water, threshold):
    mask = valid & (probability >= threshold)
    mask &= ~ndi.binary_dilation(water, structure=disk(2))
    mask = ndi.binary_closing(mask, structure=disk(3))
    mask = remove_small_objects(mask, min_size=max(24, int(valid.sum() * 0.00003)))
    # Directional continuity is learned by the multiscale/ridge features.
    # A modest closing reconnects short interruptions without creating long
    # synthetic lines across unrelated objects.
    mask = ndi.binary_closing(mask, structure=disk(5)) & valid
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    candidates = []
    for idx in range(1, count + 1):
        component = labels == idx
        points = np.argwhere(component)
        if len(points) < 24:
            continue
        centered = points - points.mean(axis=0)
        values = np.linalg.eigvalsh(np.cov(centered.T))
        elongation = float(np.sqrt(max(values[-1], 1e-6) / max(values[0], 1e-6)))
        major = float(np.sqrt(values[-1]))
        if elongation < 2.0 or major < 10:
            continue
        score = major * min(elongation, 15) * float(probability[component].mean())
        candidates.append((score, idx))
    candidates.sort(reverse=True)
    if not candidates:
        return np.zeros_like(mask)
    cutoff = candidates[0][0] * 0.24
    selected = [idx for score, idx in candidates if score >= cutoff][:4]
    return np.isin(labels, selected) & valid


def after_crop(case_id: str) -> np.ndarray:
    image = Image.open(PROJECT / "outputs_all_cases" / "cases" / case_id / "before_after.png").convert("RGB")
    width, height = image.size
    return np.asarray(image.crop((width // 2, 0, width, height)))


def save_case(case_id, probability, corridor, water):
    case_dir = OUTPUT / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(case_dir / "structured_prediction.npz", probability=probability, corridor=corridor, water=water)
    centerline = skeletonize(corridor)
    background = after_crop(case_id)
    line = resize(centerline.astype("float32"), background.shape[:2], order=0, preserve_range=True, anti_aliasing=False) > 0.5
    fig, axis = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
    axis.imshow(background)
    axis.contour(ndi.binary_dilation(line, iterations=3).astype("uint8"), [0.5], colors=["#10231D"], linewidths=4.0)
    axis.contour(line.astype("uint8"), [0.5], colors=["#7CFF4F"], linewidths=2.5)
    axis.set_title(f"{case_id}: Model-generated tornado damage path", fontsize=18, fontweight="bold")
    axis.set_axis_off()
    fig.savefig(case_dir / "model_generated_final_path.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(background); axes[0].set_title("Full AFTER imagery", fontweight="bold")
    axes[1].imshow(probability, cmap="inferno", vmin=0, vmax=1); axes[1].set_title("Structured model probability", fontweight="bold")
    axes[2].imshow(background); axes[2].contour(line.astype("uint8"), [0.5], colors=["#7CFF4F"], linewidths=2.2); axes[2].set_title("Model-generated path", fontweight="bold")
    for axis in axes: axis.set_axis_off()
    fig.savefig(case_dir / "model_prediction_panel.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    (OUTPUT / "reports").mkdir(parents=True, exist_ok=True)
    samples = {case: sample_case(case, 700, SEED) for case in MANUAL_CASES}
    groups = np.asarray(MANUAL_CASES)
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []
    splitter = GroupKFold(n_splits=3)
    indexes = np.arange(len(MANUAL_CASES))
    for fold, (train_idx, test_idx) in enumerate(splitter.split(indexes, groups=groups), 1):
        train_cases = [MANUAL_CASES[i] for i in train_idx]
        test_cases = [MANUAL_CASES[i] for i in test_idx]
        x = np.vstack([samples[c][0] for c in train_cases]); y = np.concatenate([samples[c][1] for c in train_cases])
        model = fit(x, y, SEED + fold)
        for case_id in test_cases:
            feature, valid, water, _ = inputs(case_id, 700)
            probability = infer(model, feature, valid)
            truth = label(case_id, valid.shape)
            for threshold in thresholds:
                prediction = select_corridors(probability, valid, water, threshold)
                rows.append({"fold": fold, "case_id": case_id, "threshold": threshold, **metrics(prediction, truth, valid)})
    with (OUTPUT / "reports" / "grouped_cross_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for threshold_value in thresholds:
        selected_rows = [row for row in rows if row["threshold"] == threshold_value]
        summary.append({
            "threshold": threshold_value,
            **{name: float(np.mean([row[name] for row in selected_rows])) for name in ("dice", "iou", "precision", "recall")},
        })
    summary.sort(key=lambda row: row["dice"], reverse=True)
    with (OUTPUT / "reports" / "threshold_selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    threshold = float(summary[0]["threshold"])
    x = np.vstack([samples[c][0] for c in MANUAL_CASES]); y = np.concatenate([samples[c][1] for c in MANUAL_CASES])
    model = fit(x, y, SEED)
    model_dir = OUTPUT / "models" / "structured_extra_trees"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib", compress=3)
    (model_dir / "metadata.json").write_text(json.dumps({"training_cases": MANUAL_CASES, "selected_threshold": threshold, "grouped_validation": summary[0], "labels_used_at_inference": False}, indent=2))
    inference_rows = []
    for case_id in case_ids():
        print(f"[structured] {case_id}", flush=True)
        feature, valid, water, _ = inputs(case_id, 900)
        probability = infer(model, feature, valid)
        corridor = select_corridors(probability, valid, water, threshold)
        save_case(case_id, probability, corridor, water)
        inference_rows.append({"case_id": case_id, "path_found": bool(corridor.any()), "path_pixels": int(corridor.sum()), "threshold": threshold})
    with (OUTPUT / "reports" / "all_case_inference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inference_rows[0]))
        writer.writeheader(); writer.writerows(inference_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
