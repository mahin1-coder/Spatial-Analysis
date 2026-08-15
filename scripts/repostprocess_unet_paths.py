#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.transform import resize


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs_unet_v6"
CASE_SOURCE = PROJECT / "outputs_all_cases" / "cases"
LABEL_ROOT = PROJECT / "data" / "manual_labels" / "screenshot_verified"
PERCENTILE = 97.0


def graph_diameter(component: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(component)
    points = np.argwhere(skeleton)
    output = np.zeros_like(component)
    if len(points) < 10:
        return output
    index = np.full(component.shape, -1, dtype="int32")
    index[points[:, 0], points[:, 1]] = np.arange(len(points))

    def neighbors(node: int):
        row, column = points[node]
        local = index[max(0, row - 1):row + 2, max(0, column - 1):column + 2]
        return [int(value) for value in local.ravel() if value >= 0 and value != node]

    def farthest(start: int, parents: bool = False):
        distance = np.full(len(points), -1, dtype="int32")
        parent = np.full(len(points), -1, dtype="int32")
        distance[start] = 0; queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in neighbors(current):
                if distance[nxt] >= 0:
                    continue
                distance[nxt] = distance[current] + 1; parent[nxt] = current; queue.append(nxt)
        target = int(np.argmax(distance))
        return (target, parent) if parents else target

    first = farthest(0); second, parent = farthest(first, parents=True)
    current = second
    while current >= 0:
        row, column = points[current]; output[row, column] = True
        if current == first:
            break
        current = int(parent[current])
    return output


def extract(probability: np.ndarray, valid: np.ndarray, water: np.ndarray):
    threshold = float(np.percentile(probability[valid], PERCENTILE))
    mask = valid & (probability >= threshold)
    mask &= ~ndi.binary_dilation(water, structure=disk(2))
    mask = ndi.binary_closing(mask, structure=disk(3))
    mask = remove_small_objects(mask, min_size=max(15, int(valid.sum() * 0.000025)))
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    candidates = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        line = graph_diameter(component)
        points = np.argwhere(line)
        if len(points) < 14:
            continue
        centered = points - points.mean(0)
        values = np.linalg.eigvalsh(np.cov(centered.T))
        elongation = np.sqrt(max(values[-1], 1e-6) / max(values[0], 1e-6))
        displacement = np.linalg.norm(points[-1] - points[0])
        straightness = displacement / max(len(points), 1)
        if elongation < 2.0 or straightness < 0.18:
            continue
        score = len(points) * min(elongation, 18) * max(straightness, 0.15) * float(probability[component].mean())
        candidates.append((score, component, line))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return np.zeros_like(mask), np.zeros_like(mask), threshold, 0
    selected = [item for item in candidates if item[0] >= candidates[0][0] * 0.28][:4]
    corridor = np.logical_or.reduce([item[1] for item in selected])
    lines = np.logical_or.reduce([item[2] for item in selected])
    return corridor, lines, threshold, len(selected)


def after_image(case_id: str) -> np.ndarray:
    image = Image.open(CASE_SOURCE / case_id / "before_after.png").convert("RGB")
    width, height = image.size
    return np.asarray(image.crop((width // 2, 0, width, height)))


def label_metrics(case_id: str, prediction: np.ndarray, valid: np.ndarray):
    label_path = LABEL_ROOT / case_id / "manual_damage_corridor_mask.tif"
    if not label_path.exists():
        return {}
    with rasterio.open(label_path) as src:
        truth = src.read(1) > 0
    if truth.shape != prediction.shape:
        truth = resize(truth.astype("float32"), prediction.shape, order=0, preserve_range=True, anti_aliasing=False) > 0.5
    truth &= valid; prediction &= valid
    intersection = int((truth & prediction).sum())
    return {
        "dice": 2 * intersection / max(int(truth.sum() + prediction.sum()), 1),
        "precision": intersection / max(int(prediction.sum()), 1),
        "recall": intersection / max(int(truth.sum()), 1),
    }


def main():
    rows = []
    for case_dir in sorted((ROOT / "cases").glob("TOR*"), key=lambda path: int(path.name[3:])):
        data = np.load(case_dir / "prediction.npz")
        probability = data["probability"]; valid = data["valid"].astype(bool); water = data["water"].astype(bool)
        corridor, centerline, threshold, count = extract(probability, valid, water)
        np.savez_compressed(case_dir / "calibrated_prediction.npz", probability=probability, corridor=corridor, centerline=centerline, valid=valid, water=water, threshold=threshold)
        after = after_image(case_dir.name)
        display = resize(centerline.astype("float32"), after.shape[:2], order=0, preserve_range=True, anti_aliasing=False) > 0.5
        fig, axis = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
        axis.imshow(after)
        if display.any():
            axis.contour(ndi.binary_dilation(display, iterations=2), [0.5], colors=["#10231D"], linewidths=4)
            axis.contour(display, [0.5], colors=["#FFEA00"], linewidths=2.5)
        axis.set_title(f"{case_dir.name}: Calibrated U-Net tornado damage path", fontsize=19, fontweight="bold")
        axis.set_axis_off(); fig.savefig(case_dir / "calibrated_model_final_path.png", dpi=180, facecolor="white"); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        axes[0].imshow(after); axes[0].set_title("Full AFTER image", fontweight="bold")
        image = axes[1].imshow(probability, cmap="inferno", vmin=0, vmax=1); axes[1].set_title("Out-of-fold / model probability", fontweight="bold"); fig.colorbar(image, ax=axes[1], fraction=0.04)
        axes[2].imshow(after)
        if display.any(): axes[2].contour(display, [0.5], colors=["#FFEA00"], linewidths=2.5)
        axes[2].set_title(f"{count} calibrated model path(s)", fontweight="bold")
        for axis in axes: axis.set_axis_off()
        fig.savefig(case_dir / "calibrated_model_prediction_panel.png", dpi=170, facecolor="white"); plt.close(fig)
        rows.append({"case_id": case_dir.name, "path_count": count, "adaptive_percentile": PERCENTILE, "absolute_threshold": threshold, **label_metrics(case_dir.name, corridor.copy(), valid)})
    report = ROOT / "reports" / "calibrated_path_results.csv"
    with report.open("w", newline="") as handle:
        fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
