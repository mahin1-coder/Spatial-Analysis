#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects, skeletonize
from skimage.transform import resize


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs_structured_v5"
PERCENTILE = 99.0


def after_crop(case_id: str) -> np.ndarray:
    image = Image.open(PROJECT / "outputs_all_cases" / "cases" / case_id / "before_after.png").convert("RGB")
    width, height = image.size
    return np.asarray(image.crop((width // 2, 0, width, height)))


def graph_diameter(component: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(component)
    points = np.argwhere(skeleton)
    output = np.zeros_like(component)
    if len(points) < 8:
        return output
    index_grid = np.full(component.shape, -1, dtype="int32")
    index_grid[points[:, 0], points[:, 1]] = np.arange(len(points), dtype="int32")

    def neighbors(index):
        row, column = points[index]
        local = index_grid[max(row - 1, 0) : row + 2, max(column - 1, 0) : column + 2]
        return [int(v) for v in local.ravel() if v >= 0 and v != index]

    def farthest(start, keep_parent=False):
        distance = np.full(len(points), -1, dtype="int32")
        parent = np.full(len(points), -1, dtype="int32")
        distance[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in neighbors(current):
                if distance[nxt] >= 0:
                    continue
                distance[nxt] = distance[current] + 1
                parent[nxt] = current
                queue.append(nxt)
        target = int(np.argmax(distance))
        return (target, distance, parent) if keep_parent else target

    first = farthest(0)
    second, distance, parent = farthest(first, keep_parent=True)
    current = second
    while current >= 0:
        row, column = points[current]
        output[row, column] = True
        if current == first:
            break
        current = int(parent[current])
    return output


def extract(probability: np.ndarray, water: np.ndarray):
    valid = np.isfinite(probability)
    threshold = float(np.percentile(probability[valid], PERCENTILE))
    mask = valid & (probability >= threshold)
    mask &= ~ndi.binary_dilation(water, structure=disk(2))
    mask = ndi.binary_closing(mask, structure=disk(2))
    mask = remove_small_objects(mask, min_size=max(12, int(valid.sum() * 0.000015)))
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    candidates = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        line = graph_diameter(component)
        points = np.argwhere(line)
        if len(points) < 15:
            continue
        displacement = float(np.linalg.norm(points[-1] - points[0]))
        straightness = displacement / max(len(points), 1)
        centered = points - points.mean(axis=0)
        values = np.linalg.eigvalsh(np.cov(centered.T))
        elongation = float(np.sqrt(max(values[-1], 1e-6) / max(values[0], 1e-6)))
        if straightness < 0.18 or elongation < 1.8:
            continue
        score = len(points) * max(straightness, 0.1) ** 1.5 * min(elongation, 15) * float(probability[component].mean())
        candidates.append((score, line))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return np.zeros_like(mask), threshold, 0
    cutoff = candidates[0][0] * 0.28
    selected = [line for score, line in candidates if score >= cutoff][:4]
    result = np.logical_or.reduce(selected) if selected else np.zeros_like(mask)
    return result, threshold, len(selected)


def main():
    rows = []
    for case_dir in sorted((ROOT / "cases").iterdir(), key=lambda p: int(p.name[3:])):
        bundle = np.load(case_dir / "structured_prediction.npz")
        probability = bundle["probability"]
        water = bundle["water"].astype(bool)
        line, threshold, count = extract(probability, water)
        np.savez_compressed(case_dir / "graph_path_prediction.npz", line=line, threshold=threshold)
        background = after_crop(case_dir.name)
        display_line = resize(line.astype("float32"), background.shape[:2], order=0, preserve_range=True, anti_aliasing=False) > 0.5
        fig, axis = plt.subplots(figsize=(14, 7.5), constrained_layout=True)
        axis.imshow(background)
        if display_line.any():
            axis.contour(ndi.binary_dilation(display_line, iterations=3).astype("uint8"), [0.5], colors=["#10231D"], linewidths=4)
            axis.contour(display_line.astype("uint8"), [0.5], colors=["#7CFF4F"], linewidths=2.5)
        axis.set_title(f"{case_dir.name}: Model-generated tornado damage path", fontsize=19, fontweight="bold")
        axis.set_axis_off()
        fig.savefig(case_dir / "graph_model_final_path.png", dpi=180, facecolor="white")
        plt.close(fig)
        rows.append({"case_id": case_dir.name, "path_count": count, "probability_percentile": PERCENTILE, "absolute_threshold": threshold})
    with (ROOT / "reports" / "graph_path_inference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
