#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi

from cluster_path.core import _rgb, discover_pairs, load_analysis_data


CASES = ["TOR10", "TOR70", "TOR77", "TOR78", "TOR91", "TOR95", "TOR101", "TOR111", "TOR112", "TOR114", "TOR115", "TOR123"]


def overlay_line(axis, image, line_mask, color, title):
    axis.imshow(image)
    halo = ndi.binary_dilation(line_mask, iterations=4)
    axis.contour(halo.astype("uint8"), levels=[0.5], colors=["white"], linewidths=4.0)
    axis.contour(line_mask.astype("uint8"), levels=[0.5], colors=[color], linewidths=2.4)
    axis.set_title(title, fontsize=15, fontweight="bold")
    axis.set_axis_off()


def main():
    pair_lookup = {pair.case_id: pair for pair in discover_pairs(PROJECT / "data_research_final")}
    root = PROJECT / "outputs_manual_v4" / "manual_comparisons"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in CASES:
        data = load_analysis_data(pair_lookup[case_id], max_dimension=1600)
        rgb = _rgb(data.after, data.valid)
        manual_path = PROJECT / "data" / "manual_labels" / "screenshot_verified" / case_id / "manual_centerline_mask.tif"
        manual_corridor_path = manual_path.with_name("manual_damage_corridor_mask.tif")
        prediction_path = PROJECT / "outputs_manual_v4" / "cases" / case_id / "model_damage_mask.tif"
        with rasterio.open(manual_path) as src:
            manual_line = src.read(1).astype(bool)
        with rasterio.open(manual_corridor_path) as src:
            manual_corridor = src.read(1).astype(bool)
        prediction = np.zeros(manual_corridor.shape, dtype="uint8")
        with rasterio.open(prediction_path) as src:
            reproject(
                src.read(1), prediction,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=data.transform, dst_crs=data.crs,
                resampling=Resampling.nearest,
            )
        prediction = prediction.astype(bool) & data.valid
        predicted_line = ndi.binary_erosion(prediction) ^ prediction
        intersection = int((prediction & manual_corridor & data.valid).sum())
        predicted_count = int((prediction & data.valid).sum())
        truth_count = int((manual_corridor & data.valid).sum())
        dice = 2 * intersection / max(predicted_count + truth_count, 1)
        iou = intersection / max(int(((prediction | manual_corridor) & data.valid).sum()), 1)
        status = "Pass" if dice >= 0.50 else "Fail - model revision required"

        case_dir = root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(13, 7.2), constrained_layout=True)
        overlay_line(axis, rgb, manual_line, "#111111", f"{case_id}: Expert-verified tornado path reference")
        fig.savefig(case_dir / "expert_verified_path_overlay.png", dpi=200, facecolor="white")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), constrained_layout=True)
        overlay_line(axes[0], rgb, manual_line, "#111111", "Expert-verified reference")
        overlay_line(axes[1], rgb, predicted_line, "#00B8D9", "Independent Random Forest output")
        fig.suptitle(f"{case_id}: Prediction versus verified reference | Dice {dice:.3f} | IoU {iou:.3f} | {status}", fontsize=17, fontweight="bold")
        fig.savefig(case_dir / "prediction_vs_verified_reference.png", dpi=200, facecolor="white")
        plt.close(fig)
        rows.append({"case_id": case_id, "dice": dice, "iou": iou, "status": status, "evaluation_type": "training-set diagnostic"})
    pd.DataFrame(rows).to_csv(PROJECT / "outputs_manual_v4" / "reports" / "manual_reference_evaluation.csv", index=False)


if __name__ == "__main__":
    main()
