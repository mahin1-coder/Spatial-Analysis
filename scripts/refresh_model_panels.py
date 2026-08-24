#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from refresh_final_maps import normalize


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild model panels from saved probabilities and corridors.")
    parser.add_argument("--pairing-report", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairing_report)
    after_paths = {str(row.case_id): Path(row.after) for row in pairs.itertuples()}

    for case_dir in sorted(args.cases.glob("TOR*"), key=lambda value: int(value.name[3:])):
        case = case_dir.name
        required = {
            "probability": case_dir / "model_probability.tif",
            "mask": case_dir / "model_damage_mask.tif",
            "centerline": case_dir / "model_path_centerline.geojson",
        }
        if case not in after_paths or any(not value.exists() for value in required.values()):
            continue
        with rasterio.open(required["probability"]) as source:
            probability = source.read(1)
            profile = source.profile
            extent = (source.bounds.left, source.bounds.right, source.bounds.bottom, source.bounds.top)
        with rasterio.open(required["mask"]) as source:
            corridor = source.read(1).astype(bool)
        with rasterio.open(after_paths[case]) as source:
            after = np.zeros((source.count, profile["height"], profile["width"]), dtype="float32")
            for band in range(source.count):
                reproject(
                    source=rasterio.band(source, band + 1),
                    destination=after[band],
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    resampling=Resampling.bilinear,
                )
        valid = np.isfinite(after).all(axis=0) & (np.abs(after).sum(axis=0) > 0)
        rgb = np.moveaxis(normalize(after, valid)[[2, 1, 0]], 0, -1)
        centerline = gpd.read_file(required["centerline"]).to_crs(profile["crs"])

        figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        axes[0].imshow(rgb, extent=extent, origin="upper")
        axes[0].set_title("Full AFTER image", fontweight="bold")
        limit = max(float(np.percentile(probability[valid], 99)), 1e-6)
        image = axes[1].imshow(
            np.ma.masked_where(~valid, probability),
            extent=extent,
            origin="upper",
            cmap="inferno",
            vmin=0,
            vmax=limit,
        )
        axes[1].set_title("Imagery-only damage probability", fontweight="bold")
        figure.colorbar(image, ax=axes[1], fraction=0.035, pad=0.02)
        axes[2].imshow(rgb, extent=extent, origin="upper")
        axes[2].imshow(
            np.ma.masked_where(~corridor, corridor),
            extent=extent,
            origin="upper",
            cmap=ListedColormap(["#FF8C00"]),
            alpha=0.34,
        )
        centerline.plot(ax=axes[2], color="#FFD400", linewidth=3.2)
        axes[2].set_title("Cleaned corridor and centerline", fontweight="bold")
        for axis in axes:
            axis.set_axis_off()
        figure.suptitle(f"{case}: Model Prediction Stages", fontsize=18, fontweight="bold")
        figure.savefig(case_dir / "model_prediction_panel.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        print(f"[panel] {case}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
