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
from shapely.geometry import box


def normalize(stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(stack, dtype="float32")
    for band in range(stack.shape[0]):
        values = stack[band][valid]
        low, high = np.percentile(values, [2, 98])
        result[band] = np.clip((stack[band] - low) / max(high - low, 1e-6), 0, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild final maps with NWS geometry clipped to raster bounds.")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    inventory = pd.read_csv(args.inventory)
    if "after" in inventory.columns:
        after_paths = {str(row.case_id): Path(row.after) for row in inventory.itertuples()}
    else:
        after_paths = {
            str(row.case_id): Path(row.path)
            for row in inventory.itertuples()
            if str(row.period).upper() == "AFTER"
        }
    for case_dir in sorted(args.cases.glob("TOR*"), key=lambda path: int(path.name[3:])):
        case = case_dir.name
        mask_path = case_dir / "model_damage_mask.tif"
        centerline_path = case_dir / "model_path_centerline.geojson"
        if not mask_path.exists() or not centerline_path.exists() or case not in after_paths:
            continue
        with rasterio.open(mask_path) as mask_source:
            corridor = mask_source.read(1).astype(bool)
            profile = mask_source.profile
            extent = (mask_source.bounds.left, mask_source.bounds.right, mask_source.bounds.bottom, mask_source.bounds.top)
            footprint = box(*mask_source.bounds)
        with rasterio.open(after_paths[case]) as after_source:
            after = np.zeros((after_source.count, profile["height"], profile["width"]), dtype="float32")
            for band in range(after_source.count):
                reproject(
                    source=rasterio.band(after_source, band + 1),
                    destination=after[band],
                    src_transform=after_source.transform,
                    src_crs=after_source.crs,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    resampling=Resampling.bilinear,
                )
        valid = np.isfinite(after).all(axis=0) & (np.abs(after).sum(axis=0) > 0)
        rgb = np.moveaxis(normalize(after, valid)[[2, 1, 0]], 0, -1)
        centerline = gpd.read_file(centerline_path).to_crs(profile["crs"])
        reference_path = args.references / case.lower() / "nws_dat_damage_paths.geojson"
        nws = None
        if reference_path.exists():
            nws = gpd.read_file(reference_path).to_crs(profile["crs"])
            nws = nws[nws.geometry.intersects(footprint)].copy()
            nws.geometry = nws.geometry.intersection(footprint)
            nws = nws[nws.geometry.notna() & ~nws.geometry.is_empty].copy()

        figure, axis = plt.subplots(figsize=(15, 8), constrained_layout=True)
        axis.imshow(rgb, extent=extent, origin="upper")
        axis.imshow(
            np.ma.masked_where(~corridor, corridor),
            extent=extent,
            origin="upper",
            cmap=ListedColormap(["#FF8C00"]),
            alpha=0.34,
        )
        centerline.plot(ax=axis, color="#FFD400", linewidth=3.2, label="Model centerline")
        if nws is not None and not nws.empty:
            nws.plot(ax=axis, color="#00D9FF", linewidth=3.0, label="Official NWS path")
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_title(f"{case}: Predicted Path and NWS Comparison", fontsize=18, fontweight="bold")
        axis.set_axis_off()
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="lower left", framealpha=0.92)
        figure.savefig(case_dir / "model_final_path_map.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        print(f"[map] {case}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
