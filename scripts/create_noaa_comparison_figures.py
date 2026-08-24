#!/usr/bin/env python3
"""Overlay downloaded NOAA DAT paths on existing imagery-first model maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def longest_run(values: np.ndarray) -> tuple[int, int]:
    best = (0, 0)
    start = None
    for index, value in enumerate(values.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index - 1)
            start = None
    return best


def image_data_rectangle(image: np.ndarray) -> tuple[int, int, int, int]:
    rgb = image[..., :3]
    nonwhite = np.any(rgb < 0.97, axis=2)
    y0, y1 = longest_run(nonwhite.mean(axis=1) > 0.70)
    within = nonwhite[y0 : y1 + 1]
    columns = within.mean(axis=0) > 0.70
    x0, x1 = longest_run(columns)
    if x1 - x0 < image.shape[1] * 0.70 or y1 - y0 < image.shape[0] * 0.60:
        raise ValueError("Could not locate the geospatial image rectangle reliably.")
    return x0, y0, x1, y1


def sample_line(geometry, count: int = 250):
    if geometry.length <= 0:
        return []
    return [geometry.interpolate(distance, normalized=True) for distance in np.linspace(0, 1, count)]


def mean_nearest_distance(first, second) -> float:
    values = [point.distance(second) for point in sample_line(first)]
    return float(np.mean(values)) if values else float("nan")


def official_buffer(frame: gpd.GeoDataFrame, projected_crs):
    projected = frame.to_crs(projected_crs)
    buffers = []
    radii = []
    for _, row in projected.iterrows():
        raw_width = row.get("width", np.nan)
        width_yards = float(raw_width) if raw_width is not None and np.isfinite(raw_width) and raw_width > 0 else 660.0
        radius = float(np.clip(width_yards * 0.9144 / 2.0, 180.0, 2500.0))
        radii.append(radius)
        buffers.append(row.geometry.buffer(radius))
    return gpd.GeoSeries(buffers, crs=projected_crs).union_all(), radii


def save_overlay(base_path: Path, output_path: Path, nws: gpd.GeoDataFrame, bounds) -> None:
    image = mpimg.imread(base_path)
    x0, y0, x1, y1 = image_data_rectangle(image)
    left, bottom, right, top = bounds
    figure = plt.figure(figsize=(image.shape[1] / 180, image.shape[0] / 180), dpi=180)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(image)
    for geometry in nws.geometry:
        parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
        for part in parts:
            coords = np.asarray(part.coords)
            px = x0 + (coords[:, 0] - left) / (right - left) * (x1 - x0)
            py = y0 + (top - coords[:, 1]) / (top - bottom) * (y1 - y0)
            line = axis.plot(px, py, color="#00D9FF", linewidth=5.0, solid_capstyle="round")[0]
            line.set_path_effects([path_effects.Stroke(linewidth=7.5, foreground="#062B35"), path_effects.Normal()])
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.axis("off")
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    inventory = {row["case_id"]: row for row in csv.DictReader(args.inventory.open(encoding="utf-8"))}
    rows = []
    for case, item in inventory.items():
        case_dir = args.cases / case
        reference = args.references / case.lower() / "nws_dat_damage_paths.geojson"
        if item["status"] != "matched" or not reference.exists():
            rows.append({"case_id": case, "status": "NWS DAT reference unavailable"})
            continue

        with rasterio.open(case_dir / "valid_overlap_mask.tif") as raster:
            analysis_crs = raster.crs
            bounds = raster.bounds
        nws = gpd.read_file(reference).to_crs(analysis_crs)
        footprint = box(*bounds)
        nws = nws[nws.geometry.intersects(footprint)].copy()
        nws.geometry = nws.geometry.intersection(footprint)
        nws = nws[nws.geometry.notna() & ~nws.geometry.is_empty].copy()
        if nws.empty:
            rows.append({"case_id": case, "status": "downloaded geometry does not overlap analysis raster"})
            continue
        predicted = gpd.read_file(case_dir / "model_path_centerline.geojson").to_crs(analysis_crs)
        corridor = gpd.read_file(case_dir / "model_damage_corridor.geojson").to_crs(analysis_crs)
        projected_crs = nws.estimate_utm_crs()
        official_line = nws.to_crs(projected_crs).geometry.union_all()
        predicted_line = predicted.to_crs(projected_crs).geometry.union_all()
        predicted_corridor = corridor.to_crs(projected_crs).geometry.union_all()
        truth_buffer, radii = official_buffer(nws, projected_crs)
        overlap = predicted_corridor.intersection(truth_buffer).area
        dice = 2 * overlap / max(predicted_corridor.area + truth_buffer.area, 1e-9)
        iou = overlap / max(predicted_corridor.union(truth_buffer).area, 1e-9)
        mean_distance = 0.5 * (
            mean_nearest_distance(predicted_line, official_line)
            + mean_nearest_distance(official_line, predicted_line)
        )
        save_overlay(case_dir / "model_final_path_map.png", case_dir / "noaa_dat_comparison.png", nws, bounds)
        rows.append(
            {
                "case_id": case,
                "status": "validated after imagery-only prediction",
                "event_id": item["event_id"],
                "storm_date": item["storm_date"],
                "ef_scale": item["ef_scale"],
                "match_method": item["match_method"],
                "dice_against_nws_buffer": f"{dice:.6f}",
                "iou_against_nws_buffer": f"{iou:.6f}",
                "nws_path_coverage": f"{overlap / max(truth_buffer.area, 1e-9):.6f}",
                "prediction_precision": f"{overlap / max(predicted_corridor.area, 1e-9):.6f}",
                "centerline_hausdorff_m": f"{predicted_line.hausdorff_distance(official_line):.3f}",
                "symmetric_mean_distance_m": f"{mean_distance:.3f}",
                "nws_buffer_radius_m": f"{min(radii):.1f}-{max(radii):.1f}",
            }
        )
        print(f"{case}: comparison created", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
