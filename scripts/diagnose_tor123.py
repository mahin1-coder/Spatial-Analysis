#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects
from shapely.geometry import box

from cluster_path.core import _component_properties


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "outputs_all_cases" / "cases" / "TOR123"


def main() -> int:
    with rasterio.open(CASE / "model_probability.tif") as source:
        probability = source.read(1)
        transform = source.transform
        crs = source.crs
        shape = probability.shape
        bounds = source.bounds
    with rasterio.open(CASE / "valid_overlap_mask.tif") as source:
        valid = source.read(1).astype(bool)

    frame = gpd.read_file(
        ROOT / "data" / "reference" / "noaa_dat_all" / "tor123" / "nws_dat_damage_paths.geojson"
    ).to_crs(crs)
    frame.geometry = frame.geometry.intersection(box(*bounds))
    projected_crs = frame.estimate_utm_crs()
    projected = frame.to_crs(projected_crs)
    buffers = []
    for _, row in projected.iterrows():
        radius = np.clip(float(row.get("width", 660)) * 0.9144 / 2, 180, 2500)
        buffers.append(row.geometry.buffer(radius))
    corridor = gpd.GeoSeries(buffers, crs=projected_crs).to_crs(crs)
    truth = rasterize(
        [(geometry, 1) for geometry in corridor],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
    ).astype(bool) & valid

    print(f"valid_pixels={valid.sum()} nws_corridor_pixels={truth.sum()}")
    for name, mask in (("nws", truth), ("background", valid & ~truth)):
        values = probability[mask]
        statistics = [values.mean(), np.median(values), np.percentile(values, 90), np.percentile(values, 95), values.max()]
        print(name, "mean median p90 p95 max", *(f"{value:.5f}" for value in statistics))
    threshold = float(np.percentile(probability[valid], 92))
    mask = valid & (probability >= threshold)
    mask = ndi.binary_closing(mask, structure=disk(4))
    mask = ndi.binary_opening(mask, structure=disk(1))
    minimum = max(30, int(valid.sum() * 0.00005))
    mask = remove_small_objects(mask, min_size=minimum)
    labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype="uint8"))
    overlap_by_label = np.bincount(labels[truth].ravel(), minlength=int(labels.max()) + 1)
    candidates = []
    all_components = []
    for label_id, bounds_slice in enumerate(ndi.find_objects(labels), start=1):
        if bounds_slice is None:
            continue
        local = labels[bounds_slice] == label_id
        area = int(local.sum())
        points = np.argwhere(local) + np.asarray([axis.start for axis in bounds_slice])
        props = _component_properties(points)
        mean_probability = float(np.mean(probability[bounds_slice][local]))
        overlap = int(overlap_by_label[label_id])
        all_components.append(
            {
                "label": label_id,
                "area": area,
                "elongation": props["elongation"],
                "major": props["major"],
                "mean": mean_probability,
                "overlap": overlap,
            }
        )
        if area < minimum or props["elongation"] < 2.0 or props["major"] < 8.0:
            continue
        candidates.append(
            {
                "label": label_id,
                "area": area,
                "elongation": props["elongation"],
                "major": props["major"],
                "mean": mean_probability,
                "overlap": overlap,
            }
        )
    for item in sorted(all_components, key=lambda row: row["overlap"], reverse=True)[:10]:
        print("overlap_component", item)
    for item in sorted(candidates, key=lambda row: row["mean"], reverse=True)[:10]:
        print("eligible_candidate", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
