#!/usr/bin/env python3
"""Register the user-confirmed TOR5 filter annotation as training-only data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = Path.home() / "Desktop" / "Screenshot 2026-08-17 at 1.43.57 PM.png"
REFERENCE = ROOT / "outputs_all_cases" / "cases" / "TOR5" / "model_probability.tif"
OUTPUT = ROOT / "data" / "manual_labels" / "screenshot_verified" / "TOR5"


def filter_panel_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    colorful = ((maximum - minimum) > 25) & (maximum < 250)
    labels, count = ndi.label(ndi.binary_closing(colorful, iterations=3))
    panels = []
    for label_id in range(1, count + 1):
        rows, columns = np.where(labels == label_id)
        if len(rows) >= 10_000:
            panels.append((int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1))
    if not panels:
        raise RuntimeError("Could not locate the TOR5 red/blue filter panel")
    return min(panels, key=lambda bounds: bounds[0])


def extract_line(panel: np.ndarray) -> np.ndarray:
    maximum = panel.max(axis=2)
    minimum = panel.min(axis=2)
    dark = (maximum < 75) & ((maximum - minimum) < 24)
    labels, count = ndi.label(ndi.binary_closing(dark, structure=np.ones((3, 3), dtype=bool)))
    candidates = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        rows, columns = np.where(component)
        if len(rows) < 40:
            continue
        diagonal = float(np.hypot(np.ptp(rows), np.ptp(columns)))
        if diagonal >= 80:
            candidates.append((diagonal, component))
    if not candidates:
        raise RuntimeError("Could not isolate the TOR5 black path annotation")
    return skeletonize(max(candidates, key=lambda item: item[0])[1])


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype("uint8") * 255)
    return skeletonize(np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0)


def ordered_world_coordinates(mask: np.ndarray, transform: rasterio.Affine) -> list[list[float]]:
    rows, columns = np.where(mask)
    points = np.column_stack([columns, rows]).astype("float64")
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    order = np.argsort(centered @ vectors[0])
    ordered = points[order]
    stride = max(1, len(ordered) // 250)
    sampled = ordered[::stride]
    coordinates = []
    for column, row in sampled:
        x, y = transform * (float(column) + 0.5, float(row) + 0.5)
        coordinates.append([x, y])
    return coordinates


def write_mask(path: Path, values: np.ndarray, profile: dict) -> None:
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(values.astype("uint8"), 1)


def main() -> None:
    screenshot = np.asarray(Image.open(SCREENSHOT).convert("RGB"))
    left, top, right, bottom = filter_panel_bbox(screenshot)
    screen_line = extract_line(screenshot[top:bottom, left:right])
    with rasterio.open(REFERENCE) as source:
        profile = source.profile.copy()
        shape = (source.height, source.width)
        transform = source.transform
        crs = source.crs
    line = resize_mask(screen_line, shape)
    corridor = ndi.binary_dilation(line, iterations=5)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_mask(OUTPUT / "manual_centerline_mask.tif", line, profile)
    write_mask(OUTPUT / "manual_damage_corridor_mask.tif", corridor, profile)
    feature_collection = {
        "type": "FeatureCollection",
        "name": "TOR5_training_centerline",
        "crs": {"type": "name", "properties": {"name": str(crs)}},
        "features": [{
            "type": "Feature",
            "properties": {
                "case_id": "TOR5",
                "label_source": "user_verified_red_blue_filter_annotation",
                "training_only": True,
            },
            "geometry": {"type": "LineString", "coordinates": ordered_world_coordinates(line, transform)},
        }],
    }
    (OUTPUT / "manual_path_centerline.geojson").write_text(json.dumps(feature_collection), encoding="utf-8")
    metadata = {
        "case_id": "TOR5",
        "screenshot": str(SCREENSHOT),
        "reference_raster": str(REFERENCE),
        "panel_bbox_pixels": [left, top, right, bottom],
        "path_count": 1,
        "label_source": "user_verified_red_blue_filter_annotation",
        "label_quality": "silver",
        "training_only": True,
        "visible_at_inference": False,
    }
    (OUTPUT / "label_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
