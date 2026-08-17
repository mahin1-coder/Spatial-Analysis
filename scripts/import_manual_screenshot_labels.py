#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from scipy import ndimage as ndi
from shapely.geometry import mapping
from skimage.morphology import skeletonize

from cluster_path.core import _write_raster
from cluster_path.core import centerline_from_corridor
from cluster_path.multipath import bridge_path_gaps


DESKTOP = Path.home() / "Desktop"


def screenshot(stem: str) -> Path:
    matches = sorted(DESKTOP.glob(f"{stem}*PM.png"))
    return matches[0] if matches else DESKTOP / f"{stem} PM.png"


CASES = {
    "TOR5": screenshot("Screenshot 2026-08-17 at 1.43.57"),
    "TOR70": screenshot("Screenshot 2026-08-13 at 2.09.19"),
    "TOR77": screenshot("Screenshot 2026-08-13 at 2.11.28"),
    "TOR78": screenshot("Screenshot 2026-08-13 at 2.12.30"),
    "TOR91": screenshot("Screenshot 2026-08-13 at 2.13.55"),
    "TOR95": screenshot("Screenshot 2026-08-13 at 2.14.44"),
    "TOR101": screenshot("Screenshot 2026-08-13 at 2.15.42"),
    "TOR111": screenshot("Screenshot 2026-08-13 at 2.16.57"),
    "TOR112": screenshot("Screenshot 2026-08-13 at 2.17.44"),
    "TOR114": screenshot("Screenshot 2026-08-13 at 2.18.26"),
    "TOR115": screenshot("Screenshot 2026-08-13 at 2.19.08"),
    "TOR123": screenshot("Screenshot 2026-08-13 at 2.19.42"),
    "TOR10": screenshot("Screenshot 2026-08-10 at 3.28.37"),
}

FILTER_ANNOTATION_CASES = {"TOR5"}


def longest_run(values: np.ndarray) -> tuple[int, int]:
    labels, count = ndi.label(values)
    if count == 0:
        raise ValueError("No image panel detected")
    sizes = ndi.sum(values, labels, range(1, count + 1))
    selected = int(np.argmax(sizes)) + 1
    indexes = np.flatnonzero(labels == selected)
    return int(indexes.min()), int(indexes.max()) + 1


def panel_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    normalized = rgb.astype("float32") / 255.0
    maximum = normalized.max(axis=2)
    minimum = normalized.min(axis=2)
    colorful = ((maximum - minimum) > 0.10) & (maximum < 0.995)
    rows = colorful.mean(axis=1) > 0.30
    top, bottom = longest_run(ndi.binary_closing(rows, iterations=2))
    columns = colorful[top:bottom].mean(axis=0) > 0.32
    labels, count = ndi.label(ndi.binary_closing(columns, iterations=2))
    candidates = []
    for label_id in range(1, count + 1):
        indexes = np.flatnonzero(labels == label_id)
        if len(indexes):
            candidates.append((len(indexes), int(indexes.min()), int(indexes.max()) + 1))
    if not candidates:
        raise ValueError("No probability panel columns detected")
    _, left, right = max(candidates)
    return left, top, right, bottom


def canonical_probability(path: Path, width: int, height: int) -> np.ndarray:
    with rasterio.open(path) as source:
        probability = source.read(1).astype("float32")
    valid = np.isfinite(probability)
    limit = max(float(np.percentile(probability[valid], 99)), 1e-6)
    scaled = np.clip(probability / limit, 0.0, 1.0)
    rgba = matplotlib.colormaps["inferno"](scaled, bytes=True)
    image = Image.fromarray(rgba[..., :3], mode="RGB")
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))


def annotation_mask(panel: np.ndarray, canonical: np.ndarray) -> np.ndarray:
    dark_neutral = (panel.max(axis=2) < 55) & ((panel.max(axis=2) - panel.min(axis=2)) < 22)
    difference = np.mean(np.abs(panel.astype("int16") - canonical.astype("int16")), axis=2) > 52
    mask = dark_neutral & difference
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndi.label(mask)
    accepted = np.zeros_like(mask)
    for label_id in range(1, count + 1):
        component = labels == label_id
        rows, columns = np.where(component)
        if len(rows) < 18:
            continue
        diagonal = float(np.hypot(np.ptp(rows), np.ptp(columns)))
        if diagonal < 35 or len(rows) / max(diagonal, 1.0) > 18:
            continue
        accepted |= component
    return skeletonize(accepted)


def filter_panel_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the left red/blue map in a two-panel filter screenshot."""
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    colorful = ((maximum - minimum) > 25) & (maximum < 250)
    labels, count = ndi.label(ndi.binary_closing(colorful, iterations=3))
    panels = []
    for label_id in range(1, count + 1):
        rows, columns = np.where(labels == label_id)
        if len(rows) < 10_000:
            continue
        panels.append((int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1))
    if not panels:
        raise ValueError("No red/blue filter panel detected")
    return min(panels, key=lambda bounds: bounds[0])


def filter_annotation_mask(panel: np.ndarray) -> np.ndarray:
    """Extract the long black user stroke without retaining map texture or text."""
    maximum = panel.max(axis=2)
    minimum = panel.min(axis=2)
    dark_neutral = (maximum < 75) & ((maximum - minimum) < 24)
    labels, count = ndi.label(ndi.binary_closing(dark_neutral, structure=np.ones((3, 3), dtype=bool)))
    candidates = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        rows, columns = np.where(component)
        if len(rows) < 40:
            continue
        diagonal = float(np.hypot(np.ptp(rows), np.ptp(columns)))
        if diagonal < 80:
            continue
        candidates.append((diagonal, component))
    if not candidates:
        raise ValueError("No long black annotation found in filter panel")
    return skeletonize(max(candidates, key=lambda item: item[0])[1])


def to_raster_mask(screen_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(screen_mask.astype("uint8") * 255)
    resized = np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
    return skeletonize(resized)


def main() -> None:
    output = PROJECT / "data" / "manual_labels" / "screenshot_verified"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id, screenshot_path in CASES.items():
        case_dir = output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        probability_path = PROJECT / "outputs_all_cases" / "cases" / case_id / "model_probability.tif"
        if not screenshot_path.exists() or not probability_path.exists():
            rows.append({"case_id": case_id, "status": "rejected", "reason": "missing screenshot or probability raster"})
            continue
        screenshot = np.asarray(Image.open(screenshot_path).convert("RGB"))
        if case_id in FILTER_ANNOTATION_CASES:
            left, top, right, bottom = filter_panel_bbox(screenshot)
            panel = screenshot[top:bottom, left:right]
            canonical = panel.copy()
            screen_line = filter_annotation_mask(panel)
            label_source = "user_verified_red_blue_filter_annotation"
        else:
            left, top, right, bottom = panel_bbox(screenshot)
            panel = screenshot[top:bottom, left:right]
            canonical = canonical_probability(probability_path, right - left, bottom - top)
            screen_line = annotation_mask(panel, canonical)
            label_source = "user_verified_probability_annotation"
        with rasterio.open(probability_path) as source:
            profile = source.profile.copy()
            transform = source.transform
            crs = source.crs
            shape = (source.height, source.width)
        line_mask = to_raster_mask(screen_line, shape)
        # Join only short, directionally consistent annotation gaps. This
        # preserves separate parallel paths while preventing one hand-drawn
        # path from being counted as several fragments.
        line_mask, _ = bridge_path_gaps(
            line_mask,
            np.zeros(shape, dtype=bool),
            np.ones(shape, dtype="float32"),
            np.ones(shape, dtype=bool),
            max_gap_pixels=24,
            min_water_fraction=1.0,
            max_angle_degrees=38.0,
        )
        line_mask = skeletonize(line_mask)
        labels, component_count = ndi.label(line_mask, structure=np.ones((3, 3), dtype="uint8"))
        lines = []
        for label_id in range(1, component_count + 1):
            component = labels == label_id
            if int(component.sum()) < 12:
                continue
            line = centerline_from_corridor(ndi.binary_dilation(component, iterations=2), transform)
            if line is not None and line.length > 0:
                lines.append(line)
        status = "accepted" if lines else "rejected"
        if lines:
            collection = {
                "type": "FeatureCollection",
                "name": f"{case_id}_manual_training_centerlines",
                "crs": {"type": "name", "properties": {"name": str(crs)}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"case_id": case_id, "path_id": index, "label_source": label_source},
                        "geometry": mapping(line),
                    }
                    for index, line in enumerate(lines, 1)
                ],
            }
            (case_dir / "manual_path_centerline.geojson").write_text(json.dumps(collection), encoding="utf-8")
            corridor = ndi.binary_dilation(line_mask, iterations=5)
            _write_raster(case_dir / "manual_centerline_mask.tif", line_mask, profile, "uint8", 0)
            _write_raster(case_dir / "manual_damage_corridor_mask.tif", corridor, profile, "uint8", 0)
        else:
            corridor = np.zeros(shape, dtype=bool)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        axes[0].imshow(panel)
        axes[0].set_title("User annotation crop")
        axes[1].imshow(canonical)
        axes[1].imshow(screen_line, cmap="Greens", alpha=0.8)
        axes[1].set_title("Extracted black centerline")
        axes[2].imshow(corridor, cmap="Reds")
        axes[2].set_title(f"Georeferenced label: {len(lines)} path(s)")
        for axis in axes:
            axis.set_axis_off()
        fig.savefig(case_dir / "registration_qc.png", dpi=170, facecolor="white")
        plt.close(fig)
        metadata = {
            "case_id": case_id,
            "screenshot": str(screenshot_path),
            "probability_raster": str(probability_path),
            "panel_bbox_pixels": [left, top, right, bottom],
            "path_count": len(lines),
            "label_source": label_source,
            "label_quality": "silver",
            "status": status,
            "warning": "Screenshot-derived supervision; visual registration must be reviewed before final training.",
        }
        (case_dir / "label_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        rows.append(metadata)
    columns = sorted({key for row in rows for key in row})
    with (output / "manual_label_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
