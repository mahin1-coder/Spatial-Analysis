#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
from types import SimpleNamespace

import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

CASE_PATTERN = re.compile(r"^(TOR\d+)_BEST_(BEFORE|AFTER)_.*\.tif$", re.IGNORECASE)


def discover_pairs(root: Path) -> list[SimpleNamespace]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(root.glob("*.tif")):
        match = CASE_PATTERN.match(path.name)
        if match:
            grouped.setdefault(match.group(1).upper(), {})[match.group(2).lower()] = path
    return [
        SimpleNamespace(case_id=case, before=paths["before"], after=paths["after"])
        for case, paths in sorted(grouped.items(), key=lambda item: int(item[0][3:]))
        if "before" in paths and "after" in paths
    ]


def metadata(path: Path) -> dict[str, object]:
    with rasterio.open(path) as source:
        bounds = source.bounds
        geographic = transform_bounds(source.crs, "EPSG:4326", *bounds, densify_pts=21)
        return {
            "path": str(path.resolve()),
            "crs": str(source.crs),
            "width_px": source.width,
            "height_px": source.height,
            "bands": source.count,
            "resolution_x": abs(source.transform.a),
            "resolution_y": abs(source.transform.e),
            "nodata": source.nodata,
            "left": bounds.left,
            "bottom": bounds.bottom,
            "right": bounds.right,
            "top": bounds.top,
            "lon_min": geographic[0],
            "lat_min": geographic[1],
            "lon_max": geographic[2],
            "lat_max": geographic[3],
            "readable": True,
        }


def overlap_fraction(first: dict[str, object], second: dict[str, object]) -> float:
    left = max(float(first["lon_min"]), float(second["lon_min"]))
    bottom = max(float(first["lat_min"]), float(second["lat_min"]))
    right = min(float(first["lon_max"]), float(second["lon_max"]))
    top = min(float(first["lat_max"]), float(second["lat_max"]))
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    first_area = max(0.0, float(first["lon_max"]) - float(first["lon_min"])) * max(
        0.0, float(first["lat_max"]) - float(first["lat_min"])
    )
    second_area = max(0.0, float(second["lon_max"]) - float(second["lon_min"])) * max(
        0.0, float(second["lat_max"]) - float(second["lat_min"])
    )
    return intersection / max(min(first_area, second_area), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory all tornado raster pairs.")
    parser.add_argument("--rasters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, object]] = []
    pairing_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for pair in discover_pairs(args.rasters):
        try:
            before = metadata(pair.before)
            after = metadata(pair.after)
            overlap = overlap_fraction(before, after)
            same_crs = before["crs"] == after["crs"]
            six_bands = before["bands"] == 6 and after["bands"] == 6
            status = "pass" if overlap >= 0.95 and six_bands else "review"
            pairing_rows.append(
                {
                    "case_id": pair.case_id,
                    "before_path": before["path"],
                    "after_path": after["path"],
                    "pairing_status": "unambiguous",
                }
            )
            for period, item in (("BEFORE", before), ("AFTER", after)):
                inventory_rows.append({"case_id": pair.case_id, "period": period, **item})
            quality_rows.append(
                {
                    "case_id": pair.case_id,
                    "before_crs": before["crs"],
                    "after_crs": after["crs"],
                    "same_crs": same_crs,
                    "before_bands": before["bands"],
                    "after_bands": after["bands"],
                    "six_band_input": six_bands,
                    "geographic_overlap_fraction": overlap,
                    "alignment_required": not (
                        same_crs
                        and before["width_px"] == after["width_px"]
                        and before["height_px"] == after["height_px"]
                        and before["left"] == after["left"]
                        and before["top"] == after["top"]
                    ),
                    "quality_status": status,
                }
            )
        except Exception as error:
            quality_rows.append(
                {"case_id": pair.case_id, "quality_status": "failed", "error": f"{type(error).__name__}: {error}"}
            )

    pd.DataFrame(inventory_rows).to_csv(args.output / "dataset_inventory.csv", index=False)
    pd.DataFrame(pairing_rows).to_csv(args.output / "pairing_report.csv", index=False)
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(args.output / "data_quality_report.csv", index=False)
    print(quality.to_string(index=False))
    return 1 if (quality["quality_status"] == "failed").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
