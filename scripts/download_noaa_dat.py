#!/usr/bin/env python3
"""Download NOAA DAT tornado references that intersect supplied raster footprints."""

from __future__ import annotations

import argparse
import csv
import json
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import rasterio
import certifi
from rasterio.warp import transform_bounds


SERVICE = (
    "https://services.dat.noaa.gov/arcgis/rest/services/"
    "nws_damageassessmenttoolkit/DamageViewer/FeatureServer"
)
LAYERS = {"lines": 1, "polygons": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rasters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def case_id(path: Path) -> str:
    return path.name.split("_")[0].upper()


def raster_bounds(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS: {path}")
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)


def query(layer: int, bounds: tuple[float, float, float, float]) -> tuple[dict, str]:
    params = {
        "where": "(efscale='EF4' OR efscale='EF5') AND stormdate >= DATE '2000-01-01'",
        "geometry": ",".join(f"{value:.10f}" for value in bounds),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    url = f"{SERVICE}/{layer}/query?{urlencode(params)}"
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=90, context=context) as response:
        return json.load(response), url


def event_keys(collection: dict) -> set[tuple[str, int | None]]:
    keys = set()
    for feature in collection.get("features", []):
        props = feature.get("properties", {})
        keys.add((str(props.get("event_id") or ""), props.get("stormdate")))
    return keys


def filter_events(collection: dict, keys: set[tuple[str, int | None]]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            feature
            for feature in collection.get("features", [])
            if (
                str(feature.get("properties", {}).get("event_id") or ""),
                feature.get("properties", {}).get("stormdate"),
            )
            in keys
        ],
    }


def geometry_bounds(feature: dict) -> tuple[float, float, float, float]:
    coordinates = feature.get("geometry", {}).get("coordinates", [])
    points = coordinates if coordinates and isinstance(coordinates[0][0], (int, float)) else []
    if not points:
        raise ValueError("Expected a line geometry with coordinate pairs.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def footprint_score(
    feature: dict, raster_extent: tuple[float, float, float, float]
) -> float:
    candidate = geometry_bounds(feature)
    span = (raster_extent[2] - raster_extent[0]) + (raster_extent[3] - raster_extent[1])
    return sum(abs(left - right) for left, right in zip(candidate, raster_extent)) / max(span, 1e-9)


def select_event(
    collection: dict, raster_extent: tuple[float, float, float, float]
) -> tuple[set[tuple[str, int | None]], str, float | None]:
    keys = event_keys(collection)
    if len(keys) == 1:
        return keys, "unique_ef4_ef5_intersection", 0.0
    representatives = []
    for key in keys:
        matching = filter_events(collection, {key}).get("features", [])
        score = min(footprint_score(feature, raster_extent) for feature in matching)
        representatives.append((score, key))
    representatives.sort(key=lambda item: item[0])
    if len(representatives) >= 2:
        best, second = representatives[0], representatives[1]
        if best[0] < 0.5 and second[0] - best[0] > 0.15:
            return {best[1]}, "raster_footprint_envelope_match", best[0]
    return set(), "ambiguous", representatives[0][0] if representatives else None


def iso_date(milliseconds: int | None) -> str:
    if not milliseconds:
        return ""
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).date().isoformat()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    before_files = sorted(args.rasters.glob("TOR*_BEST_BEFORE*.tif"))
    rows = []

    for raster in before_files:
        case = case_id(raster)
        bounds = raster_bounds(raster)
        lines, line_url = query(LAYERS["lines"], bounds)
        line_keys = event_keys(lines)
        selected_keys, match_method, match_score = select_event(lines, bounds)
        polygons, polygon_url = query(LAYERS["polygons"], bounds)
        selected_lines = filter_events(lines, selected_keys)
        selected_polygons = filter_events(polygons, selected_keys)
        case_dir = args.output / case.lower()
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "damage_lines_candidates.geojson").write_text(
            json.dumps(lines, indent=2), encoding="utf-8"
        )
        (case_dir / "damage_polygons_candidates.geojson").write_text(
            json.dumps(selected_polygons, indent=2), encoding="utf-8"
        )

        status = "missing"
        selected = None
        if selected_keys and selected_lines.get("features"):
            status = "matched"
            selected = selected_lines
            (case_dir / "nws_dat_damage_paths.geojson").write_text(
                json.dumps(selected, indent=2), encoding="utf-8"
            )
            if selected_polygons.get("features"):
                (case_dir / "nws_dat_damage_polys.geojson").write_text(
                    json.dumps(selected_polygons, indent=2), encoding="utf-8"
                )
        elif len(line_keys) > 1:
            status = "ambiguous"

        features = (selected or {}).get("features", [])
        props = features[0].get("properties", {}) if features else {}
        rows.append(
            {
                "case_id": case,
                "status": status,
                "line_features": len(lines.get("features", [])),
                "distinct_events": len(line_keys),
                "match_method": match_method,
                "footprint_match_score": "" if match_score is None else f"{match_score:.6f}",
                "polygon_features": len(selected_polygons.get("features", [])),
                "event_id": props.get("event_id", ""),
                "storm_date": iso_date(props.get("stormdate")),
                "ef_scale": props.get("efscale", ""),
                "wfo": props.get("wfo", ""),
                "raster_bounds_wgs84": "|".join(f"{value:.6f}" for value in bounds),
                "line_query_url": line_url,
                "polygon_query_url": polygon_url,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        print(f"{case}: {status} ({len(line_keys)} event(s))", flush=True)

    fieldnames = list(rows[0]) if rows else []
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
