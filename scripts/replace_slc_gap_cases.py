#!/usr/bin/env python3
"""Build scan-gap-free Landsat composites for the five affected cases.

The original files are never modified.  Public Collection 2 Level-2 assets are
discovered through Microsoft Planetary Computer, QA-masked, reprojected to the
original case grid, and combined with a per-pixel median.
"""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import certifi
from rasterio.enums import Resampling
from rasterio.warp import reproject


PROJECT = Path(__file__).resolve().parents[1]
STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
BAND_ASSETS = ("blue", "green", "red", "nir08", "swir16", "swir22")
BAND_DESCRIPTIONS = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7")
CASES = {
    "TOR70": {"event": "2002-04-24", "platforms": {"landsat-5"}},
    "TOR77": {"event": "2017-02-28", "platforms": {"landsat-8"}},
    "TOR111": {"event": "2008-02-05", "platforms": {"landsat-5"}},
    "TOR112": {"event": "2008-02-05", "platforms": {"landsat-5"}},
    "TOR114": {"event": "2011-04-27", "platforms": {"landsat-5"}},
    "TOR115": {"event": "2011-04-27", "platforms": {"landsat-5"}},
    "TOR123": {"event": "2022-11-04", "platforms": {"landsat-8", "landsat-9"}},
}


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    crs: object
    transform: object
    bounds: tuple[float, float, float, float]
    profile: dict


def request_json(url: str, params: dict[str, str] | None = None) -> dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "Spatial-Analysis/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=90, context=context) as response:
        return json.load(response)


def signed(href: str) -> str:
    return request_json(SAS_SIGN, {"href": href})["href"]


def find_original(case_id: str, role: str) -> Path:
    matches = [
        path for path in (PROJECT / "data_aug2").glob(f"{case_id}_BEST_{role}_*.tif")
        if " 2" not in path.stem and "-2" not in path.stem
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{case_id} {role}: expected one canonical source, found {len(matches)}")
    return matches[0]


def grid_from(path: Path) -> Grid:
    with rasterio.open(path) as source:
        return Grid(
            source.width,
            source.height,
            source.crs,
            source.transform,
            tuple(source.bounds),
            source.profile.copy(),
        )


def search_items(grid: Grid, start: date, end: date, platforms: set[str]) -> list[dict]:
    result = request_json(
        STAC_SEARCH,
        {
            "collections": "landsat-c2-l2",
            "bbox": ",".join(str(value) for value in grid.bounds),
            "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
            "limit": "100",
        },
    )
    items = [
        item for item in result.get("features", [])
        if item.get("properties", {}).get("platform") in platforms
        and all(key in item.get("assets", {}) for key in (*BAND_ASSETS, "qa_pixel"))
    ]
    return sorted(
        items,
        key=lambda item: (
            float(item.get("properties", {}).get("eo:cloud_cover", 100.0)),
            item["properties"]["datetime"],
        ),
    )


def reproject_asset(href: str, grid: Grid, *, nearest: bool, src_nodata: int | float | None) -> np.ndarray:
    destination = np.full((grid.height, grid.width), np.nan, dtype="float32")
    with rasterio.open(signed(href)) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=src_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
        )
    return destination


def item_valid_mask(item: dict, grid: Grid) -> np.ndarray:
    qa = reproject_asset(item["assets"]["qa_pixel"]["href"], grid, nearest=True, src_nodata=1)
    finite = np.isfinite(qa)
    qa_int = np.nan_to_num(qa, nan=1).astype("uint16")
    # Fill, dilated cloud, cloud, cloud shadow, and snow are invalid.
    blocked = sum(1 << bit for bit in (0, 1, 3, 4, 5))
    return finite & ((qa_int & blocked) == 0)


def rank_items(items: list[dict], grid: Grid, event: date, max_candidates: int) -> list[tuple[dict, np.ndarray, float]]:
    ranked = []
    for item in items[:max_candidates]:
        valid = item_valid_mask(item, grid)
        clear_fraction = float(valid.mean())
        acquisition = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).date()
        temporal_days = abs((acquisition - event).days)
        ranked.append((item, valid, clear_fraction, temporal_days))
        print(f"  {item['id']}: ROI clear={clear_fraction:.1%}, event distance={temporal_days} d", flush=True)
    ranked.sort(key=lambda value: (-value[2], value[3], value[0]["id"]))
    return [(item, valid, clear) for item, valid, clear, _ in ranked]


def choose_cover(ranked: list[tuple[dict, np.ndarray, float]], max_scenes: int) -> list[tuple[dict, np.ndarray, float]]:
    if not ranked:
        return []
    selected = []
    covered = np.zeros_like(ranked[0][1], dtype=bool)
    remaining = list(ranked)
    while remaining and len(selected) < max_scenes:
        best_index = max(
            range(len(remaining)),
            key=lambda index: int(np.sum(remaining[index][1] & ~covered)),
        )
        item = remaining.pop(best_index)
        gain = int(np.sum(item[1] & ~covered))
        if gain == 0:
            break
        selected.append(item)
        covered |= item[1]
        if covered.mean() >= 0.995:
            break
    return selected


def build_composite(selected: list[tuple[dict, np.ndarray, float]], grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    if not selected:
        raise RuntimeError("No usable alternate-sensor scenes found")
    scene_stacks = []
    scene_valid = []
    for item, valid, _ in selected:
        bands = []
        for asset_name in BAND_ASSETS:
            raw = reproject_asset(item["assets"][asset_name]["href"], grid, nearest=False, src_nodata=0)
            reflectance = raw * np.float32(0.0000275) - np.float32(0.2)
            reflectance[~valid] = np.nan
            bands.append(reflectance)
        stack = np.stack(bands).astype("float32")
        valid &= np.all(np.isfinite(stack), axis=0)
        stack[:, ~valid] = np.nan
        scene_stacks.append(stack)
        scene_valid.append(valid)
    with np.errstate(all="ignore"):
        composite = np.nanmedian(np.stack(scene_stacks), axis=0).astype("float32")
    valid = np.any(np.stack(scene_valid), axis=0) & np.all(np.isfinite(composite), axis=0)
    composite[:, ~valid] = np.nan
    return composite, valid


def write_composite(path: Path, composite: np.ndarray, valid: np.ndarray, grid: Grid, tags: dict[str, str]) -> None:
    profile = grid.profile.copy()
    profile.update(
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=6,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(composite)
        destination.write_mask(valid.astype("uint8") * 255)
        destination.descriptions = BAND_DESCRIPTIONS
        destination.update_tags(**tags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="Repeat to process selected cases")
    parser.add_argument("--period", choices=("BEFORE", "AFTER"), help="Restrict processing to one period")
    parser.add_argument("--window-days", type=int, default=120)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-scenes", type=int, default=6)
    parser.add_argument("--output", type=Path, default=PROJECT / "data_slc_fixed")
    parser.add_argument("--report-root", type=Path, default=PROJECT / "outputs_slc_fix" / "reports")
    return parser.parse_args()


def merge_csv_rows(path: Path, rows: list[dict], key_fields: tuple[str, ...]) -> None:
    """Update selected records while retaining results from earlier case runs."""
    if not rows:
        return
    merged: dict[tuple[str, ...], dict] = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                merged[tuple(row[field] for field in key_fields)] = row
    for row in rows:
        merged[tuple(str(row[field]) for field in key_fields)] = row
    fieldnames = list(rows[0])
    ordered = sorted(
        merged.values(),
        key=lambda row: tuple(str(row[field]) for field in key_fields),
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


def main() -> int:
    args = parse_args()
    case_ids = args.case or list(CASES)
    scene_rows = []
    quality_rows = []
    args.report_root.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        config = CASES[case_id]
        event = date.fromisoformat(config["event"])
        roles = (args.period,) if args.period else ("BEFORE", "AFTER")
        for role in roles:
            original = find_original(case_id, role)
            grid = grid_from(original)
            if role == "BEFORE":
                start, end = event - timedelta(days=args.window_days), event - timedelta(days=1)
            else:
                start, end = event + timedelta(days=1), event + timedelta(days=args.window_days)
            print(f"[{case_id} {role}] searching {start} to {end}", flush=True)
            search_platforms = set(config["platforms"]) | {"landsat-7"}
            items = search_items(grid, start, end, search_platforms)
            primary_items = [item for item in items if item["properties"]["platform"] in config["platforms"]]
            fallback_items = [item for item in items if item["properties"]["platform"] == "landsat-7"]
            ranked = rank_items(primary_items, grid, event, args.max_candidates)
            selected = choose_cover(ranked, min(3, args.max_scenes))
            covered = (
                np.logical_or.reduce([candidate_valid for _, candidate_valid, _ in selected])
                if selected else np.zeros((grid.height, grid.width), dtype=bool)
            )
            if covered.mean() < 0.95 and fallback_items:
                print(
                    f"  primary coverage is {covered.mean():.1%}; adding QA-valid multi-date "
                    "Landsat 7 gap observations",
                    flush=True,
                )
                remaining = rank_items(fallback_items, grid, event, args.max_candidates)
                while remaining and len(selected) < args.max_scenes and covered.mean() < 0.995:
                    best_index = max(
                        range(len(remaining)),
                        key=lambda index: int(np.sum(remaining[index][1] & ~covered)),
                    )
                    candidate = remaining.pop(best_index)
                    if not np.any(candidate[1] & ~covered):
                        break
                    selected.append(candidate)
                    covered |= candidate[1]
            composite, valid = build_composite(selected, grid)
            output = args.output / f"{case_id}_BEST_{role}_clean.tif"
            ids = [item["id"] for item, _, _ in selected]
            write_composite(
                output,
                composite,
                valid,
                grid,
                {
                    "source": "Microsoft Planetary Computer / USGS Landsat Collection 2 Level-2",
                    "case_id": case_id,
                    "period": role,
                    "event_date": event.isoformat(),
                    "scene_ids": ",".join(ids),
                    "qa_mask": "QA_PIXEL bits 0,1,3,4,5 excluded",
                    "composite": "per-pixel median of valid alternate-sensor scenes",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            for order, (item, _, clear_fraction) in enumerate(selected, start=1):
                scene_rows.append({
                    "case_id": case_id,
                    "period": role,
                    "selection_order": order,
                    "scene_id": item["id"],
                    "platform": item["properties"]["platform"],
                    "acquisition_datetime": item["properties"]["datetime"],
                    "scene_cloud_cover_pct": item["properties"].get("eo:cloud_cover", ""),
                    "roi_clear_fraction": clear_fraction,
                })
            with rasterio.open(original) as source:
                old = source.read(masked=True)
                old_valid = ~np.any(np.ma.getmaskarray(old), axis=0) & np.all(np.isfinite(old.filled(np.nan)), axis=0)
            quality_rows.append({
                "case_id": case_id,
                "period": role,
                "original": str(original),
                "replacement": str(output),
                "original_valid_fraction": float(old_valid.mean()),
                "replacement_valid_fraction": float(valid.mean()),
                "selected_scene_count": len(selected),
                "selected_scene_ids": "|".join(ids),
                "grid_preserved": True,
            })
            print(f"  wrote {output} with {valid.mean():.1%} valid pixels", flush=True)

    merge_csv_rows(
        args.report_root / "scene_selection.csv",
        scene_rows,
        ("case_id", "period", "selection_order"),
    )
    merge_csv_rows(
        args.report_root / "quality_report.csv",
        quality_rows,
        ("case_id", "period"),
    )
    (args.report_root / "method.json").write_text(json.dumps({
        "cases": case_ids,
        "stac": STAC_SEARCH,
        "collection": "landsat-c2-l2",
        "preferred_platforms": "Landsat 5 for 2008/2011; Landsat 8/9 for 2022",
        "fallback_platform": "Landsat 7 QA-valid observations only when preferred-sensor coverage is below 95%",
        "reason": "Multi-date valid observations fill SLC-off gaps without spatial interpolation",
        "spatial_inpainting": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
