"""Phase 3 raster validation, alignment, and preprocessing."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

from .config import ProjectConfig
from .utils import detect_before_after, detect_tornado_id

LOGGER = logging.getLogger(__name__)


def _registered_rasters(config: ProjectConfig) -> list[Path]:
    roots = [config.raw_dir / "rasters", config.raw_dir / "source_mirror"]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend([p for p in root.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}])
    return sorted(set(paths))


def _choose_candidate(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    # Prefer the primary registered copy, then larger files, then stable path order.
    def score(path: Path) -> tuple[int, int, str]:
        primary = 1 if "/data/raw/rasters/" in str(path) else 0
        return (primary, path.stat().st_size, str(path))

    return sorted(paths, key=score, reverse=True)[0]


def pair_registered_rasters(config: ProjectConfig) -> pd.DataFrame:
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: {"BEFORE": [], "AFTER": []})
    for path in _registered_rasters(config):
        tor_id = detect_tornado_id(path)
        status = detect_before_after(path)
        if tor_id and status in {"BEFORE", "AFTER"}:
            grouped[tor_id][status].append(path)

    rows = []
    for tor_id, group in sorted(grouped.items(), key=lambda item: int(item[0][3:])):
        before = _choose_candidate(group["BEFORE"])
        after = _choose_candidate(group["AFTER"])
        rows.append(
            {
                "tornado_id": tor_id,
                "before_path": str(before) if before else "",
                "after_path": str(after) if after else "",
                "before_candidate_count": len(group["BEFORE"]),
                "after_candidate_count": len(group["AFTER"]),
                "pair_status": "OK" if before and after else "MISSING",
                "warnings": "; ".join(
                    item
                    for item in [
                        "missing BEFORE" if not before else "",
                        "missing AFTER" if not after else "",
                        "multiple BEFORE candidates" if len(group["BEFORE"]) > 1 else "",
                        "multiple AFTER candidates" if len(group["AFTER"]) > 1 else "",
                    ]
                    if item
                ),
            }
        )
    return pd.DataFrame(rows)


def _metadata(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "crs": src.crs.to_string() if src.crs else "",
            "transform": tuple(src.transform),
            "bounds": tuple(src.bounds),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": src.dtypes,
            "resolution": src.res,
            "nodata": src.nodata,
        }


def _read_float_stack(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(masked=True).astype("float32")
    return np.ma.filled(arr, np.nan)


def _write_stack(path: Path, arr: np.ndarray, reference_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=arr.shape[0],
            nodata=np.nan,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr.astype("float32"))


def _align_after_to_before(before_path: Path, after_path: Path, out_dir: Path) -> tuple[Path, Path, list[str]]:
    actions: list[str] = []
    before_arr = _read_float_stack(before_path)

    with rasterio.open(before_path) as before_src, rasterio.open(after_path) as after_src:
        if (
            before_src.crs == after_src.crs
            and before_src.transform == after_src.transform
            and before_src.width == after_src.width
            and before_src.height == after_src.height
            and before_src.count == after_src.count
        ):
            after_arr = _read_float_stack(after_path)
            actions.append("rasters already aligned")
        else:
            actions.append("reprojected/resampled AFTER raster to BEFORE grid")
            after_arr = np.full(before_arr.shape, np.nan, dtype="float32")
            for band in range(1, min(before_src.count, after_src.count) + 1):
                source = after_src.read(band, masked=True).astype("float32")
                source = np.ma.filled(source, np.nan)
                reproject(
                    source=source,
                    destination=after_arr[band - 1],
                    src_transform=after_src.transform,
                    src_crs=after_src.crs,
                    dst_transform=before_src.transform,
                    dst_crs=before_src.crs,
                    src_nodata=after_src.nodata,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

    before_out = out_dir / "before_aligned.tif"
    after_out = out_dir / "after_aligned.tif"
    _write_stack(before_out, before_arr, before_path)
    _write_stack(after_out, after_arr, before_path)
    return before_out, after_out, actions


def preprocess_registered_pairs(config: ProjectConfig) -> pd.DataFrame:
    """Validate, clean, and align registered BEFORE/AFTER raster pairs."""

    config.ensure_phase1_dirs()
    pair_df = pair_registered_rasters(config)
    pair_df.to_csv(config.inventory_dir / "registered_raster_pairs.csv", index=False)

    rows: list[dict[str, object]] = []
    for row in pair_df.to_dict("records"):
        tor_id = row["tornado_id"]
        out_dir = config.outputs_dir / "preprocessed" / tor_id
        out_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, object] = {
            "tornado_id": tor_id,
            "before_path": row["before_path"],
            "after_path": row["after_path"],
            "status": "FAILED",
            "before_aligned_path": "",
            "after_aligned_path": "",
            "actions": "",
            "warnings": row.get("warnings", ""),
            "error": "",
        }

        try:
            if row["pair_status"] != "OK":
                raise ValueError(row.get("warnings") or "missing pair")
            before_path = Path(row["before_path"])
            after_path = Path(row["after_path"])
            before_meta = _metadata(before_path)
            after_meta = _metadata(after_path)

            validation = {
                "before": before_meta,
                "after": after_meta,
                "crs_match": before_meta["crs"] == after_meta["crs"],
                "transform_match": before_meta["transform"] == after_meta["transform"],
                "resolution_match": before_meta["resolution"] == after_meta["resolution"],
                "shape_match": (before_meta["width"], before_meta["height"]) == (after_meta["width"], after_meta["height"]),
                "band_count_match": before_meta["count"] == after_meta["count"],
            }

            before_out, after_out, actions = _align_after_to_before(before_path, after_path, out_dir)
            validation["actions"] = actions
            (out_dir / "preprocessing_metadata.json").write_text(json.dumps(validation, indent=2, default=str))

            result.update(
                {
                    "status": "OK",
                    "before_aligned_path": str(before_out),
                    "after_aligned_path": str(after_out),
                    "actions": "; ".join(actions),
                }
            )
        except Exception as exc:
            LOGGER.warning("Preprocessing failed for %s: %s", tor_id, exc)
            result["error"] = str(exc)
            (out_dir / "preprocessing_error.json").write_text(json.dumps(result, indent=2, default=str))

        rows.append(result)

    summary = pd.DataFrame(rows)
    summary_path = config.reports_dir / "preprocessing_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Wrote preprocessing summary: %s", summary_path)
    return summary
