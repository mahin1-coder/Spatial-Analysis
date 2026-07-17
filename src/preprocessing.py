"""Phase 3 raster validation, alignment, and preprocessing."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

from .config import ProjectConfig
from .geospatial_validation import validate_pair, write_validation_report
from .pairing import find_image_pairs, write_pairing_report

LOGGER = logging.getLogger(__name__)


def _registered_rasters(config: ProjectConfig) -> list[Path]:
    roots = [config.raw_dir / "rasters", config.raw_dir / "source_mirror"]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend([p for p in root.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}])
    return sorted(set(paths))


def pair_registered_rasters(config: ProjectConfig) -> pd.DataFrame:
    """Pair registered rasters without silently choosing ambiguous candidates."""

    root = config.raw_dir / "rasters"
    if not root.exists() or not any(root.rglob("*.tif")):
        root = config.raw_dir / "source_mirror"
    if not root.exists():
        return pd.DataFrame(
            columns=[
                "tornado_id",
                "before_path",
                "after_path",
                "before_candidate_count",
                "after_candidate_count",
                "pair_status",
                "warnings",
            ]
        )
    df = find_image_pairs(root)
    df = df[df["tornado_id"].astype(str).ne("") | df["pair_key"].astype(str).ne("")]
    out = df.rename(columns={"status": "pair_status", "reason": "warnings"}).copy()
    for col in ["tornado_id", "before_path", "after_path", "before_candidate_count", "after_candidate_count", "pair_status", "warnings"]:
        if col not in out.columns:
            out[col] = ""
    return out[
        ["tornado_id", "before_path", "after_path", "before_candidate_count", "after_candidate_count", "pair_status", "warnings"]
    ]


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
    before_out = out_dir / "before_aligned.tif"
    after_out = out_dir / "after_aligned.tif"

    with rasterio.open(before_path) as before_src, rasterio.open(after_path) as after_src:
        common_bands = min(before_src.count, after_src.count)
        profile = before_src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=common_bands,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        already_aligned = (
            before_src.crs == after_src.crs
            and before_src.transform == after_src.transform
            and before_src.width == after_src.width
            and before_src.height == after_src.height
            and before_src.count == after_src.count
        )
        actions.append("rasters already aligned; copied readable blocks" if already_aligned else "reprojected/resampled AFTER raster to BEFORE grid by blocks")

        with rasterio.open(before_out, "w", **profile) as before_dst, rasterio.open(after_out, "w", **profile) as after_dst:
            if already_aligned:
                after_reader = after_src
            else:
                after_reader = WarpedVRT(
                    after_src,
                    crs=before_src.crs,
                    transform=before_src.transform,
                    width=before_src.width,
                    height=before_src.height,
                    resampling=Resampling.bilinear,
                    nodata=after_src.nodata,
                )

            unreadable_before_blocks = 0
            unreadable_after_blocks = 0
            try:
                for _, window in before_src.block_windows(1):
                    block_shape = (common_bands, int(window.height), int(window.width))
                    before_block = np.full(block_shape, np.nan, dtype="float32")
                    after_block = np.full(block_shape, np.nan, dtype="float32")
                    try:
                        before_block = np.ma.filled(
                            before_src.read(indexes=list(range(1, common_bands + 1)), window=window, masked=True).astype("float32"),
                            np.nan,
                        )
                    except Exception:
                        unreadable_before_blocks += 1
                    try:
                        after_block = np.ma.filled(
                            after_reader.read(indexes=list(range(1, common_bands + 1)), window=window, masked=True).astype("float32"),
                            np.nan,
                        )
                    except Exception:
                        unreadable_after_blocks += 1
                    before_dst.write(before_block, window=window)
                    after_dst.write(after_block, window=window)
            finally:
                if not already_aligned:
                    after_reader.close()
            if unreadable_before_blocks or unreadable_after_blocks:
                actions.append(
                    f"unreadable blocks preserved as NoData: BEFORE={unreadable_before_blocks}, AFTER={unreadable_after_blocks}"
                )
    return before_out, after_out, actions


def preprocess_registered_pairs(config: ProjectConfig) -> pd.DataFrame:
    """Validate, clean, and align registered BEFORE/AFTER raster pairs."""

    config.ensure_phase1_dirs()
    pair_df = pair_registered_rasters(config)
    pair_df.to_csv(config.inventory_dir / "registered_raster_pairs.csv", index=False)
    write_pairing_report(pair_df, config.reports_dir / "pairing_report.csv")

    rows: list[dict[str, object]] = []
    validation_rows = []
    for row in pair_df.to_dict("records"):
        tor_id = row.get("tornado_id") or row.get("pair_key") or "unknown_case"
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
            pair_validation = validate_pair(tor_id, before_path, after_path)
            validation_rows.append(pair_validation)
            if pair_validation.status != "OK":
                raise ValueError(pair_validation.error or pair_validation.warnings or "raster validation failed")
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
                "overlap_fraction_before": pair_validation.overlap_fraction_before,
                "overlap_fraction_after": pair_validation.overlap_fraction_after,
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
    if validation_rows:
        write_validation_report(validation_rows, config.reports_dir / "raster_validation_report.csv")
    LOGGER.info("Wrote preprocessing summary: %s", summary_path)
    return summary
