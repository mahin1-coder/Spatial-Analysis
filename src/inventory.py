"""Phase 1 file, raster, and shapefile inventories."""

from __future__ import annotations

import logging
import signal
import struct
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .config import ProjectConfig
from .utils import classify_file, detect_before_after, detect_tornado_id

LOGGER = logging.getLogger(__name__)

SHAPE_TYPES = {
    0: "Null Shape",
    1: "Point",
    3: "PolyLine",
    5: "Polygon",
    8: "MultiPoint",
    11: "PointZ",
    13: "PolyLineZ",
    15: "PolygonZ",
    18: "MultiPointZ",
    21: "PointM",
    23: "PolyLineM",
    25: "PolygonM",
    28: "MultiPointM",
    31: "MultiPatch",
}


class MetadataTimeoutError(RuntimeError):
    """Raised when a metadata read exceeds the configured timeout."""


@contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):  # noqa: ARG001
        raise MetadataTimeoutError(f"metadata read exceeded {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def discover_files(source_roots: tuple[Path, ...]) -> list[Path]:
    """Recursively list files under source roots, skipping virtualenv internals."""

    files: list[Path] = []
    skipped_dirs = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}
    for root in source_roots:
        if root.is_file():
            files.append(root)
            continue
        if not root.exists():
            LOGGER.warning("Source root does not exist: %s", root)
            continue
        for path in root.rglob("*"):
            if any(part in skipped_dirs for part in path.parts):
                continue
            if path.is_file():
                files.append(path)
    return sorted(set(files))


def build_file_inventory(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "category": classify_file(path),
                "tornado_id": detect_tornado_id(path),
                "before_after": detect_before_after(path),
                "size_bytes": stat.st_size,
                "parent": str(path.parent),
            }
        )
    return pd.DataFrame(rows)


def pair_rasters(raster_paths: list[Path]) -> pd.DataFrame:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"BEFORE": [], "AFTER": [], "UNKNOWN": []})
    for path in raster_paths:
        tor_id = detect_tornado_id(path) or "UNKNOWN"
        status = detect_before_after(path) or "UNKNOWN"
        grouped[tor_id][status].append(str(path))

    rows = []
    for tor_id, groups in sorted(grouped.items(), key=lambda item: item[0]):
        before = groups["BEFORE"]
        after = groups["AFTER"]
        warnings = []
        if not before:
            warnings.append("missing BEFORE")
        if not after:
            warnings.append("missing AFTER")
        if len(before) > 1:
            warnings.append("multiple BEFORE files")
        if len(after) > 1:
            warnings.append("multiple AFTER files")
        if groups["UNKNOWN"]:
            warnings.append("unknown BEFORE/AFTER status")
        rows.append(
            {
                "tornado_id": tor_id,
                "before_count": len(before),
                "after_count": len(after),
                "unknown_count": len(groups["UNKNOWN"]),
                "before_files": " | ".join(before),
                "after_files": " | ".join(after),
                "unknown_files": " | ".join(groups["UNKNOWN"]),
                "pair_status": "OK" if len(before) == 1 and len(after) == 1 and not groups["UNKNOWN"] else "CHECK",
                "warnings": "; ".join(warnings),
            }
        )
    return pd.DataFrame(rows)


def _band_stats(dataset: rasterio.io.DatasetReader, band_index: int) -> dict[str, float | str]:
    band = dataset.read(band_index, masked=True)
    total = band.size
    mask = np.ma.getmaskarray(band)
    invalid = int(mask.sum())
    valid = band.compressed()
    nan_count = int(np.isnan(valid).sum()) if valid.size and np.issubdtype(valid.dtype, np.floating) else 0
    valid_without_nan = valid[~np.isnan(valid)] if valid.size and np.issubdtype(valid.dtype, np.floating) else valid

    if valid_without_nan.size == 0:
        stats = {"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan}
    else:
        stats = {
            "min": float(np.min(valid_without_nan)),
            "max": float(np.max(valid_without_nan)),
            "mean": float(np.mean(valid_without_nan)),
            "median": float(np.median(valid_without_nan)),
            "std": float(np.std(valid_without_nan)),
        }

    return {
        f"band_{band_index}_min": stats["min"],
        f"band_{band_index}_max": stats["max"],
        f"band_{band_index}_mean": stats["mean"],
        f"band_{band_index}_median": stats["median"],
        f"band_{band_index}_std": stats["std"],
        f"band_{band_index}_nan_pct": (nan_count / total) * 100 if total else np.nan,
        f"band_{band_index}_valid_pixel_pct": ((total - invalid - nan_count) / total) * 100 if total else np.nan,
    }


def inspect_raster(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "path": str(path),
        "filename": path.name,
        "tornado_id": detect_tornado_id(path),
        "before_after": detect_before_after(path),
        "read_status": "OK",
        "error": "",
    }
    try:
        with rasterio.open(path) as src:
            row.update(
                {
                    "width": src.width,
                    "height": src.height,
                    "band_count": src.count,
                    "crs": src.crs.to_string() if src.crs else "",
                    "affine_transform": tuple(src.transform),
                    "bounds": tuple(src.bounds),
                    "resolution_x": src.res[0],
                    "resolution_y": src.res[1],
                    "dtypes": "|".join(src.dtypes),
                    "nodata": src.nodata,
                }
            )
            band_errors = []
            for band_index in range(1, src.count + 1):
                try:
                    row.update(_band_stats(src, band_index))
                except Exception as exc:
                    band_errors.append(f"band {band_index}: {exc}")
                    for stat_name in ["min", "max", "mean", "median", "std", "nan_pct", "valid_pixel_pct"]:
                        row[f"band_{band_index}_{stat_name}"] = np.nan
            if band_errors:
                row["read_status"] = "METADATA_ONLY"
                row["error"] = " | ".join(band_errors)
    except Exception as exc:  # pragma: no cover - inventory should continue through bad files
        LOGGER.warning("Unable to inspect raster %s: %s", path, exc)
        row["read_status"] = "ERROR"
        row["error"] = str(exc)
    return row


def build_raster_inventory(raster_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for index, path in enumerate(raster_paths, start=1):
        LOGGER.info("Inspecting raster %s/%s: %s", index, len(raster_paths), path.name)
        rows.append(inspect_raster(path))
    return pd.DataFrame(rows)


def infer_shapefile_role(path: Path) -> str:
    lower = path.name.lower()
    parent = str(path.parent).lower()
    text = f"{parent}/{lower}"
    if "path" in text:
        return "tornado path"
    if "poly" in text or "polygon" in text:
        return "tornado polygon"
    if "pnt" in text or "point" in text:
        return "damage point"
    return "unknown"


def _read_shp_header(path: Path) -> dict[str, object]:
    with path.open("rb") as file:
        header = file.read(100)
    if len(header) < 100:
        raise ValueError("SHP header is shorter than 100 bytes")
    file_code = struct.unpack(">i", header[0:4])[0]
    if file_code != 9994:
        raise ValueError(f"Unexpected SHP file code: {file_code}")
    shape_type_code = struct.unpack("<i", header[32:36])[0]
    xmin, ymin, xmax, ymax = struct.unpack("<4d", header[36:68])
    return {
        "geometry_type": SHAPE_TYPES.get(shape_type_code, f"Unknown({shape_type_code})"),
        "bounds": (xmin, ymin, xmax, ymax),
    }


def _read_dbf_header(path: Path) -> tuple[int | None, list[str]]:
    dbf_path = path.with_suffix(".dbf")
    if not dbf_path.exists():
        return None, []
    with dbf_path.open("rb") as file:
        header = file.read(32)
        if len(header) < 32:
            raise ValueError("DBF header is shorter than 32 bytes")
        record_count = struct.unpack("<I", header[4:8])[0]
        header_length = struct.unpack("<H", header[8:10])[0]
        fields_raw = file.read(max(header_length - 33, 0))

    columns = []
    for offset in range(0, len(fields_raw), 32):
        descriptor = fields_raw[offset : offset + 32]
        if len(descriptor) < 32 or descriptor[0] == 0x0D:
            break
        name = descriptor[0:11].split(b"\x00", 1)[0].decode("latin1", errors="ignore").strip()
        if name:
            columns.append(name)
    return record_count, columns


def _read_prj(path: Path) -> str:
    prj_path = path.with_suffix(".prj")
    if not prj_path.exists():
        return ""
    return prj_path.read_text(errors="ignore").strip()


def inspect_shapefile(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "path": str(path),
        "filename": path.name,
        "tornado_id": detect_tornado_id(path),
        "role": infer_shapefile_role(path),
        "read_status": "OK",
        "error": "",
    }
    try:
        with time_limit(5):
            shp_header = _read_shp_header(path)
            feature_count, columns = _read_dbf_header(path)
            crs = _read_prj(path)
        row.update(
            {
                "geometry_type": shp_header["geometry_type"],
                "crs": crs,
                "bounds": shp_header["bounds"],
                "feature_count": feature_count,
                "attribute_columns": "|".join(columns),
                "appears_to_represent": infer_shapefile_role(path),
            }
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Unable to inspect shapefile %s: %s", path, exc)
        row["read_status"] = "ERROR"
        row["error"] = str(exc)
    return row


def build_shapefile_inventory(shapefile_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for index, path in enumerate(shapefile_paths, start=1):
        LOGGER.info("Inspecting shapefile %s/%s: %s", index, len(shapefile_paths), path)
        rows.append(inspect_shapefile(path))
    return pd.DataFrame(rows)


def write_phase1_inventory(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    """Run the complete Phase 1 audit and write inventory CSVs."""

    config.ensure_phase1_dirs()
    roots = config.source_roots or (config.project_root,)
    files = discover_files(roots)
    file_inventory = build_file_inventory(files)

    raster_paths = [p for p in files if p.suffix.lower() in {".tif", ".tiff"}]
    shapefile_paths = [p for p in files if p.suffix.lower() == ".shp"]

    raster_inventory = build_raster_inventory(raster_paths)
    shapefile_inventory = build_shapefile_inventory(shapefile_paths)
    pair_inventory = pair_rasters(raster_paths)

    file_inventory.to_csv(config.inventory_dir / "file_inventory.csv", index=False)
    raster_inventory.to_csv(config.inventory_dir / "raster_inventory.csv", index=False)
    shapefile_inventory.to_csv(config.inventory_dir / "shapefile_inventory.csv", index=False)
    pair_inventory.to_csv(config.inventory_dir / "raster_pair_inventory.csv", index=False)

    return {
        "file_inventory": file_inventory,
        "raster_inventory": raster_inventory,
        "shapefile_inventory": shapefile_inventory,
        "pair_inventory": pair_inventory,
    }
