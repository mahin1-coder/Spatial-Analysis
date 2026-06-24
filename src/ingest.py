"""Phase 2 automated data ingestion."""

from __future__ import annotations

import logging
import signal
import shutil
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .inventory import discover_files
from .utils import classify_file, detect_before_after, detect_tornado_id

LOGGER = logging.getLogger(__name__)
COPY_TIMEOUT_SECONDS = 5


class CopyTimeoutError(RuntimeError):
    """Raised when a raw-data copy takes too long."""


@contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):  # noqa: ARG001
        raise CopyTimeoutError(f"copy exceeded {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _safe_source_label(index: int, root: Path) -> str:
    stem = root.stem if root.is_file() else root.name
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"source_{index:02d}_{safe or 'root'}"


def _copy_without_overwrite(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    temp_dst = dst.with_name(f"{dst.name}.part")
    if temp_dst.exists():
        temp_dst.unlink()
    with time_limit(COPY_TIMEOUT_SECONDS):
        shutil.copy2(src, temp_dst)
    temp_dst.replace(dst)
    return "copied"


def ingest_sources(config: ProjectConfig) -> pd.DataFrame:
    """Register source files into ``data/raw`` without overwriting originals."""

    config.ensure_phase1_dirs()
    roots = config.source_roots or (config.data_dir / "incoming",)
    rows: list[dict[str, object]] = []

    for index, root in enumerate(roots, start=1):
        label = _safe_source_label(index, root)
        root_files = discover_files((root,))
        for path in root_files:
            if root.is_file():
                relative = Path(path.name)
            else:
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    relative = Path(path.name)

            category = classify_file(path)
            mirror_dst = config.raw_dir / "source_mirror" / label / relative
            copy_error = ""
            try:
                status = _copy_without_overwrite(path, mirror_dst)
            except Exception as exc:
                status = "copy_error"
                copy_error = str(exc)
                LOGGER.warning("Unable to copy %s: %s", path, exc)

            if category == "GeoTIFF raster":
                primary_dst = config.raw_dir / "rasters" / path.name
                try:
                    primary_status = _copy_without_overwrite(path, primary_dst)
                except Exception as exc:
                    primary_status = "copy_error"
                    copy_error = f"{copy_error}; {exc}".strip("; ")
                    LOGGER.warning("Unable to register primary raster %s: %s", path, exc)
            elif category == "shapefile component":
                primary_dst = config.raw_dir / "shapefiles" / label / relative
                try:
                    primary_status = _copy_without_overwrite(path, primary_dst)
                except Exception as exc:
                    primary_status = "copy_error"
                    copy_error = f"{copy_error}; {exc}".strip("; ")
                    LOGGER.warning("Unable to register primary shapefile component %s: %s", path, exc)
            else:
                primary_dst = ""
                primary_status = ""

            warnings = []
            if category == "GeoTIFF raster":
                if not detect_tornado_id(path):
                    warnings.append("missing tornado ID")
                if not detect_before_after(path):
                    warnings.append("missing BEFORE/AFTER marker")

            rows.append(
                {
                    "source_path": str(path),
                    "mirror_path": str(mirror_dst),
                    "primary_path": str(primary_dst),
                    "source_label": label,
                    "category": category,
                    "tornado_id": detect_tornado_id(path),
                    "before_after": detect_before_after(path),
                    "status": status,
                    "primary_status": primary_status,
                    "warnings": "; ".join(warnings),
                    "error": copy_error,
                }
            )

    manifest = pd.DataFrame(rows)
    output_path = config.inventory_dir / "ingestion_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    LOGGER.info("Wrote ingestion manifest: %s", output_path)
    return manifest
