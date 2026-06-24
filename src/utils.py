"""Shared utility functions."""

from __future__ import annotations

import logging
import re
from pathlib import Path


TORNADO_ID_RE = re.compile(r"\bTOR[_-]?(\d+)\b|TOR(\d+)", re.IGNORECASE)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("rasterio._err").setLevel(logging.ERROR)


def detect_tornado_id(path: Path) -> str | None:
    """Extract a normalized tornado ID such as TOR5 from a path or filename."""

    text = str(path)
    match = TORNADO_ID_RE.search(text)
    if not match:
        return None
    number = match.group(1) or match.group(2)
    return f"TOR{int(number)}"


def detect_before_after(path: Path) -> str | None:
    """Detect BEFORE/AFTER status from a filename."""

    name = path.name.upper()
    if "BEFORE" in name or "_PRE" in name or "-PRE" in name:
        return "BEFORE"
    if "AFTER" in name or "_POST" in name or "-POST" in name:
        return "AFTER"
    return None


def classify_file(path: Path) -> str:
    """Assign a high-level inventory category from extension/name."""

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return "GeoTIFF raster"
    if suffix in {".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx"} or path.name.lower().endswith(".shp.xml"):
        return "shapefile component"
    if suffix == ".ipynb":
        return "notebook"
    if suffix in {".pdf", ".ppt", ".pptx"}:
        return "PDF/slides"
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif"}:
        return "image"
    if suffix == ".zip":
        return "archive"
    return "other"


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
