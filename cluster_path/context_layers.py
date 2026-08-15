from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import box

from .core import AnalysisData


SUPPORTED_VECTOR_SUFFIXES = {".gpkg", ".geojson", ".json", ".shp"}


@dataclass(frozen=True)
class ContextMasks:
    water: np.ndarray
    roads: np.ndarray
    railways: np.ndarray
    exclusion: np.ndarray
    crossing: np.ndarray
    provenance: list[dict[str, Any]]


def _iter_vector_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VECTOR_SUFFIXES:
            yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_VECTOR_SUFFIXES:
                    yield candidate


def _rasterize_sources(
    paths: Iterable[str | Path],
    data: AnalysisData,
    *,
    buffer_pixels: float,
    layer_name: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    mask = np.zeros(data.valid.shape, dtype=bool)
    records: list[dict[str, Any]] = []
    footprint = gpd.GeoDataFrame(geometry=[box(*data.bounds)], crs=data.crs)
    pixel_size = max(abs(float(data.transform.a)), abs(float(data.transform.e)))
    for path in _iter_vector_files(paths):
        record: dict[str, Any] = {
            "layer": layer_name,
            "path": str(path.resolve()),
            "status": "rejected",
            "feature_count": 0,
        }
        try:
            frame = gpd.read_file(path)
            if frame.crs is None:
                record["reason"] = "missing CRS"
                records.append(record)
                continue
            frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].to_crs(data.crs)
            frame = frame.loc[frame.intersects(footprint.geometry.iloc[0])]
            if frame.empty:
                record["reason"] = "no overlap with raster"
                records.append(record)
                continue
            geometries = frame.geometry
            if buffer_pixels > 0:
                geometries = geometries.buffer(pixel_size * float(buffer_pixels))
            burned = rasterize(
                ((geometry, 1) for geometry in geometries if geometry is not None and not geometry.is_empty),
                out_shape=data.valid.shape,
                transform=data.transform,
                fill=0,
                dtype="uint8",
                all_touched=True,
            ) > 0
            mask |= burned & data.valid
            record.update(status="accepted", feature_count=int(len(frame)), pixels=int(burned.sum()))
        except Exception as exc:
            record["reason"] = str(exc)
        records.append(record)
    return mask, records


def build_context_masks(
    data: AnalysisData,
    stable_water: np.ndarray,
    config: dict[str, Any] | None = None,
) -> ContextMasks:
    """Build aligned nuisance-feature masks without using NWS/DAT geometry."""
    config = config or {}
    water_vectors, water_records = _rasterize_sources(
        config.get("hydrography_paths", []),
        data,
        buffer_pixels=float(config.get("hydrography_buffer_pixels", 2.0)),
        layer_name="hydrography",
    )
    road_vectors, road_records = _rasterize_sources(
        config.get("road_paths", []),
        data,
        buffer_pixels=float(config.get("road_buffer_pixels", 1.5)),
        layer_name="roads",
    )
    rail_vectors, rail_records = _rasterize_sources(
        config.get("railway_paths", []),
        data,
        buffer_pixels=float(config.get("railway_buffer_pixels", 1.5)),
        layer_name="railways",
    )
    water = (stable_water | water_vectors) & data.valid
    roads = road_vectors & data.valid
    railways = rail_vectors & data.valid
    exclusion = (water | roads | railways) & data.valid
    provenance = [*water_records, *road_records, *rail_records]
    provenance.append(
        {
            "layer": "spectral_stable_water",
            "status": "accepted",
            "pixels": int(stable_water.sum()),
            "method": "persistent MNDWI/NDVI evidence on BEFORE and AFTER rasters",
        }
    )
    return ContextMasks(
        water=water,
        roads=roads,
        railways=railways,
        exclusion=exclusion,
        crossing=exclusion.copy(),
        provenance=provenance,
    )


def save_context_provenance(path: Path, masks: ContextMasks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "important_rule": (
            "Context layers suppress evidence on mapped water and infrastructure. "
            "They are never tornado labels and NWS/DAT geometry is not included here."
        ),
        "layers": masks.provenance,
        "pixel_counts": {
            "water": int(masks.water.sum()),
            "roads": int(masks.roads.sum()),
            "railways": int(masks.railways.sum()),
            "combined_exclusion": int(masks.exclusion.sum()),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
