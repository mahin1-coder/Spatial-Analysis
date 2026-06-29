"""Analyze an arbitrary BEFORE/AFTER raster pair with the trained baseline model.

This is the entry point for new imagery that isn't part of the curated TOR##
dataset (e.g. a pair of GeoTIFFs handed to you directly and saved anywhere on
disk). It reuses the same windowed, corruption-aware prediction logic and the
same slide-quality showcase renderer as the dataset pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd

from .config import ProjectConfig
from .predict import predict_on_pair
from .visualization import render_showcase_map

LOGGER = logging.getLogger(__name__)


def _load_geometries_from_shapefile(path: Path) -> list[object]:
    gdf = gpd.read_file(path)
    if gdf.empty:
        return []
    if gdf.crs is None:
        raise ValueError(f"Shapefile has no CRS, cannot align with raster: {path}")
    gdf = gdf.to_crs("EPSG:4326")
    return [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]


def analyze_custom_pair(
    config: ProjectConfig,
    before_path: Path,
    after_path: Path,
    name: str,
    nws_shapefile: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Predict damage and render a showcase map for any BEFORE/AFTER raster pair."""

    model_path = config.outputs_dir / "models" / "random_forest_baseline" / "random_forest_damage_baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Train the baseline first: missing {model_path}")
    if not before_path.exists():
        raise FileNotFoundError(f"BEFORE raster not found: {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"AFTER raster not found: {after_path}")

    model = joblib.load(model_path)

    label_geoms_raw: list[object] = []
    if nws_shapefile is not None:
        label_geoms_raw = _load_geometries_from_shapefile(nws_shapefile)

    out_dir = output_dir or (config.outputs_dir / "predictions" / "custom" / name)
    metrics = predict_on_pair(
        model,
        before_path,
        after_path,
        out_dir,
        label_geoms=[{"geometry": g} for g in label_geoms_raw],
    )
    metrics["case_id"] = name

    showcase_path = out_dir / "showcase_prediction_map.png"
    render_info = render_showcase_map(
        name, Path(metrics["prediction_tif"]), after_path, label_geoms_raw, metrics, showcase_path
    )
    metrics.update(render_info)
    metrics["showcase_path"] = str(showcase_path)

    (out_dir / "result_summary.json").write_text(json.dumps(metrics, indent=2, default=str))
    pd.DataFrame([metrics]).to_csv(out_dir / "result_summary.csv", index=False)
    LOGGER.info("Custom pair analysis complete for %s -> %s", name, out_dir)
    return metrics
