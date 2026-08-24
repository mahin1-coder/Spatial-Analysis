from __future__ import annotations

import html
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # The deterministic workflow remains runnable without LangGraph.
    END = START = StateGraph = None
from shapely.geometry import LineString, Point

import run_rf_path_workflow as wf
from cluster_path.core import AnalysisData
from cluster_path.multipath import (
    centerlines_from_corridor,
    evaluate_path_set,
    extract_multiple_corridors,
    stable_water_mask,
)
from cluster_path.context_layers import build_context_masks, save_context_provenance
from .analysis import (
    _component_properties,
    create_candidate_figure,
    create_curve_figure,
    create_eda_figures,
    create_kmeans_products,
    fit_modis_curve,
    resolve_band_mapping,
    select_damage_candidate,
    write_corridor_products,
)


LOGGER = logging.getLogger("tornado.agentic")
PROJECT = Path(__file__).resolve().parents[1]


class AgentState(TypedDict, total=False):
    source: str
    shapefiles: str
    output: str
    reuse_output: str
    model_dir: str
    build_slides: bool
    cases: list[dict[str, Any]]
    aligned: dict[str, dict[str, Any]]
    results: dict[str, dict[str, Any]]
    errors: list[dict[str, str]]
    warnings: list[str]
    stage_log: list[dict[str, Any]]
    manifest: str
    presentation: str
    context_layers: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_stage(state: AgentState, agent: str, status: str, detail: str) -> list[dict[str, Any]]:
    return [*state.get("stage_log", []), {"agent": agent, "status": status, "detail": detail, "timestamp": _now()}]


def _append_error(state: AgentState, case_id: str, agent: str, error: Exception | str) -> list[dict[str, str]]:
    return [*state.get("errors", []), {"case_id": case_id, "agent": agent, "error": str(error)}]


def _safe_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def inventory_agent(state: AgentState) -> dict[str, Any]:
    source = Path(state["source"])
    shapefiles = Path(state["shapefiles"])
    output = Path(state["output"])
    (output / "reports").mkdir(parents=True, exist_ok=True)
    pairs = wf.discover_pairs(source)
    cases: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for pair in pairs:
        nws = wf.find_nws_path(pair.case_id, shapefiles)
        with rasterio.open(pair.before) as before, rasterio.open(pair.after) as after:
            same_crs = before.crs == after.crs
            intersects = bool(
                before.bounds.left < after.bounds.right
                and before.bounds.right > after.bounds.left
                and before.bounds.bottom < after.bounds.top
                and before.bounds.top > after.bounds.bottom
            ) if same_crs else None
            item = {
                "case_id": pair.case_id,
                "before": str(pair.before.resolve()),
                "after": str(pair.after.resolve()),
                "nws": str(nws.resolve()) if nws else "",
                "band_descriptions": list(before.descriptions),
            }
            cases.append(item)
            inventory_rows.append(
                {
                    "case_id": pair.case_id,
                    "before": str(pair.before),
                    "after": str(pair.after),
                    "before_crs": before.crs.to_string() if before.crs else "",
                    "after_crs": after.crs.to_string() if after.crs else "",
                    "before_dimensions": f"{before.width}x{before.height}",
                    "after_dimensions": f"{after.width}x{after.height}",
                    "before_resolution": before.res,
                    "after_resolution": after.res,
                    "bands": min(before.count, after.count),
                    "band_descriptions": ";".join(str(value or "") for value in before.descriptions),
                    "same_crs": same_crs,
                    "bounds_intersect_before_reprojection": intersects,
                    "nws_valid_geometry": bool(nws),
                    "status": "paired",
                }
            )
    pd.DataFrame(inventory_rows).to_csv(output / "reports" / "dataset_inventory.csv", index=False)
    warnings = list(state.get("warnings", []))
    if not pairs:
        warnings.append(f"No unambiguous BEFORE/AFTER pairs found in {source}")
    return {
        "cases": cases,
        "results": {},
        "errors": list(state.get("errors", [])),
        "warnings": warnings,
        "stage_log": _append_stage(state, "inventory_agent", "complete", f"Discovered {len(cases)} complete cases."),
    }


def route_after_inventory(state: AgentState) -> str:
    return "geospatial_agent" if state.get("cases") else "report_agent"


def geospatial_agent(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    reuse = Path(state.get("reuse_output", "")) if state.get("reuse_output") else None
    aligned: dict[str, dict[str, Any]] = {}
    errors = list(state.get("errors", []))
    for case in state.get("cases", []):
        case_id = case["case_id"]
        try:
            reuse_case = reuse / "cases" / case_id if reuse else None
            if reuse_case and all((reuse_case / name).exists() for name in ["aligned_before.tif", "aligned_after.tif", "valid_overlap_mask.tif", "geospatial_validation.json"]):
                before_path = reuse_case / "aligned_before.tif"
                after_path = reuse_case / "aligned_after.tif"
                mask_path = reuse_case / "valid_overlap_mask.tif"
                metadata = _safe_json(reuse_case / "geospatial_validation.json", {})
                source_mode = "reused verified hybrid alignment"
            else:
                pair = wf.Pair(case_id, Path(case["before"]), Path(case["after"]))
                before_path, after_path, mask_path, metadata = wf.align_pair(pair, output)
                source_mode = "new rasterio alignment"
            with rasterio.open(before_path) as before, rasterio.open(after_path) as after, rasterio.open(mask_path) as valid_src:
                if before.crs != after.crs or before.transform != after.transform or before.shape != after.shape:
                    raise ValueError("Aligned rasters do not share one CRS, transform, and shape.")
                valid_fraction = float(np.mean(valid_src.read(1) > 0))
                metadata.update(
                    {
                        "case_id": case_id,
                        "source_mode": source_mode,
                        "crs": before.crs.to_string() if before.crs else "",
                        "transform": tuple(before.transform),
                        "dimensions": [before.height, before.width],
                        "bands": before.count,
                        "valid_fraction": valid_fraction,
                    }
                )
            case_dir = output / "cases" / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "geospatial_validation.json").write_text(json.dumps(metadata, indent=2))
            aligned[case_id] = {
                "before": str(before_path),
                "after": str(after_path),
                "valid_mask": str(mask_path),
                "metadata": metadata,
            }
        except Exception as exc:
            errors = _append_error({**state, "errors": errors}, case_id, "geospatial_agent", exc)
    return {
        "aligned": aligned,
        "errors": errors,
        "stage_log": _append_stage(state, "geospatial_agent", "complete", f"Validated {len(aligned)} analysis grids."),
    }


def eda_agent(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    results = dict(state.get("results", {}))
    errors = list(state.get("errors", []))
    case_lookup = {case["case_id"]: case for case in state.get("cases", [])}
    for case_id, aligned in state.get("aligned", {}).items():
        try:
            case_dir = output / "cases" / case_id
            descriptions = tuple(case_lookup.get(case_id, {}).get("band_descriptions", []))
            eda = create_eda_figures(
                case_id,
                Path(aligned["before"]),
                Path(aligned["after"]),
                case_dir,
                descriptions_override=descriptions,
            )
            results.setdefault(case_id, {})["eda"] = eda
        except Exception as exc:
            errors = _append_error({**state, "errors": errors}, case_id, "eda_agent", exc)
    return {
        "results": results,
        "errors": errors,
        "stage_log": _append_stage(state, "eda_agent", "complete", "Generated statistics, band maps, distributions, composites, correlations, and NDVI diagnostics."),
    }


def clustering_agent(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    results = dict(state.get("results", {}))
    errors = list(state.get("errors", []))
    case_lookup = {case["case_id"]: case for case in state.get("cases", [])}
    for case_id, aligned in state.get("aligned", {}).items():
        try:
            descriptions = tuple(case_lookup.get(case_id, {}).get("band_descriptions", []))
            products = create_kmeans_products(
                case_id,
                Path(aligned["before"]),
                Path(aligned["after"]),
                output / "cases" / case_id,
                descriptions_override=descriptions,
            )
            results.setdefault(case_id, {})["kmeans"] = products
        except Exception as exc:
            errors = _append_error({**state, "errors": errors}, case_id, "clustering_agent", exc)
    return {
        "results": results,
        "errors": errors,
        "stage_log": _append_stage(state, "clustering_agent", "complete", "Generated K-Means change candidates without treating clusters as ground truth."),
    }


def _load_dnn(model_dir: Path, input_channels: int):
    checkpoint = model_dir / "tiny_unet_change_model.pt"
    if not checkpoint.exists():
        return None
    try:
        import torch
        import run_hybrid_path_workflow as hybrid
    except ImportError:
        LOGGER.warning("PyTorch is unavailable; continuing with Random Forest probability only.")
        return None
    model = hybrid.TinyUNet(input_channels)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _rf_probability(model, before_path: Path, after_path: Path):
    before, after, profile = wf.read_stacks(before_path, after_path)
    valid = wf.valid_mask(before, after)
    features = wf.feature_stack(before, after)
    coordinates = np.argwhere(valid)
    probability = np.zeros(valid.shape, dtype="float32")
    for start in range(0, len(coordinates), 250_000):
        block = coordinates[start : start + 250_000]
        values = features[:, block[:, 0], block[:, 1]].T
        probability[block[:, 0], block[:, 1]] = model.predict_proba(values)[:, 1]
    return probability, valid, profile, features


def prediction_agent(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    reuse = Path(state.get("reuse_output", "")) if state.get("reuse_output") else None
    model_dir = Path(state["model_dir"])
    results = dict(state.get("results", {}))
    errors = list(state.get("errors", []))
    rf_model = None
    dnn_model = None
    for case_id, aligned in state.get("aligned", {}).items():
        try:
            case_dir = output / "cases" / case_id
            reused_probability = reuse / "predictions" / case_id / "predicted_probability.tif" if reuse else None
            if reused_probability and reused_probability.exists():
                probability_path = reused_probability
                source = "reused hybrid inference probability"
            else:
                if rf_model is None:
                    rf_model = joblib.load(model_dir / "random_forest_path_model.joblib")
                rf_probability, valid, profile, features = _rf_probability(
                    rf_model, Path(aligned["before"]), Path(aligned["after"])
                )
                if dnn_model is None:
                    dnn_model = _load_dnn(model_dir, features.shape[0])
                if dnn_model:
                    import run_hybrid_path_workflow as hybrid

                    dnn_probability = hybrid.dnn_probability(
                        dnn_model,
                        features,
                        valid,
                        hybrid.cfg_for_hybrid(),
                    )
                else:
                    dnn_probability = None
                if dnn_probability is None:
                    probability = rf_probability
                    source = "Random Forest inference"
                else:
                    weight = 0.25
                    probability = ((1.0 - weight) * rf_probability + weight * dnn_probability).astype("float32")
                    probability[~valid] = 0
                    source = "75% Random Forest + 25% tiny U-Net inference"
                probability_path = case_dir / "predicted_probability.tif"
                wf.write_raster(probability_path, probability, profile, "float32", 0.0)
            results.setdefault(case_id, {})["prediction"] = {
                "probability": str(probability_path),
                "source": source,
            }
        except Exception as exc:
            errors = _append_error({**state, "errors": errors}, case_id, "prediction_agent", exc)
    return {
        "results": results,
        "errors": errors,
        "stage_log": _append_stage(state, "prediction_agent", "complete", "Produced or reused imagery-only hybrid damage probabilities."),
    }


def path_agent(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    shapefiles = Path(state["shapefiles"])
    results = dict(state.get("results", {}))
    errors = list(state.get("errors", []))
    case_lookup = {case["case_id"]: case for case in state.get("cases", [])}
    for case_id, aligned in state.get("aligned", {}).items():
        try:
            case_result = results.setdefault(case_id, {})
            case_dir = output / "cases" / case_id
            probability_path = Path(case_result["prediction"]["probability"])
            kmeans_path = Path(case_result["kmeans"]["mask"])
            with rasterio.open(probability_path) as source:
                probability = source.read(1).astype("float32")
            with rasterio.open(aligned["valid_mask"]) as source:
                valid = source.read(1) > 0
            with rasterio.open(kmeans_path) as source:
                kmeans_mask = source.read(1) > 0
            with rasterio.open(aligned["before"]) as source:
                profile = source.profile.copy()
            ndvi_path = case_dir / "ndvi_loss.tif"
            if ndvi_path.exists():
                with rasterio.open(ndvi_path) as source:
                    ndvi_loss = source.read(1)
            else:
                ndvi_loss = None
            corridor, candidate_name, candidate_records = select_damage_candidate(
                probability,
                valid,
                kmeans_mask,
                ndvi_loss,
                threshold=0.45,
                min_pixels=120,
            )
            water = np.zeros_like(valid)
            exclusion = water.copy()
            crossing = water.copy()
            try:
                with rasterio.open(aligned["before"]) as before_source, rasterio.open(aligned["after"]) as after_source:
                    if before_source.count >= 6 and after_source.count >= 6:
                        before_stack = before_source.read(range(1, 7)).astype("float32")
                        after_stack = after_source.read(range(1, 7)).astype("float32")
                        water_data = AnalysisData(
                            before_stack,
                            after_stack,
                            valid,
                            profile["transform"],
                            profile["crs"],
                            tuple(before_source.bounds),
                            profile,
                        )
                        spectral_water = stable_water_mask(water_data)
                        context_masks = build_context_masks(
                            water_data,
                            spectral_water,
                            state.get("context_layers", {}),
                        )
                        water = context_masks.water
                        exclusion = context_masks.exclusion
                        crossing = context_masks.crossing
                        save_context_provenance(
                            case_dir / "context_layer_provenance.json",
                            context_masks,
                        )
            except Exception as exc:
                LOGGER.warning("%s water mask unavailable: %s", case_id, exc)
                exclusion = water
                crossing = water
            multi_corridor, multi_diagnostics = extract_multiple_corridors(
                probability,
                valid,
                88.0,
                water_mask=water,
                exclusion_mask=exclusion,
                crossing_mask=crossing,
                max_paths=6,
            )
            if multi_diagnostics.get("path_count", 0) > 0:
                corridor = multi_corridor
                candidate_name = "water_aware_multi_path"
            skeleton_lines = centerlines_from_corridor(corridor, profile["transform"])
            skeleton_line = skeleton_lines[0] if skeleton_lines else None
            curve_line, curve_metadata = fit_modis_curve(corridor, profile["transform"])
            final_lines = skeleton_lines
            path_method = "water-aware multi-corridor skeleton graph"
            skeleton_tortuosity = max(
                (wf.line_tortuosity(line) or 0.0 for line in skeleton_lines),
                default=None,
            )
            curve_tortuosity = wf.line_tortuosity(curve_line)
            if not final_lines or (skeleton_tortuosity is not None and skeleton_tortuosity > 4.0):
                if curve_line is not None and curve_tortuosity is not None and curve_tortuosity <= 2.5:
                    final_lines = [curve_line]
                    path_method = f"Tornado_Modis-style {curve_metadata['selected_model']} curve fallback"
            props = _component_properties(corridor)
            damage_fraction = float(corridor.sum() / max(valid.sum(), 1))
            quality = wf.classify_quality(
                float(aligned["metadata"].get("valid_fraction", 0)),
                damage_fraction,
                bool(final_lines) and candidate_name != "rejected",
                {
                    "quality": {
                        "reject_valid_fraction_below": 0.02,
                        "low_confidence_valid_fraction_below": 0.10,
                        "max_prediction_fraction": 0.30,
                        "max_path_tortuosity": 4.0,
                    }
                },
                max((wf.line_tortuosity(line) or 0.0 for line in final_lines), default=None),
            )
            quality_reasons: list[str] = []
            if quality != "Rejected" and candidate_name not in {"hybrid_probability", "water_aware_multi_path"}:
                quality = "Low confidence"
                quality_reasons.append("The broad hybrid mask was implausible; a narrower independently supported candidate was required.")
            if quality != "Rejected" and max((wf.line_tortuosity(line) or 0.0 for line in final_lines), default=0.0) > 2.0:
                quality = "Low confidence"
                quality_reasons.append("The extracted centerline is unusually tortuous and needs manual review.")
            if quality == "Rejected":
                quality_reasons.append("No candidate passed the corridor extent, elongation, coverage, and continuity checks.")
            published_lines = final_lines if quality != "Rejected" else []
            write_corridor_products(corridor, published_lines, profile, case_dir)
            nws = wf.find_nws_path(case_id, shapefiles)
            descriptions = tuple(case_lookup.get(case_id, {}).get("band_descriptions", []))
            band_mapping = resolve_band_mapping(descriptions)
            rgb_indices = None
            if band_mapping:
                rgb_indices = [band_mapping["red"] + 1, band_mapping["green"] + 1, band_mapping["blue"] + 1]
            map_assets = case_dir / "map_assets"
            map_assets.mkdir(exist_ok=True)
            wf.make_maps(
                case_id,
                Path(aligned["before"]),
                Path(aligned["after"]),
                probability,
                corridor,
                published_lines,
                nws,
                map_assets,
                case_dir,
                model_label="Agentic ensemble",
                quality=quality,
                rgb_indices=rgb_indices,
            )
            candidate_figure = case_dir / "candidate_comparison.png"
            create_candidate_figure(
                case_id,
                Path(aligned["after"]),
                probability,
                kmeans_mask,
                corridor,
                candidate_records,
                candidate_figure,
                descriptions_override=descriptions,
            )
            curve_figure = case_dir / "modis_curve_fit_comparison.png"
            create_curve_figure(
                Path(aligned["after"]),
                corridor,
                skeleton_line,
                curve_line,
                curve_figure,
                descriptions_override=descriptions,
            )
            pd.DataFrame(candidate_records).to_csv(case_dir / "candidate_scores.csv", index=False)
            path_metrics = {
                "candidate": candidate_name,
                "path_method": path_method if published_lines else "rejected",
                "confidence": quality,
                "predicted_path_count": len(published_lines),
                "water_gap_bridge_count": len(multi_diagnostics.get("bridges", [])),
                "damage_fraction": damage_fraction,
                "elongation": props["elongation"],
                "path_tortuosity": max((wf.line_tortuosity(line) or 0.0 for line in published_lines), default=None),
                "path_length_crs_units": float(sum(line.length for line in published_lines)) if published_lines else None,
                "multi_path_diagnostics": multi_diagnostics,
                "curve_fit": curve_metadata,
                "candidate_scores": candidate_records,
                "quality_reasons": quality_reasons,
                "important_rule": "NWS geometry was not used to choose or draw the prediction.",
            }
            (case_dir / "path_metrics.json").write_text(json.dumps(path_metrics, indent=2))
            case_result["path"] = {
                **path_metrics,
                "centerline": str(case_dir / "predicted_path_centerline.geojson"),
                "corridor": str(case_dir / "predicted_damage_corridor.geojson"),
                "mask": str(case_dir / "predicted_damage_mask.tif"),
                "final_map": str(case_dir / "final_path_map.png"),
                "candidate_figure": str(candidate_figure),
                "curve_figure": str(curve_figure),
            }
        except Exception as exc:
            errors = _append_error({**state, "errors": errors}, case_id, "path_agent", exc)
    return {
        "results": results,
        "errors": errors,
        "stage_log": _append_stage(state, "path_agent", "complete", "Selected plausible imagery-derived corridors and extracted curved centerlines."),
    }


def _sample_geometry(line: LineString, count: int = 200) -> list[Point]:
    if line.length <= 0:
        return []
    return [line.interpolate(index / max(count - 1, 1), normalized=True) for index in range(count)]


def validation_agent(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    shapefiles = Path(state["shapefiles"])
    model_metadata = _safe_json(Path(state["model_dir"]) / "model_metadata.json", {})
    training_cases = set(model_metadata.get("training_cases", []))
    results = dict(state.get("results", {}))
    errors = list(state.get("errors", []))
    for case_id, aligned in state.get("aligned", {}).items():
        try:
            case_dir = output / "cases" / case_id
            path_result = results.setdefault(case_id, {}).get("path", {})
            centerline_path = Path(path_result.get("centerline", ""))
            predicted_lines: list[LineString] = []
            predicted_line = None
            if centerline_path.exists():
                predicted_gdf = gpd.read_file(centerline_path)
                if len(predicted_gdf) and predicted_gdf.geometry.notna().any() and (~predicted_gdf.geometry.is_empty).any():
                    for geometry in predicted_gdf.geometry:
                        if isinstance(geometry, LineString):
                            predicted_lines.append(geometry)
                        elif hasattr(geometry, "geoms"):
                            predicted_lines.extend(part for part in geometry.geoms if isinstance(part, LineString))
                    predicted_line = gpd.GeoSeries(predicted_lines, crs=predicted_gdf.crs).union_all() if predicted_lines else None
            nws_path = wf.find_nws_path(case_id, shapefiles)
            metrics: dict[str, Any] = {
                "case_id": case_id,
                "nws_available": bool(nws_path),
                "evaluation_role": "training-reference comparison" if case_id in training_cases else "unlabelled inference",
                "prediction_confidence": path_result.get("confidence", "Rejected"),
                "nws_agreement": "not available",
                "mean_nws_to_prediction_distance_pixels": None,
                "nws_path_overlap_percent": None,
                "predicted_path_count": len(predicted_lines),
                "reference_path_count": None,
                "matched_path_count": None,
                "path_count_precision": None,
                "path_count_recall": None,
                "path_count_f1": None,
            }
            if nws_path:
                with rasterio.open(aligned["before"]) as reference:
                    pixel_size = max(abs(reference.transform.a), abs(reference.transform.e))
                    target_crs = reference.crs
                nws_gdf = gpd.read_file(nws_path)
                if nws_gdf.crs and target_crs:
                    nws_gdf = nws_gdf.to_crs(target_crs)
                valid_geometries = nws_gdf.geometry[nws_gdf.geometry.notna() & ~nws_gdf.geometry.is_empty]
                if len(valid_geometries):
                    path_set_metrics, path_matches = evaluate_path_set(
                        predicted_lines,
                        nws_gdf.loc[valid_geometries.index],
                        target_crs,
                    )
                    pd.DataFrame(path_matches).to_csv(case_dir / "agentic_per_path_matching.csv", index=False)
                    nws_union = valid_geometries.union_all()
                    samples = _sample_geometry(nws_union, 250) if isinstance(nws_union, LineString) else []
                    if not samples and hasattr(nws_union, "geoms"):
                        samples = [point for geom in nws_union.geoms if isinstance(geom, LineString) for point in _sample_geometry(geom, 120)]
                    distances = (
                        [point.distance(predicted_line) / max(pixel_size, 1e-12) for point in samples]
                        if predicted_line is not None
                        else []
                    )
                    buffer_distance = 12 * pixel_size
                    overlap = (
                        float(nws_union.intersection(predicted_line.buffer(buffer_distance)).length / max(nws_union.length, 1e-12))
                        if predicted_line is not None
                        else 0.0
                    )
                    mean_distance = float(np.mean(distances)) if distances else None
                    if overlap >= 0.70 and (mean_distance is None or mean_distance <= 12):
                        agreement = "High"
                    elif overlap >= 0.40 and (mean_distance is None or mean_distance <= 30):
                        agreement = "Moderate"
                    else:
                        agreement = "Low"
                    metrics.update(
                        {
                            "nws_agreement": agreement,
                            "mean_nws_to_prediction_distance_pixels": mean_distance,
                            "nws_path_overlap_percent": 100.0 * overlap,
                            **path_set_metrics,
                        }
                    )
            (case_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
            results[case_id]["validation"] = metrics
        except Exception as exc:
            errors = _append_error({**state, "errors": errors}, case_id, "validation_agent", exc)
    return {
        "results": results,
        "errors": errors,
        "stage_log": _append_stage(state, "validation_agent", "complete", "Compared published predictions with valid NWS geometry without copying reference lines."),
    }


def report_agent(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in state.get("cases", []):
        case_id = case["case_id"]
        result = state.get("results", {}).get(case_id, {})
        path = result.get("path", {})
        validation = result.get("validation", {})
        rows.append(
            {
                "case_id": case_id,
                "confidence": path.get("confidence", "Failed"),
                "selected_candidate": path.get("candidate", ""),
                "path_method": path.get("path_method", ""),
                "damage_fraction": path.get("damage_fraction"),
                "elongation": path.get("elongation"),
                "path_tortuosity": path.get("path_tortuosity"),
                "nws_available": validation.get("nws_available", False),
                "nws_agreement": validation.get("nws_agreement", "not available"),
                "nws_overlap_percent": validation.get("nws_path_overlap_percent"),
                "predicted_path_count": validation.get("predicted_path_count"),
                "reference_path_count": validation.get("reference_path_count"),
                "matched_path_count": validation.get("matched_path_count"),
                "reference_completeness": validation.get("reference_completeness", "not available"),
                "documented_path_recovery": validation.get("documented_path_recovery"),
                "unverified_predicted_path_count": validation.get("unverified_predicted_path_count"),
                "path_count_precision": validation.get("path_count_precision"),
                "path_count_recall": validation.get("path_count_recall"),
                "path_count_f1": validation.get("path_count_f1"),
                "evaluation_role": validation.get("evaluation_role", ""),
                "final_map": path.get("final_map", ""),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = reports / "agentic_case_summary.csv"
    summary.to_csv(summary_path, index=False)
    (reports / "agent_execution_log.json").write_text(json.dumps(state.get("stage_log", []), indent=2))
    (reports / "agent_errors.json").write_text(json.dumps(state.get("errors", []), indent=2))

    if len(summary):
        colors = summary["confidence"].map({"Moderate confidence": "#007C78", "Low confidence": "#E0A100", "Rejected": "#A13A32"}).fillna("#6B7280")
        fig, axis = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
        values = summary["damage_fraction"].fillna(0)
        axis.bar(summary["case_id"], values, color=colors)
        axis.axhline(0.30, color="#A13A32", linestyle="--", label="Rejection threshold")
        axis.set_ylabel("Predicted corridor fraction")
        axis.set_title("Batch quality-control summary", fontsize=18, fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        batch_figure = reports / "batch_quality_summary.png"
        fig.savefig(batch_figure, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        batch_figure = reports / "batch_quality_summary.png"

    table_rows = []
    for row in rows:
        map_path = Path(row["final_map"]) if row["final_map"] else None
        relative_map = map_path.relative_to(output) if map_path and map_path.exists() else None
        map_link = f'<a href="{relative_map}">map</a>' if relative_map else ""
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['case_id']))}</td>"
            f"<td>{html.escape(str(row['confidence']))}</td>"
            f"<td>{html.escape(str(row['selected_candidate']))}</td>"
            f"<td>{html.escape(str(row['nws_agreement']))}</td>"
            f"<td>{map_link}</td>"
            "</tr>"
        )
    report_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Agentic Tornado Path Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;color:#16221d}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5cf;padding:9px;text-align:left}}th{{background:#eef3ef}}.warning{{color:#8b2f2a}}</style></head>
<body><h1>Agentic Tornado Damage-Path Report</h1>
<p>Predictions are derived from imagery. NWS/DAT geometry is used only for label preparation and post-prediction comparison.</p>
<p>NWS/DAT is authoritative for documented paths, but the available layer may not contain every path in the scene. Extra imagery-derived paths are therefore marked unverified rather than automatically counted as false positives.</p>
<p class="warning">Rejected means the workflow refused to publish an unreliable centerline.</p>
<table><thead><tr><th>Case</th><th>Confidence</th><th>Selected candidate</th><th>DAT documented-path agreement</th><th>Output</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></body></html>"""
    (output / "agentic_report.html").write_text(report_html)
    return {
        "stage_log": _append_stage(state, "report_agent", "complete", f"Wrote summary for {len(rows)} cases."),
    }


def _find_artifact_runtime() -> tuple[Path | None, Path | None]:
    node_candidates = list((Path.home() / ".cache" / "codex-runtimes").glob("*/dependencies/node/bin/node"))
    setup_candidates = list((Path.home() / ".codex" / "plugins" / "cache").glob("openai-primary-runtime/presentations/*/skills/presentations/container_tools/setup_artifact_tool_workspace.mjs"))
    return (node_candidates[-1] if node_candidates else None, setup_candidates[-1] if setup_candidates else None)


def _build_manifest(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    slides: list[dict[str, Any]] = []
    labels = [
        ("before_after", "Full-scene BEFORE and AFTER"),
        ("signed_difference", "Signed band differences"),
        ("absolute_difference", "Absolute band differences"),
        ("distributions", "Band distribution shifts"),
        ("statistics", "Channel means and variances"),
        ("correlations", "Band correlation matrices"),
        ("composites", "Multispectral band combinations"),
        ("ndvi", "NDVI vegetation-change map"),
    ]
    for case in state.get("cases", []):
        case_id = case["case_id"]
        result = state.get("results", {}).get(case_id, {})
        confidence = result.get("path", {}).get("confidence", "Failed")
        validation = result.get("validation", {})
        nws_agreement = validation.get("nws_agreement", "not available")
        recovery = validation.get("documented_path_recovery")
        unverified = validation.get("unverified_predicted_path_count")
        recovery_text = "not available" if recovery is None else f"{100.0 * float(recovery):.0f}%"
        unverified_text = "not available" if unverified is None else str(int(unverified))
        status = (
            f"{confidence} | DAT agreement: {nws_agreement} | "
            f"documented recovery: {recovery_text} | unverified candidates: {unverified_text}"
        )
        for key, title in labels:
            image = result.get("eda", {}).get("figures", {}).get(key, "")
            if image and Path(image).exists():
                slides.append({"case_id": case_id, "title": f"{case_id}: {title}", "status": status, "image": image})
        for image_key, title in [
            (result.get("kmeans", {}).get("figure", ""), "K-Means change candidates"),
            (result.get("path", {}).get("candidate_figure", ""), "Candidate corridor comparison"),
            (result.get("path", {}).get("curve_figure", ""), "Tornado_Modis curve comparison"),
            (result.get("path", {}).get("final_map", ""), "Final path on the AFTER image"),
        ]:
            if image_key and Path(image_key).exists():
                slides.append({"case_id": case_id, "title": f"{case_id}: {title}", "status": status, "image": image_key})
    return {
        "title": "Agentic Tornado Damage-Path Analysis",
        "subtitle": "CRS-safe alignment, multispectral EDA, multi-path extraction, water-aware gap bridging, and DAT comparison",
        "output_pptx": str(output / "presentation" / "agentic_tornado_path_analysis.pptx"),
        "preview_dir": str(output / "presentation" / "rendered"),
        "batch_figure": str(output / "reports" / "batch_quality_summary.png"),
        "slides": slides,
        "sources": [
            "Professor-provided BEFORE/AFTER Landsat surface-reflectance imagery",
            "Professor-provided NOAA/NWS DAT shapefiles where valid geometry exists; available coverage may be incomplete",
            "DiLiu2023/Tornado_Modis commit 7151aec9153dd9a655e3ac480f3f180e04861b43 (MIT), adapted for portable curve fitting",
        ],
    }


def presentation_agent(state: AgentState) -> dict[str, Any]:
    output = Path(state["output"])
    presentation_dir = output / "presentation"
    presentation_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(state)
    manifest_path = presentation_dir / "deck_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    warnings = list(state.get("warnings", []))
    presentation = ""
    if state.get("build_slides", True):
        node, setup = _find_artifact_runtime()
        if node and setup:
            workspace = presentation_dir / ".artifact_tool"
            subprocess.run([str(node), str(setup), "--workspace", str(workspace)], check=True)
            source_script = PROJECT / "tools" / "build_agentic_deck.mjs"
            runtime_script = workspace / "build_agentic_deck.mjs"
            shutil.copyfile(source_script, runtime_script)
            subprocess.run([str(node), str(runtime_script), str(manifest_path)], check=True)
            presentation = manifest["output_pptx"]
        else:
            warnings.append("Artifact-tool runtime not found; deck_manifest.json was created but PPTX export was skipped.")
    return {
        "manifest": str(manifest_path),
        "presentation": presentation,
        "warnings": warnings,
        "stage_log": _append_stage(state, "presentation_agent", "complete", f"Prepared {len(manifest['slides'])} evidence slides."),
    }


def build_graph():
    if StateGraph is None:
        class SequentialWorkflow:
            def invoke(self, initial_state, _config=None):
                state = dict(initial_state)
                inventory_update = inventory_agent(state)
                state.update(inventory_update)
                if not state.get("cases"):
                    for agent in (report_agent, presentation_agent):
                        state.update(agent(state))
                    return state
                for agent in (
                    geospatial_agent,
                    eda_agent,
                    clustering_agent,
                    prediction_agent,
                    path_agent,
                    validation_agent,
                    report_agent,
                    presentation_agent,
                ):
                    state.update(agent(state))
                state.setdefault("warnings", []).append(
                    "LangGraph is not installed; the same bounded agents ran in deterministic sequence."
                )
                return state

        return SequentialWorkflow()
    builder = StateGraph(AgentState)
    builder.add_node("inventory_agent", inventory_agent)
    builder.add_node("geospatial_agent", geospatial_agent)
    builder.add_node("eda_agent", eda_agent)
    builder.add_node("clustering_agent", clustering_agent)
    builder.add_node("prediction_agent", prediction_agent)
    builder.add_node("path_agent", path_agent)
    builder.add_node("validation_agent", validation_agent)
    builder.add_node("report_agent", report_agent)
    builder.add_node("presentation_agent", presentation_agent)
    builder.add_edge(START, "inventory_agent")
    builder.add_conditional_edges("inventory_agent", route_after_inventory, {"geospatial_agent": "geospatial_agent", "report_agent": "report_agent"})
    builder.add_edge("geospatial_agent", "eda_agent")
    builder.add_edge("eda_agent", "clustering_agent")
    builder.add_edge("clustering_agent", "prediction_agent")
    builder.add_edge("prediction_agent", "path_agent")
    builder.add_edge("path_agent", "validation_agent")
    builder.add_edge("validation_agent", "report_agent")
    builder.add_edge("report_agent", "presentation_agent")
    builder.add_edge("presentation_agent", END)
    return builder.compile()


def run_agentic_workflow(initial_state: AgentState) -> AgentState:
    graph = build_graph()
    result = graph.invoke(initial_state, {"recursion_limit": 30})
    output = Path(result["output"])
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "reports" / "final_workflow_state.json").write_text(json.dumps(result, indent=2))
    return result
