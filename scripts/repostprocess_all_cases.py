#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import rasterio
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cluster_path.core import _robust_image_normalize, discover_pairs, load_analysis_data
from cluster_path.model import postprocess_probability, save_model_case
from cluster_path.multipath import stable_water_mask


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh multi-path outputs from saved imagery-only probability rasters."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Restrict refresh to selected case IDs.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = (PROJECT / config["data"]["raster_dir"]).resolve()
    references = (PROJECT / config["data"]["shapefile_dir"]).resolve()
    output = (PROJECT / config["data"]["output_dir"]).resolve()
    analysis = config["analysis"]
    settings = config.get("postprocessing", {})
    rows = []
    requested = {value.upper() for value in args.case or []}
    for pair in discover_pairs(source):
        if requested and pair.case_id not in requested:
            continue
        case_dir = output / "cases" / pair.case_id
        probability_path = case_dir / "model_probability.tif"
        if not probability_path.exists():
            print(f"[skip] {pair.case_id}: no saved probability raster", flush=True)
            continue
        prior_metrics = case_dir / "model_case_metrics.json"
        prior = json.loads(prior_metrics.read_text()) if prior_metrics.exists() else {}
        print(f"[postprocess] {pair.case_id}", flush=True)
        data = load_analysis_data(pair, max_dimension=int(analysis["max_dimension"]))
        with rasterio.open(probability_path) as dataset:
            probability = dataset.read(1)
        if probability.shape != data.valid.shape:
            raise RuntimeError(
                f"{pair.case_id}: probability shape {probability.shape} does not match analysis grid {data.valid.shape}"
            )
        water = stable_water_mask(
            data, mndwi_threshold=float(settings.get("mndwi_threshold", 0.05))
        )
        case_settings = {**settings, **settings.get("case_overrides", {}).get(pair.case_id, {})}
        corridor, diagnostics = postprocess_probability(
            probability,
            data.valid,
            float(config["model"].get("probability_percentile", 92.0)),
            water_mask=water,
            max_paths=int(case_settings.get("max_paths", 3)),
            max_gap_pixels=int(case_settings.get("max_gap_pixels", 45)),
            min_water_fraction=float(
                case_settings.get("minimum_water_fraction_for_bridge", 0.30)
            ),
            max_bridge_angle_degrees=float(
                case_settings.get("maximum_bridge_angle_degrees", 50.0)
            ),
            min_relative_path_score=float(
                case_settings.get("minimum_relative_path_score", 0.35)
            ),
            water_exclusion_buffer_pixels=int(
                case_settings.get("water_exclusion_buffer_pixels", 6)
            ),
            axis_filter_half_width_pixels=(
                float(case_settings["axis_filter_half_width_pixels"])
                if "axis_filter_half_width_pixels" in case_settings
                else None
            ),
        )
        diagnostics["straighten_centerline"] = bool(
            case_settings.get("straighten_centerline", False)
        )
        rows.append(
            save_model_case(
                pair,
                data,
                {"after_normalized": _robust_image_normalize(data.after, data.valid)},
                probability,
                corridor,
                output,
                references,
                prior.get("evaluation_type", "unverified imagery inference"),
                diagnostics,
            )
        )
    all_rows = []
    for pair in discover_pairs(source):
        metrics_path = output / "cases" / pair.case_id / "model_case_metrics.json"
        if metrics_path.exists():
            all_rows.append(json.loads(metrics_path.read_text()))
    report = output / "reports" / "model_inference_summary.csv"
    pd.DataFrame(all_rows).to_csv(report, index=False)
    print(f"[report] {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
