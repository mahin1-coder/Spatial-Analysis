#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib

from cluster_path.core import discover_pairs, load_analysis_data
from cluster_path.multipath import stable_water_mask
from cluster_path.context_layers import build_context_masks, save_context_provenance
from cluster_path.model import (
    fit_model,
    infer_probability,
    model_features,
    postprocess_probability,
    run_grouped_model_selection,
    save_model_bundle,
    save_model_case,
    train_samples_for_cases,
)


PROJECT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run the NWS-validated tornado path model")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "cluster_path.yaml")
    parser.add_argument("--case", action="append", help="Restrict final inference to selected cases.")
    parser.add_argument("--inference-only", action="store_true", help="Reuse the saved model and refresh case outputs.")
    parser.add_argument(
        "--resume-selection",
        action="store_true",
        help="Reuse completed grouped-validation reports, refit the selected model, and run inference.",
    )
    parser.add_argument("--model-name", default="research_v2", help="Versioned model output directory name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = (PROJECT / config["data"]["raster_dir"]).resolve()
    shapefiles = (PROJECT / config["data"]["shapefile_dir"]).resolve()
    manual_label_root = (
        (PROJECT / config["data"]["manual_label_dir"]).resolve()
        if config["data"].get("manual_label_dir")
        else None
    )
    output = (PROJECT / config["data"]["output_dir"]).resolve()
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    analysis = config["analysis"]
    model_cfg = config["model"]
    postprocess_cfg = config.get("postprocessing", {})
    pairs = discover_pairs(source)

    inventory_path = reports / "noaa_dat_reference_inventory.csv"
    if not inventory_path.exists():
        raise SystemExit(f"Missing NOAA reference inventory: {inventory_path}")
    inventory = pd.read_csv(inventory_path).fillna("")
    event_groups = {
        str(row.case_id): f"{row.storm_date}|{row.event_id or row.case_id}"
        for row in inventory.itertuples()
        if str(row.status) == "matched"
    }

    model_dir = output / "models" / args.model_name
    if args.inference_only:
        model = joblib.load(model_dir / "tornado_path_model.joblib")
        metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        validation_summary = metadata.get("validation_summary", {})
        selected_kind = str(validation_summary["selected_model"])
        selected_percentile = float(validation_summary["selected_probability_percentile"])
        samples = {case: (np.empty((0, 0)), np.empty(0)) for case in metadata["training_cases"]}
        cv = pd.read_csv(reports / "cross_validation_results.csv")
        y = np.empty(0)
    else:
        samples, feature_names, label_inventory = train_samples_for_cases(
            pairs,
            shapefiles,
            max_dimension=int(analysis["max_dimension"]),
            clusters=int(analysis["clusters"]),
            seed=int(analysis["random_seed"]),
            positive_limit=int(model_cfg.get("positive_samples_per_case", 6000)),
            negative_limit=int(model_cfg.get("negative_samples_per_case", 12000)),
            manual_label_root=manual_label_root,
        )
        label_inventory.to_csv(reports / "training_label_inventory.csv", index=False)
        if len(samples) < 3:
            raise SystemExit("At least three NWS-labeled tornado cases are required.")

    if not args.inference_only:
        print(f"[model] NWS-labeled cases: {', '.join(samples)}", flush=True)
        if args.resume_selection:
            comparison = pd.read_csv(reports / "model_comparison.csv")
            winner = comparison.loc[comparison["selected"].astype(bool)].iloc[0]
            selected_kind = str(winner["model"])
            selected_percentile = float(winner["probability_percentile"])
            cv = pd.read_csv(reports / "cross_validation_results.csv")
            print("[model] Reusing completed grouped-validation selection.", flush=True)
        else:
            detailed, comparison, splits, selected_kind, selected_percentile = run_grouped_model_selection(
                pairs,
                samples,
                shapefiles,
                event_groups,
                max_dimension=int(analysis["max_dimension"]),
                clusters=int(analysis["clusters"]),
                model_kinds=list(model_cfg.get("candidates", [model_cfg.get("type", "extra_trees")])),
                percentiles=[
                    float(value)
                    for value in model_cfg.get(
                        "probability_percentiles", [model_cfg.get("probability_percentile", 92)]
                    )
                ],
                folds=int(model_cfg.get("validation_folds", 5)),
                seed=int(analysis["random_seed"]),
                manual_label_root=manual_label_root,
            )
            detailed.to_csv(reports / "threshold_analysis.csv", index=False)
            comparison["selected"] = (
                (comparison["model"] == selected_kind)
                & (comparison["probability_percentile"] == selected_percentile)
            )
            comparison.to_csv(reports / "model_comparison.csv", index=False)
            splits.to_csv(reports / "dataset_split_report.csv", index=False)
            cv = detailed[
                (detailed["model"] == selected_kind)
                & (detailed["probability_percentile"] == selected_percentile)
            ].copy()
            cv.to_csv(reports / "cross_validation_results.csv", index=False)
            cv.to_csv(reports / "leave_one_tornado_out.csv", index=False)
        print(comparison.to_string(index=False), flush=True)
        print(f"[model] Selected {selected_kind} at percentile {selected_percentile:g}", flush=True)

        x = np.vstack([samples[case][0] for case in samples])
        y = np.concatenate([samples[case][1] for case in samples])
        model = fit_model(x, y, int(analysis["random_seed"]), kind=selected_kind)
        validation_summary = {
            "selected_model": selected_kind,
            "selected_probability_percentile": selected_percentile,
            "mean_grouped_cv_dice": float(cv["dice"].mean()),
            "mean_grouped_cv_iou": float(cv["iou"].mean()),
            "mean_grouped_cv_precision": float(cv["precision"].mean()),
            "mean_grouped_cv_recall": float(cv["recall"].mean()),
            "labeled_cases": len(samples),
            "independent_event_groups": len(set(event_groups.get(case, case) for case in samples)),
        }
        save_model_bundle(
        model,
        feature_names,
        sorted(samples, key=lambda case: int(case[3:])),
            model_dir,
            config,
            validation_summary=validation_summary,
        )

    requested = {value.upper() for value in args.case or []}
    inference_pairs = [pair for pair in pairs if not requested or pair.case_id in requested]
    rows = []
    for pair in inference_pairs:
        print(f"[model] Inference {pair.case_id}", flush=True)
        case_settings = dict(postprocess_cfg)
        try:
            data = load_analysis_data(pair, max_dimension=int(analysis["max_dimension"]))
        except Exception as error:
            case_dir = output / "cases" / pair.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "case_id": pair.case_id,
                "evaluation_type": "rejected",
                "path_found": False,
                "prediction_quality_status": "Rejected",
                "error": str(error),
            }
            (case_dir / "processing_error.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            rows.append(failure)
            print(f"[model] Rejected {pair.case_id}: {error}", flush=True)
            continue
        features, _, diagnostics = model_features(
            data,
            int(analysis["clusters"]),
            int(analysis["random_seed"]) + int(pair.case_id[3:]),
        )
        probability = infer_probability(model, features, data.valid)
        spectral_water = stable_water_mask(
            data,
            mndwi_threshold=float(case_settings.get("mndwi_threshold", 0.05)),
        )
        context_masks = build_context_masks(data, spectral_water, config.get("context_layers", {}))
        corridor, postprocess = postprocess_probability(
            probability,
            data.valid,
            selected_percentile,
            water_mask=context_masks.water,
            exclusion_mask=context_masks.exclusion,
            crossing_mask=context_masks.crossing,
            max_paths=int(case_settings.get("max_paths", 6)),
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
            max_feature_follow_fraction=float(
                case_settings.get("maximum_feature_follow_fraction", 0.65)
            ),
        )
        save_context_provenance(
            output / "cases" / pair.case_id / "context_layer_provenance.json",
            context_masks,
        )
        postprocess["straighten_centerline"] = bool(
            case_settings.get("straighten_centerline", False)
        )
        evaluation_type = "training-set result" if pair.case_id in samples else "unverified imagery inference"
        row = save_model_case(
            pair,
            data,
            diagnostics,
            probability,
            corridor,
            output,
            shapefiles,
            evaluation_type,
            postprocess,
        )
        row["postprocess_reason"] = postprocess.get("reason")
        rows.append(row)
    pd.DataFrame(rows).to_csv(reports / "model_inference_summary.csv", index=False)
    summary = {
        "training_cases": sorted(samples, key=lambda case: int(case[3:])),
        "training_samples": int(len(y)) if len(y) else None,
        "positive_fraction": float(y.mean()) if len(y) else None,
        "selected_model": selected_kind,
        "selected_probability_percentile": selected_percentile,
        "mean_grouped_cv_dice": float(cv["dice"].mean()),
        "mean_grouped_cv_iou": float(cv["iou"].mean()),
        "final_inference_cases": len(rows),
        "nws_used_at_inference": False,
    }
    (reports / "model_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
