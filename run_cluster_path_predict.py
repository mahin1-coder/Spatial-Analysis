#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from cluster_path.core import discover_pairs, load_analysis_data
from cluster_path.model import (
    infer_probability,
    model_features,
    postprocess_probability,
    save_model_case,
)


PROJECT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained tornado path model on a new folder")
    parser.add_argument("--source", type=Path, required=True, help="Folder containing BEFORE/AFTER GeoTIFF pairs")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT / "outputs_all_cases" / "models" / "final_model" / "tornado_path_model.joblib",
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs_all_cases" / "future_predictions")
    parser.add_argument("--shapefiles", type=Path, default=PROJECT / "data" / "raw" / "shapefiles" / "_none")
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--probability-percentile", type=float, default=92.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = discover_pairs(args.source.resolve())
    if not pairs:
        raise SystemExit("No complete, unambiguous BEFORE/AFTER pairs were found.")
    model = joblib.load(args.model.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for pair in pairs:
        print(f"[predict] {pair.case_id}", flush=True)
        try:
            data = load_analysis_data(pair, max_dimension=args.max_dimension)
            features, names, diagnostics = model_features(
                data,
                args.clusters,
                args.seed + int(pair.case_id[3:]),
            )
            if features.shape[-1] != int(model.n_features_in_):
                raise ValueError(
                    f"Feature mismatch: model expects {model.n_features_in_}, generated {features.shape[-1]}"
                )
            probability = infer_probability(model, features, data.valid)
            corridor, postprocess = postprocess_probability(
                probability,
                data.valid,
                args.probability_percentile,
            )
            row = save_model_case(
                pair,
                data,
                diagnostics,
                probability,
                corridor,
                output,
                args.shapefiles.resolve(),
                "future inference",
            )
            row["postprocess_reason"] = postprocess.get("reason")
            row["feature_count"] = len(names)
            rows.append(row)
        except Exception as exc:
            failures.append({"case_id": pair.case_id, "error": f"{type(exc).__name__}: {exc}"})

    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(reports / "prediction_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(reports / "failures.csv", index=False)
    summary = {
        "cases_found": len(pairs),
        "cases_completed": len(rows),
        "cases_failed": len(failures),
        "model": str(args.model.resolve()),
        "nws_required_for_inference": False,
        "warning": "Unlabeled results are candidates until independently validated.",
    }
    (reports / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
