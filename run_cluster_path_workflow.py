#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from cluster_path.core import discover_pairs, process_case, result_row


PROJECT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Imagery-first tornado damage-path workflow")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "cluster_path.yaml")
    parser.add_argument("--case", action="append", help="Process one case, e.g. TOR10. Repeat as needed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = (PROJECT / config["data"]["raster_dir"]).resolve()
    shapefiles = (PROJECT / config["data"]["shapefile_dir"]).resolve()
    output = (PROJECT / config["data"]["output_dir"]).resolve()
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(source)
    requested = {value.upper() for value in args.case or []}
    if requested:
        pairs = [pair for pair in pairs if pair.case_id in requested]
    if not pairs:
        raise SystemExit("No complete BEFORE/AFTER pairs were found.")

    pairing_rows = [
        {
            "case_id": pair.case_id,
            "before": str(pair.before),
            "after": str(pair.after),
            "pairing_status": "unambiguous",
        }
        for pair in pairs
    ]
    pd.DataFrame(pairing_rows).to_csv(reports / "pairing_report.csv", index=False)

    results = []
    failures = []
    analysis = config["analysis"]
    for pair in pairs:
        print(f"[cluster-path] Processing {pair.case_id}", flush=True)
        try:
            result = process_case(
                pair,
                output,
                shapefiles,
                max_dimension=int(analysis["max_dimension"]),
                clusters=int(analysis["clusters"]),
                percentile=float(analysis["candidate_percentile"]),
                min_area_fraction=float(analysis["min_area_fraction"]),
                seed=int(analysis["random_seed"]),
            )
            results.append(result)
            print(
                f"[cluster-path] {pair.case_id}: {result.status}, "
                f"{result.confidence} confidence, NWS={result.nws_available}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"case_id": pair.case_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[cluster-path] {pair.case_id}: FAILED - {exc}", flush=True)

    summary = pd.DataFrame([result_row(result) for result in results])
    summary.to_csv(reports / "case_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(reports / "failures.csv", index=False)
    run_metadata = {
        "cases_requested": len(pairs),
        "cases_completed": len(results),
        "cases_failed": len(failures),
        "accepted": sum(result.status == "accepted" for result in results),
        "rejected": sum(result.status == "rejected" for result in results),
        "nws_validation_cases": sum(result.nws_available for result in results),
        "scientific_rule": "Predictions are derived from imagery. NWS is loaded only after prediction.",
    }
    (reports / "run_summary.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(json.dumps(run_metadata, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
