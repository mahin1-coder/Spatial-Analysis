#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs_all_cases" / "reports"


def classify(row: pd.Series) -> tuple[str, str]:
    if not bool(row.get("path_found", False)) or int(row.get("predicted_path_count", 0)) == 0:
        return "Rejected", "No plausible imagery-derived corridor was extracted."
    if bool(row.get("nws_available", False)):
        recovery = float(row.get("documented_path_recovery", 0) or 0)
        coverage = float(row.get("nws_nws_path_coverage", 0) or 0)
        precision = float(row.get("nws_prediction_precision_against_nws_buffer", 0) or 0)
        if recovery >= 1 and coverage >= 0.60 and precision >= 0.30:
            return "DAT-validated", "Imagery prediction recovered the available official DAT path."
        if recovery >= 1 and coverage >= 0.40:
            return "Partial / review", "Some DAT agreement exists, but spatial precision or coverage is insufficient."
        return "Rejected", "Imagery prediction did not recover the available official DAT path."
    return (
        "Unverified imagery candidate",
        "No official DAT geometry is available; internal geometry alone cannot confirm the path.",
    )


def main() -> int:
    source = pd.read_csv(REPORTS / "model_inference_summary.csv")
    statuses = source.apply(classify, axis=1, result_type="expand")
    source["research_status"] = statuses[0]
    source["status_reason"] = statuses[1]
    source["production_accepted"] = source["research_status"].eq("DAT-validated")
    source.to_csv(REPORTS / "research_acceptance_report.csv", index=False)
    summary = (
        source.groupby("research_status", as_index=False)
        .agg(cases=("case_id", "count"))
        .sort_values("research_status")
    )
    summary.to_csv(REPORTS / "research_acceptance_summary.csv", index=False)
    print(summary.to_string(index=False))
    rejected = source.loc[source["research_status"].eq("Rejected"), "case_id"].tolist()
    print("Rejected:", ", ".join(rejected) if rejected else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
