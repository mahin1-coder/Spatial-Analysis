#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from prithvi_pipeline.data import discover_cases
from prithvi_pipeline.pipeline import binary_metrics, process_prediction


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "outputs_prithvi_final"
DECODER = PROJECT / "outputs_prithvi"
FOREST = PROJECT / "outputs_prithvi_rf"
WEIGHT_FOREST = 0.9
THRESHOLD = 0.55
MODEL_NAME = "Prithvi-EO-2.0 ensemble: 90% embedding Extra Trees + 10% segmentation decoder"


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read(1).astype("float32")


def label_info(case_id: str) -> dict[str, np.ndarray] | None:
    mask = DECODER / "cases" / case_id / "ground_truth_damage_mask.tif"
    centerline = DECODER / "cases" / case_id / "ground_truth_centerline_mask.tif"
    if not mask.exists() or not centerline.exists():
        return None
    return {"label": read(mask) > 0, "centerline": read(centerline) > 0}


def combine(decoder: Path, forest: Path) -> np.ndarray:
    return ((1.0 - WEIGHT_FOREST) * read(decoder) + WEIGHT_FOREST * read(forest)).astype("float32")


def copy_streamed(source: Path, destination: Path) -> None:
    """Avoid macOS clone/xattr timeouts on large generated model files."""

    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "reports").mkdir(parents=True, exist_ok=True)
    cases = discover_cases(
        PROJECT / "data" / "raw" / "rasters",
        PROJECT / "outputs_hybrid_path" / "cases",
        OUTPUT,
        PROJECT / "data" / "raw" / "shapefiles",
    )

    cv_rows = []
    for case in cases:
        decoder_path = DECODER / "evaluation" / case.case_id / "leave_one_case_out_probability.tif"
        forest_path = FOREST / "evaluation" / case.case_id / "leave_one_case_out_probability.tif"
        info = label_info(case.case_id)
        if info is None or not decoder_path.exists() or not forest_path.exists():
            continue
        probability = combine(decoder_path, forest_path)
        with rasterio.open(case.valid_mask) as source:
            valid = source.read(1) > 0
        row = {
            "case_id": case.case_id,
            "evaluation_role": "leave-one-tornado-out model-selection result",
            **binary_metrics(probability, info["label"], valid, THRESHOLD),
        }
        cv_rows.append(row)
        process_prediction(
            case,
            probability,
            THRESHOLD,
            info,
            OUTPUT / "leave_one_out",
            "leave-one-tornado-out model-selection result",
            MODEL_NAME,
        )
    cv_frame = pd.DataFrame(cv_rows)
    cv_frame.to_csv(OUTPUT / "reports" / "cross_validation_results.csv", index=False)

    final_rows = []
    for case in cases:
        probability = combine(
            DECODER / "cases" / case.case_id / "predicted_probability.tif",
            FOREST / "cases" / case.case_id / "predicted_probability.tif",
        )
        info = label_info(case.case_id)
        role = "training-reference comparison (not independent)" if info is not None else "unlabelled inference; independent validation required"
        final_rows.append(
            process_prediction(case, probability, THRESHOLD, info, OUTPUT, role, MODEL_NAME)
        )
    final_frame = pd.DataFrame(final_rows)
    final_frame.to_csv(OUTPUT / "reports" / "current_dataset_results.csv", index=False)

    decoder_cv = pd.read_csv(DECODER / "reports" / "cross_validation_results.csv")
    forest_cv = pd.read_csv(FOREST / "reports" / "cross_validation_results.csv")
    comparison = pd.DataFrame(
        [
            {
                "model": "Prithvi segmentation decoder",
                **decoder_cv[["dice", "iou", "precision", "recall"]].mean().to_dict(),
                "selected": False,
            },
            {
                "model": "Prithvi embeddings + Extra Trees",
                **forest_cv[["dice", "iou", "precision", "recall"]].mean().to_dict(),
                "selected": False,
            },
            {
                "model": MODEL_NAME,
                **cv_frame[["dice", "iou", "precision", "recall"]].mean().to_dict(),
                "selected": True,
            },
        ]
    )
    comparison.to_csv(OUTPUT / "reports" / "model_comparison.csv", index=False)

    model_dir = OUTPUT / "models" / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    copy_streamed(DECODER / "models" / "final_model" / "prithvi_segmentation_head.pt", model_dir / "prithvi_segmentation_head.pt")
    copy_streamed(
        FOREST / "models" / "final_model" / "prithvi_embedding_extra_trees.joblib",
        model_dir / "prithvi_embedding_extra_trees.joblib",
    )
    metadata = {
        "model": MODEL_NAME,
        "forest_weight": WEIGHT_FOREST,
        "decoder_weight": 1.0 - WEIGHT_FOREST,
        "threshold": THRESHOLD,
        "training_cases": ["TOR10", "TOR12", "TOR13", "TOR16"],
        "validation": "leave-one-tornado-out model selection; no independent final test set",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_status": "research prototype; not validated for unattended operational use",
        "known_limitations": [
            "Only four NWS-labelled cases are available.",
            "All four held-out centerlines fail the strict NWS path-agreement gate.",
            "Unlabelled outputs are candidates requiring manual or external validation.",
        ],
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    summary = {
        "cases": len(cases),
        "labelled_cases": 4,
        "selected_model": MODEL_NAME,
        "cross_validation_macro": cv_frame[["dice", "iou", "precision", "recall"]].mean().to_dict(),
        "published_final_centerlines": int(final_frame["published_centerline"].sum()),
        "output": str(OUTPUT),
    }
    (OUTPUT / "reports" / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
