#!/usr/bin/env python3
"""Run the final U-Net on every case without exposing labels to inference."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from train_unet_path_model import (
    BASE,
    CASE_ROOT,
    LABEL_ROOT,
    OUTPUT,
    SmallUNet,
    discover_pairs,
    predict,
    prepare_case,
)


def main() -> None:
    checkpoint_override = os.getenv("UNET_CHECKPOINT", "").strip()
    checkpoint_path = (
        Path(checkpoint_override).expanduser().resolve()
        if checkpoint_override
        else OUTPUT / "models" / "final_unet" / "model.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SmallUNet(int(checkpoint["in_channels"]), int(checkpoint.get("base", BASE)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    pairs = discover_pairs()
    case_ids = sorted(
        set(pairs) & {path.name for path in CASE_ROOT.iterdir() if path.is_dir()},
        key=lambda value: int(value[3:]),
    )
    requested = {value.strip().upper() for value in os.getenv("UNET_CASES", "").split(",") if value.strip()}
    if requested:
        case_ids = [case_id for case_id in case_ids if case_id in requested]
    rows = []
    for case_id in case_ids:
        last_error = None
        for attempt in range(3):
            try:
                case = prepare_case(case_id, pairs[case_id])
                break
            except (OSError, TimeoutError) as error:
                last_error = error
                time.sleep(2 * (attempt + 1))
        else:
            rows.append({"case_id": case_id, "in_training_set": "", "probability_mean": "", "probability_max": "", "status": f"failed: {last_error}"})
            print(f"failed {case_id}: {last_error}", flush=True)
            continue
        probability = predict(model, case)
        case_dir = OUTPUT / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "deployment_probability.npz",
            probability=probability,
            valid=case["valid"],
            water=case["water"],
        )
        rows.append(
            {
                "case_id": case_id,
                "in_training_set": (LABEL_ROOT / case_id / "manual_damage_corridor_mask.tif").exists(),
                "probability_mean": float(probability[case["valid"].astype(bool)].mean()),
                "probability_max": float(probability.max()),
                "status": "success",
            }
        )
        print(f"predicted {case_id}", flush=True)

    report = OUTPUT / "reports" / "deployment_inference.csv"
    if requested and report.exists():
        with report.open(newline="") as handle:
            previous = list(csv.DictReader(handle))
        by_case = {row["case_id"]: row for row in previous}
        by_case.update({row["case_id"]: row for row in rows})
        rows = sorted(by_case.values(), key=lambda row: int(row["case_id"][3:]))
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "prediction_source": "BEFORE/AFTER imagery and derived raster features",
        "labels_available_to_inference": False,
        "training_set_cases_are_not_independent_test_results": True,
        "case_count": len(rows),
    }
    (OUTPUT / "reports" / "deployment_inference_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
