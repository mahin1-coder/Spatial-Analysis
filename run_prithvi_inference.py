#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import torch

from prithvi_pipeline.data import discover_cases, validate_prithvi_bands
from prithvi_pipeline.embedding_rf import infer_embedding_forest
from prithvi_pipeline.model import PrithviSegmentationHead, load_prithvi_encoder
from prithvi_pipeline.pipeline import infer_probability, process_prediction


PROJECT = Path(__file__).resolve().parent


def absolute(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained Prithvi ensemble on a new BEFORE/AFTER folder.")
    parser.add_argument("--source", required=True, help="Folder containing paired BEFORE and AFTER GeoTIFFs.")
    parser.add_argument("--output", required=True, help="New output folder.")
    parser.add_argument("--shapefiles", default="data/raw/shapefiles", help="Optional NWS shapefile root for comparison only.")
    parser.add_argument("--vendor-dir", default="third_party/prithvi_eo2_tiny")
    parser.add_argument("--decoder", default="outputs_prithvi/models/final_model/prithvi_segmentation_head.pt")
    parser.add_argument("--forest", default="outputs_prithvi_rf/models/final_model/prithvi_embedding_extra_trees.joblib")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument(
        "--assume-landsat-order",
        action="store_true",
        help="Explicitly treat unnamed six-band inputs as SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B7.",
    )
    args = parser.parse_args()

    output = absolute(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    print("[1/4] Discovering and geospatially aligning BEFORE/AFTER cases...", flush=True)
    cases = discover_cases(absolute(args.source), output / "cases", output, absolute(args.shapefiles))
    if not cases:
        raise SystemExit("No unambiguous BEFORE/AFTER GeoTIFF pairs were found.")
    print(f"      Found {len(cases)} case(s): {', '.join(case.case_id for case in cases)}", flush=True)
    print("[2/4] Loading the official Prithvi encoder and trained heads...", flush=True)
    encoder, _ = load_prithvi_encoder(absolute(args.vendor_dir), num_frames=2)
    decoder = PrithviSegmentationHead()
    decoder.load_state_dict(torch.load(absolute(args.decoder), map_location="cpu", weights_only=True))
    decoder.eval()
    forest = joblib.load(absolute(args.forest))
    device = torch.device("cpu")
    rows = []
    print("[3/4] Running imagery-only inference...", flush=True)
    for index, case in enumerate(cases, start=1):
        print(f"      [{index}/{len(cases)}] {case.case_id}", flush=True)
        descriptions = case.geospatial_metadata.get("band_descriptions")
        band_mapping = "GeoTIFF band descriptions"
        if args.assume_landsat_order and not any(descriptions or []):
            descriptions = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
            band_mapping = "explicit --assume-landsat-order acknowledgement"
        band_validation = validate_prithvi_bands(case.before, descriptions)
        if not band_validation["usable"]:
            rows.append(
                {
                    "case_id": case.case_id,
                    "status": "Rejected: incompatible or ambiguous band order",
                    "band_validation": band_validation,
                }
            )
            continue
        decoder_probability = infer_probability(encoder, decoder, case, device, args.batch_size, args.stride)
        forest_probability = infer_embedding_forest(encoder, forest, case, device, args.batch_size, args.stride)
        probability = (0.1 * decoder_probability + 0.9 * forest_probability).astype("float32")
        result = process_prediction(
            case,
            probability,
            0.55,
            None,
            output,
            "future unlabelled inference; independent validation required",
            "Prithvi-EO-2.0 ensemble: 90% embedding Extra Trees + 10% segmentation decoder",
        )
        result["band_mapping"] = band_mapping
        rows.append(result)
    report = output / "batch_summary.json"
    report.write_text(json.dumps(rows, indent=2))
    print("[4/4] Finished writing maps and the batch report.", flush=True)
    print(json.dumps({"cases": len(cases), "results": str(report)}, indent=2))


if __name__ == "__main__":
    main()
