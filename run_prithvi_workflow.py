#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from prithvi_pipeline.pipeline import run_prithvi_pipeline


PROJECT = Path(__file__).resolve().parent


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Prithvi-EO-2.0 tornado damage-path segmentation workflow."
    )
    parser.add_argument("--source", default="data/raw/rasters", help="Folder containing paired BEFORE/AFTER GeoTIFFs.")
    parser.add_argument("--shapefiles", default="data/raw/shapefiles", help="Root containing event-specific NWS shapefiles.")
    parser.add_argument("--aligned-root", default="outputs_hybrid_path/cases", help="Existing validated aligned-raster root.")
    parser.add_argument("--output", default="outputs_prithvi", help="Prithvi output folder.")
    parser.add_argument("--vendor-dir", default="third_party/prithvi_eo2_tiny", help="Official Prithvi checkpoint/code folder.")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patches-per-class", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove the selected output folder before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = project_path(args.output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    summary = run_prithvi_pipeline(
        source=project_path(args.source),
        shapefiles=project_path(args.shapefiles),
        aligned_root=project_path(args.aligned_root),
        output=output,
        vendor_dir=project_path(args.vendor_dir),
        epochs=args.epochs,
        patches_per_class=args.patches_per_class,
        batch_size=args.batch_size,
        stride=args.stride,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
