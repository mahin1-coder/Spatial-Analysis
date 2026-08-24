#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from prithvi_pipeline.embedding_rf import run_embedding_forest_pipeline


PROJECT = Path(__file__).resolve().parent


def absolute(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prithvi embeddings with a balanced tree classifier.")
    parser.add_argument("--source", default="data/raw/rasters")
    parser.add_argument("--shapefiles", default="data/raw/shapefiles")
    parser.add_argument("--aligned-root", default="outputs_hybrid_path/cases")
    parser.add_argument("--output", default="outputs_prithvi_rf")
    parser.add_argument("--vendor-dir", default="third_party/prithvi_eo2_tiny")
    parser.add_argument("--patches-per-class", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    output = absolute(args.output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    summary = run_embedding_forest_pipeline(
        absolute(args.source),
        absolute(args.shapefiles),
        absolute(args.aligned_root),
        output,
        absolute(args.vendor_dir),
        args.patches_per_class,
        args.batch_size,
        args.stride,
        args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
