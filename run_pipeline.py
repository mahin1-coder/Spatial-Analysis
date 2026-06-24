"""Command-line entry point for the tornado damage-path pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl_cache")

from src.config import build_config
from src.analysis import run_change_analysis
from src.baseline_model import train_random_forest_baseline
from src.ingest import ingest_sources
from src.inventory import write_phase1_inventory
from src.preprocessing import preprocess_registered_pairs
from src.predict import run_baseline_predictions
from src.shapefile_tools import run_overlays
from src.utils import setup_logging
from src.visualization import create_prediction_showcase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tornado damage-path geospatial pipeline")
    parser.add_argument(
        "--mode",
        default="inventory",
        choices=["inventory", "ingest", "preprocess", "analyze", "overlays", "labels", "baseline", "predict", "showcase", "full"],
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source file or directory to audit. May be supplied multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = build_config(project_root, args.source)

    if args.mode == "inventory":
        results = write_phase1_inventory(config)
        print("Phase 1 inventory complete.")
        print(f"Files inventoried: {len(results['file_inventory'])}")
        print(f"Rasters inspected: {len(results['raster_inventory'])}")
        print(f"Shapefiles inspected: {len(results['shapefile_inventory'])}")
        print(f"Pair rows: {len(results['pair_inventory'])}")
        print(f"Outputs written to: {config.inventory_dir}")
    elif args.mode == "analyze":
        stats = run_change_analysis(config)
        print(f"Analysis rows written: {len(stats)}")
    elif args.mode == "ingest":
        manifest = ingest_sources(config)
        print(f"Ingested/registered rows: {len(manifest)}")
    elif args.mode == "preprocess":
        preprocessing = preprocess_registered_pairs(config)
        print(f"Preprocessing rows written: {len(preprocessing)}")
    elif args.mode == "overlays":
        overlays = run_overlays(config)
        print(f"Overlay rows written: {len(overlays)}")
    elif args.mode == "full":
        manifest = ingest_sources(config)
        preprocessing = preprocess_registered_pairs(config)
        stats = run_change_analysis(config)
        overlays = run_overlays(config)
        print("Phase 2-4 pipeline complete.")
        print(f"Ingested/registered rows: {len(manifest)}")
        print(f"Preprocessing rows: {len(preprocessing)}")
        print(f"Analysis rows: {len(stats)}")
        print(f"Overlay rows: {len(overlays)}")
    elif args.mode == "labels":
        raise NotImplementedError("Label creation belongs to a later phase and was not requested.")
    elif args.mode == "baseline":
        report = train_random_forest_baseline(config)
        print(report.to_string(index=False))
    elif args.mode == "predict":
        predictions = run_baseline_predictions(config)
        print(predictions.to_string(index=False))
    elif args.mode == "showcase":
        showcase = create_prediction_showcase(config)
        print(showcase.to_string(index=False))


if __name__ == "__main__":
    main()
