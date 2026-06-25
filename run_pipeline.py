"""Command-line entry point for the tornado damage-path pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl_cache")

from src.config import build_config
from src.analysis import run_change_analysis
from src.baseline_model import train_random_forest_baseline
from src.custom_pair import analyze_custom_pair
from src.ingest import ingest_sources
from src.inventory import write_phase1_inventory
from src.path_atlas import create_tornado_path_atlas
from src.preprocessing import preprocess_registered_pairs
from src.predict import run_baseline_predictions
from src.report_generator import generate_all_eda_reports
from src.shapefile_tools import run_overlays
from src.utils import setup_logging
from src.visualization import create_prediction_showcase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tornado damage-path geospatial pipeline")
    parser.add_argument(
        "--mode",
        default="inventory",
        choices=[
            "inventory",
            "ingest",
            "preprocess",
            "analyze",
            "overlays",
            "labels",
            "baseline",
            "predict",
            "showcase",
            "path-atlas",
            "eda-report",
            "analyze-pair",
            "full",
        ],
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source file or directory to audit. May be supplied multiple times.",
    )
    parser.add_argument("--before", type=Path, help="Path to a BEFORE raster (--mode analyze-pair).")
    parser.add_argument("--after", type=Path, help="Path to an AFTER raster (--mode analyze-pair).")
    parser.add_argument("--name", help="Case name for outputs (--mode analyze-pair).")
    parser.add_argument(
        "--nws-shapefile",
        type=Path,
        default=None,
        help="Optional NWS damage path/polygon shapefile to overlay (--mode analyze-pair).",
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
    elif args.mode == "path-atlas":
        atlas_path = create_tornado_path_atlas(config)
        print(f"Tornado path atlas written to: {atlas_path}")
    elif args.mode == "eda-report":
        eda_results = generate_all_eda_reports(config)
        for row in eda_results:
            print(row)
    elif args.mode == "analyze-pair":
        if not args.before or not args.after or not args.name:
            raise SystemExit("--mode analyze-pair requires --before, --after, and --name.")
        result = analyze_custom_pair(config, args.before, args.after, args.name, args.nws_shapefile)
        print(f"Status: {result['status']}")
        print(f"Readable pixels: {result['valid_pixels']:,}")
        print(f"Predicted damage pixels: {result['predicted_damage_pixels']:,}")
        print(f"Showcase map: {result['showcase_path']}")


if __name__ == "__main__":
    main()
