#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from agentic.workflow import PROJECT, run_agentic_workflow


def absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangGraph tornado damage-path workflow.")
    parser.add_argument("--source", default="data/raw/rasters", help="Folder containing BEFORE and AFTER GeoTIFF files.")
    parser.add_argument("--shapefiles", default="data/raw/shapefiles", help="Optional NWS shapefile root.")
    parser.add_argument("--output", default="outputs_agentic", help="Output folder.")
    parser.add_argument("--reuse-output", default="outputs_hybrid_path", help="Existing aligned rasters and hybrid probabilities to reuse.")
    parser.add_argument("--model-dir", default="outputs_hybrid_path/models/final_model", help="Saved hybrid model folder for new inference.")
    parser.add_argument("--no-slides", action="store_true", help="Create deck manifest but skip PPTX export.")
    parser.add_argument("--clean", action="store_true", help="Remove the selected output folder before running.")
    parser.add_argument("--hydrography", action="append", default=[], help="Cached georeferenced USGS/3DHP/NHD water vector file or directory; repeatable.")
    parser.add_argument("--roads", action="append", default=[], help="Cached georeferenced road vector file or directory; repeatable.")
    parser.add_argument("--railways", action="append", default=[], help="Cached georeferenced railway vector file or directory; repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    source = absolute_path(args.source)
    shapefiles = absolute_path(args.shapefiles)
    output = absolute_path(args.output)
    reuse_output = absolute_path(args.reuse_output) if args.reuse_output else None
    model_dir = absolute_path(args.model_dir)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    state = run_agentic_workflow(
        {
            "source": str(source),
            "shapefiles": str(shapefiles),
            "output": str(output),
            "reuse_output": str(reuse_output) if reuse_output else "",
            "model_dir": str(model_dir),
            "build_slides": not args.no_slides,
            "errors": [],
            "warnings": [],
            "stage_log": [],
            "context_layers": {
                "hydrography_paths": [str(absolute_path(value)) for value in args.hydrography],
                "road_paths": [str(absolute_path(value)) for value in args.roads],
                "railway_paths": [str(absolute_path(value)) for value in args.railways],
                "hydrography_buffer_pixels": 2.0,
                "road_buffer_pixels": 1.5,
                "railway_buffer_pixels": 1.5,
            },
        }
    )
    print(
        json.dumps(
            {
                "cases": len(state.get("cases", [])),
                "errors": len(state.get("errors", [])),
                "report": str(output / "agentic_report.html"),
                "presentation": state.get("presentation", ""),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
