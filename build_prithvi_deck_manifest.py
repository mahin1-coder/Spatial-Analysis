#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "outputs_prithvi_final"
EDA = PROJECT / "outputs_agentic" / "cases"


def existing(path: Path) -> str | None:
    return str(path.resolve()) if path.exists() else None


def main() -> None:
    presentation = OUTPUT / "presentation"
    presentation.mkdir(parents=True, exist_ok=True)
    result_rows = list(csv.DictReader((OUTPUT / "reports" / "current_dataset_results.csv").open()))
    cv_rows = list(csv.DictReader((OUTPUT / "reports" / "cross_validation_results.csv").open()))
    comparison_rows = list(csv.DictReader((OUTPUT / "reports" / "model_comparison.csv").open()))
    macro = {
        key: sum(float(row[key]) for row in cv_rows) / len(cv_rows)
        for key in ["dice", "iou", "precision", "recall"]
    }
    slides = []
    figure_specs = [
        ("before_after_full.png", "The aligned full scenes preserve the complete raster footprint"),
        ("band_difference_maps.png", "Signed band differences show where reflectance increased or decreased"),
        ("absolute_difference_maps.png", "Absolute differences isolate the magnitude of spectral change"),
        ("distribution_histograms.png", "Band distributions quantify scene-wide radiometric shifts"),
        ("channel_statistics.png", "Channel means and variances summarize BEFORE and AFTER behavior"),
        ("correlation_matrices.png", "Band correlations reveal broad scene changes and redundancy"),
        ("band_combinations.png", "Multispectral composites expose vegetation and moisture response"),
        ("ndvi_change_map.png", "NDVI change provides an independent vegetation-loss diagnostic"),
        ("kmeans_change_analysis.png", "K-Means supplies an unsupervised change candidate, not ground truth"),
    ]
    for row in result_rows:
        case_id = row["case_id"]
        status = row["status"]
        for filename, title in figure_specs:
            asset = existing(EDA / case_id / filename)
            if asset:
                slides.append({"case_id": case_id, "title": f"{case_id}: {title}", "status": status, "image": asset})
        for path, title in [
            (OUTPUT / "cases" / case_id / "prithvi_probability_map.png", "Prithvi probability covers the full valid scene"),
            (OUTPUT / "cases" / case_id / "final_path_map.png", "Final path decision is drawn directly on the AFTER image"),
        ]:
            asset = existing(path)
            if asset:
                slides.append({"case_id": case_id, "title": f"{case_id}: {title}", "status": status, "image": asset})
        loo_map = existing(OUTPUT / "leave_one_out" / "cases" / case_id / "final_path_map.png")
        loo_comparison = existing(OUTPUT / "leave_one_out" / "cases" / case_id / "prediction_vs_nws.png")
        if loo_map:
            slides.append({"case_id": case_id, "title": f"{case_id}: held-out prediction versus the official NWS path", "status": "Leave-one-tornado-out result", "image": loo_map})
        if loo_comparison:
            slides.append({"case_id": case_id, "title": f"{case_id}: held-out corridor comparison", "status": "Cyan = NWS reference | Red = imagery-only prediction", "image": loo_comparison})

    manifest = {
        "title": "Prithvi Tornado Damage-Path Analysis",
        "subtitle": "Fourteen BEFORE/AFTER Landsat cases | NWS-supervised validation on four cases | full-scene overlays",
        "output_pptx": str((presentation / "prithvi_tornado_path_analysis.pptx").resolve()),
        "preview_dir": str((presentation / "rendered").resolve()),
        "macro": macro,
        "held_out_passes": 0,
        "held_out_cases": len(cv_rows),
        "model_rows": comparison_rows,
        "slides": slides,
        "sources": [
            "NASA/IBM Prithvi-EO-2.0-tiny-TL pretrained Earth-observation encoder (Apache-2.0).",
            "Project-provided Landsat BEFORE/AFTER surface-reflectance rasters.",
            "Project-provided NWS Damage Assessment Toolkit paths for TOR10, TOR12, TOR13, and TOR16.",
            "Predictions come from imagery. NWS geometry is used for supervised labels and post-prediction evaluation, never to select or draw an inference path.",
        ],
    }
    manifest_path = presentation / "deck_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(manifest_path)


if __name__ == "__main__":
    main()
