# Tornado Damage-Path Geospatial ML Pipeline

This project is a reproducible, automation-first geospatial pipeline for tornado damage-path analysis and eventual damage-path prediction from before/after satellite imagery.

## Objective

The long-term goal is to detect tornado damage paths with high spatial reliability using:

- BEFORE satellite GeoTIFF rasters
- AFTER satellite GeoTIFF rasters
- NWS tornado damage path shapefiles
- NWS tornado damage polygon shapefiles
- Existing exploratory notebooks, reports, and slides

Accuracy will not be judged by ordinary pixel accuracy alone because tornado damage corridors occupy a small fraction of the raster footprint. Future model evaluation should prioritize IoU, Dice/F1, precision, recall, false-negative rate, path-overlap accuracy, distance-to-path error, path-width error, and path-length error.

## Dataset Structure

The project is organized for weekly or biweekly dataset drops:

```text
data/
  incoming/      # New files from the professor before registration
  raw/           # Original registered source data, preserved unchanged
  processed/     # Cleaned or derived data products
notebooks/
  exploratory/   # Existing research notebooks copied or referenced here
outputs/
  inventory/     # Phase 1 file/raster/shapefile inventories
  preprocessed/  # Aligned rasters from later phases
  plots/         # Before/after/difference plots from later phases
  overlays/      # NWS geometry overlays from later phases
  masks/         # Rasterized ground-truth labels from later phases
  models/        # Trained models from later phases
  predictions/   # Predicted masks/probabilities from later phases
  reports/       # Statistical and evaluation reports
```

TIFF filenames are expected to contain tornado IDs such as `TOR5` or `TOR62` and a BEFORE/AFTER marker. The Phase 1 inventory automatically extracts these IDs and pairs candidate BEFORE/AFTER rasters by tornado ID.

## Ground Truth

NWS damage path and damage polygon shapefiles are treated as the official ground-truth reference. Later phases will reproject those geometries to each raster CRS, validate spatial overlap, rasterize paths/polygons onto the satellite grid, and use the resulting masks for model training and evaluation.

## Automation Workflow

Phase 1, implemented now, audits existing data and writes:

- `outputs/inventory/file_inventory.csv`
- `outputs/inventory/raster_inventory.csv`
- `outputs/inventory/shapefile_inventory.csv`
- `outputs/inventory/raster_pair_inventory.csv`

Later phases will add ingestion, raster alignment, before/after change analysis, shapefile overlays, label creation, baseline models, deep learning segmentation, and rigorous spatial evaluation.

## How To Run Phase 1

Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the inventory audit. Supply one or more source folders or files:

```bash
.venv/bin/python run_pipeline.py --mode inventory \
  --source /path/to/tornado_codex_package \
  --source /path/to/Tornado_Lab \
  --source /path/to/OneDrive_1_1-6-2026
```

The command preserves raw data and writes CSV outputs under `outputs/inventory/`.

## Planned ML Workflow

1. Register new weekly or biweekly data drops into `data/raw/`.
2. Pair BEFORE and AFTER rasters by tornado ID.
3. Validate CRS, transform, resolution, bounds, shape, and band compatibility.
4. Align rasters geospatially when required.
5. Compute per-channel before/after difference products and change statistics.
6. Overlay NWS paths/polygons and create rasterized ground-truth masks.
7. Train threshold and classical ML baselines before deep learning.
8. Train a U-Net style segmentation model using BEFORE, AFTER, and DIFFERENCE channels.
9. Evaluate with spatial metrics that penalize missed tornado paths.

## Current Status

Only Phase 1 is implemented. Model training, prediction, labels, overlays, and statistical change-analysis modes are intentionally deferred until the inventories confirm the data are complete and spatially usable.
