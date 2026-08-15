# Agentic Tornado Damage-Path Analysis

## Prithvi research workflow

The current experiment uses the official NASA/IBM Prithvi-EO-2.0-tiny-TL encoder with two Landsat dates. It compares a small segmentation decoder, a balanced Extra Trees classifier over frozen Prithvi embeddings, and a weighted ensemble.

```bash
env PYTHONPATH=. .venv/bin/python run_prithvi_workflow.py --clean
env PYTHONPATH=. .venv/bin/python run_prithvi_embedding_rf.py --clean
env PYTHONPATH=. .venv/bin/python build_prithvi_ensemble.py
```

The selected ensemble reached macro leave-one-tornado-out Dice `0.199` on the four cases with safe event-specific NWS paths. None of the four held-out centerlines passed the strict NWS path-agreement gate, so this remains a research triage system rather than a validated unattended detector. Outputs without NWS data are labelled **unverified imagery-only candidates**.

Analyze a future folder without retraining:

```bash
env PYTHONPATH=. .venv/bin/python run_prithvi_inference.py \
  --source "/path/to/new_folder" \
  --output "outputs_prithvi_future/new_batch" \
  --assume-landsat-order
```

Use `--assume-landsat-order` only for the same six-band professor data product, whose order is known to be `SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B7`. Without that explicit acknowledgement, unnamed bands are rejected. Mac users can double-click `RUN_PRITHVI_NEW_DATASET.command` and drag the new folder into Terminal.

This repository processes paired BEFORE and AFTER multispectral GeoTIFFs and produces a reviewable tornado-damage corridor, curved centerline, quality-control report, and PowerPoint deck.

The workflow uses a deterministic LangGraph state machine. Each agent has one job: inventory, geospatial alignment, EDA, K-Means candidate generation, model inference, path extraction, NWS validation, reporting, and presentation generation. The predicted red path comes from the imagery workflow. The cyan NWS path is drawn afterward for validation and is never copied into the prediction.

## Run the professor dataset

```bash
cd "/Users/m.mahin/Documents/New project 2/Spatial-Analysis"
.venv/bin/python run_agentic_workflow.py --clean
```

Open:

- `outputs_agentic/agentic_report.html`
- `outputs_agentic/presentation/agentic_tornado_path_analysis.pptx`
- `outputs_agentic/reports/agentic_case_summary.csv`

## Run a new folder

```bash
.venv/bin/python run_agentic_workflow.py \
  --source "/path/to/new/folder" \
  --output "outputs_agentic_new_batch" \
  --reuse-output ""
```

The folder must contain unambiguous BEFORE/AFTER pairs such as `TOR25_before.tif` and `TOR25_after.tif`. NWS shapefiles are optional at inference time.

On macOS, double-click `RUN_AGENTIC_WORKFLOW.command`, drag the new data folder into Terminal, and press Enter.

## Methods

- `rasterio`, `geopandas`, `shapely`, and `pyproj` for CRS-safe geospatial processing
- `numpy` and `scipy` for multispectral changes and local processing
- `scikit-learn` for K-Means and Random Forest probability inference
- `scikit-image` for morphology and skeleton concepts
- a small PyTorch U-Net blended with Random Forest when the saved checkpoint is available
- graph-based skeleton centerlines with a portable Tornado_Modis-inspired curve fallback
- `matplotlib` and artifact-tool for figures and PowerPoint output

See [docs/AGENTIC_WORKFLOW.md](docs/AGENTIC_WORKFLOW.md) for the full process and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

## Scientific limits

This is a research workflow, not an operational warning product. It rejects weak cases instead of inventing a path. NWS agreement is reported only when valid reference geometry exists, and results from cases used in training are explicitly marked as training-reference comparisons.
