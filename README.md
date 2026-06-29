# Tornado Damage Path Detection

This project detects likely tornado damage paths from paired **before** and **after** satellite imagery.

The goal is practical: when a new tornado dataset arrives, the workflow should not require someone to manually trace the path by hand. Drop in the imagery, run the pipeline, and get a slide-ready map showing the predicted damage corridor.

## What It Does

- Finds before/after image pairs in a folder
- Aligns and compares the raster data where the imagery is readable
- Runs a trained baseline model on the paired imagery
- Produces a damage mask for each tornado case
- Renders clear map outputs with a red predicted path overlay
- Supports official NWS shapefile overlays when they are available
- Builds a contact sheet for batches, so 50 pairs produce 50 reviewable maps

Typical use:

```text
2 images   -> 1 tornado path map
100 images -> about 50 tornado path maps
```

## Current Model

The current working model is a Random Forest baseline trained on readable before/after raster windows from the existing tornado dataset.

Model file:

```text
outputs/models/random_forest_baseline/random_forest_damage_baseline.joblib
```

This is a working baseline, not the final research-grade model. It is good enough to prove the automated workflow and generate visual outputs. The next major upgrade should be a segmentation model such as U-Net, trained on clean complete GeoTIFFs and stronger labels.

## Quick Start

On macOS, the easiest way is to double-click:

```text
RUN_NEW_DATASET.command
```

Then drag the professor's dataset folder into the Terminal window and press Enter.

The script will:

1. Set up Python packages if needed
2. Detect before/after pairs
3. Run the trained model
4. Create Gloria-style map figures
5. Open the output folder

Open this first:

```text
batch_contact_sheet.png
```

Each case also gets its own:

```text
showcase_prediction_map.png
```

## Folder Naming

For automatic pairing, filenames should clearly say which image is before and which is after.

Good examples:

```text
case01_before.tif
case01_after.tif

TOR_2024_01_before.tif
TOR_2024_01_after.tif

new_area_pre_event.tif
new_area_post_event.tif
```

If the folder contains 100 files, the pipeline looks for matching before/after names and processes each pair.

## Command Line Usage

From the repo folder:

```bash
cd "/Users/m.mahin/Documents/New project 2/Spatial-Analysis"
```

Analyze a folder of new images:

```bash
.venv/bin/python run_pipeline.py --mode analyze-folder \
  --source "/Users/m.mahin/Desktop/new_tornado_dataset" \
  --batch-name "professor_batch"
```

Analyze one pair manually:

```bash
.venv/bin/python run_pipeline.py --mode analyze-pair \
  --before "/path/to/case01_before.tif" \
  --after "/path/to/case01_after.tif" \
  --name "case01"
```

Analyze one pair with an official NWS shapefile:

```bash
.venv/bin/python run_pipeline.py --mode analyze-pair \
  --before "/path/to/case01_before.tif" \
  --after "/path/to/case01_after.tif" \
  --name "case01" \
  --nws-shapefile "/path/to/nws_dat_damage_paths.shp"
```

## Output Colors

- Red: model predicted tornado damage/path
- Cyan or blue: official NWS path/polygon, if supplied
- Gray or white: unreadable, missing, or corrupted image area

The pipeline does not fake results for broken imagery. If a tile cannot be read, it is marked instead of silently guessed.

## Main Output Locations

Batch results:

```text
outputs/predictions/batch/<batch_name>/
```

Single-pair results:

```text
outputs/predictions/custom/<case_name>/
```

Research dataset results:

```text
outputs/predictions/random_forest_baseline/
```

## Research Pipeline Modes

These commands are useful when working with the full historical dataset:

```bash
.venv/bin/python run_pipeline.py --mode inventory --source "/path/to/dataset"
.venv/bin/python run_pipeline.py --mode full --source "/path/to/dataset"
.venv/bin/python run_pipeline.py --mode baseline
.venv/bin/python run_pipeline.py --mode predict
.venv/bin/python run_pipeline.py --mode showcase
.venv/bin/python run_pipeline.py --mode path-atlas
```

## Data Quality Notes

The original sample data contained several truncated or unreadable TIFF tiles. That limits how much any model can learn or predict from those files.

For stronger results, future datasets should include:

- clean GeoTIFF before/after imagery
- matching resolution and projection
- minimal cloud cover
- official NWS path or polygon shapefiles when available
- enough labeled examples across EF0 to EF4 damage

## Engineering Roadmap

The current repo is set up to make the workflow usable now. The recommended next steps are:

1. Collect clean before/after GeoTIFFs for more tornado cases
2. Build stronger labels from NWS damage paths and manual QA
3. Train a U-Net or similar segmentation model
4. Add confidence maps and path centerline extraction
5. Package the workflow as a small web app for drag-and-drop use

The important part is already in place: the project can take new paired imagery and automatically produce tornado path maps without hand tracing.
