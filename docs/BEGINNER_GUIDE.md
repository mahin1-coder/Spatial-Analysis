# Beginner Guide

## Easiest Workflow

1. Double-click `RUN_NEW_DATASET.command`.
2. Drag the professor's dataset folder into Terminal.
3. Press Enter.
4. Wait for the script to finish.
5. Open `batch_report.html` or `batch_contact_sheet.png`.

## Expected Input

Use one BEFORE and one AFTER image per tornado case:

```text
TOR7_BEST_BEFORE_win25d.tif
TOR7_BEST_AFTER_win45d.tif
```

The code also accepts names like `TOR1_pre.tif`, `TOR1_post.tif`, `before_TOR1.tif`, and `after_TOR1.tif`.

## Main Outputs

Batch outputs are saved here:

```text
outputs/predictions/batch/<batch_name>/
```

Open these first:

```text
batch_report.html
batch_contact_sheet.png
```

Each case folder contains a map, mask, probability raster, and GeoJSON vector outputs.
