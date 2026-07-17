# Technical Audit

## Critical

- Pairing previously selected duplicate BEFORE/AFTER candidates silently. This can mix the wrong image pair and make the NWS path appear wrong. Fixed with explicit pairing statuses and `outputs/reports/pairing_report.csv`.
- NWS geometries were converted to EPSG:4326 while raster windows could be in another CRS. That can offset labels, metrics, and overlays. Fixed by transforming geometries into each raster CRS before intersection and rasterization.
- New-pair prediction previously assumed the two rasters were already aligned. Fixed by validating and aligning the pair before prediction.

## High

- The pipeline assumed equal dimensions implied geospatial alignment. Fixed with CRS, transform, resolution, bounds, band-count, readability, and overlap validation.
- Batch processing did not produce a formal HTML report. Added `batch_report.html` and `batch_summary.csv`.
- Prediction wrote only a binary mask. Added probability rasters and vector path products.

## Medium

- Feature generation was limited to raw before/after/difference bands. Added a reusable feature module with robust normalization, signed/absolute/relative differences, spectral magnitude, and optional local texture.
- Evaluation metrics were a placeholder. Added IoU, Dice, precision, recall, F1, false-negative rate, balanced accuracy, MCC, AP/ROC-AUC where valid, threshold analysis, and connected-component stats.
- Documentation did not clearly separate observed spectral change, predicted damage, and official ground truth. Added methodology and beginner documentation.

## Low

- `.gitignore` was minimal. It should continue excluding virtual environments, caches, local data, and generated bulky products.
- The U-Net file remains a later-stage hook. The beginner runner still defaults to the Random Forest baseline.

## Remaining Limitations

- The Random Forest baseline is not proof of a confirmed tornado path. It produces candidate damage corridors.
- If official NWS geometry does not overlap the image footprint, the correct result is “no overlapping official geometry,” not a forced overlay.
- Path centerlines are postprocessed candidates from predicted masks. Curved skeleton extraction can be improved in a later deep-learning phase.
- Vegetation indices are not calculated unless band identity is known from metadata.
