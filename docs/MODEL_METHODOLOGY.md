# Model Methodology

The workflow separates three different things:

- Observed spectral change: pixel differences between aligned BEFORE and AFTER rasters.
- Predicted damage: model output from the Random Forest baseline.
- Official ground truth: NWS path or polygon shapefiles when supplied and geographically overlapping.

The pipeline first pairs files, validates raster metadata, aligns AFTER imagery to the BEFORE grid, preserves NoData, and then predicts only on readable overlapping pixels.

The current default model is a Random Forest baseline. It is intentionally conservative and produces candidate damage corridors, probability rasters, binary masks, cleaned polygons, and candidate centerlines. It should not be presented as a perfect tornado-path detector.
