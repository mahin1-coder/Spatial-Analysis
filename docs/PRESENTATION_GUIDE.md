# Presentation Guide

Use careful language:

- Say "candidate tornado-damage corridor," not "confirmed tornado path."
- Say "predicted damage mask," not "ground truth."
- Say "official NWS path/polygon" only when a shapefile is supplied and overlaps the raster.

Explain the workflow:

1. Pair BEFORE and AFTER rasters.
2. Validate CRS, transform, bounds, bands, NoData, readability, and overlap.
3. Align AFTER to the BEFORE raster grid.
4. Compute change features.
5. Run the Random Forest baseline.
6. Generate probability map, binary mask, cleaned polygon, and candidate centerline.
7. Overlay official NWS geometry separately for comparison.
