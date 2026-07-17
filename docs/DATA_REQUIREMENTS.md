# Data Requirements

Recommended input:

- GeoTIFF BEFORE/AFTER raster pairs.
- Clear filename roles: BEFORE/AFTER or PRE/POST.
- Valid CRS metadata.
- Similar geographic coverage between BEFORE and AFTER images.
- Matching or compatible band counts.
- NWS path or polygon shapefiles when official validation is needed.

The pipeline rejects or warns on:

- unreadable TIFFs;
- missing CRS;
- low geographic overlap;
- duplicate BEFORE or AFTER candidates;
- files that do not clearly identify before/after role.
