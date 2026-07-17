# Troubleshooting

## NWS path does not show on the raster

Check whether the shapefile overlaps the raster footprint. If it does not overlap, the correct output is no official geometry for that scene.

## Pair was skipped

Open:

```text
outputs/reports/pairing_report.csv
```

If status says `AMBIGUOUS_BEFORE` or `AMBIGUOUS_AFTER`, remove duplicate copies or rename files clearly.

## Raster validation failed

Open:

```text
outputs/reports/raster_validation_report.csv
```

Common causes are unreadable TIFFs, missing CRS, different band counts, or no geographic overlap.

## Results look too empty

That usually means the image has little readable overlap, the model found no confident damage signal, or the NWS geometry is outside the raster footprint.
