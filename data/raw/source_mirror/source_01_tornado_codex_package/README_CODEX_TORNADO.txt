Tornado Research Codex Package

Contents:
- Research PDF: Datasets Analyzing Tornado Research 02_12.pdf
- Tornado path image: Tornado_path.jpg
- NWS damage path shapefile set: nws_dat_damage_paths.*
- NWS damage polygon shapefile set: nws_dat_damage_polys.*
- Jupyter notebooks: data_studying_week0301.ipynb, data_studying_week030126.ipynb, langchain_agent.ipynb, data_analyzing_tor5.ipynb, data_analyzing_week032326.ipynb
- Before/after GeoTIFF raster pairs for TOR5, TOR7, TOR10, TOR11, TOR12, TOR13, TOR15, TOR16, TOR18, TOR24, TOR61, TOR62, TOR66, TOR68

Suggested Codex instruction:
"Open this tornado research package. First inspect the notebooks and the PDF to understand the current workflow. Then create a clean Python analysis pipeline that loads each before/after TIFF pair, computes per-channel summary stats, before-after difference rasters, distributions, correlations, and saves plots/results. Also load the NWS path and polygon shapefiles and explain how they can be used to overlay tornado damage paths on the raster outputs. Keep code beginner-friendly and document each step."

Notes:
- Keep shapefile sidecar files together (.shp, .shx, .dbf, .prj, .cpg, .shp.xml).
- The TIFF files appear to contain 6 channels each.
