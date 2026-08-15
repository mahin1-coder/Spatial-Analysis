# Third-Party Method Attribution

## Tornado_Modis

The agentic workflow contains a portable, modernized curve-fitting candidate inspired by:

- Repository: https://github.com/DiLiu2023/Tornado_Modis
- Commit inspected: `7151aec9153dd9a655e3ac480f3f180e04861b43`
- Copyright: 2025 DiLiu2023
- License: MIT

The original notebook uses ArcPy, hard-coded Windows paths, affected-pixel filtering, and competing linear/nonlinear curve fits. This project does not copy the ArcPy or Windows-specific implementation. It adapts the scientific idea by fitting linear and quadratic curves in a PCA-rotated coordinate system using NumPy and selecting the lower-BIC candidate.

The fitted curve is a diagnostic or fallback candidate. The primary path remains a connected skeleton extracted from an imagery-derived damage corridor.

## Prithvi-EO-2.0

The Prithvi workflow uses the official NASA/IBM Prithvi-EO-2.0 tiny encoder and
transfer-learning checkpoint:

- Model: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL
- Source: https://github.com/NASA-IMPACT/Prithvi-EO-2.0
- License: Apache License 2.0

The upstream encoder is kept frozen. This project trains its own segmentation
head and Extra Trees classifier using the available event-specific NWS labels.
NWS geometries are used for training and validation only; they are never copied
into prediction outputs.
