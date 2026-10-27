# Provenance

## Local evidence

- Reconstruction input: `results/modality_comparison/modality_comparison_report.json`.
- Reconstruction table: `results/modality_comparison/modality_comparison.csv`.
- Reconstruction figure: `results/grating_advantage_analysis/grating_advantage_summary.png`.
- Coverage input: `results/coverage_comparison/coverage_comparison_report.json`.
- Coverage table: `results/coverage_comparison/coverage_comparison.csv`.
- Coverage figure: `results/grating_advantage_analysis/grating_advantage_coverage.png`.
- Derived report: `results/grating_advantage_analysis/grating_advantage_analysis_report.json`.
- Analysis command: `rtk python scripts/analyze_grating_advantages.py --out results/grating_advantage_analysis`.

## Benchmark conditions

- Reconstruction: periodic smooth phase, absorption, and anisotropic D phantoms;
  Poisson noise; five repeats at 300, 1000, 3000, and 10000 photons/pixel.
- Final reconstruction settings: exact FFT carrier window, 5 px phase/T
  smoothing, 16 px D smoothing.
- Coverage: monochromatic propagation over 10-16 keV and 0.08-0.40 m;
  thresholded by minimum visibility, balance CV, and contamination.

## External sources

- Zanette et al., multidirectional 2-D grating dark-field imaging:
  https://portal.fis.tum.de/en/publications/multidirectional-x-ray-dark-field-imaging-with-two-dimensional-gr/
- Malecki et al., scalar and vectorial X-ray dark-field reconstruction:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4156746/
- Wieczorek et al., anisotropic X-ray dark-field tomography:
  https://researchmgt.monash.edu/ws/portalfiles/portal/246556284/246549085_oa.pdf
- Coherent superposition in directional dark-field imaging:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3634061/
- Dark-field chest X-ray imaging and visibility/SNR:
  https://www.nature.com/articles/s43856-022-00215-3
- Spectral X-ray dark-field visibility weighting:
  https://www.nature.com/articles/s41598-020-70011-3

The performance claims are inferences from the local synthetic benchmark. The
external sources support the interpretation that directional dark-field encodes
microstructure orientation, that multiple directions enable vector/tensor
recovery, and that visibility affects dark-field noise; they do not validate the
specific numerical values for this implementation.
