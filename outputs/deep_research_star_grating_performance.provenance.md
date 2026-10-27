# Provenance

- Local benchmark: `results/modality_adaptive/modality_comparison.csv`, generated
  with `scripts/compare_modalities.py --photons 300 1000 3000 10000 --repeats 5
  --smoothing-sigma-px 5 --window-sigma-bins 0`.
- Final optimized benchmark: `results/modality_comparison/modality_comparison.csv`
  and `modality_directional_dark_field.png`, generated with 5 px phase/T
  smoothing, 16 px D smoothing, and an exact carrier window.
- Local method ablation: `results/modality_optimized/modality_comparison.csv`
  and the window sweep from `scripts/optimize_reconstruction_method.py`.
- Directional-D parameter sweep: `results/reconstruction_method_optimization_d/`
  selected 5 px phase/T smoothing, 24 px D smoothing, and an exact carrier
  window at 3000 photons/pixel.
- External source: Zanette et al., *Multidirectional X-ray dark-field imaging
  with two-dimensional gratings*,
  https://portal.fis.tum.de/en/publications/multidirectional-x-ray-dark-field-imaging-with-two-dimensional-gr/
- External source: Wu et al., *Predicting visibility of interference fringes in
  X-ray grating interferometry*, https://pmc.ncbi.nlm.nih.gov/articles/PMC5234509/
- External source: Shashev et al., *Improving the visibility of phase gratings
  for Talbot-Lau X-ray imaging*,
  https://www.degruyterbrill.com/document/doi/10.3139/120.110948/html?lang=en

The numerical conclusions are inferences from the local synthetic benchmark;
the cited papers support the physical interpretation of multidirectional
dark-field, visibility, and the three contrast channels.
