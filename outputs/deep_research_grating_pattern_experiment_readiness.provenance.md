# Provenance: Grating Pattern Experiment-Readiness Review

Date: 2026-07-13

## Local Evidence

- `src/pattern_comparison.py`: generated and propagated S4, H3, H3-4L, DRH6, FATO proxy, CUC square/hex, RBN, MPG proxy, G2, C2, and C2 two-shot families.
- `src/experiment_readiness.py`: measured noise-free sinusoidal phase/T/D transfer separately from photon-noise accuracy.
- `results/pattern_simulation_research/pattern_simulation_report.json`: 12.4 keV-equivalent monochromatic, 80 mm, 10,000 photons/pixel, five-repeat visibility-calibrated reconstruction comparison.
- `results/experiment_readiness/pattern_experiment_readiness_report.json`: transfer and accuracy scorecard data.
- `rtk python -m pytest tests`: 10 tests passed. Windows emitted a non-fatal pytest temporary-directory cleanup permission warning after completion.

## External Sources

- Wen et al. (2010): https://pmc.ncbi.nlm.nih.gov/articles/PMC3091831/
- Zanette et al. (2010): https://doi.org/10.1103/PhysRevLett.105.248102
- Croughan et al. (2023): https://pubmed.ncbi.nlm.nih.gov/37155790/
- Croughan et al. (2024): https://pubmed.ncbi.nlm.nih.gov/39090344/
- Kagias et al. (2016): https://doi.org/10.1103/PhysRevLett.116.093902
- Kagias et al. (2017): https://doi.org/10.1088/1742-6596/849/1/012053
- Kim et al. (2020): https://doi.org/10.1063/1.5145361
- Wang et al. (2016): https://pmc.ncbi.nlm.nih.gov/articles/PMC4964655/
- Wang and Sawhney (2021): https://doi.org/10.1073/pnas.2022319118
- Meyer et al. (2024): https://doi.org/10.1038/s41598-024-78133-8

## Interpretation Limit

The literature establishes modality maturity and experimental constraints. It does not establish that the current S4, H3, H3-4L, DRH6, FATO proxy, or MPG proxy layouts are superior. Local results use ideal monochromatic scalar propagation and visibility-calibrated carrier retrieval, not a complete experimental forward model.
