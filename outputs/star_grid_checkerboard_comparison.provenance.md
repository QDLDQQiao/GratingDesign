# Provenance

Local artifacts:

- `src/modality_comparison.py`
- `scripts/compare_modalities.py`
- `tests/test_modality_comparison.py`
- `results/modality_comparison/modality_comparison.csv`
- `results/modality_comparison/modality_comparison_report.json`
- `results/modality_comparison/modality_comparison.png`
- `results/modality_comparison/modality_recovered_maps.png`
- `results/modality_comparison/modality_error_maps.png`

Commands:

```powershell
rtk python scripts\compare_modalities.py --photons 3000 10000 30000 --repeats 5 --out results\modality_comparison
rtk python -m pytest tests
```

External sources:

- https://arxiv.org/abs/2506.02427
- https://arxiv.org/abs/2404.14584
- https://arxiv.org/abs/2211.11757
- https://arxiv.org/abs/2112.13968

Notes:

- Two-shot checkerboard was simulated as equal total dose split across two
  exposures.
- Two-direction grid/checkerboard directional dark-field anisotropy and angle
  are reported as non-identifiable because the cosine directional dark-field
  fit has rank 2 for three unknowns.
