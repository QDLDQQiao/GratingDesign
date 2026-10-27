# Autoresearch Log

Objective: turn the validated propagation kernel into a usable grating pattern
design and optimization program with visual outputs.

Benchmark metric: maximize a bounded score combining mean target-carrier
visibility, minimum carrier visibility, balance across the four carriers, and
local Fourier-window contamination.

Decision: use a repeatable bounded random search over four carrier amplitudes
and phases. This is not the final fabrication optimizer, but it creates a useful
baseline that is fast, auditable, and directly tied to detector-plane evidence.

Artifacts:

- `src/grating_design.py`
- `src/visualize.py`
- `scripts/run_design.py`
- `config/default.json`
- `results/design_run/design_report.json`
- `results/design_run/design_summary.png`
- `results/design_run/optimization_history.png`

## Reconstruction extension

The directional-dark-field objective was extended with a separate D-only
Gaussian smoothing parameter. A bounded sweep over phase/T smoothing, D
smoothing, and carrier-window width selected exact carrier extraction, 5 px
phase/T smoothing, and 24 px D smoothing at 3000 photons/pixel. For the final
300-10000 photon comparison, 16 px D smoothing was selected as the balanced
setting. The final figure is
`results/modality_comparison/modality_directional_dark_field.png`.

## H3 And H3-4L Joint Optimization

Objective: optimize the three-carrier H3 and four-level H3 masks fairly against
S4 using propagated physical carrier visibility, then matched phase, T, and
directional-dark-field reconstruction at 10,000 photons/pixel and five noise
repeats. The search used 160 amplitude/phase candidates per layout, shortlisted
16 by physical visibility/balance/contamination, and ranked those using the
same 5 px phase/T and 16 px D smoothing as S4.

The four-level pattern now uses phase levels `0, pi/2, pi, 3pi/2`, rather than
compressing four levels into a pi-depth span. H3 improved from `Vmin=0.041` to
`0.405`, phase RMSE `0.00553 rad`, D0 RMSE `6.55e-4`, anisotropy RMSE `0.0263`,
and D-angle MAE `1.82 deg`. H3-4L reached `Vmin=0.376`, phase RMSE `0.00556
rad`, D0 RMSE `7.12e-4`, anisotropy RMSE `0.0277`, and D-angle MAE `2.11 deg`.
The matched S4 values are `Vmin=0.194`, phase RMSE `0.00737 rad`, D0 RMSE
`9.04e-4`, and D-angle MAE `3.81 deg`.

Artifacts:

- `scripts/optimize_h3_patterns.py`
- `scripts/compare_optimized_h3.py`
- `results/h3_optimization/h3_optimization_report.json`
- `results/h3_optimization/h3_optimization_summary.png`
- `results/h3_optimized_comparison/optimized_h3_comparison.png`

Decision: the optimized H3 is now the leading simulation candidate for a
matched S4/H3/CUC-hex coupon comparison. This is still a visibility-calibrated
screen, not an end-to-end fabrication or experimental-accuracy claim.

## Export Performance Optimization

Hypothesis: the exporter is slow because S4/H3 candidate scores and the three
independent real-resolution pattern writes are evaluated serially. Candidate
workers should return scalar metrics only, while the winning candidate is
simulated once in the parent; export workers should return only downsampled
previews to avoid retaining three full-aperture arrays in the parent process.

Benchmark command: 40 S4 candidates plus 40 H3 candidates, 0.5 um small
simulation grid, 2 um export grid, and a 0.3 mm smoke aperture.

Results:

- Serial: `21.733 s`
- `--n-cores 2`: `13.665 s` (`1.59x` speedup)
- `--n-cores 4`: `11.041 s` (`1.97x` speedup)

The parallel run selected the same S4 and H3 candidates as serial execution;
TIFF outputs were byte-identical. GDS byte hashes differ only because GDS
library timestamps are generated at export time. The Windows sandbox denied
process pipes with `WinError 5`, so the implementation falls back to NumPy-
friendly thread workers; unrestricted Windows runs can use process workers.

Decision: keep the parallel implementation, expose `--n-cores`, and preserve
the small simulation versus real-resolution export separation. Full tests pass
after the change.

Final production check: the requested `13.3 mm` square export with `224 x 224`
simulation pixels at `0.5 um`, `6650 x 6650` export pixels at `2 um`, 160 S4
and H3 candidates, and `--n-cores 4` completed in `23.2 s`. Large TIFF/NPY/GDS
writes were serialized with atomic TIFF replacement after concurrent Pillow
writes produced a Windows `OSError 22`; the candidate search stayed parallel.

## High-Resolution Streaming Export

Hypothesis: a real `13.3 mm` aperture sampled at `0.05 um` cannot be assembled
as a NumPy array (`266000 x 266000`, about 66 GiB as `bool`). Regenerating a
single periodic supercell and streaming bounded NPY rows or TIFF tiles should
retain the exact periodic raster without full-aperture RAM use.

Result: the no-full-raster production path completed at `266000 x 266000` in
`13.3 s` with two-iteration smoke optimization, producing GDS, center PNG, and
overview files without the former `np.tile` allocation. A separate streamed
NPY/TIFF smoke export and a direct periodic-equivalence test passed. The writer
now retains only a one-supercell base tile; NPY uses a disk-backed memmap and
TIFF uses tiled BigTIFF output. A full 0.05 um NPY still requires about `66 GiB`
of disk per pattern, so full NPY/TIFF files remain explicit opt-in outputs.

Decision: retain serial fabrication writing to bound RAM and disk contention;
keep `--n-cores` for small optimization/validation work only.
