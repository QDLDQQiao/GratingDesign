# GratingDesign

This repository now contains a runnable SB-BSTG grating pattern design prototype.
It builds a binary phase grating from four star-carrier components, simulates
monochromatic propagation with the validated FFT propagator, scores the detector
pattern, and searches for a balanced high-visibility design.

## Run

```powershell
rtk python scripts/run_design.py --config config/default.json --iterations 80
```

Set the generated image/grid size directly with `--image-size-px`; it must be a
multiple of `samples_per_supercell` so the carrier peaks stay on FFT bins:

```powershell
rtk python scripts/run_design.py --image-size-px 512 --samples-per-supercell 128
```

The default run writes:

- `results/design_run/design_report.json`
- `results/design_run/design_summary.png`
- `results/design_run/optimization_history.png`
- `results/design_run/thickness_map_m.npy`
- `results/design_run/intensity.npy`

## Designed Pattern Images And Physical Comparison

Export phase-mask images and propagated detector intensities for every planned
family: S4, H3, H3-4L, DRH6, FATO proxy, square/hex CUC, RBN, MPG proxy, and
single- and two-shot grid/checkerboard controls:

```powershell
rtk python scripts/generate_pattern_images.py --design-report results/design_run/design_report.json
```

This writes `results/pattern_images/designed_pattern_images.png`, plus NumPy
arrays for each binary mask and detector intensity.

To export the exact S4, optimized H3, and CUC-hex candidates proposed for the
first fabrication comparison, run the S4 and H3 optimizations first, then pass
both reports to the exporter:

```powershell
rtk python scripts/generate_pattern_images.py --design-report results/design_run/design_report.json --h3-optimization-report results/h3_optimization/h3_optimization_report.json --patterns S4 H3 CUC_hex --out results/recommended_patterns
```

The exporter writes each mask as `<pattern>_phase_map_rad.npy` and a combined
inspection image. The arrays are ideal phase maps for numerical use; convert
them into a process-specific vector layout only after calculating the required
material thickness at the selected X-ray energy.

Run the corresponding physical simulation comparison:

```powershell
rtk python scripts/simulate_pattern_comparison.py --photons-per-pixel 10000 --repeats 5
```

It propagates all planned masks at the configured energy and distance, measures
their intended Fourier-carrier visibility and contamination, then runs phase,
transmission `T`, and mean dark-field `D` noise trials using each pattern's
measured visibility. RBN is propagated but deliberately excluded from the
carrier-demodulation reconstruction metric because it needs a speckle/random
mask retrieval method. FATO and MPG are labelled as proxies until their planned
optimization/fabrication models exist. The report and figures are written to
`results/pattern_simulation/`; the comparison does not yet include source
spectrum, detector MTF, or fabrication perturbations. The two-shot C2 control
has physical visibility metrics, but its phase/T/D entries remain `n/a` until a
dedicated two-image fusion reconstruction is implemented.

Measure the current reconstruction bandwidth using sinusoidal phase, `T`, and
isotropic `D` phantoms, then combine it with the photon-noise comparison:

```powershell
rtk python scripts/analyze_pattern_readiness.py --pattern-report results/pattern_simulation_all_smoke/pattern_simulation_report.json
```

The output contains a spatial-transfer figure, a phase/T/D scorecard, and a
JSON report. It reports the 50% transfer half-period of the present
reconstruction pipeline; it is not a detector-MTF or full experimental
resolution measurement.

## H3/H3-4L Joint Optimization

Optimize the H3 and four-level H3 layouts using propagated carrier visibility,
then phase, `T`, and directional-dark-field reconstruction at matched dose and
reconstruction settings:

```powershell
rtk python scripts/optimize_h3_patterns.py --iterations 160 --validation-photons-per-pixel 10000 --validation-repeats 5
```

The report under `results/h3_optimization/` stores the candidate histories,
selected phase maps, detector intensities, validation metrics, and a summary
figure. The result remains an idealized visibility-calibrated screen until the
material, detector, and sample-through-pattern forward models are implemented.

Compare the selected H3 designs against S4, CUC-hex, grid, and checkerboard at
the same 10,000 photons/pixel, five-repeat, and reconstruction settings:

```powershell
rtk python scripts/compare_optimized_h3.py
```

## 14 keV Fabrication-Raster Export

Generate binary S4, optimized H3, and fixed CUC-hex control masks for Si at
14 keV, a pi phase change, 0.4 m propagation distance, 0.05 um raster resolution,
and a 13.3 mm square field:

```powershell
rtk python scripts/export_fabrication_patterns.py --energy-kev 14 --material Si --phase-change-rad 3.141592653589793 --width-mm 13.3 --height-mm 13.3 --pixel-um 0.05 --simulation-pixel-um 0.5 --simulation-supercells 2 --z-m 0.4 --n-cores 4 --out results/fabrication_14kev
```

The default high-resolution run writes compact GDSII layouts, unlabeled central
PNG crops, and bounded overview images without materializing a full-aperture
raster in RAM. The exporter stores one periodic supercell and generates only
small windows as required for each output.

Full 13.3 mm x 13.3 mm rasters at 0.05 um are about 70.8 billion pixels:
one boolean NPY needs about 66 GiB before filesystem overhead. Request those
large files only when the required disk space is available; they are streamed
to disk in bounded chunks rather than allocated in RAM:

```powershell
rtk python scripts/export_fabrication_patterns.py --pixel-um 0.05 --save_npy --save_tiff --export-chunk-rows 64 --tiff-tile-px 1024 --out results/fabrication_14kev_full_raster
```

`False` is the zero-phase level and `True` is the pi-phase level. The metadata
records the Si optical constants and material-dependent pi-height needed to
convert the raster into the process stack.

`--simulation-pixel-um` and `--simulation-supercells` control only the small
optimization/propagation window. `--pixel-um`, `--width-mm`, and `--height-mm`
control the independent real export raster. Selected optimized coefficients are
regenerated on the export-resolution periodic tile before the full aperture is
assembled.

`--n-cores` parallelizes independent candidate scores in the small
optimization/validation stages. Fabrication file writes are serial by design,
which bounds RAM and avoids competing multi-gigabyte disk writes. In restricted
Windows sessions, optimization process workers automatically fall back to
NumPy-friendly threads.

## Focused Dammann Grating

The interactive notebook `scripts/xray_mask_generation_interactive.ipynb`
includes a binary-phase Dammann beam splitter for an external focusing optic.
`orders_per_axis=5` designs a crossed 5 x 5 beam array. Its suggested period is
calculated from focal length and target focal-plane separation:
`period = wavelength * focal_length / separation`. The notebook reports an
appropriate raster pitch, illuminated aperture, and diffraction-limited spot
width before generating PNG, TIFF, NPY, JSON, compact GDSII, and a central PNG
crop. A standalone Dammann-Fresnel mask remains optional, but its outer Fresnel
zones can require substantially finer fabrication resolution.

The same notebook cell simulates both the diffraction-limited reference focal
spot and the split 5 x 5 focal spots using a circular pupil and padded FFT
sampling. It saves focal intensity, reference intensity, the order-power map,
JSON metrics, and `dammann_5x5_f300mm_focal_simulation.png`.

The notebook also provides a single simulation cell for the complete planned
carrier library: S4, H3, H3-4L, DRH6, FATO/MPG proxies, square/hexagonal CUC,
RBN, grid, checkerboard, and the two-shot control.

## Compact Raster GDS

Raster-to-GDS conversion can become large because a naive writer emits one
rectangle for every horizontal pixel run. To compact an existing GDS produced
by this repository, preserving its hierarchy and AREFs, run:

```powershell
rtk python scripts/compact_generated_gds.py generated_patterns/dammann_5x5_f300mm.gds --output generated_patterns/dammann_5x5_f300mm_compact.gds
```

It losslessly merges vertically touching rectangles with the same x extent in
every GDS structure. For an original boolean NPY mask, use the NPY-to-GDS
variant instead:

```powershell
rtk python scripts/compact_raster_gds.py generated_patterns/dammann_5x5_f300mm.npy --pixel-um 1.0 --output generated_patterns/dammann_5x5_f300mm_compact.gds --cell-name DAMMANN_5X5
```

Both utilities preserve the same binary geometry. The NPY utility reads its
source through a memory map, so the full input mask is not loaded into RAM.

Use a faster smoke run while editing:

```powershell
rtk python scripts/run_design.py --iterations 12 --samples-per-supercell 64 --out results/smoke_design
```

## Metrics

The optimizer maximizes a score built from detector-plane Fourier visibility at
the four carrier directions. For a binary pi phase mask the default detector
metric uses harmonic `2`, matching the validated Talbot behavior where the main
intensity modulation appears at twice the grating carrier:

```text
(m, 0), (0, m), (n, n), (n, -n)
```

The score rewards mean visibility and the weakest carrier, then penalizes
carrier imbalance and local Fourier-window contamination. The output JSON stores
the full candidate history so the result can be audited or re-ranked later.

## Propagation Validation

The original propagation acceptance suite is still available:

```powershell
rtk python tests/test_propagate.py
```

It writes `results/test_propagate_report.json`.

## Phase II Reconstruction

The next-step reconstruction benchmark from `plan.md` is implemented as a
synthetic carrier-demodulation pipeline:

```powershell
rtk python scripts/run_reconstruction.py
```

To use the visibility measured from an optimized grating design:

```powershell
rtk python scripts/run_reconstruction.py --design-report results/design_run/design_report.json
```

It writes:

- `results/reconstruction_run/reconstruction_report.json`
- `results/reconstruction_run/reconstruction_summary.png`
- `results/reconstruction_run/phase_reconstruction.npy`
- `results/reconstruction_run/phase_truth.npy`

The benchmark creates a known smooth phase phantom, synthesizes sample/reference
carrier images, extracts complex envelopes around each detector carrier, fuses
directional gradients, integrates phase with a Fourier closed-form solver, and
reports separate accuracy metrics for phase, transmission `T`, and dark-field
`D`.

To search grating candidates directly by reconstruction quality:

```powershell
rtk python scripts/optimize_reconstruction.py --iterations 32 --out results/reconstruction_optimized_design
```

That optimizer evaluates each candidate on recovered phase, `T`, and `D` metrics,
using photon noise by default, then writes `reconstruction_optimization_report.json`
alongside the selected pattern and visualization.

## Star/Grid/Checkerboard Comparison

Run the modality comparison:

```powershell
rtk python scripts/compare_modalities.py --photons 3000 10000 30000 --repeats 5 --out results/modality_comparison
```

It compares the star-like four-direction single-shot layout against two-direction
grid, two-direction checkerboard, and equal-total-dose two-shot checkerboard
layouts. Outputs are written to `results/modality_comparison/`, with a research
brief in `outputs/star_grid_checkerboard_comparison.md`.
The recovered-map figure includes phase, `T`, and mean `D`; the companion
`modality_directional_dark_field.png` shows recovered anisotropy and principal
direction for the rank-identifiable methods.

The reconstruction uses exact FFT-bin carrier extraction, optional soft spectral
windows, phase wrapping rather than path-dependent 2-D unwrapping, Gaussian
denoising of the recovered maps, separate D-only denoising, and local
visibility-weighted fusion of the directional phase gradients. The validated
comparison uses an exact carrier window, 5 px phase/T smoothing, and 16 px D
smoothing. A soft window is retained for ablation because it can reduce
sideband leakage, but in the current synthetic benchmark it biases the
low-frequency transmission estimate and is not selected automatically.

To reproduce the reconstruction-method ablation and select settings at a fixed
dose:

```powershell
rtk python scripts/optimize_reconstruction_method.py --photons 3000 --repeats 3
```

The script writes a ranked CSV and JSON report under
`results/reconstruction_method_optimization/`.

## Energy/Distance Coverage

Run an energy and propagation-distance coverage comparison:

```powershell
rtk python scripts/compare_coverage.py --energies-kev 10 11 12.4 14 16 --distances-m 0.08 0.16 0.24 0.30 0.40
```

The script re-simulates star, grid, and checkerboard masks at each energy and
distance, then reports visibility coverage and writes
`results/coverage_comparison/coverage_heatmap.png`.

## Broad Advantage Analysis

Build a single summary of geometry, phase/T/D reconstruction, dose robustness,
and energy-distance coverage:

```powershell
rtk python scripts/analyze_grating_advantages.py
```

The outputs are written to `results/grating_advantage_analysis/`, including
`grating_advantage_summary.png`, `grating_advantage_coverage.png`, and a JSON
report with the measured trade-offs and limitations.
