# Phase II Reconstruction Brief

This implementation follows the `plan.md` Phase II direction: single-shot
carrier demodulation should recover attenuation, directional phase gradients,
and dark-field changes from sample/reference images. The code implements the
core measurable pieces first: Fourier-window carrier extraction, phase-shift to
directional-gradient conversion, weighted multi-direction gradient fusion,
Fourier-domain phase integration, and dark-field anisotropy fitting.

The design choice is consistent with grating and single-shot Fourier analysis
literature: grating-based phase contrast separates absorption, differential
phase, and dark-field channels by comparing sample and reference fringe
modulations; single-shot approaches use carrier-fringe Fourier windows rather
than mechanical phase stepping. Recent single-mask and UMPA work also supports
the same practical division between attenuation, phase-gradient/displacement,
and dark-field/scattering terms.

Implementation scope:

- `src/reconstruction.py` contains demodulation, gradient fusion, Fourier phase
  integration, and dark-field anisotropy fitting.
- `src/phantoms.py` contains analytic validation phantoms.
- `scripts/run_reconstruction.py` runs a synthetic end-to-end benchmark and
  writes plots plus JSON metrics.

Current validation result is an analytic synthetic benchmark rather than a full
material-forward simulation. This is intentional: it isolates the inverse
pipeline before coupling it to detector noise, polychromatic propagation, and
sample material models. The benchmark now reports separate recovery metrics for
phase, transmission `T`, dark-field mean, dark-field anisotropy, and dark-field
angle. It can also consume `results/design_run/design_report.json` so the
optimized grating visibility controls the reconstruction simulation.

The current implementation also includes a reconstruction-aware grating search
(`scripts/optimize_reconstruction.py`). Unlike the visibility-only design loop,
this evaluates each candidate under photon noise and scores recovered phase,
`T`, and `D` while penalizing weak minimum visibility, carrier imbalance, and
local spectral contamination. On the default benchmark at \(10^4\)
photons/pixel, the balanced symmetric grating remains the selected design, with
dark-field orientation error below \(5^\circ\).

Sources:

1. `plan.md`, section D, local specification for carrier demodulation and
   reconstruction.
2. Meyer et al., "X-ray Interferometry Using a Modulated Phase Grating: Theory
   and Experiments", arXiv:2404.14584.
3. De Marco et al., "High-speed processing of X-ray wavefront marking data with
   the Unified Modulated Pattern Analysis (UMPA) model", arXiv:2211.01375.
4. Yuan and Das, "Single-Shot, Single-Mask X-ray Dark-field and Phase Contrast
   Imaging", arXiv:2506.02427.
