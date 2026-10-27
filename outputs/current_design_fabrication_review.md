# Research review: current star-like grating design and fabrication readiness

## Recommendation

**Do not fabricate the current `thickness_map_m.npy` as the final experimental grating.**

The project is worth continuing, and a small process/test coupon becomes reasonable after the
material model, detector model, pattern-specific forward comparison, and minimum-feature rules
are corrected. A full-area grating order is premature.

## Major findings

### 1. The saved thickness map is not physical

`src/grating_design.py` sets `h_pi = wavelength_m / 2`, which produces a nominal relief of
0.05 nm at 12.4 keV. The required phase height is instead

`h_pi = wavelength / (2 delta)`.

Using xraylib at 12.398 keV gives approximately:

| material | pi thickness | calculated intensity transmission through pi region |
|---|---:|---:|
| Si | 15.79 um | 0.936 |
| Au | 2.69 um | 0.417 |
| Ni | 4.34 um | 0.632 |

Silicon is the most natural starting point here because it retains high transmission and the
required depth is compatible with established DRIE processes, but the pattern must be simulated
with both delta and beta rather than an ideal lossless `exp(i*pi)` mask.

### 2. The current reconstruction comparison is not end-to-end grating validation

The modality benchmark synthesizes ideal cosine carriers and reconstructs them with the matching
carrier model. It assigns the same scalar visibility to star, grid, and checkerboard methods.
The separate propagation analysis, however, predicts average visibility near 0.17 for the star
and 0.40 for the two-direction patterns over the energy-distance sweep.

Therefore the present comparison demonstrates:

- the rank advantage of four directions;
- behavior of the carrier demodulator under matched synthetic noise;
- the penalty of splitting equal total dose over two exposures.

It does **not** yet prove that the fabricated star mask gives lower phase or dark-field error than
a fabricated grid/checkerboard under their actual pattern-specific visibility, crosstalk, detector
MTF, and source coherence.

### 3. The reported accuracy is promising but moderate, not definitive

At 10,000 photons/pixel the star reports phase RMSE 0.00737 rad, mean-D RMSE 9.04e-4,
anisotropy RMSE 0.0550, and D-angle MAE 3.81 degrees.

The phase phantom has RMS 0.03445 rad, so the phase RMSE is 21.4% of the truth RMS. The D
anisotropy truth is 0.35, making the anisotropy error 15.7% of the true value. Mean-D error is
2.6% of the nominal 0.035 strength. These are useful proof-of-principle values, but the phase
result should not yet be described as high-accuracy metrology.

Directional-D accuracy also uses 16-pixel Gaussian smoothing, equal to 7 um on the current
numerical grid. The improved angle and anisotropy precision therefore comes with reduced spatial
resolution. At 300 photons/pixel, angle error rises to about 22 degrees.

### 4. Manufacturability constraints are not enforced

The current 256x256 binary mask has a 0.4375 um numerical pixel and contains one-pixel axis runs.
Its smallest connected feature is about 5.74 um2. With a 15.8 um silicon etch, isolated local
features can approach an effective 36:1 aspect ratio. The project plan specified a 2 um minimum
feature, morphological repair, and a 5% maximum repair fraction, but these gates are not present
in the current implementation.

The main carrier periods are physically sensible: 56/7 = 8.0 um axially and approximately
56/sqrt(50) = 7.92 um diagonally, with an 80 mm propagation distance corresponding to the chosen
fractional-Talbot regime. The problem is not the base period; it is the unfiltered small geometry
created by thresholding four carriers.

### 5. The robustness result does not meet the acceptance gate in `plan.md`

The saved coverage sweep uses monochromatic propagation, a minimum-visibility threshold of 0.10,
five energies, and five distances. The plan requires real detector MTF, 10,000-photon noise,
minimum visibility at least 0.15, a nonempty operating window of at least +/-5% in distance and
+/-3% in energy, and fabrication tolerances. The current 0.96 pass fraction cannot substitute for
that denser local operating-window test.

Missing experimental effects include:

- material dispersion and absorption;
- polychromatic spectrum integration;
- finite source size and partial coherence;
- pixel aperture and detector MTF;
- phase-depth, line-width, sidewall-angle, stitching, and substrate errors;
- sample-edge artefacts, flat-field drift, and non-dark-field blur;
- mixed fibre orientations and dark-field models outside the fitted `cos(2 theta)` law.

## Strengths that are supported

1. Four carrier visibilities are balanced: 0.1943-0.1964, with 0.55% coefficient of variation.
2. Ideal Fourier-window contamination is low at 0.85%.
3. The four directions give rank 3 and condition number 1.414 for the fitted directional-D model.
4. A single shot contains enough directional degrees of freedom to estimate D0, anisotropy, and
   principal angle; two-direction methods are structurally rank deficient.
5. Equal-total-dose simulations favor the star over a two-shot four-direction checkerboard.
6. The approximately 8 um base period and silicon depth near 16 um are within demonstrated
   micromachining territory after minimum-feature cleanup.

## Novelty assessment

### Concept-level novelty: low

Single-shot multi-directional or omnidirectional X-ray phase/dark-field imaging with 2-D gratings
is established prior art. Relevant examples include 2-D transmission gratings with spatial harmonic
analysis (2011), circular-unit-cell omnidirectional phase gratings (2016-2017), fabricated 2-D
hard-X-ray gratings, and single-grid directional-dark-field retrieval (2023). Broad patent claims
also exist around 2-D and single-shot X-ray phase/dark-field gratings.

### Pattern-level novelty: possible, but unproven

The potentially publishable contribution is narrower:

- a binary supercell formed from the balanced `(7,0)`, `(0,7)`, `(5,5)`, `(5,-5)` carrier set;
- simultaneous axial and diagonal phase/D channels with near-equal visibility;
- reconstruction-aware optimization under explicit manufacturability constraints;
- a measured advantage in dose, detector-resolution demand, bandwidth, or tolerance relative to
  circular-cell, grid, and checkerboard gratings.

The exact carrier construction was not found in this limited review, but that is not a formal
novelty or freedom-to-operate search. The project should avoid claiming novelty for “single-shot
directional dark-field” itself.

## Fabrication decision

| decision | verdict |
|---|---|
| Fabricate the current full mask | No |
| Continue simulation and design work | Yes |
| Fabricate a small process coupon after DRC/material correction | Yes, conditionally |
| Commit to a full-field experimental grating now | No |
| Potentially publishable idea | Yes, if the specific pattern wins a fair end-to-end comparison |

## Minimum gates before a test coupon

1. Replace `lambda/2` thickness with xraylib `delta/beta` material transmission and regenerate the
   mask for a named material and energy.
2. Enforce a fabrication rule of at least 2 um minimum line/space, remove islands and acute tips,
   and report mask-repair fraction and post-repair visibility.
3. Propagate sample + physical grating independently, then reconstruct with a separate algorithm.
4. Use each pattern's actual per-carrier visibility rather than a shared scalar visibility.
5. Add detector pixel integration, MTF, source blur, spectrum integration, and at least 30-100 noise
   realizations.
6. Run dense +/-10% distance and energy sweeps around the selected operating point and require the
   `plan.md` 0.15-visibility, +/-5% z, +/-3% energy gate.
7. Compare star, grid, checkerboard, and circular-unit-cell baselines at equal total dose and equal
   effective spatial resolution.

## Minimum decisive experiment

Fabricate the star and at least one grid/checkerboard control on the same silicon wafer and process
them to the same measured depth. First measure feature dimensions, sidewall angle, depth uniformity,
and flat-field carrier visibility. Then image an anisotropic phantom such as aligned carbon-fibre
bundles at several known orientations. Report phase RMSE against a calibrated phase object, D0 error,
anisotropy error, principal-angle error, spatial resolution, and dose. The star idea is validated only
if it retains directional accuracy after matching dose, visibility calibration, and resolution.

## Literature used

- Single-shot 2-D transmission grating phase/diffraction imaging:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3091831/
- 2-D omnidirectional hard-X-ray scattering in one shot:
  https://portal.research.lu.se/en/publications/2d-omnidirectional-hard-x-ray-scattering-sensitivity-in-a-single-/
- Circular-unit-cell phase-grating optimization:
  https://www.sljus.lu.se/matias-kagias/publication/3bec6639-db2f-4dde-a099-4c593f30e62e
- Single-grid directional-dark-field retrieval:
  https://pubmed.ncbi.nlm.nih.gov/37155790/
- Fabrication of 2-D hard-X-ray diffraction gratings:
  https://www.sciencedirect.com/science/article/abs/pii/S0167931712004935
- Experimental artefacts from source/detector blur and attenuation edges:
  https://arxiv.org/abs/2404.17612
- Practical scope of omnidirectional X-ray phase/dark-field imaging:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8617406/

This is a technical research assessment, not a formal patentability or freedom-to-operate opinion.
