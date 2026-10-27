# Detailed advantage analysis of the star-like grating

## Executive summary

The star-like pattern is not the highest-visibility pattern in every optical
condition. The two-direction grid and checkerboard patterns concentrate their
modulation into fewer carriers and therefore often have higher raw visibility.
The star pattern wins on the more important system-level objective: it measures
four non-collinear directions in one exposure, making the three-parameter
directional-dark-field model identifiable while also overdetermining the two
Cartesian phase-gradient components.

In the current 25-point energy-distance sweep, the star passes 24/25 conditions
(0.96 coverage), compared with 20/25 for the grid (0.80) and 21/25 for the
two-direction checkerboard (0.84). In the combined-carrier optical model, the
two-shot checkerboard has the same coverage as the star, but it requires two
exposures and splits the dose. That distinction is decisive in the noisy
reconstruction benchmark.

At 10,000 photons/pixel, using the optimized 5 px phase/T and 16 px D smoothing:

| pattern | phase RMSE | T RMSE | mean-D RMSE | D anisotropy RMSE | D angle MAE | D direction identifiable |
|---|---:|---:|---:|---:|---:|---|
| star, 4 directions, 1 shot | 0.00737 rad | 2.77e-4 | 9.04e-4 | 0.0550 | 3.81 deg | yes |
| grid, 2 directions, 1 shot | 0.01064 rad | 2.83e-4 | 1.30e-3 | n/a | n/a | no |
| checkerboard, 2 directions, 1 shot | 0.01065 rad | 2.76e-4 | 1.27e-3 | n/a | n/a | no |
| checkerboard, 4 directions, 2 shots | 0.01096 rad | 3.75e-4 | 1.05e-3 | 0.0718 | 5.36 deg | yes |

The star reduces phase RMSE by 30.7% relative to the grid and gives 29.3% lower
mean-D RMSE. Against the two-shot four-direction checkerboard it gives 32.8%
lower phase RMSE, 14.1% lower mean-D RMSE, 23.4% lower anisotropy RMSE, and
28.9% lower angle error.

![Reconstruction and noise comparison](../results/grating_advantage_analysis/grating_advantage_summary.png)

![Energy-distance coverage comparison](../results/grating_advantage_analysis/grating_advantage_coverage.png)

## 1. Information geometry

The directional-dark-field model used by the program is

`D(theta) = D0 + Dc cos(2 theta) + Ds sin(2 theta)`.

It has three unknowns per pixel: mean scattering `D0` and two anisotropy
components `(Dc, Ds)`. The star directions are approximately 0, 90, 45, and
-45 degrees. Its design matrix has rank 3 and condition number 1.414. The grid
and single checkerboard each provide only two directions, giving rank 2 and an
infinite condition number for this three-parameter model. No increase in photon
count can repair that missing degree of freedom; it is a geometric
non-identifiability, not merely a noise problem.

The two-shot checkerboard restores rank 3 by combining the two carrier pairs,
but the same total dose is divided across exposures. Its rank is equal to the
star's rank and its condition number is also 1.414, so the star's advantage is
acquisition efficiency rather than a fundamentally different least-squares
geometry.

This matches the physical motivation of multidirectional 2-D grating
interferometry: four in-plane directions can be acquired without rotating the
sample or interferometer ([Zanette et al.](https://portal.fis.tum.de/en/publications/multidirectional-x-ray-dark-field-imaging-with-two-dimensional-gr/)).
Directional dark-field is specifically useful because the signal depends on
microstructure orientation and can reveal structures below the detector's
spatial resolution ([Malecki et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4156746/)).

## 2. Phase reconstruction

The star has the lowest phase RMSE at every tested dose. At 300, 1,000, 3,000,
and 10,000 photons/pixel its phase RMSE is respectively 0.0419, 0.0228, 0.0154,
and 0.00737 rad. The grid values are 0.0618, 0.0337, 0.0178, and 0.0106 rad.
The improvement is largest at low dose, where the overdetermined four-channel
fusion suppresses the noise in the recovered x/y gradient field.

The two-shot checkerboard is worse than the star at every dose because each
exposure receives half of the total photon budget. At 300 photons/pixel its
phase RMSE is 0.0666 rad, 59% higher than the star; at 10,000 photons/pixel it
remains 49% higher.

This is a real phase advantage in the present model, but it should not be
interpreted as a universal visibility advantage. The star distributes the
available modulation among four carrier components. Its phase advantage comes
from direction diversity and weighted fusion, while its raw carrier visibility
is lower than the two-direction alternatives.

## 3. Transmission `T`

Transmission is mainly recovered from the DC envelope, so pattern geometry has
less effect than it has on phase and D. At 10,000 photons/pixel, all one-shot
patterns have T RMSE near `2.8e-4`; the two-shot checkerboard is worse at
`3.75e-4` because of dose splitting. At 300 photons/pixel the same ordering is
smaller but still visible: star `1.59e-3`, grid `1.60e-3`, checkerboard
`1.61e-3`, and two-shot checkerboard `2.16e-3`.

The correct conclusion is therefore that the star is approximately neutral to
slightly better for T under equal single-shot dose, not that it creates a large
transmission-contrast advantage.

## 4. Mean and directional dark-field

Mean D benefits from the star's four independent visibility measurements. At
10,000 photons/pixel, star mean-D RMSE is `9.04e-4`, compared with `1.30e-3`
for grid, `1.27e-3` for two-direction checkerboard, and `1.05e-3` for the
two-shot checkerboard.

The more important result is directional D. The star recovers anisotropy and
principal angle in a single exposure. At 10,000 photons/pixel its anisotropy
RMSE is `0.0550` and angle error is `3.81 degrees`. The two-shot checkerboard
can recover the same quantities, but its errors are larger because the dose is
split. The two-direction patterns are reported as not identifiable rather than
being assigned a misleading pseudoinverse estimate.

At 300 photons/pixel, the star still has rank-three identifiability, but its
angle error rises to `22.2 degrees` and anisotropy RMSE to `0.350`. This is the
expected precision limit of a weak dark-field signal. The distinction is useful
experimentally: the star can tell the user that directional information exists
even when the confidence interval is too large for quantitative orientation
claims.

The reconstruction optimization used a separate D-only Gaussian smoothing scale.
At 3,000 photons/pixel, a bounded sweep selected 5 px phase/T smoothing, 24 px
D smoothing, and an exact carrier window. The broader 300-10,000 photon
comparison uses 16 px D smoothing as a balanced setting. The D-specific result
is a bias-variance tradeoff: stronger smoothing improves anisotropy and angle
precision but eventually biases spatially varying mean D.

The physical interpretation is supported by the established directional
dark-field literature: anisotropic scattering encodes local orientation and
anisotropy, and multiple directional measurements are required to recover those
parameters ([Wieczorek et al.](https://researchmgt.monash.edu/ws/portalfiles/portal/246556284/246549085_oa.pdf),
[Coherent superposition in directional dark-field imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC3634061/)).

## 5. Photon-noise robustness and exposure cost

The star's ranking is strongest at low and moderate dose. At 300 photons/pixel:

- star phase RMSE is 32% below grid;
- star angle error is 22 degrees, while two-direction methods cannot estimate an angle;
- star is better than the two-shot checkerboard on phase, T, mean D, anisotropy, and angle.

At 10,000 photons/pixel the star's angle error is only 3.81 degrees. The two-shot
checkerboard reaches 5.36 degrees, but requires two frames and has higher phase
and T errors. Thus, when motion, dose, or scan time matters, the star provides
more directional information per exposure.

The visibility tradeoff must be kept explicit. In the coverage model the grid
has average mean visibility `0.402`, versus `0.170` for the star. Higher
visibility generally improves dark-field SNR ([Dark-field chest X-ray imaging](https://www.nature.com/articles/s43856-022-00215-3)),
but visibility alone does not measure the number of recoverable directional
parameters. The star exchanges raw modulation depth for four-channel
information.

## 6. Energy-distance robustness

The current coverage sweep spans 10, 11, 12.4, 14, and 16 keV and distances
0.08, 0.16, 0.24, 0.30, and 0.40 m. A condition passes when minimum carrier
visibility is at least 0.10, balance CV is at most 0.35, and contamination is
at most 0.05.

| pattern | pass fraction | average mean visibility | worst minimum visibility | mean contamination |
|---|---:|---:|---:|---:|
| star 4-dir | 0.96 | 0.170 | 0.0990 | 0.0165 |
| grid 2-dir | 0.80 | 0.402 | 0.00679 | 0.00908 |
| checkerboard 2-dir | 0.84 | 0.402 | 0.00311 | 0.0382 |
| checkerboard 4-dir / 2 shots | 0.96 | 0.170 | 0.0990 | 0.0165 |

The star's coverage is substantially more uniform in the sense relevant to
four-carrier operation: it avoids the severe low-visibility failures seen in
the two-direction patterns. The identical star and two-shot coverage numbers
are expected in this model because the two-shot pattern uses the same combined
carrier set. The star still wins operationally because it obtains those carriers
in one exposure.

Energy dependence is physically important because grating phase depth, Talbot
distance, source spectrum, and detector response all change with energy. Measured
visibility spectra are typically peaked near a design energy, and the effective
dark-field signal is weighted by both photon flux and visibility
([Grating-based spectral X-ray dark-field imaging](https://www.nature.com/articles/s41598-020-70011-3)).
The current sweep is monochromatic and therefore a screening analysis, not a
polychromatic performance guarantee.

## 7. Carrier balance, contamination, and implementation risk

The star has four carrier channels, so it exposes more opportunities for
imbalance, overlap, and fabrication error. The coverage summary shows higher
mean contamination for the checkerboard two-direction case than for the star,
but a real device will require tolerance simulations for line placement, phase
depth, duty cycle, source blur, detector MTF, and flat-field drift.

The star's strengths are:

1. rank-three directional-D information in one exposure;
2. well-conditioned anisotropy fit from four non-collinear directions;
3. overdetermined x/y phase-gradient recovery;
4. better low-dose phase and D accuracy in the current benchmark;
5. 0.96 energy-distance coverage under the current thresholds;
6. a built-in four-channel consistency check for visibility imbalance.

Its costs are:

1. lower raw visibility than two-direction patterns;
2. more complex mask synthesis and fabrication;
3. greater sensitivity to one carrier becoming weak;
4. directional-D accuracy still becomes poor at very low photon counts;
5. current coverage is based on monochromatic propagation and does not include
   source spectrum, detector MTF, or fabrication tolerances.

## Bottom line

For scalar transmission only, a two-direction pattern can be attractive because
it concentrates visibility into fewer carriers. For joint phase, transmission,
mean dark-field, anisotropy, and principal direction, the star-like grating is
the best single-shot choice in this benchmark. Its strongest advantage is not
maximum fringe contrast; it is the amount of identifiable directional information
per exposure and the resulting lower phase/D error under equal total dose.

The next experimental validation should use a polychromatic spectrum and measured
flat fields, add source-size and detector-MTF blur, perturb carrier phase/depth
and duty cycle, and report confidence intervals for the recovered D angle. That
will test whether the current 0.96 coverage and 3.81-degree high-dose angle
accuracy survive realistic instrument nonidealities.
