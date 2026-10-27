# Star-like grating performance validation

## Scope

This report evaluates the star-like four-carrier pattern against an axis-aligned
grid, a two-direction checkerboard, and a two-shot four-direction checkerboard.
The measurements use the repository's periodic phase phantom, smooth absorption
phantom, and anisotropic directional dark-field phantom. Metrics are RMSE for
phase and transmission `T`, RMSE for mean dark-field `D`, anisotropy RMSE, and
wrapped principal-angle error. The dose sweep uses 300, 1000, 3000, and 10000
photons/pixel with five random repeats.

## Method validation

The reconstruction model is a local carrier expansion. The DC Fourier envelope
estimates absorption, each carrier envelope gives a phase shift and a visibility
loss, and the phase shifts are converted to directional phase gradients. The
four star directions are fused by a per-pixel weighted least-squares solve and
integrated by a Fourier Poisson solver. Dark-field is fit to

`D(theta) = D0 + Dc cos(2 theta) + Ds sin(2 theta)`.

The implementation now uses the wrapped complex phase `angle(env_sample/env_ref)`
instead of sequential 2-D `unwrap` calls. This avoids path-dependent phase jumps
when shot noise makes a local phase cross the branch cut. It smooths the
visibility maps and uses their squared values as local weights in gradient fusion.
Dark-field log-amplitude maps receive a separate, stronger smoothing scale than
phase/T. The final comparison uses 5 px phase/T smoothing, 16 px D smoothing,
and the exact FFT window. A soft Gaussian window remains available for ablation
but biases the DC-derived transmission in this model.

## Measured results

At 10000 photons/pixel, the adaptive reconstruction gives the following means:

| pattern | phase RMSE (rad) | T RMSE | D mean RMSE | D anisotropy RMSE | D angle MAE | directional D |
|---|---:|---:|---:|---:|---:|---|
| star, 4 directions, 1 shot | 0.00737 | 2.77e-4 | 9.04e-4 | 0.0550 | 3.81 deg | yes |
| grid, 2 directions, 1 shot | 0.0106 | 2.83e-4 | 0.00163 | not identifiable | not identifiable | no |
| checkerboard, 2 directions, 1 shot | 0.0107 | 2.76e-4 | 0.00155 | not identifiable | not identifiable | no |
| checkerboard, 4 directions, 2 shots | 0.0110 | 3.75e-4 | 0.00137 | 0.0718 | 5.36 deg | yes |

At 300 photons/pixel the star still has rank-three directional dark-field
identifiability, but the angular error rises to about 22 degrees and anisotropy
RMSE remains about 0.35. This is a precision limit, not a loss of geometric
identifiability. The two-direction patterns cannot estimate both anisotropy
components at any dose because their design matrix has rank two. The two-shot
checkerboard restores rank three but pays a dose split and therefore has worse
low-dose phase, `T`, and angle errors in this benchmark.

## Advantages of the star-like pattern

1. **Single-shot directional dark-field.** Four non-collinear carrier directions
   give rank three for the `[1, cos(2 theta), sin(2 theta)]` dark-field model.
   This recovers mean scattering strength, anisotropy, and principal direction
   without rotating the sample or taking a second exposure. This is consistent
   with the established motivation for multidirectional 2-D grating
   interferometry, which provides four in-plane directions without rotating the
   interferometer or sample ([Zanette et al.](https://portal.fis.tum.de/en/publications/multidirectional-x-ray-dark-field-imaging-with-two-dimensional-gr/)).

2. **Better phase conditioning.** The four directions overdetermine the two
   Cartesian phase-gradient components. The star's phase RMSE is about 31% lower
   than the grid result at 10000 photons/pixel and about 60% lower than the
   two-shot checkerboard result in the same run.

3. **Dose efficiency for directional information.** The star obtains all four
   directional channels in one exposure. A checkerboard can reach comparable
   geometric rank only by combining exposures, so equal total dose is divided
   between shots. The measured penalty is especially visible below 3000
   photons/pixel.

4. **A natural quality-control signal.** Four carrier visibilities provide a
   per-pixel and per-direction consistency check. Strong imbalance or leakage
   can be detected before phase integration, which is useful for tolerance and
   energy/distance sweeps.

5. **Compatibility with the three standard contrast channels.** The design
   produces the same three physical readouts expected in grating interferometry:
   attenuation/transmission, differential phase, and dark-field. Visibility is
   directly tied to the noise of the retrieved images, so optimizing the weakest
   star carrier is relevant to reconstruction precision
   ([Wu et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5234509/);
   [Shashev et al.](https://www.degruyterbrill.com/document/doi/10.3139/120.110948/html?lang=en)).

## Limitations and required next validation

The benchmark is a controlled forward model, not a measured polychromatic
experiment. It assumes exact carrier-bin alignment, stationary visibility, and a
cosine dark-field angular law. The coverage model already sweeps energy and
distance, but the next physical validation should add spectrum averaging, source
blur, detector MTF, grating fabrication errors, and flat-field drift. Published
visibility theory treats partial coherence, spectral averaging, and detector
re-binning explicitly ([Wu et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5234509/));
those effects can change the best distance and can make a soft carrier window
useful after calibration.

The current conclusion is therefore bounded: under the implemented monochromatic
noise model, the star-like pattern is the strongest single-shot choice for joint
phase, transmission, and directional dark-field recovery. Its principal benefit
is directional information per exposure, while its main risk is that the fourth
channel must retain adequate visibility across energy, distance, and fabrication
tolerance variations.
