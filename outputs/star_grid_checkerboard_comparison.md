# Star-Like vs Grid/Checkerboard Reconstruction Comparison

## Scope

This comparison evaluates the current SB-BSTG star-like four-direction carrier
layout against grid/checkerboard alternatives using the repository's synthetic
carrier reconstruction benchmark. The benchmark reconstructs phase, transmission
`T`, dark-field mean `D0`, dark-field anisotropy, and dark-field direction under
Poisson noise at equal total dose.

The compared layouts are:

- `star_4dir_single`: one exposure with axial plus diagonal carriers.
- `grid_2dir_single`: one exposure with axial carriers only.
- `checkerboard_2dir_single`: one exposure with diagonal carriers only.
- `checkerboard_4dir_2shot`: two exposures combining axial and diagonal carrier
  directions, with total dose split equally across the two exposures.

## Literature Context

Recent single-mask and single-grid X-ray work supports the basic strategy used
here: attenuation, DPC, and dark-field can be recovered from local changes in a
modulated pattern or grid image. Yuan and Das describe single-exposure recovery
of attenuation, DPC, and dark-field from a single-mask design, emphasizing the
motivation to avoid multiple exposures. Meyer et al. discuss modulated phase
gratings that create directly resolvable 2D interference patterns and evaluate
fringe visibility versus geometry. Croughan et al. show that directional
dark-field retrieval requires a model of directional blur/scattering parameters,
while How and Morgan quantify single-grid dark-field from sample-induced grid
shadow changes.

The important design implication is angular rank. A dark-field model of the
form

```text
D(theta) = D0 + Dc cos(2 theta) + Ds sin(2 theta)
```

has three unknowns per pixel. A two-direction grid/checkerboard layout is
rank-deficient for directional `D` unless additional assumptions or additional
exposures are introduced. The star-like layout supplies four carrier directions
in one exposure, so the fit is overdetermined.

## Measured Results

The full results are saved in:

- `results/modality_comparison/modality_comparison.csv`
- `results/modality_comparison/modality_comparison_report.json`
- `results/modality_comparison/modality_comparison.png`
- `results/modality_comparison/modality_recovered_maps.png`
- `results/modality_comparison/modality_error_maps.png`

`modality_recovered_maps.png` shows ground-truth phase, `T`, and `D0` alongside
the reconstructed maps for each method. `modality_error_maps.png` shows the
corresponding error maps. For two-direction grid/checkerboard methods,
directional dark-field angle and anisotropy are marked non-identifiable, but
the scalar dark-field mean `D0` is still plotted.

## Reconstruction Method

The comparison uses the same Fourier carrier-demodulation pipeline for every
method.

First, a reference image is synthesized from the method's carrier set. The
sample image is generated with known ground truth: a smooth phase map
`\phi(x,y)`, transmission `T(x,y)=exp(-mu(x,y))`, and directional dark-field
channels `D_j(x,y)`. For each carrier direction `j`, the sample image contains
a local phase shift proportional to the directional phase gradient
`n_j · grad(phi)` and a fringe-amplitude reduction `exp(-D_j)`.

Second, reconstruction extracts each carrier by applying a circular Fourier
window around the carrier peak, shifting that peak to DC, and inverse
transforming to a complex envelope. Comparing sample and reference envelopes
gives:

```text
absorption:        mu = -log(|a0_sample| / |a0_ref|)
phase shift:       dpsi_j = angle(a_j_sample / a_j_ref)
dark-field:        D_j = -log((|a_j_sample|/|a0_sample|) / (|a_j_ref|/|a0_ref|))
```

The directional phase shifts are converted to directional gradients, then fused
by weighted least squares into `dphi/dx` and `dphi/dy`. Phase is recovered with
a Fourier-domain least-squares integration. Transmission is reconstructed as
`T=exp(-mu)`.

For dark-field direction, each pixel fits:

```text
D(theta_j) = D0 + Dc cos(2 theta_j) + Ds sin(2 theta_j)
```

This requires angular rank 3. The star-like four-direction layout and the
two-shot four-direction checkerboard meet that rank condition. The single-shot
two-direction grid and checkerboard do not, so their anisotropy and direction
are reported as non-identifiable rather than as misleading numerical errors.

At 30,000 photons/pixel total dose:

| modality | phase RMSE | T RMSE | D0 RMSE | D anisotropy RMSE | D angle MAE | directional D identifiable |
|---|---:|---:|---:|---:|---:|---|
| star_4dir_single | 0.001657 | 0.000193 | 0.000839 | 0.0541 | 3.36 deg | yes |
| grid_2dir_single | 0.002108 | 0.000195 | 0.001158 | n/a | n/a | no |
| checkerboard_2dir_single | 0.002352 | 0.000191 | 0.001079 | n/a | n/a | no |
| checkerboard_4dir_2shot | 0.002375 | 0.000266 | 0.001019 | 0.0869 | 4.99 deg | yes |

At 10,000 photons/pixel total dose:

| modality | phase RMSE | T RMSE | D0 RMSE | D anisotropy RMSE | D angle MAE | directional D identifiable |
|---|---:|---:|---:|---:|---:|---|
| star_4dir_single | 0.002810 | 0.000287 | 0.001462 | 0.1137 | 6.91 deg | yes |
| grid_2dir_single | 0.003608 | 0.000281 | 0.002179 | n/a | n/a | no |
| checkerboard_2dir_single | 0.004173 | 0.000289 | 0.002029 | n/a | n/a | no |
| checkerboard_4dir_2shot | 0.003937 | 0.000383 | 0.001931 | 0.1671 | 9.27 deg | yes |

## Interpretation

For phase recovery, the star-like grating is the best of the tested single-shot
layouts across the photon sweep. At 30,000 photons/pixel, it improves phase RMSE
by about 21% relative to the two-direction grid and about 30% relative to the
single checkerboard. The advantage comes from four directional slope constraints
in one exposure.

For transmission `T`, all single-shot layouts are similar because attenuation is
mostly carried by the DC channel. The two-shot checkerboard is worse at equal
total dose because each exposure receives half the photons.

For dark-field mean `D0`, the star-like grating is again the strongest
single-shot option. It is also the only tested single-shot layout that can
estimate directional dark-field anisotropy and angle under the three-parameter
cosine model. The two-direction grid and single checkerboard are explicitly
marked non-identifiable for `D` anisotropy and direction, not merely low
accuracy.

For directional dark-field, the two-shot checkerboard can become identifiable
because it combines four directions, but it pays a dose penalty. In this
equal-total-dose simulation, it is consistently noisier than the star-like
single shot: at 30,000 photons/pixel its dark-field direction error is 4.99 deg
versus 3.36 deg for the star-like layout.

## Conclusion

For the current reconstruction model, the star-like SB-BSTG layout is preferable
when the goal is single-shot recovery of phase, `T`, and directional `D`. A
grid or single checkerboard can recover phase and transmission, but it cannot
support the anisotropic dark-field fit without additional assumptions or
additional exposures. A two-shot checkerboard can recover directional
dark-field, but it is less noise-robust under equal total dose.

## References

1. Yuan and Das, "Single-Shot, Single-Mask X-ray Dark-field and Phase Contrast
   Imaging", arXiv:2506.02427.
2. Meyer et al., "X-ray Interferometry Using a Modulated Phase Grating: Theory
   and Experiments", arXiv:2404.14584.
3. Croughan et al., "Directional dark-field retrieval with single-grid x-ray
   imaging", arXiv:2211.11757.
4. How and Morgan, "Quantifying the x-ray dark-field signal in single-grid
   imaging", arXiv:2112.13968.
