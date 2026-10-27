# Literature review for the next grating-design phase

## Scope

This review supports the roadmap appended to `plan.md`. It asks which structures should be treated as
mandatory baselines and which narrower hypotheses may still provide publishable novelty.

## Findings

1. Two-dimensional single-shot differential phase and diffraction imaging is established. A new design
   cannot claim novelty solely from using a 2-D grating or Fourier spatial-harmonic retrieval.
2. Circular unit-cell phase gratings provide omnidirectional hard-X-ray scattering sensitivity, and
   hexagonal packing has already been experimentally compared with Cartesian packing.
3. Directional dark-field retrieval from single grids and random/speckle modulators is established,
   including anisotropic-diffusion models that recover a local scattering ellipse.
4. Modulated phase gratings are an active route for producing fringes resolvable by conventional
   detectors; they are an important baseline for detector-resolution claims.
5. Dark-field correlation length depends on wavelength, geometry, and grating period. Multiple carrier
   radii could therefore encode more than one scale, but the exact single-shot dual-radius directional
   phase-grating hypothesis requires careful novelty review and experimental proof.
6. A three-direction hexagonal carrier set is mathematically sufficient for the three-parameter
   `D0/Dc/Ds` model. Related hexagonal directional patterns exist, especially in neutron and circular-cell
   work, so novelty must be claimed through the X-ray phase-mask implementation, system optimization,
   and demonstrated performance rather than symmetry alone.

## Recommended research ranking

| priority | candidate | reason |
|---:|---|---|
| 1 | H3 three-direction hexagonal phase carrier | highest probability of improving visibility while retaining rank-3 directional D |
| 2 | fabrication-aware corrected S4 | required reference and potentially publishable optimization baseline |
| 3 | H3 four-level | likely efficiency improvement if multi-depth fabrication is acceptable |
| 4 | circular-unit-cell/hex-packed baseline | strongest established omnidirectional comparator |
| 5 | random/blue-noise modulator | broadband and fabrication-robust fallback |
| 6 | DRH6 dual-radius hexagonal | highest novelty potential, but only after H3 is validated |
| 7 | modulated phase grating | comparator when detector resolution is the limiting constraint |

## Conclusion

The next design should not be selected by appearance or raw visibility. The best near-term hypothesis
is H3 because it removes one carrier while preserving directional-D rank. The strongest high-risk
hypothesis is DRH6 because it may add scale sensitivity, but it should be pursued only after the
single-radius physical model and fabrication constraints pass. A publishable claim requires a fair
comparison against circular-unit-cell, random-mask, and modulated-phase-grating baselines.
