# Grating Pattern Experiment-Readiness Review

Date: 2026-07-13

This report combines a literature review with the repository's 10,000 photons/pixel, five-repeat pattern screen and its reconstruction-transfer analysis.

## Decision

Fabricate a small comparative X-ray coupon, not a full-area or publication-claim device. Use a corrected S4 star as the primary joint phase/transmission/directional-dark-field pattern. Include an independently optimized H3 binary and H3-4L pair, a hex-packed circular-unit-cell (CUC-hex) reference, and G2/C2 controls. S4 is the only all-channel baseline supported by the current screen. CUC-hex is the required low-risk directional-dark-field reference. The current H3 realization is not ready to fabricate. DRH6 needs a multiscale dark-field forward/retrieval model before fabrication. RBN and MPG need their own retrieval methods, so the carrier benchmark cannot rank them fairly.

Two-dimensional single-shot phase/dark-field acquisition is established. Novelty must therefore be a measured advantage in dose, accuracy, spatial resolution, tolerance, or detector requirement; it cannot be the presence of multiple carrier directions alone.

## What Was Simulated

Actual S4, H3, H3-4L, DRH6, FATO proxy, CUC square/hex, RBN, MPG proxy, G2, C2, and two-shot C2 masks were propagated with the repository's monochromatic scalar model. The configuration is `lambda=1.0e-10 m` (about 12.4 keV), `z=80 mm`, `112 um x 112 um` field, and `0.4375 um` sampling. Accuracy uses 10,000 photons/pixel and five noise repeats. For each carrier pattern, the physical simulation's mean visibility calibrates a separate synthetic carrier-demodulation reference/sample benchmark.

This is an idealized carrier-screening study, not an experimental-accuracy prediction. It omits source size/coherence, spectrum, detector pixel integration/MTF, material absorption, fabrication defects, and actual sample-through-grating propagation. The phase/T/D RMSE values are therefore not reconstructed from the propagated pattern images. RBN has no carrier channels, MPG is a proxy, and the two-shot C2 row has no two-image fusion reconstruction.

## Resolution: The Main Current Limitation

Noise-free sinusoidal phase, transmission, and isotropic-D transfer tests give:

| Channel | 50% transfer half-period |
|---|---:|
| Phase | 16.16 um |
| Transmission | 16.16 um |
| Mean dark field | 18.68 um |

Nearly every carrier layout gives the same result because the current reconstruction fixes a three-bin Fourier envelope, 5 px phase/T smoothing, and 16 px D smoothing. The envelope preserves field frequencies 1--3 and rejects the fourth. The values are pipeline bandwidth limits, not proof that patterns have the same intrinsic resolution; they do not include detector MTF, 2-D or orientation-dependent resolution, or directional-D resolution. They also miss the plan's `<=10 um` D-resolution gate.

![Resolution transfer](../results/experiment_readiness/pattern_resolution_transfer.png)

![Accuracy and D-resolution scorecard](../results/experiment_readiness/pattern_experiment_scorecard.png)

## Quantitative Screening at 10,000 Photons/Pixel

Lower error is better. Directional-D metrics are only meaningful for rank-3 patterns. Every carrier entry has the same phase/T/D half-period shown above, so it is omitted from the table.

| Pattern | Mean V | Phase RMSE (rad) | T RMSE | D0 RMSE | D anisotropy RMSE | D angle MAE | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| S4 | 0.195 | 0.00737 | 2.77e-4 | 9.04e-4 | 0.0550 | 3.81 deg | Best current all-channel baseline |
| H3 | 0.041 | 0.0511 | 2.79e-4 | 4.66e-3 | 0.359 | 20.37 deg | Not viable as currently designed |
| H3-4L | 0.069 | 0.0287 | 2.81e-4 | 2.78e-3 | 0.197 | 10.96 deg | Improvement, still below visibility gate |
| DRH6 | 0.181 | 0.00850 | 2.87e-4 | 8.76e-4 | 0.0445 | 3.17 deg | Promising, but not multiscale validated |
| FATO proxy | 0.195 | 0.00771 | 2.84e-4 | 9.15e-4 | 0.0990 | 7.30 deg | Not topology optimization evidence |
| CUC square | 0.354 | 0.00565 | 2.87e-4 | 7.59e-4 | n/a | n/a | Strong phase/T/isotropic-D control |
| CUC hex | 0.204 | 0.0107 | 2.85e-4 | 1.05e-3 | 0.100 | 9.02 deg | Required directional-D literature control |
| G2 grid | 0.405 | 0.00528 | 3.02e-4 | 6.66e-4 | n/a | n/a | Phase/T upper-bound control |
| C2 checkerboard | 0.414 | 0.00514 | 2.80e-4 | 6.95e-4 | n/a | n/a | Phase/T upper-bound control |
| RBN | n/a | n/a | n/a | n/a | n/a | n/a | Needs speckle retrieval |
| MPG proxy | 0.003 | 2.89 | 2.77e-4 | 8.12e-2 | n/a | n/a | Not a valid MPG benchmark |
| C2 two-shot | 0.410 | n/a | n/a | n/a | n/a | n/a | Needs two-image fusion |

G2/C2 have high visibility because modulation is concentrated into two carriers. They cannot identify the three coefficients of an in-plane directional-D tensor. DRH6 uses one scalar-D truth for all six carriers, so it does not demonstrate its intended two-correlation-length function.

## Pattern Conclusions

### S4

S4 has balanced contrast and the strongest implemented joint phase/T/directional-D result. Its four directions provide redundancy against calibration or channel loss. It should be the first fabricated comparison pattern, but is not yet a fabrication-ready design because material physics and detector/source effects are missing.

### H3 and H3-4L

Three approximately 0/60/120-degree axes give a full-rank, well-conditioned fit for `D(theta)=D0+Dc cos(2theta)+Ds sin(2theta)`. H3 is the strongest novelty hypothesis: fewer channels may preserve directional D while raising per-channel modulation. Current binary H3 does the opposite (`V=0.041` versus S4 `0.195`). H3-4L improves to `0.069`, but both fail the planned `V >= 0.15` requirement. Optimize integer vectors, phase levels, fill, and DRC before putting H3 on the coupon; do not fabricate the current realization at full scale.

### DRH6

DRH6 is close to S4 for ordinary phase/T/D0 and has slightly lower simulated directional-D error. That is not proof of multiscale D: separate scatterer sizes or correlation lengths are neither synthesized nor reconstructed. Hold DRH6 until it recovers two known scatterer populations with less than 15% error, D angle MAE below 7 degrees, and all six post-detector/spectrum channels above `V=0.12`.

### Circular cells, random masks, and MPG

CUC-hex is the correct external directional-D baseline because circular and hex-packed cells have direct single-shot X-ray omnidirectional scattering evidence. CUC square, G2, and C2 are useful phase/T/isotropic-D controls, not directional-D substitutes. Random masks are experimentally credible but trade local-window resolution, dose, frames, and calibration complexity; they need a speckle-specific retrieval. MPG is the practical fallback when detector pixels cannot resolve S4/H3/COC fringes, but its experimentally validated strength is phase/T/isotropic D rather than directional-D tensor recovery; the present MPG proxy is not physically meaningful.

## Literature Position

- Two-dimensional transmission gratings already demonstrate single-shot multidirectional phase and dark-field retrieval: [Wen et al., 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC3091831/) and [Zanette et al., 2010](https://doi.org/10.1103/PhysRevLett.105.248102).
- Directional single-grid dark field can recover scatter direction under noise, but source/detector blur and attenuation/phase edges create quantitative artifacts unless corrected: [Croughan et al., 2023](https://pubmed.ncbi.nlm.nih.gov/37155790/) and [Croughan et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39090344/).
- Circular phase gratings provide single-shot omnidirectional hard-X-ray sensitivity; hex packing is a direct reference: [Kagias et al., 2016](https://doi.org/10.1103/PhysRevLett.116.093902), [Kagias et al., 2017](https://doi.org/10.1088/1742-6596/849/1/012053), and [Kim et al., 2020](https://doi.org/10.1063/1.5145361).
- Random masks have phase/D evidence at high energy and with a laboratory microfocus source: [Wang et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4964655/) and [Wang and Sawhney, 2021](https://doi.org/10.1073/pnas.2022319118).
- MPG has microfocus-source phase/T/D evidence with directly resolvable envelopes, but uses phase stepping for higher-resolution quantification: [Meyer et al., 2024](https://doi.org/10.1038/s41598-024-78133-8).

No primary source found a common-dose, common-detector, equal-resolution comparison across S4, H3, CUC, random masks, and MPG. Treat this as a preregistered comparison opportunity, not a confirmed field-wide ranking.

## Required Gates Before X-ray Fabrication

1. Repair the material model: `h_pi` must be `lambda/(2*delta(E, material))` with beta-based absorption. The current `lambda/2` thickness is not a material pi-phase height.
2. Implement spectrum, source blur/coherence, pixel aperture, detector MTF/read noise, real mask absorption, grating CD/depth/sidewall errors, and actual sample-through-grating propagation.
3. Replace the transfer proxy with phase/T/D0/anisotropy/angle edge or point MTF. Sweep the carrier window and D regularization, then freeze the reconstruction before comparison.
4. Use independent phase, attenuation, isotropic-scatter, single-fibre, crossed-fibre, and attenuation-matched null phantoms. Match dose, spectrum, FOV, geometry, and effective resolution. Test centre and edge over three realignments.
5. Advance only candidates that meet target-frequency MTF >=0.20, phase/T calibration slopes 0.95--1.05, D slope 0.90--1.10, crosstalk <=10% of the smallest useful signal, and all gates with one-sided 95% confidence.

## Reproduce

```powershell
rtk python scripts/simulate_pattern_comparison.py --out results/pattern_simulation_research --photons-per-pixel 10000 --repeats 5
rtk python scripts/analyze_pattern_readiness.py --pattern-report results/pattern_simulation_research/pattern_simulation_report.json --out results/experiment_readiness
```

Key artifacts: `results/pattern_simulation_research/pattern_simulation_report.json`, `results/experiment_readiness/pattern_experiment_readiness_report.json`, `results/experiment_readiness/pattern_resolution_transfer.png`, and `results/experiment_readiness/pattern_experiment_scorecard.png`.

## Addendum: Optimized H3/H3-4L

The original H3 values above describe equal-weight, zero-phase layouts and should not be used as the final H3 comparison. A 160-candidate joint physical-pattern/reconstruction search optimized the three carrier amplitudes and phases, shortlisted by propagated visibility, and validated at the same 10,000 photons/pixel, five-repeat, 5 px phase/T and 16 px D settings as S4. H3-4L uses actual `0, pi/2, pi, 3pi/2` phase levels.

| Pattern | Min V | Phase RMSE | D0 RMSE | D anisotropy RMSE | D angle MAE |
|---|---:|---:|---:|---:|---:|
| S4 | 0.194 | 0.00737 rad | 9.04e-4 | 0.0550 | 3.81 deg |
| Optimized H3 | 0.405 | 0.00553 rad | 6.55e-4 | 0.0263 | 1.82 deg |
| Optimized H3-4L | 0.376 | 0.00556 rad | 7.12e-4 | 0.0277 | 2.11 deg |

Under this idealized, visibility-calibrated screen, H3 is the leading candidate for the matched S4/H3/CUC-hex coupon comparison: it reaches grid-level visibility while retaining rank-3 directional-D recovery. This supersedes the unoptimized-H3 screening conclusion, but does not remove the critical limitations above: material phase height, source/detector/spectrum physics, fabrication tolerance, and end-to-end sample-through-pattern dark-field modeling remain required before claiming experimental superiority.

Figures: `results/h3_optimization/h3_optimization_summary.png` and `results/h3_optimized_comparison/optimized_h3_comparison.png`.
