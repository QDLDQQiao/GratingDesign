"""Noise-aware reconstruction benchmark for grating candidate evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from grating_design import DesignConfig, DesignResult
from phantoms import periodic_directional_dark_field, smooth_absorption, smooth_periodic_phase
from reconstruction import (
    ReconstructionConfig,
    align_phase_offset,
    fit_dark_field_anisotropy,
    reconstruct_sample,
    synthesize_reference,
    synthesize_sample,
)


@dataclass
class BenchmarkConfig:
    photons_per_pixel: float = 0.0
    phase_amplitude_rad: float = 0.08
    absorption_peak_mu: float = 0.05
    dark_field_mean: float = 0.035
    dark_field_anisotropy: float = 0.35
    dark_field_angle_rad: float = 0.55
    guard_bins: int = 3
    smoothing_sigma_px: float = 0.0
    dark_field_smoothing_sigma_px: float | None = 0.0
    seed: int = 11


@dataclass
class BenchmarkMetrics:
    score: float
    phase_rmse_rad: float
    transmission_rmse: float
    dark_field_mean_rmse: float
    dark_field_anisotropy_rmse: float
    dark_field_angle_mae_deg: float
    mean_visibility: float
    min_visibility: float


def benchmark_design_reconstruction(
    design_result: DesignResult,
    benchmark: BenchmarkConfig | None = None,
) -> BenchmarkMetrics:
    """Evaluate phase, T, and D recovery using visibility from a design result."""

    if benchmark is None:
        benchmark = BenchmarkConfig()
    design = design_result.config
    recon_config = ReconstructionConfig(
        dx_m=design.dx_m,
        wavelength_m=design.wavelength_m,
        z_eff_m=design.z_m,
        carriers=design.detector_carriers,
        samples_per_supercell=design.samples_per_supercell,
        n_supercells=design.n_supercells,
        guard_bins=benchmark.guard_bins,
        visibility=design_result.metrics.mean_visibility,
        smoothing_sigma_px=benchmark.smoothing_sigma_px,
        dark_field_smoothing_sigma_px=benchmark.dark_field_smoothing_sigma_px,
    )
    shape = design.shape
    truth_phase = smooth_periodic_phase(shape, amplitude_rad=benchmark.phase_amplitude_rad)
    absorption = smooth_absorption(shape, peak_mu=benchmark.absorption_peak_mu)
    truth_t = np.exp(-absorption)
    directions = _directions(recon_config.carriers)
    truth_d = periodic_directional_dark_field(
        shape,
        directions,
        angle_rad=benchmark.dark_field_angle_rad,
        mean_strength=benchmark.dark_field_mean,
        anisotropy=benchmark.dark_field_anisotropy,
    )
    reference = synthesize_reference(recon_config, shape)
    sample = synthesize_sample(recon_config, truth_phase, absorption=absorption, dark_field=truth_d)
    if benchmark.photons_per_pixel > 0.0:
        rng = np.random.default_rng(benchmark.seed)
        reference = add_poisson_noise(reference, benchmark.photons_per_pixel, rng)
        sample = add_poisson_noise(sample, benchmark.photons_per_pixel, rng)
    result = reconstruct_sample(sample, reference, recon_config)
    phase = align_phase_offset(result.phase, truth_phase)
    phase_rmse = _rmse(phase, truth_phase)
    t_rmse = _rmse(result.transmission, truth_t)
    d0_true, dani_true, dangle_true = fit_dark_field_anisotropy(truth_d, result.demodulation.carrier_vectors)
    valid = d0_true > max(float(np.max(d0_true)) * 0.05, 1e-12)
    d0_rmse = _rmse(result.dark_field_mean[valid], d0_true[valid])
    dani_rmse = _rmse(result.dark_field_anisotropy[valid], dani_true[valid])
    dangle_mae = float(np.mean(np.abs(_wrapped_angle_error(result.dark_field_angle[valid], dangle_true[valid]))) * 180.0 / np.pi)
    score = reconstruction_score(
        phase_rmse,
        t_rmse,
        d0_rmse,
        dani_rmse,
        dangle_mae,
        design_result.metrics.min_visibility,
        design_result.metrics.balance_cv,
        design_result.metrics.contamination_mean,
    )
    return BenchmarkMetrics(
        score=score,
        phase_rmse_rad=phase_rmse,
        transmission_rmse=t_rmse,
        dark_field_mean_rmse=d0_rmse,
        dark_field_anisotropy_rmse=dani_rmse,
        dark_field_angle_mae_deg=dangle_mae,
        mean_visibility=design_result.metrics.mean_visibility,
        min_visibility=design_result.metrics.min_visibility,
    )


def reconstruction_score(
    phase_rmse_rad: float,
    transmission_rmse: float,
    dark_field_mean_rmse: float,
    dark_field_anisotropy_rmse: float,
    dark_field_angle_mae_deg: float,
    min_visibility: float = 1.0,
    balance_cv: float = 0.0,
    contamination_mean: float = 0.0,
) -> float:
    """Higher-is-better score from normalized inverse errors."""

    penalty = (
        phase_rmse_rad / 1.0e-3
        + transmission_rmse / 1.0e-3
        + dark_field_mean_rmse / 1.0e-3
        + dark_field_anisotropy_rmse / 5.0e-2
        + dark_field_angle_mae_deg / 2.0
        + max(0.0, 0.10 - min_visibility) / 0.02
        + balance_cv / 0.25
        + contamination_mean / 0.05
    )
    return float(1.0 / (1.0 + penalty))


def add_poisson_noise(image: np.ndarray, photons_per_pixel: float, rng: np.random.Generator) -> np.ndarray:
    """Apply Poisson noise while preserving the image scale."""

    clipped = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    counts = rng.poisson(clipped * photons_per_pixel)
    return counts.astype(np.float64) / photons_per_pixel


def metrics_to_dict(metrics: BenchmarkMetrics) -> dict:
    return asdict(metrics)


def _directions(carriers: Sequence[tuple[int, int]]) -> np.ndarray:
    out = []
    for kx, ky in carriers:
        norm = math.hypot(kx, ky)
        out.append((kx / norm, ky / norm))
    return np.asarray(out, dtype=np.float64)


def _rmse(actual: np.ndarray, truth: np.ndarray) -> float:
    error = np.asarray(actual, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def _wrapped_angle_error(angle: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return 0.5 * np.angle(np.exp(2j * (angle - truth)))
