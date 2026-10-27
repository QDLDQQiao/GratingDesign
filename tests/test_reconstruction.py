"""Tests for Phase II carrier demodulation and phase reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import DesignConfig
from phantoms import periodic_directional_dark_field, smooth_absorption, smooth_periodic_phase
from reconstruction import (
    ReconstructionConfig,
    align_phase_offset,
    fit_dark_field_anisotropy,
    reconstruct_sample,
    synthesize_reference,
    synthesize_sample,
)
from reconstruction_benchmark import BenchmarkConfig, benchmark_design_reconstruction


def test_phase_reconstruction_on_synthetic_zernike_phantom():
    design = DesignConfig(samples_per_supercell=64, n_supercells=2)
    config = ReconstructionConfig(
        dx_m=design.dx_m,
        wavelength_m=design.wavelength_m,
        z_eff_m=design.z_m,
        carriers=design.detector_carriers,
        samples_per_supercell=design.samples_per_supercell,
        n_supercells=design.n_supercells,
        guard_bins=3,
        visibility=0.2,
    )
    truth = smooth_periodic_phase(design.shape, amplitude_rad=0.04)
    absorption = smooth_absorption(design.shape)
    truth_t = np.exp(-absorption)
    directions = np.asarray(resulting_directions(config), dtype=float)
    truth_d = periodic_directional_dark_field(design.shape, directions)
    reference = synthesize_reference(config, design.shape)
    sample = synthesize_sample(config, truth, absorption=absorption, dark_field=truth_d)
    result = reconstruct_sample(sample, reference, config)
    aligned = align_phase_offset(result.phase, truth)
    rmse = np.sqrt(np.mean((aligned - truth) ** 2))
    assert rmse < 1e-4
    assert np.sqrt(np.mean((result.transmission - truth_t) ** 2)) < 2e-3
    d0_true, dani_true, dangle_true = fit_dark_field_anisotropy(truth_d, result.demodulation.carrier_vectors)
    valid = d0_true > np.max(d0_true) * 0.05
    assert np.sqrt(np.mean((result.dark_field_mean[valid] - d0_true[valid]) ** 2)) < 3e-3
    assert np.sqrt(np.mean((result.dark_field_anisotropy[valid] - dani_true[valid]) ** 2)) < 8e-2


def test_design_reconstruction_benchmark_reports_all_channels():
    design = DesignConfig(samples_per_supercell=64, n_supercells=2)
    from grating_design import default_candidate, simulate_design

    design_result = simulate_design(design, default_candidate())
    metrics = benchmark_design_reconstruction(design_result, BenchmarkConfig(photons_per_pixel=0.0))
    assert metrics.phase_rmse_rad < 1e-4
    assert metrics.transmission_rmse < 2e-3
    assert metrics.dark_field_mean_rmse < 3e-3
    assert metrics.dark_field_anisotropy_rmse < 8e-2
    assert metrics.score > 0.5


def resulting_directions(config):
    out = []
    for kx, ky in config.carriers:
        norm = (kx * kx + ky * ky) ** 0.5
        out.append((kx / norm, ky / norm))
    return out
