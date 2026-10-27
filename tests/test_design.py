"""Smoke tests for the grating design optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import DesignConfig, default_candidate, optimize_design, simulate_design


def test_default_design_scores_detector_harmonics():
    config = DesignConfig(samples_per_supercell=64, n_supercells=2)
    result = simulate_design(config, default_candidate())
    assert result.metrics.mean_visibility > 0.18
    assert result.metrics.balance_cv < 0.05
    assert result.metrics.contamination_mean < 0.08
    assert result.metrics.phase_noise_factor > 0.0
    assert result.metrics.dark_field_noise_factor > 0.0
    assert abs(result.metrics.fill_factor - 0.5) < 1e-12


def test_optimizer_returns_auditable_history():
    config = DesignConfig(samples_per_supercell=64, n_supercells=2)
    result, history = optimize_design(config, iterations=5, seed=3)
    assert len(history) == 5
    assert result.metrics.score == max(row["score"] for row in history)


def test_parallel_optimizer_matches_serial_candidate():
    config = DesignConfig(samples_per_supercell=32, n_supercells=2)
    serial_result, serial_history = optimize_design(config, iterations=4, seed=11, n_cores=1)
    parallel_result, parallel_history = optimize_design(config, iterations=4, seed=11, n_cores=2)
    assert parallel_result.candidate == serial_result.candidate
    assert [row["score"] for row in parallel_history] == [row["score"] for row in serial_history]
