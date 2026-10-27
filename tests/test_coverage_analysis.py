"""Tests for energy-distance coverage comparison."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverage_analysis import CoverageConfig, build_pattern_mask, default_coverage_patterns, run_coverage_comparison
from grating_design import DesignConfig


def test_image_size_override_and_pattern_mask():
    config = DesignConfig(samples_per_supercell=32, image_size_px=96)
    assert config.shape == (96, 96)
    assert config.effective_n_supercells == 3
    mask = build_pattern_mask(config, default_coverage_patterns(config)[0].mask_carriers)
    assert mask.shape == config.shape
    assert abs(mask.mean() - config.fill_factor) < 0.01


def test_coverage_comparison_smoke(tmp_path):
    config = DesignConfig(samples_per_supercell=32, image_size_px=64, z_m=0.3)
    coverage = CoverageConfig(energies_keV=[12.4], distances_m=[0.3], visibility_threshold=0.0)
    report = run_coverage_comparison(config, coverage, tmp_path)
    assert len(report["rows"]) == 4
    assert (tmp_path / "coverage_comparison.csv").exists()
