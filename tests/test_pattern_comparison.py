"""Tests for physical grating-pattern generation and comparison."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import Candidate, DesignConfig
from pattern_comparison import default_patterns, simulate_pattern


def test_patterns_generate_binary_masks_and_carrier_metrics():
    config = DesignConfig(samples_per_supercell=64, n_supercells=2)
    simulations = [simulate_pattern(config, pattern) for pattern in default_patterns(config)]
    names = [item["pattern"]["name"] for item in simulations]
    assert {"S4", "H3", "H3_4L", "DRH6", "FATO_proxy", "CUC_square", "CUC_hex", "RBN", "MPG_proxy", "G2_grid", "C2_checkerboard", "C2_4dir_2shot"} == set(names)
    for item in simulations:
        assert item["mask"].shape == config.shape
        if item["pattern"]["implementation"] in {"binary_carrier", "rounded_carrier", "blue_noise", "modulated_phase"}:
            assert abs(float(item["mask"].mean()) - config.fill_factor) < 1e-12
        else:
            assert abs(float(item["mask"].mean()) - config.fill_factor) < 0.02
        if item["pattern"]["carrier_reconstruction"]:
            assert item["min_visibility"] > 0.0
        else:
            assert item["min_visibility"] is None
        assert len(item["visibility"]) == len(item["pattern"]["carriers"])


def test_h3_candidate_and_four_level_phase_map_are_applied():
    config = DesignConfig(samples_per_supercell=64, n_supercells=2)
    patterns = {item.name: item for item in default_patterns(config)}
    candidate = Candidate(amplitudes=[1.4, 0.7, 1.1], phases_rad=[0.3, -0.8, 1.2])
    default_h3 = simulate_pattern(config, patterns["H3"])
    optimized_h3 = simulate_pattern(config, patterns["H3"], candidate)
    assert not (default_h3["phase_map"] == optimized_h3["phase_map"]).all()
    h3_4l = simulate_pattern(config, patterns["H3_4L"], candidate)
    assert len(set(h3_4l["phase_map"].ravel())) == 4
