"""Tests for star/grid/checkerboard modality comparison."""

from __future__ import annotations

import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import DesignConfig
from modality_comparison import ComparisonConfig, default_modalities, evaluate_modality, reconstruct_modality_maps


def test_directional_dark_field_identifiability_rank():
    design = DesignConfig(samples_per_supercell=64, n_supercells=2)
    modalities = {item.name: item for item in default_modalities(design)}
    config = ComparisonConfig(photons_per_pixel=[0.0], repeats=1, visibility=0.2, smoothing_sigma_px=0.0)

    star = evaluate_modality(design, modalities["star_4dir_single"], config, photons_per_pixel=0.0, repeat=0)
    grid = evaluate_modality(design, modalities["grid_2dir_single"], config, photons_per_pixel=0.0, repeat=0)
    checker = evaluate_modality(design, modalities["checkerboard_2dir_single"], config, photons_per_pixel=0.0, repeat=0)

    assert star["directional_dark_field_identifiable"]
    assert star["dark_field_fit_rank"] == 3
    assert not grid["directional_dark_field_identifiable"]
    assert not checker["directional_dark_field_identifiable"]
    assert math.isnan(grid["dark_field_anisotropy_rmse"])
    assert math.isnan(checker["dark_field_angle_mae_deg"])


def test_recovered_maps_include_phase_t_and_d():
    design = DesignConfig(samples_per_supercell=64, n_supercells=2)
    modality = default_modalities(design)[0]
    config = ComparisonConfig(photons_per_pixel=[0.0], repeats=1, visibility=0.2, smoothing_sigma_px=0.0)
    maps = reconstruct_modality_maps(design, modality, config, photons_per_pixel=0.0)
    for key in [
        "phase_truth",
        "phase_reconstruction",
        "phase_error",
        "transmission_truth",
        "transmission_reconstruction",
        "transmission_error",
        "dark_field_mean_truth",
        "dark_field_mean_reconstruction",
        "dark_field_mean_error",
    ]:
        assert key in maps
        assert maps[key].shape == design.shape
