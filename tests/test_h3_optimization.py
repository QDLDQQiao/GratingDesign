"""Focused tests for the H3 joint-candidate scoring functions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from h3_optimization import _joint_score, _physical_score


def test_joint_score_rewards_better_phase_and_directional_dark_field():
    row = {"physical_score": 1.0}
    stronger = {
        "phase_rmse_rad": 0.005,
        "transmission_rmse": 0.0003,
        "dark_field_mean_rmse": 0.0007,
        "dark_field_anisotropy_rmse": 0.03,
        "dark_field_angle_mae_deg": 2.0,
    }
    weaker = {
        "phase_rmse_rad": 0.03,
        "transmission_rmse": 0.001,
        "dark_field_mean_rmse": 0.005,
        "dark_field_anisotropy_rmse": 0.3,
        "dark_field_angle_mae_deg": 20.0,
    }
    assert _joint_score(row, stronger) > _joint_score(row, weaker)


def test_physical_score_rewards_visibility_and_penalizes_imbalance():
    strong = {"min_visibility": 0.3, "mean_visibility": 0.35, "balance_cv": 0.02, "contamination_mean": 0.0}
    weak = {"min_visibility": 0.1, "mean_visibility": 0.35, "balance_cv": 0.20, "contamination_mean": 0.04}
    assert _physical_score(strong) > _physical_score(weak)
