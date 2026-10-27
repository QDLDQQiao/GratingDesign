"""Tests for spatial-transfer summary calculations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_readiness import _half_period_at_50


def test_half_period_at_50_interpolates_transfer_crossing():
    resolution = _half_period_at_50([0.9, 0.6, 0.2], [1, 2, 3], field_um=120.0)
    assert resolution is not None
    assert abs(resolution - (120.0 / 4.5)) < 1e-12


def test_half_period_at_50_returns_none_without_cutoff():
    assert _half_period_at_50([0.9, 0.7, 0.6], [1, 2, 3], field_um=120.0) is None
