"""Analytic phantom generators for reconstruction validation."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def zernike_like_phase(shape: Tuple[int, int], amplitude_rad: float = 0.08) -> np.ndarray:
    """Return a smooth phase phantom resembling defocus, astigmatism, and coma."""

    ny, nx = shape
    y = np.linspace(-1.0, 1.0, ny, endpoint=False)
    x = np.linspace(-1.0, 1.0, nx, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx * xx + yy * yy)
    aperture = rr <= 0.92
    defocus = 2.0 * rr * rr - 1.0
    astig = xx * xx - yy * yy
    coma = (3.0 * rr * rr - 2.0) * xx
    phase = amplitude_rad * (0.65 * defocus + 0.45 * astig + 0.30 * coma)
    phase = phase * aperture
    phase -= phase.mean()
    return phase.astype(np.float64)


def smooth_periodic_phase(shape: Tuple[int, int], amplitude_rad: float = 0.04) -> np.ndarray:
    """Band-limited periodic phase phantom for quantitative demodulation tests."""

    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) / ny
    x = np.arange(nx, dtype=np.float64) / nx
    xx, yy = np.meshgrid(x, y)
    phase = (
        0.70 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
        + 0.25 * np.cos(4.0 * np.pi * xx + 0.4)
        + 0.18 * np.sin(2.0 * np.pi * (xx + yy))
    )
    phase -= phase.mean()
    phase /= max(float(np.max(np.abs(phase))), 1e-12)
    return amplitude_rad * phase


def smooth_absorption(shape: Tuple[int, int], peak_mu: float = 0.05) -> np.ndarray:
    """Band-limited absorption phantom; transmission is exp(-mu)."""

    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) / ny
    x = np.arange(nx, dtype=np.float64) / nx
    xx, yy = np.meshgrid(x, y)
    mu = 0.55 + 0.25 * np.sin(2.0 * np.pi * xx - 0.2) + 0.20 * np.cos(2.0 * np.pi * yy + 0.3)
    mu += 0.12 * np.sin(2.0 * np.pi * (xx - yy))
    mu -= mu.min()
    mu /= max(float(mu.max()), 1e-12)
    return peak_mu * mu


def anisotropic_dark_field(shape: Tuple[int, int], angle_rad: float = 0.6, strength: float = 0.04) -> np.ndarray:
    """Smooth scalar dark-field strength map for synthetic image generation."""

    ny, nx = shape
    y = np.linspace(-1.0, 1.0, ny, endpoint=False)
    x = np.linspace(-1.0, 1.0, nx, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    rotated = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
    envelope = np.exp(-(rotated / 0.45) ** 2)
    return strength * envelope


def directional_dark_field(
    shape: Tuple[int, int],
    directions: np.ndarray,
    angle_rad: float = 0.6,
    mean_strength: float = 0.035,
    anisotropy: float = 0.45,
) -> np.ndarray:
    """Return per-direction dark-field maps following a cos(2 theta) model."""

    base = anisotropic_dark_field(shape, angle_rad=angle_rad, strength=mean_strength)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    channels = []
    for theta in angles:
        directional_factor = 1.0 + anisotropy * np.cos(2.0 * (theta - angle_rad))
        channels.append(np.maximum(base * directional_factor, 0.0))
    return np.asarray(channels, dtype=np.float64)


def periodic_directional_dark_field(
    shape: Tuple[int, int],
    directions: np.ndarray,
    angle_rad: float = 0.55,
    mean_strength: float = 0.035,
    anisotropy: float = 0.35,
) -> np.ndarray:
    """Low-frequency directional dark-field phantom for recovery metrics."""

    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) / ny
    x = np.arange(nx, dtype=np.float64) / nx
    xx, yy = np.meshgrid(x, y)
    base = mean_strength * (0.65 + 0.20 * np.sin(2.0 * np.pi * xx) + 0.15 * np.cos(2.0 * np.pi * yy))
    base = np.maximum(base, mean_strength * 0.2)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    channels = []
    for theta in angles:
        directional_factor = 1.0 + anisotropy * np.cos(2.0 * (theta - angle_rad))
        channels.append(np.maximum(base * directional_factor, 0.0))
    return np.asarray(channels, dtype=np.float64)
