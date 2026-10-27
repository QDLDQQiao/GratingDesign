"""Monochromatic X-ray propagation utilities.

All lengths are in meters. Two-dimensional arrays use axis 0 for y and
axis 1 for x. Frequency grids are built directly from unshifted fftfreq
arrays so they match numpy.fft.fft2 without any shift bookkeeping.
"""

from __future__ import annotations

import numpy as np


def propagate(
    U0: np.ndarray,
    dx: float,
    wavelength: float,
    z: float,
    kernel: str = "asm",
    bandlimit: bool = False,
) -> np.ndarray:
    """Propagate a 2D scalar complex field by angular spectrum or Fresnel FFT.

    Parameters
    ----------
    U0:
        Complex input field with shape (Ny, Nx).
    dx:
        Square-grid sample spacing in meters.
    wavelength:
        Wavelength in meters.
    z:
        Propagation distance in meters.
    kernel:
        ``"asm"`` for the full angular spectrum kernel or ``"fresnel"`` for
        the paraxial Fresnel kernel. The Fresnel kernel omits the global
        ``exp(i*k*z)`` phase because intensity tests are phase-insensitive.
    bandlimit:
        Apply a conservative Matsushima-style rectangular band limit. The
        validation suite intentionally leaves this disabled.
    """

    field = np.asarray(U0)
    if field.ndim != 2:
        raise ValueError("U0 must be a two-dimensional array")
    if not np.iscomplexobj(field):
        field = field.astype(np.complex128)
    elif field.dtype != np.complex128:
        field = field.astype(np.complex128)
    if dx <= 0:
        raise ValueError("dx must be positive")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")

    ny, nx = field.shape
    fy = np.fft.fftfreq(ny, d=dx)
    fx = np.fft.fftfreq(nx, d=dx)
    fxx, fyy = np.meshgrid(fx, fy)
    f2 = fxx * fxx + fyy * fyy

    kernel_name = kernel.lower()
    if kernel_name == "asm":
        lam2_f2 = (wavelength * wavelength) * f2
        propagating = lam2_f2 <= 1.0
        root = np.zeros_like(f2, dtype=np.float64)
        root[propagating] = np.sqrt(1.0 - lam2_f2[propagating])
        H = np.zeros_like(f2, dtype=np.complex128)
        H[propagating] = np.exp(1j * (2.0 * np.pi / wavelength) * z * root[propagating])
    elif kernel_name == "fresnel":
        H = np.exp(-1j * np.pi * wavelength * z * f2)
    else:
        raise ValueError("kernel must be 'asm' or 'fresnel'")

    if bandlimit:
        H = H * _bandlimit_mask(nx, ny, dx, wavelength, z, fx, fy)

    return np.fft.ifft2(np.fft.fft2(field) * H).astype(np.complex128, copy=False)


def _bandlimit_mask(
    nx: int,
    ny: int,
    dx: float,
    wavelength: float,
    z: float,
    fx: np.ndarray,
    fy: np.ndarray,
) -> np.ndarray:
    """Return a conservative rectangular angular-spectrum band-limit mask."""

    if z == 0:
        return np.ones((ny, nx), dtype=bool)
    lx = nx * dx
    ly = ny * dx
    ax = 1.0 / np.sqrt(1.0 + (2.0 * z / lx) ** 2) / wavelength
    ay = 1.0 / np.sqrt(1.0 + (2.0 * z / ly) ** 2) / wavelength
    return (np.abs(fx)[None, :] <= ax) & (np.abs(fy)[:, None] <= ay)


def make_pi_grating(
    Ny: int,
    Nx: int,
    dx: float,
    p: float,
    phase: float = np.pi,
) -> np.ndarray:
    """Create a 1D binary phase grating along x, tiled uniformly along y."""

    mask = _half_period_mask(Nx, dx, p)
    line = np.ones(Nx, dtype=np.complex128)
    line[mask] = np.exp(1j * phase)
    return np.broadcast_to(line, (Ny, Nx)).copy()


def make_amp_grating(Ny: int, Nx: int, dx: float, p: float) -> np.ndarray:
    """Create a 1D 50% duty-cycle binary absorption grating along x."""

    mask = _half_period_mask(Nx, dx, p)
    line = np.zeros(Nx, dtype=np.complex128)
    line[mask] = 1.0 + 0.0j
    return np.broadcast_to(line, (Ny, Nx)).copy()


def _half_period_mask(Nx: int, dx: float, p: float) -> np.ndarray:
    if Nx <= 0:
        raise ValueError("Nx must be positive")
    if dx <= 0:
        raise ValueError("dx must be positive")
    if p <= 0:
        raise ValueError("p must be positive")

    samples_per_period = p / dx
    rounded = int(round(samples_per_period))
    if rounded <= 0 or not np.isclose(samples_per_period, rounded, rtol=0.0, atol=1e-9):
        raise ValueError("p/dx must be an integer for exact 50% grating construction")
    if rounded % 2 != 0:
        raise ValueError("p/dx must be even for exact 50% duty cycle")

    idx = np.arange(Nx) - (Nx // 2)
    phase_index = np.mod(idx, rounded)
    return phase_index < (rounded // 2)


def fourier_visibility(I: np.ndarray, dx: float, f_target: float) -> float:
    """Measure Fourier visibility at a grid-aligned spatial frequency.

    The intensity is averaged along y first. Visibility is defined as
    ``V = 2*abs(I_hat(f_target))/abs(I_hat(0))``.
    """

    intensity = np.asarray(I, dtype=np.float64)
    if intensity.ndim != 2:
        raise ValueError("I must be a two-dimensional intensity array")
    if dx <= 0:
        raise ValueError("dx must be positive")

    profile = intensity.mean(axis=0)
    freq = np.fft.fftfreq(profile.size, d=dx)
    df = abs(freq[1] - freq[0]) if profile.size > 1 else np.inf
    matches = np.flatnonzero(np.isclose(freq, f_target, rtol=0.0, atol=1e-6 * df))
    if matches.size != 1:
        raise ValueError("f_target must land exactly on the FFT frequency grid")

    spectrum = np.fft.fft(profile) / profile.size
    dc_index = int(np.flatnonzero(np.isclose(freq, 0.0, rtol=0.0, atol=1e-12))[0])
    dc = abs(spectrum[dc_index])
    if dc == 0.0:
        raise ValueError("zero DC intensity prevents visibility normalization")
    return float(2.0 * abs(spectrum[int(matches[0])]) / dc)
