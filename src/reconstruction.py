"""Carrier demodulation and phase/dark-field reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


Carrier = Tuple[int, int]


@dataclass
class ReconstructionConfig:
    dx_m: float
    wavelength_m: float
    z_eff_m: float
    carriers: Sequence[Carrier]
    samples_per_supercell: int
    n_supercells: int = 2
    guard_bins: int = 3
    visibility: float = 0.18
    mean_intensity: float = 1.0
    smoothing_sigma_px: float = 0.0
    dark_field_smoothing_sigma_px: float | None = None
    window_sigma_bins: float = 0.0
    adaptive_channel_weights: bool = True


@dataclass
class DemodulationResult:
    absorption: np.ndarray
    phase_shift: np.ndarray
    directional_gradient: np.ndarray
    dark_field: np.ndarray
    visibility: np.ndarray
    carrier_vectors: np.ndarray


@dataclass
class ReconstructionResult:
    phase: np.ndarray
    transmission: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    dark_field_mean: np.ndarray
    dark_field_anisotropy: np.ndarray
    dark_field_angle: np.ndarray
    demodulation: DemodulationResult


def synthesize_reference(config: ReconstructionConfig, shape: Tuple[int, int]) -> np.ndarray:
    """Create a carrier-fringe reference image from detector carrier bins."""

    yy, xx = _normalized_grid(shape)
    image = np.full(shape, config.mean_intensity, dtype=np.float64)
    for kx, ky in config.carriers:
        phase = 2.0 * np.pi * config.n_supercells * (kx * xx + ky * yy)
        image += config.mean_intensity * config.visibility * np.cos(phase)
    floor = float(image.min())
    if floor <= 0.0:
        image = image + (abs(floor) + 1e-6)
    return image


def synthesize_sample(
    config: ReconstructionConfig,
    phase: np.ndarray,
    absorption: np.ndarray | None = None,
    dark_field: np.ndarray | None = None,
) -> np.ndarray:
    """Create a sample image with carrier phase shifts from a known phase map."""

    shape = phase.shape
    yy, xx = _normalized_grid(shape)
    grad_y, grad_x = np.gradient(phase, config.dx_m, edge_order=2)
    if absorption is None:
        absorption = np.zeros(shape, dtype=np.float64)
    if dark_field is None:
        dark_field = np.zeros((len(config.carriers),) + shape, dtype=np.float64)
    else:
        dark_field = np.asarray(dark_field, dtype=np.float64)
        if dark_field.ndim == 2:
            dark_field = np.broadcast_to(dark_field, (len(config.carriers),) + shape)
        if dark_field.shape != (len(config.carriers),) + shape:
            raise ValueError("dark_field must have shape (Ny, Nx) or (n_carriers, Ny, Nx)")

    image = config.mean_intensity * np.exp(-absorption)
    for j, (kx, ky) in enumerate(config.carriers):
        f_abs = _carrier_frequency_abs(config, (kx, ky))
        nx, ny = _carrier_unit((kx, ky))
        directional_gradient = nx * grad_x + ny * grad_y
        delta_psi = config.wavelength_m * config.z_eff_m * f_abs * directional_gradient
        carrier_phase = 2.0 * np.pi * config.n_supercells * (kx * xx + ky * yy)
        image += (
            config.mean_intensity
            * config.visibility
            * np.exp(-absorption)
            * np.exp(-dark_field[j])
            * np.cos(carrier_phase + delta_psi)
        )
    floor = float(image.min())
    if floor <= 0.0:
        image = image + (abs(floor) + 1e-6)
    return image


def demodulate_carriers(
    sample: np.ndarray,
    reference: np.ndarray,
    config: ReconstructionConfig,
) -> DemodulationResult:
    """Extract absorption, DPC phase shifts, and dark-field per carrier."""

    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if sample.shape != reference.shape:
        raise ValueError("sample and reference images must have the same shape")

    a0_s = _extract_envelope(sample, (0, 0), config.guard_bins, config.window_sigma_bins)
    a0_r = _extract_envelope(reference, (0, 0), config.guard_bins, config.window_sigma_bins)
    eps = 1e-12
    absorption = -np.log((np.abs(a0_s) + eps) / (np.abs(a0_r) + eps))

    phase_shift = []
    directional_gradient = []
    dark_field = []
    visibility = []
    carrier_vectors = []
    for carrier in config.carriers:
        index = _carrier_to_fft_index(sample.shape, carrier, config.n_supercells)
        env_s = _extract_envelope(sample, index, config.guard_bins, config.window_sigma_bins)
        env_r = _extract_envelope(reference, index, config.guard_bins, config.window_sigma_bins)
        ratio = env_s / (env_r + eps)
        dpsi = np.angle(ratio)
        f_abs = _carrier_frequency_abs(config, carrier)
        directional_gradient.append(dpsi / (config.wavelength_m * config.z_eff_m * f_abs))
        phase_shift.append(dpsi)
        dark_field.append(-np.log(((np.abs(env_s) + eps) / (np.abs(a0_s) + eps)) / ((np.abs(env_r) + eps) / (np.abs(a0_r) + eps))))
        visibility.append(2.0 * np.abs(env_s) / (np.abs(a0_s) + eps))
        carrier_vectors.append(_carrier_unit(carrier))

    phase_shift_arr = np.asarray(phase_shift)
    gradient_arr = np.asarray(directional_gradient)
    dark_arr = np.asarray(dark_field)
    visibility_arr = np.asarray(visibility)
    if config.smoothing_sigma_px > 0.0:
        absorption = _gaussian_smooth_fft(absorption, config.smoothing_sigma_px)
        gradient_arr = np.asarray([_gaussian_smooth_fft(channel, config.smoothing_sigma_px) for channel in gradient_arr])
        visibility_arr = np.asarray([_gaussian_smooth_fft(channel, config.smoothing_sigma_px) for channel in visibility_arr])
    dark_sigma = config.dark_field_smoothing_sigma_px
    if dark_sigma is None:
        dark_sigma = config.smoothing_sigma_px
    if dark_sigma > 0.0:
        dark_arr = np.asarray([_gaussian_smooth_fft(channel, dark_sigma) for channel in dark_arr])

    return DemodulationResult(
        absorption=absorption,
        phase_shift=phase_shift_arr,
        directional_gradient=gradient_arr,
        dark_field=dark_arr,
        visibility=visibility_arr,
        carrier_vectors=np.asarray(carrier_vectors, dtype=np.float64),
    )


def reconstruct_phase(
    demod: DemodulationResult,
    dx_m: float,
    regularization: float = 1e-12,
    adaptive_channel_weights: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse directional gradients and integrate phase with a Fourier closed form."""

    directions = demod.carrier_vectors
    gradients = demod.directional_gradient
    if adaptive_channel_weights:
        weights = np.maximum(np.nan_to_num(demod.visibility, nan=0.0), 1e-6) ** 2
    else:
        weights = np.maximum(np.nan_to_num(demod.visibility.mean(axis=(1, 2)), nan=0.0), 1e-6) ** 2
    grad_x, grad_y = fuse_gradients(gradients, directions, weights)
    phase = integrate_gradients_fourier(grad_x, grad_y, dx_m, regularization=regularization)
    return phase, grad_x, grad_y


def fuse_gradients(gradients: np.ndarray, directions: np.ndarray, weights: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares fusion from directional slopes to x/y slopes."""

    n_channels, ny, nx = gradients.shape
    if weights is None:
        weights = np.ones(n_channels, dtype=np.float64)
    a = np.asarray(directions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        normal = a.T @ (w[:, None] * a)
        inv = np.linalg.pinv(normal)
        rhs_x = np.zeros((ny, nx), dtype=np.float64)
        rhs_y = np.zeros((ny, nx), dtype=np.float64)
        for j in range(n_channels):
            rhs_x += w[j] * a[j, 0] * gradients[j]
            rhs_y += w[j] * a[j, 1] * gradients[j]
        grad_x = inv[0, 0] * rhs_x + inv[0, 1] * rhs_y
        grad_y = inv[1, 0] * rhs_x + inv[1, 1] * rhs_y
    elif w.shape == gradients.shape:
        nxx = np.sum(w * a[:, 0, None, None] ** 2, axis=0)
        nyy = np.sum(w * a[:, 1, None, None] ** 2, axis=0)
        nxy = np.sum(w * a[:, 0, None, None] * a[:, 1, None, None], axis=0)
        rhs_x = np.sum(w * a[:, 0, None, None] * gradients, axis=0)
        rhs_y = np.sum(w * a[:, 1, None, None] * gradients, axis=0)
        det = np.maximum(nxx * nyy - nxy * nxy, 1e-12)
        grad_x = (nyy * rhs_x - nxy * rhs_y) / det
        grad_y = (-nxy * rhs_x + nxx * rhs_y) / det
    else:
        raise ValueError("weights must have shape (n_channels,) or match gradients")
    return grad_x, grad_y


def integrate_gradients_fourier(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    dx_m: float,
    regularization: float = 1e-12,
) -> np.ndarray:
    """Solve min_phi ||dphi/dx-gx||^2 + ||dphi/dy-gy||^2 in Fourier space."""

    ny, nx = grad_x.shape
    fy = np.fft.fftfreq(ny, d=dx_m)
    fx = np.fft.fftfreq(nx, d=dx_m)
    fxx, fyy = np.meshgrid(fx, fy)
    qx = 2.0 * np.pi * fxx
    qy = 2.0 * np.pi * fyy
    gx_hat = np.fft.fft2(grad_x)
    gy_hat = np.fft.fft2(grad_y)
    denom = qx * qx + qy * qy + regularization
    phi_hat = (-1j * qx * gx_hat - 1j * qy * gy_hat) / denom
    phi_hat[0, 0] = 0.0
    return np.fft.ifft2(phi_hat).real


def fit_dark_field_anisotropy(dark_field: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit D(theta)=D0+Dc*cos(2theta)+Ds*sin(2theta) per pixel."""

    angles = np.arctan2(directions[:, 1], directions[:, 0])
    design = np.stack([np.ones_like(angles), np.cos(2.0 * angles), np.sin(2.0 * angles)], axis=1)
    coeff = np.linalg.pinv(design) @ dark_field.reshape(dark_field.shape[0], -1)
    d0 = coeff[0].reshape(dark_field.shape[1:])
    dc = coeff[1].reshape(dark_field.shape[1:])
    ds = coeff[2].reshape(dark_field.shape[1:])
    signal_floor = max(float(np.nanmax(np.abs(d0))) * 1e-3, 1e-12)
    anisotropy = np.zeros_like(d0)
    valid = np.abs(d0) >= signal_floor
    anisotropy[valid] = np.sqrt(dc[valid] * dc[valid] + ds[valid] * ds[valid]) / np.abs(d0[valid])
    angle = 0.5 * np.arctan2(ds, dc)
    return d0, anisotropy, angle


def reconstruct_sample(sample: np.ndarray, reference: np.ndarray, config: ReconstructionConfig) -> ReconstructionResult:
    demod = demodulate_carriers(sample, reference, config)
    phase, grad_x, grad_y = reconstruct_phase(
        demod,
        config.dx_m,
        adaptive_channel_weights=config.adaptive_channel_weights,
    )
    d0, dani, dangle = fit_dark_field_anisotropy(demod.dark_field, demod.carrier_vectors)
    transmission = np.exp(-demod.absorption)
    return ReconstructionResult(phase, transmission, grad_x, grad_y, d0, dani, dangle, demod)


def save_reconstruction_report(
    result: ReconstructionResult,
    truth_phase: np.ndarray,
    config: ReconstructionConfig,
    out_dir: Path,
    truth_transmission: np.ndarray | None = None,
    truth_dark_field: np.ndarray | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_aligned = align_phase_offset(result.phase, truth_phase)
    error = phase_aligned - truth_phase
    metrics = {
        "phase_rmse_rad": float(np.sqrt(np.mean(error * error))),
        "phase_mae_rad": float(np.mean(np.abs(error))),
        "gradient_x_rmse": float(np.sqrt(np.mean((result.grad_x - np.gradient(truth_phase, config.dx_m, axis=1, edge_order=2)) ** 2))),
        "gradient_y_rmse": float(np.sqrt(np.mean((result.grad_y - np.gradient(truth_phase, config.dx_m, axis=0, edge_order=2)) ** 2))),
    }
    if truth_transmission is not None:
        t_err = result.transmission - truth_transmission
        metrics["transmission_rmse"] = float(np.sqrt(np.mean(t_err * t_err)))
        metrics["transmission_mae"] = float(np.mean(np.abs(t_err)))
    if truth_dark_field is not None:
        d0_true, dani_true, dangle_true = fit_dark_field_anisotropy(truth_dark_field, result.demodulation.carrier_vectors)
        d0_err = result.dark_field_mean - d0_true
        dani_err = result.dark_field_anisotropy - dani_true
        angle_err = _wrapped_angle_error(result.dark_field_angle, dangle_true)
        valid = d0_true > max(float(np.max(d0_true)) * 0.05, 1e-12)
        metrics["dark_field_mean_rmse"] = float(np.sqrt(np.mean(d0_err[valid] * d0_err[valid])))
        metrics["dark_field_anisotropy_rmse"] = float(np.sqrt(np.mean(dani_err[valid] * dani_err[valid])))
        metrics["dark_field_angle_mae_deg"] = float(np.mean(np.abs(angle_err[valid])) * 180.0 / np.pi)

    report = {
        "config": asdict(config),
        "metrics": metrics,
        "files": {
            "phase_reconstruction": "phase_reconstruction.npy",
            "phase_truth": "phase_truth.npy",
            "phase_error": "phase_error.npy",
            "transmission_reconstruction": "transmission_reconstruction.npy",
            "dark_field_mean": "dark_field_mean.npy",
            "dark_field_anisotropy": "dark_field_anisotropy.npy",
            "summary_plot": "reconstruction_summary.png",
        },
    }
    np.save(out_dir / "phase_reconstruction.npy", phase_aligned)
    np.save(out_dir / "phase_truth.npy", truth_phase)
    np.save(out_dir / "phase_error.npy", error)
    np.save(out_dir / "transmission_reconstruction.npy", result.transmission)
    np.save(out_dir / "dark_field_mean.npy", result.dark_field_mean)
    np.save(out_dir / "dark_field_anisotropy.npy", result.dark_field_anisotropy)
    if truth_transmission is not None:
        np.save(out_dir / "transmission_truth.npy", truth_transmission)
        np.save(out_dir / "transmission_error.npy", result.transmission - truth_transmission)
        report["files"]["transmission_truth"] = "transmission_truth.npy"
        report["files"]["transmission_error"] = "transmission_error.npy"
    if truth_dark_field is not None:
        d0_true, _, _ = fit_dark_field_anisotropy(truth_dark_field, result.demodulation.carrier_vectors)
        np.save(out_dir / "dark_field_truth.npy", truth_dark_field)
        np.save(out_dir / "dark_field_mean_truth.npy", d0_true)
        np.save(out_dir / "dark_field_mean_error.npy", result.dark_field_mean - d0_true)
        report["files"]["dark_field_truth"] = "dark_field_truth.npy"
        report["files"]["dark_field_mean_truth"] = "dark_field_mean_truth.npy"
        report["files"]["dark_field_mean_error"] = "dark_field_mean_error.npy"
    with (out_dir / "reconstruction_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def align_phase_offset(phase: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return phase - np.mean(phase - truth)


def _wrapped_angle_error(angle: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return 0.5 * np.angle(np.exp(2j * (angle - truth)))


def _normalized_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) / ny
    x = np.arange(nx, dtype=np.float64) / nx
    return np.meshgrid(x, y)[::-1]


def _carrier_unit(carrier: Carrier) -> Tuple[float, float]:
    kx, ky = carrier
    norm = math.hypot(kx, ky)
    if norm == 0:
        raise ValueError("zero carrier has no direction")
    return kx / norm, ky / norm


def _carrier_frequency_abs(config: ReconstructionConfig, carrier: Carrier) -> float:
    kx, ky = carrier
    n = math.hypot(kx, ky)
    period_m = config.dx_m * config.samples_per_supercell
    return n / period_m


def _carrier_to_fft_index(shape: Tuple[int, int], carrier: Carrier, n_supercells: int) -> Tuple[int, int]:
    ny, nx = shape
    kx, ky = carrier
    return (ky * n_supercells) % ny, (kx * n_supercells) % nx


def _extract_envelope(image: np.ndarray, index: Tuple[int, int], radius: int, sigma_bins: float = 0.0) -> np.ndarray:
    spectrum = np.fft.fft2(image)
    mask = np.zeros_like(spectrum, dtype=np.float64)
    iy, ix = index
    ny, nx = image.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance2 = dy * dy + dx * dx
            if distance2 <= radius * radius:
                value = 1.0
                if sigma_bins > 0.0:
                    value = math.exp(-0.5 * distance2 / (sigma_bins * sigma_bins))
                mask[(iy + dy) % ny, (ix + dx) % nx] = value
    shifted = np.roll(np.roll(spectrum * mask, -iy, axis=0), -ix, axis=1)
    return np.fft.ifft2(shifted)


def _unwrap_2d(phase: np.ndarray) -> np.ndarray:
    return np.unwrap(np.unwrap(phase, axis=1), axis=0)


def _gaussian_smooth_fft(image: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0.0:
        return image
    ny, nx = image.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    fxx, fyy = np.meshgrid(fx, fy)
    kernel = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (fxx * fxx + fyy * fyy))
    return np.fft.ifft2(np.fft.fft2(image) * kernel).real
