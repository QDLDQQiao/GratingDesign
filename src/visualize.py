"""Visualization helpers for grating design results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grating_design import DesignResult
from reconstruction import ReconstructionResult, align_phase_offset, fit_dark_field_anisotropy


def save_visualizations(result: DesignResult, history: Sequence[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_design_summary(result, out_dir / "design_summary.png")
    save_history(history, out_dir / "optimization_history.png")


def save_design_summary(result: DesignResult, path: Path) -> None:
    cfg = result.config
    extent_um = [0, cfg.shape[1] * cfg.dx_m * 1e6, 0, cfg.shape[0] * cfg.dx_m * 1e6]
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(result.intensity)))
    spectrum = np.log10(spectrum / max(float(spectrum.max()), 1e-30) + 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    im0 = axes[0, 0].imshow(result.design_field, cmap="coolwarm", origin="lower", extent=extent_um)
    axes[0, 0].set_title("Continuous design field")
    axes[0, 0].set_xlabel("x (um)")
    axes[0, 0].set_ylabel("y (um)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(result.thickness_map_m * 1e9, cmap="gray", origin="lower", extent=extent_um)
    axes[0, 1].set_title("Binary grating thickness (nm)")
    axes[0, 1].set_xlabel("x (um)")
    axes[0, 1].set_ylabel("y (um)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(result.intensity, cmap="magma", origin="lower", extent=extent_um)
    axes[1, 0].set_title(f"Detector intensity at z={cfg.z_m * 1e3:.1f} mm")
    axes[1, 0].set_xlabel("x (um)")
    axes[1, 0].set_ylabel("y (um)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(spectrum, cmap="viridis", origin="lower")
    axes[1, 1].set_title("Log intensity spectrum")
    axes[1, 1].set_xlabel("FFT x bin")
    axes[1, 1].set_ylabel("FFT y bin")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    fig.suptitle(
        "Optimized SB-BSTG grating: "
        f"score {result.metrics.score:.3f}, mean V {result.metrics.mean_visibility:.3f}, "
        f"min V {result.metrics.min_visibility:.3f}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_history(history: Sequence[dict], path: Path) -> None:
    if not history:
        return
    iterations = [row["iteration"] for row in history]
    if "score" in history[0]:
        scores = [row["score"] for row in history]
        mean_visibility = [row["mean_visibility"] for row in history]
        min_visibility = [row["min_visibility"] for row in history]
        contamination = [row["contamination_mean"] for row in history]
    else:
        scores = [row["reconstruction_metrics"]["score"] for row in history]
        mean_visibility = [row["design_metrics"]["mean_visibility"] for row in history]
        min_visibility = [row["design_metrics"]["min_visibility"] for row in history]
        contamination = [row["design_metrics"]["contamination_mean"] for row in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(iterations, scores, label="score", color="#222222", linewidth=1.6)
    axes[0].plot(iterations, mean_visibility, label="mean visibility", color="#0072B2", linewidth=1.2)
    axes[0].plot(iterations, min_visibility, label="min visibility", color="#D55E00", linewidth=1.2)
    axes[0].set_ylabel("metric")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(iterations, contamination, label="contamination", color="#009E73", linewidth=1.2)
    axes[1].set_xlabel("candidate")
    axes[1].set_ylabel("window contamination")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.suptitle("Optimization history")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reconstruction_summary(
    result: ReconstructionResult,
    truth_phase: np.ndarray,
    sample: np.ndarray,
    reference: np.ndarray,
    path: Path,
    truth_transmission: np.ndarray | None = None,
    truth_dark_field: np.ndarray | None = None,
) -> None:
    phase = align_phase_offset(result.phase, truth_phase)
    phase_error = phase - truth_phase

    if truth_transmission is None:
        truth_transmission = np.ones_like(result.transmission)
    transmission_error = result.transmission - truth_transmission

    if truth_dark_field is None:
        truth_dark_mean = np.zeros_like(result.dark_field_mean)
    else:
        truth_dark_mean, _, _ = fit_dark_field_anisotropy(truth_dark_field, result.demodulation.carrier_vectors)
    dark_error = result.dark_field_mean - truth_dark_mean

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11), constrained_layout=True)
    rows = [
        ("Phase (rad)", truth_phase, phase, phase_error, "coolwarm"),
        ("T", truth_transmission, result.transmission, transmission_error, "viridis"),
        ("D mean", truth_dark_mean, result.dark_field_mean, dark_error, "viridis"),
    ]
    for row_idx, (label, truth, recon, err, cmap) in enumerate(rows):
        vmin = min(float(np.nanmin(truth)), float(np.nanmin(recon)))
        vmax = max(float(np.nanmax(truth)), float(np.nanmax(recon)))
        err_abs = max(float(np.nanmax(np.abs(err))), 1e-12)
        panels = [
            (truth, f"{label} truth", cmap, vmin, vmax),
            (recon, f"{label} reconstruction", cmap, vmin, vmax),
            (err, f"{label} error", "coolwarm", -err_abs, err_abs),
        ]
        for ax, (data, title, panel_cmap, panel_vmin, panel_vmax) in zip(axes[row_idx], panels):
            im = ax.imshow(data, origin="lower", cmap=panel_cmap, vmin=panel_vmin, vmax=panel_vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    phase_rmse = float(np.sqrt(np.mean(phase_error * phase_error)))
    t_rmse = float(np.sqrt(np.mean(transmission_error * transmission_error)))
    d_rmse = float(np.sqrt(np.mean(dark_error * dark_error)))
    fig.suptitle(
        "Carrier demodulation reconstruction: "
        f"phase RMSE {phase_rmse:.3g} rad, T RMSE {t_rmse:.3g}, D RMSE {d_rmse:.3g}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reconstruction_input_summary(
    sample: np.ndarray,
    reference: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, data, title in [
        (axes[0], reference, "Reference image"),
        (axes[1], sample, "Sample image"),
    ]:
        im = ax.imshow(data, origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=180)
    plt.close(fig)
