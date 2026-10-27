"""Resolution-aware analysis for choosing grating patterns for an X-ray coupon."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grating_design import Candidate, DesignConfig
from pattern_comparison import default_patterns, simulate_pattern
from reconstruction import ReconstructionConfig, align_phase_offset, reconstruct_sample, synthesize_reference, synthesize_sample


def run_readiness_analysis(
    design: DesignConfig,
    candidate: Candidate,
    pattern_report: Path,
    out_dir: Path,
    frequencies: list[int] | None = None,
) -> dict:
    """Combine noisy accuracy with a noise-free spatial-frequency transfer test.

    The transfer test is deliberately noise-free. It estimates the intrinsic
    reconstruction bandwidth of the present carrier-window and smoothing setup;
    accuracy remains sourced from the photon-noise trials in ``pattern_report``.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    frequencies = frequencies or [1, 2, 3, 4, 5, 6]
    with pattern_report.open("r", encoding="utf-8") as fh:
        accuracy_report = json.load(fh)
    accuracy_rows = {row["pattern"]: row for row in accuracy_report["rows"]}
    transfer_rows = []
    for pattern in default_patterns(design):
        simulation = simulate_pattern(design, pattern, candidate)
        row = _transfer_for_pattern(design, pattern, simulation, frequencies)
        row.update(_accuracy_fields(accuracy_rows.get(pattern.name, {})))
        transfer_rows.append(row)

    _save_transfer_figure(transfer_rows, design, out_dir / "pattern_resolution_transfer.png")
    _save_scorecard(transfer_rows, out_dir / "pattern_experiment_scorecard.png")
    report = {
        "design": asdict(design),
        "frequency_cycles_per_field": frequencies,
        "method": (
            "Accuracy uses the visibility-calibrated photon-noise comparison. Spatial response uses noise-free sinusoidal "
            "phase, transmission, and isotropic dark-field phantoms, reconstructed with the current 3-bin Fourier envelope, "
            "5 px phase/T smoothing, and 16 px D smoothing. The 50% cutoff is a reconstruction-pipeline metric, not a detector MTF."
        ),
        "rows": transfer_rows,
        "files": {
            "resolution_transfer": "pattern_resolution_transfer.png",
            "scorecard": "pattern_experiment_scorecard.png",
        },
    }
    with (out_dir / "pattern_experiment_readiness_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def _transfer_for_pattern(design: DesignConfig, pattern, simulation: dict, frequencies: list[int]) -> dict:
    row = {
        "pattern": pattern.name,
        "is_proxy": pattern.is_proxy,
        "carrier_count": len(pattern.carriers),
        "mean_visibility": simulation["mean_visibility"],
        "directional_dark_field_identifiable": False,
        "phase_transfer": [],
        "transmission_transfer": [],
        "dark_field_transfer": [],
        "phase_resolution_um": None,
        "transmission_resolution_um": None,
        "dark_field_resolution_um": None,
        "resolution_note": None,
    }
    if not pattern.carrier_reconstruction or pattern.implementation == "two_shot_control":
        row["resolution_note"] = "Not evaluated by the current single-image carrier-demodulation resolution test."
        return row

    carriers = [(design.detector_harmonic * kx, design.detector_harmonic * ky) for kx, ky in pattern.carriers]
    recon_config = ReconstructionConfig(
        dx_m=design.dx_m,
        wavelength_m=design.wavelength_m,
        z_eff_m=design.z_m,
        carriers=carriers,
        samples_per_supercell=design.samples_per_supercell,
        n_supercells=design.effective_n_supercells,
        guard_bins=3,
        visibility=float(simulation["mean_visibility"]),
        smoothing_sigma_px=5.0,
        dark_field_smoothing_sigma_px=16.0,
        window_sigma_bins=0.0,
    )
    shape = design.shape
    reference = synthesize_reference(recon_config, shape)
    for q in frequencies:
        phase, absorption, dark_field = _sinusoid_truth(shape, q, len(carriers))
        sample = synthesize_sample(recon_config, phase, absorption=absorption, dark_field=dark_field)
        result = reconstruct_sample(sample, reference, recon_config)
        phase = align_phase_offset(phase, phase)
        phase_rec = align_phase_offset(result.phase, phase)
        transmission_truth = np.exp(-absorption)
        dark_truth = np.mean(dark_field, axis=0)
        row["phase_transfer"].append(_amplitude_transfer(phase_rec, phase, q))
        row["transmission_transfer"].append(_amplitude_transfer(result.transmission, transmission_truth, q))
        row["dark_field_transfer"].append(_amplitude_transfer(result.dark_field_mean, dark_truth, q))
    row["directional_dark_field_identifiable"] = _direction_rank(carriers) >= 3
    field_um = design.shape[1] * design.dx_m * 1e6
    row["phase_resolution_um"] = _half_period_at_50(row["phase_transfer"], frequencies, field_um)
    row["transmission_resolution_um"] = _half_period_at_50(row["transmission_transfer"], frequencies, field_um)
    row["dark_field_resolution_um"] = _half_period_at_50(row["dark_field_transfer"], frequencies, field_um)
    return row


def _sinusoid_truth(shape: tuple[int, int], q: int, channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(shape[1], dtype=np.float64) / shape[1]
    sine = np.broadcast_to(np.sin(2.0 * np.pi * q * x), shape)
    phase = 0.04 * sine
    absorption = 0.025 + 0.010 * sine
    dark_mean = 0.035 + 0.010 * sine
    return phase, absorption, np.broadcast_to(dark_mean, (channels,) + shape).copy()


def _amplitude_transfer(reconstruction: np.ndarray, truth: np.ndarray, q: int) -> float:
    x = np.arange(truth.shape[1], dtype=np.float64) / truth.shape[1]
    basis = np.broadcast_to(np.sin(2.0 * np.pi * q * x), truth.shape)
    truth_component = float(np.sum((truth - truth.mean()) * basis))
    reconstruction_component = float(np.sum((reconstruction - reconstruction.mean()) * basis))
    return abs(reconstruction_component / truth_component) if abs(truth_component) > 1e-15 else float("nan")


def _half_period_at_50(transfer: list[float], frequencies: list[int], field_um: float) -> float | None:
    values = np.asarray(transfer, dtype=np.float64)
    for idx in range(1, len(values)):
        if values[idx] <= 0.5 < values[idx - 1]:
            f0, f1 = frequencies[idx - 1], frequencies[idx]
            q50 = f0 + (0.5 - values[idx - 1]) * (f1 - f0) / (values[idx] - values[idx - 1])
            return float(field_um / (2.0 * q50))
    return None


def _accuracy_fields(row: dict) -> dict:
    return {
        "phase_rmse_rad": row.get("phase_rmse_rad"),
        "transmission_rmse": row.get("transmission_rmse"),
        "dark_field_mean_rmse": row.get("dark_field_mean_rmse"),
        "dark_field_anisotropy_rmse": row.get("dark_field_anisotropy_rmse"),
        "dark_field_angle_mae_deg": row.get("dark_field_angle_mae_deg"),
        "accuracy_note": row.get("reconstruction_note"),
    }


def _direction_rank(carriers) -> int:
    angles = np.arctan2([item[1] for item in carriers], [item[0] for item in carriers])
    matrix = np.stack([np.ones_like(angles), np.cos(2.0 * angles), np.sin(2.0 * angles)], axis=1)
    return int(np.linalg.matrix_rank(matrix, tol=1e-10))


def _save_transfer_figure(rows: list[dict], design: DesignConfig, path: Path) -> None:
    colors = plt.get_cmap("tab20").colors
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    frequencies = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
    field_um = design.shape[1] * design.dx_m * 1e6
    spatial_period_um = field_um / frequencies
    for idx, (key, title) in enumerate([
        ("phase_transfer", "Phase transfer"),
        ("transmission_transfer", "Transmission transfer"),
        ("dark_field_transfer", "Dark-field transfer"),
    ]):
        for color_idx, row in enumerate(rows):
            values = row[key]
            if values:
                axes[idx].plot(spatial_period_um, values, marker="o", ms=3, lw=1.25, color=colors[color_idx % len(colors)], label=row["pattern"])
        axes[idx].axhline(0.5, color="#555555", ls="--", lw=1)
        axes[idx].set_xscale("log")
        axes[idx].invert_xaxis()
        axes[idx].set_ylim(-0.05, 1.1)
        axes[idx].set_title(title)
        axes[idx].set_xlabel("sinusoid period (um), finer to the right")
        axes[idx].set_ylabel("amplitude transfer")
        axes[idx].grid(alpha=0.25)
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.suptitle("Current reconstruction spatial transfer: 3-bin carrier envelope and tuned smoothing")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_scorecard(rows: list[dict], path: Path) -> None:
    names = [row["pattern"] for row in rows]
    y = np.arange(len(rows))
    colors = ["#CC79A7" if row["is_proxy"] else "#0072B2" for row in rows]
    fig, axes = plt.subplots(1, 4, figsize=(18, 0.65 * len(rows) + 3), sharey=True, constrained_layout=True)
    for ax, key, title, xlabel in [
        (axes[0], "phase_rmse_rad", "Phase accuracy", "RMSE (rad)"),
        (axes[1], "transmission_rmse", "Transmission accuracy", "RMSE"),
        (axes[2], "dark_field_mean_rmse", "D mean accuracy", "RMSE"),
    ]:
        values = np.asarray([np.nan if row[key] is None else row[key] for row in rows], dtype=np.float64)
        valid = np.isfinite(values) & (values > 0.0)
        if np.any(valid):
            lower = float(np.nanmin(values[valid])) / 1.7
            ax.hlines(y[valid], lower, values[valid], color="#bbbbbb", lw=0.8)
        ax.scatter(values[valid], y[valid], c=[colors[idx] for idx in np.flatnonzero(valid)], s=34, zorder=3)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel + " (log)")
        ax.grid(axis="x", alpha=0.25)
    ax = axes[3]
    key = "dark_field_resolution_um"
    values = [np.nan if row[key] is None else row[key] for row in rows]
    ax.barh(y, values, color=colors)
    ax.set_title("D resolution")
    ax.set_xlabel("50% half-period (um)")
    ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    fig.suptitle("Experiment-readiness scorecard: blue = specified layout, magenta = proxy; blank = not applicable")
    fig.savefig(path, dpi=180)
    plt.close(fig)
