"""Compare star, grid, and checkerboard carrier layouts for reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from grating_design import DesignConfig
from phantoms import periodic_directional_dark_field, smooth_absorption, smooth_periodic_phase
from reconstruction import (
    ReconstructionConfig,
    align_phase_offset,
    fit_dark_field_anisotropy,
    reconstruct_sample,
    synthesize_reference,
    synthesize_sample,
)
from reconstruction_benchmark import add_poisson_noise


Carrier = Tuple[int, int]


@dataclass
class Modality:
    name: str
    carriers: List[Carrier]
    exposures: int
    description: str


@dataclass
class ComparisonConfig:
    photons_per_pixel: List[float]
    repeats: int = 5
    visibility: float = 0.142470873981641
    smoothing_sigma_px: float = 5.0
    dark_field_smoothing_sigma_px: float | None = 16.0
    window_sigma_bins: float = 0.0
    seed: int = 23
    phase_amplitude_rad: float = 0.08
    absorption_peak_mu: float = 0.05
    dark_field_mean: float = 0.035
    dark_field_anisotropy: float = 0.35
    dark_field_angle_rad: float = 0.55


def default_modalities(design: DesignConfig) -> List[Modality]:
    axial = [(design.detector_harmonic * design.m, 0), (0, design.detector_harmonic * design.m)]
    diagonal = [
        (design.detector_harmonic * design.n, design.detector_harmonic * design.n),
        (design.detector_harmonic * design.n, -design.detector_harmonic * design.n),
    ]
    return [
        Modality("star_4dir_single", axial + diagonal, 1, "Star-like SB-BSTG single-shot four-direction carrier set."),
        Modality("grid_2dir_single", axial, 1, "Axis-aligned grid single-shot two-direction carrier set."),
        Modality("checkerboard_2dir_single", diagonal, 1, "Single checkerboard-like diagonal two-direction carrier set."),
        Modality("checkerboard_4dir_2shot", axial + diagonal, 2, "Two checkerboard/grid exposures with total dose split across shots."),
    ]


def run_modality_comparison(
    design: DesignConfig,
    config: ComparisonConfig,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for modality in default_modalities(design):
        for photons in config.photons_per_pixel:
            metrics = [
                evaluate_modality(design, modality, config, photons, repeat)
                for repeat in range(config.repeats)
            ]
            rows.append(_aggregate_rows(modality, photons, metrics))

    csv_path = out_dir / "modality_comparison.csv"
    _write_csv(csv_path, rows)
    report = {
        "design": asdict(design),
        "config": asdict(config),
        "modalities": [asdict(item) for item in default_modalities(design)],
        "rows": rows,
        "files": {
            "csv": "modality_comparison.csv",
            "plot": "modality_comparison.png",
            "recovered_maps_plot": "modality_recovered_maps.png",
            "error_maps_plot": "modality_error_maps.png",
            "directional_dark_field_plot": "modality_directional_dark_field.png",
        },
    }
    with (out_dir / "modality_comparison_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def evaluate_modality(
    design: DesignConfig,
    modality: Modality,
    comparison: ComparisonConfig,
    photons_per_pixel: float,
    repeat: int,
) -> dict:
    recon_config = ReconstructionConfig(
        dx_m=design.dx_m,
        wavelength_m=design.wavelength_m,
        z_eff_m=design.z_m,
        carriers=modality.carriers,
        samples_per_supercell=design.samples_per_supercell,
        n_supercells=design.n_supercells,
        guard_bins=3,
        visibility=comparison.visibility,
        smoothing_sigma_px=comparison.smoothing_sigma_px,
        dark_field_smoothing_sigma_px=comparison.dark_field_smoothing_sigma_px,
        window_sigma_bins=comparison.window_sigma_bins,
    )
    shape = design.shape
    truth_phase = smooth_periodic_phase(shape, amplitude_rad=comparison.phase_amplitude_rad)
    absorption = smooth_absorption(shape, peak_mu=comparison.absorption_peak_mu)
    truth_t = np.exp(-absorption)
    directions = _directions(modality.carriers)
    truth_d = periodic_directional_dark_field(
        shape,
        directions,
        angle_rad=comparison.dark_field_angle_rad,
        mean_strength=comparison.dark_field_mean,
        anisotropy=comparison.dark_field_anisotropy,
    )
    reference = synthesize_reference(recon_config, shape)
    sample = synthesize_sample(recon_config, truth_phase, absorption=absorption, dark_field=truth_d)
    if photons_per_pixel > 0.0:
        rng = np.random.default_rng(comparison.seed + 1009 * repeat + int(photons_per_pixel))
        per_exposure = photons_per_pixel / max(modality.exposures, 1)
        reference = add_poisson_noise(reference, per_exposure, rng)
        sample = add_poisson_noise(sample, per_exposure, rng)

    result = reconstruct_sample(sample, reference, recon_config)
    phase = align_phase_offset(result.phase, truth_phase)
    d0_true, dani_true, dangle_true = fit_dark_field_anisotropy(truth_d, result.demodulation.carrier_vectors)
    valid = d0_true > max(float(np.max(d0_true)) * 0.05, 1e-12)
    rank = _direction_rank(directions)
    if rank >= 3:
        dani_rmse = _rmse(result.dark_field_anisotropy[valid], dani_true[valid])
        angle_error = _wrapped_angle_error(result.dark_field_angle, dangle_true)
        angle_mae = float(np.mean(np.abs(angle_error[valid])) * 180.0 / np.pi)
    else:
        dani_rmse = float("nan")
        angle_mae = float("nan")
    return {
        "phase_rmse_rad": _rmse(phase, truth_phase),
        "transmission_rmse": _rmse(result.transmission, truth_t),
        "dark_field_mean_rmse": _rmse(result.dark_field_mean[valid], d0_true[valid]),
        "dark_field_anisotropy_rmse": dani_rmse,
        "dark_field_angle_mae_deg": angle_mae,
        "directional_dark_field_identifiable": rank >= 3,
        "dark_field_fit_rank": rank,
    }


def reconstruct_modality_maps(
    design: DesignConfig,
    modality: Modality,
    comparison: ComparisonConfig,
    photons_per_pixel: float,
    repeat: int = 0,
) -> dict:
    """Return truth/reconstruction/error maps for visual comparison."""

    recon_config = ReconstructionConfig(
        dx_m=design.dx_m,
        wavelength_m=design.wavelength_m,
        z_eff_m=design.z_m,
        carriers=modality.carriers,
        samples_per_supercell=design.samples_per_supercell,
        n_supercells=design.n_supercells,
        guard_bins=3,
        visibility=comparison.visibility,
        smoothing_sigma_px=comparison.smoothing_sigma_px,
        dark_field_smoothing_sigma_px=comparison.dark_field_smoothing_sigma_px,
        window_sigma_bins=comparison.window_sigma_bins,
    )
    shape = design.shape
    truth_phase = smooth_periodic_phase(shape, amplitude_rad=comparison.phase_amplitude_rad)
    absorption = smooth_absorption(shape, peak_mu=comparison.absorption_peak_mu)
    truth_t = np.exp(-absorption)
    directions = _directions(modality.carriers)
    truth_d = periodic_directional_dark_field(
        shape,
        directions,
        angle_rad=comparison.dark_field_angle_rad,
        mean_strength=comparison.dark_field_mean,
        anisotropy=comparison.dark_field_anisotropy,
    )
    reference = synthesize_reference(recon_config, shape)
    sample = synthesize_sample(recon_config, truth_phase, absorption=absorption, dark_field=truth_d)
    if photons_per_pixel > 0.0:
        rng = np.random.default_rng(comparison.seed + 1009 * repeat + int(photons_per_pixel))
        per_exposure = photons_per_pixel / max(modality.exposures, 1)
        reference = add_poisson_noise(reference, per_exposure, rng)
        sample = add_poisson_noise(sample, per_exposure, rng)

    result = reconstruct_sample(sample, reference, recon_config)
    phase = align_phase_offset(result.phase, truth_phase)
    d0_true, _, _ = fit_dark_field_anisotropy(truth_d, result.demodulation.carrier_vectors)
    return {
        "phase_truth": truth_phase,
        "phase_reconstruction": phase,
        "phase_error": phase - truth_phase,
        "transmission_truth": truth_t,
        "transmission_reconstruction": result.transmission,
        "transmission_error": result.transmission - truth_t,
        "dark_field_mean_truth": d0_true,
        "dark_field_mean_reconstruction": result.dark_field_mean,
        "dark_field_mean_error": result.dark_field_mean - d0_true,
        "dark_field_anisotropy_truth": fit_dark_field_anisotropy(truth_d, result.demodulation.carrier_vectors)[1],
        "dark_field_anisotropy_reconstruction": result.dark_field_anisotropy,
        "dark_field_anisotropy_error": result.dark_field_anisotropy - fit_dark_field_anisotropy(truth_d, result.demodulation.carrier_vectors)[1],
        "dark_field_angle_truth": fit_dark_field_anisotropy(truth_d, result.demodulation.carrier_vectors)[2],
        "dark_field_angle_reconstruction": result.dark_field_angle,
        "dark_field_angle_error": _wrapped_angle_error(
            result.dark_field_angle,
            fit_dark_field_anisotropy(truth_d, result.demodulation.carrier_vectors)[2],
        ),
        "directional_dark_field_identifiable": _direction_rank(directions) >= 3,
    }


def save_comparison_plot(report: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]
    metrics = [
        ("phase_rmse_rad_mean", "Phase RMSE (rad)"),
        ("transmission_rmse_mean", "T RMSE"),
        ("dark_field_mean_rmse_mean", "D mean RMSE"),
        ("dark_field_angle_mae_deg_mean", "D angle MAE (deg)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for modality in sorted({row["modality"] for row in rows}):
            subset = [row for row in rows if row["modality"] == modality]
            subset.sort(key=lambda item: item["photons_per_pixel"])
            x = [item["photons_per_pixel"] for item in subset]
            y = [item[metric] for item in subset]
            ax.plot(x, y, marker="o", label=modality)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("photons per pixel, total dose")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Star-like vs grid/checkerboard reconstruction comparison")
    fig.savefig(out_dir / "modality_comparison.png", dpi=180)
    plt.close(fig)


def save_recovered_maps_figure(
    design: DesignConfig,
    comparison: ComparisonConfig,
    out_dir: Path,
    photons_per_pixel: float,
    repeat: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modalities = default_modalities(design)
    maps = {
        modality.name: reconstruct_modality_maps(design, modality, comparison, photons_per_pixel, repeat)
        for modality in modalities
    }
    rows = [
        ("phase_reconstruction", "phase_truth", "phase_error", "Phase reconstruction", "Phase error", "coolwarm"),
        ("transmission_reconstruction", "transmission_truth", "transmission_error", "T reconstruction", "T error", "viridis"),
        ("dark_field_mean_reconstruction", "dark_field_mean_truth", "dark_field_mean_error", "D mean reconstruction", "D mean error", "viridis"),
    ]
    fig, axes = plt.subplots(len(rows), len(modalities) + 1, figsize=(17, 10), constrained_layout=True)
    for row_idx, (recon_key, truth_key, err_key, recon_label, err_label, cmap) in enumerate(rows):
        truth = next(iter(maps.values()))[truth_key]
        vmin = float(np.nanmin(truth))
        vmax = float(np.nanmax(truth))
        im = axes[row_idx, 0].imshow(truth, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_title(f"{recon_label.split()[0]} truth")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        fig.colorbar(im, ax=axes[row_idx, 0], fraction=0.046)
        max_err = max(float(np.nanmax(np.abs(item[err_key]))) for item in maps.values())
        max_err = max(max_err, 1e-12)
        for col_idx, modality in enumerate(modalities, start=1):
            item = maps[modality.name]
            if row_idx == 0:
                panel = item[recon_key]
                panel_title = modality.name
                panel_cmap = cmap
                panel_vmin = min(vmin, float(np.nanmin(panel)))
                panel_vmax = max(vmax, float(np.nanmax(panel)))
            elif row_idx == 1:
                panel = item[recon_key]
                panel_title = recon_label
                panel_cmap = cmap
                panel_vmin = min(vmin, float(np.nanmin(panel)))
                panel_vmax = max(vmax, float(np.nanmax(panel)))
            else:
                panel = item[recon_key]
                panel_title = recon_label
                panel_cmap = cmap
                panel_vmin = min(vmin, float(np.nanmin(panel)))
                panel_vmax = max(vmax, float(np.nanmax(panel)))
            im = axes[row_idx, col_idx].imshow(panel, origin="lower", cmap=panel_cmap, vmin=panel_vmin, vmax=panel_vmax)
            if not item["directional_dark_field_identifiable"] and row_idx == 2:
                panel_title += "\n(direction only n/a)"
            axes[row_idx, col_idx].set_title(panel_title, fontsize=9)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            fig.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046)
    fig.suptitle(f"Recovered phase, T, and D mean by method at {photons_per_pixel:g} photons/pixel")
    fig.savefig(out_dir / "modality_recovered_maps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(rows), len(modalities), figsize=(14, 9), constrained_layout=True)
    for row_idx, (_, _, err_key, recon_label, _, _) in enumerate(rows):
        max_err = max(float(np.nanmax(np.abs(item[err_key]))) for item in maps.values())
        max_err = max(max_err, 1e-12)
        for col_idx, modality in enumerate(modalities):
            item = maps[modality.name]
            im = axes[row_idx, col_idx].imshow(
                item[err_key],
                origin="lower",
                cmap="coolwarm",
                vmin=-max_err,
                vmax=max_err,
            )
            title = modality.name if row_idx == 0 else recon_label.replace("reconstruction", "error")
            if not item["directional_dark_field_identifiable"] and row_idx == 2:
                title += "\n(direction only n/a)"
            axes[row_idx, col_idx].set_title(title, fontsize=9)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            fig.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046)
    fig.suptitle(f"Phase, T, and D mean error maps by method at {photons_per_pixel:g} photons/pixel")
    fig.savefig(out_dir / "modality_error_maps.png", dpi=180)
    plt.close(fig)

    if any(item["directional_dark_field_identifiable"] for item in maps.values()):
        directional_rows = [
            ("dark_field_anisotropy_reconstruction", "dark_field_anisotropy_truth", "dark_field_anisotropy_error", "D anisotropy", "magma"),
            ("dark_field_angle_reconstruction", "dark_field_angle_truth", "dark_field_angle_error", "D angle (rad)", "twilight"),
        ]
        fig, axes = plt.subplots(len(directional_rows), len(modalities) + 1, figsize=(17, 7), constrained_layout=True)
        for row_idx, (recon_key, truth_key, err_key, label, cmap) in enumerate(directional_rows):
            truth = next(iter(maps.values()))[truth_key]
            vmin = float(np.nanmin(truth))
            vmax = float(np.nanmax(truth))
            im = axes[row_idx, 0].imshow(truth, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, 0].set_title(f"{label} truth")
            axes[row_idx, 0].set_xticks([])
            axes[row_idx, 0].set_yticks([])
            fig.colorbar(im, ax=axes[row_idx, 0], fraction=0.046)
            for col_idx, modality in enumerate(modalities, start=1):
                item = maps[modality.name]
                panel = item[recon_key]
                title = modality.name
                if not item["directional_dark_field_identifiable"]:
                    panel = np.full_like(panel, np.nan)
                    title += "\n(direction only n/a)"
                im = axes[row_idx, col_idx].imshow(panel, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                axes[row_idx, col_idx].set_title(title, fontsize=9)
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])
                fig.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046)
        fig.suptitle(f"Directional dark-field recovery by method at {photons_per_pixel:g} photons/pixel")
        fig.savefig(out_dir / "modality_directional_dark_field.png", dpi=180)
        plt.close(fig)


def _aggregate_rows(modality: Modality, photons: float, metrics: Sequence[dict]) -> dict:
    row = {
        "modality": modality.name,
        "photons_per_pixel": photons,
        "exposures": modality.exposures,
        "directional_dark_field_identifiable": bool(metrics[0]["directional_dark_field_identifiable"]),
        "dark_field_fit_rank": int(metrics[0]["dark_field_fit_rank"]),
    }
    numeric_keys = [
        "phase_rmse_rad",
        "transmission_rmse",
        "dark_field_mean_rmse",
        "dark_field_anisotropy_rmse",
        "dark_field_angle_mae_deg",
    ]
    for key in numeric_keys:
        values = np.asarray([item[key] for item in metrics], dtype=np.float64)
        row[f"{key}_mean"] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
        row[f"{key}_std"] = float(np.nanstd(values, ddof=0)) if not np.all(np.isnan(values)) else float("nan")
    return row


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _directions(carriers: Sequence[Carrier]) -> np.ndarray:
    directions = []
    for kx, ky in carriers:
        norm = math.hypot(kx, ky)
        directions.append((kx / norm, ky / norm))
    return np.asarray(directions, dtype=np.float64)


def _direction_rank(directions: np.ndarray) -> int:
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    design = np.stack([np.ones_like(angles), np.cos(2.0 * angles), np.sin(2.0 * angles)], axis=1)
    return int(np.linalg.matrix_rank(design, tol=1e-10))


def _rmse(actual: np.ndarray, truth: np.ndarray) -> float:
    error = np.asarray(actual, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def _wrapped_angle_error(angle: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return 0.5 * np.angle(np.exp(2j * (angle - truth)))
