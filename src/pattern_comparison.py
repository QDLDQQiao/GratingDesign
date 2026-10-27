"""Generate and compare every grating family proposed in ``plan.md``.

The module deliberately labels FATO and MPG as proxy layouts: the plan proposes
them as optimization/fabrication tracks, not as already-finalized geometries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grating_design import Candidate, Carrier, DesignConfig, binarize_field, default_candidate
from modality_comparison import ComparisonConfig, Modality, evaluate_modality
from propagate import propagate


@dataclass(frozen=True)
class PatternSpec:
    """One planned grating family and its current simulation realization."""

    name: str
    carriers: tuple[Carrier, ...]
    description: str
    implementation: str = "binary_carrier"
    phase_levels: int = 2
    phase_span_rad: float | None = None
    carrier_reconstruction: bool = True
    is_proxy: bool = False
    exposures: int = 1


def default_patterns(config: DesignConfig) -> list[PatternSpec]:
    """Return the complete S4/H3/DRH6/FATO/CUC/RBN/MPG/control library."""

    h3 = ((6, 0), (3, 5), (-3, 5))
    drh6 = h3 + ((8, 0), (4, 7), (-4, 7))
    return [
        PatternSpec("S4", config.carriers, "Current four-carrier star baseline.", "binary_carrier"),
        PatternSpec("H3", h3, "Three approximately 60-degree carriers for rank-3 directional D.", "binary_carrier"),
        PatternSpec(
            "H3_4L",
            h3,
            "Four-level H3 phase grating for diffraction-efficiency testing.",
            "multilevel_carrier",
            phase_levels=4,
            phase_span_rad=1.5 * math.pi,
        ),
        PatternSpec("DRH6", drh6, "Dual-radius H3: two three-angle carrier radii for multiscale D.", "binary_carrier"),
        PatternSpec("FATO_proxy", config.carriers, "Rounded S4 proxy; not yet Fisher-information topology optimized.", "rounded_carrier", is_proxy=True),
        PatternSpec("CUC_square", config.carriers[:2], "Circular-unit-cell square-packed established baseline.", "circular_square"),
        PatternSpec("CUC_hex", h3, "Circular-unit-cell hex-packed established baseline.", "circular_hex"),
        PatternSpec("RBN", (), "Deterministic blue-noise-like random phase modulator baseline.", "blue_noise", carrier_reconstruction=False),
        PatternSpec("MPG_proxy", ((2, 0), (0, 2)), "2-D modulated phase-grating proxy with resolvable envelope carriers.", "modulated_phase", is_proxy=True),
        PatternSpec("G2_grid", config.carriers[:2], "Orthogonal grid control.", "binary_carrier"),
        PatternSpec("C2_checkerboard", config.carriers[2:], "Diagonal checkerboard control.", "binary_carrier"),
        PatternSpec("C2_4dir_2shot", config.carriers, "Two-shot grid plus checkerboard control with dose split across exposures.", "two_shot_control", exposures=2),
    ]


def make_pattern_field(config: DesignConfig, carriers: Sequence[Carrier], candidate: Candidate | None = None) -> np.ndarray:
    """Return an equal-weight cosine design field for an arbitrary carrier set."""

    if not carriers:
        raise ValueError("at least one carrier is required")
    xx, yy = _coordinates(config)
    amplitudes = np.ones(len(carriers), dtype=np.float64)
    phases = np.zeros(len(carriers), dtype=np.float64)
    if candidate is not None and len(candidate.amplitudes) == len(carriers):
        amplitudes = np.asarray(candidate.amplitudes, dtype=np.float64)
        phases = np.asarray(candidate.phases_rad, dtype=np.float64)
    field = np.zeros(config.shape, dtype=np.float64)
    for amplitude, phase, (kx, ky) in zip(amplitudes, phases, carriers):
        field += amplitude * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phase)
    return field


def simulate_pattern(config: DesignConfig, pattern: PatternSpec, star_candidate: Candidate | None = None) -> dict:
    """Build, propagate, and evaluate one planned pattern realization."""

    phase_map, field = _make_phase_map(config, pattern, star_candidate)
    if pattern.implementation == "two_shot_control":
        checker_phase = config.phase_depth_rad * binarize_field(make_pattern_field(config, config.carriers[2:]), config.fill_factor)
        grid_intensity = np.abs(propagate(np.exp(1j * phase_map), config.dx_m, config.wavelength_m, config.z_m, config.kernel)) ** 2
        checker_intensity = np.abs(propagate(np.exp(1j * checker_phase), config.dx_m, config.wavelength_m, config.z_m, config.kernel)) ** 2
        grid_visibility, grid_contamination = _measure_carriers(config, grid_intensity, config.carriers[:2])
        checker_visibility, checker_contamination = _measure_carriers(config, checker_intensity, config.carriers[2:])
        intensity = grid_intensity
        visibility, contamination = grid_visibility + checker_visibility, grid_contamination + checker_contamination
    else:
        transmission = np.exp(1j * phase_map)
        intensity = np.abs(propagate(transmission, config.dx_m, config.wavelength_m, config.z_m, config.kernel)) ** 2
        visibility, contamination = _measure_carriers(config, intensity, pattern.carriers)
    return {
        "pattern": asdict(pattern),
        "field": field,
        "phase_map": phase_map,
        "mask": phase_map > 0.5 * np.pi,
        "intensity": intensity,
        "visibility": visibility,
        "mean_visibility": float(np.mean(visibility)) if visibility else None,
        "min_visibility": float(np.min(visibility)) if visibility else None,
        "balance_cv": float(np.std(visibility) / np.mean(visibility)) if visibility and np.mean(visibility) else None,
        "contamination_mean": float(np.mean(contamination)) if contamination else None,
    }


def run_pattern_comparison(
    config: DesignConfig,
    out_dir: Path,
    photons_per_pixel: float = 10_000.0,
    repeats: int = 5,
    star_candidate: Candidate | None = None,
) -> dict:
    """Compare physical propagation and visibility-calibrated reconstruction noise."""

    out_dir.mkdir(parents=True, exist_ok=True)
    simulations = [simulate_pattern(config, pattern, star_candidate) for pattern in default_patterns(config)]
    rows = [_evaluate_simulation(config, simulation, photons_per_pixel, repeats) for simulation in simulations]
    _save_pattern_figure(simulations, config, out_dir / "designed_pattern_images.png")
    _save_comparison_figure(rows, config, out_dir / "pattern_simulation_comparison.png")
    report = {
        "design": asdict(config),
        "photons_per_pixel": photons_per_pixel,
        "repeats": repeats,
        "method": (
            "All masks use monochromatic scalar propagation. Carrier reconstruction trials use each layout's measured "
            "mean visibility as calibrated contrast. RBN is propagated but excluded from carrier-demodulation metrics. "
            "FATO_proxy and MPG_proxy are explicit surrogate geometries, not final optimized designs."
        ),
        "rows": rows,
        "files": {"pattern_images": "designed_pattern_images.png", "comparison_plot": "pattern_simulation_comparison.png"},
    }
    with (out_dir / "pattern_simulation_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def load_star_candidate(report_path: Path | None) -> tuple[DesignConfig | None, Candidate]:
    if report_path is None or not report_path.exists():
        return None, default_candidate()
    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    return DesignConfig(**report["config"]), Candidate(**report["candidate"])


def _evaluate_simulation(config: DesignConfig, simulation: dict, photons: float, repeats: int) -> dict:
    pattern = simulation["pattern"]
    row = {
        "pattern": pattern["name"], "description": pattern["description"], "implementation": pattern["implementation"],
        "is_proxy": pattern["is_proxy"], "carrier_count": len(pattern["carriers"]), "exposures": pattern["exposures"],
        "mean_visibility": simulation["mean_visibility"], "min_visibility": simulation["min_visibility"],
        "balance_cv": simulation["balance_cv"], "contamination_mean": simulation["contamination_mean"],
    }
    metric_keys = ("phase_rmse_rad", "transmission_rmse", "dark_field_mean_rmse", "dark_field_anisotropy_rmse", "dark_field_angle_mae_deg")
    if pattern["implementation"] == "two_shot_control":
        row.update({key: None for key in metric_keys})
        row["directional_dark_field_identifiable"] = True
        row["reconstruction_note"] = "Physical visibility is measured across both exposures; phase/T/D metrics await a dedicated two-image fusion reconstruction."
        return row
    if not pattern["carrier_reconstruction"] or not pattern["carriers"]:
        row.update({key: None for key in metric_keys})
        row["directional_dark_field_identifiable"] = False
        row["reconstruction_note"] = "Requires a speckle/random-mask retrieval method; carrier-demodulation metrics are not applicable."
        return row
    modality = Modality(
        name=pattern["name"], carriers=[(config.detector_harmonic * x, config.detector_harmonic * y) for x, y in pattern["carriers"]],
        exposures=int(pattern["exposures"]), description=pattern["description"],
    )
    comparison = ComparisonConfig(photons_per_pixel=[photons], repeats=repeats, visibility=float(simulation["mean_visibility"]))
    trials = [evaluate_modality(config, modality, comparison, photons, repeat) for repeat in range(repeats)]
    for key in metric_keys:
        values = np.asarray([trial[key] for trial in trials], dtype=np.float64)
        row[key] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else None
    row["directional_dark_field_identifiable"] = bool(trials[0]["directional_dark_field_identifiable"])
    row["reconstruction_note"] = "Visibility-calibrated carrier-demodulation benchmark."
    return row


def _make_phase_map(config: DesignConfig, pattern: PatternSpec, star_candidate: Candidate | None) -> tuple[np.ndarray, np.ndarray]:
    candidate = star_candidate
    if pattern.implementation in {"binary_carrier", "multilevel_carrier"}:
        field = make_pattern_field(config, pattern.carriers, candidate)
        if pattern.phase_levels == 2:
            return config.phase_depth_rad * binarize_field(field, config.fill_factor), field
        ranks = np.argsort(np.argsort(field, axis=None), axis=None).reshape(field.shape)
        levels = np.floor(pattern.phase_levels * ranks / field.size).clip(0, pattern.phase_levels - 1)
        phase_span = pattern.phase_span_rad if pattern.phase_span_rad is not None else config.phase_depth_rad
        return phase_span * levels / (pattern.phase_levels - 1), field
    if pattern.implementation == "rounded_carrier":
        field = make_pattern_field(config, pattern.carriers)
        smooth = _gaussian_lowpass(field, sigma_px=1.1)
        return config.phase_depth_rad * binarize_field(smooth, config.fill_factor), smooth
    if pattern.implementation == "circular_square":
        field, phase = _circular_phase_map(config, ((config.m, 0), (0, config.m)))
        return phase, field
    if pattern.implementation == "circular_hex":
        field, phase = _circular_phase_map(config, ((6, 0), (3, 5)))
        return phase, field
    if pattern.implementation == "blue_noise":
        field = _blue_noise_field(config)
        return config.phase_depth_rad * binarize_field(field, config.fill_factor), field
    if pattern.implementation == "modulated_phase":
        xx, yy = _coordinates(config)
        envelope = 0.5 + 0.5 * np.cos(2.0 * np.pi * 2.0 * xx) * np.cos(2.0 * np.pi * 2.0 * yy)
        carrier = np.cos(2.0 * np.pi * config.m * xx) + np.cos(2.0 * np.pi * config.m * yy)
        field = envelope * carrier
        phase = config.phase_depth_rad * binarize_field(field, config.fill_factor)
        return phase, field
    if pattern.implementation == "two_shot_control":
        field = make_pattern_field(config, config.carriers[:2])
        return config.phase_depth_rad * binarize_field(field, config.fill_factor), field
    raise ValueError(f"unsupported pattern implementation: {pattern.implementation}")


def _coordinates(config: DesignConfig) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(config.shape[0], dtype=np.float64) / config.samples_per_supercell
    x = np.arange(config.shape[1], dtype=np.float64) / config.samples_per_supercell
    return np.meshgrid(x, y)


def _circular_phase_map(config: DesignConfig, reciprocal: tuple[Carrier, Carrier]) -> tuple[np.ndarray, np.ndarray]:
    """Construct a periodic circular-cell lattice from two reciprocal vectors."""

    xx, yy = _coordinates(config)
    reciprocal_matrix = np.asarray(reciprocal, dtype=np.float64)
    lattice_coords = reciprocal_matrix @ np.stack([xx.ravel(), yy.ravel()])
    u = lattice_coords[0].reshape(config.shape) % 1.0
    v = lattice_coords[1].reshape(config.shape) % 1.0
    distance = np.sqrt(np.minimum(u, 1.0 - u) ** 2 + np.minimum(v, 1.0 - v) ** 2)
    radius = math.sqrt(config.fill_factor / math.pi)
    phase = config.phase_depth_rad * (distance <= radius)
    return -distance, phase


def _blue_noise_field(config: DesignConfig) -> np.ndarray:
    rng = np.random.default_rng(702)
    noise = rng.standard_normal(config.shape)
    fy = np.fft.fftfreq(config.shape[0])[:, None]
    fx = np.fft.fftfreq(config.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    highpass = 1.0 - np.exp(-(radius / 0.09) ** 2)
    return np.fft.ifft2(np.fft.fft2(noise) * highpass).real


def _gaussian_lowpass(field: np.ndarray, sigma_px: float) -> np.ndarray:
    fy = np.fft.fftfreq(field.shape[0])[:, None]
    fx = np.fft.fftfreq(field.shape[1])[None, :]
    kernel = np.exp(-2.0 * np.pi * np.pi * sigma_px * sigma_px * (fx * fx + fy * fy))
    return np.fft.ifft2(np.fft.fft2(field) * kernel).real


def _measure_carriers(config: DesignConfig, intensity: np.ndarray, carriers: Sequence[Carrier]) -> tuple[list[float], list[float]]:
    spectrum = np.fft.fft2(np.asarray(intensity, dtype=np.float64)) / intensity.size
    dc = abs(spectrum[0, 0])
    visibility, contamination = [], []
    for kx, ky in carriers:
        ix = (config.detector_harmonic * kx * config.effective_n_supercells) % config.shape[1]
        iy = (config.detector_harmonic * ky * config.effective_n_supercells) % config.shape[0]
        peak = abs(spectrum[iy, ix])
        visibility.append(float(2.0 * peak / dc))
        contamination.append(_window_contamination(spectrum, iy, ix, config.guard_bins))
    return visibility, contamination


def _window_contamination(spectrum: np.ndarray, iy: int, ix: int, radius: int) -> float:
    peak = abs(spectrum[iy, ix]) ** 2
    if peak == 0.0:
        return float("inf")
    total = sum(
        abs(spectrum[(iy + dy) % spectrum.shape[0], (ix + dx) % spectrum.shape[1]]) ** 2
        for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius
    )
    return float(max(total - peak, 0.0) / peak)


def _save_pattern_figure(simulations: Sequence[dict], config: DesignConfig, path: Path) -> None:
    extent = [0, config.shape[1] * config.dx_m * 1e6, 0, config.shape[0] * config.dx_m * 1e6]
    fig, axes = plt.subplots(len(simulations), 2, figsize=(10, 3.1 * len(simulations)), constrained_layout=True)
    for row, simulation in enumerate(simulations):
        pattern = simulation["pattern"]
        im = axes[row, 0].imshow(simulation["phase_map"] / np.pi, origin="lower", cmap="gray", extent=extent, interpolation="nearest", vmin=0, vmax=1)
        suffix = " (proxy)" if pattern["is_proxy"] else ""
        if pattern["exposures"] > 1:
            suffix += " (shot 1 of 2)"
        axes[row, 0].set_title(f"{pattern['name']} phase map" + suffix)
        axes[row, 0].set_xlabel("x (um)")
        axes[row, 0].set_ylabel("y (um)")
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046, label="phase / pi")
        im = axes[row, 1].imshow(simulation["intensity"], origin="lower", cmap="magma", extent=extent)
        visibility = simulation["mean_visibility"]
        suffix = f"V={visibility:.3f}" if visibility is not None else "non-carrier"
        intensity_label = f"Detector intensity, {suffix}"
        if pattern["exposures"] > 1:
            intensity_label += " (shot 1)"
        axes[row, 1].set_title(intensity_label)
        axes[row, 1].set_xlabel("x (um)")
        axes[row, 1].set_ylabel("y (um)")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046)
    fig.suptitle(f"Selected phase-mask families at z={config.z_m * 1e3:.1f} mm")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_comparison_figure(rows: Sequence[dict], config: DesignConfig, path: Path) -> None:
    labels = [row["pattern"] for row in rows]
    x = np.arange(len(rows))
    colors = ["#0072B2" if not row["is_proxy"] else "#CC79A7" for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    metrics = [
        ("mean_visibility", "Physical carrier visibility", "visibility"), ("min_visibility", "Weakest carrier visibility", "visibility"),
        ("contamination_mean", "Fourier-window contamination", "ratio"), ("phase_rmse_rad", "Phase reconstruction", "RMSE (rad)"),
        ("transmission_rmse", "Transmission reconstruction", "RMSE"), ("dark_field_mean_rmse", "Dark-field mean reconstruction", "RMSE"),
    ]
    for ax, (key, title, ylabel) in zip(axes.ravel(), metrics):
        values = [np.nan if row[key] is None else row[key] for row in rows]
        ax.bar(x, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"All planned grating families: physical propagation and calibrated reconstruction ({config.z_m * 1e3:.1f} mm)")
    fig.savefig(path, dpi=180)
    plt.close(fig)
