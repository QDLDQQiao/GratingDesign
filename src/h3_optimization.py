"""Joint physical-pattern and reconstruction optimization for H3 layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grating_design import Candidate, DesignConfig
from modality_comparison import ComparisonConfig, Modality, evaluate_modality
from pattern_comparison import PatternSpec, default_patterns, simulate_pattern


@dataclass
class H3OptimizationConfig:
    iterations: int = 160
    shortlisted_candidates: int = 16
    search_photons_per_pixel: float = 3_000.0
    search_repeats: int = 2
    validation_photons_per_pixel: float = 10_000.0
    validation_repeats: int = 5
    seed: int = 73
    phase_smoothing_sigma_px: float = 5.0
    dark_field_smoothing_sigma_px: float = 16.0
    n_cores: int = 1


def optimize_h3_family(
    design: DesignConfig,
    config: H3OptimizationConfig,
    out_dir: Path,
    s4_candidate: Candidate | None = None,
) -> dict:
    """Optimize H3 and H3-4L against physical visibility and reconstruction quality."""

    if config.iterations < 2:
        raise ValueError("iterations must be at least 2")
    if config.n_cores < 1:
        raise ValueError("n_cores must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = {pattern.name: pattern for pattern in default_patterns(design)}
    results = {}
    for index, name in enumerate(("H3", "H3_4L")):
        result = _optimize_pattern(design, patterns[name], config, config.seed + 1009 * index)
        results[name] = result
        np.save(out_dir / f"{name}_phase_map_rad.npy", result["simulation"]["phase_map"])
        np.save(out_dir / f"{name}_intensity.npy", result["simulation"]["intensity"])

    baseline = _evaluate_pattern(
        design, patterns["S4"], s4_candidate, config, config.validation_photons_per_pixel, config.validation_repeats
    )
    report = {
        "design": asdict(design),
        "optimization": asdict(config),
        "baseline": baseline,
        "patterns": {name: _serializable_result(result) for name, result in results.items()},
        "method": (
            "Stage 1 ranks random amplitude/phase candidates using propagated target-carrier visibility, balance, and contamination. "
            "Stage 2 ranks the strongest candidates by matched reconstruction settings and dose. The phase/T/D reconstruction remains "
            "visibility-calibrated rather than end-to-end sample-through-pattern propagation."
        ),
        "files": {
            "figure": "h3_optimization_summary.png",
            "H3_phase_map": "H3_phase_map_rad.npy",
            "H3_4L_phase_map": "H3_4L_phase_map_rad.npy",
        },
    }
    _save_summary_figure(results, baseline, design, out_dir / "h3_optimization_summary.png")
    with (out_dir / "h3_optimization_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def compare_optimized_h3(
    design: DesignConfig,
    s4_candidate: Candidate | None,
    optimization_report: Path,
    out_dir: Path,
) -> dict:
    """Compare selected H3 designs with S4 and established controls at fixed settings."""

    with optimization_report.open("r", encoding="utf-8") as fh:
        optimized = json.load(fh)
    config = H3OptimizationConfig(**optimized["optimization"])
    candidates = {
        "S4": s4_candidate,
        "H3": Candidate(**optimized["patterns"]["H3"]["candidate"]),
        "H3_4L": Candidate(**optimized["patterns"]["H3_4L"]["candidate"]),
    }
    patterns = {pattern.name: pattern for pattern in default_patterns(design)}
    names = ("S4", "H3", "H3_4L", "CUC_hex", "G2_grid", "C2_checkerboard")
    rows = []
    simulations = []
    for name in names:
        pattern = patterns[name]
        candidate = candidates.get(name)
        metrics = _evaluate_pattern(
            design,
            pattern,
            candidate,
            config,
            config.validation_photons_per_pixel,
            config.validation_repeats,
        )
        rows.append({"pattern": name, **metrics, "directional_dark_field_identifiable": len(pattern.carriers) >= 3})
        simulations.append(simulate_pattern(design, pattern, candidate))

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_comparison_figure(rows, out_dir / "optimized_h3_comparison.png")
    _save_selected_patterns_figure(simulations, design, out_dir / "optimized_h3_patterns.png")
    report = {
        "design": asdict(design),
        "reconstruction_settings": asdict(config),
        "method": "All patterns use the same propagated visibility measurement, 10,000 photons/pixel, five noise repeats, 5 px phase/T smoothing, and 16 px D smoothing. H3 and H3-4L use their selected joint-optimization candidates.",
        "rows": rows,
        "files": {"comparison": "optimized_h3_comparison.png", "patterns": "optimized_h3_patterns.png"},
    }
    with (out_dir / "optimized_h3_comparison_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def _optimize_pattern(design: DesignConfig, pattern: PatternSpec, config: H3OptimizationConfig, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    candidates = [Candidate(amplitudes=[1.0] * len(pattern.carriers), phases_rad=[0.0] * len(pattern.carriers))]
    for _ in range(config.iterations - 1):
        amplitudes = rng.uniform(0.35, 1.65, size=len(pattern.carriers))
        phases = rng.uniform(-np.pi, np.pi, size=len(pattern.carriers))
        candidates.append(Candidate(amplitudes=amplitudes.tolist(), phases_rad=phases.tolist()))

    physical_payloads = [(design, pattern, iteration, candidate) for iteration, candidate in enumerate(candidates)]
    if config.n_cores == 1 or len(physical_payloads) == 1:
        physical_rows = [_score_h3_physical_candidate(payload) for payload in physical_payloads]
    else:
        workers = min(config.n_cores, len(physical_payloads))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                physical_rows = list(executor.map(_score_h3_physical_candidate, physical_payloads))
        except (PermissionError, OSError):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                physical_rows = list(executor.map(_score_h3_physical_candidate, physical_payloads))
    physical_rows.sort(key=lambda row: row["physical_score"], reverse=True)
    finalists = physical_rows[: min(config.shortlisted_candidates, len(physical_rows))]
    finalist_payloads = [(design, pattern, config, row) for row in finalists]
    if config.n_cores == 1 or len(finalist_payloads) == 1:
        reconstruction_rows = [_score_h3_finalist(payload) for payload in finalist_payloads]
    else:
        workers = min(config.n_cores, len(finalist_payloads))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                reconstruction_rows = list(executor.map(_score_h3_finalist, finalist_payloads))
        except (PermissionError, OSError):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                reconstruction_rows = list(executor.map(_score_h3_finalist, finalist_payloads))
    reconstruction_rows.sort(key=lambda row: row["joint_score"], reverse=True)
    best = reconstruction_rows[0]
    best_candidate = Candidate(**best["candidate"])
    validation_metrics = _evaluate_pattern(
        design, pattern, best_candidate, config, config.validation_photons_per_pixel, config.validation_repeats
    )
    simulation = simulate_pattern(design, pattern, best_candidate)
    return {
        "candidate": best_candidate,
        "simulation": simulation,
        "search_metrics": best["search_metrics"],
        "validation_metrics": validation_metrics,
        "physical_history": [{key: value for key, value in row.items() if key != "simulation"} for row in physical_rows],
        "shortlist": [{key: value for key, value in row.items() if key != "simulation"} for row in reconstruction_rows],
    }


def _score_h3_physical_candidate(payload: tuple[DesignConfig, PatternSpec, int, Candidate]) -> dict:
    """Process-pool worker for the visibility/contamination screen."""

    design, pattern, iteration, candidate = payload
    simulation = simulate_pattern(design, pattern, candidate)
    return {
        "iteration": iteration,
        "candidate": asdict(candidate),
        "physical_score": _physical_score(simulation),
        "mean_visibility": simulation["mean_visibility"],
        "min_visibility": simulation["min_visibility"],
        "balance_cv": simulation["balance_cv"],
        "contamination_mean": simulation["contamination_mean"],
    }


def _score_h3_finalist(payload: tuple[DesignConfig, PatternSpec, H3OptimizationConfig, dict]) -> dict:
    """Process-pool worker for the short-list reconstruction screen."""

    design, pattern, config, row = payload
    metrics = _evaluate_pattern(
        design,
        pattern,
        Candidate(**row["candidate"]),
        config,
        config.search_photons_per_pixel,
        config.search_repeats,
    )
    return {**row, "search_metrics": metrics, "joint_score": _joint_score(row, metrics)}


def _evaluate_pattern(
    design: DesignConfig,
    pattern: PatternSpec,
    candidate: Candidate | None,
    config: H3OptimizationConfig,
    photons_per_pixel: float,
    repeats: int,
) -> dict:
    simulation = simulate_pattern(design, pattern, candidate)
    modality = Modality(
        name=pattern.name,
        carriers=[(design.detector_harmonic * x, design.detector_harmonic * y) for x, y in pattern.carriers],
        exposures=1,
        description=pattern.description,
    )
    comparison = ComparisonConfig(
        photons_per_pixel=[photons_per_pixel],
        repeats=repeats,
        visibility=float(simulation["mean_visibility"]),
        smoothing_sigma_px=config.phase_smoothing_sigma_px,
        dark_field_smoothing_sigma_px=config.dark_field_smoothing_sigma_px,
    )
    trials = [evaluate_modality(design, modality, comparison, photons_per_pixel, repeat) for repeat in range(repeats)]
    metric_names = (
        "phase_rmse_rad",
        "transmission_rmse",
        "dark_field_mean_rmse",
        "dark_field_anisotropy_rmse",
        "dark_field_angle_mae_deg",
    )
    metrics = {name: float(np.mean([trial[name] for trial in trials])) for name in metric_names}
    metrics.update({
        "mean_visibility": simulation["mean_visibility"],
        "min_visibility": simulation["min_visibility"],
        "balance_cv": simulation["balance_cv"],
        "contamination_mean": simulation["contamination_mean"],
    })
    return metrics


def _physical_score(simulation: dict) -> float:
    return (
        3.0 * float(simulation["min_visibility"])
        + float(simulation["mean_visibility"])
        - 0.30 * float(simulation["balance_cv"])
        - 0.50 * float(simulation["contamination_mean"])
    )


def _joint_score(row: dict, metrics: dict) -> float:
    reconstruction_quality = (
        0.32 * min(0.015 / max(metrics["phase_rmse_rad"], 1e-5), 3.0)
        + 0.08 * min(5e-4 / max(metrics["transmission_rmse"], 1e-6), 3.0)
        + 0.25 * min(0.002 / max(metrics["dark_field_mean_rmse"], 1e-6), 3.0)
        + 0.20 * min(0.12 / max(metrics["dark_field_anisotropy_rmse"], 1e-4), 3.0)
        + 0.15 * min(7.0 / max(metrics["dark_field_angle_mae_deg"], 1e-3), 3.0)
    )
    return 0.25 * row["physical_score"] + reconstruction_quality


def _serializable_result(result: dict) -> dict:
    return {
        "candidate": asdict(result["candidate"]),
        "physical_metrics": {
            key: result["simulation"][key]
            for key in ("visibility", "mean_visibility", "min_visibility", "balance_cv", "contamination_mean")
        },
        "search_metrics": result["search_metrics"],
        "validation_metrics": result["validation_metrics"],
        "physical_history": result["physical_history"],
        "shortlist": result["shortlist"],
    }


def _save_summary_figure(results: dict, baseline: dict, design: DesignConfig, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    extent = [0, design.shape[1] * design.dx_m * 1e6, 0, design.shape[0] * design.dx_m * 1e6]
    for row, (name, result) in enumerate(results.items()):
        simulation = result["simulation"]
        im = axes[row, 0].imshow(simulation["phase_map"] / np.pi, origin="lower", cmap="gray", extent=extent, vmin=0, vmax=1)
        axes[row, 0].set_title(f"{name} optimized phase map")
        axes[row, 0].set_xlabel("x (um)")
        axes[row, 0].set_ylabel("y (um)")
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046)
        im = axes[row, 1].imshow(simulation["intensity"], origin="lower", cmap="magma", extent=extent)
        axes[row, 1].set_title(f"Detector intensity, V={simulation['mean_visibility']:.3f}")
        axes[row, 1].set_xlabel("x (um)")
        axes[row, 1].set_ylabel("y (um)")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046)
        history = result["physical_history"]
        axes[row, 2].scatter(
            [item["min_visibility"] for item in history],
            [item["mean_visibility"] for item in history],
            s=10,
            alpha=0.45,
            color="#777777",
        )
        axes[row, 2].scatter(
            simulation["min_visibility"], simulation["mean_visibility"], s=55, color="#D55E00", label="selected")
        axes[row, 2].set_title(f"{name} physical search")
        axes[row, 2].set_xlabel("minimum visibility")
        axes[row, 2].set_ylabel("mean visibility")
        axes[row, 2].grid(alpha=0.25)
        axes[row, 2].legend(loc="best")
    fig.suptitle("Joint H3 physical-pattern and reconstruction optimization at matched reconstruction settings")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_comparison_figure(rows: Sequence[dict], path: Path) -> None:
    labels = [row["pattern"] for row in rows]
    x = np.arange(len(rows))
    colors = ["#0072B2" if row["directional_dark_field_identifiable"] else "#D55E00" for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = [
        ("min_visibility", "Minimum physical visibility", "visibility"),
        ("phase_rmse_rad", "Phase reconstruction", "RMSE (rad)"),
        ("transmission_rmse", "Transmission reconstruction", "RMSE"),
        ("dark_field_mean_rmse", "D mean reconstruction", "RMSE"),
        ("dark_field_anisotropy_rmse", "D anisotropy", "RMSE"),
        ("dark_field_angle_mae_deg", "D direction", "MAE (deg)"),
    ]
    for ax, (key, title, ylabel) in zip(axes.ravel(), metrics):
        values = [np.nan if row[key] is None else row[key] for row in rows]
        ax.bar(x, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Matched comparison: optimized H3/H3-4L versus S4 and established controls")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_selected_patterns_figure(simulations: Sequence[dict], design: DesignConfig, path: Path) -> None:
    extent = [0, design.shape[1] * design.dx_m * 1e6, 0, design.shape[0] * design.dx_m * 1e6]
    fig, axes = plt.subplots(len(simulations), 2, figsize=(10, 3.1 * len(simulations)), constrained_layout=True)
    for index, simulation in enumerate(simulations):
        pattern = simulation["pattern"]
        im = axes[index, 0].imshow(simulation["phase_map"] / np.pi, origin="lower", cmap="gray", extent=extent, vmin=0, vmax=1)
        axes[index, 0].set_title(f"{pattern['name']} phase map")
        axes[index, 0].set_xlabel("x (um)")
        axes[index, 0].set_ylabel("y (um)")
        fig.colorbar(im, ax=axes[index, 0], fraction=0.046)
        im = axes[index, 1].imshow(simulation["intensity"], origin="lower", cmap="magma", extent=extent)
        axes[index, 1].set_title(f"Detector intensity, V={simulation['mean_visibility']:.3f}")
        axes[index, 1].set_xlabel("x (um)")
        axes[index, 1].set_ylabel("y (um)")
        fig.colorbar(im, ax=axes[index, 1], fraction=0.046)
    fig.suptitle("Patterns used in the matched optimized-H3 comparison")
    fig.savefig(path, dpi=180)
    plt.close(fig)
