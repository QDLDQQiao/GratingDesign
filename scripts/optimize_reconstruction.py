"""Optimize grating candidates using reconstruction-quality metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import Candidate, DesignConfig, default_candidate, load_config, save_result, simulate_design
from reconstruction_benchmark import BenchmarkConfig, benchmark_design_reconstruction, metrics_to_dict
from visualize import save_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search grating candidates by reconstruction accuracy.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "reconstruction_optimized_design")
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--photons-per-pixel", type=float, default=1.0e4)
    parser.add_argument("--samples-per-supercell", type=int, default=None)
    parser.add_argument("--smoothing-sigma-px", type=float, default=5.0)
    parser.add_argument("--dark-field-smoothing-sigma-px", type=float, default=16.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.samples_per_supercell is not None:
        config = DesignConfig(**{**config.__dict__, "samples_per_supercell": args.samples_per_supercell})
    benchmark = BenchmarkConfig(
        photons_per_pixel=args.photons_per_pixel,
        smoothing_sigma_px=args.smoothing_sigma_px,
        dark_field_smoothing_sigma_px=args.dark_field_smoothing_sigma_px,
    )
    rng = np.random.default_rng(args.seed)
    candidates = [default_candidate()]
    for _ in range(args.iterations - 1):
        amplitudes = np.clip(1.0 + rng.uniform(-0.4, 0.4, size=4), 0.1, None)
        phases = rng.uniform(-np.pi, np.pi, size=4)
        candidates.append(Candidate(amplitudes=amplitudes.tolist(), phases_rad=phases.tolist()))

    best_design = None
    best_metrics = None
    history = []
    for idx, candidate in enumerate(candidates):
        design = simulate_design(config, candidate)
        recon_metrics = benchmark_design_reconstruction(design, benchmark)
        row = {
            "iteration": idx,
            "candidate": asdict(candidate),
            "design_metrics": asdict(design.metrics),
            "reconstruction_metrics": metrics_to_dict(recon_metrics),
        }
        history.append(row)
        if best_metrics is None or recon_metrics.score > best_metrics.score:
            best_design = design
            best_metrics = recon_metrics
    assert best_design is not None and best_metrics is not None

    args.out.mkdir(parents=True, exist_ok=True)
    save_result(best_design, history, args.out)
    save_visualizations(best_design, history, args.out)
    with (args.out / "reconstruction_optimization_report.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": asdict(config),
                "benchmark": asdict(benchmark),
                "best_candidate": asdict(best_design.candidate),
                "best_design_metrics": asdict(best_design.metrics),
                "best_reconstruction_metrics": metrics_to_dict(best_metrics),
                "history": history,
            },
            fh,
            indent=2,
        )
    print("Best reconstruction-aware design")
    print(f"score={best_metrics.score:.6g}")
    print(f"phase_rmse_rad={best_metrics.phase_rmse_rad:.6g}")
    print(f"transmission_rmse={best_metrics.transmission_rmse:.6g}")
    print(f"dark_field_mean_rmse={best_metrics.dark_field_mean_rmse:.6g}")
    print(f"dark_field_anisotropy_rmse={best_metrics.dark_field_anisotropy_rmse:.6g}")
    print(f"dark_field_angle_mae_deg={best_metrics.dark_field_angle_mae_deg:.6g}")
    print(f"outputs={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
