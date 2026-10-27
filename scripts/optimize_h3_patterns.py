"""Optimize H3 and H3-4L physical patterns and matched reconstruction metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import load_config
from h3_optimization import H3OptimizationConfig, optimize_h3_family
from pattern_comparison import load_star_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize H3 and H3-4L with propagated visibility and phase/T/D reconstruction metrics.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--design-report", type=Path, default=ROOT / "results" / "design_run" / "design_report.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "h3_optimization")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--shortlist", type=int, default=16)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--search-photons-per-pixel", type=float, default=3_000.0)
    parser.add_argument("--search-repeats", type=int, default=2)
    parser.add_argument("--validation-photons-per-pixel", type=float, default=10_000.0)
    parser.add_argument("--validation-repeats", type=int, default=5)
    parser.add_argument("--n-cores", type=int, default=1, help="Worker processes for independent H3 candidate evaluations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_config, s4_candidate = load_star_candidate(args.design_report)
    config = H3OptimizationConfig(
        iterations=args.iterations,
        shortlisted_candidates=args.shortlist,
        search_photons_per_pixel=args.search_photons_per_pixel,
        search_repeats=args.search_repeats,
        validation_photons_per_pixel=args.validation_photons_per_pixel,
        validation_repeats=args.validation_repeats,
        seed=args.seed,
        n_cores=args.n_cores,
    )
    report = optimize_h3_family(report_config or load_config(args.config), config, args.out, s4_candidate)
    print(f"H3 optimization written to {args.out}")
    for name, result in report["patterns"].items():
        metrics = result["validation_metrics"]
        print(f"{name}: Vmin={metrics['min_visibility']:.3f}, phase={metrics['phase_rmse_rad']:.4g}, D-angle={metrics['dark_field_angle_mae_deg']:.3g} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
