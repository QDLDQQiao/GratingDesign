"""Run the physical comparison for every grating family in ``plan.md``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import load_config
from pattern_comparison import load_star_candidate, run_pattern_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all planned propagated grating patterns and calibrated reconstruction noise.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--design-report", type=Path, default=ROOT / "results" / "design_run" / "design_report.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "pattern_simulation")
    parser.add_argument("--photons-per-pixel", type=float, default=10_000.0)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_config, candidate = load_star_candidate(args.design_report)
    report = run_pattern_comparison(
        report_config or load_config(args.config), args.out, args.photons_per_pixel, args.repeats, candidate
    )
    print(f"Pattern simulation comparison written to {args.out}")
    for row in report["rows"]:
        if row["mean_visibility"] is None:
            print(f"{row['pattern']}: non-carrier pattern; phase/T carrier metrics=n/a")
        elif row["phase_rmse_rad"] is None:
            print(f"{row['pattern']}: V={row['mean_visibility']:.3f}; phase/T carrier metrics=n/a")
        else:
            print(f"{row['pattern']}: V={row['mean_visibility']:.3f}, phase={row['phase_rmse_rad']:.4g}, T={row['transmission_rmse']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
