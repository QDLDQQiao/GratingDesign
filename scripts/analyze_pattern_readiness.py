"""Run resolution-aware readiness analysis for every pattern in ``plan.md``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_readiness import run_readiness_analysis
from grating_design import load_config
from pattern_comparison import load_star_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all planned patterns for accuracy and reconstruction spatial transfer.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--design-report", type=Path, default=ROOT / "results" / "design_run" / "design_report.json")
    parser.add_argument("--pattern-report", type=Path, default=ROOT / "results" / "pattern_simulation_all_smoke" / "pattern_simulation_report.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "experiment_readiness")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_config, candidate = load_star_candidate(args.design_report)
    report = run_readiness_analysis(report_config or load_config(args.config), candidate, args.pattern_report, args.out)
    print(f"Readiness analysis written to {args.out}")
    for row in report["rows"]:
        print(f"{row['pattern']}: D-resolution={row['dark_field_resolution_um']}, phase-RMSE={row['phase_rmse_rad']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
