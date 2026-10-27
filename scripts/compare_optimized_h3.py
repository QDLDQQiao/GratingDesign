"""Compare optimized H3 layouts against S4 and grid/circular controls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import load_config
from h3_optimization import compare_optimized_h3
from pattern_comparison import load_star_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare optimized H3/H3-4L candidates at matched phase/T/D reconstruction settings.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--design-report", type=Path, default=ROOT / "results" / "design_run" / "design_report.json")
    parser.add_argument("--optimization-report", type=Path, default=ROOT / "results" / "h3_optimization" / "h3_optimization_report.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "h3_optimized_comparison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_config, s4_candidate = load_star_candidate(args.design_report)
    report = compare_optimized_h3(report_config or load_config(args.config), s4_candidate, args.optimization_report, args.out)
    print(f"Optimized H3 comparison written to {args.out}")
    for row in report["rows"]:
        print(f"{row['pattern']}: Vmin={row['min_visibility']:.3f}, phase={row['phase_rmse_rad']:.4g}, D-angle={row['dark_field_angle_mae_deg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
