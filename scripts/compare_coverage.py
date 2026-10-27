"""Compare energy-distance coverage for different grating patterns."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverage_analysis import CoverageConfig, run_coverage_comparison, save_coverage_heatmap
from grating_design import DesignConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run energy/distance coverage comparison.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "coverage_comparison")
    parser.add_argument("--energies-kev", type=float, nargs="+", default=[10.0, 11.0, 12.4, 14.0, 16.0])
    parser.add_argument("--distances-m", type=float, nargs="+", default=[0.08, 0.16, 0.24, 0.30, 0.40])
    parser.add_argument("--image-size-px", type=int, default=None)
    parser.add_argument("--samples-per-supercell", type=int, default=None)
    parser.add_argument("--visibility-threshold", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    design = load_config(args.config)
    overrides = {}
    if args.image_size_px is not None:
        overrides["image_size_px"] = args.image_size_px
    if args.samples_per_supercell is not None:
        overrides["samples_per_supercell"] = args.samples_per_supercell
    if overrides:
        design = DesignConfig(**{**design.__dict__, **overrides})
    coverage = CoverageConfig(
        energies_keV=args.energies_kev,
        distances_m=args.distances_m,
        visibility_threshold=args.visibility_threshold,
    )
    report = run_coverage_comparison(design, coverage, args.out)
    save_coverage_heatmap(report, args.out)
    print("Coverage comparison complete")
    print(f"outputs={args.out}")
    for row in report["summary"]:
        print(
            f"{row['pattern']}: coverage={row['coverage_fraction']:.3f}, "
            f"avgV={row['mean_visibility_avg']:.4g}, worstMinV={row['min_visibility_worst']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
