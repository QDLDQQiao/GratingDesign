"""Compare star-like, grid, and checkerboard reconstruction performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import DesignConfig, load_config
from modality_comparison import (
    ComparisonConfig,
    run_modality_comparison,
    save_comparison_plot,
    save_recovered_maps_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run star/grid/checkerboard reconstruction comparison.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--design-report", type=Path, default=ROOT / "results" / "design_run" / "design_report.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "modality_comparison")
    parser.add_argument("--photons", type=float, nargs="+", default=[300.0, 1000.0, 3000.0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoothing-sigma-px", type=float, default=5.0)
    parser.add_argument("--dark-field-smoothing-sigma-px", type=float, default=16.0)
    parser.add_argument("--window-sigma-bins", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    design = load_config(args.config)
    visibility = 0.142470873981641
    if args.design_report.exists():
        with args.design_report.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
        design = DesignConfig(**report["config"])
        visibility = float(report["metrics"]["mean_visibility"])
    comparison = ComparisonConfig(
        photons_per_pixel=args.photons,
        repeats=args.repeats,
        visibility=visibility,
        smoothing_sigma_px=args.smoothing_sigma_px,
        dark_field_smoothing_sigma_px=args.dark_field_smoothing_sigma_px,
        window_sigma_bins=args.window_sigma_bins,
    )
    report = run_modality_comparison(design, comparison, args.out)
    save_comparison_plot(report, args.out)
    save_recovered_maps_figure(design, comparison, args.out, photons_per_pixel=args.photons[-1], repeat=0)
    print("Modality comparison complete")
    print(f"outputs={args.out}")
    for row in report["rows"]:
        if row["photons_per_pixel"] == args.photons[-1]:
            print(
                f"{row['modality']}: phase={row['phase_rmse_rad_mean']:.4g}, "
                f"T={row['transmission_rmse_mean']:.4g}, "
                f"D_angle={row['dark_field_angle_mae_deg_mean']:.3g} deg, "
                f"D_identifiable={row['directional_dark_field_identifiable']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
