"""Run the Phase II carrier-demodulation reconstruction benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import DesignConfig
from phantoms import periodic_directional_dark_field, smooth_absorption, smooth_periodic_phase
from reconstruction import (
    ReconstructionConfig,
    reconstruct_sample,
    save_reconstruction_report,
    synthesize_reference,
    synthesize_sample,
)
from visualize import save_reconstruction_input_summary, save_reconstruction_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic SB-BSTG signal reconstruction benchmark.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "reconstruction_run")
    parser.add_argument("--samples-per-supercell", type=int, default=128)
    parser.add_argument("--n-supercells", type=int, default=2)
    parser.add_argument("--visibility", type=float, default=0.18)
    parser.add_argument("--design-report", type=Path, default=None, help="Use mean visibility/carriers from a design_report.json.")
    parser.add_argument("--phase-amplitude", type=float, default=0.08)
    parser.add_argument("--guard-bins", type=int, default=3)
    parser.add_argument("--smoothing-sigma-px", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    design = DesignConfig(samples_per_supercell=args.samples_per_supercell, n_supercells=args.n_supercells)
    visibility = args.visibility
    carriers = design.detector_carriers
    if args.design_report is not None:
        with args.design_report.open("r", encoding="utf-8") as fh:
            design_report = json.load(fh)
        design = DesignConfig(**design_report["config"])
        visibility = float(design_report["metrics"]["mean_visibility"])
        carriers = design.detector_carriers
    config = ReconstructionConfig(
        dx_m=design.dx_m,
        wavelength_m=design.wavelength_m,
        z_eff_m=design.z_m,
        carriers=carriers,
        samples_per_supercell=design.samples_per_supercell,
        n_supercells=design.n_supercells,
        guard_bins=args.guard_bins,
        visibility=visibility,
        smoothing_sigma_px=args.smoothing_sigma_px,
    )
    shape = design.shape
    truth_phase = smooth_periodic_phase(shape, amplitude_rad=args.phase_amplitude)
    absorption = smooth_absorption(shape)
    truth_transmission = np.exp(-absorption)
    reference = synthesize_reference(config, shape)
    directions = []
    for kx, ky in config.carriers:
        norm = (kx * kx + ky * ky) ** 0.5
        directions.append((kx / norm, ky / norm))
    dark = periodic_directional_dark_field(shape, directions=np.asarray(directions, dtype=float))
    sample = synthesize_sample(config, truth_phase, absorption=absorption, dark_field=dark)
    result = reconstruct_sample(sample, reference, config)
    report = save_reconstruction_report(
        result,
        truth_phase,
        config,
        args.out,
        truth_transmission=truth_transmission,
        truth_dark_field=dark,
    )
    save_reconstruction_summary(
        result,
        truth_phase,
        sample,
        reference,
        args.out / "reconstruction_summary.png",
        truth_transmission=truth_transmission,
        truth_dark_field=dark,
    )
    save_reconstruction_input_summary(sample, reference, args.out / "input_images.png")
    print("Reconstruction benchmark")
    print(f"phase_rmse_rad={report['metrics']['phase_rmse_rad']:.6g}")
    print(f"phase_mae_rad={report['metrics']['phase_mae_rad']:.6g}")
    print(f"transmission_rmse={report['metrics']['transmission_rmse']:.6g}")
    print(f"dark_field_mean_rmse={report['metrics']['dark_field_mean_rmse']:.6g}")
    print(f"dark_field_anisotropy_rmse={report['metrics']['dark_field_anisotropy_rmse']:.6g}")
    print(f"dark_field_angle_mae_deg={report['metrics']['dark_field_angle_mae_deg']:.6g}")
    print(f"gradient_x_rmse={report['metrics']['gradient_x_rmse']:.6g}")
    print(f"gradient_y_rmse={report['metrics']['gradient_y_rmse']:.6g}")
    print(f"outputs={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
