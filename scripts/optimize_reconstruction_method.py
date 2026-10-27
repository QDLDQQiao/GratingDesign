"""Select reconstruction settings from a phase/T/dark-field noise benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import DesignConfig, load_config
from modality_comparison import ComparisonConfig, default_modalities, evaluate_modality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep reconstruction settings for phase, T, and directional D.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--design-report", type=Path, default=ROOT / "results" / "design_run" / "design_report.json")
    parser.add_argument("--photons", type=float, default=3000.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "reconstruction_method_optimization")
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

    modality = default_modalities(design)[0]
    rows = []
    for smoothing in (0.0, 5.0, 8.0):
        for dark_smoothing in (8.0, 12.0, 16.0, 20.0, 24.0):
            for window_sigma in (0.0, 2.0):
                comparison = ComparisonConfig(
                    photons_per_pixel=[args.photons],
                    repeats=args.repeats,
                    visibility=visibility,
                    smoothing_sigma_px=smoothing,
                    dark_field_smoothing_sigma_px=dark_smoothing,
                    window_sigma_bins=window_sigma,
                )
                metrics = [
                    evaluate_modality(design, modality, comparison, args.photons, repeat)
                    for repeat in range(args.repeats)
                ]
                row = {
                    "smoothing_sigma_px": smoothing,
                    "dark_field_smoothing_sigma_px": dark_smoothing,
                    "window_sigma_bins": window_sigma,
                    "phase_rmse_rad": _mean(metrics, "phase_rmse_rad"),
                    "transmission_rmse": _mean(metrics, "transmission_rmse"),
                    "dark_field_mean_rmse": _mean(metrics, "dark_field_mean_rmse"),
                    "dark_field_anisotropy_rmse": _mean(metrics, "dark_field_anisotropy_rmse"),
                    "dark_field_angle_mae_deg": _mean(metrics, "dark_field_angle_mae_deg"),
                }
                row["score"] = _score(row)
                rows.append(row)

    rows.sort(key=lambda item: item["score"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "reconstruction_method_optimization.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "design": design.__dict__,
        "photons_per_pixel": args.photons,
        "repeats": args.repeats,
        "objective": "normalized sum of phase, T, D mean, D anisotropy, and D angle errors",
        "best": rows[0],
        "rows": rows,
    }
    with (args.out / "reconstruction_method_optimization_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(
        "best: smoothing_sigma_px={smoothing_sigma_px:g}, window_sigma_bins={window_sigma_bins:g}, "
        "score={score:.6g}".format(**rows[0])
    )
    return 0


def _mean(metrics: list[dict], key: str) -> float:
    values = np.asarray([item[key] for item in metrics], dtype=float)
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")


def _score(row: dict) -> float:
    terms = (
        row["phase_rmse_rad"] / 0.02,
        row["transmission_rmse"] / 0.001,
        row["dark_field_mean_rmse"] / 0.002,
        row["dark_field_anisotropy_rmse"] / 0.15,
        row["dark_field_angle_mae_deg"] / 10.0,
    )
    return float(np.nansum(terms))


if __name__ == "__main__":
    raise SystemExit(main())
