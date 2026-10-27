"""Build a broad, reproducible advantage analysis from saved benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze star-like grating advantages across reconstruction and coverage metrics.")
    parser.add_argument("--modality-report", type=Path, default=ROOT / "results" / "modality_comparison" / "modality_comparison_report.json")
    parser.add_argument("--coverage-report", type=Path, default=ROOT / "results" / "coverage_comparison" / "coverage_comparison_report.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "grating_advantage_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modality = _load_json(args.modality_report)
    coverage = _load_json(args.coverage_report)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    geometry = _geometry_summary(modality["modalities"])
    dose_rows = modality["rows"]
    coverage_rows = coverage["rows"]
    coverage_summary = coverage["summary"]
    selected_photons = max(float(row["photons_per_pixel"]) for row in dose_rows)
    high_dose = [row for row in dose_rows if float(row["photons_per_pixel"]) == selected_photons]
    high_dose_advantage = _high_dose_advantage(high_dose)
    dose_advantage = _dose_advantage(dose_rows)
    coverage_metrics = _coverage_metrics(coverage_rows, coverage_summary)
    overall = _overall_summary(geometry, high_dose_advantage, dose_advantage, coverage_metrics)

    report = {
        "inputs": {
            "modality_report": str(args.modality_report),
            "coverage_report": str(args.coverage_report),
        },
        "geometry": geometry,
        "high_dose_advantage": high_dose_advantage,
        "dose_advantage": dose_advantage,
        "coverage": coverage_metrics,
        "overall_summary": overall,
        "files": {
            "summary_csv": "grating_advantage_summary.csv",
            "figure": "grating_advantage_summary.png",
            "coverage_figure": "grating_advantage_coverage.png",
        },
    }
    with (out / "grating_advantage_analysis_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=True)
    _write_summary_csv(out / "grating_advantage_summary.csv", high_dose_advantage, coverage_metrics)
    _save_summary_figure(out, dose_rows, high_dose_advantage, coverage_metrics)
    _save_coverage_figure(out, coverage_rows)
    print(f"outputs={out}")
    print(json.dumps(overall, indent=2))
    return 0


def _geometry_summary(modalities: list[dict]) -> list[dict]:
    rows = []
    for item in modalities:
        directions = np.asarray([_unit(carrier) for carrier in item["carriers"]], dtype=float)
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        design = np.stack([np.ones_like(angles), np.cos(2 * angles), np.sin(2 * angles)], axis=1)
        rank = int(np.linalg.matrix_rank(design, tol=1e-10))
        singular = np.linalg.svd(design, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if rank == 3 else float("inf")
        rows.append({
            "modality": item["name"],
            "directions": len(item["carriers"]),
            "exposures": int(item["exposures"]),
            "dark_field_model_rank": rank,
            "dark_field_design_condition": condition,
            "directional_dark_field_identifiable": rank >= 3,
        })
    return rows


def _high_dose_advantage(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda row: row["modality"])
    star = next(row for row in rows if row["modality"] == "star_4dir_single")
    result = []
    for row in rows:
        result.append({
            "modality": row["modality"],
            "photons_per_pixel": row["photons_per_pixel"],
            "phase_rmse_rad": row["phase_rmse_rad_mean"],
            "transmission_rmse": row["transmission_rmse_mean"],
            "dark_field_mean_rmse": row["dark_field_mean_rmse_mean"],
            "dark_field_anisotropy_rmse": row["dark_field_anisotropy_rmse_mean"],
            "dark_field_angle_mae_deg": row["dark_field_angle_mae_deg_mean"],
            "relative_to_star_phase": _ratio(row["phase_rmse_rad_mean"], star["phase_rmse_rad_mean"]),
            "relative_to_star_transmission": _ratio(row["transmission_rmse_mean"], star["transmission_rmse_mean"]),
            "relative_to_star_dark_field_mean": _ratio(row["dark_field_mean_rmse_mean"], star["dark_field_mean_rmse_mean"]),
            "relative_to_star_dark_field_angle": _ratio(row["dark_field_angle_mae_deg_mean"], star["dark_field_angle_mae_deg_mean"]),
            "directional_dark_field_identifiable": bool(row["directional_dark_field_identifiable"]),
        })
    return result


def _dose_advantage(rows: list[dict]) -> list[dict]:
    modalities = sorted({row["modality"] for row in rows})
    out = []
    for modality in modalities:
        subset = sorted((row for row in rows if row["modality"] == modality), key=lambda row: row["photons_per_pixel"])
        out.append({
            "modality": modality,
            "photons_per_pixel": [row["photons_per_pixel"] for row in subset],
            "phase_rmse_rad": [row["phase_rmse_rad_mean"] for row in subset],
            "transmission_rmse": [row["transmission_rmse_mean"] for row in subset],
            "dark_field_mean_rmse": [row["dark_field_mean_rmse_mean"] for row in subset],
            "dark_field_anisotropy_rmse": [row["dark_field_anisotropy_rmse_mean"] for row in subset],
            "dark_field_angle_mae_deg": [row["dark_field_angle_mae_deg_mean"] for row in subset],
        })
    return out


def _coverage_metrics(rows: list[dict], summary: list[dict]) -> dict:
    result = []
    for item in summary:
        subset = [row for row in rows if row["pattern"] == item["pattern"]]
        result.append({
            **item,
            "mean_contamination": float(np.mean([row["contamination_mean"] for row in subset])),
            "pass_count": int(sum(bool(row["pass_coverage"]) for row in subset)),
            "total_conditions": len(subset),
            "energy_count": len({row["energy_keV"] for row in subset}),
            "distance_count": len({row["distance_m"] for row in subset}),
        })
    return {"patterns": result}


def _overall_summary(geometry: list[dict], high_dose: list[dict], dose: list[dict], coverage: dict) -> dict:
    star_geo = next(row for row in geometry if row["modality"] == "star_4dir_single")
    star_cov = next(row for row in coverage["patterns"] if row["pattern"] == "star_4dir")
    grid_cov = next(row for row in coverage["patterns"] if row["pattern"] == "grid_2dir")
    star_high = next(row for row in high_dose if row["modality"] == "star_4dir_single")
    grid_high = next(row for row in high_dose if row["modality"] == "grid_2dir_single")
    return {
        "star_single_shot_directional_dark_field": star_geo["directional_dark_field_identifiable"],
        "star_dark_field_model_rank": star_geo["dark_field_model_rank"],
        "star_vs_grid_phase_rmse_reduction_fraction": 1.0 - star_high["phase_rmse_rad"] / grid_high["phase_rmse_rad"],
        "star_vs_grid_transmission_rmse_reduction_fraction": 1.0 - star_high["transmission_rmse"] / grid_high["transmission_rmse"],
        "star_coverage_fraction": star_cov["coverage_fraction"],
        "grid_coverage_fraction": grid_cov["coverage_fraction"],
        "star_coverage_advantage_fraction_points": star_cov["coverage_fraction"] - grid_cov["coverage_fraction"],
        "star_coverage_vs_grid_visibility_tradeoff": "star has lower raw mean visibility but covers four directions; grid has higher visibility but only two directions",
        "main_advantage": "single-shot rank-three directional dark-field with overdetermined phase-gradient fusion",
        "main_tradeoff": "four simultaneous carriers divide modulation budget and reduce raw visibility versus two-direction patterns",
    }


def _save_summary_figure(out: Path, dose_rows: list[dict], high_dose: list[dict], coverage: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "star_4dir_single": "#c44e52",
        "grid_2dir_single": "#4c72b0",
        "checkerboard_2dir_single": "#55a868",
        "checkerboard_4dir_2shot": "#8172b2",
    }
    labels = {
        "star_4dir_single": "Star 4-dir / 1 shot",
        "grid_2dir_single": "Grid 2-dir / 1 shot",
        "checkerboard_2dir_single": "Checkerboard 2-dir / 1 shot",
        "checkerboard_4dir_2shot": "Checkerboard 4-dir / 2 shots",
    }
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    metrics = [
        ("phase_rmse_rad_mean", "Phase RMSE (rad)"),
        ("transmission_rmse_mean", "Transmission RMSE"),
        ("dark_field_mean_rmse_mean", "Mean D RMSE"),
        ("dark_field_anisotropy_rmse_mean", "D anisotropy RMSE"),
        ("dark_field_angle_mae_deg_mean", "D angle MAE (deg)"),
    ]
    for ax, (key, ylabel) in zip(axes.ravel()[:5], metrics):
        for modality in labels:
            subset = sorted((row for row in dose_rows if row["modality"] == modality), key=lambda row: row["photons_per_pixel"])
            y = np.asarray([row[key] for row in subset], dtype=float)
            if np.all(np.isnan(y)):
                continue
            ax.plot([row["photons_per_pixel"] for row in subset], y, marker="o", label=labels[modality], color=colors[modality])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total photons/pixel")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)

    ax = axes[1, 2]
    names = [row["modality"] for row in high_dose]
    star = next(row for row in high_dose if row["modality"] == "star_4dir_single")
    keys = [("phase_rmse_rad", "phase"), ("transmission_rmse", "T"), ("dark_field_mean_rmse", "D mean"), ("dark_field_anisotropy_rmse", "D anisotropy"), ("dark_field_angle_mae_deg", "D angle")]
    x = np.arange(len(names))
    width = 0.15
    for i, (key, label) in enumerate(keys):
        values = [row[key] / star[key] if np.isfinite(row[key]) and star[key] else np.nan for row in high_dose]
        ax.bar(x + (i - 2) * width, values, width, label=label)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [labels.get(name, name).replace(" / ", "\n") for name in names], rotation=0, fontsize=7)
    ax.set_ylabel("error relative to star (1 = star)")
    ax.set_title("High-dose error ratio")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Star-like grating: reconstruction accuracy, noise robustness, and trade-offs")
    fig.savefig(out / "grating_advantage_summary.png", dpi=190)
    plt.close(fig)


def _save_coverage_figure(out: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    patterns = sorted({row["pattern"] for row in rows})
    energies = sorted({row["energy_keV"] for row in rows})
    distances = sorted({row["distance_m"] for row in rows})
    fig, axes = plt.subplots(2, len(patterns), figsize=(4.2 * len(patterns), 8), constrained_layout=True)
    for col, pattern in enumerate(patterns):
        for row_idx, (metric, label, cmap) in enumerate((("min_visibility", "minimum visibility", "viridis"), ("contamination_mean", "carrier contamination", "magma"))):
            image = np.full((len(distances), len(energies)), np.nan)
            for item in rows:
                if item["pattern"] == pattern:
                    image[distances.index(item["distance_m"]), energies.index(item["energy_keV"])] = item[metric]
            im = axes[row_idx, col].imshow(image, origin="lower", aspect="auto", cmap=cmap)
            axes[row_idx, col].set_title(pattern if row_idx == 0 else "")
            axes[row_idx, col].set_xlabel("energy (keV)")
            axes[row_idx, col].set_ylabel("distance (m)")
            axes[row_idx, col].set_xticks(range(len(energies)), [f"{x:g}" for x in energies], rotation=45)
            axes[row_idx, col].set_yticks(range(len(distances)), [f"{x:g}" for x in distances])
            fig.colorbar(im, ax=axes[row_idx, col], fraction=0.046, label=label)
    fig.suptitle("Energy-distance robustness: visibility and contamination")
    fig.savefig(out / "grating_advantage_coverage.png", dpi=190)
    plt.close(fig)


def _write_summary_csv(path: Path, high_dose: list[dict], coverage: dict) -> None:
    rows = []
    for item in high_dose:
        rows.append({"section": "high_dose_reconstruction", **item})
    for item in coverage["patterns"]:
        rows.append({"section": "energy_distance_coverage", **item})
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _unit(carrier: list[int]) -> tuple[float, float]:
    kx, ky = carrier
    norm = float(np.hypot(kx, ky))
    return kx / norm, ky / norm


def _ratio(value: float, reference: float) -> float:
    if not np.isfinite(value) or reference == 0 or not np.isfinite(reference):
        return float("nan")
    return float(value / reference)


if __name__ == "__main__":
    raise SystemExit(main())
