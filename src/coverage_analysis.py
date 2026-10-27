"""Energy and distance coverage comparison for grating carrier patterns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from grating_design import Candidate, DesignConfig, binarize_field, score_intensity
from propagate import propagate


HC_KEV_NM = 1.239841984
Carrier = Tuple[int, int]


@dataclass
class CoveragePattern:
    name: str
    mask_carriers: List[Carrier]
    detector_carriers: List[Carrier]
    description: str


@dataclass
class CoverageConfig:
    energies_keV: List[float]
    distances_m: List[float]
    reference_energy_keV: float = 12.39841984
    visibility_threshold: float = 0.10
    max_balance_cv: float = 0.35
    max_contamination: float = 0.05


def default_coverage_patterns(config: DesignConfig) -> List[CoveragePattern]:
    axial = [(config.m, 0), (0, config.m)]
    diagonal = [(config.n, config.n), (config.n, -config.n)]
    return [
        CoveragePattern(
            "star_4dir",
            axial + diagonal,
            [(config.detector_harmonic * x, config.detector_harmonic * y) for x, y in axial + diagonal],
            "Star-like four-direction carrier pattern.",
        ),
        CoveragePattern(
            "grid_2dir",
            axial,
            [(config.detector_harmonic * x, config.detector_harmonic * y) for x, y in axial],
            "Axis-aligned two-direction grid pattern.",
        ),
        CoveragePattern(
            "checkerboard_2dir",
            diagonal,
            [(config.detector_harmonic * x, config.detector_harmonic * y) for x, y in diagonal],
            "Diagonal two-direction checkerboard-like pattern.",
        ),
        CoveragePattern(
            "checkerboard_4dir_2shot",
            axial + diagonal,
            [(config.detector_harmonic * x, config.detector_harmonic * y) for x, y in axial + diagonal],
            "Two-shot checkerboard/grid four-direction coverage; optical coverage equals combined carriers.",
        ),
    ]


def run_coverage_comparison(
    design: DesignConfig,
    coverage: CoverageConfig,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pattern in default_coverage_patterns(design):
        mask = build_pattern_mask(design, pattern.mask_carriers)
        for energy in coverage.energies_keV:
            wavelength = energy_to_wavelength_m(energy)
            phase_depth = design.phase_depth_rad * (coverage.reference_energy_keV / energy)
            transmission = np.exp(1j * phase_depth * mask.astype(np.float64)).astype(np.complex128)
            for distance in coverage.distances_m:
                intensity = np.abs(propagate(transmission, design.dx_m, wavelength, distance, kernel=design.kernel)) ** 2
                metrics = score_pattern_intensity(design, pattern.detector_carriers, intensity)
                passed = (
                    metrics["min_visibility"] >= coverage.visibility_threshold
                    and metrics["balance_cv"] <= coverage.max_balance_cv
                    and metrics["contamination_mean"] <= coverage.max_contamination
                )
                rows.append(
                    {
                        "pattern": pattern.name,
                        "energy_keV": energy,
                        "distance_m": distance,
                        "mean_visibility": metrics["mean_visibility"],
                        "min_visibility": metrics["min_visibility"],
                        "balance_cv": metrics["balance_cv"],
                        "contamination_mean": metrics["contamination_mean"],
                        "pass_coverage": passed,
                    }
                )

    summary = summarize_coverage(rows)
    _write_csv(out_dir / "coverage_comparison.csv", rows)
    report = {
        "design": asdict(design),
        "coverage_config": asdict(coverage),
        "patterns": [asdict(pattern) for pattern in default_coverage_patterns(design)],
        "summary": summary,
        "rows": rows,
        "files": {
            "csv": "coverage_comparison.csv",
            "heatmap": "coverage_heatmap.png",
        },
    }
    with (out_dir / "coverage_comparison_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def build_pattern_mask(config: DesignConfig, carriers: Sequence[Carrier]) -> np.ndarray:
    ny, nx = config.shape
    y = np.arange(ny, dtype=np.float64) / config.samples_per_supercell
    x = np.arange(nx, dtype=np.float64) / config.samples_per_supercell
    xx, yy = np.meshgrid(x, y)
    field = np.zeros((ny, nx), dtype=np.float64)
    for kx, ky in carriers:
        field += np.cos(2.0 * np.pi * (kx * xx + ky * yy))
    return binarize_field(field, config.fill_factor)


def score_pattern_intensity(config: DesignConfig, detector_carriers: Sequence[Carrier], intensity: np.ndarray) -> dict:
    spectrum = np.fft.fft2(np.asarray(intensity, dtype=np.float64)) / intensity.size
    dc = abs(spectrum[0, 0])
    visibility = []
    contamination = []
    for carrier in detector_carriers:
        iy, ix = _carrier_to_fft_index(config, carrier)
        peak = abs(spectrum[iy, ix])
        visibility.append(float(2.0 * peak / dc))
        contamination.append(_window_contamination(spectrum, iy, ix, config.guard_bins))
    values = np.asarray(visibility, dtype=np.float64)
    mean_visibility = float(values.mean())
    return {
        "mean_visibility": mean_visibility,
        "min_visibility": float(values.min()),
        "balance_cv": float(values.std() / mean_visibility) if mean_visibility else float("inf"),
        "contamination_mean": float(np.mean(contamination)),
        "visibility": visibility,
    }


def summarize_coverage(rows: Sequence[dict]) -> List[dict]:
    summary = []
    for pattern in sorted({row["pattern"] for row in rows}):
        subset = [row for row in rows if row["pattern"] == pattern]
        pass_count = sum(1 for row in subset if row["pass_coverage"])
        summary.append(
            {
                "pattern": pattern,
                "coverage_fraction": pass_count / len(subset),
                "mean_visibility_avg": float(np.mean([row["mean_visibility"] for row in subset])),
                "min_visibility_worst": float(np.min([row["min_visibility"] for row in subset])),
                "balance_cv_worst": float(np.max([row["balance_cv"] for row in subset])),
            }
        )
    return summary


def save_coverage_heatmap(report: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]
    energies = sorted({row["energy_keV"] for row in rows})
    distances = sorted({row["distance_m"] for row in rows})
    patterns = sorted({row["pattern"] for row in rows})
    fig, axes = plt.subplots(1, len(patterns), figsize=(4.2 * len(patterns), 4.2), constrained_layout=True)
    if len(patterns) == 1:
        axes = [axes]
    for ax, pattern in zip(axes, patterns):
        image = np.full((len(distances), len(energies)), np.nan)
        for row in rows:
            if row["pattern"] != pattern:
                continue
            i = distances.index(row["distance_m"])
            j = energies.index(row["energy_keV"])
            image[i, j] = row["min_visibility"]
        im = ax.imshow(image, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(pattern)
        ax.set_xlabel("energy (keV)")
        ax.set_ylabel("distance (m)")
        ax.set_xticks(range(len(energies)))
        ax.set_xticklabels([f"{value:g}" for value in energies], rotation=45)
        ax.set_yticks(range(len(distances)))
        ax.set_yticklabels([f"{value:g}" for value in distances])
        fig.colorbar(im, ax=ax, fraction=0.046, label="min visibility")
    fig.suptitle("Energy-distance coverage by pattern")
    fig.savefig(out_dir / "coverage_heatmap.png", dpi=180)
    plt.close(fig)


def energy_to_wavelength_m(energy_keV: float) -> float:
    return (HC_KEV_NM / energy_keV) * 1.0e-9


def _carrier_to_fft_index(config: DesignConfig, carrier: Carrier) -> Tuple[int, int]:
    ny, nx = config.shape
    n_supercells = config.effective_n_supercells
    kx, ky = carrier
    return (ky * n_supercells) % ny, (kx * n_supercells) % nx


def _window_contamination(spectrum: np.ndarray, iy: int, ix: int, radius: int) -> float:
    ny, nx = spectrum.shape
    peak = abs(spectrum[iy, ix]) ** 2
    if peak == 0.0:
        return float("inf")
    total = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                total += abs(spectrum[(iy + dy) % ny, (ix + dx) % nx]) ** 2
    return float(max(total - peak, 0.0) / peak)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
