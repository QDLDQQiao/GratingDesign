"""SB-BSTG grating pattern design, simulation, and optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from propagate import propagate


Carrier = Tuple[int, int]


@dataclass
class DesignConfig:
    """Numerical and physical settings for a star-carrier binary grating."""

    supercell_um: float = 56.0
    m: int = 7
    n: int = 5
    samples_per_supercell: int = 128
    n_supercells: int = 2
    image_size_px: int | None = None
    wavelength_m: float = 1.0e-10
    z_m: float = 0.08
    fill_factor: float = 0.5
    phase_depth_rad: float = math.pi
    detector_harmonic: int = 2
    guard_bins: int = 2
    kernel: str = "asm"

    @property
    def supercell_m(self) -> float:
        return self.supercell_um * 1.0e-6

    @property
    def dx_m(self) -> float:
        return self.supercell_m / self.samples_per_supercell

    @property
    def shape(self) -> Tuple[int, int]:
        n = self.image_size_px
        if n is None:
            n = self.samples_per_supercell * self.n_supercells
        if n <= 0:
            raise ValueError("image_size_px must be positive")
        if n % self.samples_per_supercell != 0:
            raise ValueError("image_size_px must be a multiple of samples_per_supercell")
        return n, n

    @property
    def effective_n_supercells(self) -> int:
        return self.shape[0] // self.samples_per_supercell

    @property
    def carriers(self) -> Tuple[Carrier, ...]:
        return ((self.m, 0), (0, self.m), (self.n, self.n), (self.n, -self.n))

    @property
    def detector_carriers(self) -> Tuple[Carrier, ...]:
        h = self.detector_harmonic
        return tuple((h * kx, h * ky) for kx, ky in self.carriers)


@dataclass
class Candidate:
    amplitudes: List[float]
    phases_rad: List[float]


@dataclass
class DesignMetrics:
    score: float
    mean_visibility: float
    min_visibility: float
    balance_cv: float
    contamination_mean: float
    fill_factor: float
    visibility: List[float]
    phase_noise_factor: float
    dark_field_noise_factor: float


@dataclass
class DesignResult:
    config: DesignConfig
    candidate: Candidate
    metrics: DesignMetrics
    thickness_map_m: np.ndarray
    transmission: np.ndarray
    intensity: np.ndarray
    design_field: np.ndarray


def default_candidate() -> Candidate:
    return Candidate(amplitudes=[1.0, 1.0, 1.0, 1.0], phases_rad=[0.0, 0.0, 0.0, 0.0])


def make_design_field(config: DesignConfig, candidate: Candidate) -> np.ndarray:
    """Build the continuous star-carrier design field before thresholding."""

    _validate_candidate(config, candidate)
    ny, nx = config.shape
    y = np.arange(ny, dtype=np.float64) / config.samples_per_supercell
    x = np.arange(nx, dtype=np.float64) / config.samples_per_supercell
    xx, yy = np.meshgrid(x, y)
    field = np.zeros((ny, nx), dtype=np.float64)
    for amp, phase, (kx, ky) in zip(candidate.amplitudes, candidate.phases_rad, config.carriers):
        field += amp * np.cos(2.0 * np.pi * (kx * xx + ky * yy) + phase)
    return field


def binarize_field(field: np.ndarray, fill_factor: float) -> np.ndarray:
    """Threshold a continuous field to a binary mask with the requested fill."""

    if not 0.0 < fill_factor < 1.0:
        raise ValueError("fill_factor must be between 0 and 1")
    threshold = np.quantile(field, 1.0 - fill_factor, method="nearest")
    mask = field >= threshold
    target_count = int(round(fill_factor * field.size))
    current = int(mask.sum())
    if current != target_count:
        flat = field.ravel()
        order = np.argsort(flat, kind="mergesort")
        exact = np.zeros(flat.size, dtype=bool)
        exact[order[-target_count:]] = True
        mask = exact.reshape(field.shape)
    return mask


def build_grating(config: DesignConfig, candidate: Candidate) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return thickness map, complex transmission, and continuous design field."""

    field = make_design_field(config, candidate)
    mask = binarize_field(field, config.fill_factor)
    h_pi = config.wavelength_m / 2.0
    thickness = h_pi * mask.astype(np.float64)
    transmission = np.exp(1j * config.phase_depth_rad * mask.astype(np.float64)).astype(np.complex128)
    return thickness, transmission, field


def simulate_design(config: DesignConfig, candidate: Candidate) -> DesignResult:
    thickness, transmission, field = build_grating(config, candidate)
    output = propagate(transmission, config.dx_m, config.wavelength_m, config.z_m, kernel=config.kernel)
    intensity = np.abs(output) ** 2
    metrics = score_intensity(config, intensity)
    metrics.fill_factor = float(np.mean(thickness > 0.0))
    return DesignResult(config, candidate, metrics, thickness, transmission, intensity, field)


def score_intensity(config: DesignConfig, intensity: np.ndarray) -> DesignMetrics:
    spectrum = np.fft.fft2(np.asarray(intensity, dtype=np.float64)) / intensity.size
    dc = abs(spectrum[0, 0])
    if dc == 0.0:
        raise ValueError("zero DC intensity prevents scoring")

    visibility = []
    contamination = []
    phase_noise_terms = []
    dark_noise_terms = []
    for carrier in config.detector_carriers:
        iy, ix = _carrier_to_fft_index(config, carrier)
        peak = abs(spectrum[iy, ix])
        v = float(2.0 * peak / dc)
        visibility.append(v)
        contamination.append(_window_contamination(spectrum, iy, ix, config.guard_bins))
        f_abs = math.hypot(carrier[0], carrier[1]) / config.supercell_m
        phase_noise_terms.append(1.0 / max(v * f_abs, 1e-30))
        dark_noise_terms.append(1.0 / max(v, 1e-30))

    vis = np.asarray(visibility, dtype=np.float64)
    mean_visibility = float(vis.mean())
    min_visibility = float(vis.min())
    balance_cv = float(vis.std() / mean_visibility) if mean_visibility else float("inf")
    contamination_mean = float(np.mean(contamination))
    score = mean_visibility + 0.25 * min_visibility - 0.35 * balance_cv - 0.20 * contamination_mean
    return DesignMetrics(
        score=float(score),
        mean_visibility=mean_visibility,
        min_visibility=min_visibility,
        balance_cv=balance_cv,
        contamination_mean=contamination_mean,
        fill_factor=float(np.mean(intensity > np.median(intensity))),
        visibility=visibility,
        phase_noise_factor=float(np.mean(phase_noise_terms)),
        dark_field_noise_factor=float(np.mean(dark_noise_terms)),
    )


def optimize_design(
    config: DesignConfig,
    iterations: int = 80,
    seed: int = 7,
    amplitude_span: float = 0.35,
    phase_span_rad: float = math.pi,
    n_cores: int = 1,
) -> Tuple[DesignResult, List[dict]]:
    """Run a bounded random search and return the best design and history.

    Candidate scoring is embarrassingly parallel. Workers return only scalar
    metrics, then the winning candidate is simulated once in the parent so
    large propagated arrays are not copied between processes.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if n_cores < 1:
        raise ValueError("n_cores must be positive")
    rng = np.random.default_rng(seed)
    history = []
    candidates = [default_candidate()]
    for _ in range(iterations - 1):
        amplitudes = np.clip(1.0 + rng.uniform(-amplitude_span, amplitude_span, size=4), 0.05, None)
        phases = rng.uniform(-phase_span_rad, phase_span_rad, size=4)
        candidates.append(Candidate(amplitudes=amplitudes.tolist(), phases_rad=phases.tolist()))

    payloads = [(config, candidate) for candidate in candidates]
    if n_cores == 1 or len(candidates) == 1:
        metric_rows = [_score_design_candidate(payload) for payload in payloads]
    else:
        workers = min(n_cores, len(candidates))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                metric_rows = list(executor.map(_score_design_candidate, payloads))
        except (PermissionError, OSError):
            # Windows sandboxed sessions can deny multiprocessing pipes. NumPy
            # releases the GIL in FFT/array kernels, so threads remain useful.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                metric_rows = list(executor.map(_score_design_candidate, payloads))

    best_candidate = None
    best_score = -float("inf")
    history = []
    for idx, (candidate, metrics) in enumerate(zip(candidates, metric_rows)):
        row = {
            "iteration": idx,
            "score": metrics["score"],
            "mean_visibility": metrics["mean_visibility"],
            "min_visibility": metrics["min_visibility"],
            "balance_cv": metrics["balance_cv"],
            "contamination_mean": metrics["contamination_mean"],
            "amplitudes": candidate.amplitudes,
            "phases_rad": candidate.phases_rad,
        }
        history.append(row)
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_candidate = candidate
    assert best_candidate is not None
    return simulate_design(config, best_candidate), history


def _score_design_candidate(payload: tuple[DesignConfig, Candidate]) -> dict:
    """Process-pool worker returning only metrics for one S4 candidate."""

    config, candidate = payload
    return asdict(simulate_design(config, candidate).metrics)


def save_result(result: DesignResult, history: Sequence[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "thickness_map_m.npy", result.thickness_map_m)
    np.save(out_dir / "transmission_real_imag.npy", np.stack([result.transmission.real, result.transmission.imag]))
    np.save(out_dir / "intensity.npy", result.intensity)
    report = {
        "config": asdict(result.config),
        "candidate": asdict(result.candidate),
        "metrics": asdict(result.metrics),
        "history": list(history),
        "files": {
            "thickness_map_m": "thickness_map_m.npy",
            "transmission_real_imag": "transmission_real_imag.npy",
            "intensity": "intensity.npy",
            "summary_plot": "design_summary.png",
            "history_plot": "optimization_history.png",
        },
    }
    with (out_dir / "design_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def load_config(path: Path | None) -> DesignConfig:
    if path is None:
        return DesignConfig()
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return DesignConfig(**data)


def _validate_candidate(config: DesignConfig, candidate: Candidate) -> None:
    if len(candidate.amplitudes) != len(config.carriers):
        raise ValueError("candidate must provide one amplitude per carrier")
    if len(candidate.phases_rad) != len(config.carriers):
        raise ValueError("candidate must provide one phase per carrier")


def _carrier_to_fft_index(config: DesignConfig, carrier: Carrier) -> Tuple[int, int]:
    ny, nx = config.shape
    n_supercells = config.effective_n_supercells
    ky, kx = carrier[1] * n_supercells, carrier[0] * n_supercells
    return ky % ny, kx % nx


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


def format_metrics(metrics: DesignMetrics) -> str:
    return (
        f"score={metrics.score:.4f}, mean_visibility={metrics.mean_visibility:.4f}, "
        f"min_visibility={metrics.min_visibility:.4f}, balance_cv={metrics.balance_cv:.4f}, "
        f"contamination_mean={metrics.contamination_mean:.4f}"
    )
