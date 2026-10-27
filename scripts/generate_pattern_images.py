"""Export images for selected grating families proposed in ``plan.md``."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import Candidate, load_config
from pattern_comparison import default_patterns, load_star_candidate, simulate_pattern, _save_pattern_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate phase-mask and propagated-intensity images for planned grating families.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.json")
    parser.add_argument("--design-report", type=Path, default=ROOT / "results" / "design_run" / "design_report.json")
    parser.add_argument(
        "--h3-optimization-report",
        type=Path,
        default=None,
        help="Optional H3/H3-4L optimization report. Uses its selected candidates when exporting those patterns.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "pattern_images")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="Optional pattern IDs, for example: S4 CUC_hex H3 H3_4L. Defaults to all patterns.",
    )
    return parser.parse_args()


def load_h3_candidates(report_path: Path | None) -> tuple[dict | None, dict[str, Candidate]]:
    """Load selected H3-family candidates from a joint optimization report."""

    if report_path is None:
        return None, {}
    if not report_path.exists():
        raise FileNotFoundError(f"H3 optimization report does not exist: {report_path}")
    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    patterns = report.get("patterns")
    design = report.get("design")
    if not isinstance(patterns, dict) or not isinstance(design, dict):
        raise ValueError(f"invalid H3 optimization report: {report_path}")
    candidates = {}
    for name in ("H3", "H3_4L"):
        result = patterns.get(name)
        if result is not None and "candidate" in result:
            candidates[name] = Candidate(**result["candidate"])
    if not candidates:
        raise ValueError(f"H3 optimization report contains no H3 candidates: {report_path}")
    return design, candidates


def main() -> int:
    args = parse_args()
    report_config, candidate = load_star_candidate(args.design_report)
    config = report_config or load_config(args.config)
    optimized_design, h3_candidates = load_h3_candidates(args.h3_optimization_report)
    if optimized_design is not None and optimized_design != asdict(config):
        raise ValueError("the H3 optimization report and design report must use the same design configuration")
    patterns = default_patterns(config)
    if args.patterns is not None:
        requested = set(args.patterns)
        available = {pattern.name for pattern in patterns}
        unknown = requested - available
        if unknown:
            raise ValueError(f"unknown pattern IDs: {', '.join(sorted(unknown))}; available: {', '.join(sorted(available))}")
        patterns = [pattern for pattern in patterns if pattern.name in requested]
    simulations = [simulate_pattern(config, pattern, h3_candidates.get(pattern.name, candidate)) for pattern in patterns]
    args.out.mkdir(parents=True, exist_ok=True)
    _save_pattern_figure(simulations, config, args.out / "designed_pattern_images.png")
    for simulation in simulations:
        name = simulation["pattern"]["name"]
        np.save(args.out / f"{name}_phase_map_rad.npy", simulation["phase_map"])
        np.save(args.out / f"{name}_intensity.npy", simulation["intensity"])
    print(f"Pattern images written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
