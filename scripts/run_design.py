"""Run SB-BSTG grating pattern design and optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grating_design import DesignConfig, format_metrics, load_config, optimize_design, save_result
from visualize import save_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize and visualize an SB-BSTG grating pattern.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "design_run", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=80, help="Number of bounded random-search candidates.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable optimization.")
    parser.add_argument("--samples-per-supercell", type=int, default=None, help="Override config grid density.")
    parser.add_argument("--n-supercells", type=int, default=None, help="Override number of tiled supercells.")
    parser.add_argument("--image-size-px", type=int, default=None, help="Override output image width/height in pixels.")
    parser.add_argument("--z-mm", type=float, default=None, help="Override propagation distance in millimeters.")
    parser.add_argument("--n-cores", type=int, default=1, help="Worker processes for independent candidate evaluations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    overrides = {}
    if args.samples_per_supercell is not None:
        overrides["samples_per_supercell"] = args.samples_per_supercell
    if args.n_supercells is not None:
        overrides["n_supercells"] = args.n_supercells
    if args.image_size_px is not None:
        overrides["image_size_px"] = args.image_size_px
    if args.z_mm is not None:
        overrides["z_m"] = args.z_mm * 1e-3
    if overrides:
        config = DesignConfig(**{**config.__dict__, **overrides})

    result, history = optimize_design(config, iterations=args.iterations, seed=args.seed, n_cores=args.n_cores)
    save_result(result, history, args.out)
    save_visualizations(result, history, args.out)

    print("Best design")
    print(format_metrics(result.metrics))
    print(f"amplitudes={result.candidate.amplitudes}")
    print(f"phases_rad={result.candidate.phases_rad}")
    print(f"outputs={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
