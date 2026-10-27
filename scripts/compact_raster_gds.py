"""Losslessly compact a binary NPY mask into a GDSII layout.

The original fabrication exporter writes one rectangle for every horizontal
pixel run. This utility merges runs with the same x extent across adjacent rows
into one taller rectangle. It is exact at the supplied raster pitch and avoids
loading the entire NPY mask into RAM by using NumPy memory mapping.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
import re
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_fabrication_patterns import (
    _gds_ascii,
    _gds_dates,
    _gds_real8,
    _gds_record,
    _gds_xy,
    _write_gds_structure_header,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Losslessly compact a binary NPY raster into GDSII rectangles.")
    parser.add_argument("input_npy", type=Path, help="Boolean or integer 2-D NPY mask; nonzero pixels are written to GDS.")
    parser.add_argument("--pixel-um", type=float, required=True, help="Physical raster pitch in micrometers.")
    parser.add_argument("--output", type=Path, default=None, help="Output GDS path; defaults to <input>_compact.gds.")
    parser.add_argument("--cell-name", default="COMPACT_MASK", help="Top-level GDS structure name.")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--datatype", type=int, default=0)
    parser.add_argument("--invert", action="store_true", help="Write zero-valued pixels instead of nonzero pixels.")
    return parser.parse_args()


def _row_runs(row: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield [start, stop) intervals of set pixels in one raster row."""

    values = np.asarray(row, dtype=bool)
    edges = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
    for start, stop in zip(edges[:-1], edges[1:]):
        if values[start]:
            yield int(start), int(stop)


def iter_merged_rectangles(mask: np.ndarray, invert: bool = False) -> Iterator[tuple[int, int, int, int]]:
    """Yield exact raster rectangles as (x0, x1, y0, y1) pixel bounds."""

    if mask.ndim != 2:
        raise ValueError("input_npy must contain a two-dimensional raster")
    active: dict[tuple[int, int], int] = {}
    for y, source_row in enumerate(mask):
        row = np.asarray(source_row, dtype=bool)
        if invert:
            row = ~row
        runs = set(_row_runs(row))
        for run, y0 in list(active.items()):
            if run not in runs:
                yield run[0], run[1], y0, y
                del active[run]
        for run in runs:
            active.setdefault(run, y)
    final_y = int(mask.shape[0])
    for (x0, x1), y0 in active.items():
        yield x0, x1, y0, final_y


def _safe_cell_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", value.upper())[:32]
    return name or "COMPACT_MASK"


def _write_rectangle(fh, rectangle: tuple[int, int, int, int], pixel_um: float, db_per_um: float, layer: int, datatype: int) -> None:
    x0, x1, y0, y1 = rectangle
    points = [
        (x0 * pixel_um, y0 * pixel_um),
        (x1 * pixel_um, y0 * pixel_um),
        (x1 * pixel_um, y1 * pixel_um),
        (x0 * pixel_um, y1 * pixel_um),
        (x0 * pixel_um, y0 * pixel_um),
    ]
    fh.write(_gds_record(0x0800, 0x00))
    fh.write(_gds_record(0x0D02, 0x02, struct.pack(">h", layer)))
    fh.write(_gds_record(0x0E02, 0x02, struct.pack(">h", datatype)))
    fh.write(_gds_record(0x1003, 0x03, _gds_xy(points, db_per_um)))
    fh.write(_gds_record(0x1100, 0x00))


def compact_mask_to_gds(
    input_npy: Path,
    output_gds: Path,
    pixel_um: float,
    cell_name: str = "COMPACT_MASK",
    layer: int = 1,
    datatype: int = 0,
    invert: bool = False,
) -> int:
    """Write a lossless, vertically merged GDSII representation of an NPY mask."""

    if pixel_um <= 0:
        raise ValueError("pixel-um must be positive")
    if not 0 <= layer <= 32767 or not 0 <= datatype <= 32767:
        raise ValueError("layer and datatype must be between 0 and 32767")
    mask = np.load(input_npy, mmap_mode="r", allow_pickle=False)
    if mask.ndim != 2:
        raise ValueError("input_npy must contain a two-dimensional raster")
    output_gds.parent.mkdir(parents=True, exist_ok=True)
    rectangles = 0
    db_per_um = 1000.0
    cell_name = _safe_cell_name(cell_name)
    with output_gds.open("wb") as fh:
        fh.write(_gds_record(0x0000, 0x02, struct.pack(">h", 600)))
        fh.write(_gds_record(0x0102, 0x02, _gds_dates()))
        fh.write(_gds_record(0x0206, 0x06, _gds_ascii("GRATING_MASKS")))
        fh.write(_gds_record(0x0305, 0x05, _gds_real8(1e-6) + _gds_real8(1e-9)))
        _write_gds_structure_header(fh, cell_name)
        for rectangle in iter_merged_rectangles(mask, invert=invert):
            _write_rectangle(fh, rectangle, pixel_um, db_per_um, layer, datatype)
            rectangles += 1
        fh.write(_gds_record(0x0700, 0x00))
        fh.write(_gds_record(0x0400, 0x00))
    return rectangles


def main() -> int:
    args = parse_args()
    output = args.output or args.input_npy.with_name(f"{args.input_npy.stem}_compact.gds")
    rectangles = compact_mask_to_gds(
        args.input_npy,
        output,
        args.pixel_um,
        cell_name=args.cell_name,
        layer=args.layer,
        datatype=args.datatype,
        invert=args.invert,
    )
    print(f"Wrote {output}")
    print(f"Raster: {np.load(args.input_npy, mmap_mode='r', allow_pickle=False).shape[1]} x {np.load(args.input_npy, mmap_mode='r', allow_pickle=False).shape[0]} px")
    print(f"Merged rectangles: {rectangles:,}")
    print(f"GDS size: {output.stat().st_size / 1024**2:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
