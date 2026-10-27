"""Optimize and export binary S4, H3, and CUC-hex pattern masks.

The exported raster is sampled at the requested detector resolution. The GDS
uses reusable tile references and the metadata records the material-dependent
pi height needed for conversion to the actual process stack.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import xraydb

from grating_design import DesignConfig, binarize_field, optimize_design, save_result
from h3_optimization import H3OptimizationConfig, optimize_h3_family
from pattern_comparison import Candidate, default_patterns
from visualize import save_visualizations


HC_KEV_M = 1.2398419843320026e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize and export binary S4, H3, and CUC-hex masks.")
    parser.add_argument("--energy-kev", type=float, default=14.0)
    parser.add_argument("--material", default="Si")
    parser.add_argument("--density-g-cm3", type=float, default=2.329)
    parser.add_argument("--phase-change-rad", type=float, default=np.pi)
    parser.add_argument("--width-mm", type=float, default=13.3)
    parser.add_argument("--height-mm", type=float, default=13.3)
    parser.add_argument("--pixel-um", type=float, default=0.05, help="Real export raster pixel size in um.")
    parser.add_argument("--simulation-pixel-um", type=float, default=0.5, help="Small optimization/simulation pixel size in um.")
    parser.add_argument("--simulation-supercells", type=int, default=2, help="Small periodic simulation window size in supercells per axis.")
    parser.add_argument("--z-m", type=float, default=0.4)
    parser.add_argument("--supercell-um", type=float, default=56.0)
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--fill-factor", type=float, default=0.5)
    parser.add_argument("--s4-iterations", type=int, default=160)
    parser.add_argument("--h3-iterations", type=int, default=160)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--n-cores", type=int, default=16, help="Worker processes used by the optimization stages.")
    parser.add_argument("--png-center-px", type=int, default=1024, help="Side length of the central input crop in raster pixels.")
    parser.add_argument("--png-output-px", type=int, default=2048, help="Side length of each central PNG output.")
    parser.add_argument("--export-chunk-rows", type=int, default=64, help="Bounded row chunk size for streamed NPY export.")
    parser.add_argument("--tiff-tile-px", type=int, default=1024, help="Square TIFF tile size in pixels; must be a multiple of 16.")
    parser.add_argument("--save_npy", action="store_true", help="Save full real-export boolean NPY masks.")
    parser.add_argument("--save_tiff", action="store_true", help="Save full real-export boolean TIFF masks.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "fabrication_14kev")
    return parser.parse_args()


def _make_config(args: argparse.Namespace) -> tuple[DesignConfig, dict]:
    if args.energy_kev <= 0 or args.pixel_um <= 0 or args.simulation_pixel_um <= 0 or args.supercell_um <= 0:
        raise ValueError("energy, pixel, and supercell dimensions must be positive")
    if args.simulation_supercells < 1:
        raise ValueError("simulation-supercells must be positive")
    simulation_samples = int(round(args.supercell_um / args.simulation_pixel_um))
    if not np.isclose(args.supercell_um / simulation_samples, args.simulation_pixel_um, rtol=0.0, atol=1e-9):
        raise ValueError("supercell_um must be an integer multiple of simulation-pixel-um")
    wavelength_m = HC_KEV_M / args.energy_kev
    delta, beta, attenuation_length_cm = xraydb.xray_delta_beta(
        args.material, args.density_g_cm3, args.energy_kev * 1000.0
    )
    pi_height_m = wavelength_m * args.phase_change_rad / (2.0 * np.pi * delta)
    config = DesignConfig(
        supercell_um=args.supercell_um,
        m=args.m,
        n=args.n,
        samples_per_supercell=simulation_samples,
        n_supercells=args.simulation_supercells,
        image_size_px=None,
        wavelength_m=wavelength_m,
        z_m=args.z_m,
        fill_factor=args.fill_factor,
        phase_depth_rad=args.phase_change_rad,
        detector_harmonic=2,
        guard_bins=2,
        kernel="asm",
    )
    physics = {
        "energy_kev": args.energy_kev,
        "material": args.material,
        "density_g_cm3": args.density_g_cm3,
        "delta": float(delta),
        "beta": float(beta),
        "wavelength_m": wavelength_m,
        "attenuation_length_mm": float(attenuation_length_cm * 10.0),
        "requested_phase_change_rad": args.phase_change_rad,
        "phase_height_m": float(pi_height_m),
        "phase_height_um": float(pi_height_m * 1e6),
        "simulation_pixel_um": args.simulation_pixel_um,
        "simulation_supercells": args.simulation_supercells,
        "export_pixel_um": args.pixel_um,
    }
    return config, physics


def _make_real_export_base_masks(
    config: DesignConfig,
    selected: dict[str, Candidate | None],
    export_pixel_um: float,
) -> dict[str, np.ndarray]:
    """Regenerate one periodic tile at the real export raster resolution."""

    export_samples = int(round(config.supercell_um / export_pixel_um))
    if not np.isclose(config.supercell_um / export_samples, export_pixel_um, rtol=0.0, atol=1e-9):
        raise ValueError("supercell_um must be an integer multiple of pixel-um for real export generation")
    # Every carrier has integral indices, so one supercell is the minimal
    # periodic export tile. Broadcast 1-D axes to avoid two extra full grids.
    tile_pixels = export_samples
    axis = np.arange(tile_pixels, dtype=np.float64) * export_pixel_um / config.supercell_um
    x = axis[None, :]
    y = axis[:, None]
    patterns = {pattern.name: pattern for pattern in default_patterns(config)}
    output = {}
    for name, candidate in selected.items():
        pattern = patterns[name]
        if name == "CUC_hex":
            u = (6.0 * x) % 1.0
            v = (3.0 * x + 5.0 * y) % 1.0
            distance = np.sqrt(np.minimum(u, 1.0 - u) ** 2 + np.minimum(v, 1.0 - v) ** 2)
            phase_mask = distance <= math.sqrt(config.fill_factor / math.pi)
        else:
            if candidate is None or len(candidate.amplitudes) != len(pattern.carriers):
                amplitudes = np.ones(len(pattern.carriers), dtype=np.float64)
                phases = np.zeros(len(pattern.carriers), dtype=np.float64)
            else:
                amplitudes = np.asarray(candidate.amplitudes, dtype=np.float64)
                phases = np.asarray(candidate.phases_rad, dtype=np.float64)
            field = np.zeros((tile_pixels, tile_pixels), dtype=np.float64)
            for amplitude, phase, (kx, ky) in zip(amplitudes, phases, pattern.carriers):
                field += amplitude * np.cos(2.0 * np.pi * (kx * x + ky * y) + phase)
            phase_mask = binarize_field(field, config.fill_factor)
        output[name] = np.asarray(phase_mask, dtype=bool)
    return output


def _tiled_window(base_mask: np.ndarray, row_start: int, row_count: int, col_start: int, col_count: int) -> np.ndarray:
    """Return a bounded window of the infinite periodic binary mask."""

    row_indices = np.arange(row_start, row_start + row_count, dtype=np.intp) % base_mask.shape[0]
    col_indices = np.arange(col_start, col_start + col_count, dtype=np.intp) % base_mask.shape[1]
    return np.ascontiguousarray(base_mask[np.ix_(row_indices, col_indices)], dtype=bool)


def _iter_tiled_row_chunks(base_mask: np.ndarray, rows: int, cols: int, chunk_rows: int):
    if chunk_rows < 1:
        raise ValueError("export-chunk-rows must be positive")
    col_indices = np.arange(cols, dtype=np.intp) % base_mask.shape[1]
    for row_start in range(0, rows, chunk_rows):
        row_count = min(chunk_rows, rows - row_start)
        row_indices = np.arange(row_start, row_start + row_count, dtype=np.intp) % base_mask.shape[0]
        yield row_start, np.ascontiguousarray(base_mask[np.ix_(row_indices, col_indices)], dtype=bool)


def _minimum_run_pixels(mask: np.ndarray) -> int:
    """Return the shortest constant binary run along either image axis."""

    mask = np.asarray(mask, dtype=bool)
    lengths = []
    for rows in (mask, mask.T):
        for row in rows:
            boundaries = np.flatnonzero(np.r_[True, row[:-1] != row[1:], True])
            lengths.extend(np.diff(boundaries).tolist())
    return int(min(lengths)) if lengths else 0


def _resolution_report(config: DesignConfig, base_masks: dict[str, np.ndarray], pixel_um: float) -> dict:
    carrier_sets = {
        "S4": config.carriers,
        "H3": ((6, 0), (3, 5), (-3, 5)),
        "CUC_hex": ((6, 0), (3, 5)),
    }
    carrier_periods = {}
    for name, carriers in carrier_sets.items():
        periods_um = [config.supercell_um / float(np.hypot(kx, ky)) for kx, ky in carriers]
        carrier_periods[name] = {
            "period_um": [float(period) for period in periods_um],
            "samples_per_period": [float(period / pixel_um) for period in periods_um],
            "minimum_samples_per_period": float(min(period / pixel_um for period in periods_um)),
        }
    shortest_period_um = min(
        period
        for values in carrier_periods.values()
        for period in values["period_um"]
    )
    minimum_runs = {name: _minimum_run_pixels(mask) for name, mask in base_masks.items()}
    return {
        "pixel_um": pixel_um,
        "carrier_periods": carrier_periods,
        "minimum_binary_run_pixels": minimum_runs,
        "minimum_binary_run_um": {name: float(run * pixel_um) for name, run in minimum_runs.items()},
        "carrier_sampling_status": "marginal" if min(v["minimum_samples_per_period"] for v in carrier_periods.values()) < 8.0 else "adequate",
        "binary_edge_sampling_status": "under_resolved" if min(minimum_runs.values()) < 4 else "adequate",
        "recommended_layout_pixel_um": float(shortest_period_um / 16.0),
        "recommendation": "Use approximately 0.5 um or finer layout sampling for fabrication; 2 um is a detector-resolution raster and is not sufficient for clean binary edge geometry.",
    }


def _require_free_space(path: Path, required_bytes: int) -> None:
    free_bytes = shutil.disk_usage(path.parent).free
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free disk space for {path.name}: need at least "
            f"{required_bytes / 1024**3:.1f} GiB, have {free_bytes / 1024**3:.1f} GiB"
        )


def _save_tiled_npy(base_mask: np.ndarray, path: Path, rows: int, cols: int, chunk_rows: int) -> None:
    """Stream the periodic full aperture to an on-disk NPY memmap."""

    required_bytes = rows * cols
    _require_free_space(path, required_bytes)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.bool_, shape=(rows, cols))
    try:
        for row_start, chunk in _iter_tiled_row_chunks(base_mask, rows, cols, chunk_rows):
            output[row_start:row_start + chunk.shape[0]] = chunk
        output.flush()
    finally:
        del output
    os.replace(temporary, path)


def _iter_tiff_tiles(base_mask: np.ndarray, rows: int, cols: int, tile_px: int):
    if tile_px < 16 or tile_px % 16:
        raise ValueError("tiff-tile-px must be a multiple of 16 and at least 16")
    for row_start in range(0, rows, tile_px):
        row_count = min(tile_px, rows - row_start)
        for col_start in range(0, cols, tile_px):
            col_count = min(tile_px, cols - col_start)
            tile = np.zeros((tile_px, tile_px), dtype=np.uint8)
            tile[:row_count, :col_count] = _tiled_window(
                base_mask, row_start, row_count, col_start, col_count
            ).astype(np.uint8) * 255
            yield tile


def _save_tiled_tiff(base_mask: np.ndarray, path: Path, rows: int, cols: int, tile_px: int) -> None:
    """Write a tiled BigTIFF without retaining the full-aperture image in RAM."""

    # Compression determines final size, but reserve the uncompressed size to
    # avoid failing part-way through a multi-hour fabrication export.
    _require_free_space(path, rows * cols)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    with tifffile.TiffWriter(temporary, bigtiff=rows * cols > 4 * 1024**3) as writer:
        writer.write(
            _iter_tiff_tiles(base_mask, rows, cols, tile_px),
            shape=(rows, cols),
            dtype=np.uint8,
            tile=(tile_px, tile_px),
            compression="zlib",
            photometric="minisblack",
            metadata=None,
        )
    os.replace(temporary, path)


def _save_center_png(mask: np.ndarray, path: Path, crop_px: int, output_px: int) -> None:
    """Save a label-free, nearest-neighbor enlargement of the central mask."""

    if crop_px < 1 or output_px < 1:
        raise ValueError("PNG crop and output sizes must be positive")
    crop_px = min(crop_px, mask.shape[0], mask.shape[1])
    y0 = (mask.shape[0] - crop_px) // 2
    x0 = (mask.shape[1] - crop_px) // 2
    center = np.asarray(mask[y0:y0 + crop_px, x0:x0 + crop_px], dtype=np.uint8) * 255
    image = Image.fromarray(center, mode="L")
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    image = image.resize((output_px, output_px), resample=nearest)
    image.save(path, dpi=(600, 600), optimize=True)


def _save_center_png_from_base(
    base_mask: np.ndarray,
    rows: int,
    cols: int,
    path: Path,
    crop_px: int,
    output_px: int,
) -> None:
    """Save a central PNG directly from the periodic base tile."""

    if crop_px < 1 or output_px < 1:
        raise ValueError("PNG crop and output sizes must be positive")
    crop_px = min(crop_px, rows, cols)
    center = _tiled_window(base_mask, (rows - crop_px) // 2, crop_px, (cols - crop_px) // 2, crop_px)
    _save_center_png(center, path, crop_px, output_px)


def _preview_from_base(base_mask: np.ndarray, rows: int, cols: int, max_size: int = 1200) -> np.ndarray:
    """Subsample the periodic aperture to a bounded overview array."""

    step = max(1, math.ceil(max(rows, cols) / max_size))
    row_indices = np.arange(0, rows, step, dtype=np.intp) % base_mask.shape[0]
    col_indices = np.arange(0, cols, step, dtype=np.intp) % base_mask.shape[1]
    return np.ascontiguousarray(base_mask[np.ix_(row_indices, col_indices)], dtype=bool)


def _gds_real8(value: float) -> bytes:
    """Encode a positive/negative number in the GDSII base-16 real format."""

    if value == 0.0:
        return b"\x00" * 8
    sign = 0x80 if value < 0.0 else 0
    value = abs(value)
    exponent = int(np.floor(np.log(value) / np.log(16.0))) + 1
    mantissa = value / (16.0 ** exponent)
    mantissa_int = int(round(mantissa * (1 << 56)))
    if mantissa_int >= (1 << 56):
        mantissa_int >>= 4
        exponent += 1
    exponent_byte = exponent + 64
    if not 0 <= exponent_byte <= 127:
        raise ValueError("GDS real8 exponent out of range")
    return bytes([sign | exponent_byte]) + mantissa_int.to_bytes(7, byteorder="big")


def _gds_record(record_type: int, data_type: int, payload: bytes = b"") -> bytes:
    # Call sites use the conventional GDS tag notation, e.g. 0x1003 for
    # XY/int4. The binary record header stores the two bytes separately.
    if record_type > 0xFF:
        embedded_type, embedded_data = record_type >> 8, record_type & 0xFF
        if embedded_data != data_type:
            raise ValueError("GDS record tag and data type disagree")
        record_type = embedded_type
    length = 4 + len(payload)
    if length % 2 or length > 0xFFFF:
        raise ValueError("invalid GDS record payload length")
    return struct.pack(">HBB", length, record_type, data_type) + payload


def _gds_ascii(value: str) -> bytes:
    data = value.encode("ascii")
    return data + (b"\x00" if len(data) % 2 else b"")


def _gds_dates() -> bytes:
    now = datetime.now()
    values = [now.year, now.month, now.day, now.hour, now.minute, now.second] * 2
    return struct.pack(">12h", *values)


def _gds_xy(points_um: list[tuple[float, float]], db_per_um: float) -> bytes:
    values = []
    for x, y in points_um:
        values.extend((int(round(x * db_per_um)), int(round(y * db_per_um))))
    return struct.pack(f">{len(values)}i", *values)


def _write_gds_boundary(fh, mask: np.ndarray, pixel_um: float, db_per_um: float, layer: int = 1) -> None:
    """Write horizontal-run rectangles for one reusable GDS tile cell."""

    for iy, row in enumerate(np.asarray(mask, dtype=bool)):
        ix = 0
        while ix < row.size:
            if not row[ix]:
                ix += 1
                continue
            start = ix
            while ix < row.size and row[ix]:
                ix += 1
            x0, x1 = start * pixel_um, ix * pixel_um
            y0, y1 = iy * pixel_um, (iy + 1) * pixel_um
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            fh.write(_gds_record(0x0800, 0x00))
            fh.write(_gds_record(0x0D02, 0x02, struct.pack(">h", layer)))
            fh.write(_gds_record(0x0E02, 0x02, struct.pack(">h", 0)))
            fh.write(_gds_record(0x1003, 0x03, _gds_xy(points, db_per_um)))
            fh.write(_gds_record(0x1100, 0x00))


def _write_gds_structure_header(fh, name: str) -> None:
    fh.write(_gds_record(0x0502, 0x02, _gds_dates()))
    fh.write(_gds_record(0x0606, 0x06, _gds_ascii(name)))


def _write_gds_aref(fh, cell_name: str, columns: int, rows: int, origin_um: tuple[float, float], pitch_um: tuple[float, float], db_per_um: float) -> None:
    if columns < 1 or rows < 1:
        return
    ox, oy = origin_um
    px, py = pitch_um
    points = [(ox, oy), (ox + (columns - 1) * px, oy), (ox, oy + (rows - 1) * py)]
    fh.write(_gds_record(0x0B00, 0x00))
    fh.write(_gds_record(0x1206, 0x06, _gds_ascii(cell_name)))
    fh.write(_gds_record(0x1302, 0x02, struct.pack(">2h", columns, rows)))
    fh.write(_gds_record(0x1003, 0x03, _gds_xy(points, db_per_um)))
    fh.write(_gds_record(0x1100, 0x00))


def _write_gds_cell(fh, name: str, mask: np.ndarray, pixel_um: float, db_per_um: float) -> None:
    _write_gds_structure_header(fh, name)
    _write_gds_boundary(fh, mask, pixel_um, db_per_um)
    fh.write(_gds_record(0x0700, 0x00))


def _write_gds_rectangle(fh, x0_um: float, y0_um: float, x1_um: float, y1_um: float, layer: int, db_per_um: float) -> None:
    points = [(x0_um, y0_um), (x1_um, y0_um), (x1_um, y1_um), (x0_um, y1_um), (x0_um, y0_um)]
    fh.write(_gds_record(0x0800, 0x00))
    fh.write(_gds_record(0x0D02, 0x02, struct.pack(">h", layer)))
    fh.write(_gds_record(0x0E02, 0x02, struct.pack(">h", 0)))
    fh.write(_gds_record(0x1003, 0x03, _gds_xy(points, db_per_um)))
    fh.write(_gds_record(0x1100, 0x00))


def _write_gds_polygon(fh, points_um: list[tuple[float, float]], layer: int, db_per_um: float) -> None:
    """Write one closed GDS boundary."""

    if len(points_um) < 3:
        raise ValueError("A GDS polygon requires at least three vertices")
    closed = points_um if points_um[0] == points_um[-1] else [*points_um, points_um[0]]
    fh.write(_gds_record(0x0800, 0x00))
    fh.write(_gds_record(0x0D02, 0x02, struct.pack(">h", layer)))
    fh.write(_gds_record(0x0E02, 0x02, struct.pack(">h", 0)))
    fh.write(_gds_record(0x1003, 0x03, _gds_xy(closed, db_per_um)))
    fh.write(_gds_record(0x1100, 0x00))


def _random_hartmann_vertices(
    hole: dict[str, float],
    hole_shape: str,
    pixel_um: float,
    n_sides: int,
) -> list[tuple[float, float]]:
    """Return a fabrication-grid-aligned vector approximation of one hole."""

    shape = hole_shape.lower()
    size = float(hole["hole_size_um"])
    angle = math.radians(float(hole["rotation_deg"]))
    if shape == "square":
        local = [(-size / 2.0, -size / 2.0), (size / 2.0, -size / 2.0), (size / 2.0, size / 2.0), (-size / 2.0, size / 2.0)]
    elif shape == "regular_polygon":
        if n_sides < 3:
            raise ValueError("n_sides must be at least three")
        local = [
            (size / 2.0 * math.cos(2.0 * math.pi * index / n_sides), size / 2.0 * math.sin(2.0 * math.pi * index / n_sides))
            for index in range(n_sides)
        ]
    elif shape == "circle":
        radius = size / 2.0
        # Bound the sagitta of each chord to no more than half a fabrication pixel.
        if radius > pixel_um / 2.0:
            angle_step = math.acos(max(-1.0, 1.0 - pixel_um / (2.0 * radius)))
            segments = max(16, int(math.ceil(math.pi / angle_step)))
        else:
            segments = 16
        local = [
            (radius * math.cos(2.0 * math.pi * index / segments), radius * math.sin(2.0 * math.pi * index / segments))
            for index in range(segments)
        ]
    else:
        raise ValueError("hole_shape must be 'circle', 'square', or 'regular_polygon'")

    c, s = math.cos(angle), math.sin(angle)
    return [
        (
            round((float(hole["x_um"]) + c * x - s * y) / pixel_um) * pixel_um,
            round((float(hole["y_um"]) + s * x + c * y) / pixel_um) * pixel_um,
        )
        for x, y in local
    ]


def save_random_hartmann_gds(
    holes: list[dict[str, float]],
    path: Path,
    field_size_um: tuple[float, float],
    pixel_um: float,
    cell_name: str,
    hole_shape: str,
    n_sides: int = 6,
    layer: int = 1,
) -> None:
    """Write a sparse Random Hartmann GDS without a full-aperture raster.

    Hole vertices are quantized to ``pixel_um``. This preserves the requested
    fabrication coordinate grid while avoiding a huge raster and millions of
    staircase rectangles for a sparse mask.
    """

    if not holes:
        raise ValueError("holes must contain at least one aperture")
    if pixel_um <= 0 or field_size_um[0] <= 0 or field_size_um[1] <= 0:
        raise ValueError("field_size_um and pixel_um must be positive")
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in cell_name.upper())[:24] or "RANDOM_HARTMANN"
    db_per_um = 1000.0
    height_um, width_um = field_size_um
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(_gds_record(0x0000, 0x02, struct.pack(">h", 600)))
        fh.write(_gds_record(0x0102, 0x02, _gds_dates()))
        fh.write(_gds_record(0x0206, 0x06, _gds_ascii("RANDOM_HARTMANN")))
        fh.write(_gds_record(0x0305, 0x05, _gds_real8(1e-6) + _gds_real8(1e-9)))
        _write_gds_structure_header(fh, safe_name)
        for hole in holes:
            vertices = _random_hartmann_vertices(hole, hole_shape, pixel_um, n_sides)
            translated = [(x + width_um / 2.0, y + height_um / 2.0) for x, y in vertices]
            _write_gds_polygon(fh, translated, layer, db_per_um)
        fh.write(_gds_record(0x0700, 0x00))
        fh.write(_gds_record(0x0400, 0x00))


def save_multilevel_gds(
    levels: np.ndarray,
    path: Path,
    pixel_um: float,
    cell_name: str,
    frame_size_um: tuple[float, float] | None = None,
    frame_width_um: float = 1.0,
    frame_layer: int = 100,
) -> None:
    """Write cumulative multilevel etch masks on GDS layers 1 through N.

    Layer ``i`` contains every pixel with level at least ``i`` and therefore
    represents one aligned incremental etch exposure.  If a support frame is
    requested, it is written as a rectangular outline on ``frame_layer``.
    """

    level_map = np.asarray(levels)
    if level_map.ndim != 2 or not np.issubdtype(level_map.dtype, np.integer):
        raise ValueError("levels must be a two-dimensional integer array")
    if pixel_um <= 0 or frame_width_um <= 0:
        raise ValueError("pixel_um and frame_width_um must be positive")
    maximum_level = int(level_map.max(initial=0))
    if maximum_level < 1:
        raise ValueError("levels must contain at least one etched level")
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in cell_name.upper())[:24] or "KINOFORM"
    db_per_um = 1000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(_gds_record(0x0000, 0x02, struct.pack(">h", 600)))
        fh.write(_gds_record(0x0102, 0x02, _gds_dates()))
        fh.write(_gds_record(0x0206, 0x06, _gds_ascii("KINOFORM_MASKS")))
        fh.write(_gds_record(0x0305, 0x05, _gds_real8(1e-6) + _gds_real8(1e-9)))
        _write_gds_structure_header(fh, safe_name)
        for level in range(1, maximum_level + 1):
            _write_gds_boundary(fh, level_map >= level, pixel_um, db_per_um, layer=level)
        if frame_size_um is not None:
            frame_width, frame_height = frame_size_um
            if frame_width <= 2.0 * frame_width_um or frame_height <= 2.0 * frame_width_um:
                raise ValueError("frame_size_um must exceed twice frame_width_um")
            _write_gds_rectangle(fh, 0.0, 0.0, frame_width, frame_width_um, frame_layer, db_per_um)
            _write_gds_rectangle(fh, 0.0, frame_height - frame_width_um, frame_width, frame_height, frame_layer, db_per_um)
            _write_gds_rectangle(fh, 0.0, frame_width_um, frame_width_um, frame_height - frame_width_um, frame_layer, db_per_um)
            _write_gds_rectangle(fh, frame_width - frame_width_um, frame_width_um, frame_width, frame_height - frame_width_um, frame_layer, db_per_um)
        fh.write(_gds_record(0x0700, 0x00))
        fh.write(_gds_record(0x0400, 0x00))


def save_fixed_depth_gds(
    mask: np.ndarray,
    path: Path,
    pixel_um: float,
    cell_name: str,
    frame_size_um: tuple[float, float] | None = None,
    frame_width_um: float = 1.0,
    frame_layer: int = 100,
) -> None:
    """Write a single-layer fixed-depth etch mask and optional support frame.

    Set ``frame_width_um`` to zero to omit the support-frame geometry.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if not np.any(binary):
        raise ValueError("mask must contain at least one etched pixel")
    if frame_width_um < 0:
        raise ValueError("frame_width_um must be non-negative")
    include_frame = frame_size_um is not None and frame_width_um > 0
    save_multilevel_gds(
        binary.astype(np.uint8),
        path,
        pixel_um,
        cell_name,
        frame_size_um=frame_size_um if include_frame else None,
        frame_width_um=frame_width_um if include_frame else 1.0,
        frame_layer=frame_layer,
    )


def _save_binary_gds(mask: np.ndarray, path: Path, pixel_um: float, rows: int, cols: int, cell_prefix: str) -> None:
    """Write a compact GDSII layout using AREFs for the tiled binary mask."""

    base = np.asarray(mask, dtype=bool)
    base_rows, base_cols = base.shape
    full_rows, rem_rows = divmod(rows, base_rows)
    full_cols, rem_cols = divmod(cols, base_cols)
    db_per_um = 1000.0
    base_width_um = base_cols * pixel_um
    base_height_um = base_rows * pixel_um
    x_full_um = full_cols * base_width_um
    y_full_um = full_rows * base_height_um
    base_name = f"{cell_prefix}_TILE"
    partial_x_name = f"{cell_prefix}_PX"
    partial_y_name = f"{cell_prefix}_PY"
    partial_xy_name = f"{cell_prefix}_PXY"
    top_name = f"{cell_prefix}_TOP"

    with path.open("wb") as fh:
        # HEADER has record type 0 and int2 data type 2.
        fh.write(_gds_record(0x0000, 0x02, struct.pack(">h", 600)))
        fh.write(_gds_record(0x0102, 0x02, _gds_dates()))
        fh.write(_gds_record(0x0206, 0x06, _gds_ascii("GRATING_MASKS")))
        fh.write(_gds_record(0x0305, 0x05, _gds_real8(1e-6) + _gds_real8(1e-9)))
        _write_gds_cell(fh, base_name, base, pixel_um, db_per_um)
        if rem_cols:
            _write_gds_cell(fh, partial_x_name, base[:, :rem_cols], pixel_um, db_per_um)
        if rem_rows:
            _write_gds_cell(fh, partial_y_name, base[:rem_rows, :], pixel_um, db_per_um)
        if rem_rows and rem_cols:
            _write_gds_cell(fh, partial_xy_name, base[:rem_rows, :rem_cols], pixel_um, db_per_um)
        _write_gds_structure_header(fh, top_name)
        _write_gds_aref(fh, base_name, full_cols, full_rows, (0.0, 0.0), (base_width_um, base_height_um), db_per_um)
        if rem_cols:
            _write_gds_aref(fh, partial_x_name, 1, full_rows, (x_full_um, 0.0), (base_width_um, base_height_um), db_per_um)
        if rem_rows:
            _write_gds_aref(fh, partial_y_name, full_cols, 1, (0.0, y_full_um), (base_width_um, base_height_um), db_per_um)
        if rem_rows and rem_cols:
            _write_gds_aref(fh, partial_xy_name, 1, 1, (x_full_um, y_full_um), (base_width_um, base_height_um), db_per_um)
        fh.write(_gds_record(0x0700, 0x00))
        fh.write(_gds_record(0x0400, 0x00))


def _save_overview(masks: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, len(masks), figsize=(12, 4), constrained_layout=True)
    for ax, (name, mask) in zip(np.atleast_1d(axes), masks.items()):
        step = max(1, max(mask.shape) // 1200)
        ax.imshow(mask[::step, ::step], cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("binary raster")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=180, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def _export_one_pattern(payload: tuple) -> tuple[str, np.ndarray]:
    """Build and write one real-resolution pattern, returning only a preview."""

    (
        name,
        base_mask,
        target_rows,
        target_cols,
        pixel_um,
        out_dir,
        save_npy,
        save_tiff,
        png_center_px,
        png_output_px,
        chunk_rows,
        tiff_tile_px,
    ) = payload
    out_dir = Path(out_dir)
    if save_npy:
        _save_tiled_npy(base_mask, out_dir / f"{name}_mask.npy", target_rows, target_cols, chunk_rows)
    if save_tiff:
        _save_tiled_tiff(base_mask, out_dir / f"{name}_mask.tiff", target_rows, target_cols, tiff_tile_px)
    _save_center_png_from_base(base_mask, target_rows, target_cols, out_dir / f"{name}_center.png", png_center_px, png_output_px)
    _save_binary_gds(
        base_mask,
        out_dir / f"{name}_mask.gds",
        pixel_um,
        target_rows,
        target_cols,
        name.upper().replace("-", "_")[:20],
    )
    return name, _preview_from_base(base_mask, target_rows, target_cols)


def main() -> int:
    args = parse_args()
    config, physics = _make_config(args)
    args.out.mkdir(parents=True, exist_ok=True)
    optimization_dir = args.out / "optimization"

    s4_result, s4_history = optimize_design(config, iterations=args.s4_iterations, seed=args.seed, n_cores=args.n_cores)
    save_result(s4_result, s4_history, optimization_dir / "S4")
    save_visualizations(s4_result, s4_history, optimization_dir / "S4")

    h3_config = H3OptimizationConfig(
        iterations=args.h3_iterations,
        shortlisted_candidates=min(16, args.h3_iterations),
        search_photons_per_pixel=3_000.0,
        search_repeats=2,
        validation_photons_per_pixel=10_000.0,
        validation_repeats=5,
        seed=args.seed,
        n_cores=args.n_cores,
    )
    h3_report = optimize_h3_family(config, h3_config, optimization_dir / "H3", s4_result.candidate)

    selected = {
        "S4": s4_result.candidate,
        "H3": Candidate(**h3_report["patterns"]["H3"]["candidate"]),
        "CUC_hex": None,
    }
    target_rows = int(round(args.height_mm * 1000.0 / args.pixel_um))
    target_cols = int(round(args.width_mm * 1000.0 / args.pixel_um))
    # The optimization/simulation used config.shape. Export masks are rebuilt
    # independently at the real export pixel size from the selected candidates.
    base_masks = _make_real_export_base_masks(config, selected, args.pixel_um)
    resolution = _resolution_report(config, base_masks, args.pixel_um)
    export_payloads = [
        (
            name, base_mask, target_rows, target_cols, args.pixel_um, str(args.out), args.save_npy,
            args.save_tiff, args.png_center_px, args.png_output_px, args.export_chunk_rows,
            args.tiff_tile_px,
        )
        for name, base_mask in base_masks.items()
    ]
    # Fabrication writes intentionally stay serial: this bounds RAM and avoids
    # concurrent multi-gigabyte disk writes. --n-cores still parallelizes the
    # small optimization/validation stages above.
    previews = [_export_one_pattern(payload) for payload in export_payloads]
    preview_masks = dict(previews)
    _save_overview(preview_masks, args.out / "patterns_overview.tiff")

    metadata = {
        "physics": physics,
        "requested_geometry": {
            "width_mm": args.width_mm,
            "height_mm": args.height_mm,
            "export_pixel_um": args.pixel_um,
            "z_m": args.z_m,
        },
        "simulation_geometry": {
            "pixel_um": args.simulation_pixel_um,
            "supercells_per_axis": args.simulation_supercells,
            "shape_px": list(config.shape),
            "extent_mm": [config.shape[1] * config.dx_m * 1e3, config.shape[0] * config.dx_m * 1e3],
        },
        "export_geometry": {
            "rows": target_rows,
            "cols": target_cols,
            "actual_width_mm": target_cols * args.pixel_um / 1000.0,
            "actual_height_mm": target_rows * args.pixel_um / 1000.0,
            "base_supercell_um": args.supercell_um,
            "base_tile_pixels": list(next(iter(base_masks.values())).shape),
            "base_supercell_pixels": int(round(args.supercell_um / args.pixel_um)),
            "periodic_base_shape_px": list(config.shape),
            "periodic_base_extent_mm": [config.shape[1] * config.dx_m * 1e3, config.shape[0] * config.dx_m * 1e3],
            "streaming_export": {
                "full_aperture_array_in_memory": False,
                "npy_chunk_rows": args.export_chunk_rows,
                "tiff_tile_px": args.tiff_tile_px,
                "save_npy": args.save_npy,
                "save_tiff": args.save_tiff,
            },
        },
        "design": asdict(config),
        "resolution": resolution,
        "patterns": list(base_masks),
        "files": {
            name: {
                "gds": f"{name}_mask.gds",
                "png": f"{name}_center.png",
                "encoding": "boolean mask; false=0 phase, true=target phase",
                "png_content": f"central {min(args.png_center_px, target_rows, target_cols)} px crop enlarged to {args.png_output_px} px with nearest-neighbor sampling",
            }
            for name in base_masks
        },
        "note": (
            "Optimization and propagation use a small periodic simulation window. The selected candidates are regenerated on a real-pixel "
            "periodic export tile before the full aperture is assembled. "
            "The requested 13.3 mm extent is 237.5 nominal 56 um supercells; the raster therefore ends halfway through a supercell. "
            "The GDS uses reusable tile and partial-tile cells with AREF placement. Use a finer fabrication writer grid and convert "
            "true/false levels to material heights using phase_height_um."
        ),
    }
    if args.save_npy:
        for name in base_masks:
            metadata["files"][name]["npy"] = f"{name}_mask.npy"
    if args.save_tiff:
        for name in base_masks:
            metadata["files"][name]["tiff"] = f"{name}_mask.tiff"
    with (args.out / "fabrication_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Exported masks to {args.out}")
    print(f"Si phase height for {args.energy_kev:g} keV and {args.phase_change_rad:.6g} rad: {physics['phase_height_um']:.6g} um")
    print(f"Raster: {target_cols} x {target_rows} pixels at {args.pixel_um:g} um = {metadata['export_geometry']['actual_width_mm']:.6g} x {metadata['export_geometry']['actual_height_mm']:.6g} mm")
    print(f"Resolution: carrier sampling={resolution['carrier_sampling_status']}, binary edges={resolution['binary_edge_sampling_status']}, recommended layout pixel={resolution['recommended_layout_pixel_um']:.3g} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
