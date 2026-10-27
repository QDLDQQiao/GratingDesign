from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from grating_design import DesignConfig


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_pattern_images.py"
SPEC = importlib.util.spec_from_file_location("generate_pattern_images", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EXPORT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_fabrication_patterns.py"
EXPORT_SPEC = importlib.util.spec_from_file_location("export_fabrication_patterns", EXPORT_SCRIPT)
assert EXPORT_SPEC is not None and EXPORT_SPEC.loader is not None
EXPORT_MODULE = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(EXPORT_MODULE)

COMPACT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compact_raster_gds.py"
COMPACT_SPEC = importlib.util.spec_from_file_location("compact_raster_gds", COMPACT_SCRIPT)
assert COMPACT_SPEC is not None and COMPACT_SPEC.loader is not None
COMPACT_MODULE = importlib.util.module_from_spec(COMPACT_SPEC)
COMPACT_SPEC.loader.exec_module(COMPACT_MODULE)

DIRECT_COMPACT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compact_generated_gds.py"
DIRECT_COMPACT_SPEC = importlib.util.spec_from_file_location("compact_generated_gds", DIRECT_COMPACT_SCRIPT)
assert DIRECT_COMPACT_SPEC is not None and DIRECT_COMPACT_SPEC.loader is not None
DIRECT_COMPACT_MODULE = importlib.util.module_from_spec(DIRECT_COMPACT_SPEC)
DIRECT_COMPACT_SPEC.loader.exec_module(DIRECT_COMPACT_MODULE)


def test_load_h3_candidates_reads_selected_candidates(tmp_path: Path) -> None:
    report_path = tmp_path / "h3_optimization_report.json"
    report_path.write_text(
        json.dumps(
            {
                "design": asdict(DesignConfig()),
                "patterns": {
                    "H3": {"candidate": {"amplitudes": [1.0, 1.1, 1.2], "phases_rad": [0.0, 0.1, 0.2]}},
                    "H3_4L": {"candidate": {"amplitudes": [1.2, 1.1, 1.0], "phases_rad": [0.3, 0.2, 0.1]}},
                },
            }
        ),
        encoding="utf-8",
    )

    design, candidates = MODULE.load_h3_candidates(report_path)

    assert design == asdict(DesignConfig())
    assert candidates["H3"].amplitudes == [1.0, 1.1, 1.2]
    assert candidates["H3_4L"].phases_rad == [0.3, 0.2, 0.1]


def test_binary_gds_and_center_png_export(tmp_path: Path) -> None:
    mask = np.array([[True, False, True], [False, True, False]], dtype=bool)
    gds_path = tmp_path / "test.gds"
    png_path = tmp_path / "center.png"

    EXPORT_MODULE._save_binary_gds(mask, gds_path, pixel_um=2.0, rows=5, cols=7, cell_prefix="TEST")
    EXPORT_MODULE._save_center_png(mask, png_path, crop_px=2, output_px=16)

    assert gds_path.stat().st_size > 100
    assert gds_path.read_bytes()[2:4] == b"\x00\x02"
    assert b"TEST_TOP" in gds_path.read_bytes()
    assert Image.open(png_path).size == (16, 16)


def test_sparse_random_hartmann_gds_is_grid_quantized(tmp_path: Path) -> None:
    holes = [
        {"x_um": -2.1, "y_um": 1.9, "hole_size_um": 4.0, "rotation_deg": 25.0, "bounding_radius_um": 3.0},
        {"x_um": 3.8, "y_um": -2.6, "hole_size_um": 4.0, "rotation_deg": 25.0, "bounding_radius_um": 3.0},
    ]
    gds_path = tmp_path / "random_hartmann.gds"

    EXPORT_MODULE.save_random_hartmann_gds(
        holes,
        gds_path,
        field_size_um=(20.0, 20.0),
        pixel_um=0.2,
        cell_name="RANDOM",
        hole_shape="square",
    )

    assert gds_path.read_bytes()[2:4] == b"\x00\x02"
    assert gds_path.stat().st_size > 100


def test_streamed_exports_match_periodic_mask(tmp_path: Path) -> None:
    base_mask = np.array([[True, False, False], [False, True, True]], dtype=bool)
    rows, cols = 32, 48
    expected = np.tile(base_mask, (16, 16))[:rows, :cols]
    npy_path = tmp_path / "mask.npy"
    tiff_path = tmp_path / "mask.tiff"
    png_path = tmp_path / "center.png"

    EXPORT_MODULE._save_tiled_npy(base_mask, npy_path, rows, cols, chunk_rows=5)
    EXPORT_MODULE._save_tiled_tiff(base_mask, tiff_path, rows, cols, tile_px=16)
    EXPORT_MODULE._save_center_png_from_base(base_mask, rows, cols, png_path, crop_px=12, output_px=24)

    np.testing.assert_array_equal(np.load(npy_path, allow_pickle=False), expected)
    np.testing.assert_array_equal(tifffile.imread(tiff_path), expected.astype(np.uint8) * 255)
    assert Image.open(png_path).size == (24, 24)


def test_compact_gds_vertical_run_merging_is_lossless(tmp_path: Path) -> None:
    mask = np.zeros((12, 16), dtype=bool)
    mask[1:8, 2:6] = True
    mask[4:12, 9:14] = True
    rectangles = list(COMPACT_MODULE.iter_merged_rectangles(mask))
    rebuilt = np.zeros_like(mask)
    for x0, x1, y0, y1 in rectangles:
        rebuilt[y0:y1, x0:x1] = True
    np.testing.assert_array_equal(rebuilt, mask)

    npy_path = tmp_path / "mask.npy"
    compact_path = tmp_path / "compact.gds"
    baseline_path = tmp_path / "baseline.gds"
    np.save(npy_path, mask, allow_pickle=False)
    COMPACT_MODULE.compact_mask_to_gds(npy_path, compact_path, pixel_um=1.0)
    EXPORT_MODULE._save_binary_gds(mask, baseline_path, pixel_um=1.0, rows=mask.shape[0], cols=mask.shape[1], cell_prefix="BASE")

    assert compact_path.read_bytes()[2:4] == b"\x00\x02"
    assert compact_path.stat().st_size < baseline_path.stat().st_size


def test_direct_gds_compactor_preserves_generated_hierarchy(tmp_path: Path) -> None:
    mask = np.zeros((12, 16), dtype=bool)
    mask[1:8, 2:6] = True
    mask[4:12, 9:14] = True
    source = tmp_path / "source.gds"
    output = tmp_path / "compact.gds"
    EXPORT_MODULE._save_binary_gds(mask, source, pixel_um=1.0, rows=mask.shape[0], cols=mask.shape[1], cell_prefix="BASE")

    input_rectangles, output_rectangles = DIRECT_COMPACT_MODULE.compact_generated_gds(source, output)

    assert output.read_bytes()[2:4] == b"\x00\x02"
    assert output_rectangles <= input_rectangles
    assert output.stat().st_size <= source.stat().st_size
    assert b"BASE_TOP" in output.read_bytes()
