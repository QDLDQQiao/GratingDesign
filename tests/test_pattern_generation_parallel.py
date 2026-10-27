from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pattern_generation import DammannGrating, KinoformCRL1D, RandomHartmann


def test_random_hartmann_chunked_raster_matches_serial() -> None:
    params = {
        "field_size_um": (256, 256),
        "raster_resolution_um": 1.0,
        "target_density_per_mm2": 40,
        "hole_shape": "square",
        "hole_rotation": 25.0,
        "min_center_distance_um": 35.0,
        "hole_size_um": 25.0,
        "edge_clearance_um": 25.0,
        "seed": 123,
        "supersample": 1,
        "chunk_rows": 32,
    }
    serial = RandomHartmann(**params, n_cores=1).to_array()
    parallel = RandomHartmann(**params, n_cores=2).to_array()

    np.testing.assert_array_equal(parallel, serial)


def test_random_hartmann_preview_avoids_full_fabrication_raster() -> None:
    pattern = RandomHartmann(
        field_size_um=(1_300.0, 1_300.0),
        raster_resolution_um=0.2,
        target_density_per_mm2=40.0,
        hole_shape="square",
        hole_rotation=25.0,
        min_center_distance_um=113.0,
        hole_size_um=55.0,
        edge_clearance_um=20.0,
        seed=123,
    )

    preview = pattern.preview_array(max_pixels=120)

    assert preview.shape[0] <= 120
    assert preview.shape[1] <= 120
    assert pattern.holes


def test_random_hartmann_rejects_impossible_density_immediately() -> None:
    pattern = RandomHartmann(
        field_size_um=(13_000.0, 13_000.0),
        raster_resolution_um=0.2,
        target_density_per_mm2=160.0,
        min_center_distance_um=113.0,
        hole_shape="square",
        hole_size_um=55.0,
    )

    try:
        pattern.generate_holes()
    except ValueError as exc:
        assert "hexagonal-packing upper bound" in str(exc)
    else:
        raise AssertionError("Expected infeasible density to be rejected")


def test_random_hartmann_placement_limits_include_effective_spacing() -> None:
    pattern = RandomHartmann(
        field_size_um=(13_000.0, 13_000.0),
        raster_resolution_um=0.2,
        target_density_per_mm2=130.0,
        min_center_distance_um=50.0,
        hole_shape="square",
        hole_size_um=55.0,
        hole_rotation=25.0,
    )

    limits = pattern.placement_limits()

    assert limits["requested_hole_count"] == 21_970
    assert limits["effective_min_center_distance_um"] > pattern.min_center_distance_um
    assert limits["theoretical_hexagonal_max_density_per_mm2"] > limits["requested_density_per_mm2"]
    assert limits["random_sequential_reference_density_per_mm2"] < limits["requested_density_per_mm2"]
    assert limits["full_raster_shape_yx"] == [65_000, 65_000]


def test_dammann_grating_designs_binary_five_by_five_array(tmp_path: Path) -> None:
    pattern = DammannGrating(
        field_size_um=(128, 128),
        raster_resolution_um=1.0,
        xray_energy_keV=12.0,
        period_um=32.0,
        focal_length_m=0.3,
        orders_per_axis=5,
        transition_count=5,
        optimization_iterations=120,
        raster_chunk_rows=32,
    )
    mask = pattern.to_array()
    report = pattern.diffraction_report()
    focal = pattern.simulate_focal_plane(mask)
    detector = pattern.simulate_detector_plane(0.5, mask, padding_factor=2)
    paths = pattern.save_all(tmp_path, "dammann", array=mask)

    assert mask.shape == (128, 128)
    assert set(np.unique(mask)) <= {0, 1}
    assert report["beam_count"] == 25
    assert report["array_uniformity_min_over_mean"] > 0.5
    assert focal["model"] == "ideal_focusing_optic"
    assert focal["order_power"].shape == (5, 5)
    assert focal["captured_target_power"] > 0.0
    assert focal["reference_intensity"].shape == focal["intensity"].shape
    assert detector["intensity"].shape == (256, 256)
    assert np.isclose(detector["first_order_spacing_um"], report["first_order_spacing_um_at_focus"] * (1.0 + 0.5 / 0.3))
    assert detector["nominal_beamlet_envelope_diameter_um"] > 0.0
    assert all(path.exists() for path in paths.values())


def test_dammann_suggested_period_matches_requested_focal_separation() -> None:
    suggestion = DammannGrating.suggest_parameters(energy_keV=12.0, focal_length_m=0.3, desired_separation_um=2.0)

    assert suggestion["beam_count"] == 25
    assert np.isclose(suggestion["period_um"], 15.498, rtol=1e-12)
    assert suggestion["recommended_simulation_field_size_um"] == suggestion["recommended_field_size_um"]
    assert suggestion["recommended_simulation_raster_resolution_um"] == suggestion["recommended_raster_resolution_um"]
    assert suggestion["recommended_raster_resolution_um"] < suggestion["period_um"]
    assert suggestion["predicted_diffraction_limited_fwhm_um"] < suggestion["desired_separation_um"]


def test_dammann_focal_simulation_accepts_phase_depth_tolerance() -> None:
    pattern = DammannGrating(
        field_size_um=(64, 64),
        raster_resolution_um=1.0,
        xray_energy_keV=12.0,
        period_um=16.0,
        focal_length_m=0.3,
        orders_per_axis=5,
        transition_count=5,
        optimization_iterations=60,
        raster_chunk_rows=32,
    )

    focal = pattern.simulate_focal_plane(phase_depth_rad=0.9 * np.pi)

    assert np.isclose(focal["phase_depth_rad"], 0.9 * np.pi)
    assert focal["order_power"].shape == (5, 5)


def test_kinoform_crl_builds_quantized_silicon_relief(tmp_path: Path) -> None:
    from export_fabrication_patterns import save_fixed_depth_gds

    pattern = KinoformCRL1D(
        radius_um=20.0,
        xray_energy_keV=12.0,
        number_of_crls=3,
        crl_spacing_um=10.0,
        array_count=2,
        array_spacing_um=15.0,
        aperture_um=80.0,
        support_width_um=100.0,
        raster_resolution_um=0.5,
        fixed_etch_depth_um=40.0,
        central_wall_um=15.0,
        frame_margin_um=4.0,
        support_web_width_um=2.0,
    )

    surface_mask = pattern.surface_mask()
    optics = pattern.optical_parameters()

    assert surface_mask.ndim == 2
    assert set(np.unique(surface_mask)) <= {0, 1}
    assert np.any(surface_mask > 0)
    frame_pixels = int(pattern.frame_margin_um / pattern.raster_resolution_um)
    assert not np.any(surface_mask[:frame_pixels, :])
    assert not np.any(surface_mask[:, :frame_pixels])
    assert pattern.metadata()["array_count"] == 2
    assert pattern.field_size_um()[1] > pattern.single_array_width_um
    assert np.isclose(
        pattern.effective_crl_length_um,
        2.0 * pattern.effective_surface_length_um + pattern.central_wall_um,
    )
    central_wall_row = int(
        (pattern.frame_margin_um + pattern.effective_surface_length_um + pattern.central_wall_um / 2.0)
        / pattern.raster_resolution_um
    )
    assert not np.any(surface_mask[central_wall_row, :])
    assert np.all(pattern.etch_depth_map_um()[surface_mask > 0] == pattern.fixed_etch_depth_um)
    assert pattern.frame_rectangle_um()[0] > pattern.aperture_um
    assert np.isclose(optics["focal_length_um"], pattern.radius_um / (2.0 * pattern.number_of_crls * optics["delta"]))
    assert np.isclose(optics["two_pi_path_jump_um"], optics["wavelength_um"] / optics["delta"])
    profile = pattern.surface_profile_um()
    width_um = pattern.field_size_um()[1]
    profile_x = (np.arange(profile.size) + 0.5) * pattern.raster_resolution_um - width_um / 2.0
    first_array_center = -width_um / 2.0 + pattern.frame_margin_um + pattern.single_array_width_um / 2.0
    profile_index = int(np.argmin(np.abs(profile_x - (first_array_center + 10.0))))
    local_x = profile_x[profile_index] - first_array_center
    assert np.isclose(profile[profile_index], local_x * local_x / (2.0 * pattern.radius_um))

    focal = pattern.simulate_focal_spot(
        fabrication_resolution_um=0.1,
        simulation_resolution_um=0.1,
        source_size_fwhm_um=30.0,
        source_distance_m=30.0,
        source_modes=5,
        padding_factor=2,
    )
    assert set(focal["intensities"]) == {"ideal_coherent", "fabricated_coherent", "ideal_partial", "fabricated_partial"}
    assert focal["source_blur_sigma_um"] > 0.0
    assert focal["relative_peak"]["fabricated_over_ideal_coherent"] > 0.0

    gds_path = tmp_path / "kinoform.gds"
    save_fixed_depth_gds(surface_mask, gds_path, pattern.raster_resolution_um, "KINOFORM", frame_size_um=pattern.frame_rectangle_um())
    assert gds_path.exists()
    assert gds_path.stat().st_size > 100

    gds_path_without_frame = tmp_path / "kinoform_without_frame.gds"
    save_fixed_depth_gds(
        surface_mask,
        gds_path_without_frame,
        pattern.raster_resolution_um,
        "KINOFORM_NO_FRAME",
        frame_size_um=pattern.frame_rectangle_um(),
        frame_width_um=0.0,
    )
    assert gds_path_without_frame.exists()
    assert gds_path_without_frame.stat().st_size > 100
