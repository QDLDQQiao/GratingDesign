"""X-ray wavefront sensing mask generation utilities.

The module is designed for interactive notebook use, but it also exposes a
small command line interface. All geometry is specified in physical units:
micrometers for features and millimeters for convenience field sizes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage


ArrayLike = np.ndarray
Point = Tuple[float, float]
# Pillow 9+ exposes resampling filters through Image.Resampling, while older
# Pillow releases keep the constants directly on Image.
_NEAREST_RESAMPLING = getattr(getattr(Image, "Resampling", Image), "NEAREST")


def _random_hartmann_chunk(task: Tuple[int, int, float, float, float, List[Dict[str, float]], str, int]) -> Tuple[int, ArrayLike]:
    """Rasterize one RandomHartmann row range without allocating the full field."""

    row_start, row_stop, height_um, width_um, resolution_um, holes, hole_shape, n_sides = task
    nx = int(np.ceil(width_um / resolution_um))
    mask = np.zeros((row_stop - row_start, nx), dtype=bool)
    for hole in holes:
        radius = float(hole["bounding_radius_um"])
        col_start = max(0, int(math.ceil((hole["x_um"] - radius + width_um / 2.0) / resolution_um - 0.5)))
        col_stop = min(nx, int(math.floor((hole["x_um"] + radius + width_um / 2.0) / resolution_um - 0.5)) + 1)
        global_row_start = max(row_start, int(math.ceil((hole["y_um"] - radius + height_um / 2.0) / resolution_um - 0.5)))
        global_row_stop = min(row_stop, int(math.floor((hole["y_um"] + radius + height_um / 2.0) / resolution_um - 0.5)) + 1)
        if col_start >= col_stop or global_row_start >= global_row_stop:
            continue
        x = ((np.arange(col_start, col_stop, dtype=np.float64) + 0.5) * resolution_um - width_um / 2.0)[None, :]
        y = ((np.arange(global_row_start, global_row_stop, dtype=np.float64) + 0.5) * resolution_um - height_um / 2.0)[:, None]
        mask[global_row_start - row_start:global_row_stop - row_start, col_start:col_stop] |= _shape_mask(
            x - hole["x_um"],
            y - hole["y_um"],
            hole_shape,
            hole["hole_size_um"],
            hole["rotation_deg"],
            n_sides,
        )
    return row_start, mask.astype(np.uint8)


def wavelength_nm(energy_keV: float) -> float:
    """Return X-ray wavelength in nm from photon energy in keV."""
    if energy_keV <= 0:
        raise ValueError("energy_keV must be positive")
    return 1.23984 / energy_keV


def rotate_xy(x: ArrayLike, y: ArrayLike, angle_deg: float) -> Tuple[ArrayLike, ArrayLike]:
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return c * x + s * y, -s * x + c * y


def _grid(field_size_um: Tuple[float, float], resolution_um: float, supersample: int = 1) -> Tuple[ArrayLike, ArrayLike]:
    height_um, width_um = field_size_um
    if resolution_um <= 0:
        raise ValueError("raster_resolution_um must be positive")
    if height_um <= 0 or width_um <= 0:
        raise ValueError("field_size_um entries must be positive")
    step = resolution_um / supersample
    ny = int(np.ceil(height_um / step))
    nx = int(np.ceil(width_um / step))
    x = (np.arange(nx, dtype=np.float64) + 0.5) * step - width_um / 2.0
    y = (np.arange(ny, dtype=np.float64) + 0.5) * step - height_um / 2.0
    return np.meshgrid(x, y)


def _downsample(mask: ArrayLike, supersample: int) -> ArrayLike:
    if supersample <= 1:
        return mask.astype(np.uint8)
    ny = (mask.shape[0] // supersample) * supersample
    nx = (mask.shape[1] // supersample) * supersample
    cropped = mask[:ny, :nx]
    averaged = cropped.reshape(ny // supersample, supersample, nx // supersample, supersample).mean(axis=(1, 3))
    return (averaged >= 0.5).astype(np.uint8)


def _as_uint8_image(mask: ArrayLike) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


@dataclass
class MaskPattern:
    """Base class for binary mask patterns.

    Parameters
    ----------
    field_size_um:
        ``(height_um, width_um)`` of the mask.
    raster_resolution_um:
        Pixel size of raster output. Use smaller values for simulation fidelity.
    xray_energy_keV:
        Energy used only for metadata and physics helper calculations.
    polarity:
        ``"holes_transmit"`` means logical 1 is an open/transmitting feature.
        ``"absorber_transmit"`` inverts the generated geometry.
    supersample:
        Rasterization oversampling factor. Values 3-6 are useful for rotated
        or circular features.
    """

    field_size_um: Tuple[float, float] = (1000.0, 1000.0)
    raster_resolution_um: float = 0.25
    xray_energy_keV: float = 12.0
    polarity: str = "holes_transmit"
    supersample: int = 4

    def _validate_base(self) -> None:
        if self.polarity not in {"holes_transmit", "absorber_transmit"}:
            raise ValueError("polarity must be 'holes_transmit' or 'absorber_transmit'")
        if self.supersample < 1:
            raise ValueError("supersample must be >= 1")

    def _apply_polarity(self, mask: ArrayLike) -> ArrayLike:
        mask = mask.astype(np.uint8)
        if self.polarity == "absorber_transmit":
            mask = 1 - mask
        return mask

    def _binary(self) -> ArrayLike:
        raise NotImplementedError

    def to_array(self) -> ArrayLike:
        self._validate_base()
        return self._apply_polarity(self._binary())

    def to_tiff(self, filename: Union[str, Path]) -> Path:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        _as_uint8_image(self.to_array()).save(filename)
        return filename

    def to_png(self, filename: Union[str, Path]) -> Path:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        _as_uint8_image(self.to_array()).save(filename)
        return filename

    def to_npy(self, filename: Union[str, Path]) -> Path:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(filename, self.to_array())
        return filename

    def preview(self, max_size: int = 1200) -> Image.Image:
        img = _as_uint8_image(self.to_array())
        img.thumbnail((max_size, max_size), _NEAREST_RESAMPLING)
        return img

    def save_preview(self, filename: Union[str, Path], max_size: int = 1200) -> Path:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.preview(max_size=max_size).save(filename)
        return filename

    def metadata(self, array_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        data = asdict(self)
        data["pattern_class"] = self.__class__.__name__
        data["wavelength_nm"] = wavelength_nm(self.xray_energy_keV)
        data["array_shape_yx"] = list(array_shape if array_shape is not None else self.to_array().shape)
        return _jsonable(data)

    def save_metadata(self, filename: Union[str, Path], array_shape: Optional[Tuple[int, int]] = None) -> Path:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(json.dumps(self.metadata(array_shape=array_shape), indent=2), encoding="utf-8")
        return filename

    def save_all(self, output_dir: Union[str, Path], stem: str, array: Optional[ArrayLike] = None) -> Dict[str, Path]:
        """Write all raster artifacts, optionally reusing an already-generated array."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = self.to_array() if array is None else np.asarray(array, dtype=np.uint8)
        png_path = output_dir / f"{stem}.png"
        tiff_path = output_dir / f"{stem}.tiff"
        npy_path = output_dir / f"{stem}.npy"
        preview_path = output_dir / f"{stem}_preview.png"
        _as_uint8_image(mask).save(png_path)
        _as_uint8_image(mask).save(tiff_path)
        np.save(npy_path, mask)
        preview = _as_uint8_image(mask)
        preview.thumbnail((1200, 1200), _NEAREST_RESAMPLING)
        preview.save(preview_path)
        paths = {
            "png": png_path,
            "tiff": tiff_path,
            "npy": npy_path,
            "metadata": self.save_metadata(output_dir / f"{stem}.json", array_shape=mask.shape),
            "preview": preview_path,
        }
        return paths

    def to_polygons(self) -> List[Any]:
        """Return vector polygons when optional geometry libraries are available.

        For now this returns an empty list for notebook workflows that only need
        raster masks. Use the metadata JSON for exact RandomHartmann geometry.
        """

        return []

    def to_gds(self, filename: Union[str, Path]) -> Path:
        raise NotImplementedError("GDS export needs gdstk/gdspy vectorization; raster and metadata export are implemented.")

    def to_dxf(self, filename: Union[str, Path]) -> Path:
        raise NotImplementedError("DXF export needs ezdxf vectorization; raster and metadata export are implemented.")


@dataclass
class Grating(MaskPattern):
    period_um: float = 20.0
    duty_cycle: float = 0.5
    dimensionality: str = "1D"
    orientation_deg: float = 0.0
    period_x_um: Optional[float] = None
    period_y_um: Optional[float] = None
    duty_cycle_x: Optional[float] = None
    duty_cycle_y: Optional[float] = None
    combine_mode: str = "and"

    def _binary(self) -> ArrayLike:
        px = self.period_x_um or self.period_um
        py = self.period_y_um or self.period_um
        dcx = self.duty_cycle_x if self.duty_cycle_x is not None else self.duty_cycle
        dcy = self.duty_cycle_y if self.duty_cycle_y is not None else self.duty_cycle
        for name, period, dc in [("x", px, dcx), ("y", py, dcy)]:
            if period <= 0:
                raise ValueError(f"period_{name}_um must be positive")
            if not 0 < dc < 1:
                raise ValueError(f"duty_cycle_{name} must be between 0 and 1")
        x, y = _grid(self.field_size_um, self.raster_resolution_um, self.supersample)
        xr, yr = rotate_xy(x, y, self.orientation_deg)
        mx = np.mod(xr + 0.5 * px, px) < (dcx * px)
        if self.dimensionality.upper() == "1D":
            return _downsample(mx, self.supersample)
        my = np.mod(yr + 0.5 * py, py) < (dcy * py)
        if self.combine_mode.lower() == "and":
            mask = mx & my
        elif self.combine_mode.lower() == "or":
            mask = mx | my
        else:
            raise ValueError("combine_mode must be 'and' or 'or'")
        return _downsample(mask, self.supersample)


@dataclass
class Checkerboard(MaskPattern):
    period_um: float = 20.0
    duty_cycle: float = 0.5
    orientation_deg: float = 0.0

    def _binary(self) -> ArrayLike:
        if self.period_um <= 0:
            raise ValueError("period_um must be positive")
        if not 0 < self.duty_cycle < 1:
            raise ValueError("duty_cycle must be between 0 and 1")
        x, y = _grid(self.field_size_um, self.raster_resolution_um, self.supersample)
        xr, yr = rotate_xy(x, y, self.orientation_deg)
        ix = np.floor((xr + self.period_um * 1000000.0) / (self.period_um / 2.0)).astype(np.int64)
        iy = np.floor((yr + self.period_um * 1000000.0) / (self.period_um / 2.0)).astype(np.int64)
        mask = (ix + iy) % 2 == 0
        if self.duty_cycle != 0.5:
            gx = np.mod(xr + 0.5 * self.period_um, self.period_um) < (self.period_um * self.duty_cycle)
            gy = np.mod(yr + 0.5 * self.period_um, self.period_um) < (self.period_um * self.duty_cycle)
            mask = mask & (gx | gy)
        return _downsample(mask, self.supersample)


@dataclass
class DammannGrating(MaskPattern):
    """Binary phase Dammann beam splitter for a focusing optic.

    The periodic 1-D Dammann cell is optimized for equal odd-order splitting
    and crossed in x/y. By default the mask is designed for a separate focusing
    optic. Set ``include_binary_fresnel_lens=True`` only when a standalone
    Dammann-Fresnel zone-plate mask is specifically required.
    """

    orders_per_axis: int = 5
    period_um: float = 100.0
    focal_length_m: float = 0.3
    transition_count: int = 5
    optimization_iterations: int = 1200
    transition_seed: int = 23
    transitions_half_period: List[float] = field(default_factory=list)
    raster_chunk_rows: int = 256
    include_binary_fresnel_lens: bool = False

    @staticmethod
    def suggest_parameters(
        energy_keV: float,
        focal_length_m: float,
        desired_separation_um: float,
        orders_per_axis: int = 5,
        samples_per_period: int = 128,
        illuminated_aperture_periods: int = 8,
        focal_padding_factor: int = 3,
    ) -> Dict[str, float | int]:
        """Suggest a compact simulation grid from focal-plane separation.

        The grating period follows ``period = wavelength * focal_length /
        separation``. The returned field size and pixel pitch are for a compact
        numerical pupil with approximately eight Dammann periods.  They are
        deliberately independent of a fabrication aperture: a fabricated
        periodic mask may cover a much larger field while retaining the same
        Dammann period and feature-resolution requirement.
        """

        if energy_keV <= 0 or focal_length_m <= 0 or desired_separation_um <= 0:
            raise ValueError("energy_keV, focal_length_m, and desired_separation_um must be positive")
        if orders_per_axis < 3 or orders_per_axis % 2 == 0:
            raise ValueError("orders_per_axis must be an odd integer of at least 3")
        if samples_per_period < 32 or illuminated_aperture_periods < 4 or focal_padding_factor < 1:
            raise ValueError("samples_per_period >= 32, illuminated_aperture_periods >= 4, and focal_padding_factor >= 1 are required")
        wavelength_um = wavelength_nm(energy_keV) / 1000.0
        focal_um = focal_length_m * 1e6
        period_um = wavelength_um * focal_um / desired_separation_um
        aperture_um = illuminated_aperture_periods * period_um
        focal_fwhm_um = 1.028 * wavelength_um * focal_um / aperture_um
        outer_zone_um = wavelength_um * focal_um / aperture_um
        return {
            "energy_keV": energy_keV,
            "focal_length_m": focal_length_m,
            "desired_separation_um": desired_separation_um,
            "orders_per_axis": orders_per_axis,
            "beam_count": orders_per_axis**2,
            "wavelength_um": wavelength_um,
            "period_um": period_um,
            "recommended_simulation_raster_resolution_um": period_um / samples_per_period,
            "recommended_simulation_field_size_um": aperture_um,
            "recommended_simulation_pupil_diameter_um": aperture_um,
            # Retained for existing notebooks and scripts using the original
            # single-grid naming scheme.
            "recommended_raster_resolution_um": period_um / samples_per_period,
            "recommended_illuminated_aperture_um": aperture_um,
            "recommended_field_size_um": aperture_um,
            "predicted_diffraction_limited_fwhm_um": focal_fwhm_um,
            "separation_to_fwhm_ratio": desired_separation_um / focal_fwhm_um,
            "focal_padding_factor": focal_padding_factor,
            "standalone_binary_lens_outermost_zone_um": outer_zone_um,
            "standalone_binary_lens_recommended_raster_um": outer_zone_um / 4.0,
        }

    def _validate_design(self) -> None:
        if self.orders_per_axis < 3 or self.orders_per_axis % 2 == 0:
            raise ValueError("orders_per_axis must be an odd integer of at least 3")
        if self.period_um <= 0:
            raise ValueError("period_um must be positive")
        if self.focal_length_m <= 0:
            raise ValueError("focal_length_m must be positive")
        if self.transition_count < 1:
            raise ValueError("transition_count must be positive")
        if self.optimization_iterations < 0:
            raise ValueError("optimization_iterations must be non-negative")
        if self.raster_chunk_rows < 1:
            raise ValueError("raster_chunk_rows must be positive")

    def _transition_score(self, transitions: np.ndarray) -> float:
        # Optimize at the actual raster sampling of one Dammann period so the
        # predicted order balance matches the fabricated binary pixel mask.
        samples = max(32, int(round(self.period_um / self.raster_resolution_um)))
        coordinate = np.mod((np.arange(samples, dtype=np.float64) + 0.5) / samples + 0.5, 1.0) - 0.5
        bit = (np.searchsorted(transitions, np.abs(coordinate), side="right") % 2).astype(np.int8)
        coefficients = np.fft.fftshift(np.fft.fft(1.0 - 2.0 * bit)) / samples
        half_orders = self.orders_per_axis // 2
        target = np.abs(coefficients[samples // 2 - half_orders:samples // 2 + half_orders + 1]) ** 2
        mean = float(np.mean(target))
        if mean <= 0:
            return -np.inf
        uniformity = float(np.min(target) / mean)
        energy = float(np.sum(target))
        return 1.8 * uniformity + 0.2 * energy

    def design_transitions(self) -> List[float]:
        """Optimize symmetric 1-D Dammann transition positions on [0, 0.5]."""

        self._validate_design()
        if self.transitions_half_period:
            transitions = np.asarray(self.transitions_half_period, dtype=np.float64)
            if transitions.shape != (self.transition_count,) or np.any(transitions <= 0) or np.any(transitions >= 0.5):
                raise ValueError("transitions_half_period must contain sorted positions strictly between 0 and 0.5")
            if np.any(np.diff(transitions) <= 0):
                raise ValueError("transitions_half_period must be strictly increasing")
            return transitions.tolist()

        rng = np.random.default_rng(self.transition_seed)
        minimum_gap = 0.5 / (12.0 * (self.transition_count + 1))
        base = np.linspace(0.5 / (self.transition_count + 1), 0.5 - 0.5 / (self.transition_count + 1), self.transition_count)
        best = base
        best_score = self._transition_score(best)
        # A small multi-start local search is deterministic and runs once per
        # design. The grating period/focus do not affect these normalized steps.
        for start in range(12):
            current = base if start == 0 else np.sort(rng.uniform(minimum_gap, 0.5 - minimum_gap, self.transition_count))
            if np.any(np.diff(current) < minimum_gap):
                continue
            current_score = self._transition_score(current)
            for iteration in range(self.optimization_iterations // 12 + 1):
                scale = 0.05 * (1.0 - iteration / max(1, self.optimization_iterations // 12 + 1)) + 0.002
                candidate = np.sort(np.clip(current + rng.normal(0.0, scale, self.transition_count), minimum_gap, 0.5 - minimum_gap))
                if np.any(np.diff(candidate) < minimum_gap):
                    continue
                candidate_score = self._transition_score(candidate)
                if candidate_score > current_score:
                    current, current_score = candidate, candidate_score
            if current_score > best_score:
                best, best_score = current, current_score
        self.transitions_half_period = [float(value) for value in best]
        return self.transitions_half_period

    def diffraction_report(self) -> Dict[str, Any]:
        """Return normalized 1-D order powers and the predicted square array."""

        transitions = np.asarray(self.design_transitions(), dtype=np.float64)
        samples = max(32, int(round(self.period_um / self.raster_resolution_um)))
        coordinate = np.mod((np.arange(samples, dtype=np.float64) + 0.5) / samples + 0.5, 1.0) - 0.5
        bit = np.searchsorted(transitions, np.abs(coordinate), side="right") % 2
        coefficients = np.fft.fftshift(np.fft.fft(1.0 - 2.0 * bit)) / samples
        half_orders = self.orders_per_axis // 2
        orders = np.arange(-half_orders, half_orders + 1)
        powers = np.abs(coefficients[samples // 2 + orders]) ** 2
        array_powers = np.outer(powers, powers)
        wavelength_um = wavelength_nm(self.xray_energy_keV) / 1000.0
        focal_um = self.focal_length_m * 1e6
        return {
            "orders_per_axis": self.orders_per_axis,
            "beam_count": self.orders_per_axis**2,
            "transitions_half_period": [float(value) for value in transitions],
            "order_indices": orders.tolist(),
            "one_dimensional_order_power": powers.tolist(),
            "one_dimensional_uniformity_min_over_mean": float(np.min(powers) / np.mean(powers)),
            "target_order_energy_fraction_1d": float(np.sum(powers)),
            "target_order_energy_fraction_2d": float(np.sum(array_powers)),
            "array_uniformity_min_over_mean": float(np.min(array_powers) / np.mean(array_powers)),
            "focal_length_m": self.focal_length_m,
            "first_order_spacing_um_at_focus": float(wavelength_um * focal_um / self.period_um),
            "order_coordinates_um_at_focus": (orders * wavelength_um * focal_um / self.period_um).tolist(),
            "phase_levels_rad": [0.0, float(np.pi)],
        }

    def _binary(self) -> ArrayLike:
        transitions = np.asarray(self.design_transitions(), dtype=np.float64)
        height_um, width_um = self.field_size_um
        resolution_um = self.raster_resolution_um
        rows = int(np.ceil(height_um / resolution_um))
        cols = int(np.ceil(width_um / resolution_um))
        wavelength_um = wavelength_nm(self.xray_energy_keV) / 1000.0
        focal_um = self.focal_length_m * 1e6
        x = ((np.arange(cols, dtype=np.float64) + 0.5) * resolution_um - width_um / 2.0)[None, :]
        x_bit = np.searchsorted(transitions, np.abs(np.mod(x / self.period_um + 0.5, 1.0) - 0.5), side="right") % 2
        mask = np.empty((rows, cols), dtype=np.uint8)
        for row_start in range(0, rows, self.raster_chunk_rows):
            row_stop = min(rows, row_start + self.raster_chunk_rows)
            y = ((np.arange(row_start, row_stop, dtype=np.float64) + 0.5) * resolution_um - height_um / 2.0)[:, None]
            y_bit = np.searchsorted(transitions, np.abs(np.mod(y / self.period_um + 0.5, 1.0) - 0.5), side="right") % 2
            dammann_bit = x_bit ^ y_bit
            if self.include_binary_fresnel_lens:
                # Binary phase Fresnel lens: adjacent half-wave zones differ by pi.
                lens_bit = np.floor((x * x + y * y) / (wavelength_um * focal_um)).astype(np.int64) & 1
                mask[row_start:row_stop] = (dammann_bit ^ lens_bit).astype(np.uint8)
            else:
                mask[row_start:row_stop] = dammann_bit.astype(np.uint8)
        return mask

    def simulate_focal_plane(
        self,
        array: Optional[ArrayLike] = None,
        model: str = "ideal_focusing_optic",
        pupil_diameter_um: Optional[float] = None,
        pupil_shape: str = "circular",
        padding_factor: int = 1,
        phase_depth_rad: float = np.pi,
    ) -> Dict[str, Any]:
        """Simulate the Dammann beam array in the plane of a focusing optic.

        ``ideal_focusing_optic`` evaluates the Dammann pattern in the Fourier
        plane of an ideal lens, which isolates beam-splitting performance and
        also returns the unaberrated reference focal spot. ``binary_fresnel_mask``
        instead propagates the fabricated XOR-combined binary Fresnel mask and
        therefore includes lens-order contamination. ``padding_factor``
        zero-pads the pupil before the FFT to sample spot profiles more finely.
        ``phase_depth_rad`` allows tolerance studies for an etched phase depth
        that differs from the nominal pi phase step.
        """

        binary_mask = self.to_array() if array is None else np.asarray(array, dtype=np.uint8)
        if binary_mask.ndim != 2:
            raise ValueError("array must be a two-dimensional binary mask")
        if padding_factor < 1:
            raise ValueError("padding_factor must be positive")
        if not np.isfinite(phase_depth_rad):
            raise ValueError("phase_depth_rad must be finite")
        if pupil_shape not in {"circular", "square"}:
            raise ValueError("pupil_shape must be 'circular' or 'square'")
        wavelength_um = wavelength_nm(self.xray_energy_keV) / 1000.0
        focal_um = self.focal_length_m * 1e6
        dx_um = self.raster_resolution_um
        rows, cols = binary_mask.shape
        x = (np.arange(cols, dtype=np.float64) - cols / 2.0 + 0.5) * dx_um
        y = (np.arange(rows, dtype=np.float64) - rows / 2.0 + 0.5) * dx_um
        default_diameter = min(rows * dx_um, cols * dx_um)
        pupil_diameter_um = default_diameter if pupil_diameter_um is None else float(pupil_diameter_um)
        if not 0 < pupil_diameter_um <= default_diameter:
            raise ValueError("pupil_diameter_um must be positive and no larger than the mask field")
        if pupil_shape == "circular":
            pupil = (x[None, :] ** 2 + y[:, None] ** 2) <= (pupil_diameter_um / 2.0) ** 2
        else:
            pupil = (np.abs(x)[None, :] <= pupil_diameter_um / 2.0) & (np.abs(y)[:, None] <= pupil_diameter_um / 2.0)
        transitions = np.asarray(self.design_transitions(), dtype=np.float64)
        x_bit = np.searchsorted(transitions, np.abs(np.mod(x / self.period_um + 0.5, 1.0) - 0.5), side="right") % 2
        y_bit = np.searchsorted(transitions, np.abs(np.mod(y / self.period_um + 0.5, 1.0) - 0.5), side="right") % 2
        dammann_phase = x_bit[None, :] ^ y_bit[:, None]
        if model == "ideal_focusing_optic":
            # An ideal lens maps the mask Fourier transform to its focal plane.
            transmission = np.exp(1j * phase_depth_rad * dammann_phase)
        elif model == "binary_fresnel_mask":
            quadratic_input = np.exp(1j * np.pi * (x[None, :] ** 2 + y[:, None] ** 2) / (wavelength_um * focal_um))
            transmission = np.exp(1j * phase_depth_rad * binary_mask) * quadratic_input
        else:
            raise ValueError("model must be 'ideal_focusing_optic' or 'binary_fresnel_mask'")
        padded_rows, padded_cols = rows * padding_factor, cols * padding_factor

        def focal_fft(input_field: np.ndarray) -> np.ndarray:
            padded = np.zeros((padded_rows, padded_cols), dtype=np.complex128)
            y0 = (padded_rows - rows) // 2
            x0 = (padded_cols - cols) // 2
            padded[y0:y0 + rows, x0:x0 + cols] = input_field
            return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))

        field = focal_fft(pupil * transmission)
        intensity = np.abs(field) ** 2
        del field
        reference_field = focal_fft(pupil.astype(np.complex128))
        reference_intensity = np.abs(reference_field) ** 2
        del reference_field
        intensity /= max(float(np.sum(intensity)), np.finfo(np.float64).tiny)
        reference_intensity /= max(float(np.sum(reference_intensity)), np.finfo(np.float64).tiny)
        focal_x = np.fft.fftshift(np.fft.fftfreq(padded_cols, d=dx_um)) * wavelength_um * focal_um
        focal_y = np.fft.fftshift(np.fft.fftfreq(padded_rows, d=dx_um)) * wavelength_um * focal_um
        report = self.diffraction_report()
        orders = np.asarray(report["order_indices"], dtype=int)
        order_centers = np.asarray(report["order_coordinates_um_at_focus"], dtype=np.float64)
        spacing = float(report["first_order_spacing_um_at_focus"])
        integration_radius = max(0.15 * spacing, min(abs(focal_x[1] - focal_x[0]), abs(focal_y[1] - focal_y[0])))
        powers = np.zeros((self.orders_per_axis, self.orders_per_axis), dtype=np.float64)
        for iy, y_center in enumerate(order_centers):
            y_selector = np.abs(focal_y - y_center) <= integration_radius
            for ix, x_center in enumerate(order_centers):
                x_selector = np.abs(focal_x - x_center) <= integration_radius
                powers[iy, ix] = float(np.sum(intensity[np.ix_(y_selector, x_selector)]))
        return {
            "focal_x_um": focal_x,
            "focal_y_um": focal_y,
            "intensity": intensity,
            "reference_intensity": reference_intensity,
            "order_indices": orders,
            "order_coordinates_um": order_centers,
            "order_power": powers,
            "captured_target_power": float(np.sum(powers)),
            "order_uniformity_min_over_mean": float(np.min(powers) / np.mean(powers)) if np.mean(powers) else 0.0,
            "integration_radius_um": integration_radius,
            "model": model,
            "pupil_shape": pupil_shape,
            "pupil_diameter_um": pupil_diameter_um,
            "padding_factor": padding_factor,
            "phase_depth_rad": float(phase_depth_rad),
            "focal_pixel_um": float(abs(focal_x[1] - focal_x[0])),
            "reference_fwhm_estimate_um": float(1.028 * wavelength_um * focal_um / pupil_diameter_um),
        }

    def simulate_detector_plane(
        self,
        distance_after_focus_m: float,
        array: Optional[ArrayLike] = None,
        pupil_diameter_um: Optional[float] = None,
        pupil_shape: str = "circular",
        padding_factor: int = 2,
        phase_depth_rad: float = np.pi,
    ) -> Dict[str, Any]:
        """Propagate the focused Dammann beam array to a downstream detector.

        The model applies an ideal thin focusing lens to the Dammann phase mask
        and then uses paraxial Fresnel propagation to a plane
        ``distance_after_focus_m`` beyond the nominal focus.  Padding expands
        the detector field of view while preserving the physical sampling pitch.
        """

        if distance_after_focus_m < 0:
            raise ValueError("distance_after_focus_m must be non-negative")
        if padding_factor < 1:
            raise ValueError("padding_factor must be positive")
        if not np.isfinite(phase_depth_rad):
            raise ValueError("phase_depth_rad must be finite")
        if pupil_shape not in {"circular", "square"}:
            raise ValueError("pupil_shape must be 'circular' or 'square'")
        binary_mask = self.to_array() if array is None else np.asarray(array, dtype=np.uint8)
        if binary_mask.ndim != 2:
            raise ValueError("array must be a two-dimensional binary mask")

        wavelength_um = wavelength_nm(self.xray_energy_keV) / 1000.0
        focal_um = self.focal_length_m * 1e6
        propagation_um = focal_um + distance_after_focus_m * 1e6
        dx_um = self.raster_resolution_um
        rows, cols = binary_mask.shape
        x = (np.arange(cols, dtype=np.float64) - cols / 2.0 + 0.5) * dx_um
        y = (np.arange(rows, dtype=np.float64) - rows / 2.0 + 0.5) * dx_um
        default_diameter = min(rows * dx_um, cols * dx_um)
        pupil_diameter_um = default_diameter if pupil_diameter_um is None else float(pupil_diameter_um)
        if not 0 < pupil_diameter_um <= default_diameter:
            raise ValueError("pupil_diameter_um must be positive and no larger than the mask field")
        if pupil_shape == "circular":
            pupil = (x[None, :] ** 2 + y[:, None] ** 2) <= (pupil_diameter_um / 2.0) ** 2
        else:
            pupil = (np.abs(x)[None, :] <= pupil_diameter_um / 2.0) & (np.abs(y)[:, None] <= pupil_diameter_um / 2.0)

        transitions = np.asarray(self.design_transitions(), dtype=np.float64)
        x_bit = np.searchsorted(transitions, np.abs(np.mod(x / self.period_um + 0.5, 1.0) - 0.5), side="right") % 2
        y_bit = np.searchsorted(transitions, np.abs(np.mod(y / self.period_um + 0.5, 1.0) - 0.5), side="right") % 2
        dammann_phase = x_bit[None, :] ^ y_bit[:, None]
        thin_lens = np.exp(-1j * np.pi * (x[None, :] ** 2 + y[:, None] ** 2) / (wavelength_um * focal_um))
        input_field = pupil * np.exp(1j * phase_depth_rad * dammann_phase) * thin_lens

        padded_rows, padded_cols = rows * padding_factor, cols * padding_factor
        padded = np.zeros((padded_rows, padded_cols), dtype=np.complex128)
        y0 = (padded_rows - rows) // 2
        x0 = (padded_cols - cols) // 2
        padded[y0:y0 + rows, x0:x0 + cols] = input_field
        input_x = (np.arange(padded_cols, dtype=np.float64) - padded_cols / 2.0 + 0.5) * dx_um
        input_y = (np.arange(padded_rows, dtype=np.float64) - padded_rows / 2.0 + 0.5) * dx_um
        # A single scaled Fresnel step from the lens plane preserves the
        # physical detector coordinate system at long propagation distances.
        input_chirp = np.exp(1j * np.pi * (input_y[:, None] ** 2 + input_x[None, :] ** 2) / (wavelength_um * propagation_um))
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded * input_chirp)))
        detector_x = np.fft.fftshift(np.fft.fftfreq(padded_cols, d=dx_um)) * wavelength_um * propagation_um
        detector_y = np.fft.fftshift(np.fft.fftfreq(padded_rows, d=dx_um)) * wavelength_um * propagation_um
        output_chirp = np.exp(1j * np.pi * (detector_y[:, None] ** 2 + detector_x[None, :] ** 2) / (wavelength_um * propagation_um))
        field = spectrum * output_chirp
        intensity = np.abs(field) ** 2
        del field
        intensity /= max(float(np.sum(intensity)), np.finfo(np.float64).tiny)

        orders = np.arange(-(self.orders_per_axis // 2), self.orders_per_axis // 2 + 1)
        spacing = wavelength_um * propagation_um / self.period_um
        order_centers = orders * spacing
        beamlet_envelope_diameter_um = (
            1.028 * wavelength_um * focal_um / pupil_diameter_um
            if distance_after_focus_m == 0
            else pupil_diameter_um * distance_after_focus_m * 1e6 / focal_um
        )
        return {
            "detector_x_um": detector_x,
            "detector_y_um": detector_y,
            "intensity": intensity,
            "order_indices": orders,
            "order_coordinates_um": order_centers,
            "nominal_beamlet_envelope_diameter_um": float(beamlet_envelope_diameter_um),
            "distance_after_focus_m": float(distance_after_focus_m),
            "propagation_distance_from_lens_m": float(propagation_um / 1e6),
            "pupil_shape": pupil_shape,
            "pupil_diameter_um": pupil_diameter_um,
            "padding_factor": padding_factor,
            "phase_depth_rad": float(phase_depth_rad),
            "detector_pixel_um": float(abs(detector_x[1] - detector_x[0])),
            "first_order_spacing_um": float(spacing),
        }

    def metadata(self, array_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        report = self.diffraction_report()
        data = super().metadata(array_shape=array_shape)
        data["transitions_half_period"] = report["transitions_half_period"]
        data["dammann_design"] = report
        data["mask_semantics"] = (
            "0=0 rad phase, 1=pi rad phase; crossed Dammann beam-splitter phase XOR binary Fresnel-lens phase"
            if self.include_binary_fresnel_lens
            else "0=0 rad phase, 1=pi rad phase; crossed Dammann beam-splitter phase for an external focusing optic"
        )
        return _jsonable(data)


@dataclass
class KinoformCRL1D:
    """Fixed-depth, surface-patterned silicon kinoform for parallel incidence.

    X-rays propagate parallel to the silicon surface. The GDS therefore
    contains a binary plan-view kinoform pattern that is etched vertically to
    one fixed depth; it is not a multilevel relief-height layout. Each
    cylindrical CRL focuses in ``x``. ``number_of_crls`` elements form one
    serial CRL train along ``y``; ``array_count`` repeats complete trains
    side-by-side along ``x`` to create parallel kinoform arrays. Each CRL
    contains two mirrored kinoform surfaces separated by a retained central
    silicon wall.
    """

    radius_um: float = 100.0
    xray_energy_keV: float = 12.0
    number_of_crls: int = 10
    crl_spacing_um: float = 20.0
    array_count: int = 1
    array_spacing_um: float = 20.0
    aperture_um: float = 100.0
    support_width_um: float = 140.0
    raster_resolution_um: float = 0.25
    fixed_etch_depth_um: float = 50.0
    crl_length_um: Optional[float] = None
    central_wall_um: float = 15.0
    frame_margin_um: float = 10.0
    support_web_width_um: float = 2.0
    material: str = "Si"
    density_g_cm3: float = 2.329

    def _validate(self) -> None:
        if self.radius_um <= 0 or self.xray_energy_keV <= 0 or self.aperture_um <= 0:
            raise ValueError("radius_um, xray_energy_keV, and aperture_um must be positive")
        if self.number_of_crls < 1 or self.array_count < 1 or self.fixed_etch_depth_um <= 0:
            raise ValueError("number_of_crls, array_count, and fixed_etch_depth_um must be positive")
        if self.crl_spacing_um < 0 or self.array_spacing_um < 0 or self.central_wall_um < 0 or self.raster_resolution_um <= 0 or self.frame_margin_um < 0 or self.support_web_width_um < 0:
            raise ValueError("crl_spacing_um, array_spacing_um, central_wall_um, frame_margin_um, and support_web_width_um must be non-negative; raster_resolution_um must be positive")
        if self.support_width_um <= self.aperture_um:
            raise ValueError("support_width_um must be larger than aperture_um")
        if 2.0 * self.support_web_width_um >= self.aperture_um:
            raise ValueError("support_web_width_um must be less than half the aperture_um")
        if self.crl_length_um is not None and self.crl_length_um <= 0:
            raise ValueError("crl_length_um must be positive when provided")

    @property
    def effective_surface_length_um(self) -> float:
        """Return the y length occupied by one kinoform surface."""

        if self.crl_length_um is not None:
            return float(self.crl_length_um)
        return float(self.optical_parameters()["two_pi_path_jump_um"])

    @property
    def effective_crl_length_um(self) -> float:
        """Return the complete double-surface CRL-cell length along y."""

        return 2.0 * self.effective_surface_length_um + self.central_wall_um

    @property
    def single_array_height_um(self) -> float:
        """Return the height of one serial double-surface CRL train."""

        return self.number_of_crls * self.effective_crl_length_um + (self.number_of_crls - 1) * self.crl_spacing_um

    @property
    def single_array_width_um(self) -> float:
        """Return the lateral width allocated to one parallel kinoform array."""

        return self.support_width_um

    def optical_parameters(self) -> Dict[str, float]:
        """Return silicon optics, in-plane 2-pi path jump, and CRL focus."""

        self._validate()
        beta, delta = _optical_constants(self.xray_energy_keV, self.material, self.density_g_cm3)
        wavelength_um = wavelength_nm(self.xray_energy_keV) / 1000.0
        two_pi_path_jump_um = wavelength_um / delta
        focal_length_um = self.radius_um / (2.0 * self.number_of_crls * delta)
        return {
            "wavelength_um": float(wavelength_um),
            "delta": float(delta),
            "beta": float(beta),
            "two_pi_path_jump_um": float(two_pi_path_jump_um),
            "fixed_etch_depth_um": float(self.fixed_etch_depth_um),
            "focal_length_um": float(focal_length_um),
            "focal_length_m": float(focal_length_um * 1e-6),
        }

    def field_size_um(self) -> Tuple[float, float]:
        active_width = self.array_count * self.single_array_width_um + (self.array_count - 1) * self.array_spacing_um
        return (
            self.single_array_height_um + 2.0 * self.frame_margin_um,
            active_width + 2.0 * self.frame_margin_um,
        )

    def surface_profile_um(self) -> ArrayLike:
        """Return the in-plane kinoform path profile for all parallel arrays."""

        optics = self.optical_parameters()
        _, width_um = self.field_size_um()
        cols = int(np.ceil(width_um / self.raster_resolution_um))
        x = (np.arange(cols, dtype=np.float64) + 0.5) * self.raster_resolution_um - width_um / 2.0
        profile = np.zeros(cols, dtype=np.float64)
        active_x0 = -width_um / 2.0 + self.frame_margin_um
        for array_index in range(self.array_count):
            array_center = active_x0 + array_index * (self.single_array_width_um + self.array_spacing_um) + self.single_array_width_um / 2.0
            local_x = x - array_center
            inside_array = np.abs(local_x) <= self.single_array_width_um / 2.0
            inside_aperture = inside_array & (np.abs(local_x) <= self.aperture_um / 2.0)
            # Each surface contributes x^2 / (2 R). The two mirrored
            # surfaces therefore give the total x^2 / R path term and
            # f = R / (2 N delta). A 2-pi material thickness remains
            # lambda / delta for each individual surface.
            profile[inside_aperture] = np.mod(
                local_x[inside_aperture] * local_x[inside_aperture] / (2.0 * self.radius_um),
                optics["two_pi_path_jump_um"],
            )
        return profile

    def surface_mask(self) -> ArrayLike:
        """Return the binary fixed-depth etch mask around supported kinoforms.

        A value of one is material removed by the fixed-depth etch.  The
        concave kinoform profiles, their central wall, outer frame,
        inter-array web, and lateral support rails remain silicon (zero) so
        the CRLs are not isolated.
        """

        height_um, width_um = self.field_size_um()
        rows = int(np.ceil(height_um / self.raster_resolution_um))
        profile = self.surface_profile_um()
        mask = np.zeros((rows, profile.size), dtype=np.uint8)
        active_height = self.single_array_height_um
        active_y0 = (height_um - active_height) / 2.0
        x = (np.arange(profile.size, dtype=np.float64) + 0.5) * self.raster_resolution_um - width_um / 2.0
        active_x0 = -width_um / 2.0 + self.frame_margin_um
        etchable_x = np.zeros(profile.size, dtype=bool)
        for array_index in range(self.array_count):
            array_center = active_x0 + array_index * (self.single_array_width_um + self.array_spacing_um) + self.single_array_width_um / 2.0
            local_x = x - array_center
            etchable_x |= np.abs(local_x) <= self.aperture_um / 2.0 - self.support_web_width_um
        for index in range(self.number_of_crls):
            cell_start = active_y0 + index * (self.effective_crl_length_um + self.crl_spacing_um)
            row_start = max(0, int(np.floor(cell_start / self.raster_resolution_um)))
            row_stop = min(rows, int(np.ceil((cell_start + self.effective_crl_length_um) / self.raster_resolution_um)))
            local_y = (np.arange(row_start, row_stop, dtype=np.float64) + 0.5) * self.raster_resolution_um - cell_start
            surface_length = self.effective_surface_length_um
            central_wall_start = surface_length
            central_wall_stop = surface_length + self.central_wall_um
            retain_lower_surface = local_y[:, None] < profile[None, :]
            retain_central_wall = (local_y[:, None] >= central_wall_start) & (local_y[:, None] < central_wall_stop)
            retain_upper_surface = local_y[:, None] >= (2.0 * surface_length + self.central_wall_um - profile[None, :])
            mask[row_start:row_stop, :] = (
                etchable_x[None, :] & ~(retain_lower_surface | retain_central_wall | retain_upper_surface)
            ).astype(np.uint8)
        return mask

    def etch_depth_map_um(self) -> ArrayLike:
        return self.surface_mask().astype(np.float64) * self.fixed_etch_depth_um

    def to_array(self) -> ArrayLike:
        """Return the binary union of all etched kinoform regions."""

        return self.surface_mask()

    def simulate_focal_spot(
        self,
        fabrication_resolution_um: float = 0.1,
        simulation_resolution_um: float = 0.02,
        gaussian_intensity_fwhm_um: Optional[float] = None,
        source_size_fwhm_um: float = 30.0,
        source_distance_m: float = 30.0,
        source_modes: int = 15,
        padding_factor: int = 4,
    ) -> Dict[str, Any]:
        """Simulate ideal and resolution-limited coherent/partial-coherent focus.

        The extended Gaussian source is represented by an incoherent weighted
        sum of tilted coherent modes.  ``fabrication_resolution_um`` samples
        the in-plane kinoform path profile using zero-order-hold pixels, while
        the fixed vertical etch depth remains unchanged.
        """

        if fabrication_resolution_um <= 0 or simulation_resolution_um <= 0:
            raise ValueError("fabrication_resolution_um and simulation_resolution_um must be positive")
        if source_size_fwhm_um < 0 or source_distance_m <= 0:
            raise ValueError("source_size_fwhm_um must be non-negative and source_distance_m must be positive")
        if source_modes < 1 or padding_factor < 1:
            raise ValueError("source_modes and padding_factor must be positive")
        optics = self.optical_parameters()
        intensity_fwhm_um = self.aperture_um if gaussian_intensity_fwhm_um is None else float(gaussian_intensity_fwhm_um)
        if intensity_fwhm_um <= 0:
            raise ValueError("gaussian_intensity_fwhm_um must be positive")
        base_samples = max(64, int(np.ceil(self.aperture_um / simulation_resolution_um)))
        if base_samples % 2:
            base_samples += 1
        padded_samples = base_samples * padding_factor
        x = (np.arange(base_samples, dtype=np.float64) - base_samples / 2.0 + 0.5) * simulation_resolution_um
        inside_aperture = np.abs(x) <= self.aperture_um / 2.0
        intensity_sigma_um = intensity_fwhm_um / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        amplitude = np.exp(-x * x / (4.0 * intensity_sigma_um * intensity_sigma_um)) * inside_aperture
        wavelength_um = optics["wavelength_um"]
        two_pi_jump_um = optics["two_pi_path_jump_um"]
        ideal_profile = np.mod(x * x / self.radius_um, two_pi_jump_um)
        ideal_profile[~inside_aperture] = 0.0
        fabricated_profile = np.zeros_like(x)
        coarse_count = int(np.ceil(self.aperture_um / fabrication_resolution_um))
        coarse_centers = -self.aperture_um / 2.0 + (np.arange(coarse_count, dtype=np.float64) + 0.5) * fabrication_resolution_um
        coarse_profile = np.mod(coarse_centers * coarse_centers / self.radius_um, two_pi_jump_um)
        coarse_index = np.floor((x + self.aperture_um / 2.0) / fabrication_resolution_um).astype(int)
        valid = inside_aperture & (coarse_index >= 0) & (coarse_index < coarse_count)
        fabricated_profile[valid] = coarse_profile[coarse_index[valid]]
        focal_length_um = optics["focal_length_um"]
        phase_scale = -2.0 * np.pi * self.number_of_crls * optics["delta"] / wavelength_um
        # The ideal unwrapped phase gives the exact thin-lens phase. The
        # fabricated kinoform retains the wrapped profile sampled at the
        # specified lithographic resolution.
        ideal_transmission = np.exp(-1j * np.pi * x * x / (wavelength_um * focal_length_um))
        fabricated_transmission = np.exp(1j * phase_scale * fabricated_profile)
        focal_x = np.fft.fftshift(np.fft.fftfreq(padded_samples, d=simulation_resolution_um)) * wavelength_um * focal_length_um
        wave_number_per_um = 2.0 * np.pi / wavelength_um
        focus_chirp = np.exp(1j * np.pi * x * x / (wavelength_um * focal_length_um))

        def focal_intensity(transmission: ArrayLike, angle_rad: float) -> ArrayLike:
            field = amplitude * transmission * focus_chirp * np.exp(1j * wave_number_per_um * angle_rad * x)
            padded = np.zeros(padded_samples, dtype=np.complex128)
            start = (padded_samples - base_samples) // 2
            padded[start:start + base_samples] = field
            intensity = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(padded)))) ** 2
            return intensity / max(float(np.sum(intensity)), np.finfo(np.float64).tiny)

        def partial_intensity(transmission: ArrayLike, sigma_angle_rad: float) -> ArrayLike:
            if sigma_angle_rad == 0.0:
                return focal_intensity(transmission, 0.0)
            nodes, weights = np.polynomial.hermite.hermgauss(source_modes)
            result = np.zeros(padded_samples, dtype=np.float64)
            for node, weight in zip(nodes, weights):
                result += float(weight / math.sqrt(np.pi)) * focal_intensity(transmission, math.sqrt(2.0) * sigma_angle_rad * node)
            return result / max(float(np.sum(result)), np.finfo(np.float64).tiny)

        def fwhm_um(intensity: ArrayLike) -> float:
            center = int(np.argmin(np.abs(focal_x)))
            peak = float(intensity[center])
            half = peak / 2.0
            left = center
            while left > 0 and intensity[left] >= half:
                left -= 1
            right = center
            while right < intensity.size - 1 and intensity[right] >= half:
                right += 1
            return float(focal_x[right] - focal_x[left]) if left < center < right else 0.0

        source_sigma_angle_rad = source_size_fwhm_um / (2.0 * math.sqrt(2.0 * math.log(2.0)) * source_distance_m * 1e6)
        coherent_ideal = focal_intensity(ideal_transmission, 0.0)
        coherent_fabricated = focal_intensity(fabricated_transmission, 0.0)
        partial_ideal = partial_intensity(ideal_transmission, source_sigma_angle_rad)
        partial_fabricated = partial_intensity(fabricated_transmission, source_sigma_angle_rad)
        labels = {
            "ideal_coherent": coherent_ideal,
            "fabricated_coherent": coherent_fabricated,
            "ideal_partial": partial_ideal,
            "fabricated_partial": partial_fabricated,
        }
        peak_intensity = {name: float(np.max(intensity)) for name, intensity in labels.items()}
        return {
            "x_um": x,
            "focal_x_um": focal_x,
            "ideal_profile_um": ideal_profile,
            "fabricated_profile_um": fabricated_profile,
            "source_sigma_angle_rad": float(source_sigma_angle_rad),
            "source_blur_sigma_um": float(source_sigma_angle_rad * focal_length_um),
            "focal_length_m": optics["focal_length_m"],
            "fabrication_resolution_um": float(fabrication_resolution_um),
            "simulation_resolution_um": float(simulation_resolution_um),
            "gaussian_intensity_fwhm_um": intensity_fwhm_um,
            "source_size_fwhm_um": float(source_size_fwhm_um),
            "source_distance_m": float(source_distance_m),
            "source_modes": source_modes,
            "intensities": labels,
            "fwhm_um": {name: fwhm_um(intensity) for name, intensity in labels.items()},
            "peak_intensity": peak_intensity,
            "relative_peak": {
                "fabricated_over_ideal_coherent": peak_intensity["fabricated_coherent"] / peak_intensity["ideal_coherent"],
                "fabricated_over_ideal_partial": peak_intensity["fabricated_partial"] / peak_intensity["ideal_partial"],
            },
        }

    def frame_rectangle_um(self) -> Tuple[float, float]:
        """Return width and height of the rectangular support-outline geometry."""

        height_um, width_um = self.field_size_um()
        return (width_um, height_um)

    def metadata(self) -> Dict[str, Any]:
        optics = self.optical_parameters()
        height_um, width_um = self.field_size_um()
        data = asdict(self)
        data.update(optics)
        data.update(
            {
                "pattern_class": self.__class__.__name__,
                "array_shape_yx": [int(np.ceil(height_um / self.raster_resolution_um)), int(np.ceil(width_um / self.raster_resolution_um))],
                "field_size_um": [height_um, width_um],
                "support_rectangle_width_um": width_um,
                "single_surface_length_um": self.effective_surface_length_um,
                "crl_cell_length_um": self.effective_crl_length_um,
                "central_wall_um": self.central_wall_um,
                "surface_sag_formula": "x^2 / (2 * R) per surface; x^2 / R for the two-surface CRL",
                "two_pi_phase_thickness_formula": "lambda / delta",
                "single_array_height_um": self.single_array_height_um,
                "single_array_width_um": self.single_array_width_um,
                "array_layout": "array_count parallel serial CRL trains aligned side-by-side along x",
                "focal_length_formula": "f = R / (2 * N * delta)",
                "incidence_geometry": "X-rays propagate parallel to the silicon surface; the plan-view kinoform is etched vertically to fixed_etch_depth_um",
                "mask_semantics": "one denotes a fixed-depth etched void; zero denotes retained silicon, including the concave kinoform and its support frame/webs",
                "gds_layer_semantics": "layer 1 is the fixed-depth etch around the retained kinoform material; layer 100 is an optional support-outline annotation",
            }
        )
        return _jsonable(data)


@dataclass
class HartmannPlate(MaskPattern):
    period_um: float = 80.0
    lattice: str = "square"
    hole_shape: str = "circle"
    duty_cycle: float = 0.35
    hole_size_um: Optional[float] = None
    n_sides: int = 6
    hole_rotation_deg: float = 0.0
    hole_rotation: Optional[float] = None
    array_rotation_deg: float = 0.0

    def _centers(self) -> List[Point]:
        h, w = self.field_size_um
        margin = self.period_um
        centers: List[Point] = []
        if self.lattice == "square":
            xs = np.arange(-w / 2 - margin, w / 2 + margin, self.period_um)
            ys = np.arange(-h / 2 - margin, h / 2 + margin, self.period_um)
            centers = [(float(x), float(y)) for y in ys for x in xs]
        elif self.lattice == "hexagonal":
            dy = self.period_um * math.sqrt(3.0) / 2.0
            ys = np.arange(-h / 2 - margin, h / 2 + margin, dy)
            for j, y in enumerate(ys):
                offset = 0.5 * self.period_um if j % 2 else 0.0
                xs = np.arange(-w / 2 - margin, w / 2 + margin, self.period_um) + offset
                centers.extend((float(x), float(y)) for x in xs)
        else:
            raise ValueError("lattice must be 'square' or 'hexagonal'")
        theta = np.deg2rad(self.array_rotation_deg)
        c, s = math.cos(theta), math.sin(theta)
        return [(c * x - s * y, s * x + c * y) for x, y in centers]

    def _binary(self) -> ArrayLike:
        hole = self.hole_size_um if self.hole_size_um is not None else self.period_um * self.duty_cycle
        if hole <= 0:
            raise ValueError("hole_size_um must be positive")
        x, y = _grid(self.field_size_um, self.raster_resolution_um, self.supersample)
        mask = np.zeros_like(x, dtype=bool)
        for cx, cy in self._centers():
            local_x = x - cx
            local_y = y - cy
            if abs(cx) > self.field_size_um[1] / 2 + hole or abs(cy) > self.field_size_um[0] / 2 + hole:
                continue
            rotation = self.hole_rotation_deg if self.hole_rotation is None else float(self.hole_rotation)
            mask |= _shape_mask(local_x, local_y, self.hole_shape, hole, rotation, self.n_sides)
        return _downsample(mask, self.supersample)


@dataclass
class RandomBinaryMask(MaskPattern):
    pitch_um: float = 20.0
    fill_fraction: float = 0.5
    seed: int = 1
    connected: bool = False

    def _binary(self) -> ArrayLike:
        if self.pitch_um <= 0:
            raise ValueError("pitch_um must be positive")
        if not 0 <= self.fill_fraction <= 1:
            raise ValueError("fill_fraction must be between 0 and 1")
        rng = np.random.default_rng(self.seed)
        h, w = self.field_size_um
        n_y = int(np.ceil(h / self.pitch_um))
        n_x = int(np.ceil(w / self.pitch_um))
        cells = (rng.random((n_y, n_x)) < self.fill_fraction).astype(np.uint8)
        if self.connected:
            cells = _connect_material(cells)
        scale = max(1, int(round(self.pitch_um / self.raster_resolution_um)))
        mask = np.repeat(np.repeat(cells, scale, axis=0), scale, axis=1)
        target_y = int(np.ceil(h / self.raster_resolution_um))
        target_x = int(np.ceil(w / self.raster_resolution_um))
        return mask[:target_y, :target_x].astype(np.uint8)


@dataclass
class RandomHartmann(MaskPattern):
    n_holes: Optional[int] = 100
    target_density_per_mm2: Optional[float] = None
    min_center_distance_um: float = 60.0
    hole_shape: str = "circle"
    hole_size_um: Union[float, Dict[str, float]] = 20.0
    hole_rotation: Union[float, str, Dict[str, float]] = 0.0
    seed: int = 1
    edge_clearance_um: float = 10.0
    n_sides: int = 6
    n_cores: int = 1
    chunk_rows: int = 256
    holes: List[Dict[str, float]] = field(default_factory=list)

    def _hole_size(self, rng: np.random.Generator) -> float:
        spec = self.hole_size_um
        if isinstance(spec, dict):
            if spec.get("type") == "uniform":
                return float(rng.uniform(spec["low"], spec["high"]))
            if spec.get("type") == "normal":
                return float(max(spec.get("min", 0.0), rng.normal(spec["mean"], spec["std"])))
            raise ValueError("hole_size_um distribution type must be 'uniform' or 'normal'")
        return float(spec)

    def _rotation(self, rng: np.random.Generator) -> float:
        spec = self.hole_rotation
        if spec == "random":
            return float(rng.uniform(0.0, 360.0))
        if isinstance(spec, dict):
            if spec.get("type") == "uniform":
                return float(rng.uniform(spec["low"], spec["high"]))
            if spec.get("type") == "normal":
                return float(rng.normal(spec["mean"], spec["std"]))
            raise ValueError("hole_rotation distribution type must be 'uniform' or 'normal'")
        return float(spec)

    def _maximum_bounding_radius_um(self) -> float:
        """Return a conservative footprint bound for spatial-hash placement."""

        spec = self.hole_size_um
        if isinstance(spec, dict):
            if spec.get("type") == "uniform":
                size = float(spec["high"])
            elif spec.get("type") == "normal":
                size = float(spec.get("max", spec["mean"] + 4.0 * spec["std"]))
            else:
                raise ValueError("hole_size_um distribution type must be 'uniform' or 'normal'")
        else:
            size = float(spec)
        return _bounding_radius(size, self.hole_shape, self.n_sides)

    def _target_hole_count(self) -> int:
        h, w = self.field_size_um
        if self.target_density_per_mm2 is not None:
            return int(round(self.target_density_per_mm2 * h * w / 1e6))
        return int(self.n_holes or 0)

    def placement_limits(self, random_sequential_fraction: float = 0.58) -> Dict[str, float]:
        """Return density and raster limits for the current hole geometry.

        The hexagonal result is a strict geometric upper bound. The random
        sequential reference is deliberately lower: it is a practical target
        for this non-overlapping random-placement algorithm, not a physical
        packing bound.
        """

        if not 0.0 < random_sequential_fraction < 1.0:
            raise ValueError("random_sequential_fraction must be between zero and one")
        height_um, width_um = self.field_size_um
        area_mm2 = height_um * width_um / 1e6
        max_radius_um = self._maximum_bounding_radius_um()
        effective_min_center_distance_um = max(self.min_center_distance_um, 2.0 * max_radius_um)
        theoretical_max_density_per_mm2 = 2e6 / (math.sqrt(3.0) * effective_min_center_distance_um**2)
        target_holes = self._target_hole_count()
        requested_density_per_mm2 = target_holes / area_mm2
        full_rows = int(np.ceil(height_um / self.raster_resolution_um))
        full_cols = int(np.ceil(width_um / self.raster_resolution_um))
        recommended_density_per_mm2 = random_sequential_fraction * theoretical_max_density_per_mm2
        return {
            "field_area_mm2": float(area_mm2),
            "requested_hole_count": int(target_holes),
            "requested_density_per_mm2": float(requested_density_per_mm2),
            "maximum_hole_bounding_radius_um": float(max_radius_um),
            "effective_min_center_distance_um": float(effective_min_center_distance_um),
            "theoretical_hexagonal_max_density_per_mm2": float(theoretical_max_density_per_mm2),
            "random_sequential_reference_fraction": float(random_sequential_fraction),
            "random_sequential_reference_density_per_mm2": float(recommended_density_per_mm2),
            "random_sequential_reference_hole_count": int(math.floor(recommended_density_per_mm2 * area_mm2)),
            "full_raster_shape_yx": [full_rows, full_cols],
            "full_boolean_raster_gib": float(full_rows * full_cols / 1024**3),
        }

    def preview_array(self, max_pixels: int = 1200) -> ArrayLike:
        """Rasterize a bounded preview without allocating the fabrication mask.

        The result is only for display. GDS export should use the original
        ``raster_resolution_um`` through ``save_random_hartmann_gds``.
        """

        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        holes = self.holes or self.generate_holes()
        height_um, width_um = self.field_size_um
        preview_resolution_um = max(
            self.raster_resolution_um,
            max(height_um, width_um) / max_pixels,
        )
        rows = int(np.ceil(height_um / preview_resolution_um))
        _, preview = _random_hartmann_chunk(
            (
                0,
                rows,
                height_um,
                width_um,
                preview_resolution_um,
                holes,
                self.hole_shape,
                self.n_sides,
            )
        )
        return self._apply_polarity(preview)

    def generate_holes(self) -> List[Dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        h, w = self.field_size_um
        target = self._target_hole_count()
        if target <= 0:
            raise ValueError("n_holes or target_density_per_mm2 must request at least one hole")
        limits = self.placement_limits()
        max_radius = limits["maximum_hole_bounding_radius_um"]
        minimum_spacing = limits["effective_min_center_distance_um"]
        max_density_per_mm2 = limits["theoretical_hexagonal_max_density_per_mm2"]
        requested_density_per_mm2 = limits["requested_density_per_mm2"]
        if requested_density_per_mm2 > max_density_per_mm2:
            raise ValueError(
                f"Requested density {requested_density_per_mm2:.3g} holes/mm^2 exceeds the "
                f"hexagonal-packing upper bound {max_density_per_mm2:.3g} holes/mm^2 for "
                f"minimum spacing {minimum_spacing:.3g} um. Reduce target_density_per_mm2 "
                "or min_center_distance_um."
            )
        holes: List[Dict[str, float]] = []
        # A spatial hash limits every trial to nearby candidates rather than
        # comparing it with every previously accepted hole.
        cell_size = minimum_spacing
        spatial_hash: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
        attempts = max(2000, target * 500)
        for _ in range(attempts):
            if len(holes) >= target:
                break
            size = self._hole_size(rng)
            radius = _bounding_radius(size, self.hole_shape, self.n_sides)
            margin = self.edge_clearance_um + radius
            if margin * 2 >= w or margin * 2 >= h:
                raise RuntimeError(
                    "Hole size plus edge_clearance_um does not fit inside field_size_um. "
                    "Reduce hole_size_um/edge_clearance_um or increase field_size_um."
                )
            x = float(rng.uniform(-w / 2 + margin, w / 2 - margin))
            y = float(rng.uniform(-h / 2 + margin, h / 2 - margin))
            ok = True
            cell_x = int(math.floor((x + w / 2.0) / cell_size))
            cell_y = int(math.floor((y + h / 2.0) / cell_size))
            for neighbor_y in range(cell_y - 1, cell_y + 2):
                for neighbor_x in range(cell_x - 1, cell_x + 2):
                    for other in spatial_hash.get((neighbor_x, neighbor_y), []):
                        min_sep = max(self.min_center_distance_um, radius + other["bounding_radius_um"])
                        if (x - other["x_um"]) ** 2 + (y - other["y_um"]) ** 2 < min_sep**2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                hole = {
                    "x_um": x,
                    "y_um": y,
                    "hole_size_um": size,
                    "rotation_deg": self._rotation(rng),
                    "bounding_radius_um": radius,
                }
                holes.append(hole)
                spatial_hash.setdefault((cell_x, cell_y), []).append(hole)
        if len(holes) < target:
            raise RuntimeError(
                f"Could only place {len(holes)} of {target} holes. "
                "Reduce n_holes, increase field_size_um, or reduce min_center_distance_um."
            )
        self.holes = holes
        return holes

    def _binary(self) -> ArrayLike:
        holes = self.holes or self.generate_holes()
        if self.n_cores < 1:
            raise ValueError("n_cores must be positive")
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")

        # Chunked multiprocessing preserves the exact geometry for the common
        # supersample=1 case while keeping each worker's temporary arrays small.
        # The original path remains the reference for supersampled masks because
        # downsampling must not split a supersample block across workers.
        if self.supersample != 1:
            x, y = _grid(self.field_size_um, self.raster_resolution_um, self.supersample)
            mask = np.zeros_like(x, dtype=bool)
            for hole in holes:
                mask |= _shape_mask(
                    x - hole["x_um"],
                    y - hole["y_um"],
                    self.hole_shape,
                    hole["hole_size_um"],
                    hole["rotation_deg"],
                    self.n_sides,
                )
            return _downsample(mask, self.supersample)

        height_um, width_um = self.field_size_um
        rows = int(np.ceil(height_um / self.raster_resolution_um))
        tasks = [
            (
                row_start,
                min(rows, row_start + self.chunk_rows),
                height_um,
                width_um,
                self.raster_resolution_um,
                holes,
                self.hole_shape,
                self.n_sides,
            )
            for row_start in range(0, rows, self.chunk_rows)
        ]
        output = np.empty((rows, int(np.ceil(width_um / self.raster_resolution_um))), dtype=np.uint8)
        workers = min(self.n_cores, len(tasks))
        try:
            if self.n_cores == 1:
                chunks = map(_random_hartmann_chunk, tasks)
                for row_start, chunk in chunks:
                    output[row_start:row_start + chunk.shape[0]] = chunk
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    chunks = executor.map(_random_hartmann_chunk, tasks)
                    for row_start, chunk in chunks:
                        output[row_start:row_start + chunk.shape[0]] = chunk
        except (OSError, RuntimeError, PermissionError) as exc:
            # Restricted Windows/Jupyter sessions may deny process pipes. NumPy
            # releases the GIL in the raster operations, so threads remain useful.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                chunks = executor.map(_random_hartmann_chunk, tasks)
                for row_start, chunk in chunks:
                    output[row_start:row_start + chunk.shape[0]] = chunk
        return output

    def metadata(self, array_shape: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        data = super().metadata(array_shape=array_shape)
        data["holes"] = self.holes or self.generate_holes()
        return _jsonable(data)


def _shape_mask(
    x: ArrayLike,
    y: ArrayLike,
    shape: str,
    size_um: float,
    rotation_deg: float = 0.0,
    n_sides: int = 6,
) -> ArrayLike:
    shape = shape.lower()
    if shape == "circle":
        return x**2 + y**2 <= (size_um / 2.0) ** 2
    xr, yr = rotate_xy(x, y, rotation_deg)
    if shape == "square":
        return (np.abs(xr) <= size_um / 2.0) & (np.abs(yr) <= size_um / 2.0)
    if shape == "regular_polygon":
        if n_sides < 3:
            raise ValueError("n_sides must be >= 3")
        radius = size_um / 2.0
        angles = np.arctan2(yr, xr)
        radii = np.sqrt(xr**2 + yr**2)
        sector = 2 * np.pi / n_sides
        local = np.mod(angles + sector / 2.0, sector) - sector / 2.0
        boundary = radius * np.cos(np.pi / n_sides) / np.cos(local)
        return radii <= boundary
    raise ValueError("hole_shape must be 'circle', 'square', or 'regular_polygon'")


def _bounding_radius(size_um: float, shape: str, n_sides: int = 6) -> float:
    if shape == "square":
        return math.sqrt(2.0) * size_um / 2.0
    return size_um / 2.0


def _connect_material(cells: ArrayLike) -> ArrayLike:
    material = 1 - cells.astype(np.uint8)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labeled, n_components = ndimage.label(material, structure=structure)
    if n_components <= 1:
        return cells
    sizes = ndimage.sum(material, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes) + 1)
    main = labeled == largest_label
    dist, indices = ndimage.distance_transform_edt(1 - main, return_indices=True)
    for label_id in range(1, n_components + 1):
        if label_id == largest_label:
            continue
        island = labeled == label_id
        if not np.any(island):
            continue
        island_dist = np.where(island, dist, np.inf)
        p0 = np.array(np.unravel_index(np.argmin(island_dist), island_dist.shape))
        p1 = np.array([indices[0, p0[0], p0[1]], indices[1, p0[0], p0[1]]])
        steps = max(1, int(np.ceil(np.linalg.norm(p1 - p0) * 3)))
        rr = np.round(np.linspace(p0[0], p1[0], steps)).astype(int)
        cc = np.round(np.linspace(p0[1], p1[1], steps)).astype(int)
        material[rr, cc] = 1
    return 1 - material


def absorber_requirements(
    energy_keV: float,
    material: str = "Au",
    transmission_target: float = 0.03,
    min_feature_um: Optional[float] = None,
    grating_period_um: Optional[float] = None,
    talbot_orders: Sequence[int] = (1, 2, 3),
) -> Dict[str, Any]:
    """Estimate absorber/phase thicknesses from lightweight fallback data.

    The values are engineering estimates when xraylib is unavailable. They are
    intended for feasibility screening, not final fabrication sign-off.
    """

    lam_nm = wavelength_nm(energy_keV)
    lam_um = lam_nm / 1000.0
    beta, delta = _optical_constants_fallback(energy_keV, material)
    amp_t_um = -math.log(transmission_target) * lam_um / (4.0 * math.pi * beta)
    pi_t_um = lam_um / (2.0 * delta)
    half_pi_t_um = pi_t_um / 2.0
    result: Dict[str, Any] = {
        "energy_keV": energy_keV,
        "wavelength_nm": lam_nm,
        "material": material,
        "beta_estimate": beta,
        "delta_estimate": delta,
        "amplitude_thickness_um": amp_t_um,
        "pi_phase_thickness_um": pi_t_um,
        "half_pi_phase_thickness_um": half_pi_t_um,
        "warnings": [],
    }
    if min_feature_um:
        aspect = amp_t_um / min_feature_um
        result["amplitude_aspect_ratio"] = aspect
        if energy_keV >= 2 and aspect > 10:
            result["warnings"].append("Hard X-ray absorber aspect ratio exceeds 10:1.")
        if energy_keV < 2 and aspect > 3:
            result["warnings"].append("Soft X-ray absorber aspect ratio exceeds 3:1.")
    if energy_keV < 1:
        result["warnings"].append("Soft X-ray masks below about 1 keV often need free-standing membranes or bridges.")
    if grating_period_um:
        result["fractional_talbot_distances_m"] = {
            int(n): float(n * grating_period_um**2 / (8.0 * lam_um) * 1e-6) for n in talbot_orders
        }
    return result


def _optical_constants_fallback(energy_keV: float, material: str) -> Tuple[float, float]:
    material = material.lower()
    base = {
        "au": (2.9e-6, 1.7e-5),
        "ni": (8.5e-7, 6.0e-6),
        "w": (2.4e-6, 1.4e-5),
        "ta": (2.2e-6, 1.3e-5),
        "si": (1.2e-8, 1.5e-6),
    }
    beta_10kev, delta_10kev = base.get(material, base["au"])
    scale = (10.0 / energy_keV) ** 2
    return beta_10kev * scale, delta_10kev * scale


def _optical_constants(energy_keV: float, material: str, density_g_cm3: float) -> Tuple[float, float]:
    """Return beta and delta, preferring xraydb when it is available."""

    if density_g_cm3 <= 0:
        raise ValueError("density_g_cm3 must be positive")
    try:
        import xraydb

        delta, beta, _ = xraydb.xray_delta_beta(material, density_g_cm3, energy_keV * 1000.0)
        return float(beta), float(delta)
    except (ImportError, ValueError, TypeError):
        return _optical_constants_fallback(energy_keV, material)


def analytic_hartmann_first_guess(
    energy_keV: float,
    mask_to_detector_m: float,
    pixel_size_um: float,
    source_to_mask_m: Union[float, str] = math.inf,
    source_size_um: float = 0.0,
    detector_psf_fwhm_um: float = 0.0,
    min_feature_um: float = 1.0,
    max_slope_urad: float = 0.0,
    separation_factor: float = 3.0,
) -> Dict[str, float]:
    """Compute a practical first guess for RandomHartmann parameters."""

    lam_um = wavelength_nm(energy_keV) / 1000.0
    l1 = math.inf if source_to_mask_m == "inf" else float(source_to_mask_m)
    l2_um = mask_to_detector_m * 1e6
    magnification = 1.0 if math.isinf(l1) else (l1 + mask_to_detector_m) / l1
    d_opt = math.sqrt(max(2.44 * lam_um * l2_um / magnification, 0.0))
    d_um = max(d_opt, min_feature_um)
    source_blur = 0.0 if math.isinf(l1) else source_size_um * mask_to_detector_m / l1
    geom = magnification * d_um
    diffraction = 2.44 * lam_um * l2_um / d_um
    fwhm_um = math.sqrt(geom**2 + diffraction**2 + source_blur**2 + detector_psf_fwhm_um**2)
    target_min = 3.0 * pixel_size_um
    target_max = 7.0 * pixel_size_um
    if fwhm_um < target_min:
        d_um = max(d_um, target_min / max(magnification, 1e-9))
    elif fwhm_um > target_max and geom > diffraction:
        d_um = max(min_feature_um, target_max / max(magnification, 1e-9))
    geom = magnification * d_um
    diffraction = 2.44 * lam_um * l2_um / d_um
    fwhm_um = math.sqrt(geom**2 + diffraction**2 + source_blur**2 + detector_psf_fwhm_um**2)
    slope_margin_um = abs(max_slope_urad) * 1e-6 * l2_um
    detector_spacing_um = separation_factor * fwhm_um + 2.0 * slope_margin_um
    mask_spacing_um = detector_spacing_um / magnification
    return {
        "wavelength_nm": wavelength_nm(energy_keV),
        "magnification": magnification,
        "hole_size_um": d_um,
        "spot_fwhm_um": fwhm_um,
        "spot_fwhm_pixels": fwhm_um / pixel_size_um,
        "min_center_distance_um": mask_spacing_um,
        "detector_min_spacing_um": detector_spacing_um,
    }


def generate_pattern(
    pattern_type: str,
    area_size_mm: Tuple[float, float],
    pixel_size_um: float,
    period_um: float = 10.0,
    duty_cycle: float = 0.5,
    random_pitch_um: float = 10.0,
    connected: bool = False,
    seed: int = 1,
    **kwargs: Any,
) -> Tuple[ArrayLike, Tuple[float, float, float, float]]:
    """Backward-compatible wrapper around the class-based API."""

    field = (area_size_mm[0] * 1000.0, area_size_mm[1] * 1000.0)
    common = {"field_size_um": field, "raster_resolution_um": pixel_size_um, **kwargs}
    pt = pattern_type.lower()
    if pt in {"1dgrating", "grating"}:
        pattern = Grating(period_um=period_um, duty_cycle=duty_cycle, dimensionality="1D", **common)
    elif pt == "grid":
        pattern = Grating(period_um=period_um, duty_cycle=duty_cycle, dimensionality="2D", combine_mode="and", **common)
    elif pt == "checkerboard":
        pattern = Checkerboard(period_um=period_um, duty_cycle=duty_cycle, **common)
    elif pt in {"dammann", "dammann_grating"}:
        pattern = DammannGrating(period_um=period_um, **common)
    elif pt in {"hartmann", "hartmannplate", "hartmann_plate"}:
        pattern = HartmannPlate(period_um=period_um, duty_cycle=duty_cycle, **common)
    elif pt == "random":
        pattern = RandomBinaryMask(pitch_um=random_pitch_um, fill_fraction=duty_cycle, connected=connected, seed=seed, **common)
    elif pt == "random_hartmann":
        pattern = RandomHartmann(seed=seed, **common)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    extent = (0, area_size_mm[1], 0, area_size_mm[0])
    return pattern.to_array(), extent


def _load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install PyYAML to read YAML configs, or use JSON.") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def pattern_from_config(config: Dict[str, Any]) -> MaskPattern:
    kind = config.pop("type").lower()
    cls = {
        "grating": Grating,
        "checkerboard": Checkerboard,
        "dammann": DammannGrating,
        "dammann_grating": DammannGrating,
        "hartmann": HartmannPlate,
        "hartmannplate": HartmannPlate,
        "random": RandomBinaryMask,
        "randombinarymask": RandomBinaryMask,
        "random_hartmann": RandomHartmann,
        "randomhartmann": RandomHartmann,
    }.get(kind)
    if cls is None:
        raise ValueError(f"Unknown config pattern type: {kind}")
    return cls(**config)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate X-ray wavefront-sensing binary mask patterns.")
    parser.add_argument("--config", help="JSON/YAML config file. Must include a 'type' field.")
    parser.add_argument("--output-dir", default="generated_patterns")
    parser.add_argument("--stem", default="mask")
    parser.add_argument("--preview", action="store_true", help="Also save a downsampled preview PNG.")
    args = parser.parse_args(argv)
    if not args.config:
        parser.error("--config is required")
    pattern = pattern_from_config(_load_config(args.config))
    paths = pattern.save_all(args.output_dir, args.stem)
    if not args.preview:
        paths["preview"].unlink(missing_ok=True)
        del paths["preview"]
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
