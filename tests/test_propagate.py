"""Validation suite for src/propagate.py.

Run as:
    python tests/test_propagate.py

The script prints a pass/fail table and writes results/test_propagate_report.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from propagate import fourier_visibility, make_amp_grating, make_pi_grating, propagate


WAVELENGTH = 1.0e-10
P = 8.0e-6
LX = 8 * P
NX = 4096
NY = 8
DX = LX / NX
Z1 = P * P / (8 * WAVELENGTH)
Z2 = P * P / (4 * WAVELENGTH)
ZT = 2 * P * P / WAVELENGTH
FOUR_OVER_PI = 4.0 / math.pi


def main() -> int:
    results = run_acceptance_suite()

    print_report(results)
    write_json_report(results)
    return 0 if all(item["passed"] for item in results) else 1


def test_propagation_acceptance_suite():
    results = run_acceptance_suite()
    assert all(item["passed"] for item in results), results


def run_acceptance_suite():
    results = []
    results.extend(run_energy_conservation())
    results.extend(run_phase_grating_fractional_talbot())
    results.extend(run_absorption_grating_talbot())
    results.extend(run_sampling_convergence())
    return results


def add_result(results, name, measured, expected, tolerance, passed, details=None):
    results.append(
        {
            "name": name,
            "measured": _json_safe(measured),
            "expected": _json_safe(expected),
            "tolerance": _json_safe(tolerance),
            "passed": bool(passed),
            "details": details or "",
        }
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_energy_conservation():
    results = []
    rng = np.random.default_rng(42)
    phase = rng.integers(0, 2, size=(1024, 1024), dtype=np.int8) * np.pi
    u0 = np.exp(1j * phase).astype(np.complex128)
    uz = propagate(u0, dx=15.625e-9, wavelength=WAVELENGTH, z=Z1, kernel="asm")
    e0 = float(np.sum(np.abs(u0) ** 2))
    ez = float(np.sum(np.abs(uz) ** 2))
    rel_error = abs(ez - e0) / e0
    add_result(
        results,
        "1 energy conservation",
        rel_error,
        "<= 1e-10",
        1e-10,
        rel_error <= 1e-10,
    )
    return results


def run_phase_grating_fractional_talbot():
    results = []
    _assert_geometry()
    grating = make_pi_grating(NY, NX, DX, P)
    duty = float(np.mean(np.isclose(grating, -1.0 + 0.0j)))
    add_result(
        results,
        "2 geometry pi grating duty cycle",
        duty,
        0.5,
        1e-12,
        abs(duty - 0.5) < 1e-12,
    )

    uz1 = propagate(grating, DX, WAVELENGTH, Z1, kernel="asm")
    i1 = np.abs(uz1) ** 2
    v1 = fourier_visibility(i1, DX, 2.0 / P)
    sorted_i = np.sort(i1.ravel())
    low_median = float(np.median(sorted_i[: int(0.4 * sorted_i.size)]))
    high_median = float(np.median(sorted_i[int(0.6 * sorted_i.size) :]))
    mean_error = abs(float(i1.mean()) - 1.0)
    pass_2a = (
        abs(v1 - FOUR_OVER_PI) <= 0.006
        and 1.95 <= high_median <= 2.05
        and low_median <= 0.05
        and mean_error <= 1e-10
    )
    add_result(
        results,
        "2a pi grating peak visibility at 80 mm",
        {"visibility": v1, "low_median": low_median, "high_median": high_median, "mean_error": mean_error},
        {"visibility": FOUR_OVER_PI, "low_median": "<= 0.05", "high_median": "1.95..2.05", "mean_error": "<= 1e-10"},
        {"visibility": 0.006},
        pass_2a,
    )

    uz2_asm = propagate(grating, DX, WAVELENGTH, Z2, kernel="asm")
    i2_asm = np.abs(uz2_asm) ** 2
    v2_asm = fourier_visibility(i2_asm, DX, 2.0 / P)
    cv2_asm = float(i2_asm.std() / i2_asm.mean())
    uz2_fresnel = propagate(grating, DX, WAVELENGTH, Z2, kernel="fresnel")
    i2_fresnel = np.abs(uz2_fresnel) ** 2
    cv2_fresnel = float(i2_fresnel.std() / i2_fresnel.mean())
    pass_2b = v2_asm <= 0.005 and cv2_asm <= 1e-3 and cv2_fresnel <= 1e-9
    add_result(
        results,
        "2b pi grating zero contrast at 160 mm",
        {"visibility_asm": v2_asm, "cv_asm": cv2_asm, "cv_fresnel": cv2_fresnel},
        {"visibility_asm": "<= 0.005", "cv_asm": "<= 1e-3", "cv_fresnel": "<= 1e-9"},
        None,
        pass_2b,
    )

    z_mm = [20, 40, 60, 80, 100, 120, 140, 160, 200, 240]
    measured = []
    expected = []
    max_abs_error = 0.0
    for mm in z_mm:
        z = mm * 1e-3
        v = fourier_visibility(np.abs(propagate(grating, DX, WAVELENGTH, z, kernel="asm")) ** 2, DX, 2.0 / P)
        ref = FOUR_OVER_PI * abs(math.sin(4.0 * math.pi * WAVELENGTH * z / (P * P)))
        measured.append(v)
        expected.append(ref)
        max_abs_error = max(max_abs_error, abs(v - ref))
    add_result(
        results,
        "2c pi grating visibility curve",
        {"max_abs_error": max_abs_error, "values": measured},
        {"values": expected},
        0.01,
        max_abs_error <= 0.01,
    )
    return results


def run_absorption_grating_talbot():
    results = []
    grating = make_amp_grating(NY, NX, DX, P)
    duty = float(np.mean(np.isclose(grating, 1.0 + 0.0j)))
    add_result(
        results,
        "3 geometry absorption grating duty cycle",
        duty,
        0.5,
        1e-12,
        abs(duty - 0.5) < 1e-12,
    )

    i0 = np.abs(grating) ** 2
    profile0 = i0.mean(axis=0)
    izt_asm = np.abs(propagate(grating, DX, WAVELENGTH, ZT, kernel="asm")) ** 2
    ncc_zt = pearson(izt_asm.mean(axis=0), profile0)
    izt_fresnel = np.abs(propagate(grating, DX, WAVELENGTH, ZT, kernel="fresnel")) ** 2
    max_diff_fresnel = float(np.max(np.abs(izt_fresnel - i0)))
    add_result(
        results,
        "3a absorption grating integer Talbot image",
        {"ncc_asm": ncc_zt, "max_diff_fresnel": max_diff_fresnel},
        {"ncc_asm": ">= 0.999", "max_diff_fresnel": "<= 1e-8"},
        None,
        ncc_zt >= 0.999 and max_diff_fresnel <= 1e-8,
    )

    izh = np.abs(propagate(grating, DX, WAVELENGTH, ZT / 2.0, kernel="asm")) ** 2
    samples_per_period = int(round(P / DX))
    shifted = np.roll(profile0, samples_per_period // 2)
    ncc_half = pearson(izh.mean(axis=0), shifted)
    add_result(
        results,
        "3b absorption grating half-Talbot shift",
        ncc_half,
        ">= 0.999",
        None,
        ncc_half >= 0.999,
    )

    z_values = [0.08, 0.16, 0.32, 0.48, 0.64]
    measured = []
    expected = []
    max_abs_error = 0.0
    for z in z_values:
        v = fourier_visibility(np.abs(propagate(grating, DX, WAVELENGTH, z, kernel="asm")) ** 2, DX, 1.0 / P)
        ref = FOUR_OVER_PI * abs(math.cos(math.pi * WAVELENGTH * z / (P * P)))
        measured.append(v)
        expected.append(ref)
        max_abs_error = max(max_abs_error, abs(v - ref))
    add_result(
        results,
        "3c absorption grating visibility curve",
        {"max_abs_error": max_abs_error, "values": measured},
        {"values": expected},
        0.01,
        max_abs_error <= 0.01,
    )
    return results


def run_sampling_convergence():
    results = []
    values = []
    errors = []
    for nx in (512, 1024, 4096):
        dx = LX / nx
        grating = make_pi_grating(NY, nx, dx, P)
        intensity = np.abs(propagate(grating, dx, WAVELENGTH, Z1, kernel="asm")) ** 2
        v = fourier_visibility(intensity, dx, 2.0 / P)
        values.append(v)
        errors.append(abs(v - FOUR_OVER_PI))
    monotonic = errors[0] >= errors[1] >= errors[2]
    rel_1024_4096 = abs(values[1] - values[2]) / values[2]
    add_result(
        results,
        "4 sampling convergence",
        {"values": values, "errors": errors, "relative_1024_4096": rel_1024_4096},
        {"monotonic_error": True, "relative_1024_4096": "<= 0.005"},
        None,
        monotonic and rel_1024_4096 <= 0.005,
    )
    return results


def _assert_geometry():
    assert math.isclose(P * P / WAVELENGTH, 0.64, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(Z1, 0.080, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(Z2, 0.160, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(ZT, 1.28, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(DX, 15.625e-9, rel_tol=0.0, abs_tol=1e-18)


def pearson(a, b):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def print_report(results):
    print("\nPropagation validation report")
    print("-" * 104)
    print(f"{'status':<8} {'test':<46} {'measured':<28} {'expected/tolerance'}")
    print("-" * 104)
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        measured = _short(item["measured"])
        expected = _short(item["expected"])
        tolerance = _short(item["tolerance"])
        print(f"{status:<8} {item['name']:<46} {measured:<28} {expected} tol={tolerance}")
    print("-" * 104)


def _short(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 27 else text[:24] + "..."


def write_json_report(results):
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "test_propagate_report.json").open("w", encoding="utf-8") as fh:
        json.dump({"all_passed": all(item["passed"] for item in results), "tests": results}, fh, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
