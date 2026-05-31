"""engine_analytic.py — deterministic analytic fallback solver.

NOT a real FDTD solve. It produces physically-plausible, SEED-DETERMINISTIC
result vectors in the em.result.v1 shape so the runner always emits a valid
result contract for dev/CI and for bundles where the native engine library is
unavailable. Real engines (engine_gprmax/openems/meep) call into this only when
their heavy dependency cannot be imported. Marked with engine_version suffix
"+analytic" by the caller so QA / aggregation can tell it apart.

Determinism: vectors depend only on (job seed, params, frequency grid), never on
wall-clock or RNG without a fixed seed — required for reproducible QA.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from geometry import Geometry


def _freq_grid(job: Dict[str, Any], geo: Geometry) -> List[float]:
    freq = job.get("frequency", {}) or {}
    n = int(freq.get("points", 51))
    n = max(1, n)
    c = 299792458.0
    if "range_nm" in freq and freq["range_nm"]:
        lo_nm, hi_nm = float(freq["range_nm"][0]), float(freq["range_nm"][1])
        # convert wavelength endpoints to frequency, ascending.
        f_lo = c / (hi_nm * 1e-9)
        f_hi = c / (lo_nm * 1e-9)
    elif "range_ghz" in freq and freq["range_ghz"]:
        f_lo = float(freq["range_ghz"][0]) * 1e9
        f_hi = float(freq["range_ghz"][1]) * 1e9
    else:
        # single-frequency source +/- 10%.
        f0 = geo.sources[0].center_hz if geo.sources else 1.0e14
        f_lo, f_hi = 0.9 * f0, 1.1 * f0
    if n == 1:
        return [(f_lo + f_hi) / 2.0]
    step = (f_hi - f_lo) / (n - 1)
    return [f_lo + i * step for i in range(n)]


def _seed_phase(job: Dict[str, Any]) -> float:
    seed = job.get("seed")
    if seed is None:
        return 0.0
    # bounded deterministic phase offset in [0, 2pi)
    return (int(seed) % 1000) / 1000.0 * 2 * math.pi


def solve(geo: Geometry, job: Dict[str, Any]) -> Dict[str, Any]:
    freqs = _freq_grid(job, geo)
    f0 = geo.sources[0].center_hz if geo.sources else (sum(freqs) / len(freqs))
    bw = geo.sources[0].bandwidth_hz if geo.sources else (f0 * 0.1)
    bw = max(bw, 1.0)
    phase_off = _seed_phase(job)

    transmission: List[float] = []
    reflection: List[float] = []
    phase_rad: List[float] = []
    s11_re: List[float] = []
    s11_im: List[float] = []
    s21_re: List[float] = []
    s21_im: List[float] = []
    s31_re: List[float] = []
    s31_im: List[float] = []

    for f in freqs:
        # Lorentzian transmission resonance centered at f0.
        x = (f - f0) / bw
        t = 1.0 / (1.0 + 4.0 * x * x)
        r = 1.0 - t
        ph = math.atan2(2.0 * x, 1.0) + phase_off
        transmission.append(t)
        reflection.append(r)
        phase_rad.append(((ph + math.pi) % (2 * math.pi)) - math.pi)
        mag_s21 = math.sqrt(max(t, 0.0))
        mag_s11 = math.sqrt(max(r, 0.0))
        s21_re.append(mag_s21 * math.cos(ph))
        s21_im.append(mag_s21 * math.sin(ph))
        s11_re.append(mag_s11 * math.cos(ph))
        s11_im.append(mag_s11 * math.sin(ph))
        # cross port (used by waveguide_coupler) = complement of through.
        s31_re.append(math.sqrt(max(1.0 - t, 0.0)) * math.cos(ph))
        s31_im.append(math.sqrt(max(1.0 - t, 0.0)) * math.sin(ph))

    dims = geo.grid_dims_cells()
    cells = dims["nx"] * dims["ny"] * dims["nz"]

    result: Dict[str, Any] = {
        "freqs_hz": freqs,
        "transmission": transmission,
        "reflection": reflection,
        "phase_rad": phase_rad,
        "s_params": {
            "s11_re": s11_re, "s11_im": s11_im,
            "s21_re": s21_re, "s21_im": s21_im,
            "s31_re": s31_re, "s31_im": s31_im,
        },
        "mesh_cells": cells,
        "converged": True,
        # marks this as the analytic fallback, NOT a real FDTD solve. run.py
        # appends "+analytic" to engine_version so QA/aggregation can tell the
        # difference (architecture doc §3.2 determinism note).
        "solver": "analytic",
    }

    # Radiation pattern only for templates that asked for it (antenna/metasurface).
    wants = (job.get("outputs", {}) or {}).get("want", []) or []
    has_rad_monitor = any(m.kind == "radiation" for m in geo.monitors)
    if "radiation_pattern" in wants or has_rad_monitor:
        thetas = [i * (math.pi / 18) for i in range(19)]   # 0..180 deg, 10 deg steps
        phis = [0.0, math.pi / 2]
        gain_db: List[float] = []
        for th in thetas:
            # simple broadside cos pattern, deterministic.
            g = max(math.cos(th), 1e-3)
            gain_db.append(10.0 * math.log10(g))
        result["radiation_pattern"] = {
            "theta": thetas,
            "phi": phis,
            "gain_db": gain_db,
        }

    return result
