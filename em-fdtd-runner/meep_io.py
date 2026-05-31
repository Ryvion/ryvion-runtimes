"""meep_io.py — Geometry IR -> Meep mp.Simulation translation skeleton.

// TODO(em-verify): Meep is not installed in CI/dev, so none of this is
exercised here. It is a structural skeleton parallel to gprmax_io.py showing how
the engine-neutral Geometry IR maps onto a Meep simulation (mp.Block per Box,
GaussianSource, FluxRegion / mode monitors) and how get_eigenmode_coefficients
gives transmission / phase / s_params. engine_meep.py only calls these when
`import meep` succeeds (Linux OCI image); otherwise it degrades to analytic.

Meep API references (for the implementer finishing on a Linux box):
  - mp.Simulation(cell_size, geometry, sources, boundary_layers, resolution)
  - mp.Block(size, center, material=mp.Medium(epsilon=...))
  - mp.GaussianSource(frequency, fwidth), mp.EigenModeSource for waveguides.
  - sim.add_mode_monitor / sim.add_flux ; sim.run(until_after_sources=stop_when_decayed)
  - sim.get_eigenmode_coefficients(mon, [1]).alpha -> complex mode amplitudes;
    transmission = |alpha_out|^2 / |alpha_in|^2 ; phase = angle(alpha_out).

All heavy imports are LOCAL to the functions so importing this module in CI
(py_compile) does not require Meep.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from geometry import Box, Geometry

C0 = 299792458.0


def _meep_freqs(job: Dict[str, Any], geo: Geometry):
    """Meep frequency units are 1/a with a = 1 um; we work in physical Hz here
    and let the caller scale by the chosen characteristic length a.
    // TODO(em-verify): pin the unit scale (a) consistently with build_simulation.
    """
    freq = job.get("frequency", {}) or {}
    n = max(1, int(freq.get("points", 51)))
    if "range_nm" in freq and freq["range_nm"]:
        lo_nm, hi_nm = float(freq["range_nm"][0]), float(freq["range_nm"][1])
        f_lo, f_hi = C0 / (hi_nm * 1e-9), C0 / (lo_nm * 1e-9)
    else:
        f0 = geo.sources[0].center_hz if geo.sources else 2e14
        f_lo, f_hi = 0.9 * f0, 1.1 * f0
    if n == 1:
        return [(f_lo + f_hi) / 2.0]
    step = (f_hi - f_lo) / (n - 1)
    return [f_lo + i * step for i in range(n)]


def build_simulation(geo: Geometry, job: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Translate the IR into an mp.Simulation + a monitor table.

    // TODO(em-verify): finish + verify on a Linux box with Meep installed.
    The characteristic length a (Meep's unit) is taken as 1 um; all IR meter
    coordinates are divided by a to get Meep units.
    """
    import meep as mp  # type: ignore

    a = 1e-6  # 1 um characteristic length (photonics).
    sx, sy, sz = (s / a for s in geo.domain.size)
    is2d = geo.domain.dimensionality == 2
    cell = mp.Vector3(sx, sy, 0 if is2d else sz)

    geometry: List[Any] = []
    for box in geo.shapes:
        lo = [c / a for c in box.lo]
        hi = [c / a for c in box.hi]
        size = mp.Vector3(hi[0] - lo[0], hi[1] - lo[1], 0 if is2d else (hi[2] - lo[2]))
        center = mp.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                            0 if is2d else (lo[2] + hi[2]) / 2)
        if box.is_metal or box.material == "pec":
            material = mp.metal
        else:
            material = mp.Medium(epsilon=float(box.eps or 1.0))
        geometry.append(mp.Block(size=size, center=center, material=material))

    src = geo.sources[0] if geo.sources else None
    sources: List[Any] = []
    if src is not None:
        fcen = src.center_hz * a / C0  # -> Meep frequency units (a/lambda).
        df = max(src.bandwidth_hz * a / C0, 0.1 * fcen)
        comp = {"te": mp.Ez, "tm": mp.Hz}.get(src.polarization.lower(), mp.Ez)
        sources.append(mp.Source(mp.GaussianSource(fcen, fwidth=df), component=comp,
                                 center=mp.Vector3(*[c / a for c in src.position][:3])))

    pml = [mp.PML(geo.domain.pml_layers / (geo.domain.resolution * a))]
    resolution = max(10, int(geo.domain.resolution * a))

    sim = mp.Simulation(cell_size=cell, geometry=geometry, sources=sources,
                        boundary_layers=pml, resolution=resolution)

    monitors: Dict[str, Any] = {}
    for mon in geo.monitors:
        if mon.kind in ("flux", "mode"):
            fcen = (src.center_hz * a / C0) if src else 0.2
            df = (max(src.bandwidth_hz, src.center_hz * 0.2) * a / C0) if src else 0.1
            region = mp.FluxRegion(center=mp.Vector3(*[c / a for c in mon.position][:3]))
            monitors[mon.name] = sim.add_mode_monitor(fcen, df, max(1, int((job.get("frequency", {}) or {}).get("points", 51))), region)
    return sim, monitors


def run_until_decay(sim: Any, geo: Geometry, job: Dict[str, Any]) -> None:
    """Run until the fields decay below the configured threshold.

    // TODO(em-verify): finish on a Linux box. Uses Meep's stop_when_fields_decayed.
    """
    import meep as mp  # type: ignore

    run = job.get("run", {}) or {}
    thresh_db = float(run.get("decay_threshold_db", -40))
    decay = 10 ** (thresh_db / 10.0)
    src = geo.sources[0] if geo.sources else None
    comp = mp.Ez
    where = mp.Vector3()
    if geo.monitors:
        where = mp.Vector3(*[c / 1e-6 for c in geo.monitors[0].position][:3])
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, comp, where, decay))


def extract(sim: Any, monitors: Dict[str, Any], job: Dict[str, Any], geo: Geometry) -> Dict[str, Any]:
    """Read mode coefficients into em.result.v1 vectors.

    // TODO(em-verify): finish on a Linux box. transmission = |alpha_out|^2 /
    |alpha_in|^2 ; phase = angle(alpha_out / alpha_in).
    """
    import numpy as np  # type: ignore

    freqs = _meep_freqs(job, geo)
    # Pick the input mode monitor and the output ("T") monitor by name.
    in_name = next(iter(monitors), None)
    out_name = "T" if "T" in monitors else in_name
    if in_name is None or out_name is None:
        raise RuntimeError("meep extract: no mode monitors")

    in_coeffs = sim.get_eigenmode_coefficients(monitors[in_name], [1]).alpha[0, :, 0]
    out_coeffs = sim.get_eigenmode_coefficients(monitors[out_name], [1]).alpha[0, :, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        s21 = out_coeffs / in_coeffs
    s21 = np.nan_to_num(s21, nan=0.0)
    trans = np.clip(np.abs(s21) ** 2, 0.0, 1.0)
    phase = np.angle(s21)

    return {
        "freqs_hz": [float(x) for x in freqs],
        "transmission": [float(x) for x in trans],
        "reflection": [float(1.0 - x) for x in trans],
        "phase_rad": [float(x) for x in phase],
        "s_params": {
            "s21_re": [float(x) for x in s21.real],
            "s21_im": [float(x) for x in s21.imag],
            "s11_re": [float(math.sqrt(max(1.0 - t, 0.0))) for t in trans],
            "s11_im": [0.0 for _ in trans],
        },
        "mesh_cells": geo.grid_dims_cells()["nx"] * geo.grid_dims_cells()["ny"] * geo.grid_dims_cells()["nz"],
        "converged": True,
        "solver": "meep",
    }
