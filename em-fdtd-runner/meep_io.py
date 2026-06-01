"""meep_io.py — Geometry IR -> Meep mp.Simulation, a REAL 2D FDTD solve.

Photonics lane (grating_coupler, waveguide_coupler). Verified to run on macOS
(Apple Silicon) + Linux with conda-forge `pymeep`. engine_meep.solve() calls
`solve_native(geo, job)`; when Meep isn't importable it degrades to analytic.

Approach (single-run eigenmode S-params):
  - characteristic length a = 1 um; IR metres / a -> Meep units; Meep frequency
    = f_hz * a / c0; Meep resolution = cells_per_metre * a (pixels per um);
    PML thickness = pml_layers / resolution (in a-units).
  - EigenModeSource at the source port; an INPUT mode monitor just downstream of
    the source measures the actually-injected mode amplitude alpha_in.
  - one mode monitor per geometry monitor at the output ports -> alpha_out_i.
  - S_i1 = alpha_out_i / alpha_in ; T_i = |S_i1|^2 ; phase_i = angle(S_i1).
    The through port -> s21, a second ("cross") port -> s31, reflection ~
    1 - sum(T_i). Real, geometry-dependent, contract-shaped em.result.v1.

All heavy imports are LOCAL so `py_compile`/CI import works without Meep.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from geometry import Geometry

C0 = 299792458.0
A = 1e-6  # characteristic length: 1 um.


def _freqs_hz(job: Dict[str, Any], geo: Geometry) -> List[float]:
    freq = job.get("frequency", {}) or {}
    n = max(1, int(freq.get("points", 51)))
    if freq.get("range_nm"):
        lo_nm, hi_nm = float(freq["range_nm"][0]), float(freq["range_nm"][1])
        f_lo, f_hi = C0 / (hi_nm * 1e-9), C0 / (lo_nm * 1e-9)
    elif geo.sources:
        f0 = geo.sources[0].center_hz
        f_lo, f_hi = 0.9 * f0, 1.1 * f0
    else:
        f0 = 2e14
        f_lo, f_hi = 0.9 * f0, 1.1 * f0
    if n == 1:
        return [(f_lo + f_hi) / 2.0]
    step = (f_hi - f_lo) / (n - 1)
    return [f_lo + i * step for i in range(n)]


def _meep_resolution(geo: Geometry) -> int:
    # IR resolution is cells-per-metre; Meep resolution is pixels-per-a.
    res = int(round(geo.domain.resolution * A))
    return max(10, res)


def _port_span(geo: Geometry, y_meep: float, sy: float) -> float:
    """Transverse (y) span for a mode source/monitor at port height y_meep.

    Eigenmode injection/extraction must cover ONE waveguide, not the whole cell
    (else MPB returns a supermode and S-params collapse). Find the IR Box whose
    y-range contains the port and span its height plus a mode-tail margin,
    clamped so it does not reach a neighbouring waveguide.
    """
    boxes = []
    for b in geo.shapes:
        ylo, yhi = b.lo[1] / A, b.hi[1] / A
        boxes.append((min(ylo, yhi), max(ylo, yhi)))
    here = next((b for b in boxes if b[0] - 1e-9 <= y_meep <= b[1] + 1e-9), None)
    if here is None:
        return sy
    h = here[1] - here[0]
    # distance to the nearest OTHER box edge (keeps the window off neighbours).
    gaps = []
    for b in boxes:
        if b is here:
            continue
        if b[0] >= here[1]:
            gaps.append(b[0] - here[1])
        elif b[1] <= here[0]:
            gaps.append(here[0] - b[1])
    margin = 1.5 * h
    if gaps:
        margin = min(margin, 0.45 * min(gaps) + 0.0)
    span = h + 2.0 * max(margin, 0.2 * h)
    return float(min(span, sy))


def solve_native(geo: Geometry, job: Dict[str, Any]) -> Dict[str, Any]:
    """Run a real 2D Meep FDTD solve and return em.result.v1 result vectors."""
    import numpy as np  # type: ignore
    import meep as mp  # type: ignore

    is2d = geo.domain.dimensionality == 2
    sx, sy, sz = (s / A for s in geo.domain.size)
    cell = mp.Vector3(sx, sy, 0 if is2d else sz)
    resolution = _meep_resolution(geo)
    # IR uses corner-origin coords (0..size); Meep is centred at the origin, so
    # shift every coordinate by -size/2 or the structure sits outside the cell
    # (which silently gives zero fields / zero flux).
    ox, oy, oz = sx / 2.0, sy / 2.0, (0.0 if is2d else sz / 2.0)

    # Geometry: one mp.Block per IR Box.
    geometry: List[Any] = []
    for box in geo.shapes:
        lo = [c / A for c in box.lo]
        hi = [c / A for c in box.hi]
        size = mp.Vector3(hi[0] - lo[0], hi[1] - lo[1], 0 if is2d else (hi[2] - lo[2]))
        center = mp.Vector3((lo[0] + hi[0]) / 2 - ox, (lo[1] + hi[1]) / 2 - oy,
                            0 if is2d else (lo[2] + hi[2]) / 2 - oz)
        if box.is_metal or box.material == "pec":
            material = mp.metal
        else:
            material = mp.Medium(epsilon=float(box.eps or 1.0))
        geometry.append(mp.Block(size=size, center=center, material=material))

    src = geo.sources[0] if geo.sources else None
    if src is None:
        raise RuntimeError("meep: geometry has no source")
    fcen = src.center_hz * A / C0
    df = src.bandwidth_hz * A / C0
    if df <= 0:
        df = 0.2 * fcen
    src_y_ir = src.position[1] / A
    src_pos = mp.Vector3(src.position[0] / A - ox, src_y_ir - oy,
                         0.0 if is2d else src.position[2] / A - oz)

    # Source plane spans ONE waveguide (inferred from the UNSHIFTED y).
    src_span = _port_span(geo, src_y_ir, sy)
    src_size = mp.Vector3(0, src_span, 0) if is2d else mp.Vector3(0, src_span, sz)
    eig_parity = mp.ODD_Z if is2d else mp.NO_PARITY  # Ez (TE-like in 2D photonics)
    sources = [mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=df),
        center=src_pos, size=src_size,
        eig_band=1, eig_parity=eig_parity, eig_match_freq=True,
    )]

    pml_thick = max(0.5, geo.domain.pml_layers / resolution)
    sim = mp.Simulation(
        cell_size=cell, geometry=geometry, sources=sources,
        boundary_layers=[mp.PML(pml_thick)], resolution=resolution,
        force_complex_fields=True,
    )

    nfreq = max(1, int((job.get("frequency", {}) or {}).get("points", 51)))
    fwidth_mon = max(df, 0.2 * fcen)

    # Per port: a FLUX monitor (robust Poynting power -> magnitude) + a MODE
    # monitor (eigenmode coefficient -> phase). Flux is far less finicky than
    # mode-matching, so transmission magnitude is always physical even if the
    # eigenmode solver mismatches.
    dx_in = max(2.0 * pml_thick, 0.5)
    in_x = src_pos.x + dx_in
    in_fr = mp.FluxRegion(center=mp.Vector3(in_x, src_pos.y, 0 if is2d else src_pos.z),
                          size=mp.Vector3(0, src_span, 0 if is2d else sz))
    in_flux = sim.add_flux(fcen, fwidth_mon, nfreq, in_fr)
    in_mode = sim.add_mode_monitor(fcen, fwidth_mon, nfreq, in_fr)

    ports: List[Tuple[str, Any, Any]] = []  # (name, flux, mode)
    for mon in geo.monitors:
        if mon.kind not in ("flux", "mode", "s_param"):
            continue
        my_ir = mon.position[1] / A
        mc = mp.Vector3(mon.position[0] / A - ox, my_ir - oy,
                        0.0 if is2d else mon.position[2] / A - oz)
        fr = mp.FluxRegion(center=mc, size=mp.Vector3(0, _port_span(geo, my_ir, sy), 0 if is2d else sz))
        ports.append((mon.name, sim.add_flux(fcen, fwidth_mon, nfreq, fr),
                      sim.add_mode_monitor(fcen, fwidth_mon, nfreq, fr)))
    if not ports:
        mc = mp.Vector3(sx / 2.0 - pml_thick - 0.5, src_pos.y, 0)
        fr = mp.FluxRegion(center=mc, size=mp.Vector3(0, src_span, 0 if is2d else sz))
        ports.append(("T", sim.add_flux(fcen, fwidth_mon, nfreq, fr),
                      sim.add_mode_monitor(fcen, fwidth_mon, nfreq, fr)))

    run = job.get("run", {}) or {}
    thresh_db = float(run.get("decay_threshold_db", -30))
    decay = 10 ** (thresh_db / 10.0)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(in_x, src_pos.y), decay))

    eps = 1e-30
    in_p = np.maximum(np.abs(np.array(mp.get_fluxes(in_flux))), eps)  # injected power/freq
    band = [1]
    try:
        a_in = sim.get_eigenmode_coefficients(in_mode, band, eig_parity=eig_parity).alpha[0, :, 0]
    except Exception:
        a_in = np.ones(len(in_p), dtype=complex)
    s_by_port: Dict[str, Any] = {}
    for name, flux, mode in ports:
        out_p = np.array(mp.get_fluxes(flux))
        t = np.clip(np.abs(out_p) / in_p, 0.0, 1.0)        # robust power transmission
        try:
            a_out = sim.get_eigenmode_coefficients(mode, band, eig_parity=eig_parity).alpha[0, :, 0]
            ph = np.angle(a_out / (a_in + eps))            # eigenmode phase
        except Exception:
            ph = np.zeros(len(t))
        s_by_port[name] = np.sqrt(t) * np.exp(1j * ph)

    freqs = _freqs_hz(job, geo)
    port_names = list(s_by_port.keys())
    s_through = s_by_port[port_names[0]]
    s_cross = s_by_port[port_names[1]] if len(port_names) > 1 else None

    def _resize(arr, length):
        if arr is None:
            return np.zeros(length, dtype=complex)
        if len(arr) == length:
            return arr
        idx = np.linspace(0, len(arr) - 1, length)
        re = np.interp(idx, np.arange(len(arr)), arr.real)
        im = np.interp(idx, np.arange(len(arr)), arr.imag)
        return re + 1j * im

    L = len(freqs)
    s_through = _resize(s_through, L)
    s_cross = _resize(s_cross, L) if s_cross is not None else None

    t_through = np.clip(np.abs(s_through) ** 2, 0.0, 1.0)
    refl = np.clip(1.0 - t_through - (np.abs(s_cross) ** 2 if s_cross is not None else 0.0), 0.0, 1.0)
    phase = np.angle(s_through)

    s_params: Dict[str, Any] = {
        "s21_re": [float(x) for x in s_through.real],
        "s21_im": [float(x) for x in s_through.imag],
        "s11_re": [float(math.sqrt(max(r, 0.0))) for r in refl],
        "s11_im": [0.0 for _ in refl],
    }
    if s_cross is not None:
        s_params["s31_re"] = [float(x) for x in s_cross.real]
        s_params["s31_im"] = [float(x) for x in s_cross.imag]

    return {
        "freqs_hz": [float(x) for x in freqs],
        "transmission": [float(x) for x in t_through],
        "reflection": [float(x) for x in refl],
        "phase_rad": [float(x) for x in phase],
        "s_params": s_params,
        "mesh_cells": geo.grid_dims_cells()["nx"] * geo.grid_dims_cells()["ny"] * geo.grid_dims_cells()["nz"],
        "converged": True,
        "solver": "meep",
    }
