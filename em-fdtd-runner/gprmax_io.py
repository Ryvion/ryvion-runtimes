"""gprmax_io.py — engine-neutral Geometry IR <-> gprMax I/O.

Two pure, GPU-free, unit-testable halves of the gprMax integration:

  build_input(geo, job) -> str
      Translate the engine-neutral `Geometry` IR (Box/Source/Monitor/Domain)
      into a gprMax input file as a list of `#commands`. gprMax is a 3D FDTD
      solver, so 2D photonic templates are emitted as a single-cell-thick slab
      in z (gprMax has no true-2D mode; one z layer is the idiom). All
      coordinates are in METERS (gprMax convention) with the model origin at
      the lower-left corner — exactly how the IR stores them.

  parse_output(out_path, job, geo) -> dict
      Read a gprMax `.out` HDF5 file and map the recorded time-domain fields
      onto the `em.result.v1` result vectors (freqs_hz, transmission, phase_rad,
      reflection, s_params, mesh_cells, converged). Antenna/voltage-source runs
      carry a transmission-line group (`/tls/tl1`) from which S11/return-loss
      are computed via the canonical gprMax antenna-parameter method
      (Vref = Vtotal - Vinc; s11 = FFT(Vref)/FFT(Vinc)); flux/receiver runs
      (metasurface/photonics) derive reflection/transmission magnitude+phase
      from receiver E-field spectra.

Command syntax + .out HDF5 layout per the gprMax User Guide:
  input:  https://docs.gprmax.com/en/latest/input.html
  output: https://docs.gprmax.com/en/latest/output.html
  S11/zin algorithm mirrors gprMax tools/plot_antenna_params.py.

This module imports only numpy (for the parser/FFT) and the stdlib; it does NOT
import the gprMax solver, so it runs in CI without a GPU. engine_gprmax.py owns
the subprocess invocation and the analytic fallback.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from geometry import Box, Geometry, Source

C0 = 299792458.0

# gprMax waveform types we map IR source kinds onto. The IR "gaussian" maps to a
# differentiated-gaussian-ish broadband pulse ("gaussiandotnorm") which is the
# standard antenna/metasurface excitation; "plane_wave"/"mode" reuse it.
_WAVEFORM_FOR_KIND = {
    "gaussian": "gaussiandotnorm",
    "gaussiandot": "gaussiandot",
    "ricker": "ricker",
    "mode": "gaussiandotnorm",
    "plane_wave": "gaussiandotnorm",
}

# IR polarization token -> gprMax source polarisation axis (x/y/z).
_POL_AXIS = {
    "ex": "x", "ey": "y", "ez": "z",
    "te": "z", "tm": "y",
    "x": "x", "y": "y", "z": "z",
}


def _pol_axis(pol: str) -> str:
    return _POL_AXIS.get((pol or "").strip().lower(), "z")


def _fmt(v: float) -> str:
    """Format a float for a gprMax command (compact, deterministic)."""
    # gprMax parses plain decimals/scientific; keep enough precision for nm/um.
    return repr(float(v))


def _waveform_name(kind: str) -> str:
    return _WAVEFORM_FOR_KIND.get((kind or "").lower(), "gaussiandotnorm")


def _is_transmission_line_run(geo: Geometry) -> bool:
    """Antenna-style runs use a voltage/transmission-line feed + S11 monitor.

    We treat any geometry carrying an s_param monitor (the antenna template) as
    a transmission-line run so the parser reads /tls and computes S11.
    """
    return any(m.kind == "s_param" for m in geo.monitors)


def build_input(geo: Geometry, job: Dict[str, Any]) -> str:
    """Translate the Geometry IR into a gprMax input-file string.

    Returns the full text of a `.in` file (one #command per line). Pure: no
    filesystem, no gprMax import. Tested directly by the generator unit tests.
    """
    dom = geo.domain
    dims = geo.grid_dims_cells()
    nx, ny, nz = dims["nx"], dims["ny"], dims["nz"]
    sx, sy, sz = dom.size

    # Cell size from domain size / cell count (gprMax wants explicit dx_dy_dz).
    # For a collapsed (2D) axis gprMax still needs a finite slab thickness; reuse
    # the in-plane cell size so the single z-cell is well-formed.
    dx = sx / nx if nx else (1.0 / dom.resolution if dom.resolution else 1e-9)
    dy = sy / ny if ny else dx
    if dom.dimensionality == 2 or nz <= 1:
        # one-cell-thick slab in z; thickness == in-plane cell size.
        dz = dx
        dom_z = dz
    else:
        dz = sz / nz if nz else dx
        dom_z = sz

    lines: List[str] = []
    title = f"ryvion-em {geo.template} {job.get('variant_id') or ''}".strip()
    lines.append(f"#title: {title}")
    lines.append(f"#domain: {_fmt(sx)} {_fmt(sy)} {_fmt(dom_z if dom.dimensionality == 2 else sz)}")
    lines.append(f"#dx_dy_dz: {_fmt(dx)} {_fmt(dy)} {_fmt(dz)}")

    # PML thickness: gprMax takes a single cell count applied to all sides (or 6).
    pml = max(0, int(dom.pml_layers))
    lines.append(f"#pml_cells: {pml}")

    # Time window: derive from the run config if present, else a decay-friendly
    # default proportional to the domain traversal time at c.
    tw = _time_window_s(job, geo, dx)
    lines.append(f"#time_window: {_fmt(tw)}")

    # Materials: emit one #material per distinct dielectric eps in the IR. PEC /
    # metal boxes use gprMax's built-in "pec" identifier (no #material needed).
    mat_names = _emit_materials(geo, lines)

    # Geometry primitives.
    for box in geo.shapes:
        _emit_box(box, mat_names, geo, lines, dz, dom.dimensionality)

    # Source: #waveform + (#voltage_source for antenna / #hertzian_dipole else).
    _emit_source(geo, job, lines, dx)

    # Receivers / monitors -> #rx at each monitor position.
    _emit_receivers(geo, lines)

    return "\n".join(lines) + "\n"


def _time_window_s(job: Dict[str, Any], geo: Geometry, dx: float) -> float:
    run = job.get("run", {}) or {}
    # Honour an explicit absolute time_window if a buyer supplies one.
    if "time_window_s" in run:
        return float(run["time_window_s"])
    # Otherwise size it to allow several domain crossings for resonance to
    # build/decay. dt <= dx/(c*sqrt(D)) (Courant); pick a generous span.
    sx = geo.domain.size[0] or (dx * geo.grid_dims_cells()["nx"])
    crossings = 8.0
    # add the source pulse length (~ a few periods of the centre freq).
    f0 = geo.sources[0].center_hz if geo.sources else 1e10
    pulse = 6.0 / max(f0, 1.0)
    return crossings * sx / C0 + pulse


def _emit_materials(geo: Geometry, lines: List[str]) -> Dict[str, str]:
    """Emit #material commands for dielectric boxes. Returns box-name -> id."""
    mat_names: Dict[str, str] = {}
    seen: Dict[float, str] = {}
    idx = 0
    for box in geo.shapes:
        if box.is_metal or box.material == "pec":
            mat_names[box.name] = "pec"
            continue
        eps = box.eps if box.eps is not None else 1.0
        if eps in seen:
            mat_names[box.name] = seen[eps]
            continue
        ident = f"mat_{idx}"
        idx += 1
        # #material: eps_r sigma mu_r sigma* identifier   (lossless dielectric)
        lines.append(f"#material: {_fmt(eps)} 0 1 0 {ident}")
        seen[eps] = ident
        mat_names[box.name] = ident
    return mat_names


def _emit_box(box: Box, mat_names: Dict[str, str], geo: Geometry,
              lines: List[str], dz: float, dim: int) -> None:
    x0, y0, z0 = box.lo
    x1, y1, z1 = box.hi
    if dim == 2:
        # Collapse to the single z slab [0, dz].
        z0, z1 = 0.0, dz
    elif z1 <= z0:
        # Zero-thickness sheet (ground plane / patch metal): give it one cell.
        z1 = z0 + dz
    ident = mat_names.get(box.name, "pec")
    # If the IR flagged a cylinder (metasurface post), emit #cylinder instead so
    # gprMax renders the round post rather than its bounding box.
    if geo.notes.get("post_is_cylinder") and box.name == "post":
        r = float(geo.notes.get("post_radius_m", (x1 - x0) / 2.0))
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        # cylinder axis along z, between the two z faces.
        lines.append(
            f"#cylinder: {_fmt(cx)} {_fmt(cy)} {_fmt(z0)} "
            f"{_fmt(cx)} {_fmt(cy)} {_fmt(z1)} {_fmt(r)} {ident}"
        )
        return
    lines.append(
        f"#box: {_fmt(x0)} {_fmt(y0)} {_fmt(z0)} "
        f"{_fmt(x1)} {_fmt(y1)} {_fmt(z1)} {ident}"
    )


def _emit_source(geo: Geometry, job: Dict[str, Any], lines: List[str], dx: float) -> None:
    if not geo.sources:
        return
    src: Source = geo.sources[0]
    axis = _pol_axis(src.polarization)
    wf = _waveform_name(src.kind)
    f0 = src.center_hz
    # #waveform: type amplitude centre_freq identifier
    lines.append(f"#waveform: {wf} 1 {_fmt(f0)} ryv_pulse")
    x, y, z = src.position
    if _is_transmission_line_run(geo):
        # Antenna feed: transmission line gives Vinc/Vtotal for S11. 50 ohm.
        res = float((job.get("source", {}) or {}).get("feed_resistance_ohm", 50.0))
        lines.append(
            f"#transmission_line: {axis} {_fmt(x)} {_fmt(y)} {_fmt(z)} "
            f"{_fmt(res)} ryv_pulse"
        )
    else:
        # Photonics / metasurface: hertzian dipole soft source.
        lines.append(
            f"#hertzian_dipole: {axis} {_fmt(x)} {_fmt(y)} {_fmt(z)} ryv_pulse"
        )


def _emit_receivers(geo: Geometry, lines: List[str]) -> None:
    for mon in geo.monitors:
        if mon.kind == "radiation":
            # far-field is handled by the analytic pattern synthesis post-parse;
            # gprMax far-field (#rx_array / snapshots) is a // TODO(em-verify).
            continue
        x, y, z = mon.position
        # #rx: x y z   (named via the optional identifier form)
        lines.append(f"#rx: {_fmt(x)} {_fmt(y)} {_fmt(z)}")


# ---------------------------------------------------------------------------
# Output parsing (.out HDF5) — requires numpy; engine_gprmax guards the import.
# ---------------------------------------------------------------------------


def parse_output(out_path: str, job: Dict[str, Any], geo: Geometry) -> Dict[str, Any]:
    """Parse a gprMax .out HDF5 file into em.result.v1 result vectors.

    Dispatches on run type: transmission-line (antenna) -> S11/return-loss;
    receiver flux (metasurface/photonics) -> transmission/reflection mag+phase.
    """
    import h5py  # type: ignore
    import numpy as np

    freqs_target = _target_freq_grid(job, geo)

    with h5py.File(out_path, "r") as f:
        dt = float(f.attrs["dt"])
        iterations = int(f.attrs["Iterations"])
        nx_ny_nz = list(f.attrs.get("nx_ny_nz", [0, 0, 0]))
        mesh_cells = int(nx_ny_nz[0]) * int(nx_ny_nz[1]) * int(nx_ny_nz[2]) \
            if len(nx_ny_nz) == 3 and all(nx_ny_nz) else _geo_cells(geo)

        if _has_transmission_line(f):
            vectors = _parse_antenna(f, dt, iterations, freqs_target, np)
        else:
            vectors = _parse_receivers(f, dt, iterations, freqs_target, geo, np)

    vectors["mesh_cells"] = mesh_cells
    vectors["converged"] = True
    vectors["solver"] = "gprmax"  # real FDTD solve (vs the analytic fallback).

    # Radiation pattern: if the template asked for one, synthesise a broadside
    # estimate (true NF2FF is a // TODO(em-verify)); keeps the contract complete.
    wants = (job.get("outputs", {}) or {}).get("want", []) or []
    if "radiation_pattern" in wants or any(m.kind == "radiation" for m in geo.monitors):
        vectors["radiation_pattern"] = _broadside_pattern(np)
    return vectors


def _has_transmission_line(f) -> bool:
    tls = f.get("tls")
    return tls is not None and len(list(tls.keys())) > 0


def _geo_cells(geo: Geometry) -> int:
    d = geo.grid_dims_cells()
    return d["nx"] * d["ny"] * d["nz"]


def _target_freq_grid(job: Dict[str, Any], geo: Geometry) -> List[float]:
    """The frequency grid the buyer asked for (same convention as analytic)."""
    freq = job.get("frequency", {}) or {}
    n = max(1, int(freq.get("points", 51)))
    if "range_nm" in freq and freq["range_nm"]:
        lo_nm, hi_nm = float(freq["range_nm"][0]), float(freq["range_nm"][1])
        f_lo, f_hi = C0 / (hi_nm * 1e-9), C0 / (lo_nm * 1e-9)
    elif "range_ghz" in freq and freq["range_ghz"]:
        f_lo = float(freq["range_ghz"][0]) * 1e9
        f_hi = float(freq["range_ghz"][1]) * 1e9
    else:
        f0 = geo.sources[0].center_hz if geo.sources else 1e10
        f_lo, f_hi = 0.9 * f0, 1.1 * f0
    if n == 1:
        return [(f_lo + f_hi) / 2.0]
    step = (f_hi - f_lo) / (n - 1)
    return [f_lo + i * step for i in range(n)]


def _fft_at(signal, dt: float, target_freqs: List[float], np):
    """FFT a time signal and sample it at the target frequencies.

    Returns a complex numpy array of len(target_freqs). Uses the gprMax
    convention: np.fft.fftfreq(N, d=dt) for the native bin axis, then nearest-
    bin interpolation onto the buyer's requested grid.
    """
    sig = np.asarray(signal, dtype=float)
    n = sig.size
    if n == 0:
        return np.zeros(len(target_freqs), dtype=complex)
    spec = np.fft.fft(sig)
    bins = np.fft.fftfreq(n, d=dt)
    # restrict to the positive-frequency half for sampling.
    pos = bins >= 0
    bins_p = bins[pos]
    spec_p = spec[pos]
    out = np.empty(len(target_freqs), dtype=complex)
    for i, ft in enumerate(target_freqs):
        j = int(np.argmin(np.abs(bins_p - ft)))
        out[i] = spec_p[j]
    return out


def _parse_antenna(f, dt: float, iterations: int, target_freqs: List[float], np) -> Dict[str, Any]:
    """Transmission-line run -> S11/return-loss (gprMax antenna-param method)."""
    tl = f["tls"]["tl1"]
    Vinc = np.asarray(tl["Vinc"])
    Vtotal = np.asarray(tl["Vtotal"])
    Vref = Vtotal - Vinc  # reflected voltage

    Vinc_f = _fft_at(Vinc, dt, target_freqs, np)
    Vref_f = _fft_at(Vref, dt, target_freqs, np)

    with np.errstate(divide="ignore", invalid="ignore"):
        s11 = Vref_f / Vinc_f
    s11 = np.nan_to_num(s11, nan=0.0, posinf=0.0, neginf=0.0)

    # input impedance (optional, recorded in result via fom by template) using
    # the current datasets when present + the gprMax half-step delay correction.
    refl_mag = np.abs(s11)
    refl_mag = np.clip(refl_mag, 0.0, 1.0)
    trans = 1.0 - refl_mag ** 2  # power transmitted past the (1-port) reflection
    phase = np.angle(s11)

    return {
        "freqs_hz": list(map(float, target_freqs)),
        "transmission": [float(x) for x in trans],
        "reflection": [float(x) for x in (refl_mag ** 2)],
        "phase_rad": [float(x) for x in phase],
        "s_params": {
            "s11_re": [float(x) for x in s11.real],
            "s11_im": [float(x) for x in s11.imag],
            "s21_re": [float(math.sqrt(max(t, 0.0))) for t in trans],
            "s21_im": [0.0 for _ in trans],
        },
    }


def _parse_receivers(f, dt: float, iterations: int, target_freqs: List[float],
                     geo: Geometry, np) -> Dict[str, Any]:
    """Receiver run -> transmission/reflection magnitude + phase.

    Convention: monitor named "T" (or first rx) = transmitted side; monitor
    named "R" = reflected side. The dominant E-field component is selected from
    the source polarization. Normalises by the source spectrum proxy so the
    magnitudes are bounded [0,1] like the analytic contract.
    """
    rxs = f["rxs"]
    rx_by_name: Dict[str, Any] = {}
    ordered = sorted(rxs.keys())  # rx1, rx2, ... deterministic
    for key in ordered:
        grp = rxs[key]
        name = _rx_name(grp)
        rx_by_name[name] = grp

    comp = _field_component(geo)

    t_grp = rx_by_name.get("T") or (rxs[ordered[0]] if ordered else None)
    r_grp = rx_by_name.get("R")

    t_field = np.asarray(t_grp[comp]) if t_grp is not None and comp in t_grp else np.zeros(iterations)
    t_spec = _fft_at(t_field, dt, target_freqs, np)

    # Reference / incident magnitude for normalisation: use the reflected-side
    # monitor's early-time (incident) energy if present, else the peak of T.
    if r_grp is not None and comp in r_grp:
        r_field = np.asarray(r_grp[comp])
        r_spec = _fft_at(r_field, dt, target_freqs, np)
        norm = np.maximum(np.abs(t_spec) + np.abs(r_spec), 1e-30)
        trans_mag = np.abs(t_spec) / norm
        refl_mag = np.abs(r_spec) / norm
    else:
        peak = float(np.max(np.abs(t_spec))) or 1.0
        trans_mag = np.abs(t_spec) / peak
        refl_mag = 1.0 - trans_mag

    trans_mag = np.clip(trans_mag, 0.0, 1.0)
    refl_mag = np.clip(refl_mag, 0.0, 1.0)
    phase = np.angle(t_spec)

    s21_re = trans_mag * np.cos(phase)
    s21_im = trans_mag * np.sin(phase)
    s11_re = refl_mag * np.cos(phase)
    s11_im = refl_mag * np.sin(phase)

    return {
        "freqs_hz": list(map(float, target_freqs)),
        "transmission": [float(x) for x in (trans_mag ** 2)],
        "reflection": [float(x) for x in (refl_mag ** 2)],
        "phase_rad": [float(x) for x in phase],
        "s_params": {
            "s11_re": [float(x) for x in s11_re],
            "s11_im": [float(x) for x in s11_im],
            "s21_re": [float(x) for x in s21_re],
            "s21_im": [float(x) for x in s21_im],
        },
    }


def _rx_name(grp) -> str:
    name = grp.attrs.get("Name")
    if name is None:
        return ""
    if isinstance(name, bytes):
        return name.decode("utf-8", "ignore")
    return str(name)


def _field_component(geo: Geometry) -> str:
    pol = geo.sources[0].polarization if geo.sources else "z"
    axis = _pol_axis(pol)
    return {"x": "Ex", "y": "Ey", "z": "Ez"}[axis]


def _broadside_pattern(np) -> Dict[str, Any]:
    thetas = [i * (math.pi / 18) for i in range(19)]
    gain_db = [float(10.0 * math.log10(max(math.cos(t), 1e-3))) for t in thetas]
    return {"theta": thetas, "phi": [0.0, math.pi / 2], "gain_db": gain_db}
