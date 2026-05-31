"""metasurface_unit_cell.py — 6G metasurface element (3D, periodic).

Params: element_size_um, post_h_um, post_r_um, lattice_period_um, freq_ghz
Output: reflection/transmission magnitude + phase (the phase-vs-geometry map
that surrogate-model buyers want).
"""
from __future__ import annotations

from typing import Any, Dict

from geometry import UM, Box, Domain, Geometry, Monitor, Source

DEFAULT_OUTPUTS = ["reflection", "transmission", "phase", "fom"]

_REQUIRED = ["post_h_um", "post_r_um", "lattice_period_um", "freq_ghz"]


def _validate(params: Dict[str, Any]) -> None:
    for k in _REQUIRED:
        if k not in params:
            raise ValueError(f"metasurface_unit_cell missing required param {k!r}")


def build(params: Dict[str, Any], job: Dict[str, Any]) -> Geometry:
    _validate(params)
    post_h = float(params["post_h_um"]) * UM
    post_r = float(params["post_r_um"]) * UM
    period = float(params["lattice_period_um"]) * UM
    f0 = float(params["freq_ghz"]) * 1e9

    materials = job.get("materials", {}) or {}
    post_eps = float(materials.get("post_eps", 11.7))  # silicon-like default

    grid = job.get("grid", {}) or {}
    res_per_um = float(grid.get("resolution", 20))
    res = res_per_um / UM
    pml = int(grid.get("pml_layers", 8))

    # Periodic unit cell: one lattice period in x,y; air padding + PML in z.
    pad = 4.0 * UM
    sx = period
    sy = period
    sz = post_h + 2 * pad
    dom = Domain(size=(sx, sy, sz), resolution=res, pml_layers=pml,
                 dimensionality=int(grid.get("dimensionality", 3)),
                 symmetry=grid.get("symmetry"))
    geo = Geometry(template="metasurface_unit_cell", domain=dom,
                   materials={"post_eps": post_eps})
    geo.notes["periodic_xy"] = True
    geo.notes["element_size_um"] = float(params.get("element_size_um", 2 * post_r / UM))

    # Dielectric post centered in the cell. Represented as a box bounding the
    # cylinder; the engine layer refines to a cylinder where supported.
    cx, cy = sx / 2.0, sy / 2.0
    z0 = pad
    geo.add_box(Box("post", (cx - post_r, cy - post_r, z0),
                    (cx + post_r, cy + post_r, z0 + post_h),
                    material="post", eps=post_eps))
    geo.notes["post_is_cylinder"] = True
    geo.notes["post_radius_m"] = post_r

    bw = float((job.get("source", {}) or {}).get("bandwidth", 0.2)) * f0
    geo.add_source(Source(kind="plane_wave", center_hz=f0, bandwidth_hz=bw,
                          polarization=str((job.get("source", {}) or {}).get("polarization", "Ex")),
                          port="top", position=(cx, cy, sz - pad / 2.0)))

    geo.add_monitor(Monitor("R", "flux", position=(cx, cy, sz - pad / 4.0), normal="z"))
    geo.add_monitor(Monitor("T", "flux", position=(cx, cy, pad / 4.0), normal="z"))
    return geo


def extract_fom(result_vectors: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Transmission phase at the design frequency (the surrogate target)."""
    phase = result_vectors.get("phase_rad") or []
    trans = result_vectors.get("transmission") or []
    if not phase:
        return {"transmission_phase_rad": None, "transmission_mag": None}
    mid = len(phase) // 2
    return {
        "transmission_phase_rad": phase[mid],
        "transmission_mag": trans[mid] if mid < len(trans) else None,
    }
