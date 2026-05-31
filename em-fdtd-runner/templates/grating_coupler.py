"""grating_coupler.py — silicon photonics fiber-to-chip grating coupler (2D).

Params: period_nm, fill_factor, etch_depth_nm, n_teeth, wg_thickness_nm,
        fiber_angle_deg
Output: coupling efficiency + transmission/phase vs wavelength.

This is a PR1 template and MUST produce a valid result contract.
"""
from __future__ import annotations

from typing import Any, Dict

from geometry import NM, UM, Box, Domain, Geometry, Monitor, Source

DEFAULT_OUTPUTS = ["transmission", "phase", "fom"]

_REQUIRED = ["period_nm", "fill_factor", "etch_depth_nm", "n_teeth", "wg_thickness_nm"]


def _validate(params: Dict[str, Any]) -> None:
    for k in _REQUIRED:
        if k not in params:
            raise ValueError(f"grating_coupler missing required param {k!r}")
    if not (0.0 < float(params["fill_factor"]) < 1.0):
        raise ValueError("fill_factor must be in (0,1)")
    if int(params["n_teeth"]) < 1:
        raise ValueError("n_teeth must be >= 1")


def build(params: Dict[str, Any], job: Dict[str, Any]) -> Geometry:
    _validate(params)
    period = float(params["period_nm"]) * NM
    ff = float(params["fill_factor"])
    etch = float(params["etch_depth_nm"]) * NM
    n_teeth = int(params["n_teeth"])
    wg_t = float(params["wg_thickness_nm"]) * NM

    materials = job.get("materials", {}) or {}
    core_index = float(materials.get("core_index", 3.48))
    clad_index = float(materials.get("clad_index", 1.44))

    grid = job.get("grid", {}) or {}
    resolution_per_um = float(grid.get("resolution", 30))
    res = resolution_per_um / UM  # cells per meter
    pml = int(grid.get("pml_layers", 8))

    # Domain: grating region length + waveguide lead-in + PML padding.
    grating_len = n_teeth * period
    pad = 3.0 * UM
    sx = grating_len + 2 * pad
    sy = max(4.0 * UM, wg_t + 4.0 * UM)

    dom = Domain(
        size=(sx, sy, 0.0),
        resolution=res,
        pml_layers=pml,
        dimensionality=int(grid.get("dimensionality", 2)),
        symmetry=grid.get("symmetry"),
    )
    geo = Geometry(template="grating_coupler", domain=dom, materials=materials)

    # Slab waveguide (core).
    y0 = sy / 2.0 - wg_t / 2.0
    geo.add_box(Box("slab", (0.0, y0, 0.0), (sx, y0 + wg_t, 0.0),
                    material="core", eps=core_index ** 2))

    # Grating teeth: etched grooves (clad index) into the top of the slab.
    x = pad
    for i in range(n_teeth):
        groove_w = period * (1.0 - ff)
        gx0 = x + period * ff
        geo.add_box(Box(f"groove_{i}", (gx0, y0 + wg_t - etch, 0.0),
                        (gx0 + groove_w, y0 + wg_t, 0.0),
                        material="clad", eps=clad_index ** 2))
        x += period

    # Source: gaussian beam from "fiber" port above the grating.
    src = job.get("source", {}) or {}
    lam = float(src.get("center_wavelength_nm", 1550)) * NM
    c = 299792458.0
    f0 = c / lam
    bw_nm = float(src.get("bandwidth", 100)) * NM
    fbw = (c / (lam - bw_nm / 2)) - (c / (lam + bw_nm / 2))
    geo.add_source(Source(kind=str(src.get("type", "gaussian")), center_hz=f0,
                          bandwidth_hz=abs(fbw),
                          polarization=str(src.get("polarization", "TE")),
                          port=str(src.get("port", "fiber")),
                          position=(sx / 2.0, y0 + wg_t + 1.0 * UM, 0.0)))

    # Monitors: transmission flux + waveguide mode at the left output.
    geo.add_monitor(Monitor("T", "flux", position=(pad / 2.0, y0 + wg_t / 2, 0.0), normal="x"))
    geo.add_monitor(Monitor("mode", "mode", position=(pad / 2.0, y0 + wg_t / 2, 0.0), normal="x"))
    geo.notes["fiber_angle_deg"] = float(params.get("fiber_angle_deg", 0.0))
    return geo


def extract_fom(result_vectors: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Coupling efficiency (dB) at the peak of the transmission spectrum."""
    import math

    t = result_vectors.get("transmission") or []
    if not t:
        return {"coupling_efficiency_db": None}
    peak = max(t)
    peak = max(peak, 1e-9)
    return {"coupling_efficiency_db": 10.0 * math.log10(peak)}
