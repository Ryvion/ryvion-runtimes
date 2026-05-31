"""waveguide_coupler.py — directional waveguide coupler (2D photonics).

Params: gap_nm, coupling_length_um, wg_width_nm
Output: S-params (through / cross).
"""
from __future__ import annotations

from typing import Any, Dict

from geometry import NM, UM, Box, Domain, Geometry, Monitor, Source

DEFAULT_OUTPUTS = ["s_params", "transmission", "fom"]

_REQUIRED = ["gap_nm", "coupling_length_um", "wg_width_nm"]


def _validate(params: Dict[str, Any]) -> None:
    for k in _REQUIRED:
        if k not in params:
            raise ValueError(f"waveguide_coupler missing required param {k!r}")


def build(params: Dict[str, Any], job: Dict[str, Any]) -> Geometry:
    _validate(params)
    gap = float(params["gap_nm"]) * NM
    clen = float(params["coupling_length_um"]) * UM
    w = float(params["wg_width_nm"]) * NM

    materials = job.get("materials", {}) or {}
    core_index = float(materials.get("core_index", 3.48))

    grid = job.get("grid", {}) or {}
    res = float(grid.get("resolution", 30)) / UM
    pml = int(grid.get("pml_layers", 8))

    pad = 3.0 * UM
    sx = clen + 2 * pad
    sy = 2 * w + gap + 4 * UM
    dom = Domain(size=(sx, sy, 0.0), resolution=res, pml_layers=pml,
                 dimensionality=int(grid.get("dimensionality", 2)),
                 symmetry=grid.get("symmetry"))
    geo = Geometry(template="waveguide_coupler", domain=dom, materials=materials)

    cy = sy / 2.0
    y_top = cy + gap / 2.0
    y_bot = cy - gap / 2.0 - w
    eps = core_index ** 2
    geo.add_box(Box("wg_through", (0.0, y_top, 0.0), (sx, y_top + w, 0.0),
                    material="core", eps=eps))
    geo.add_box(Box("wg_cross", (pad, y_bot, 0.0), (pad + clen, y_bot + w, 0.0),
                    material="core", eps=eps))

    src = job.get("source", {}) or {}
    lam = float(src.get("center_wavelength_nm", 1550)) * NM
    c = 299792458.0
    f0 = c / lam
    bw_nm = float(src.get("bandwidth", 100)) * NM
    fbw = (c / (lam - bw_nm / 2)) - (c / (lam + bw_nm / 2))
    geo.add_source(Source(kind=str(src.get("type", "mode")), center_hz=f0,
                          bandwidth_hz=abs(fbw),
                          polarization=str(src.get("polarization", "TE")),
                          port=str(src.get("port", "through_in")),
                          position=(pad / 2.0, y_top + w / 2.0, 0.0)))

    geo.add_monitor(Monitor("through", "mode", position=(sx - pad / 2.0, y_top + w / 2.0, 0.0), normal="x"))
    geo.add_monitor(Monitor("cross", "mode", position=(sx - pad / 2.0, y_bot + w / 2.0, 0.0), normal="x"))
    return geo


def extract_fom(result_vectors: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Splitting ratio: cross / (through + cross) at band center."""
    s = result_vectors.get("s_params") or {}
    s21 = s.get("s21_re") or []
    s31 = s.get("s31_re") or []
    if not s21 or not s31:
        return {"splitting_ratio": None}
    mid = len(s21) // 2
    thru = abs(s21[mid]) ** 2
    cross = abs(s31[mid]) ** 2 if mid < len(s31) else 0.0
    denom = thru + cross
    return {"splitting_ratio": (cross / denom) if denom > 0 else None}
