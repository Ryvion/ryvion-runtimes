"""antenna_unit_cell.py — microstrip patch antenna unit cell (3D, RF).

Params: patch_w_mm, patch_l_mm, substrate_eps, substrate_h_mm, feed_pos
Output: S11 / resonance + sampled radiation pattern.

This is a PR1 template and MUST produce a valid result contract. It leads the
NATIVE bundle path (gprMax/openEMS).
"""
from __future__ import annotations

from typing import Any, Dict

from geometry import MM, Box, Domain, Geometry, Monitor, Source

DEFAULT_OUTPUTS = ["s_params", "fom", "radiation_pattern"]

_REQUIRED = ["patch_w_mm", "patch_l_mm", "substrate_eps", "substrate_h_mm"]


def _validate(params: Dict[str, Any]) -> None:
    for k in _REQUIRED:
        if k not in params:
            raise ValueError(f"antenna_unit_cell missing required param {k!r}")
    if float(params["substrate_eps"]) < 1.0:
        raise ValueError("substrate_eps must be >= 1.0")


def build(params: Dict[str, Any], job: Dict[str, Any]) -> Geometry:
    _validate(params)
    pw = float(params["patch_w_mm"]) * MM
    pl = float(params["patch_l_mm"]) * MM
    eps_r = float(params["substrate_eps"])
    h = float(params["substrate_h_mm"]) * MM
    feed_pos = float(params.get("feed_pos", 0.3))  # fractional offset along length

    grid = job.get("grid", {}) or {}
    # resolution given as cells per mm here; convert to cells per meter.
    res_per_mm = float(grid.get("resolution", 4))
    res = res_per_mm / MM
    pml = int(grid.get("pml_layers", 8))

    # Domain: patch + lambda/4-ish air padding all around (3D).
    pad = 8.0 * MM
    sx = pw + 2 * pad
    sy = pl + 2 * pad
    sz = h + 2 * pad
    dom = Domain(size=(sx, sy, sz), resolution=res, pml_layers=pml,
                 dimensionality=int(grid.get("dimensionality", 3)),
                 symmetry=grid.get("symmetry"))
    geo = Geometry(template="antenna_unit_cell", domain=dom,
                   materials={"substrate_eps": eps_r})

    z0 = pad
    # Substrate dielectric slab.
    geo.add_box(Box("substrate", (pad, pad, z0), (pad + pw, pad + pl, z0 + h),
                    material="substrate", eps=eps_r))
    # Ground plane (PEC) at the bottom of the substrate.
    geo.add_box(Box("ground", (pad, pad, z0), (pad + pw, pad + pl, z0),
                    material="pec", is_metal=True))
    # Radiating patch (PEC) on top.
    geo.add_box(Box("patch", (pad, pad, z0 + h), (pad + pw, pad + pl, z0 + h),
                    material="pec", is_metal=True))

    # Source: lumped/voltage feed at the inset feed position.
    src = job.get("source", {}) or {}
    f0_ghz = float(src.get("center_freq_ghz", 0.0))
    if f0_ghz <= 0:
        # estimate dominant resonance from patch length (effective).
        c = 299792458.0
        eps_eff = (eps_r + 1) / 2
        f0 = c / (2 * pl * (eps_eff ** 0.5))
    else:
        f0 = f0_ghz * 1e9
    bw = float(src.get("bandwidth", 0.5)) * 1e9
    fx = pad + feed_pos * pw
    geo.add_source(Source(kind=str(src.get("type", "gaussian")), center_hz=f0,
                          bandwidth_hz=bw,
                          polarization=str(src.get("polarization", "Ez")),
                          port=str(src.get("port", "feed")),
                          position=(fx, pad + pl / 2.0, z0 + h / 2.0)))

    # Monitors: S-parameter port at feed + far-field radiation box.
    geo.add_monitor(Monitor("s11", "s_param", position=(fx, pad + pl / 2.0, z0)))
    geo.add_monitor(Monitor("rad", "radiation"))
    return geo


def extract_fom(result_vectors: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Resonant frequency = min |S11| (dB), with that minimum return loss."""
    import math

    s = result_vectors.get("s_params") or {}
    freqs = result_vectors.get("freqs_hz") or []
    s11_re = s.get("s11_re") or []
    s11_im = s.get("s11_im") or []
    if not s11_re or not freqs or len(s11_re) != len(freqs):
        return {"resonance_hz": None, "return_loss_db": None}
    best_db = None
    best_f = None
    for i, f in enumerate(freqs):
        mag = math.hypot(s11_re[i], s11_im[i] if i < len(s11_im) else 0.0)
        mag = max(mag, 1e-9)
        db = 20.0 * math.log10(mag)
        if best_db is None or db < best_db:
            best_db = db
            best_f = f
    return {"resonance_hz": best_f, "return_loss_db": best_db}
