"""engine.py — engine layer dispatch.

Selects the FDTD backend from job["engine"] and runs the Geometry, returning a
dict of result vectors in the engine-neutral `em.result.v1` shape (freqs_hz,
transmission, phase_rad, s_params, radiation_pattern, mesh_cells, converged).

Native-first ordering per architecture §10:
  - gprmax / openems  -> native bundle path (antenna + metasurface), preferred.
  - meep              -> OCI/Linux photonics lane.

Each concrete engine is a thin module (engine_gprmax / engine_openems /
engine_meep). When the heavy native library is not importable (e.g. CI syntax
check, or a bundle that hasn't been provisioned), the engine falls back to an
ANALYTIC solver so the result CONTRACT is always satisfied. The analytic path
is clearly marked converged=true with engine_version suffix "+analytic" so QA /
aggregation can distinguish a real solve. This guarantees grating_coupler and
antenna_unit_cell always emit a valid result.npz (doc requirement).
"""
from __future__ import annotations

from typing import Any, Dict

from geometry import Geometry

# Engine name -> module path. Real solves live in engine_<name>.py; missing
# heavy deps trigger the analytic fallback inside each module.
_ENGINES = {
    "gprmax": "engine_gprmax",
    "openems": "engine_openems",
    "meep": "engine_meep",
}


def run(engine_name: str, geo: Geometry, job: Dict[str, Any]) -> Dict[str, Any]:
    """Run the geometry on the requested engine, returning result vectors."""
    name = (engine_name or "").lower()
    if name not in _ENGINES:
        raise ValueError(f"unknown engine {engine_name!r}; known: {sorted(_ENGINES)}")

    import importlib

    mod = importlib.import_module(_ENGINES[name])
    return mod.solve(geo, job)
