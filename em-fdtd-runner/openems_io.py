"""openems_io.py — Geometry IR -> openEMS CSX translation skeleton.

// TODO(em-verify): openEMS is not installed in CI/dev, so none of this is
exercised here. It is a structural skeleton parallel to gprmax_io.py: it shows
EXACTLY how the engine-neutral Geometry IR maps onto openEMS' CSXCAD geometry +
FDTD ports, and how S-params / radiation patterns are read back into the
em.result.v1 vectors. engine_openems.py only calls these functions when the
openEMS python binding is importable; otherwise it degrades to analytic.

openEMS API references (for the implementer finishing this on a target box):
  - CSXCAD.ContinuousStructure, AddMaterial/AddMetal, AddBox/AddCylinder
  - openEMS.openEMS, SetGaussExcite, SetBoundaryCond, AddLumpedPort (S-params),
    CreateNF2FFBox (radiation pattern).
  - port.CalcPort(sim_path, freqs) -> uf_inc/uf_ref -> S11 = uf_ref/uf_inc.

All heavy imports are LOCAL to the functions so importing this module in CI
(py_compile) does not require openEMS.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

from geometry import Box, Geometry

C0 = 299792458.0


def _freq_grid(job: Dict[str, Any], geo: Geometry) -> List[float]:
    freq = job.get("frequency", {}) or {}
    n = max(1, int(freq.get("points", 51)))
    if "range_ghz" in freq and freq["range_ghz"]:
        f_lo = float(freq["range_ghz"][0]) * 1e9
        f_hi = float(freq["range_ghz"][1]) * 1e9
    elif "range_nm" in freq and freq["range_nm"]:
        lo_nm, hi_nm = float(freq["range_nm"][0]), float(freq["range_nm"][1])
        f_lo, f_hi = C0 / (hi_nm * 1e-9), C0 / (lo_nm * 1e-9)
    else:
        f0 = geo.sources[0].center_hz if geo.sources else 1e10
        f_lo, f_hi = 0.9 * f0, 1.1 * f0
    if n == 1:
        return [(f_lo + f_hi) / 2.0]
    step = (f_hi - f_lo) / (n - 1)
    return [f_lo + i * step for i in range(n)]


def build_csx(geo: Geometry, job: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Translate the IR into a CSXCAD ContinuousStructure + a port table.

    // TODO(em-verify): finish + verify on a box with openEMS installed.
    Returns (csx, ports) where `ports` maps logical port names to openEMS port
    objects used later by extract().
    """
    from CSXCAD import ContinuousStructure  # type: ignore
    from openEMS import openEMS  # type: ignore

    fdtd = openEMS(NrTS=int((job.get("run", {}) or {}).get("max_steps", 30000)),
                   EndCriteria=1e-4)
    csx = ContinuousStructure()
    fdtd.SetCSX(csx)

    # Mesh from the IR domain (meters -> openEMS works in the chosen unit; we
    # keep meters via drawing_unit = 1).
    csx.GetGrid().SetDeltaUnit(1.0)
    dims = geo.grid_dims_cells()
    sx, sy, sz = geo.domain.size
    _add_uniform_mesh(csx, sx, sy, sz, dims)

    # Materials + primitives.
    for box in geo.shapes:
        if box.is_metal or box.material == "pec":
            metal = csx.AddMetal(box.name)
            _add_box(metal, box)
        else:
            mat = csx.AddMaterial(box.name)
            mat.SetMaterialProperty(epsilon=float(box.eps or 1.0))
            _add_box(mat, box)

    # Gaussian broadband excitation centered on the source.
    src = geo.sources[0] if geo.sources else None
    if src is not None:
        f0 = src.center_hz
        fc = max(src.bandwidth_hz, f0 * 0.2)
        fdtd.SetGaussExcite(f0, fc)
    fdtd.SetBoundaryCond(["PML_8"] * 6)

    # Lumped port at the antenna feed (the s_param monitor position).
    ports: Dict[str, Any] = {}
    for mon in geo.monitors:
        if mon.kind == "s_param":
            x, y, z = mon.position
            start = [x, y, z]
            stop = [x, y, z + (sz * 0.02)]
            ports["p1"] = fdtd.AddLumpedPort(
                1, 50.0, start, stop, "z", 1.0, priority=5)
    # NF2FF box for radiation pattern when requested.
    if any(m.kind == "radiation" for m in geo.monitors):
        ports["nf2ff"] = fdtd.CreateNF2FFBox()

    return (csx, {"fdtd": fdtd, "ports": ports})


def _add_uniform_mesh(csx, sx: float, sy: float, sz: float, dims: Dict[str, int]) -> None:
    grid = csx.GetGrid()
    for axis, length, n in (("x", sx, dims["nx"]), ("y", sy, dims["ny"]), ("z", sz, dims["nz"])):
        n = max(1, int(n))
        step = length / n if n else length
        lines = [i * step for i in range(n + 1)]
        grid.AddLine(axis, lines)


def _add_box(prim, box: Box) -> None:
    prim.AddBox(priority=10, start=list(box.lo), stop=list(box.hi))


def run_csx(csx: Any, ports: Dict[str, Any], geo: Geometry, job: Dict[str, Any]) -> str:
    """Run the openEMS FDTD solve, returning the sim output directory.

    // TODO(em-verify): finish on a target box. openEMS writes per-port voltage/
    current files into sim_path that CalcPort() reads back.
    """
    work = os.environ.get("RYVION_WORK_DIR", "/work")
    sim_path = os.path.join(work, "openems")
    os.makedirs(sim_path, exist_ok=True)
    fdtd = ports["fdtd"]
    fdtd.Run(sim_path, cleanup=True)
    return sim_path


def extract(sim_path: str, ports: Dict[str, Any], job: Dict[str, Any], geo: Geometry) -> Dict[str, Any]:
    """Read S-params / NF2FF back into em.result.v1 vectors.

    // TODO(em-verify): port.CalcPort(sim_path, freqs) gives uf_inc / uf_ref;
    S11 = uf_ref / uf_inc. Finish + verify on a box with openEMS installed.
    """
    import numpy as np  # type: ignore

    freqs = _freq_grid(job, geo)
    port_tbl = ports["ports"]
    p1 = port_tbl.get("p1")
    if p1 is None:
        # no port -> let the caller's exception path degrade to analytic.
        raise RuntimeError("openems extract: no port defined")

    p1.CalcPort(sim_path, np.asarray(freqs))
    s11 = p1.uf_ref / p1.uf_inc
    refl = np.clip(np.abs(s11), 0.0, 1.0)
    trans = 1.0 - refl ** 2
    phase = np.angle(s11)

    result = {
        "freqs_hz": [float(x) for x in freqs],
        "transmission": [float(x) for x in trans],
        "reflection": [float(x) for x in (refl ** 2)],
        "phase_rad": [float(x) for x in phase],
        "s_params": {
            "s11_re": [float(x) for x in s11.real],
            "s11_im": [float(x) for x in s11.imag],
            "s21_re": [float(math.sqrt(max(t, 0.0))) for t in trans],
            "s21_im": [0.0 for _ in trans],
        },
        "mesh_cells": geo.grid_dims_cells()["nx"] * geo.grid_dims_cells()["ny"] * geo.grid_dims_cells()["nz"],
        "converged": True,
        "solver": "openems",
    }
    return result
