"""engine_openems.py — openEMS backend (native bundle path, RF/antenna).

openEMS ships prebuilt binaries (incl. Windows) so it packages cleanly into the
native bundle and is a second native lead for antenna/RF templates. This module
mirrors engine_gprmax.py's shape exactly:

  1. availability check  — openEMS python binding (CSXCAD + openEMS) importable.
  2. IR -> CSX            — openems_io.build_csx(geo, job) translates the IR.
  3. run                  — openems_io.run_csx(...) drives the FDTD engine.
  4. extract              — openems_io.extract(...) -> em.result.v1 vectors.

// TODO(em-verify): the CSX translation + S-param/NF2FF extraction are written
as a skeleton in openems_io.py but cannot be verified here (openEMS not
installed). Until then we DEGRADE to engine_analytic so the contract holds and
the runner never crashes. Same fallback discipline as gprMax.
"""
from __future__ import annotations

import sys
from typing import Any, Dict

import engine_analytic
from geometry import Geometry


def _native_available() -> bool:
    try:
        import CSXCAD  # type: ignore  # noqa: F401
        import openEMS  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def solve(geo: Geometry, job: Dict[str, Any]) -> Dict[str, Any]:
    if not _native_available():
        return engine_analytic.solve(geo, job)
    try:
        import openems_io

        csx, ports = openems_io.build_csx(geo, job)            # TODO(em-verify)
        sim_dir = openems_io.run_csx(csx, ports, geo, job)      # TODO(em-verify)
        return openems_io.extract(sim_dir, ports, job, geo)     # TODO(em-verify)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"engine_openems: native solve failed, degrading to analytic: {exc}\n")
        return engine_analytic.solve(geo, job)
