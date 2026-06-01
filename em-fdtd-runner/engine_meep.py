"""engine_meep.py — Meep backend (OCI/Linux photonics lane).

Meep (C++/MPI/conda) is hardest to ship native, so it is the OCI/Linux path for
photonics templates (grating_coupler, waveguide_coupler). This module mirrors
engine_gprmax.py's shape:

  1. availability check  — `import meep` succeeds (Linux OCI image).
  2. IR -> mp.Simulation — meep_io.build_simulation(geo, job).
  3. run                 — meep_io.run_until_decay(...).
  4. extract             — meep_io.extract(...) -> em.result.v1 vectors
                           (flux/get_eigenmode_coefficients -> transmission /
                           phase / s_params).

// TODO(em-verify): the mp.Simulation translation + mode-coefficient extraction
are written as a skeleton in meep_io.py but cannot be verified here (Meep not
installed). Until then we DEGRADE to engine_analytic so the contract holds and
the runner never crashes.
"""
from __future__ import annotations

import sys
from typing import Any, Dict

import engine_analytic
from geometry import Geometry


def _native_available() -> bool:
    try:
        import meep  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def solve(geo: Geometry, job: Dict[str, Any]) -> Dict[str, Any]:
    if not _native_available():
        return engine_analytic.solve(geo, job)
    try:
        import meep_io

        return meep_io.solve_native(geo, job)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"engine_meep: native solve failed, degrading to analytic: {exc}\n")
        return engine_analytic.solve(geo, job)
