#!/usr/bin/env python3
"""smoke_real_physics.py — one-command "is the EM solve REAL?" check.

Run this ON A GPU NODE that has gprMax (+ pycuda + h5py + the CUDA toolkit)
installed. It drives the actual runner entrypoint (run.py) end-to-end against a
sample device job and then tells you, unambiguously, whether you got a REAL
gprMax FDTD solve or the deterministic ANALYTIC PLACEHOLDER:

  * REAL  -> receipt/result engine_version has NO "+analytic" suffix AND a gprMax
            .out HDF5 file was produced in the work dir.
  * PLACEHOLDER -> the runner could not import/run gprMax and fell back to the
            analytic solver (engine_version ends with "+analytic").

This is the founder hardware step: no hub, no node-agent, no bundle — just proof
that the physics path produces real numbers on this machine. Once this prints
REAL, the same engine wired through the native bundle / hub produces real data.

Usage:
  python tools/smoke_real_physics.py                         # antenna, auto GPU
  python tools/smoke_real_physics.py --template metasurface_unit_cell
  python tools/smoke_real_physics.py --job my_point.json     # your own em.job.v1
  python tools/smoke_real_physics.py --gpu cuda --keep       # force GPU, keep dir

Exit code: 0 = REAL solve confirmed; 1 = analytic fallback or runner failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.dirname(_TOOLS)
_RUN_PY = os.path.join(_RUNNER, "run.py")


def _sample_job(template: str) -> Dict[str, Any]:
    """A small, fast, physically-meaningful job per native-lead template."""
    base = {
        "schema_version": "em.job.v1",
        "task": "em_simulation",
        "engine": "gprmax",
        "engine_version": "smoke",
        "study_id": "smoke",
        "variant_id": "smoke#0000",
        "point_index": 0,
        "seed": 7,
        "device_template": template,
    }
    if template == "antenna_unit_cell":
        base.update({
            "params": {"patch_w_mm": 12.0, "patch_l_mm": 16.0, "substrate_eps": 4.4,
                       "substrate_h_mm": 1.6, "feed_pos": 0.3},
            "source": {"type": "gaussian", "center_freq_ghz": 5.0, "bandwidth": 4.0,
                       "polarization": "Ez", "port": "feed"},
            "grid": {"resolution": 4, "pml_layers": 8, "dimensionality": 3},
            "frequency": {"points": 41, "range_ghz": [3.0, 8.0]},
            "outputs": {"want": ["s_params", "fom", "radiation_pattern"]},
            "budget": {"max_cells": 30000000, "est_vram_mb": 4000, "est_runtime_s": 300},
        })
    elif template == "metasurface_unit_cell":
        base.update({
            "params": {"element_size_um": 4.0, "post_h_um": 30.0, "post_r_um": 8.0,
                       "lattice_period_um": 40.0, "freq_ghz": 100.0},
            "materials": {"post_eps": 11.7},
            # resolution = cells/µm; 2 keeps the smoke grid small + fast (~0.5M
            # cells) while still resolving the post. Real studies tune this up.
            "grid": {"resolution": 2, "pml_layers": 8, "dimensionality": 3},
            "frequency": {"points": 21, "range_ghz": [80.0, 120.0]},
            "outputs": {"want": ["reflection", "transmission", "phase"]},
            "budget": {"max_cells": 30000000, "est_vram_mb": 4000, "est_runtime_s": 300},
        })
    else:
        raise SystemExit(
            f"no built-in sample for template {template!r}; pass --job <file>. "
            f"native-lead templates: antenna_unit_cell, metasurface_unit_cell")
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify a REAL gprMax FDTD solve on this machine.")
    ap.add_argument("--template", default="antenna_unit_cell",
                    help="built-in sample template (antenna_unit_cell|metasurface_unit_cell)")
    ap.add_argument("--job", default="", help="path to your own em.job.v1 job.json (overrides --template)")
    ap.add_argument("--gpu", default="auto", choices=("auto", "cuda", "none"),
                    help="RYVION_EM_GPU: auto/cuda use the GPU, none forces CPU")
    ap.add_argument("--keep", action="store_true", help="keep the work dir for inspection")
    args = ap.parse_args(argv)

    work = tempfile.mkdtemp(prefix="ryv_em_smoke_")
    job_path = os.path.join(work, "job.json")
    if args.job:
        with open(args.job, encoding="utf-8") as fh:
            job = json.load(fh)
    else:
        job = _sample_job(args.template)
    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump(job, fh)

    env = dict(os.environ)
    env["RYVION_EM_GPU"] = args.gpu
    env["RYVION_WORK_DIR"] = work

    print(f"[smoke] runner   : {_RUN_PY}")
    print(f"[smoke] template : {job.get('device_template')}")
    print(f"[smoke] work dir : {work}")
    print(f"[smoke] GPU mode : {args.gpu}")
    print(f"[smoke] python   : {sys.executable}")
    print("[smoke] running gprMax FDTD solve (this can take minutes)...\n")

    proc = subprocess.run(
        [sys.executable, _RUN_PY, "--job", job_path, "--work", work],
        env=env, cwd=work,
    )

    receipt_path = os.path.join(work, "receipt.json")
    result_path = os.path.join(work, "output", "result.json")
    out_dir = os.path.join(work, "gprmax")
    out_files = [f for f in (os.listdir(out_dir) if os.path.isdir(out_dir) else []) if f.endswith(".out")]

    def _finish(code: int):
        if args.keep:
            print(f"\n[smoke] work dir kept at: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
        return code

    if proc.returncode != 0 or not os.path.exists(receipt_path):
        print(f"\n  RESULT: ✗ RUNNER FAILED (exit {proc.returncode}); no receipt written.")
        return _finish(1)

    with open(receipt_path, encoding="utf-8") as fh:
        receipt = json.load(fh)
    engine_version = str(receipt.get("engine_version", ""))
    converged = bool(receipt.get("converged"))
    analytic = engine_version.endswith("+analytic")

    print("\n" + "=" * 64)
    print(f"  engine          : {receipt.get('engine')}")
    print(f"  engine_version  : {engine_version}")
    print(f"  converged       : {converged}")
    print(f"  mesh_cells      : {receipt.get('mesh_cells')}")
    print(f"  gprMax .out file: {out_files[0] if out_files else '(none)'}")
    if os.path.exists(result_path):
        print(f"  result.json     : {result_path}")
    print("=" * 64)

    if analytic or not out_files:
        print("\n  RESULT: ✗ ANALYTIC PLACEHOLDER — this was NOT a real gprMax solve.")
        print("  gprMax/pycuda/h5py are not importable+runnable in THIS python, or")
        print("  the solve failed and degraded to analytic. Fix the install (see")
        print("  GPU_NODE_PROVISIONING.md), then re-run. The work was NOT real physics.")
        return _finish(1)

    print("\n  RESULT: ✓ REAL gprMax FDTD SOLVE CONFIRMED on this machine.")
    print("  The native bundle / hub path now produces real EM data with this engine.")
    return _finish(0)


if __name__ == "__main__":
    raise SystemExit(main())
