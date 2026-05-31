#!/usr/bin/env python3
"""run.py — EM FDTD runner entrypoint.

Implements the Ryvion runner contract (read /work/job.json, write
/work/output/<artifact>, /work/receipt.json, /work/metrics.json) following the
archived finetune-runner skeleton:
  * SIGTERM -> graceful exit(143)
  * load_job() from /work/job.json
  * fail() ALWAYS writes an error receipt with a hash (never hang the hub)
  * sha256 of the artifact -> receipt output_hash

Works identically whether launched inside the OCI image or by the node-agent
native bundle entrypoint (it only touches /work and the engine modules).
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, Optional

def _arg_value(argv, name: str) -> Optional[str]:
    """Return the value of `--name X` or `--name=X` from argv, else None."""
    flag = "--" + name
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _resolve_work_dir(argv) -> str:
    """Work-dir precedence: --work argv > RYVION_WORK_DIR > RYV_WORK_DIR > /work.

    The OCI image launches run.py with no args and RYVION_WORK_DIR=/work; the
    node-agent native bundle launches it with `--work <dir>` (and RYV_WORK_DIR).
    Honouring both keeps ONE runner working in both lanes.
    """
    work = _arg_value(argv, "work")
    if not work:
        work = os.environ.get("RYVION_WORK_DIR") or os.environ.get("RYV_WORK_DIR") or "/work"
    return work


_ARGV = sys.argv[1:]
WORK_DIR = _resolve_work_dir(_ARGV)
# Make the resolved work dir the single source of truth for every engine module
# (engine_gprmax reads RYVION_WORK_DIR for its run dir), regardless of how the
# runner was launched.
os.environ["RYVION_WORK_DIR"] = WORK_DIR
JOB_PATH = _arg_value(_ARGV, "job") or os.path.join(WORK_DIR, "job.json")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
RECEIPT_PATH = os.path.join(WORK_DIR, "receipt.json")
METRICS_PATH = os.path.join(WORK_DIR, "metrics.json")

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

_terminating = False


def _on_sigterm(signum, frame):  # noqa: ARG001
    global _terminating
    _terminating = True
    # Graceful abort: write an error receipt then exit 143 (128+SIGTERM).
    try:
        fail("terminated by SIGTERM", load_job_safe(), exit_code=143)
    finally:
        sys.exit(143)


def load_job_safe() -> Dict[str, Any]:
    try:
        return load_job()
    except Exception:
        return {}


def load_job() -> Dict[str, Any]:
    with open(JOB_PATH, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    if job.get("schema_version") != "em.job.v1":
        raise ValueError(f"unsupported schema_version {job.get('schema_version')!r}")
    if job.get("task") != "em_simulation":
        raise ValueError(f"unsupported task {job.get('task')!r}")
    return job


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def fail(message: str, job: Optional[Dict[str, Any]] = None, exit_code: int = 1) -> None:
    """ALWAYS write an error receipt + metrics so the hub gets a clean fail+refund."""
    job = job or {}
    receipt = {
        "output_hash": EMPTY_SHA256,
        "variant_id": job.get("variant_id"),
        "study_id": job.get("study_id"),
        "converged": False,
        "engine": job.get("engine"),
        "engine_version": job.get("engine_version"),
        "mesh_cells": 0,
        "duration_ms": 0,
        "error": message,
    }
    metrics = {
        "output_name": "",
        "duration_ms": 0,
        "cells": 0,
        "vram_estimate_mb": 0,
        "steps": 0,
        "converged": False,
        "error": message,
    }
    try:
        write_json(RECEIPT_PATH, receipt)
        write_json(METRICS_PATH, metrics)
    except Exception:
        # last resort: print so the node captures it in logs.
        sys.stderr.write(f"FATAL: could not write error receipt: {message}\n")
    sys.stderr.write(f"em-fdtd-runner FAILED: {message}\n")
    sys.exit(exit_code)


def main() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    started = time.time()
    job: Dict[str, Any] = {}
    try:
        job = load_job()
    except Exception as exc:  # noqa: BLE001
        fail(f"load_job failed: {exc}", job)
        return

    try:
        # Imports are local so a top-level import error still routes through fail().
        import budget
        import engine as engine_mod
        import results as results_mod
        from templates import registry

        template_name = job.get("device_template")
        template = registry.get(template_name)
        params = job.get("params", {}) or {}

        geo = template.build(params, job)

        dims = geo.grid_dims_cells()
        bud = budget.estimate(dims, max_steps=int((job.get("run", {}) or {}).get("max_steps", 200000)))

        engine_name = job.get("engine", "gprmax")
        vectors = engine_mod.run(engine_name, geo, job)

        # Mark the analytic fallback in the reported engine_version when the
        # native library was not available. Engine modules tag their result with
        # a "solver" marker ("analytic" vs the real engine name); we append
        # "+analytic" so QA / aggregation can distinguish a real FDTD solve from
        # the deterministic placeholder (architecture doc §3.2).
        engine_version = job.get("engine_version") or f"{engine_name}-unknown"
        if vectors.get("solver") == "analytic" and "+analytic" not in engine_version:
            engine_version = f"{engine_version}+analytic"

        fom = template.extract_fom(vectors, params)
        wall_time_s = time.time() - started

        record = results_mod.build_record(job, vectors, fom, engine_version, wall_time_s)
        npz_path, json_path = results_mod.write(OUTPUT_DIR, record)
        out_name = results_mod.output_name(npz_path, json_path)
        artifact_path = npz_path or json_path

        output_hash = sha256_file(artifact_path)
        duration_ms = int(wall_time_s * 1000)

        receipt = {
            "output_hash": output_hash,
            "variant_id": job.get("variant_id"),
            "study_id": job.get("study_id"),
            "converged": bool(vectors.get("converged", False)),
            "engine": engine_name,
            "engine_version": engine_version,
            "mesh_cells": int(vectors.get("mesh_cells", 0)),
            "duration_ms": duration_ms,
        }
        metrics = {
            # metricsOutputName() in oci.go reads output_name to locate the
            # artifact — MUST be set to the actual file we wrote.
            "output_name": out_name,
            "duration_ms": duration_ms,
            "cells": int(vectors.get("mesh_cells", bud["max_cells"])),
            "vram_estimate_mb": int(bud["est_vram_mb"]),
            "steps": int(bud["est_steps"]),
            "converged": bool(vectors.get("converged", False)),
        }
        write_json(RECEIPT_PATH, receipt)
        write_json(METRICS_PATH, metrics)
        sys.stdout.write(
            f"em-fdtd-runner OK: {out_name} hash={output_hash[:12]} "
            f"cells={metrics['cells']} dur={duration_ms}ms\n"
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        fail(f"{type(exc).__name__}: {exc}", job)


if __name__ == "__main__":
    main()
