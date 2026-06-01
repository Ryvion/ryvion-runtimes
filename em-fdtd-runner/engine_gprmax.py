"""engine_gprmax.py — gprMax backend (native bundle path, antenna + 6G + GPR).

gprMax is pure-Python + CUDA (pip), which packages cleanly into the native
`~/.ryvion/runtimes/em` bundle (NATIVE_BUNDLE.md). It is the native-first,
antenna / 6G-metasurface lead engine.

Real solve pipeline (this module owns it; the pure, testable halves live in
gprmax_io.py):
  1. availability check  — gprMax import + a runnable entrypoint + numpy/h5py.
  2. IR -> gprMax input   — gprmax_io.build_input(geo, job) writes a `.in` file.
  3. run gprMax           — `python -m gprMax model.in [-gpu]` via subprocess,
                            offline, in the work dir, with a time budget.
  4. parse `.out` HDF5    — gprmax_io.parse_output(out, job, geo) -> em.result.v1
                            vectors (S11/return-loss for antenna; reflection /
                            transmission mag+phase for metasurface/photonics).

DEGRADE-TO-ANALYTIC is mandatory: when gprMax / h5py / numpy are unavailable
(CI, dev, unprovisioned bundle) OR the subprocess fails / produces no `.out`,
we fall back to engine_analytic.solve so the result CONTRACT always holds and
the runner never crashes. The analytic vs real distinction is surfaced by the
"+analytic" engine_version suffix that the caller appends (results.py / run.py).

GPU: we pass `-gpu` to gprMax only when a CUDA device is detected (env override
RYVION_EM_GPU=cuda|none|auto). gprMax selects the device; absence of CUDA simply
runs the CPU/OpenMP build. // TODO(em-verify): exercised only on a real
gprMax+GPU bundle (not installed in CI/dev here).
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, Optional

import engine_analytic
from geometry import Geometry

# Soft caps so a runaway solve cannot hang a node; the hub also sets
# TimeoutSeconds and the node kills on timeout. This is a belt-and-braces guard.
_DEFAULT_SUBPROCESS_TIMEOUT_S = 3600


def _have_numpy_h5py() -> bool:
    try:
        import h5py  # type: ignore  # noqa: F401
        import numpy  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _native_available() -> bool:
    """True only when gprMax is importable AND the output parser deps exist."""
    if not _have_numpy_h5py():
        return False
    try:
        import gprMax  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _gpu_flag() -> Optional[str]:
    """Return a gprMax CLI GPU flag value, or None to run on CPU.

    RYVION_EM_GPU env: "none" forces CPU; "cuda"/"auto" enables CUDA when a
    device is visible. We detect CUDA cheaply via env + a best-effort probe so
    we never pass -gpu to a CPU-only gprMax build (which would error out and
    needlessly trip the analytic fallback).
    """
    pref = (os.environ.get("RYVION_EM_GPU", "auto") or "auto").lower()
    if pref == "none":
        return None
    if _cuda_visible():
        return "0"  # gprMax -gpu takes an optional device id; 0 = first device.
    if pref == "cuda":
        # explicitly requested but no device detected -> CPU (fall through).
        return None
    return None


def _cuda_visible() -> bool:
    # CUDA_VISIBLE_DEVICES set non-empty, or pycuda importable with a device.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() not in ("", "-1"):
        return True
    try:
        import pycuda.driver as drv  # type: ignore

        drv.init()
        return drv.Device.count() > 0
    except Exception:
        return False


def solve(geo: Geometry, job: Dict[str, Any]) -> Dict[str, Any]:
    if not _native_available():
        # CI / dev / unprovisioned bundle: keep the contract via analytic.
        return engine_analytic.solve(geo, job)

    try:
        return _solve_native(geo, job)
    except Exception as exc:  # noqa: BLE001 — never crash the runner on a solve.
        sys.stderr.write(f"engine_gprmax: native solve failed, degrading to analytic: {exc}\n")
        return engine_analytic.solve(geo, job)


def _solve_native(geo: Geometry, job: Dict[str, Any]) -> Dict[str, Any]:
    """Run the real gprMax solve. // TODO(em-verify) needs gprMax+GPU bundle."""
    import gprmax_io

    work = os.environ.get("RYVION_WORK_DIR", "/work")
    run_dir = os.path.join(work, "gprmax")
    os.makedirs(run_dir, exist_ok=True)
    in_path = os.path.join(run_dir, "model.in")
    out_path = os.path.join(run_dir, "model.out")

    in_text = gprmax_io.build_input(geo, job)
    with open(in_path, "w", encoding="utf-8") as fh:
        fh.write(in_text)

    cmd = [sys.executable, "-m", "gprMax", in_path]
    gpu = _gpu_flag()
    # gprMax's CUDA solver does NOT support #transmission_line (the antenna S11
    # feed). Such a run must execute on CPU, or gprMax errors out and we lose the
    # real solve to the analytic fallback. Detect it and drop -gpu so antennas
    # still produce REAL physics (CPU-bound). Metasurface/grating/waveguide use a
    # dipole/plane-wave source + receivers and stay on the GPU.
    # TODO(em-verify): a GPU antenna path needs the #voltage_source + reference-run
    # S11 method (two solves) instead of the one-run transmission-line method.
    if gpu is not None and "#transmission_line" in in_text:
        sys.stderr.write(
            "engine_gprmax: #transmission_line present -> running on CPU "
            "(gprMax GPU has no transmission-line support)\n")
        gpu = None
    if gpu is not None:
        cmd += ["-gpu", gpu]

    timeout = int((job.get("budget", {}) or {}).get(
        "est_runtime_s", _DEFAULT_SUBPROCESS_TIMEOUT_S)) or _DEFAULT_SUBPROCESS_TIMEOUT_S
    # generous: allow 3x the estimate before the hard kill, capped.
    timeout = min(max(timeout * 3, 120), _DEFAULT_SUBPROCESS_TIMEOUT_S)

    env = dict(os.environ)
    # Offline: gprMax needs no network; mirror --network=none intent.
    env.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() or 1)))

    proc = subprocess.run(
        cmd, cwd=run_dir, env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gprMax exited {proc.returncode}: "
            f"{(proc.stdout or b'')[-800:].decode('utf-8', 'ignore')}"
        )
    if not os.path.exists(out_path):
        # gprMax names the output after the input stem; locate it if renamed.
        out_path = _find_out_file(run_dir, out_path)
        if out_path is None:
            raise RuntimeError("gprMax produced no .out file")

    return gprmax_io.parse_output(out_path, job, geo)


def _find_out_file(run_dir: str, preferred: str) -> Optional[str]:
    if os.path.exists(preferred):
        return preferred
    candidates = [p for p in os.listdir(run_dir) if p.endswith(".out")]
    if not candidates:
        return None
    return os.path.join(run_dir, sorted(candidates)[0])
