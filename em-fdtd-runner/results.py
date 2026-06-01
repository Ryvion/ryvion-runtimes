"""results.py — assemble engine result vectors into the em.result.v1 contract.

Writes a TINY artifact (KB-MB, never field volumes):
  /work/output/result.npz   (numpy savez_compressed — the artifact, 1 dataset row)
  /work/output/result.json  (human/JS-readable mirror of the same payload)

The npz holds arrays; the json holds the full nested payload. Both encode the
same em.result.v1 record. numpy is optional: if unavailable we still write the
json mirror and a minimal .npz-less marker so the contract degrades gracefully
(the runner reports the actual output_name it produced).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

RESULT_SCHEMA = "em.result.v1"

try:
    import numpy as _np  # type: ignore

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover - exercised only without numpy
    _np = None
    _HAVE_NUMPY = False


def build_record(
    job: Dict[str, Any],
    vectors: Dict[str, Any],
    fom: Dict[str, Any],
    engine_version: str,
    wall_time_s: float,
) -> Dict[str, Any]:
    """Compose the em.result.v1 record (echoing params for the dataset row)."""
    return {
        "schema_version": RESULT_SCHEMA,
        "variant_id": job.get("variant_id"),
        "study_id": job.get("study_id"),
        "point_index": job.get("point_index"),
        "device_template": job.get("device_template"),
        "params": job.get("params", {}),
        "freqs_hz": vectors.get("freqs_hz", []),
        "transmission": vectors.get("transmission", []),
        "reflection": vectors.get("reflection", []),
        "phase_rad": vectors.get("phase_rad", []),
        "s_params": vectors.get("s_params", {}),
        "fom": fom,
        "radiation_pattern": vectors.get("radiation_pattern"),
        "best_geometry": vectors.get("best_geometry"),
        "converged": bool(vectors.get("converged", False)),
        "engine": job.get("engine"),
        "engine_version": engine_version,
        # The actual solver that produced these vectors ("gprmax"|"meep"|"openems"
        # |"analytic"). Lets the dataset/QA distinguish a real FDTD solve from the
        # deterministic analytic fallback without re-parsing the engine_version
        # suffix. Mirrors run.py's "+analytic" tagging (architecture doc §3.2).
        "solver": vectors.get("solver"),
        "mesh_cells": int(vectors.get("mesh_cells", 0)),
        "wall_time_s": wall_time_s,
    }


def _flatten_for_npz(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map the nested record onto flat npz keys (arrays + a json blob)."""
    flat: Dict[str, Any] = {}
    for key in ("freqs_hz", "transmission", "reflection", "phase_rad"):
        flat[key] = _np.asarray(record.get(key) or [], dtype=float)
    sp = record.get("s_params") or {}
    for k, v in sp.items():
        flat[f"s_params__{k}"] = _np.asarray(v or [], dtype=float)
    # everything non-array (params, fom, ids, radiation pattern) goes in a json
    # sidecar key so the row is fully self-describing inside the single npz.
    meta = {k: record[k] for k in record if k not in (
        "freqs_hz", "transmission", "reflection", "phase_rad", "s_params")}
    flat["meta_json"] = _np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype="uint8")
    return flat


def write(out_dir: str, record: Dict[str, Any]) -> Tuple[str, str]:
    """Write result.npz (if numpy) + result.json mirror. Returns (npz_path_or_'', json_path)."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)

    npz_path = ""
    if _HAVE_NUMPY:
        npz_path = os.path.join(out_dir, "result.npz")
        _np.savez_compressed(npz_path, **_flatten_for_npz(record))
    return npz_path, json_path


def output_name(npz_path: str, json_path: str) -> str:
    """The artifact basename the node should upload (npz preferred)."""
    chosen = npz_path or json_path
    return os.path.basename(chosen)
