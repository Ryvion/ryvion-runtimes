"""budget.py — cell/VRAM/runtime estimator for EM FDTD jobs.

SINGLE SOURCE OF TRUTH. This is a PURE FORMULA with no engine dependency so the
hub (StudyService / em_billing) can mirror it exactly in Go before dispatch to
set MinVRAMMB and TimeoutSeconds. Any change here MUST be mirrored hub-side.

The estimate is intentionally conservative (over-estimate VRAM with a safety
margin) because a low est_vram_mb -> OOM/exit-137 -> instant fail+refund with no
retry (see architecture doc risk #2).
"""
from __future__ import annotations

from typing import Any, Dict

# Bytes of GPU memory consumed per Yee cell, summed across E/H field components
# plus PML auxiliary fields, single precision. Empirically padded.
BYTES_PER_CELL = 96.0
# Safety multiplier applied to the raw field memory to cover engine overhead,
# monitor buffers, and allocator fragmentation.
VRAM_SAFETY = 1.6
# Fixed engine/runtime base footprint (CUDA/ROCm context, python, libs).
VRAM_BASE_MB = 800.0
# Nanoseconds of wall time per (cell * timestep), a rough consumer-GPU throughput
# proxy. Padded so timeouts do not trip honest-but-slow runs.
NS_PER_CELL_STEP = 0.9
# Floor / ceiling guards.
MIN_RUNTIME_S = 30
MAX_RUNTIME_S = 3600  # ~1hr cap per the doc; hub may clamp lower.


def estimate_cells(grid_dims_cells: Dict[str, int]) -> int:
    """Total Yee cells = product of per-axis cell counts (1 for absent axes)."""
    nx = max(1, int(grid_dims_cells.get("nx", 1)))
    ny = max(1, int(grid_dims_cells.get("ny", 1)))
    nz = max(1, int(grid_dims_cells.get("nz", 1)))
    return nx * ny * nz


def estimate_vram_mb(cells: int) -> int:
    """Conservative GPU memory estimate in MiB for a run of `cells` Yee cells."""
    field_mb = (cells * BYTES_PER_CELL) / (1024.0 * 1024.0)
    return int(VRAM_BASE_MB + field_mb * VRAM_SAFETY)


def estimate_steps(cells: int, max_steps: int) -> int:
    """Heuristic timestep count, bounded by the job's max_steps.

    A larger domain needs more steps for the source to traverse and decay; we
    use a cube-root-of-cells proxy times a constant, clamped to max_steps.
    """
    proxy = int(2000 + 40 * (cells ** (1.0 / 3.0)))
    return max(1, min(proxy, int(max_steps)))


def estimate_runtime_s(cells: int, steps: int) -> int:
    """Conservative wall-time estimate in seconds, clamped to [MIN, MAX]."""
    raw_ns = float(cells) * float(steps) * NS_PER_CELL_STEP
    secs = int(raw_ns / 1e9)
    return max(MIN_RUNTIME_S, min(secs, MAX_RUNTIME_S))


def estimate(grid_dims_cells: Dict[str, int], max_steps: int = 200000) -> Dict[str, Any]:
    """Return the full budget dict used by both runner and hub.

    grid_dims_cells: {"nx":..,"ny":..,"nz":..} cell counts derived by a template.
    """
    cells = estimate_cells(grid_dims_cells)
    steps = estimate_steps(cells, max_steps)
    return {
        "max_cells": cells,
        "est_vram_mb": estimate_vram_mb(cells),
        "est_runtime_s": estimate_runtime_s(cells, steps),
        "est_steps": steps,
    }
