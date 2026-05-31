"""templates/registry.py — device-template registry.

Adding a template = one file + one entry here. The job contract does not change.
Each template module MUST expose:
    build(params: dict, job: dict) -> geometry.Geometry
    extract_fom(result_vectors: dict, params: dict) -> dict   # canonical FoM
    DEFAULT_OUTPUTS: list[str]
"""
from __future__ import annotations

from typing import Callable, Dict

from . import (
    antenna_unit_cell,
    grating_coupler,
    metasurface_unit_cell,
    waveguide_coupler,
)

# template name -> module
REGISTRY: Dict[str, object] = {
    "grating_coupler": grating_coupler,
    "waveguide_coupler": waveguide_coupler,
    "antenna_unit_cell": antenna_unit_cell,
    "metasurface_unit_cell": metasurface_unit_cell,
}


def get(template_name: str):
    """Return the template module or raise a clear error."""
    mod = REGISTRY.get(template_name)
    if mod is None:
        raise ValueError(
            f"unknown device_template {template_name!r}; "
            f"known: {sorted(REGISTRY)}"
        )
    return mod


def names() -> list:
    return sorted(REGISTRY)
