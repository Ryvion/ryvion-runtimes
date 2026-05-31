"""geometry.py — engine-neutral geometry intermediate representation.

Templates emit a `Geometry` (a list of primitive shapes + a domain + source +
monitors + a derived grid cell count) which the engine layer translates into
its own native objects (gprMax #commands, openEMS CSX, or Meep geometry list).
Keeping this layer engine-neutral is what lets one job.json run on any engine.

All lengths are in METERS internally; templates convert from nm/um/mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

NM = 1e-9
UM = 1e-6
MM = 1e-3


@dataclass
class Box:
    """Axis-aligned box primitive (meters)."""
    name: str
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    material: str  # key into materials map / engine material table
    eps: Optional[float] = None  # relative permittivity if a dielectric
    is_metal: bool = False


@dataclass
class Source:
    kind: str            # "gaussian" | "mode" | "plane_wave"
    center_hz: float
    bandwidth_hz: float
    polarization: str    # "TE" | "TM" | "Ex" ...
    port: str            # named injection port / plane
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Monitor:
    name: str
    kind: str            # "flux" | "mode" | "s_param" | "field" | "radiation"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: str = "x"


@dataclass
class Domain:
    size: Tuple[float, float, float]      # meters
    resolution: float                      # cells per meter (or per wavelength scaled)
    pml_layers: int
    dimensionality: int                    # 2 or 3
    symmetry: Optional[str] = None
    dx: float = 0.0                        # cell size (meters), set by finalize()


@dataclass
class Geometry:
    template: str
    domain: Domain
    shapes: List[Box] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    monitors: List[Monitor] = field(default_factory=list)
    materials: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def grid_dims_cells(self) -> Dict[str, int]:
        """Derive integer Yee-cell counts per axis from domain size+resolution.

        `resolution` is interpreted as cells-per-meter. For 2D, the out-of-plane
        axis collapses to 1 cell.
        """
        sx, sy, sz = self.domain.size
        res = self.domain.resolution
        nx = max(1, int(round(sx * res)))
        ny = max(1, int(round(sy * res)))
        nz = max(1, int(round(sz * res)))
        if self.domain.dimensionality == 2:
            nz = 1
        return {"nx": nx, "ny": ny, "nz": nz}

    def add_box(self, box: Box) -> None:
        self.shapes.append(box)

    def add_source(self, src: Source) -> None:
        self.sources.append(src)

    def add_monitor(self, mon: Monitor) -> None:
        self.monitors.append(mon)
