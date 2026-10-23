"""Radial-grid specifications for two-coordinate spherical calculations."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
    RadialSincDVR,
)


GridFamily = Literal["femdvr", "sinc"]


class RadialGridSpec(Protocol):
    """Protocol implemented by supported radial-grid specifications."""

    family: GridFamily

    def build(self, l_max: int) -> "RadialGrid": ...


@dataclass(frozen=True)
class RadialGrid:
    """Prepared radial grid with boundary points removed where appropriate."""

    family: GridFamily
    dvr: object
    r: np.ndarray
    weights: np.ndarray
    l_max: int

    @property
    def n_points(self) -> int:
        return self.r.size

    def first_derivative(self, l: int) -> np.ndarray:
        self._validate_l(l)
        if self.family == "sinc":
            return self.dvr.D1_l[l]
        return self.dvr.D1[1:-1, 1:-1]

    def second_derivative(self, l: int) -> np.ndarray:
        self._validate_l(l)
        if self.family == "sinc":
            return self.dvr.D2_l[l]
        return self.dvr.D2[1:-1, 1:-1]

    def _validate_l(self, l: int) -> None:
        if not 0 <= l <= self.l_max:
            raise ValueError(f"l must satisfy 0 <= l <= {self.l_max}; got {l}")


@dataclass(frozen=True)
class FEMDVRGridSpec:
    """Finite-element DVR grid specified by element nodes and point counts."""

    nodes: tuple[float, ...] | list[float] | np.ndarray
    n_points: tuple[int, ...] | list[int] | np.ndarray
    family: GridFamily = field(default="femdvr", init=False)

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        n_points = np.asarray(self.n_points, dtype=int)
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("nodes must be a one-dimensional array of length >= 2")
        if nodes[0] != 0.0 or np.any(np.diff(nodes) <= 0.0):
            raise ValueError("nodes must start at zero and be strictly increasing")
        if n_points.shape != (nodes.size - 1,):
            raise ValueError("n_points must contain one entry per finite element")
        if np.any(n_points < 2):
            raise ValueError("each finite element must contain at least two points")
        object.__setattr__(self, "nodes", tuple(float(x) for x in nodes))
        object.__setattr__(self, "n_points", tuple(int(x) for x in n_points))

    def build(self, l_max: int) -> RadialGrid:
        if l_max < 0:
            raise ValueError("l_max must be non-negative")
        dvr = FEMDVR(
            np.asarray(self.nodes),
            np.asarray(self.n_points),
            Linear_map,
            GaussLegendreLobatto,
        )
        return RadialGrid(
            family=self.family,
            dvr=dvr,
            r=dvr.r[1:-1],
            weights=dvr.weights[1:-1],
            l_max=l_max,
        )


@dataclass(frozen=True)
class RadialSincDVRGridSpec:
    """Parity-adapted radial sinc DVR on ``0 < r <= r_max``."""

    r_max: float
    n_points: int
    family: GridFamily = field(default="sinc", init=False)

    def __post_init__(self) -> None:
        if self.r_max <= 0.0:
            raise ValueError("r_max must be positive")
        if self.n_points < 1:
            raise ValueError("n_points must be at least one")

    def build(self, l_max: int) -> RadialGrid:
        if l_max < 0:
            raise ValueError("l_max must be non-negative")
        dvr = RadialSincDVR(self.r_max, self.n_points, l_max=l_max)
        return RadialGrid(
            family=self.family,
            dvr=dvr,
            r=dvr.r,
            weights=dvr.weights,
            l_max=l_max,
        )


def require_same_grid_family(grid1: RadialGrid, grid2: RadialGrid) -> None:
    """Reject mixed FEM-DVR/sinc calculations."""

    if grid1.family != grid2.family:
        raise ValueError(
            "two-radial coordinates must use the same grid family; "
            f"got {grid1.family!r} and {grid2.family!r}"
        )
