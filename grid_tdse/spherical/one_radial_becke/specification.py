"""Immutable numerical specifications for the Becke prototype."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FEMDVRQuadratureSpec:
    """Element-local Gauss-Lobatto radial quadrature specification.

    Element endpoints are retained separately in each element. Consequently,
    an internal finite-element boundary occurs twice, once with each adjacent
    element's quadrature weight. This is required by later weak-form assembly.

    Parameters
    ----------
    nodes : array_like
        Strictly increasing finite-element boundaries starting at zero.
    n_points : array_like
        Number of Gauss-Lobatto points in every element. Each value must be
        an integer of at least two.
    """

    nodes: tuple[float, ...] | list[float] | np.ndarray
    n_points: tuple[int, ...] | list[int] | np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        raw_n_points = np.asarray(self.n_points)
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("nodes must be one-dimensional with at least two entries")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("nodes must contain only finite values")
        if nodes[0] != 0.0 or np.any(np.diff(nodes) <= 0.0):
            raise ValueError("nodes must start at zero and be strictly increasing")
        if raw_n_points.shape != (nodes.size - 1,):
            raise ValueError("n_points must contain one entry per finite element")
        if not np.issubdtype(raw_n_points.dtype, np.integer):
            raise TypeError("n_points must contain integers")
        n_points = raw_n_points.astype(int, copy=False)
        if np.any(n_points < 2):
            raise ValueError("each finite element must contain at least two points")

        object.__setattr__(self, "nodes", tuple(float(value) for value in nodes))
        object.__setattr__(
            self, "n_points", tuple(int(value) for value in n_points)
        )

    @property
    def n_elements(self) -> int:
        """Number of finite elements."""

        return len(self.n_points)

    @property
    def r_max(self) -> float:
        """Outer radial boundary."""

        return self.nodes[-1]

    @property
    def nonzero_shell_count(self) -> int:
        """Number of element-local radial nodes after omitting the origin."""

        return sum(self.n_points) - 1
