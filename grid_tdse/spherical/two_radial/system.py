"""Preparation of grid and angular data for two-radial Hamiltonians."""

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .angular import AngularBasisSpec, AngularPairBasis, CoulombMultipoleSpec
from .coulomb import (
    CoulombTerm,
    RadialCoulombMethod,
    RadialCoulombMultipoles,
    build_radial_coulomb_multipoles,
)
from .couplings import CoulombCouplings, build_coulomb_couplings
from .grids import RadialGrid, RadialGridSpec, require_same_grid_family


class RadialDiagonalTerm(Protocol):
    """Term diagonal in the angular basis and radial DVR representation."""

    def diagonal(self, l: int, m: int, grid: RadialGrid) -> np.ndarray: ...


@dataclass(frozen=True)
class RadialPotential:
    """Radial potential shared by all angular channels."""

    values: float | np.ndarray | Callable[[np.ndarray], np.ndarray]

    def diagonal(self, l: int, m: int, grid: RadialGrid) -> np.ndarray:
        del l, m
        return _as_radial_values(self.values, grid.r)


@dataclass(frozen=True)
class AngularRadialPotential:
    """Radial potential that may depend on the ``(l, m)`` channel."""

    values: Callable[[int, int, np.ndarray], np.ndarray]

    def diagonal(self, l: int, m: int, grid: RadialGrid) -> np.ndarray:
        return _as_radial_values(self.values(l, m, grid.r), grid.r)


@dataclass(frozen=True)
class CoordinateSpec:
    """Grid, angular basis, and one-body terms for one coordinate."""

    grid: RadialGridSpec
    basis: AngularBasisSpec
    kinetic_prefactor: float
    one_body_terms: tuple[RadialDiagonalTerm, ...] = field(default_factory=tuple)

    def build_grid(self) -> RadialGrid:
        return self.grid.build(self.basis.l_max)

    def build_one_body_operator(self, grid: RadialGrid | None = None) -> np.ndarray:
        grid = self.build_grid() if grid is None else grid
        operator = np.zeros(
            (self.basis.n_channels, grid.n_points, grid.n_points), dtype=float
        )
        for channel, (l, m) in enumerate(self.basis.states):
            radial_laplacian = grid.second_derivative(l) - np.diag(
                l * (l + 1) / grid.r**2
            )
            block = self.kinetic_prefactor * radial_laplacian
            for term in self.one_body_terms:
                block = block + np.diag(term.diagonal(l, m, grid))
            operator[channel] = block
        return operator


@dataclass(frozen=True)
class TwoRadialSystemSpec:
    """Complete setup recipe for a two-radial Hamiltonian foundation.

    ``M_tot=None`` uses the full Cartesian product of one-coordinate angular
    channels. An integer ``M_tot`` retains only pairs with ``m1 + m2=M_tot``.
    """

    coordinate1: CoordinateSpec
    coordinate2: CoordinateSpec
    multipoles: CoulombMultipoleSpec
    coulomb_terms: tuple[CoulombTerm, ...]
    M_tot: int | None = None
    radial_coulomb_method: RadialCoulombMethod | None = None

    def build(self) -> "TwoRadialSystem":
        grid1 = self.coordinate1.build_grid()
        grid2 = self.coordinate2.build_grid()
        require_same_grid_family(grid1, grid2)
        radial_multipoles = build_radial_coulomb_multipoles(
            grid1,
            grid2,
            self.multipoles,
            self.coulomb_terms,
            method=self.radial_coulomb_method,
        )
        angular_pairs = AngularPairBasis(
            self.coordinate1.basis,
            self.coordinate2.basis,
            self.M_tot,
        )
        couplings = build_coulomb_couplings(
            self.coordinate1.basis,
            self.coordinate2.basis,
            self.multipoles,
            angular_pairs=angular_pairs,
        )
        return TwoRadialSystem(
            grid1=grid1,
            grid2=grid2,
            basis1=self.coordinate1.basis,
            basis2=self.coordinate2.basis,
            angular_pairs=angular_pairs,
            h1=self.coordinate1.build_one_body_operator(grid1),
            h2=self.coordinate2.build_one_body_operator(grid2),
            multipoles=self.multipoles,
            radial_multipoles=radial_multipoles,
            couplings=couplings,
        )


@dataclass(frozen=True)
class TwoRadialSystem:
    """Prepared arrays and sparse couplings for a two-radial system."""

    grid1: RadialGrid
    grid2: RadialGrid
    basis1: AngularBasisSpec
    basis2: AngularBasisSpec
    h1: np.ndarray
    h2: np.ndarray
    multipoles: CoulombMultipoleSpec
    radial_multipoles: RadialCoulombMultipoles
    couplings: CoulombCouplings
    angular_pairs: AngularPairBasis | None = None

    def __post_init__(self) -> None:
        angular_pairs = self.angular_pairs
        if angular_pairs is None:
            angular_pairs = AngularPairBasis(self.basis1, self.basis2)
            object.__setattr__(self, "angular_pairs", angular_pairs)
        elif (
            angular_pairs.basis1 != self.basis1
            or angular_pairs.basis2 != self.basis2
        ):
            raise ValueError("angular_pairs does not match the coordinate bases")

    @property
    def M_tot(self) -> int | None:
        return self.angular_pairs.M_tot

    @property
    def uses_compact_angular_storage(self) -> bool:
        return self.angular_pairs.is_compact

    @property
    def full_psi_shape(self) -> tuple[int, int, int, int]:
        return (
            self.basis1.n_channels,
            self.basis2.n_channels,
            self.grid1.n_points,
            self.grid2.n_points,
        )

    @property
    def psi_shape(self) -> tuple[int, ...]:
        if self.uses_compact_angular_storage:
            return (
                self.angular_pairs.n_pairs,
                self.grid1.n_points,
                self.grid2.n_points,
            )
        return self.full_psi_shape

    def angular_block(
        self, state: np.ndarray, channel1: int, channel2: int
    ) -> np.ndarray:
        """Return one retained angular block without copying it."""

        if self.uses_compact_angular_storage:
            return state[self.angular_pairs.index(channel1, channel2)]
        return state[channel1, channel2]

    def set_angular_block(
        self,
        state: np.ndarray,
        channel1: int,
        channel2: int,
        values: np.ndarray,
    ) -> None:
        if self.uses_compact_angular_storage:
            state[self.angular_pairs.index(channel1, channel2)] = values
        else:
            state[channel1, channel2] = values

    def add_to_angular_block(
        self,
        state: np.ndarray,
        channel1: int,
        channel2: int,
        values: np.ndarray,
    ) -> None:
        if self.uses_compact_angular_storage:
            pair = self.angular_pairs.get(channel1, channel2)
            if pair is None:
                raise ValueError(
                    f"angular channel pair {(channel1, channel2)} is not retained"
                )
            state[pair] += values
        else:
            state[channel1, channel2] += values

    def pack_state(self, state: np.ndarray) -> np.ndarray:
        """Pack a full angular-product state into this representation."""

        values = np.asarray(state)
        if values.shape == self.psi_shape:
            return values.copy()
        if values.shape != self.full_psi_shape:
            raise ValueError(
                f"state has shape {values.shape}; expected {self.psi_shape} or "
                f"{self.full_psi_shape}"
            )
        if not self.uses_compact_angular_storage:
            return values.copy()
        return np.ascontiguousarray(
            np.asarray([values[i, j] for i, j in self.angular_pairs.pairs])
        )

    def unpack_state(self, state: np.ndarray) -> np.ndarray:
        """Return a full angular-product array, inserting absent blocks as zero."""

        values = np.asarray(state)
        if values.shape != self.psi_shape:
            raise ValueError(
                f"state has shape {values.shape}; expected {self.psi_shape}"
            )
        if not self.uses_compact_angular_storage:
            return values.copy()
        result = np.zeros(self.full_psi_shape, dtype=values.dtype)
        for pair, (i, j) in enumerate(self.angular_pairs.pairs):
            result[i, j] = values[pair]
        return result

    def hamiltonian(self, *, tile_size: int = 2048):
        from .hamiltonian import TwoRadialHamiltonian

        return TwoRadialHamiltonian(
            self.h1,
            self.h2,
            self.radial_multipoles,
            self.couplings,
            angular_pairs=self.angular_pairs,
            tile_size=tile_size,
        )

    def inner_product(self, left: np.ndarray, right: np.ndarray) -> complex:
        """Return the quadrature-weighted two-radial inner product."""

        left = np.asarray(left)
        right = np.asarray(right)
        if left.shape != self.psi_shape:
            raise ValueError(
                f"left state has shape {left.shape}; expected {self.psi_shape}"
            )
        if right.shape != self.psi_shape:
            raise ValueError(
                f"right state has shape {right.shape}; expected {self.psi_shape}"
            )
        subscripts = (
            "pab,pab,a,b->"
            if self.uses_compact_angular_storage
            else "ijab,ijab,a,b->"
        )
        return np.einsum(
            subscripts,
            left.conj(),
            right,
            self.grid1.weights,
            self.grid2.weights,
            optimize=True,
        )


def radial_potential(
    values: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
) -> RadialPotential:
    return RadialPotential(values)


def angular_radial_potential(
    values: Callable[[int, int, np.ndarray], np.ndarray],
) -> AngularRadialPotential:
    return AngularRadialPotential(values)


def _as_radial_values(values, r: np.ndarray) -> np.ndarray:
    if callable(values):
        values = values(r)
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full_like(r, values, dtype=float)
    if values.shape != r.shape:
        raise ValueError(
            f"radial diagonal has shape {values.shape}; expected {r.shape}"
        )
    return values
