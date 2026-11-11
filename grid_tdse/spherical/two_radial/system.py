"""Preparation of grid and angular data for two-radial Hamiltonians."""

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np

from .angular import AngularBasisSpec, AngularPairBasis, CoulombMultipoleSpec
from .coulomb import (
    CoulombTerm,
    RadialCoulombMethod,
    RadialCoulombMultipoles,
    build_radial_coulomb_multipoles,
)
from .couplings import CoulombCouplings, build_coulomb_couplings
from .coupled_angular import CoupledAngularBasisSpec
from .grids import RadialGrid, RadialGridSpec, require_same_grid_family

RadialOperatorMode = Literal["global", "element-local"]
CoulombKernelMode = Literal["reference", "multipole-grouped"]
RadialKernelMode = Literal["reference", "row-wise"]


@runtime_checkable
class TwoRadialRepresentation(Protocol):
    """Representation-independent serial two-radial state contract."""

    grid1: RadialGrid
    grid2: RadialGrid

    @property
    def psi_shape(self) -> tuple[int, ...]: ...

    def inner_product(self, left: np.ndarray, right: np.ndarray) -> complex: ...


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

    def build_one_body_diagonal(
        self, grid: RadialGrid | None = None
    ) -> np.ndarray:
        """Return centrifugal and potential terms for every angular channel."""

        grid = self.build_grid() if grid is None else grid
        diagonal = np.empty((self.basis.n_channels, grid.n_points), dtype=float)
        for channel, (l, m) in enumerate(self.basis.states):
            values = -self.kinetic_prefactor * l * (l + 1) / grid.r**2
            for term in self.one_body_terms:
                values = values + term.diagonal(l, m, grid)
            diagonal[channel] = values
        return diagonal

    def build_one_body_operator(
        self,
        grid: RadialGrid | None = None,
        diagonal: np.ndarray | None = None,
    ) -> np.ndarray:
        grid = self.build_grid() if grid is None else grid
        if diagonal is None:
            diagonal = self.build_one_body_diagonal(grid)
        diagonal = np.asarray(diagonal)
        expected_shape = (self.basis.n_channels, grid.n_points)
        if diagonal.shape != expected_shape:
            raise ValueError(
                f"diagonal has shape {diagonal.shape}; expected {expected_shape}"
            )
        operator = np.zeros(
            (self.basis.n_channels, grid.n_points, grid.n_points), dtype=float
        )
        for channel, (l, _) in enumerate(self.basis.states):
            operator[channel] = self.kinetic_prefactor * grid.second_derivative(l)
            operator[channel] += np.diag(diagonal[channel])
        return operator

    def build_one_body_channel(
        self,
        l: int,
        m: int,
        grid: RadialGrid | None = None,
        diagonal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one dense radial block without constructing all channels."""

        grid = self.build_grid() if grid is None else grid
        channel = self.basis.index(l, m)
        if diagonal is None:
            values = self.build_one_body_diagonal(grid)[channel]
        else:
            diagonal = np.asarray(diagonal)
            expected_shape = (self.basis.n_channels, grid.n_points)
            if diagonal.shape != expected_shape:
                raise ValueError(
                    f"diagonal has shape {diagonal.shape}; expected "
                    f"{expected_shape}"
                )
            values = diagonal[channel]
        operator = self.kinetic_prefactor * grid.second_derivative(l)
        return operator + np.diag(values)


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

    def build(
        self,
        *,
        radial_operator_mode: RadialOperatorMode = "global",
    ) -> "TwoRadialSystem":
        """Prepare a representation for global or element-local radial actions.

        The lightweight ``"element-local"`` path retains the grids and
        channel diagonals required by FEM-DVR element actions, but never
        constructs the dense per-channel one-body matrices.
        """

        if radial_operator_mode not in {"global", "element-local"}:
            raise ValueError(
                "radial_operator_mode must be 'global' or 'element-local'"
            )
        grid1 = self.coordinate1.build_grid()
        grid2 = self.coordinate2.build_grid()
        require_same_grid_family(grid1, grid2)
        if radial_operator_mode == "element-local" and grid1.family != "femdvr":
            raise ValueError(
                "element-local radial operators require FEM-DVR grids"
            )
        h1_diagonal = self.coordinate1.build_one_body_diagonal(grid1)
        h2_diagonal = self.coordinate2.build_one_body_diagonal(grid2)
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
        retain_global_operators = radial_operator_mode == "global"
        return TwoRadialSystem(
            grid1=grid1,
            grid2=grid2,
            basis1=self.coordinate1.basis,
            basis2=self.coordinate2.basis,
            angular_pairs=angular_pairs,
            h1=(
                self.coordinate1.build_one_body_operator(grid1, h1_diagonal)
                if retain_global_operators
                else None
            ),
            h2=(
                self.coordinate2.build_one_body_operator(grid2, h2_diagonal)
                if retain_global_operators
                else None
            ),
            multipoles=self.multipoles,
            radial_multipoles=radial_multipoles,
            couplings=couplings,
            kinetic_prefactor1=self.coordinate1.kinetic_prefactor,
            kinetic_prefactor2=self.coordinate2.kinetic_prefactor,
            h1_diagonal=h1_diagonal,
            h2_diagonal=h2_diagonal,
            radial_operator_mode=radial_operator_mode,
        )

    def build_coupled(
        self,
        angular_basis: CoupledAngularBasisSpec,
    ) -> "CoupledTwoRadialSystem":
        """Prepare radial data for a coupled-angular representation.

        This path deliberately constructs neither uncoupled angular couplings
        nor one-body Hamiltonian matrices.  Those actions are introduced in
        later coupled-representation implementation steps.
        """

        if not isinstance(angular_basis, CoupledAngularBasisSpec):
            raise TypeError("angular_basis must be a CoupledAngularBasisSpec")
        if self.coordinate1.basis.l_max != angular_basis.l1_max:
            raise ValueError("coordinate1 l_max must equal angular_basis.l1_max")
        if self.coordinate2.basis.l_max != angular_basis.l2_max:
            raise ValueError("coordinate2 l_max must equal angular_basis.l2_max")
        if self.M_tot not in (None, angular_basis.M_tot):
            raise ValueError("system M_tot disagrees with angular_basis.M_tot")

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
        return CoupledTwoRadialSystem(
            grid1=grid1,
            grid2=grid2,
            angular_basis=angular_basis,
            multipoles=self.multipoles,
            radial_multipoles=radial_multipoles,
            coordinate1=self.coordinate1,
            coordinate2=self.coordinate2,
        )


@dataclass(frozen=True)
class CoupledTwoRadialSystem:
    """Prepared radial data and state operations for coupled angular channels.

    Serial states have shape ``(coupled_channel, r_1, r_2)``.  The object does
    not retain a dense coupled/uncoupled transformation or any Hamiltonian
    matrices.
    """

    grid1: RadialGrid
    grid2: RadialGrid
    angular_basis: CoupledAngularBasisSpec
    multipoles: CoulombMultipoleSpec
    radial_multipoles: RadialCoulombMultipoles
    coordinate1: CoordinateSpec
    coordinate2: CoordinateSpec

    def __post_init__(self) -> None:
        if not isinstance(self.angular_basis, CoupledAngularBasisSpec):
            raise TypeError("angular_basis must be a CoupledAngularBasisSpec")
        require_same_grid_family(self.grid1, self.grid2)
        if self.grid1.l_max != self.angular_basis.l1_max:
            raise ValueError("grid1 l_max disagrees with angular_basis.l1_max")
        if self.grid2.l_max != self.angular_basis.l2_max:
            raise ValueError("grid2 l_max disagrees with angular_basis.l2_max")
        if self.coordinate1.basis.l_max != self.angular_basis.l1_max:
            raise ValueError(
                "coordinate1 l_max disagrees with angular_basis.l1_max"
            )
        if self.coordinate2.basis.l_max != self.angular_basis.l2_max:
            raise ValueError(
                "coordinate2 l_max disagrees with angular_basis.l2_max"
            )
        expected_radial_shape = (self.grid1.n_points, self.grid2.n_points)
        if self.radial_multipoles.values.shape[1:] != expected_radial_shape:
            raise ValueError("radial multipoles do not match the radial grids")
        if self.radial_multipoles.lambdas != self.multipoles.lambdas:
            raise ValueError("radial multipoles disagree with multipole selection")

    @property
    def M_tot(self) -> int:
        return self.angular_basis.M_tot

    @property
    def psi_shape(self) -> tuple[int, int, int]:
        return (
            self.angular_basis.n_channels,
            self.grid1.n_points,
            self.grid2.n_points,
        )

    @property
    def radial_shape(self) -> tuple[int, int]:
        return (self.grid1.n_points, self.grid2.n_points)

    def validate_state(self, state: np.ndarray, *, name: str = "state") -> np.ndarray:
        """Return an array view after validating coupled state storage."""

        values = np.asarray(state)
        if values.shape != self.psi_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected {self.psi_shape}"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"{name} must have a numerical dtype")
        return values

    def zeros(self, dtype=np.complex128) -> np.ndarray:
        """Allocate a zero state in coupled serial storage."""

        return np.zeros(self.psi_shape, dtype=dtype)

    def random_state(
        self,
        rng: np.random.Generator | None = None,
        *,
        dtype=np.complex128,
        normalize: bool = False,
    ) -> np.ndarray:
        """Create a random real or complex state, optionally normalized."""

        rng = np.random.default_rng() if rng is None else rng
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            values = rng.normal(size=self.psi_shape) + 1j * rng.normal(
                size=self.psi_shape
            )
        elif np.issubdtype(dtype, np.floating):
            values = rng.normal(size=self.psi_shape)
        else:
            raise TypeError("random state dtype must be real or complex floating")
        result = np.asarray(values, dtype=dtype)
        if normalize:
            result /= self.norm(result)
        return result

    def copy_state(self, state: np.ndarray) -> np.ndarray:
        """Return an independent contiguous copy of a validated state."""

        return np.ascontiguousarray(self.validate_state(state).copy())

    def inner_product(self, left: np.ndarray, right: np.ndarray) -> complex:
        """Return the quadrature-weighted two-radial inner product."""

        left = self.validate_state(left, name="left state")
        right = self.validate_state(right, name="right state")
        return np.einsum(
            "cab,cab,a,b->",
            left.conj(),
            right,
            self.grid1.weights,
            self.grid2.weights,
            optimize=True,
        )

    def norm(self, state: np.ndarray) -> float:
        """Return the physical quadrature norm of a coupled state."""

        value = float(self.inner_product(state, state).real)
        return float(np.sqrt(max(value, 0.0)))

    def radial_marginals(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return pointwise radial probability densities for both coordinates."""

        values = self.validate_state(state)
        population = np.einsum("cab,cab->ab", values.conj(), values).real
        density1 = np.einsum("ab,b->a", population, self.grid2.weights, optimize=True)
        density2 = np.einsum("ab,a->b", population, self.grid1.weights, optimize=True)
        return density1, density2

    def state_nbytes(self, dtype=np.complex128) -> int:
        """Return the storage required by one state with the given dtype."""

        return int(np.prod(self.psi_shape, dtype=np.int64) * np.dtype(dtype).itemsize)

    def to_metadata(self) -> dict[str, object]:
        """Return deterministic JSON-compatible representation metadata."""

        return {
            "type": "coupled-two-radial-system",
            "version": 1,
            "angular_basis": self.angular_basis.to_metadata(),
            "grid_family": self.grid1.family,
            "radial_shape": list(self.radial_shape),
            "psi_shape": list(self.psi_shape),
            "multipoles": {
                "lambda_max": self.multipoles.lambda_max,
                "mu_max": self.multipoles.mu_max,
                "lambda_parity": self.multipoles.lambda_parity,
                "lambdas": list(self.multipoles.lambdas),
            },
            "radial_multipole_representation": self.radial_multipoles.representation,
        }

    def one_body_hamiltonian(
        self,
        *,
        radial_operator_mode: RadialOperatorMode = "element-local",
    ):
        """Build the coupled field-free one-body action."""

        from .coupled_hamiltonian import CoupledTwoRadialOneBodyHamiltonian

        return CoupledTwoRadialOneBodyHamiltonian(
            self,
            radial_operator_mode=radial_operator_mode,
        )

    def coulomb_hamiltonian(
        self,
        *,
        coefficient_method="reduced",
        tolerance: float = 1.0e-12,
        max_pairs: int = 20_000_000,
    ):
        """Build the coupled electron--nuclear Coulomb action."""

        from .coupled_coulomb import CoupledTwoRadialCoulombHamiltonian

        return CoupledTwoRadialCoulombHamiltonian(
            self,
            coefficient_method=coefficient_method,
            tolerance=tolerance,
            max_pairs=max_pairs,
        )


@dataclass(frozen=True)
class TwoRadialSystem:
    """Prepared arrays and sparse couplings for a two-radial system."""

    grid1: RadialGrid
    grid2: RadialGrid
    basis1: AngularBasisSpec
    basis2: AngularBasisSpec
    h1: np.ndarray | None
    h2: np.ndarray | None
    multipoles: CoulombMultipoleSpec
    radial_multipoles: RadialCoulombMultipoles
    couplings: CoulombCouplings
    angular_pairs: AngularPairBasis | None = None
    kinetic_prefactor1: float | None = None
    kinetic_prefactor2: float | None = None
    h1_diagonal: np.ndarray | None = None
    h2_diagonal: np.ndarray | None = None
    radial_operator_mode: RadialOperatorMode = "global"

    def __post_init__(self) -> None:
        if self.radial_operator_mode not in {"global", "element-local"}:
            raise ValueError(
                "radial_operator_mode must be 'global' or 'element-local'"
            )
        if (self.h1 is None) != (self.h2 is None):
            raise ValueError(
                "h1 and h2 must either both be present or both be absent"
            )
        if self.radial_operator_mode == "global" and self.h1 is None:
            raise ValueError("global radial operators require both h1 and h2")
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
    def has_global_radial_operators(self) -> bool:
        """Whether dense per-channel one-body matrices are retained."""

        return self.h1 is not None

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

    def hamiltonian(
        self,
        *,
        tile_size: int = 2048,
        radial_operator_mode: RadialOperatorMode | None = None,
        coulomb_kernel: CoulombKernelMode = "reference",
        radial_kernel: RadialKernelMode = "reference",
    ):
        from .hamiltonian import TwoRadialHamiltonian

        if radial_operator_mode is None:
            radial_operator_mode = self.radial_operator_mode
        if (
            radial_operator_mode == "global"
            and not self.has_global_radial_operators
        ):
            raise ValueError(
                "global radial operators were not retained; rebuild the system "
                "with radial_operator_mode='global'"
            )

        return TwoRadialHamiltonian(
            self.h1,
            self.h2,
            self.radial_multipoles,
            self.couplings,
            angular_pairs=self.angular_pairs,
            grid1=self.grid1,
            grid2=self.grid2,
            kinetic_prefactor1=self.kinetic_prefactor1,
            kinetic_prefactor2=self.kinetic_prefactor2,
            h1_diagonal=self.h1_diagonal,
            h2_diagonal=self.h2_diagonal,
            radial_operator_mode=radial_operator_mode,
            tile_size=tile_size,
            coulomb_kernel=coulomb_kernel,
            radial_kernel=radial_kernel,
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
