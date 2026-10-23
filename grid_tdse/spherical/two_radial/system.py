import numpy as np

from dataclasses import dataclass, field
from typing import Callable, Protocol

from grid_lib.pseudospectral_grids.sinc_dvr import RadialSincDVR
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    get_y,
    get_ybar,
    number_of_lm_states,
)


class RadialDiagonalTerm(Protocol):
    """Term diagonal in the angular basis and on the radial DVR grid."""

    def diagonal(self, l: int, m: int, grid: "RadialGrid") -> np.ndarray:
        """Return the diagonal radial values for one angular channel."""


@dataclass(frozen=True)
class RadialGrid:
    """Concrete radial grid data used by the two-radial setup."""

    dvr: object
    r: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    weights: np.ndarray

    @property
    def n_points(self) -> int:
        return self.r.size


@dataclass(frozen=True)
class RadialGridSpec:
    """Specification for one radial DVR grid."""

    r_max: float
    n_points: int
    grid_cls: type = RadialSincDVR

    def build(self) -> RadialGrid:
        dvr = self.grid_cls(self.r_max, self.n_points)
        return RadialGrid(
            dvr=dvr,
            r=dvr.r,
            D1=dvr.D1,
            D2=dvr.D2,
            weights=dvr.weights,
        )


@dataclass(frozen=True)
class AngularBasisSpec:
    """Angular momentum truncation for one radial coordinate."""

    l_max: int
    m_max: int | None = None

    def __post_init__(self):
        m_max = self.l_max if self.m_max is None else self.m_max
        if self.l_max < 0:
            raise ValueError("l_max must be non-negative")
        if m_max < 0:
            raise ValueError("m_max must be non-negative")
        if m_max > self.l_max:
            raise ValueError("m_max cannot exceed l_max")
        object.__setattr__(self, "m_max", m_max)

    @property
    def n_lm(self) -> int:
        return number_of_lm_states(self.l_max, self.m_max)

    def iter_lm(self):
        for m in range(-self.m_max, self.m_max + 1):
            for l in range(abs(m), self.l_max + 1):
                yield l, m

    def lm_index(self, l: int, m: int) -> int:
        return LM_to_I(l, m, self.l_max, self.m_max)


@dataclass(frozen=True)
class RadialPotential:
    """Radial potential shared by all angular channels."""

    values: float | np.ndarray | Callable[[np.ndarray], np.ndarray]

    def diagonal(self, l: int, m: int, grid: RadialGrid) -> np.ndarray:
        del l, m
        return _as_radial_values(self.values, grid.r)


@dataclass(frozen=True)
class LMRadialPotential:
    """Radial potential that may depend on the angular channel."""

    values: Callable[[int, int, np.ndarray], np.ndarray]

    def diagonal(self, l: int, m: int, grid: RadialGrid) -> np.ndarray:
        return _as_radial_values(self.values(l, m, grid.r), grid.r)


@dataclass(frozen=True)
class CoordinateSpec:
    """One radial coordinate, its angular basis, and one-body operators."""

    grid: RadialGridSpec
    basis: AngularBasisSpec
    kinetic_prefactor: float
    one_body_terms: tuple[RadialDiagonalTerm, ...] = field(default_factory=tuple)

    def build_one_body_operator(self, grid: RadialGrid | None = None) -> np.ndarray:
        grid = self.grid.build() if grid is None else grid
        operator = np.zeros(
            (self.basis.n_lm, grid.n_points, grid.n_points), dtype=np.float64
        )

        for l, m in self.basis.iter_lm():
            radial_laplacian = grid.D2 - np.diag(l * (l + 1) / grid.r**2)
            block = self.kinetic_prefactor * radial_laplacian
            for term in self.one_body_terms:
                block = block + np.diag(term.diagonal(l, m, grid))
            operator[self.basis.lm_index(l, m)] = block

        return operator


@dataclass(frozen=True)
class TwoRadialSystemSpec:
    """Complete setup recipe for a two-radial Hamiltonian.

    ``W_LM`` should follow the current Hamiltonian convention: it is indexed as
    ``(LM, r1, r2)`` and includes the second-coordinate quadrature weights.
    """

    coord1: CoordinateSpec
    coord2: CoordinateSpec
    W_LM: np.ndarray
    L_max: int | None = None
    M_max: int | None = None
    interaction_sign: float = 1.0
    frame: str = "nuclear_center"

    def build(self) -> "TwoRadialSystem":
        grid1 = self.coord1.grid.build()
        grid2 = self.coord2.grid.build()

        L_max = (
            self.coord1.basis.l_max + self.coord2.basis.l_max
            if self.L_max is None
            else self.L_max
        )
        M_max = L_max if self.M_max is None else self.M_max
        _validate_lm_cutoffs(L_max, M_max)

        W_LM = np.asarray(self.W_LM)
        expected_shape = (
            number_of_lm_states(L_max, M_max),
            grid1.n_points,
            grid2.n_points,
        )
        if W_LM.shape != expected_shape:
            raise ValueError(f"W_LM has shape {W_LM.shape}; expected {expected_shape}")

        return TwoRadialSystem(
            grid1=grid1,
            grid2=grid2,
            basis1=self.coord1.basis,
            basis2=self.coord2.basis,
            h1=self.coord1.build_one_body_operator(grid1),
            h2=self.coord2.build_one_body_operator(grid2),
            W_LM=W_LM,
            y=get_y(self.coord2.basis.l_max, self.coord2.basis.m_max, L_max, M_max),
            y_bar=get_ybar(
                self.coord1.basis.l_max, self.coord1.basis.m_max, L_max, M_max
            ),
            L_max=L_max,
            M_max=M_max,
            interaction_sign=self.interaction_sign,
            frame=self.frame,
        )


@dataclass(frozen=True)
class TwoRadialSystem:
    """Prepared two-radial system arrays."""

    grid1: RadialGrid
    grid2: RadialGrid
    basis1: AngularBasisSpec
    basis2: AngularBasisSpec
    h1: np.ndarray
    h2: np.ndarray
    W_LM: np.ndarray
    y: np.ndarray
    y_bar: np.ndarray
    L_max: int
    M_max: int
    interaction_sign: float = 1.0
    frame: str = "nuclear_center"

    @property
    def psi_shape(self) -> tuple[int, int, int, int]:
        return (
            self.basis1.n_lm,
            self.basis2.n_lm,
            self.grid1.n_points,
            self.grid2.n_points,
        )

    def hamiltonian(self, OPT_SWITCH: bool = True):
        from .hamiltonian import Hamiltonian

        return Hamiltonian(
            self.h1,
            self.h2,
            self.W_LM,
            self.y,
            self.y_bar,
            self.L_max,
            self.M_max,
            self.grid1.weights,
            self.grid2.weights,
            frame=self.frame,
            OPT_SWITCH=OPT_SWITCH,
        )


@dataclass(frozen=True)
class CoulombCenter:
    """One Coulomb center placed along the second radial direction."""

    charge_product: float
    radial_scale: float = 1.0
    angular_parity: int = 1

    def __post_init__(self):
        if self.radial_scale < 0:
            raise ValueError("radial_scale must be non-negative")
        if self.angular_parity not in (-1, 1):
            raise ValueError("angular_parity must be either -1 or 1")


def radial_potential(
    values: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
) -> RadialPotential:
    return RadialPotential(values)


def lm_radial_potential(
    values: Callable[[int, int, np.ndarray], np.ndarray],
) -> LMRadialPotential:
    return LMRadialPotential(values)


def coulomb_w_lm(
    r: np.ndarray,
    R: np.ndarray,
    L_max: int,
    M_max: int,
    centers: tuple[CoulombCenter, ...] = (CoulombCenter(1.0),),
    weights_R: np.ndarray | None = None,
) -> np.ndarray:
    """Build multipole-expanded Coulomb interaction as a signed W_LM array.

    Pass ``weights_R`` to get a Hamiltonian-ready array. If ``weights_R`` is
    omitted, the returned array contains raw radial multipoles.
    """

    _validate_lm_cutoffs(L_max, M_max)
    L_vec = np.arange(L_max + 1)[:, None, None]
    W_L = np.zeros((L_max + 1, r.size, R.size), dtype=np.float64)
    r_grid = r[:, None]

    for center in centers:
        center_R = center.radial_scale * R[None, :]
        r_min = np.minimum(r_grid, center_R)[None, :, :]
        r_max = np.maximum(r_grid, center_R)[None, :, :]
        base = (
            4 * np.pi / (2 * L_vec + 1) * (r_min**L_vec) / (r_max ** (L_vec + 1))
        )
        parity = center.angular_parity ** np.arange(L_max + 1)
        W_L += center.charge_product * parity[:, None, None] * base

    if weights_R is not None:
        W_L = W_L * weights_R[None, None, :]

    return expand_L_to_LM(W_L, L_max, M_max)


def expand_L_to_LM(W_L: np.ndarray, L_max: int, M_max: int) -> np.ndarray:
    """Copy L-resolved radial multipoles into all valid LM rows."""

    _validate_lm_cutoffs(L_max, M_max)
    W_L = np.asarray(W_L)
    if W_L.ndim != 3:
        raise ValueError(f"W_L must be three-dimensional; got shape {W_L.shape}")
    expected_shape = (L_max + 1, W_L.shape[1], W_L.shape[2])
    if W_L.shape != expected_shape:
        raise ValueError(f"W_L has shape {W_L.shape}; expected {expected_shape}")

    W_LM = np.zeros(
        (number_of_lm_states(L_max, M_max), W_L.shape[1], W_L.shape[2]),
        dtype=W_L.dtype,
    )
    for M in range(-M_max, M_max + 1):
        for L in range(abs(M), L_max + 1):
            W_LM[LM_to_I(L, M, L_max, M_max)] = W_L[L]
    return W_LM


def _as_radial_values(values, r: np.ndarray) -> np.ndarray:
    if callable(values):
        values = values(r)

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        return np.full_like(r, values, dtype=np.float64)
    if values.shape != r.shape:
        raise ValueError(f"radial diagonal has shape {values.shape}; expected {r.shape}")
    return values


def _validate_lm_cutoffs(L_max: int, M_max: int) -> None:
    if L_max < 0:
        raise ValueError("L_max must be non-negative")
    if M_max < 0:
        raise ValueError("M_max must be non-negative")
    if M_max > L_max:
        raise ValueError("M_max cannot exceed L_max")
