"""Wavefunction-storage estimates for non-BO diatomic calculations."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np

from .angular import AngularBasisSpec, AngularPairBasis, Parity
from .grids import FEMDVRGridSpec


FEMDVRGridInput: TypeAlias = (
    FEMDVRGridSpec | tuple[Sequence[float], int | Sequence[int]]
)
PairExchangeParity: TypeAlias = Parity | None | Literal["auto"]


@dataclass(frozen=True)
class WavefunctionStorageSize:
    """Storage occupied by one wavefunction array."""

    shape: tuple[int, ...]
    n_angular_pairs: int
    radial_shape: tuple[int, int]
    dtype: str
    itemsize: int
    n_coefficients: int
    n_bytes: int

    @property
    def gigabytes(self) -> float:
        """Size in decimal gigabytes (GB)."""

        return self.n_bytes / 1.0e9

    @property
    def gibibytes(self) -> float:
        """Size in binary gibibytes (GiB)."""

        return self.n_bytes / 1024**3


@dataclass(frozen=True)
class NBOWavefunctionSizeReport:
    """Storage estimates for production and unrestricted NBO states."""

    system: str
    l_max: int
    m_max: int
    M_tot: int | None
    inversion_parity: Parity
    pair_exchange_parity: Parity | None
    symmetry_adapted_basis: bool
    radial_shape: tuple[int, int]
    ground_state: WavefunctionStorageSize
    time_dependent_state: WavefunctionStorageSize
    unrestricted_real_state: WavefunctionStorageSize
    unrestricted_complex_state: WavefunctionStorageSize

    def __str__(self) -> str:
        lines = [
            f"System: {self.system}",
            f"Radial shape (r, R): {self.radial_shape}",
            f"Angular cutoffs: l_max={self.l_max}, m_max={self.m_max}",
            f"M_tot: {self.M_tot}",
        ]
        for label, size in (
            ("Real ground state", self.ground_state),
            ("Complex time-dependent state", self.time_dependent_state),
            ("Unrestricted real state", self.unrestricted_real_state),
            ("Unrestricted complex state", self.unrestricted_complex_state),
        ):
            lines.append(
                f"{label}: {size.n_coefficients} coefficients, "
                f"{size.gigabytes:.6g} GB ({size.gibibytes:.6g} GiB), "
                f"shape={size.shape}"
            )
        return "\n".join(lines)


def get_size_NBO_Psi(
    system: str,
    electronic_grid: FEMDVRGridInput,
    internuclear_grid: FEMDVRGridInput,
    *,
    l_max: int,
    m_max: int | None = None,
    M_tot: int | None = 0,
    inversion_parity: Parity = 1,
    pair_exchange_parity: PairExchangeParity = "auto",
    symmetry_adapted_basis: bool | None = True,
) -> NBOWavefunctionSizeReport:
    r"""Estimate the storage occupied by one non-BO diatomic wavefunction.

    ``electronic_grid`` and ``internuclear_grid`` may be existing
    :class:`FEMDVRGridSpec` objects or ``(nodes, n_points)`` tuples. In the
    tuple form, ``n_points`` may be one integer used for every element or a
    sequence with one entry per element.

    The result contains sizes for a real ``float64`` field-free ground state,
    a complex ``complex128`` z-polarized time-dependent state, and
    unrestricted real and complex reference arrays. The unrestricted
    references use ``M_tot=None`` and no coordinate-parity reduction.

    A z-polarized field preserves ``M_tot`` but couples even and odd
    electronic angular momenta. The time-dependent estimate therefore keeps
    both electronic parities through ``l_max``. For identical nuclei, pair
    exchange remains a symmetry and restricts the internuclear angular basis.

    Only permanent storage for one wavefunction is reported. Hamiltonian data,
    Krylov vectors, work arrays, MPI halos, and replicated or distributed
    operator storage are excluded.

    Examples
    --------
    >>> r_grid = FEMDVRGridSpec([0, 4, 8, 12, 16], [11] * 4)
    >>> R_grid = ([0, 2, 4, 6, 8], 21)
    >>> sizes = get_size_NBO_Psi(
    ...     "H2+", r_grid, R_grid, l_max=2, m_max=2
    ... )
    >>> sizes.ground_state.n_coefficients
    24648
    """

    system = _canonical_system(system)
    l_max = _nonnegative_integer(l_max, "l_max")
    if m_max is None:
        m_max = l_max
    else:
        m_max = _nonnegative_integer(m_max, "m_max")
    if m_max > l_max:
        raise ValueError("m_max must satisfy 0 <= m_max <= l_max")
    if M_tot is not None:
        if isinstance(M_tot, bool) or not isinstance(M_tot, Integral):
            raise TypeError("M_tot must be an integer or None")
        M_tot = int(M_tot)
    if inversion_parity not in (-1, 1):
        raise ValueError("inversion_parity must be +1 or -1")
    if symmetry_adapted_basis is not None and not isinstance(
        symmetry_adapted_basis, (bool, np.bool_)
    ):
        raise TypeError("symmetry_adapted_basis must be boolean or None")
    symmetry_adapted_basis = bool(symmetry_adapted_basis)

    identical_pair = system != "HD+"
    pair_exchange_parity = _resolved_pair_exchange_parity(
        system,
        identical_pair,
        pair_exchange_parity,
    )
    radial_shape = (
        _femdvr_interior_points(electronic_grid, "electronic_grid"),
        _femdvr_interior_points(internuclear_grid, "internuclear_grid"),
    )

    if symmetry_adapted_basis and identical_pair:
        ground_electron_parity = inversion_parity * pair_exchange_parity
        ground_pair_parity = pair_exchange_parity
    else:
        ground_electron_parity = None
        ground_pair_parity = None

    ground_basis1 = AngularBasisSpec(
        l_max,
        m_max,
        coordinate_parity=ground_electron_parity,
    )
    ground_basis2 = AngularBasisSpec(
        l_max,
        m_max,
        coordinate_parity=ground_pair_parity,
    )
    ground_pairs = AngularPairBasis(ground_basis1, ground_basis2, M_tot)

    time_basis1 = AngularBasisSpec(l_max, m_max)
    time_basis2 = AngularBasisSpec(
        l_max,
        m_max,
        coordinate_parity=(pair_exchange_parity if identical_pair else None),
    )
    time_pairs = AngularPairBasis(time_basis1, time_basis2, M_tot)

    full_basis1 = AngularBasisSpec(l_max, m_max)
    full_basis2 = AngularBasisSpec(l_max, m_max)
    full_pairs = AngularPairBasis(full_basis1, full_basis2, None)

    return NBOWavefunctionSizeReport(
        system=system,
        l_max=l_max,
        m_max=m_max,
        M_tot=M_tot,
        inversion_parity=inversion_parity,
        pair_exchange_parity=pair_exchange_parity,
        symmetry_adapted_basis=symmetry_adapted_basis,
        radial_shape=radial_shape,
        ground_state=_storage_size(
            ground_basis1,
            ground_basis2,
            ground_pairs,
            radial_shape,
            np.float64,
        ),
        time_dependent_state=_storage_size(
            time_basis1,
            time_basis2,
            time_pairs,
            radial_shape,
            np.complex128,
        ),
        unrestricted_real_state=_storage_size(
            full_basis1,
            full_basis2,
            full_pairs,
            radial_shape,
            np.float64,
        ),
        unrestricted_complex_state=_storage_size(
            full_basis1,
            full_basis2,
            full_pairs,
            radial_shape,
            np.complex128,
        ),
    )


def _storage_size(
    basis1: AngularBasisSpec,
    basis2: AngularBasisSpec,
    pairs: AngularPairBasis,
    radial_shape: tuple[int, int],
    dtype,
) -> WavefunctionStorageSize:
    dtype = np.dtype(dtype)
    if pairs.is_compact:
        shape = (pairs.n_pairs, *radial_shape)
    else:
        shape = (basis1.n_channels, basis2.n_channels, *radial_shape)
    n_coefficients = int(np.prod(shape, dtype=np.int64))
    n_bytes = n_coefficients * dtype.itemsize
    return WavefunctionStorageSize(
        shape=shape,
        n_angular_pairs=pairs.n_pairs,
        radial_shape=radial_shape,
        dtype=dtype.name,
        itemsize=dtype.itemsize,
        n_coefficients=n_coefficients,
        n_bytes=n_bytes,
    )


def _femdvr_interior_points(grid: FEMDVRGridInput, name: str) -> int:
    if isinstance(grid, FEMDVRGridSpec):
        specification = grid
    else:
        if not isinstance(grid, tuple) or len(grid) != 2:
            raise TypeError(
                f"{name} must be a FEMDVRGridSpec or a (nodes, n_points) tuple"
            )
        nodes, n_points = grid
        n_elements = len(nodes) - 1
        if isinstance(n_points, Integral) and not isinstance(n_points, bool):
            n_points = [int(n_points)] * n_elements
        specification = FEMDVRGridSpec(nodes, n_points)

    # Adjacent GLL elements share one endpoint, and the two outer Dirichlet
    # boundary points are not stored in a two-radial wavefunction.
    n_elements = len(specification.n_points)
    return sum(specification.n_points) - (n_elements - 1) - 2


def _canonical_system(system: str) -> str:
    if system not in ("H2+", "HD+", "D2+"):
        raise ValueError("system must be 'H2+', 'HD+', or 'D2+'")
    return system


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _resolved_pair_exchange_parity(
    system: str,
    identical_pair: bool,
    parity: PairExchangeParity,
) -> Parity | None:
    if parity == "auto":
        return 1 if identical_pair else None
    if identical_pair:
        if parity not in (-1, 1):
            raise ValueError(
                f"pair_exchange_parity must be +1 or -1 for {system}"
            )
        return parity
    if parity is not None:
        raise ValueError(f"pair_exchange_parity is not defined for {system}")
    return None
