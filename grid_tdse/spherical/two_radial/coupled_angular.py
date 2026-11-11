"""Coupled angular-channel tables for two-radial spherical calculations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from grid_lib.spherical_coordinates import clebsch_gordan

from .angular import AngularBasisSpec, AngularPairBasis, Parity

CoupledAngularState = tuple[int, int, int, int]
UncoupledAngularState = tuple[int, int, int, int]

_METADATA_TYPE = "coupled-bipolar-spherical-harmonics"
_METADATA_VERSION = 1


@dataclass(frozen=True)
class CoupledAngularBasisSpec:
    r"""Immutable ``(l_1, l_2, L, M)`` coupled-channel specification.

    ``M_tot`` selects one total laboratory-frame projection.  Optional
    inversion and pair-exchange parities retain channels satisfying
    ``(-1)**(l_1 + l_2)`` and ``(-1)**l_2``, respectively.

    The class describes angular channels only.  Use
    :meth:`for_diatomic_ion` when the availability of nuclear-exchange
    symmetry must be validated from a physical diatomic system.
    """

    l1_max: int
    l2_max: int
    L_max: int
    M_tot: int = 0
    inversion_parity: Parity | None = None
    pair_exchange_parity: Parity | None = None
    states: tuple[CoupledAngularState, ...] = field(init=False)
    _indices: dict[CoupledAngularState, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        l1_max = _nonnegative_integer(self.l1_max, "l1_max")
        l2_max = _nonnegative_integer(self.l2_max, "l2_max")
        L_max = _nonnegative_integer(self.L_max, "L_max")
        M_tot = _integer(self.M_tot, "M_tot")
        inversion_parity = _parity(self.inversion_parity, "inversion_parity")
        pair_exchange_parity = _parity(
            self.pair_exchange_parity,
            "pair_exchange_parity",
        )

        if L_max > l1_max + l2_max:
            raise ValueError("L_max must not exceed l1_max + l2_max")
        if abs(M_tot) > L_max:
            raise ValueError("abs(M_tot) must not exceed L_max")

        states = tuple(
            (l1, l2, L, M_tot)
            for l1 in range(l1_max + 1)
            for l2 in range(l2_max + 1)
            for L in range(abs(l1 - l2), min(l1 + l2, L_max) + 1)
            if abs(M_tot) <= L
            if inversion_parity is None or _angular_parity(l1 + l2) == inversion_parity
            if pair_exchange_parity is None
            or _angular_parity(l2) == pair_exchange_parity
        )
        if not states:
            raise ValueError("angular selections leave no coupled channels")

        object.__setattr__(self, "l1_max", l1_max)
        object.__setattr__(self, "l2_max", l2_max)
        object.__setattr__(self, "L_max", L_max)
        object.__setattr__(self, "M_tot", M_tot)
        object.__setattr__(self, "inversion_parity", inversion_parity)
        object.__setattr__(self, "pair_exchange_parity", pair_exchange_parity)
        object.__setattr__(self, "states", states)
        object.__setattr__(
            self,
            "_indices",
            {state: index for index, state in enumerate(states)},
        )

    @classmethod
    def for_diatomic_ion(
        cls,
        l1_max: int,
        l2_max: int,
        L_max: int,
        *,
        has_identical_pair: bool,
        M_tot: int = 0,
        inversion_parity: Parity | None = None,
        pair_exchange_parity: Parity | None = None,
    ) -> CoupledAngularBasisSpec:
        """Construct channels with physical nuclear-exchange validation."""

        if not isinstance(has_identical_pair, bool):
            raise TypeError("has_identical_pair must be a bool")
        if not has_identical_pair and pair_exchange_parity is not None:
            raise ValueError(
                "pair_exchange_parity is unavailable for distinguishable nuclei"
            )
        return cls(
            l1_max=l1_max,
            l2_max=l2_max,
            L_max=L_max,
            M_tot=M_tot,
            inversion_parity=inversion_parity,
            pair_exchange_parity=pair_exchange_parity,
        )

    @property
    def n_channels(self) -> int:
        """Return the number of retained coupled channels."""

        return len(self.states)

    def index(self, l1: int, l2: int, L: int, M: int) -> int:
        """Return the index of a retained state or raise ``ValueError``."""

        state = (l1, l2, L, M)
        try:
            return self._indices[state]
        except KeyError as exc:
            raise ValueError(f"coupled angular state {state} is not retained") from exc

    def get(self, l1: int, l2: int, L: int, M: int) -> int | None:
        """Return the index of a retained state, or ``None`` when absent."""

        return self._indices.get((l1, l2, L, M))

    def to_metadata(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible channel description."""

        return {
            "type": _METADATA_TYPE,
            "version": _METADATA_VERSION,
            "l1_max": self.l1_max,
            "l2_max": self.l2_max,
            "L_max": self.L_max,
            "M_tot": self.M_tot,
            "inversion_parity": self.inversion_parity,
            "pair_exchange_parity": self.pair_exchange_parity,
            "states": [list(state) for state in self.states],
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> CoupledAngularBasisSpec:
        """Reconstruct and verify a serialized channel description."""

        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if metadata.get("type") != _METADATA_TYPE:
            raise ValueError("metadata does not describe a coupled angular basis")
        if metadata.get("version") != _METADATA_VERSION:
            raise ValueError("unsupported coupled angular metadata version")

        required = (
            "l1_max",
            "l2_max",
            "L_max",
            "M_tot",
            "inversion_parity",
            "pair_exchange_parity",
            "states",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise ValueError(
                "coupled angular metadata is missing " + ", ".join(missing)
            )

        result = cls(
            l1_max=metadata["l1_max"],
            l2_max=metadata["l2_max"],
            L_max=metadata["L_max"],
            M_tot=metadata["M_tot"],
            inversion_parity=metadata["inversion_parity"],
            pair_exchange_parity=metadata["pair_exchange_parity"],
        )
        stored_states = _metadata_states(metadata["states"])
        if stored_states != result.states:
            raise ValueError(
                "stored coupled angular states disagree with the specified cutoffs"
            )
        return result


@dataclass(frozen=True)
class CoupledAngularProjection:
    """Result of projecting an uncoupled array into a coupled basis.

    The reported norms are Euclidean sums over the angular and all trailing
    axes.  Radial quadrature weights can be applied independently when a
    physical two-radial norm is required.
    """

    coefficients: np.ndarray
    input_norm_squared: float
    retained_norm_squared: float
    discarded_norm_squared: float

    @property
    def discarded_fraction(self) -> float:
        """Return the fraction of input norm removed by the projection."""

        if self.input_norm_squared == 0.0:
            return 0.0
        return self.discarded_norm_squared / self.input_norm_squared


@dataclass(frozen=True)
class CoupledAngularTransform:
    r"""Fixed-``M`` Clebsch--Gordan transformation oracle.

    The coefficient matrix has rows labelled by uncoupled states
    ``(l_1, m_1, l_2, m_2)`` and columns labelled by the coupled states in
    :class:`CoupledAngularBasisSpec`.  Its entries are

    ``<l_1 m_1, l_2 m_2 | L M>``.

    Arrays use the angular channel as their first axis and may have any number
    of trailing axes.  The dense matrix is intentionally an oracle for tests
    and validation; production Hamiltonian paths should use sparse reduced
    coupling tables instead.
    """

    basis: CoupledAngularBasisSpec
    uncoupled_states: tuple[UncoupledAngularState, ...] = field(init=False)
    coefficients: np.ndarray = field(init=False, repr=False)
    _uncoupled_indices: dict[UncoupledAngularState, int] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.basis, CoupledAngularBasisSpec):
            raise TypeError("basis must be a CoupledAngularBasisSpec")

        basis1 = AngularBasisSpec(self.basis.l1_max)
        basis2 = AngularBasisSpec(self.basis.l2_max)
        pair_basis = AngularPairBasis(basis1, basis2, self.basis.M_tot)
        states = tuple(
            (l1, m1, l2, m2)
            for channel1, channel2 in pair_basis.pairs
            for l1, m1 in (basis1.states[channel1],)
            for l2, m2 in (basis2.states[channel2],)
            if self.basis.inversion_parity is None
            or _angular_parity(l1 + l2) == self.basis.inversion_parity
            if self.basis.pair_exchange_parity is None
            or _angular_parity(l2) == self.basis.pair_exchange_parity
        )
        indices = {state: index for index, state in enumerate(states)}
        coefficients = np.zeros((len(states), self.basis.n_channels))

        for coupled, (l1, l2, L, M) in enumerate(self.basis.states):
            for m1 in range(-l1, l1 + 1):
                m2 = M - m1
                uncoupled = indices.get((l1, m1, l2, m2))
                if uncoupled is not None:
                    coefficients[uncoupled, coupled] = clebsch_gordan(
                        l1,
                        m1,
                        l2,
                        m2,
                        L,
                        M,
                    )

        coefficients.setflags(write=False)
        object.__setattr__(self, "uncoupled_states", states)
        object.__setattr__(self, "_uncoupled_indices", indices)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def n_uncoupled(self) -> int:
        """Return the number of retained fixed-``M`` uncoupled states."""

        return len(self.uncoupled_states)

    @property
    def n_coupled(self) -> int:
        """Return the number of retained coupled channels."""

        return self.basis.n_channels

    @property
    def is_complete(self) -> bool:
        """Whether the coupled basis spans the selected uncoupled space."""

        return self.n_coupled == self.n_uncoupled

    def uncoupled_index(self, l1: int, m1: int, l2: int, m2: int) -> int:
        """Return an uncoupled state index or raise ``ValueError``."""

        state = (l1, m1, l2, m2)
        try:
            return self._uncoupled_indices[state]
        except KeyError as exc:
            raise ValueError(
                f"uncoupled angular state {state} is not retained"
            ) from exc

    def to_uncoupled(self, values: np.ndarray) -> np.ndarray:
        """Transform coupled coefficients to the fixed-``M`` uncoupled basis."""

        values = _angular_array(values, self.n_coupled, "coupled")
        return np.ascontiguousarray(
            np.tensordot(self.coefficients, values, axes=(1, 0))
        )

    def to_coupled(self, values: np.ndarray) -> np.ndarray:
        """Apply the adjoint transform to fixed-``M`` uncoupled coefficients."""

        values = _angular_array(values, self.n_uncoupled, "uncoupled")
        return np.ascontiguousarray(
            np.tensordot(self.coefficients.conj().T, values, axes=(1, 0))
        )

    def project(self, values: np.ndarray) -> CoupledAngularProjection:
        """Project uncoupled values and report the norm discarded by ``L``."""

        values = _angular_array(values, self.n_uncoupled, "uncoupled")
        coupled = self.to_coupled(values)
        retained = self.to_uncoupled(coupled)
        residual = values - retained
        input_norm_squared = _norm_squared(values)
        retained_norm_squared = _norm_squared(retained)
        discarded_norm_squared = _norm_squared(residual)
        return CoupledAngularProjection(
            coefficients=coupled,
            input_norm_squared=input_norm_squared,
            retained_norm_squared=retained_norm_squared,
            discarded_norm_squared=discarded_norm_squared,
        )


def _metadata_states(values: Any) -> tuple[CoupledAngularState, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError("metadata states must be a sequence")
    states = []
    for state in values:
        if not isinstance(state, (list, tuple)) or len(state) != 4:
            raise ValueError("each metadata state must contain four integers")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in state
        ):
            raise TypeError("metadata state values must be integers")
        states.append(tuple(int(value) for value in state))
    return tuple(states)


def _nonnegative_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _parity(value: Any, name: str) -> Parity | None:
    if value not in (None, -1, 1) or isinstance(value, bool):
        raise ValueError(f"{name} must be None, -1, or 1")
    return value


def _angular_parity(angular_momentum: int) -> Parity:
    return -1 if angular_momentum % 2 else 1


def _angular_array(values: Any, n_channels: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim == 0 or result.shape[0] != n_channels:
        raise ValueError(
            f"{name} values have shape {result.shape}; expected first axis "
            f"of length {n_channels}"
        )
    return result


def _norm_squared(values: np.ndarray) -> float:
    return float(np.vdot(values, values).real)
