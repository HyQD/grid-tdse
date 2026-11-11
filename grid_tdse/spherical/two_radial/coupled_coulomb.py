"""Sparse Coulomb couplings for coupled two-radial angular channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from grid_lib.spherical_coordinates import clebsch_gordan, wigner_6j
from grid_lib.spherical_coordinates.angular_momentum import (
    get_y_value,
    get_ybar_value,
)
from numba import njit, prange

from .angular import CoulombMultipoleSpec
from .coupled_angular import CoupledAngularBasisSpec, CoupledAngularState
from .system import CoupledTwoRadialSystem

CoupledCoulombCoefficientMethod = Literal["direct", "reduced"]


@dataclass(frozen=True)
class CoupledCoulombCouplings:
    """Sparse transitions grouped by coupled output channel."""

    output: np.ndarray
    offsets: np.ndarray
    input: np.ndarray
    radial_index: np.ndarray
    coefficient: np.ndarray
    n_channels: int

    def __post_init__(self) -> None:
        arrays = (
            self.output,
            self.offsets,
            self.input,
            self.radial_index,
            self.coefficient,
        )
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("coupled Coulomb arrays must be one-dimensional")
        n_groups = self.output.size
        n_pairs = self.coefficient.size
        if self.offsets.size != n_groups + 1:
            raise ValueError("invalid coupled Coulomb output-group arrays")
        if (
            self.input.size != n_pairs
            or self.radial_index.size != n_pairs
        ):
            raise ValueError("invalid coupled Coulomb transition arrays")
        if self.offsets[0] != 0 or self.offsets[-1] != n_pairs:
            raise ValueError(
                "coupled Coulomb offsets do not span all transitions"
            )
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("coupled Coulomb offsets must be nondecreasing")
        if self.n_channels < 1:
            raise ValueError("n_channels must be positive")
        if n_groups and (
            self.output.min() < 0 or self.output.max() >= self.n_channels
        ):
            raise ValueError("coupled Coulomb output channel is out of range")
        if n_groups > 1 and np.any(np.diff(self.output) <= 0):
            raise ValueError(
                "coupled Coulomb output channels must be strictly increasing"
            )
        if n_pairs and (
            self.input.min() < 0 or self.input.max() >= self.n_channels
        ):
            raise ValueError("coupled Coulomb input channel is out of range")

    @property
    def n_pairs(self) -> int:
        """Return the number of retained channel transitions."""

        return self.coefficient.size


def coupled_coulomb_coefficient_direct(
    output_state: CoupledAngularState,
    input_state: CoupledAngularState,
    lambda_: int,
) -> float:
    r"""Evaluate one coefficient by its defining Clebsch--Gordan sum.

    This deliberately direct implementation is the convention oracle.  It
    uses the same one-coordinate ``Y`` and ``Ybar`` matrix elements as the
    uncoupled Coulomb builder.
    """

    l1_out, l2_out, L_out, M_out = output_state
    l1_in, l2_in, L_in, M_in = input_state
    if L_out != L_in or M_out != M_in:
        return 0.0

    value = 0.0
    for m1_out in range(-l1_out, l1_out + 1):
        m2_out = M_out - m1_out
        if abs(m2_out) > l2_out:
            continue
        cg_out = clebsch_gordan(
            l1_out,
            m1_out,
            l2_out,
            m2_out,
            L_out,
            M_out,
        )
        if cg_out == 0.0:
            continue

        for m1_in in range(-l1_in, l1_in + 1):
            m2_in = M_in - m1_in
            if abs(m2_in) > l2_in:
                continue
            mu = m1_in - m1_out
            if abs(mu) > lambda_:
                continue
            cg_in = clebsch_gordan(
                l1_in,
                m1_in,
                l2_in,
                m2_in,
                L_in,
                M_in,
            )
            if cg_in == 0.0:
                continue
            value += (
                cg_out
                * cg_in
                * get_ybar_value(
                    lambda_,
                    mu,
                    l1_out,
                    m1_out,
                    l1_in,
                    m1_in,
                )
                * get_y_value(
                    lambda_,
                    mu,
                    l2_out,
                    m2_out,
                    l2_in,
                    m2_in,
                )
            )
    return float(value)


def coupled_coulomb_coefficient_reduced(
    output_state: CoupledAngularState,
    input_state: CoupledAngularState,
    lambda_: int,
) -> float:
    r"""Evaluate one scalar Coulomb coefficient with a Wigner 6-j symbol."""

    l1_out, l2_out, L_out, M_out = output_state
    l1_in, l2_in, L_in, M_in = input_state
    if L_out != L_in or M_out != M_in:
        return 0.0
    if not _one_coordinate_allowed(l1_out, l1_in, lambda_):
        return 0.0
    if not _one_coordinate_allowed(l2_out, l2_in, lambda_):
        return 0.0

    reduced1 = _reduced_spherical_harmonic(l1_out, l1_in, lambda_)
    reduced2 = _reduced_spherical_harmonic(l2_out, l2_in, lambda_)
    phase = -1.0 if (l1_in + l2_out + L_out) % 2 else 1.0
    return float(
        phase
        * wigner_6j(
            l2_in,
            l1_in,
            L_in,
            l1_out,
            l2_out,
            lambda_,
        )
        * reduced1
        * reduced2
    )


def build_coupled_coulomb_couplings(
    basis: CoupledAngularBasisSpec,
    multipoles: CoulombMultipoleSpec,
    *,
    coefficient_method: CoupledCoulombCoefficientMethod = "reduced",
    tolerance: float = 1.0e-12,
    max_pairs: int = 20_000_000,
) -> CoupledCoulombCouplings:
    """Build a sparse, output-grouped scalar Coulomb transition table."""

    if not isinstance(basis, CoupledAngularBasisSpec):
        raise TypeError("basis must be a CoupledAngularBasisSpec")
    if not isinstance(multipoles, CoulombMultipoleSpec):
        raise TypeError("multipoles must be a CoulombMultipoleSpec")
    if multipoles.mu_max < max(multipoles.lambdas):
        raise ValueError(
            "coupled scalar Coulomb interactions require mu_max to cover "
            "the largest retained lambda"
        )
    if coefficient_method not in {"direct", "reduced"}:
        raise ValueError("coefficient_method must be 'direct' or 'reduced'")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")

    coefficient_function = (
        coupled_coulomb_coefficient_direct
        if coefficient_method == "direct"
        else coupled_coulomb_coefficient_reduced
    )
    lambda_indices = {
        lambda_: radial_index
        for radial_index, lambda_ in enumerate(multipoles.lambdas)
    }
    rows: list[tuple[int, int, int, float]] = []

    for output, output_state in enumerate(basis.states):
        l1_out, l2_out, L_out, M_out = output_state
        for input_, input_state in enumerate(basis.states):
            l1_in, l2_in, L_in, M_in = input_state
            if L_out != L_in or M_out != M_in:
                continue
            for lambda_ in multipoles.lambdas:
                if not _one_coordinate_allowed(l1_out, l1_in, lambda_):
                    continue
                if not _one_coordinate_allowed(l2_out, l2_in, lambda_):
                    continue
                coefficient = coefficient_function(
                    output_state,
                    input_state,
                    lambda_,
                )
                if abs(coefficient) <= tolerance:
                    continue
                rows.append(
                    (
                        output,
                        input_,
                        lambda_indices[lambda_],
                        coefficient,
                    )
                )
                if len(rows) > max_pairs:
                    raise ValueError(
                        f"more than {max_pairs} coupled Coulomb transitions; "
                        "reduce the angular truncation or raise max_pairs"
                    )

    rows.sort(key=lambda row: (row[0], row[2], row[1]))
    if not rows:
        empty = _readonly([], dtype=np.int32)
        return CoupledCoulombCouplings(
            output=empty,
            offsets=_readonly([0], dtype=np.int64),
            input=empty,
            radial_index=empty,
            coefficient=_readonly([], dtype=float),
            n_channels=basis.n_channels,
        )

    values = np.asarray(rows)
    output = values[:, 0].astype(np.int32)
    starts = np.flatnonzero(np.r_[True, output[1:] != output[:-1]])
    return CoupledCoulombCouplings(
        output=_readonly(output[starts], dtype=np.int32),
        offsets=_readonly(np.r_[starts, len(rows)], dtype=np.int64),
        input=_readonly(values[:, 1], dtype=np.int32),
        radial_index=_readonly(values[:, 2], dtype=np.int32),
        coefficient=_readonly(values[:, 3]),
        n_channels=basis.n_channels,
    )


class CoupledTwoRadialCoulombHamiltonian:
    """Apply sparse coupled-channel Coulomb transitions pointwise in radius."""

    def __init__(
        self,
        representation: CoupledTwoRadialSystem,
        *,
        couplings: CoupledCoulombCouplings | None = None,
        coefficient_method: CoupledCoulombCoefficientMethod = "reduced",
        tolerance: float = 1.0e-12,
        max_pairs: int = 20_000_000,
    ) -> None:
        if not isinstance(representation, CoupledTwoRadialSystem):
            raise TypeError("representation must be a CoupledTwoRadialSystem")
        if couplings is None:
            couplings = build_coupled_coulomb_couplings(
                representation.angular_basis,
                representation.multipoles,
                coefficient_method=coefficient_method,
                tolerance=tolerance,
                max_pairs=max_pairs,
            )
        elif not isinstance(couplings, CoupledCoulombCouplings):
            raise TypeError("couplings must be CoupledCoulombCouplings")
        if couplings.n_channels != representation.angular_basis.n_channels:
            raise ValueError(
                "coupled Coulomb channel count disagrees with representation"
            )
        if couplings.n_pairs and (
            couplings.radial_index.min() < 0
            or couplings.radial_index.max()
            >= representation.radial_multipoles.values.shape[0]
        ):
            raise ValueError(
                "coupled Coulomb transition references a missing radial multipole"
            )

        self.representation = representation
        self.couplings = couplings
        self._radial_flat = np.ascontiguousarray(
            representation.radial_multipoles.values.reshape(
                representation.radial_multipoles.values.shape[0],
                -1,
            )
        )

    @property
    def psi_shape(self) -> tuple[int, int, int]:
        return self.representation.psi_shape

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Apply the electron--nuclear Coulomb interaction."""

        return self.apply(state)

    def apply(
        self,
        state: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the interaction, optionally accumulating into ``out``."""

        state = self.representation.validate_state(state)
        dtype = np.result_type(
            state.dtype,
            self._radial_flat.dtype,
            self.couplings.coefficient.dtype,
        )
        if out is None:
            out = np.zeros(state.shape, dtype=dtype)
        elif out.shape != state.shape:
            raise ValueError(
                f"out has shape {out.shape}; expected {state.shape}"
            )
        elif not np.can_cast(dtype, out.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot accumulate {dtype} Coulomb action in "
                f"{out.dtype} output"
            )
        elif not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")

        _apply_coupled_coulomb_kernel(
            out.reshape(state.shape[0], -1),
            state.reshape(state.shape[0], -1),
            self._radial_flat,
            self.couplings.output,
            self.couplings.offsets,
            self.couplings.input,
            self.couplings.radial_index,
            self.couplings.coefficient,
        )
        return out


def _one_coordinate_allowed(l_out: int, l_in: int, lambda_: int) -> bool:
    return (
        abs(l_out - l_in) <= lambda_ <= l_out + l_in
        and (l_out + l_in + lambda_) % 2 == 0
    )


def _reduced_spherical_harmonic(
    l_out: int,
    l_in: int,
    lambda_: int,
) -> float:
    cg_zero = clebsch_gordan(l_in, 0, lambda_, 0, l_out, 0)
    if cg_zero == 0.0:
        return 0.0
    return float(
        get_y_value(lambda_, 0, l_out, 0, l_in, 0)
        * np.sqrt(2 * l_out + 1)
        / cg_zero
    )


def _readonly(values, *, dtype=None) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    result.setflags(write=False)
    return result


@njit(cache=True, parallel=True)
def _apply_coupled_coulomb_kernel(
    output,
    wavefunction,
    radial_multipoles,
    output_channel,
    offsets,
    input_channel,
    radial_index,
    coefficient,
):
    for group in prange(output_channel.size):
        target = output_channel[group]
        for pair in range(offsets[group], offsets[group + 1]):
            source = input_channel[pair]
            radial = radial_multipoles[radial_index[pair]]
            value = coefficient[pair]
            for point in range(wavefunction.shape[1]):
                output[target, point] += (
                    value * radial[point] * wavefunction[source, point]
                )
