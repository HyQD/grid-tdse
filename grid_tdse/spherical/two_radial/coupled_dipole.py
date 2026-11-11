"""Rank-one dipole couplings for coupled two-radial angular channels."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from grid_lib.spherical_coordinates import clebsch_gordan, wigner_6j
from grid_lib.spherical_coordinates.angular_momentum import get_y_value
from numba import njit, prange

from .coupled_angular import CoupledAngularBasisSpec, CoupledAngularState


CoupledDipoleCoefficientMethod = Literal["direct", "reduced"]
CoupledDipoleCoordinate = Literal[1, 2]

_C1_NORMALIZATION = np.sqrt(4.0 * np.pi / 3.0)


@dataclass(frozen=True)
class CoupledDipoleCouplings:
    r"""Sparse rank-one transitions grouped by coupled output channel.

    The coefficients include the coordinate's effective charge but not its
    radial factor or the overall length-gauge interaction sign.  Thus Step 9
    can form ``-xi_r r C_0^(1)`` or ``-xi_R R C_0^(1)`` without rebuilding
    the angular table.
    """

    output: np.ndarray
    offsets: np.ndarray
    input: np.ndarray
    coefficient: np.ndarray
    n_output_channels: int
    n_input_channels: int
    coordinate: CoupledDipoleCoordinate
    q: int
    coupling: float

    def __post_init__(self) -> None:
        arrays = self.output, self.offsets, self.input, self.coefficient
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("coupled dipole arrays must be one-dimensional")
        n_groups = self.output.size
        n_pairs = self.coefficient.size
        if self.offsets.size != n_groups + 1:
            raise ValueError("invalid coupled dipole output-group arrays")
        if self.input.size != n_pairs:
            raise ValueError("invalid coupled dipole transition arrays")
        if self.offsets[0] != 0 or self.offsets[-1] != n_pairs:
            raise ValueError(
                "coupled dipole offsets do not span all transitions"
            )
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("coupled dipole offsets must be nondecreasing")
        if self.n_output_channels < 1 or self.n_input_channels < 1:
            raise ValueError("coupled dipole channel counts must be positive")
        if n_groups and (
            self.output.min() < 0
            or self.output.max() >= self.n_output_channels
        ):
            raise ValueError("coupled dipole output channel is out of range")
        if n_groups > 1 and np.any(np.diff(self.output) <= 0):
            raise ValueError(
                "coupled dipole output channels must be strictly increasing"
            )
        if n_pairs and (
            self.input.min() < 0
            or self.input.max() >= self.n_input_channels
        ):
            raise ValueError("coupled dipole input channel is out of range")
        _coordinate(self.coordinate)
        _q(self.q)
        _finite_real(self.coupling, "coupling")

    @property
    def n_pairs(self) -> int:
        """Return the number of retained channel transitions."""

        return self.coefficient.size

    def apply(
        self,
        state: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the angular table to arrays with a leading channel axis."""

        values = np.asarray(state)
        if values.ndim < 1 or values.shape[0] != self.n_input_channels:
            raise ValueError(
                "state must have the coupled input channel as its first axis"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("state must contain numeric values")
        output_shape = (self.n_output_channels, *values.shape[1:])
        dtype = np.result_type(values.dtype, self.coefficient.dtype)
        if out is None:
            out = np.zeros(output_shape, dtype=dtype)
        elif out.shape != output_shape:
            raise ValueError(
                f"out has shape {out.shape}; expected {output_shape}"
            )
        elif not np.can_cast(dtype, out.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot accumulate {dtype} dipole action in {out.dtype} output"
            )
        elif not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")

        _apply_coupled_dipole_kernel(
            out.reshape(self.n_output_channels, -1),
            values.reshape(self.n_input_channels, -1),
            self.output,
            self.offsets,
            self.input,
            self.coefficient,
        )
        return out

    __call__ = apply


def coupled_dipole_coefficient_direct(
    output_state: CoupledAngularState,
    input_state: CoupledAngularState,
    coordinate: CoupledDipoleCoordinate,
    *,
    q: int = 0,
) -> float:
    r"""Evaluate one ``C_q^(1)`` coefficient by explicit CG summation."""

    coordinate = _coordinate(coordinate)
    q = _q(q)
    l1_out, l2_out, L_out, M_out = output_state
    l1_in, l2_in, L_in, M_in = input_state
    if M_out != M_in + q:
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
            if coordinate == 1:
                if l2_out != l2_in or m2_out != m2_in:
                    continue
                one_coordinate = _C1_NORMALIZATION * get_y_value(
                    1,
                    q,
                    l1_out,
                    m1_out,
                    l1_in,
                    m1_in,
                )
            else:
                if l1_out != l1_in or m1_out != m1_in:
                    continue
                one_coordinate = _C1_NORMALIZATION * get_y_value(
                    1,
                    q,
                    l2_out,
                    m2_out,
                    l2_in,
                    m2_in,
                )
            value += cg_out * cg_in * one_coordinate
    return float(value)


def coupled_dipole_coefficient_reduced(
    output_state: CoupledAngularState,
    input_state: CoupledAngularState,
    coordinate: CoupledDipoleCoordinate,
    *,
    q: int = 0,
) -> float:
    r"""Evaluate one ``C_q^(1)`` coefficient with a Wigner 6-j symbol."""

    coordinate = _coordinate(coordinate)
    q = _q(q)
    l1_out, l2_out, L_out, M_out = output_state
    l1_in, l2_in, L_in, M_in = input_state
    if M_out != M_in + q:
        return 0.0
    total_cg = clebsch_gordan(L_in, M_in, 1, q, L_out, M_out)
    if total_cg == 0.0:
        return 0.0

    if coordinate == 1:
        if l2_out != l2_in or abs(l1_out - l1_in) != 1:
            return 0.0
        phase = -1.0 if (l1_out + l2_out + L_in + 1) % 2 else 1.0
        six_j = wigner_6j(
            l1_out,
            L_out,
            l2_out,
            L_in,
            l1_in,
            1,
        )
        reduced = _reduced_c1(l1_out, l1_in)
    else:
        if l1_out != l1_in or abs(l2_out - l2_in) != 1:
            return 0.0
        phase = -1.0 if (l1_out + l2_in + L_out + 1) % 2 else 1.0
        six_j = wigner_6j(
            l2_out,
            L_out,
            l1_out,
            L_in,
            l2_in,
            1,
        )
        reduced = _reduced_c1(l2_out, l2_in)

    return float(
        total_cg
        / np.sqrt(2 * L_out + 1)
        * phase
        * np.sqrt((2 * L_out + 1) * (2 * L_in + 1))
        * six_j
        * reduced
    )


def build_coupled_dipole_couplings(
    input_basis: CoupledAngularBasisSpec,
    coordinate: CoupledDipoleCoordinate,
    coupling: float,
    *,
    q: int = 0,
    output_basis: CoupledAngularBasisSpec | None = None,
    coefficient_method: CoupledDipoleCoefficientMethod = "reduced",
    tolerance: float = 1.0e-12,
    max_pairs: int = 20_000_000,
) -> CoupledDipoleCouplings:
    """Build a sparse, output-grouped rank-one dipole transition table."""

    if not isinstance(input_basis, CoupledAngularBasisSpec):
        raise TypeError("input_basis must be a CoupledAngularBasisSpec")
    if output_basis is None:
        output_basis = input_basis
    elif not isinstance(output_basis, CoupledAngularBasisSpec):
        raise TypeError("output_basis must be a CoupledAngularBasisSpec")
    coordinate = _coordinate(coordinate)
    q = _q(q)
    coupling = _finite_real(coupling, "coupling")
    if output_basis.M_tot != input_basis.M_tot + q:
        raise ValueError("output M_tot must equal input M_tot + q")
    if coefficient_method not in {"direct", "reduced"}:
        raise ValueError("coefficient_method must be 'direct' or 'reduced'")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")

    coefficient_function = (
        coupled_dipole_coefficient_direct
        if coefficient_method == "direct"
        else coupled_dipole_coefficient_reduced
    )
    rows: list[tuple[int, int, float]] = []
    if coupling != 0.0:
        for output, output_state in enumerate(output_basis.states):
            for input_, input_state in enumerate(input_basis.states):
                coefficient = coupling * coefficient_function(
                    output_state,
                    input_state,
                    coordinate,
                    q=q,
                )
                if abs(coefficient) <= tolerance:
                    continue
                rows.append((output, input_, coefficient))
                if len(rows) > max_pairs:
                    raise ValueError(
                        f"more than {max_pairs} coupled dipole transitions; "
                        "reduce the angular truncation or raise max_pairs"
                    )

    if not rows:
        empty = _readonly([], dtype=np.int32)
        return CoupledDipoleCouplings(
            output=empty,
            offsets=_readonly([0], dtype=np.int64),
            input=empty,
            coefficient=_readonly([], dtype=float),
            n_output_channels=output_basis.n_channels,
            n_input_channels=input_basis.n_channels,
            coordinate=coordinate,
            q=q,
            coupling=coupling,
        )

    rows.sort(key=lambda row: (row[0], row[1]))
    values = np.asarray(rows)
    output = values[:, 0].astype(np.int32)
    starts = np.flatnonzero(np.r_[True, output[1:] != output[:-1]])
    return CoupledDipoleCouplings(
        output=_readonly(output[starts], dtype=np.int32),
        offsets=_readonly(np.r_[starts, len(rows)], dtype=np.int64),
        input=_readonly(values[:, 1], dtype=np.int32),
        coefficient=_readonly(values[:, 2]),
        n_output_channels=output_basis.n_channels,
        n_input_channels=input_basis.n_channels,
        coordinate=coordinate,
        q=q,
        coupling=coupling,
    )


def _reduced_c1(l_out: int, l_in: int) -> float:
    cg_zero = clebsch_gordan(l_in, 0, 1, 0, l_out, 0)
    if cg_zero == 0.0:
        return 0.0
    return float(
        _C1_NORMALIZATION
        * get_y_value(1, 0, l_out, 0, l_in, 0)
        * np.sqrt(2 * l_out + 1)
        / cg_zero
    )


def _coordinate(value) -> CoupledDipoleCoordinate:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("coordinate must be an integer")
    result = int(value)
    if result not in (1, 2):
        raise ValueError("coordinate must be 1 or 2")
    return result


def _q(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("q must be an integer")
    result = int(value)
    if result not in (-1, 0, 1):
        raise ValueError("q must be -1, 0, or 1")
    return result


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _readonly(values, *, dtype=None) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    result.setflags(write=False)
    return result


@njit(cache=True, parallel=True)
def _apply_coupled_dipole_kernel(
    result,
    state,
    output_channel,
    offsets,
    input_channel,
    coefficient,
):
    for group in prange(output_channel.size):
        target = output_channel[group]
        for point in range(state.shape[1]):
            value = state[0, point] * 0.0
            for pair in range(offsets[group], offsets[group + 1]):
                value += coefficient[pair] * state[input_channel[pair], point]
            result[target, point] += value
