"""Rank-one dipole couplings for coupled two-radial angular channels."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from grid_lib.spherical_coordinates import clebsch_gordan, wigner_6j
from grid_lib.spherical_coordinates.angular_momentum import get_y_value
from numba import njit, prange

from .coupled_angular import CoupledAngularBasisSpec, CoupledAngularState
from .dipole import (
    AxialDipoleCouplings,
    DipoleGauge,
    _field_profiles,
    _gauge,
    _radial_operator_mode,
)
from .system import CoupledTwoRadialSystem, RadialOperatorMode


CoupledDipoleCoefficientMethod = Literal["direct", "reduced"]
CoupledDipoleCoordinate = Literal[1, 2]

_C1_NORMALIZATION = np.sqrt(4.0 * np.pi / 3.0)


@dataclass(frozen=True)
class CoupledDipoleCouplings:
    r"""Sparse rank-one transitions grouped by coupled output channel.

    The coefficients include the coordinate's effective charge but not its
    radial factor or the overall length-gauge interaction sign. This lets the
    coordinate operator form ``-xi_r r C_0^(1)`` or
    ``-xi_R R C_0^(1)`` without rebuilding the angular table.
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


class CoupledAxialCoordinateDipole:
    r"""Apply one coordinate's coupled-angular axial interaction.

    The state axes are ``(coupled_channel, radial1, radial2)``.  For a
    coordinate coupling ``c``, this operator applies ``-c z`` in length
    gauge and ``-c p_z`` in velocity gauge. It excludes the temporal field
    amplitude in both cases.
    """

    def __init__(
        self,
        representation: CoupledTwoRadialSystem,
        coordinate: CoupledDipoleCoordinate,
        *,
        gauge: DipoleGauge = "length",
        coupling: float,
        radial_operator_mode: RadialOperatorMode = "element-local",
        couplings: CoupledDipoleCouplings | None = None,
        coefficient_method: CoupledDipoleCoefficientMethod = "reduced",
        tolerance: float = 1.0e-12,
        max_pairs: int = 20_000_000,
    ) -> None:
        if not isinstance(representation, CoupledTwoRadialSystem):
            raise TypeError("representation must be a CoupledTwoRadialSystem")
        coordinate = _coordinate(coordinate)
        gauge = _gauge(gauge)
        coupling = _finite_real(coupling, "coupling")
        radial_operator_mode = _radial_operator_mode(radial_operator_mode)
        basis = representation.angular_basis
        if coupling != 0.0 and basis.inversion_parity is not None:
            raise ValueError(
                "a field-coupled basis must retain both inversion parities"
            )
        if (
            coupling != 0.0
            and coordinate == 2
            and basis.pair_exchange_parity is not None
        ):
            raise ValueError(
                "a field-coupled internuclear coordinate must retain both "
                "pair-exchange parities"
            )
        if couplings is None:
            couplings = build_coupled_dipole_couplings(
                basis,
                coordinate,
                coupling,
                coefficient_method=coefficient_method,
                tolerance=tolerance,
                max_pairs=max_pairs,
            )
        else:
            _validate_coordinate_couplings(
                couplings,
                basis,
                coordinate,
                coupling,
            )

        self.representation = representation
        self.coordinate = coordinate
        self.gauge = gauge
        self.coupling = coupling
        self.radial_operator_mode = radial_operator_mode
        self.couplings = couplings
        self.radial = (
            representation.grid1.r
            if coordinate == 1
            else representation.grid2.r
        )
        self.grid = (
            representation.grid1 if coordinate == 1 else representation.grid2
        )
        self._velocity_angular2 = (
            None
            if gauge == "length"
            else _velocity_angular2_coefficients(
                couplings,
                basis,
                coordinate,
            )
        )

    def apply(
        self,
        state: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the field-independent coordinate term, accumulating in out."""

        values = self.representation.validate_state(state)
        dtype = np.result_type(
            values.dtype,
            self.couplings.coefficient.dtype,
            complex if self.gauge == "velocity" else float,
        )
        if out is None:
            out = np.zeros(self.representation.psi_shape, dtype=dtype)
        elif out.shape != self.representation.psi_shape:
            raise ValueError(
                f"out has shape {out.shape}; expected "
                f"{self.representation.psi_shape}"
            )
        elif not np.can_cast(dtype, out.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot accumulate {dtype} dipole action in {out.dtype} output"
            )
        elif not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")

        if self.gauge == "length":
            _apply_coupled_length_dipole_kernel(
                out,
                values,
                self.radial,
                self.coordinate,
                self.couplings.output,
                self.couplings.offsets,
                self.couplings.input,
                self.couplings.coefficient,
            )
            return out

        divided = values / (
            self.radial[None, :, None]
            if self.coordinate == 1
            else self.radial[None, None, :]
        )
        if self.radial_operator_mode == "element-local":
            derivative = self.grid.apply_local_first_derivative(
                values,
                axis=self.coordinate,
            )
        elif self.coordinate == 1:
            derivative = np.einsum(
                "ij,cjk->cik",
                self.grid.first_derivative(0),
                values,
                optimize=True,
            )
        else:
            derivative = np.einsum(
                "ij,crj->cri",
                self.grid.first_derivative(0),
                values,
                optimize=True,
            )
        _apply_coupled_velocity_dipole_kernel(
            out,
            derivative - divided,
            divided,
            self.couplings.output,
            self.couplings.offsets,
            self.couplings.input,
            self.couplings.coefficient,
            self._velocity_angular2,
        )
        return out

    def apply_length_gauge_radial_tile(
        self,
        state: np.ndarray,
        radial: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the length-gauge operator on an owned radial tile."""

        if self.gauge != "length":
            raise ValueError("operator is not configured for length gauge")
        values, radial, out = self._validate_radial_tile(
            state, radial, out, complex_output=False
        )
        _apply_coupled_length_dipole_kernel(
            out,
            values,
            radial,
            self.coordinate,
            self.couplings.output,
            self.couplings.offsets,
            self.couplings.input,
            self.couplings.coefficient,
        )
        return out

    def apply_velocity_gauge_radial_tile(
        self,
        state: np.ndarray,
        radial: np.ndarray,
        derivative: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the velocity-gauge operator on an owned radial tile."""

        if self.gauge != "velocity":
            raise ValueError("operator is not configured for velocity gauge")
        values, radial, out = self._validate_radial_tile(
            state, radial, out, complex_output=True
        )
        derivative = np.asarray(derivative)
        if derivative.shape != values.shape:
            raise ValueError("derivative shape must match the radial tile")
        divided = values / (
            radial[None, :, None]
            if self.coordinate == 1
            else radial[None, None, :]
        )
        _apply_coupled_velocity_dipole_kernel(
            out,
            derivative - divided,
            divided,
            self.couplings.output,
            self.couplings.offsets,
            self.couplings.input,
            self.couplings.coefficient,
            self._velocity_angular2,
        )
        return out

    def _validate_radial_tile(self, state, radial, out, *, complex_output):
        values = np.asarray(state)
        if values.ndim != 3 or values.shape[0] != self.couplings.n_input_channels:
            raise ValueError(
                "state must have shape (n_channels, local_nr, local_nR)"
            )
        radial = np.ascontiguousarray(np.asarray(radial))
        if radial.ndim != 1 or radial.size != values.shape[self.coordinate]:
            raise ValueError("radial coordinates do not match the tile")
        dtype = np.result_type(
            values.dtype,
            self.couplings.coefficient.dtype,
            complex if complex_output else float,
        )
        if out is None:
            out = np.zeros(values.shape, dtype=dtype)
        elif out.shape != values.shape:
            raise ValueError(f"out has shape {out.shape}; expected {values.shape}")
        elif not np.can_cast(dtype, out.dtype, casting="same_kind"):
            raise TypeError(f"cannot accumulate {dtype} dipole action in {out.dtype}")
        elif not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
        return values, radial, out

    __call__ = apply


class CoupledTwoRadialAxialDipoleInteraction:
    r"""Apply a serial coupled-angular, z-polarized dipole interaction.

    Length-gauge profiles return ``E_z(t)`` and the interaction is
    ``-E_z(t) * (xi_r z_r + xi_R z_R)``. Velocity-gauge profiles return
    ``A_z(t)`` and the interaction is
    ``-A_z(t) * (nu_r p_z,r + nu_R p_z,R)``.

    The spatially uniform quadratic vector-potential term is omitted because
    it changes only the global phase.
    """

    def __init__(
        self,
        representation: CoupledTwoRadialSystem,
        *,
        gauge: DipoleGauge,
        fields: Callable[[float], Real] | Iterable[Callable[[float], Real]],
        couplings: AxialDipoleCouplings,
        radial_operator_mode: RadialOperatorMode = "element-local",
        coefficient_method: CoupledDipoleCoefficientMethod = "reduced",
        coupling_tolerance: float = 1.0e-12,
        max_coupling_pairs: int = 20_000_000,
    ) -> None:
        if not isinstance(representation, CoupledTwoRadialSystem):
            raise TypeError("representation must be a CoupledTwoRadialSystem")
        gauge = _gauge(gauge)
        if not isinstance(couplings, AxialDipoleCouplings):
            raise TypeError("couplings must be an AxialDipoleCouplings")

        self.representation = representation
        self.gauge = gauge
        self.couplings = couplings
        self.radial_operator_mode = _radial_operator_mode(radial_operator_mode)
        self.fields = _field_profiles(fields, gauge)
        coordinate_operators = []
        for coordinate, coupling in (
            (1, couplings.coordinate1),
            (2, couplings.coordinate2),
        ):
            if coupling != 0.0:
                coordinate_operators.append(
                    CoupledAxialCoordinateDipole(
                        representation,
                        coordinate,
                        gauge=gauge,
                        coupling=coupling,
                        radial_operator_mode=self.radial_operator_mode,
                        coefficient_method=coefficient_method,
                        tolerance=coupling_tolerance,
                        max_pairs=max_coupling_pairs,
                    )
                )
        if not coordinate_operators:
            raise ValueError("at least one dipole coupling must be nonzero")
        self.coordinate_operators = tuple(coordinate_operators)

    @classmethod
    def for_diatomic_ion(
        cls,
        representation: CoupledTwoRadialSystem,
        parameters,
        *,
        gauge: DipoleGauge,
        fields: Callable[[float], Real] | Iterable[Callable[[float], Real]],
        radial_operator_mode: RadialOperatorMode = "element-local",
        coefficient_method: CoupledDipoleCoefficientMethod = "reduced",
        coupling_tolerance: float = 1.0e-12,
        max_coupling_pairs: int = 20_000_000,
    ) -> CoupledTwoRadialAxialDipoleInteraction:
        """Construct a non-BO diatomic-ion interaction."""

        return cls(
            representation,
            gauge=gauge,
            fields=fields,
            couplings=AxialDipoleCouplings.for_diatomic_ion(
                parameters,
                gauge,
            ),
            radial_operator_mode=radial_operator_mode,
            coefficient_method=coefficient_method,
            coupling_tolerance=coupling_tolerance,
            max_coupling_pairs=max_coupling_pairs,
        )

    def field_amplitude(self, time: float) -> float:
        """Evaluate and sum the scalar z-polarized electric fields."""

        amplitude = 0.0
        for index, field in enumerate(self.fields):
            value = np.asarray(field(time))
            if (
                value.ndim != 0
                or not np.issubdtype(value.dtype, np.number)
                or np.iscomplexobj(value)
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"field {index} must return a finite real scalar"
                )
            amplitude += float(value)
        return amplitude

    def apply(self, state: np.ndarray, time: float) -> np.ndarray:
        """Return the complete time-dependent dipole action."""

        values = self.representation.validate_state(state)
        amplitude = self.field_amplitude(time)
        dtype = np.result_type(
            values.dtype,
            complex if self.gauge == "velocity" else float,
        )
        if amplitude == 0.0:
            return np.zeros(self.representation.psi_shape, dtype=dtype)
        return amplitude * self._spatial_action_values(values, dtype)

    def spatial_action(self, state: np.ndarray) -> np.ndarray:
        """Return the field-independent coupled dipole action."""

        values = self.representation.validate_state(state)
        dtype = np.result_type(
            values.dtype,
            complex if self.gauge == "velocity" else float,
        )
        return self._spatial_action_values(values, dtype)

    def _spatial_action_values(self, values, dtype):
        result = np.zeros(self.representation.psi_shape, dtype=dtype)
        for operator in self.coordinate_operators:
            operator.apply(values, out=result)
        return result

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


def _validate_coordinate_couplings(
    couplings,
    basis: CoupledAngularBasisSpec,
    coordinate: CoupledDipoleCoordinate,
    coupling: float,
) -> None:
    if not isinstance(couplings, CoupledDipoleCouplings):
        raise TypeError("couplings must be a CoupledDipoleCouplings")
    if couplings.coordinate != coordinate:
        raise ValueError("couplings use the wrong radial coordinate")
    if couplings.q != 0:
        raise ValueError("an axial interaction requires q=0 couplings")
    if (
        couplings.n_input_channels != basis.n_channels
        or couplings.n_output_channels != basis.n_channels
    ):
        raise ValueError("coupled dipole channel count disagrees with basis")
    if couplings.coupling != coupling:
        raise ValueError("couplings use a different effective charge")


def _velocity_angular2_coefficients(
    couplings: CoupledDipoleCouplings,
    basis: CoupledAngularBasisSpec,
    coordinate: CoupledDipoleCoordinate,
) -> np.ndarray:
    """Return the angular coefficients multiplying the radial ``1/r`` term."""

    result = np.empty(couplings.n_pairs, dtype=couplings.coefficient.dtype)
    angular_index = coordinate - 1
    for group, output in enumerate(couplings.output):
        l_out = basis.states[int(output)][angular_index]
        for pair in range(couplings.offsets[group], couplings.offsets[group + 1]):
            l_in = basis.states[int(couplings.input[pair])][angular_index]
            if l_out == l_in + 1:
                factor = l_in
            elif l_out == l_in - 1:
                factor = -(l_in + 1)
            else:  # defensive: rank-one coordinate tables require delta l = +/-1
                raise ValueError(
                    "velocity-gauge coupling does not connect adjacent l values"
                )
            result[pair] = factor * couplings.coefficient[pair]
    return np.ascontiguousarray(result)


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


@njit(cache=True, parallel=True)
def _apply_coupled_length_dipole_kernel(
    result,
    state,
    radial,
    coordinate,
    output_channel,
    offsets,
    input_channel,
    coefficient,
):
    n_radial1 = state.shape[1]
    n_radial2 = state.shape[2]
    for group in prange(output_channel.size):
        target = output_channel[group]
        for radial1 in range(n_radial1):
            for radial2 in range(n_radial2):
                value = state[0, radial1, radial2] * 0.0
                for pair in range(offsets[group], offsets[group + 1]):
                    value += (
                        coefficient[pair]
                        * state[input_channel[pair], radial1, radial2]
                    )
                radial_value = (
                    radial[radial1] if coordinate == 1 else radial[radial2]
                )
                result[target, radial1, radial2] -= radial_value * value


@njit(parallel=True, cache=True)
def _apply_coupled_velocity_dipole_kernel(
    result,
    derivative_tilde,
    divided,
    output_channel,
    output_offsets,
    input_channel,
    angular1,
    angular2,
):
    n_radial1 = derivative_tilde.shape[1]
    n_radial2 = derivative_tilde.shape[2]
    for group in prange(output_channel.size):
        target = output_channel[group]
        for radial1 in range(n_radial1):
            for radial2 in range(n_radial2):
                value = derivative_tilde[0, radial1, radial2] * 0.0j
                for pair in range(output_offsets[group], output_offsets[group + 1]):
                    source = input_channel[pair]
                    value += (
                        angular1[pair]
                        * derivative_tilde[source, radial1, radial2]
                        - angular2[pair] * divided[source, radial1, radial2]
                    )
                result[target, radial1, radial2] += 1j * value
