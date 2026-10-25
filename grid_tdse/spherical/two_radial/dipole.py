r"""Axial electric-dipole interactions for two-radial FEM-DVR states."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from grid_lib.spherical_coordinates.momentum_operators import MomentumOperators

from .system import TwoRadialSystem


DipoleGauge = Literal["length", "velocity"]


@dataclass(frozen=True)
class AxialDipoleCouplings:
    r"""Effective couplings for the two coordinates in one gauge.

    In length gauge these are the reduced charges ``xi_i``. In velocity
    gauge they are ``nu_i = xi_i / mu_i``.
    """

    coordinate1: float
    coordinate2: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinate1",
            _finite_real(self.coordinate1, "coordinate1 coupling"),
        )
        object.__setattr__(
            self,
            "coordinate2",
            _finite_real(self.coordinate2, "coordinate2 coupling"),
        )

    @classmethod
    def for_two_electron_atom(
        cls, parameters, gauge: DipoleGauge
    ) -> "AxialDipoleCouplings":
        """Return equal couplings for the two electronic coordinates."""

        gauge = _gauge(gauge)
        charge = _finite_real(parameters.electron_charge, "electron charge")
        if gauge == "length":
            coefficient = charge
        else:
            mass = _positive_real(parameters.electron_mass, "electron mass")
            coefficient = charge / mass
        return cls(coefficient, coefficient)

    @classmethod
    def for_diatomic_ion(
        cls, parameters, gauge: DipoleGauge
    ) -> "AxialDipoleCouplings":
        """Return electronic and internuclear Jacobi-coordinate couplings."""

        gauge = _gauge(gauge)
        if gauge == "length":
            return cls(
                parameters.electronic_reduced_charge,
                parameters.internuclear_reduced_charge,
            )
        return cls(
            parameters.electronic_velocity_coupling,
            parameters.internuclear_velocity_coupling,
        )


class AxialCoordinateDipole:
    r"""Apply one coordinate's field-independent axial dipole operator.

    The state axes are ``(angular1, angular2, radial1, radial2)`` and
    ``coordinate`` is either 1 or 2. For a coupling ``c``, this object applies

    ``-c z`` in length gauge and ``-c p_z`` in velocity gauge.

    Multiplication by the temporal electric-field or vector-potential
    amplitude is performed by :class:`TwoRadialAxialDipoleInteraction`.
    """

    def __init__(
        self,
        representation: TwoRadialSystem,
        coordinate: Literal[1, 2],
        *,
        gauge: DipoleGauge,
        coupling: float,
    ) -> None:
        if coordinate not in (1, 2):
            raise ValueError("coordinate must be 1 or 2")
        self.representation = representation
        self.coordinate = coordinate
        self.gauge = _gauge(gauge)
        self.coupling = _finite_real(coupling, "coupling")

        if coordinate == 1:
            grid = representation.grid1
            basis = representation.basis1
        else:
            grid = representation.grid2
            basis = representation.basis2
        if grid.family != "femdvr":
            raise ValueError("axial dipole interactions currently require FEM-DVR")
        if basis.coordinate_parity is not None:
            raise ValueError(
                f"coordinate {coordinate} is field-coupled and must retain both "
                "even- and odd-l channels"
            )

        self.grid = grid
        self.basis = basis
        self._momentum = MomentumOperators(
            basis.l_max,
            basis.m_max,
            grid.r,
            grid.weights,
            grid.first_derivative(0),
        )
        expected_states = tuple(
            (l_value, m_value)
            for m_value in range(-basis.m_max, basis.m_max + 1)
            for l_value in range(abs(m_value), basis.l_max + 1)
        )
        if basis.states != expected_states:
            raise ValueError(
                "angular state ordering is incompatible with grid-lib momentum "
                "operators"
            )
        self.cos_theta = self._momentum.pz_lm_1.copy()
        self._compact_transitions = (
            self._build_compact_transitions()
            if representation.uses_compact_angular_storage
            else None
        )

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Return the field-independent coordinate action."""

        values = _state_array(state, self.representation.psi_shape)
        return self._apply_values(values)

    def _apply_values(self, values: np.ndarray) -> np.ndarray:
        if self.representation.uses_compact_angular_storage:
            return self._apply_compact(values)
        batched = self._as_radial_angular_batch(values)
        if self.gauge == "length":
            angular = np.matmul(batched, self.cos_theta.T)
            result = -self.coupling * self.grid.r[None, :, None] * angular
        else:
            result = -self.coupling * self._momentum.pz(batched)
        return self._from_radial_angular_batch(result)

    def _build_compact_transitions(self):
        pair_basis = self.representation.angular_pairs
        transitions = [[] for _ in pair_basis.pairs]
        angular_nonzero = tuple(
            np.flatnonzero(
                (self._momentum.pz_lm_1[:, input_] != 0.0)
                | (self._momentum.pz_lm_2[:, input_] != 0.0)
            )
            for input_ in range(self.basis.n_channels)
        )
        for input_pair, (in1, in2) in enumerate(pair_basis.pairs):
            angular_input = in1 if self.coordinate == 1 else in2
            for angular_output in angular_nonzero[angular_input]:
                if self.coordinate == 1:
                    output_pair = pair_basis.get(int(angular_output), in2)
                else:
                    output_pair = pair_basis.get(in1, int(angular_output))
                if output_pair is None:
                    continue
                transitions[output_pair].append(
                    (
                        input_pair,
                        self._momentum.pz_lm_1[angular_output, angular_input],
                        self._momentum.pz_lm_2[angular_output, angular_input],
                    )
                )
        return tuple(tuple(row) for row in transitions)

    def _apply_compact(self, values: np.ndarray) -> np.ndarray:
        dtype = np.result_type(
            values.dtype,
            complex if self.gauge == "velocity" else float,
        )
        result = np.zeros(values.shape, dtype=dtype)
        if self.gauge == "length":
            for output, transitions in enumerate(self._compact_transitions):
                for input_, angular1, _ in transitions:
                    result[output] += angular1 * values[input_]
            if self.coordinate == 1:
                result *= self.grid.r[None, :, None]
            else:
                result *= self.grid.r[None, None, :]
            return -self.coupling * result

        if self.coordinate == 1:
            derivative = np.matmul(self._momentum.D1_tilde, values)
            divided = self.grid.r[None, :, None] ** -1 * values
        else:
            derivative = np.matmul(values, self._momentum.D1_tilde.T)
            divided = self.grid.r[None, None, :] ** -1 * values
        for output, transitions in enumerate(self._compact_transitions):
            for input_, angular1, angular2 in transitions:
                result[output] += (
                    angular1 * derivative[input_] - angular2 * divided[input_]
                )
        return 1j * self.coupling * result

    __call__ = apply

    def _as_radial_angular_batch(self, state: np.ndarray) -> np.ndarray:
        n1, n2, nr1, nr2 = self.representation.psi_shape
        if self.coordinate == 1:
            return state.transpose(1, 3, 2, 0).reshape(n2 * nr2, nr1, n1)
        return state.transpose(0, 2, 3, 1).reshape(n1 * nr1, nr2, n2)

    def _from_radial_angular_batch(self, values: np.ndarray) -> np.ndarray:
        n1, n2, nr1, nr2 = self.representation.psi_shape
        if self.coordinate == 1:
            return values.reshape(n2, nr2, nr1, n1).transpose(3, 0, 2, 1)
        return values.reshape(n1, nr1, nr2, n2).transpose(0, 3, 1, 2)


class TwoRadialAxialDipoleInteraction:
    r"""Apply a spatially uniform, z-polarized dipole interaction.

    Length-gauge profiles return ``E_z(t)`` and the interaction is

    ``-E_z(t) * sum_i xi_i z_i``.

    Velocity-gauge profiles return ``A_z(t)`` and the interaction is

    ``-A_z(t) * sum_i nu_i p_{z,i}``.

    The spatially uniform quadratic vector-potential term is omitted. It is a
    time-dependent scalar and therefore changes only the global phase.
    """

    def __init__(
        self,
        representation: TwoRadialSystem,
        *,
        gauge: DipoleGauge,
        fields: Callable[[float], Real] | Iterable[Callable[[float], Real]],
        couplings: AxialDipoleCouplings,
    ) -> None:
        self.representation = representation
        self.gauge = _gauge(gauge)
        if not isinstance(couplings, AxialDipoleCouplings):
            raise TypeError("couplings must be an AxialDipoleCouplings")
        self.couplings = couplings
        self.fields = _field_profiles(fields, self.gauge)
        coordinate_operators = []
        for coordinate, coupling in (
            (1, couplings.coordinate1),
            (2, couplings.coordinate2),
        ):
            if coupling != 0.0:
                coordinate_operators.append(
                    AxialCoordinateDipole(
                        representation,
                        coordinate,
                        gauge=self.gauge,
                        coupling=coupling,
                    )
                )
        if not coordinate_operators:
            raise ValueError("at least one dipole coupling must be nonzero")
        self.coordinate_operators = tuple(coordinate_operators)

    @classmethod
    def for_two_electron_atom(
        cls,
        representation: TwoRadialSystem,
        parameters,
        *,
        gauge: DipoleGauge,
        fields: Callable[[float], Real] | Iterable[Callable[[float], Real]],
    ) -> "TwoRadialAxialDipoleInteraction":
        """Construct the interaction for an infinite-mass two-electron atom."""

        return cls(
            representation,
            gauge=gauge,
            fields=fields,
            couplings=AxialDipoleCouplings.for_two_electron_atom(
                parameters, gauge
            ),
        )

    @classmethod
    def for_diatomic_ion(
        cls,
        representation: TwoRadialSystem,
        parameters,
        *,
        gauge: DipoleGauge,
        fields: Callable[[float], Real] | Iterable[Callable[[float], Real]],
    ) -> "TwoRadialAxialDipoleInteraction":
        """Construct the interaction for non-BO diatomic Jacobi coordinates."""

        return cls(
            representation,
            gauge=gauge,
            fields=fields,
            couplings=AxialDipoleCouplings.for_diatomic_ion(parameters, gauge),
        )

    def field_amplitude(self, time: float) -> float:
        """Evaluate and sum the scalar z-polarized field profiles."""

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

        values = _state_array(state, self.representation.psi_shape)
        amplitude = self.field_amplitude(time)
        dtype = np.result_type(
            values.dtype, complex if self.gauge == "velocity" else float
        )
        if amplitude == 0.0:
            return np.zeros(self.representation.psi_shape, dtype=dtype)
        result = self.coordinate_operators[0]._apply_values(values).astype(
            dtype, copy=False
        )
        for operator in self.coordinate_operators[1:]:
            result += operator._apply_values(values)
        return amplitude * result

    __call__ = apply


def _field_profiles(fields, gauge: DipoleGauge) -> tuple[Callable, ...]:
    if callable(fields):
        profiles = (fields,)
    elif isinstance(fields, (str, bytes)) or not isinstance(fields, Iterable):
        raise TypeError("fields must be a callable or an iterable of callables")
    else:
        profiles = tuple(fields)
    if not profiles:
        raise ValueError("fields must contain at least one callable")
    if not all(callable(field) for field in profiles):
        raise TypeError("every field must be callable")
    for field in profiles:
        field_gauge = getattr(field, "gauge", gauge)
        if field_gauge != gauge:
            raise ValueError(
                f"field {type(field).__name__} uses gauge {field_gauge!r}, "
                f"but the interaction uses {gauge!r}"
            )
    return profiles


def _state_array(state, expected_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(state)
    if values.shape != expected_shape:
        raise ValueError(f"state has shape {values.shape}; expected {expected_shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("state must contain numeric values")
    return values


def _gauge(value: str) -> DipoleGauge:
    if value not in ("length", "velocity"):
        raise ValueError("gauge must be 'length' or 'velocity'")
    return value


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_real(value, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
