r"""Shared serial and distributed sampling for non-BO diatomic dynamics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Literal

import numpy as np

from .dipole import AxialCoordinateDipole, DipoleGauge
from .observables import (
    DiatomicIonObservableEvaluator,
    DiatomicIonObservableNumerators,
)
from .system import RadialOperatorMode, TwoRadialSystem


@dataclass(frozen=True)
class DiatomicIonConditionalObservables:
    r"""Norm-divided observables conditional on survival in the grid."""

    r: float
    R: float
    r_squared: float
    R_squared: float
    r_dot_R: float
    total_l_squared: float
    total_l_z: float
    l_r_squared: float
    l_R_squared: float
    l_r_z: float
    l_R_z: float
    cos_theta_R: float
    cos2_theta_R: float
    coordinate_r: tuple[float, float, float]
    coordinate_R: tuple[float, float, float]
    dipole_r: tuple[float, float, float]
    dipole_R: tuple[float, float, float]
    dipole_total: tuple[float, float, float]
    inversion: float
    pair_exchange: float | None


@dataclass(frozen=True)
class DiatomicIonElectronCenterDistances:
    r"""Electron--center distance diagnostics in Jacobi coordinates.

    The squared-distance numerators are unnormalized expectations.  The RMS
    distances divide those numerators by the state norm before taking the
    square root.  The two ``legacy_*_proxy`` values reproduce the historical
    ``pulse.py`` expression, which uses squared mean radii and is therefore
    neither a mean distance nor an RMS distance.
    """

    reference_squared_numerator: float
    partner_squared_numerator: float
    reference_rms: float
    partner_rms: float
    legacy_reference_proxy: float
    legacy_partner_proxy: float


@dataclass(frozen=True)
class DiatomicIonDynamicsSnapshot:
    r"""Physical observables evaluated for one dynamics state.

    ``energy_numerator`` is the unnormalized expectation of the physical
    gauge-covariant energy operator.  It excludes masks and complex absorbing
    potentials.  ``conditional_energy`` is explicitly divided by ``norm``.
    """

    gauge: DipoleGauge
    field_symbol: Literal["E", "A"]
    field_amplitude: float
    energy_numerator: float
    conditional_energy: float
    numerators: DiatomicIonObservableNumerators
    conditional: DiatomicIonConditionalObservables
    electron_center_distances: DiatomicIonElectronCenterDistances

    @property
    def norm(self) -> float:
        return self.numerators.norm


@dataclass(frozen=True)
class DiatomicIonPropagationDiagnostics:
    """Integrator diagnostics associated with one accepted outer step."""

    wall_time: float = 0.0
    krylov_operator_calls: int = 0
    krylov_accepted_substeps: int = 0
    krylov_rejected_substeps: int = 0
    krylov_refinement_depth: int = 0
    krylov_dimension: int = 0
    krylov_error_estimate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wall_time",
            _nonnegative_real(self.wall_time, "wall_time"),
        )
        object.__setattr__(
            self,
            "krylov_error_estimate",
            _nonnegative_real(
                self.krylov_error_estimate,
                "krylov_error_estimate",
            ),
        )
        for name in (
            "krylov_operator_calls",
            "krylov_accepted_substeps",
            "krylov_rejected_substeps",
            "krylov_refinement_depth",
            "krylov_dimension",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name),
            )


@dataclass(frozen=True)
class DiatomicIonDynamicsSample:
    """One row of the common non-BO dynamics time-series schema.

    ``norm_divergence`` is the signed change from the initially evaluated
    norm.  Energy gains likewise use the initial raw numerator and initial
    conditional energy as their respective references.
    """

    accepted_step: int
    time: float
    snapshot: DiatomicIonDynamicsSnapshot
    norm_divergence: float
    energy_gain_numerator: float
    conditional_energy_gain: float
    propagation: DiatomicIonPropagationDiagnostics
    is_initial: bool = False
    is_final: bool = False
    is_checkpoint: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible, recursively flattened record."""

        nested = asdict(self)
        record = _flatten_mapping(nested)
        record["snapshot.numerators.z_e"] = (
            self.snapshot.numerators.coordinate_r[2]
        )
        record["snapshot.numerators.z_R"] = (
            self.snapshot.numerators.coordinate_R[2]
        )
        record["snapshot.conditional.z_e"] = (
            self.snapshot.conditional.coordinate_r[2]
        )
        record["snapshot.conditional.z_R"] = (
            self.snapshot.conditional.coordinate_R[2]
        )
        return record


@dataclass(frozen=True)
class DiatomicIonDynamicsTimeSeries:
    """Immutable shared serial/MPI dynamics time series."""

    samples: tuple[DiatomicIonDynamicsSample, ...]

    @property
    def accepted_steps(self) -> np.ndarray:
        return np.asarray([sample.accepted_step for sample in self.samples], dtype=int)

    @property
    def times(self) -> np.ndarray:
        return np.asarray([sample.time for sample in self.samples])

    def records(self) -> tuple[dict[str, object], ...]:
        """Return flattened records suitable for JSONL or tabular output."""

        return tuple(sample.as_dict() for sample in self.samples)


@dataclass(frozen=True)
class DiatomicIonDynamicsSamplingPolicy:
    """Accepted-step selection for dynamics sampling."""

    interval: int | None = 10
    sample_initial: bool = True
    sample_final: bool = True
    sample_checkpoints: bool = True

    def __post_init__(self) -> None:
        if self.interval is not None:
            object.__setattr__(
                self,
                "interval",
                _positive_integer(self.interval, "interval"),
            )
        for name in ("sample_initial", "sample_final", "sample_checkpoints"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(value))

    def includes(
        self,
        accepted_step: int,
        *,
        final: bool = False,
        checkpoint: bool = False,
    ) -> bool:
        """Return whether an accepted state must be sampled."""

        step = _nonnegative_integer(accepted_step, "accepted_step")
        if step == 0 and self.sample_initial:
            return True
        if final and self.sample_final:
            return True
        if checkpoint and self.sample_checkpoints:
            return True
        return self.interval is not None and step % self.interval == 0


class DiatomicIonDynamicsObservableEvaluator:
    r"""Evaluate one physical dynamics snapshot for serial or MPI states.

    The supplied ``field_free_action`` must contain only the physical
    field-free Hamiltonian.  In particular, it must not contain a mask or CAP.
    In velocity gauge, ``velocity_action`` applies

    ``-(nu_r p_r,z + nu_R p_R,z)``.

    The uniform quadratic vector-potential contribution is restored from the
    diatomic reduced charges and masses.  Multiple field profiles are summed
    before this amplitude is squared.
    """

    def __init__(
        self,
        system: TwoRadialSystem,
        parameters,
        field_free_action: Callable,
        *,
        gauge: DipoleGauge = "length",
        fields: Callable[[float], Real] | Iterable[Callable[[float], Real]] = (),
        velocity_action: Callable | None = None,
        radial_operator_mode: RadialOperatorMode = "global",
        reality_tolerance: float = 1.0e-10,
    ) -> None:
        if not isinstance(system, TwoRadialSystem):
            raise TypeError("system must be a TwoRadialSystem")
        if not callable(field_free_action):
            raise TypeError("field_free_action must be callable")
        if gauge not in ("length", "velocity"):
            raise ValueError("gauge must be 'length' or 'velocity'")
        for name in (
            "electronic_reduced_charge",
            "internuclear_reduced_charge",
            "electronic_reduced_mass",
            "internuclear_reduced_mass",
            "electronic_velocity_coupling",
            "internuclear_velocity_coupling",
            "reference_center_scale",
            "partner_center_scale",
        ):
            if not hasattr(parameters, name):
                raise TypeError(f"parameters is missing {name!r}")
        self.system = system
        self.parameters = parameters
        self.field_free_action = field_free_action
        self.gauge = gauge
        self.fields = _field_profiles(fields, gauge)
        self.reality_tolerance = _positive_real(reality_tolerance, "reality_tolerance")
        self.observable_evaluator = DiatomicIonObservableEvaluator(system, parameters)
        if gauge == "velocity" and velocity_action is None:
            velocity_action = _serial_velocity_action(
                system,
                parameters,
                radial_operator_mode=radial_operator_mode,
            )
        if velocity_action is not None and not callable(velocity_action):
            raise TypeError("velocity_action must be callable or None")
        self.velocity_action = velocity_action
        xi_r = float(parameters.electronic_reduced_charge)
        xi_R = float(parameters.internuclear_reduced_charge)
        mu_r = _positive_real(
            parameters.electronic_reduced_mass,
            "electronic_reduced_mass",
        )
        mu_R = _positive_real(
            parameters.internuclear_reduced_mass,
            "internuclear_reduced_mass",
        )
        self._quadratic_coefficient = 0.5 * (xi_r**2 / mu_r + xi_R**2 / mu_R)

    def __call__(self, state: np.ndarray, time: float) -> DiatomicIonDynamicsSnapshot:
        """Return a serial dynamics snapshot."""

        numerators = self.observable_evaluator.numerators(state)
        return self._snapshot(
            state,
            time,
            numerators,
            self.system.inner_product,
        )

    def distributed(
        self, state, reductions, time: float
    ) -> DiatomicIonDynamicsSnapshot:
        """Collectively return a one-coordinate MPI dynamics snapshot."""

        numerators = self.observable_evaluator.distributed_numerators(state, reductions)
        return self._snapshot(state, time, numerators, reductions.expectation)

    def distributed_2d(
        self, state, reductions, time: float
    ) -> DiatomicIonDynamicsSnapshot:
        """Collectively return a Cartesian-MPI dynamics snapshot."""

        numerators = self.observable_evaluator.distributed_2d_numerators(
            state, reductions
        )
        return self._snapshot(state, time, numerators, reductions.expectation)

    def field_amplitude(self, time: float) -> float:
        """Return the sum of all electric fields or vector potentials."""

        time = _finite_real(time, "time")
        amplitude = 0.0
        for index, field in enumerate(self.fields):
            value = np.asarray(field(time))
            if (
                value.ndim != 0
                or not np.issubdtype(value.dtype, np.number)
                or np.iscomplexobj(value)
                or not np.isfinite(value)
            ):
                raise ValueError(f"field {index} must return a finite real scalar")
            amplitude += float(value)
        return amplitude

    def _snapshot(self, state, time, numerators, inner_product):
        time = _finite_real(time, "time")
        amplitude = self.field_amplitude(time)
        field_free_state = self.field_free_action(state)
        energy = self._real_expectation(
            inner_product(state, field_free_state),
            "field-free energy numerator",
        )
        if self.gauge == "velocity":
            if self.velocity_action is None:  # defensive; constructor supplies it
                raise RuntimeError("velocity gauge requires a velocity action")
            velocity_state = self.velocity_action(state)
            energy += amplitude * self._real_expectation(
                inner_product(state, velocity_state),
                "velocity energy numerator",
            )
            energy += amplitude**2 * self._quadratic_coefficient * numerators.norm
        conditional = _conditional_observables(numerators)
        return DiatomicIonDynamicsSnapshot(
            gauge=self.gauge,
            field_symbol="E" if self.gauge == "length" else "A",
            field_amplitude=amplitude,
            energy_numerator=energy,
            conditional_energy=energy / numerators.norm,
            numerators=numerators,
            conditional=conditional,
            electron_center_distances=_electron_center_distances(
                numerators,
                conditional,
                float(self.parameters.reference_center_scale),
                float(self.parameters.partner_center_scale),
            ),
        )

    def _real_expectation(self, value, name):
        value = complex(value)
        scale = max(1.0, abs(value.real))
        if abs(value.imag) > self.reality_tolerance * scale:
            raise ValueError(f"{name} has a significant imaginary part: {value.imag}")
        return float(value.real)


class DiatomicIonDynamicsSampler:
    """Collect a common time series using one of the evaluator paths."""

    def __init__(
        self,
        evaluator: DiatomicIonDynamicsObservableEvaluator,
        *,
        policy: DiatomicIonDynamicsSamplingPolicy | None = None,
    ) -> None:
        if not isinstance(evaluator, DiatomicIonDynamicsObservableEvaluator):
            raise TypeError(
                "evaluator must be a DiatomicIonDynamicsObservableEvaluator"
            )
        if policy is None:
            policy = DiatomicIonDynamicsSamplingPolicy()
        if not isinstance(policy, DiatomicIonDynamicsSamplingPolicy):
            raise TypeError("policy must be a DiatomicIonDynamicsSamplingPolicy")
        self.evaluator = evaluator
        self.policy = policy
        self._samples: list[DiatomicIonDynamicsSample] = []
        self._initial_norm: float | None = None
        self._initial_energy_numerator: float | None = None
        self._initial_conditional_energy: float | None = None

    @property
    def time_series(self) -> DiatomicIonDynamicsTimeSeries:
        return DiatomicIonDynamicsTimeSeries(tuple(self._samples))

    def sample(
        self,
        state,
        accepted_step: int,
        time: float,
        *,
        diagnostics: DiatomicIonPropagationDiagnostics | object | None = None,
        final: bool = False,
        checkpoint: bool = False,
    ) -> DiatomicIonDynamicsSample | None:
        """Possibly sample a serial state according to the policy."""

        return self._sample(
            lambda: self.evaluator(state, time),
            accepted_step,
            time,
            diagnostics,
            final,
            checkpoint,
        )

    def sample_distributed(
        self,
        state,
        reductions,
        accepted_step: int,
        time: float,
        *,
        diagnostics: DiatomicIonPropagationDiagnostics | object | None = None,
        final: bool = False,
        checkpoint: bool = False,
    ) -> DiatomicIonDynamicsSample | None:
        """Possibly sample a one-coordinate MPI state collectively."""

        return self._sample(
            lambda: self.evaluator.distributed(state, reductions, time),
            accepted_step,
            time,
            diagnostics,
            final,
            checkpoint,
        )

    def sample_distributed_2d(
        self,
        state,
        reductions,
        accepted_step: int,
        time: float,
        *,
        diagnostics: DiatomicIonPropagationDiagnostics | object | None = None,
        final: bool = False,
        checkpoint: bool = False,
    ) -> DiatomicIonDynamicsSample | None:
        """Possibly sample a Cartesian-MPI state collectively."""

        return self._sample(
            lambda: self.evaluator.distributed_2d(state, reductions, time),
            accepted_step,
            time,
            diagnostics,
            final,
            checkpoint,
        )

    def _sample(
        self,
        evaluate,
        accepted_step,
        time,
        diagnostics,
        final,
        checkpoint,
    ):
        step = _nonnegative_integer(accepted_step, "accepted_step")
        time = _finite_real(time, "time")
        included = self.policy.includes(
            step,
            final=bool(final),
            checkpoint=bool(checkpoint),
        )
        if not included and not (step == 0 and self._initial_norm is None):
            return None
        snapshot = evaluate()
        if self._initial_norm is None:
            self._initial_norm = snapshot.norm
            self._initial_energy_numerator = snapshot.energy_numerator
            self._initial_conditional_energy = snapshot.conditional_energy
        if not included:
            return None
        propagation = _propagation_diagnostics(diagnostics)
        sample = DiatomicIonDynamicsSample(
            accepted_step=step,
            time=time,
            snapshot=snapshot,
            norm_divergence=snapshot.norm - self._initial_norm,
            energy_gain_numerator=(
                snapshot.energy_numerator - self._initial_energy_numerator
            ),
            conditional_energy_gain=(
                snapshot.conditional_energy - self._initial_conditional_energy
            ),
            propagation=propagation,
            is_initial=step == 0,
            is_final=bool(final),
            is_checkpoint=bool(checkpoint),
        )
        if self._samples and self._samples[-1].accepted_step == step:
            previous = self._samples[-1]
            sample = replace(
                sample,
                is_initial=sample.is_initial or previous.is_initial,
                is_final=sample.is_final or previous.is_final,
                is_checkpoint=sample.is_checkpoint or previous.is_checkpoint,
            )
            self._samples[-1] = sample
        else:
            self._samples.append(sample)
        return sample


def _conditional_observables(numerators):
    norm = numerators.norm

    def divide(values):
        return tuple(value / norm for value in values)

    return DiatomicIonConditionalObservables(
        r=numerators.r / norm,
        R=numerators.R / norm,
        r_squared=numerators.r_squared / norm,
        R_squared=numerators.R_squared / norm,
        r_dot_R=numerators.r_dot_R / norm,
        total_l_squared=numerators.total_l_squared / norm,
        total_l_z=numerators.total_l_z / norm,
        l_r_squared=numerators.l_r_squared / norm,
        l_R_squared=numerators.l_R_squared / norm,
        l_r_z=numerators.l_r_z / norm,
        l_R_z=numerators.l_R_z / norm,
        cos_theta_R=numerators.cos_theta_R / norm,
        cos2_theta_R=numerators.cos2_theta_R / norm,
        coordinate_r=divide(numerators.coordinate_r),
        coordinate_R=divide(numerators.coordinate_R),
        dipole_r=divide(numerators.dipole_r),
        dipole_R=divide(numerators.dipole_R),
        dipole_total=divide(numerators.dipole_total),
        inversion=numerators.inversion / norm,
        pair_exchange=(
            numerators.pair_exchange / norm
            if numerators.pair_exchange is not None
            else None
        ),
    )


def _electron_center_distances(
    numerators,
    conditional,
    reference_scale,
    partner_scale,
):
    def squared_numerator(scale):
        return (
            numerators.r_squared
            + scale**2 * numerators.R_squared
            - 2.0 * scale * numerators.r_dot_R
        )

    def legacy_proxy(scale):
        squared = (
            conditional.r**2
            + scale**2 * conditional.R**2
            - 2.0 * scale * conditional.r_dot_R
        )
        return _sqrt_nonnegative(squared, "legacy distance proxy")

    reference = squared_numerator(reference_scale)
    partner = squared_numerator(partner_scale)
    return DiatomicIonElectronCenterDistances(
        reference_squared_numerator=reference,
        partner_squared_numerator=partner,
        reference_rms=_sqrt_nonnegative(
            reference / numerators.norm, "reference RMS distance"
        ),
        partner_rms=_sqrt_nonnegative(
            partner / numerators.norm, "partner RMS distance"
        ),
        legacy_reference_proxy=legacy_proxy(reference_scale),
        legacy_partner_proxy=legacy_proxy(partner_scale),
    )


def _serial_velocity_action(system, parameters, *, radial_operator_mode):
    operators = []
    for coordinate, coupling in (
        (1, float(parameters.electronic_velocity_coupling)),
        (2, float(parameters.internuclear_velocity_coupling)),
    ):
        if coupling != 0.0:
            operators.append(
                AxialCoordinateDipole(
                    system,
                    coordinate,
                    gauge="velocity",
                    coupling=coupling,
                    radial_operator_mode=radial_operator_mode,
                )
            )
    if not operators:
        raise ValueError("velocity gauge requires a nonzero velocity coupling")

    def action(state):
        result = operators[0](state)
        for operator in operators[1:]:
            result = result + operator(state)
        return result

    return action


def _field_profiles(fields, gauge):
    if callable(fields):
        profiles = (fields,)
    elif isinstance(fields, (str, bytes)):
        raise TypeError("fields must contain callables")
    else:
        profiles = tuple(fields)
    for field in profiles:
        if not callable(field):
            raise TypeError("every field must be callable")
        field_gauge = getattr(field, "gauge", gauge)
        if field_gauge != gauge:
            raise ValueError(f"field uses gauge {field_gauge!r}, expected {gauge!r}")
    return profiles


def _propagation_diagnostics(value):
    if value is None:
        return DiatomicIonPropagationDiagnostics()
    if isinstance(value, DiatomicIonPropagationDiagnostics):
        return value
    aliases = {
        "wall_time": ("wall_time", "step_wall_time"),
        "krylov_operator_calls": ("krylov_operator_calls",),
        "krylov_accepted_substeps": ("krylov_accepted_substeps",),
        "krylov_rejected_substeps": ("krylov_rejected_substeps",),
        "krylov_refinement_depth": ("krylov_refinement_depth",),
        "krylov_dimension": ("krylov_dimension",),
        "krylov_error_estimate": ("krylov_error_estimate",),
    }
    values = {}
    for target, names in aliases.items():
        values[target] = next(
            (getattr(value, name) for name in names if hasattr(value, name)),
            0.0 if target in ("wall_time", "krylov_error_estimate") else 0,
        )
    return DiatomicIonPropagationDiagnostics(**values)


def _flatten_mapping(values: Mapping, prefix=""):
    result = {}
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            result.update(_flatten_mapping(value, key))
        elif isinstance(value, tuple):
            labels = ("x", "y", "z") if len(value) == 3 else range(len(value))
            for label, item in zip(labels, value, strict=True):
                result[f"{key}.{label}"] = item
        else:
            result[key] = value
    return result


def _sqrt_nonnegative(value, name):
    value = float(value)
    tolerance = 128.0 * np.finfo(float).eps * max(1.0, abs(value))
    if value < -tolerance:
        raise ValueError(f"{name} squared is negative: {value}")
    return float(np.sqrt(max(0.0, value)))


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_real(value, name):
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value, name):
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_integer(value, name):
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result
