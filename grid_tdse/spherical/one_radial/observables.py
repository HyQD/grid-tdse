"""Raw observables and radial sampling for one-radial dynamics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .dynamics import (
    SphericalOneRadialFixedMTimeDependentHamiltonian,
    SphericalOneRadialFullAngularTimeDependentHamiltonian,
)


@dataclass(frozen=True)
class SphericalOneRadialRadialProbabilitySample:
    """One unnormalized radial probability distribution."""

    accepted_step: int
    time: float
    norm: float
    probability: np.ndarray
    is_initial: bool = False
    is_final: bool = False
    is_checkpoint: bool = False


@dataclass(frozen=True)
class SphericalOneRadialRadialProbabilitySeries:
    """Cadence-controlled radial distributions on one FEM-DVR grid."""

    radial_points: np.ndarray
    radial_weights: np.ndarray
    samples: tuple[SphericalOneRadialRadialProbabilitySample, ...]

    @property
    def accepted_steps(self) -> np.ndarray:
        return np.asarray(
            [sample.accepted_step for sample in self.samples], dtype=int
        )

    @property
    def times(self) -> np.ndarray:
        return np.asarray([sample.time for sample in self.samples])

    @property
    def norms(self) -> np.ndarray:
        return np.asarray([sample.norm for sample in self.samples])

    @property
    def probabilities(self) -> np.ndarray:
        if not self.samples:
            return np.empty((0, self.radial_points.size))
        return np.asarray([sample.probability for sample in self.samples])


class SphericalOneRadialDynamicsObservables:
    """Raw scalar samplers and cadence-aware ``P(r)`` collection.

    All scalar expectation values are unnormalized numerators. Conditional
    values remain postprocessing quantities obtained by division by the
    separately sampled norm.
    """

    def __init__(
        self,
        dynamics: SphericalOneRadialFixedMTimeDependentHamiltonian,
        initial_state,
        *,
        reference_states: Mapping[str, np.ndarray] | None = None,
        radial_probability_interval: int | None = 1,
        reality_tolerance: float = 1.0e-10,
    ) -> None:
        if not isinstance(
            dynamics, SphericalOneRadialFixedMTimeDependentHamiltonian
        ):
            raise TypeError(
                "dynamics must be a "
                "SphericalOneRadialFixedMTimeDependentHamiltonian"
            )
        self.dynamics = dynamics
        self.hamiltonian = dynamics.field_free_hamiltonian
        self.system = dynamics.system
        self.m = dynamics.m
        self.reality_tolerance = _positive_real(
            reality_tolerance, "reality_tolerance"
        )
        self.radial_probability_interval = _optional_positive_integer(
            radial_probability_interval, "radial_probability_interval"
        )
        initial = self._state(initial_state)
        initial_norm = self.norm(initial)
        if initial_norm == 0.0:
            raise ValueError("initial_state norm must be nonzero")
        self.initial_norm = initial_norm
        self.initial_total_dipole = (
            self.total_dipole_numerator(initial) / initial_norm
        )
        self._reference_labels, self._reference_states = self._references(
            reference_states
        )
        self._radial_samples: list[
            SphericalOneRadialRadialProbabilitySample
        ] = []

    @property
    def reference_labels(self) -> tuple[str, ...]:
        """Return the ordering used by ``state_populations``."""

        return self._reference_labels

    @property
    def samplers(self) -> dict:
        """Return scalar samplers accepted by ``RealTimePropagator``."""

        samplers = {
            "norm": self.norm,
            "field_free_energy_numerator": self.energy_numerator,
            "electronic_dipole_numerator": (
                self.electronic_dipole_numerator
            ),
            "total_dipole_numerator": self.total_dipole_numerator,
            "induced_dipole_numerator": self.induced_dipole_numerator,
            "canonical_momentum_numerator": (
                self.canonical_momentum_numerator
            ),
        }
        if self.dynamics.field_gauge == "velocity":
            samplers["mechanical_momentum_numerator"] = (
                self.mechanical_momentum_numerator
            )
        if self._reference_states is not None:
            samplers["state_populations"] = self.state_populations
        return samplers

    @property
    def state_observer(self):
        """Return the radial observer, or ``None`` when sampling is disabled."""

        if self.radial_probability_interval is None:
            return None
        return self.observe_radial_probability

    @property
    def propagation_kwargs(self) -> dict:
        """Return sampler arguments for the dynamics propagation wrapper."""

        options = {"samplers": self.samplers}
        if self.state_observer is not None:
            options["state_observer"] = self.state_observer
        return options

    @property
    def radial_probability_series(
        self,
    ) -> SphericalOneRadialRadialProbabilitySeries:
        """Return immutable access to the radial samples collected so far."""

        radial_points = self.hamiltonian.r.copy()
        radial_weights = self.hamiltonian.weights.copy()
        radial_points.setflags(write=False)
        radial_weights.setflags(write=False)
        return SphericalOneRadialRadialProbabilitySeries(
            radial_points=radial_points,
            radial_weights=radial_weights,
            samples=tuple(self._radial_samples),
        )

    def reset_radial_probability_samples(self) -> None:
        """Discard samples retained by an earlier propagation call."""

        self._radial_samples.clear()

    @property
    def metadata(self) -> dict:
        """Return a JSON-compatible description of sampled quantities."""

        return {
            "m": self.m,
            "normalization_policy": "raw_unnormalized_numerators",
            "initial_norm": self.initial_norm,
            "initial_total_dipole": self.initial_total_dipole,
            "scalar_sample_interval": 1,
            "radial_probability_interval": (
                self.radial_probability_interval
            ),
            "radial_probability_includes_initial": (
                self.radial_probability_interval is not None
            ),
            "radial_probability_includes_final": (
                self.radial_probability_interval is not None
            ),
            "reference_state_labels": list(self.reference_labels),
            "canonical_momentum_sampled": True,
            "mechanical_momentum_sampled": (
                self.dynamics.field_gauge == "velocity"
            ),
            "mechanical_momentum_formula": (
                None
                if self.dynamics.field_gauge != "velocity"
                else "P_canonical - electron_charge * A_z(t) * norm"
            ),
            "energy_operator": "physical_field_free_hamiltonian",
            "cap_excluded_from_energy": True,
        }

    def norm(self, state, time=None) -> float:
        """Return the physical, generally unnormalized state norm."""

        del time
        values = self._state(state)
        return self._real(
            self.hamiltonian.inner_product(values, values), "norm"
        )

    def energy_numerator(self, state, time=None) -> float:
        """Return ``Re <Psi|H0|Psi>`` without any CAP or mask."""

        del time
        values = self._state(state)
        return self._real(
            self.hamiltonian.inner_product(
                values, self.dynamics.field_free_action(values)
            ),
            "field-free energy numerator",
        )

    def electronic_dipole_numerator(self, state, time=None) -> float:
        r"""Return raw ``<Psi|q_e*(z-origin_z)|Psi>``."""

        del time
        values = self._state(state)
        norm = self.norm(values)
        result = self.system.electron_charge * (
            self.hamiltonian.inner_product(
                values,
                self.hamiltonian.apply_z_fixed_m(values, self.m),
            )
            - self._origin_z * norm
        )
        return self._real(result, "electronic dipole numerator")

    def total_dipole_numerator(self, state, time=None) -> float:
        """Return the raw electronic plus clamped-nuclear dipole."""

        values = self._state(state)
        return self.electronic_dipole_numerator(values, time) + (
            self._nuclear_dipole_z * self.norm(values)
        )

    def induced_dipole_numerator(self, state, time=None) -> float:
        """Return total dipole relative to the initial conditional dipole."""

        values = self._state(state)
        return self.total_dipole_numerator(values, time) - (
            self.initial_total_dipole * self.norm(values)
        )

    def canonical_momentum_numerator(self, state, time=None) -> float:
        r"""Return raw ``Re <Psi|p_z|Psi>``."""

        del time
        values = self._state(state)
        result = self.hamiltonian.inner_product(
            values, self.hamiltonian.apply_pz_fixed_m(values, self.m)
        )
        return self._real(result, "canonical momentum numerator")

    def mechanical_momentum_numerator(self, state, time) -> float:
        r"""Return raw ``<p_z-q_e*A_z(t)>`` in velocity gauge."""

        if self.dynamics.field_gauge != "velocity":
            raise RuntimeError(
                "mechanical momentum sampling requires velocity gauge"
            )
        values = self._state(state)
        canonical = self.canonical_momentum_numerator(values)
        amplitude = self._field_amplitude(time)
        return canonical - self.system.electron_charge * amplitude * (
            self.norm(values)
        )

    def state_populations(self, state, time=None) -> np.ndarray:
        """Return raw populations in the configured normalized states."""

        del time
        if self._reference_states is None:
            raise RuntimeError("no reference states were configured")
        values = self._state(state)
        overlaps = self.hamiltonian.inner_products(
            self._reference_states, values
        )
        return np.abs(overlaps) ** 2

    def radial_probability(self, state) -> np.ndarray:
        r"""Return raw ``P(r)=sum_l |u_lm(r)|**2``."""

        values = self._state(state)
        return np.sum(np.abs(values) ** 2, axis=-1)

    def observe_radial_probability(
        self,
        state,
        accepted_step,
        time,
        diagnostics,
        final,
        checkpoint,
    ) -> None:
        """Collect initial, interval, checkpoint, and final radial samples."""

        del diagnostics
        if self.radial_probability_interval is None:
            return
        step = _nonnegative_integer(accepted_step, "accepted_step")
        included = (
            step == 0
            or step % self.radial_probability_interval == 0
            or bool(final)
            or bool(checkpoint)
        )
        if not included:
            return
        values = self._state(state)
        probability = np.asarray(
            self.radial_probability(values), dtype=float
        ).copy()
        probability.setflags(write=False)
        sample = SphericalOneRadialRadialProbabilitySample(
            accepted_step=step,
            time=float(time),
            norm=self.norm(values),
            probability=probability,
            is_initial=step == 0,
            is_final=bool(final),
            is_checkpoint=bool(checkpoint),
        )
        if (
            self._radial_samples
            and self._radial_samples[-1].accepted_step == step
        ):
            previous = self._radial_samples[-1]
            sample = SphericalOneRadialRadialProbabilitySample(
                accepted_step=sample.accepted_step,
                time=sample.time,
                norm=sample.norm,
                probability=sample.probability,
                is_initial=sample.is_initial or previous.is_initial,
                is_final=sample.is_final or previous.is_final,
                is_checkpoint=sample.is_checkpoint or previous.is_checkpoint,
            )
            self._radial_samples[-1] = sample
        else:
            self._radial_samples.append(sample)

    @property
    def _origin_z(self) -> float:
        interaction = self.dynamics.dipole_interaction
        return float(getattr(interaction, "origin_z", 0.0))

    @property
    def _nuclear_dipole_z(self) -> float:
        return float(
            self.system.nuclear_dipole_about(
                np.array([0.0, 0.0, self._origin_z])
            )[2]
        )

    def _field_amplitude(self, time) -> float:
        interaction = self.dynamics.dipole_interaction
        if interaction is None:
            return 0.0
        return interaction.field_amplitude(time)

    def _state(self, state) -> np.ndarray:
        values = self.hamiltonian.validate_fixed_m_state(state, self.m)
        if values.ndim != 2:
            raise ValueError("state must contain exactly one fixed-m state")
        return values

    def _references(self, references):
        if references is None:
            return (), None
        if not isinstance(references, Mapping):
            raise TypeError("reference_states must be a mapping or None")
        labels = []
        states = []
        for label, state in references.items():
            if not isinstance(label, str) or not label:
                raise ValueError(
                    "reference-state labels must be nonempty strings"
                )
            values = self._state(state)
            norm = self.norm(values)
            if norm == 0.0:
                raise ValueError(f"reference state {label!r} has zero norm")
            normalized = np.asarray(values, dtype=complex).copy()
            normalized /= np.sqrt(norm)
            normalized.setflags(write=False)
            labels.append(label)
            states.append(normalized)
        if not labels:
            return (), None
        state_array = np.asarray(states)
        state_array.setflags(write=False)
        return tuple(labels), state_array

    def _real(self, value, name: str) -> float:
        result = complex(value)
        scale = max(1.0, abs(result.real))
        if abs(result.imag) > self.reality_tolerance * scale:
            raise ValueError(f"{name} has a significant imaginary part")
        return float(result.real)


class SphericalOneRadialFullAngularDynamicsObservables:
    """Raw vector observables and radial samples for full-angular states."""

    _COMPONENTS = ("x", "y", "z")

    def __init__(
        self,
        dynamics: SphericalOneRadialFullAngularTimeDependentHamiltonian,
        initial_state,
        *,
        reference_states: Mapping[str, np.ndarray] | None = None,
        radial_probability_interval: int | None = 1,
        reality_tolerance: float = 1.0e-10,
    ) -> None:
        if not isinstance(
            dynamics,
            SphericalOneRadialFullAngularTimeDependentHamiltonian,
        ):
            raise TypeError(
                "dynamics must be a "
                "SphericalOneRadialFullAngularTimeDependentHamiltonian"
            )
        self.dynamics = dynamics
        self.hamiltonian = dynamics.field_free_hamiltonian
        self.system = dynamics.system
        self.origin = dynamics.origin
        self.reality_tolerance = _positive_real(
            reality_tolerance,
            "reality_tolerance",
        )
        self.radial_probability_interval = _optional_positive_integer(
            radial_probability_interval,
            "radial_probability_interval",
        )
        initial = self._state(initial_state)
        initial_norm = self.norm(initial)
        if initial_norm == 0.0:
            raise ValueError("initial_state norm must be nonzero")
        self.initial_norm = initial_norm
        self.initial_total_dipole = (
            self.total_dipole_numerator(initial) / initial_norm
        )
        self._reference_labels, self._reference_states = self._references(
            reference_states
        )
        self._radial_samples: list[
            SphericalOneRadialRadialProbabilitySample
        ] = []

    @property
    def reference_labels(self) -> tuple[str, ...]:
        """Return the ordering used by ``state_populations``."""

        return self._reference_labels

    @property
    def samplers(self) -> dict:
        """Return scalar and vector samplers for real-time propagation."""

        samplers = {
            "norm": self.norm,
            "field_free_energy_numerator": self.energy_numerator,
            "electronic_dipole_numerator": (
                self.electronic_dipole_numerator
            ),
            "nuclear_dipole_numerator": self.nuclear_dipole_numerator,
            "total_dipole_numerator": self.total_dipole_numerator,
            "induced_dipole_numerator": self.induced_dipole_numerator,
            "canonical_momentum_numerator": (
                self.canonical_momentum_numerator
            ),
        }
        if self.dynamics.field_gauge == "velocity":
            samplers["mechanical_momentum_numerator"] = (
                self.mechanical_momentum_numerator
            )
        if self._reference_states is not None:
            samplers["state_populations"] = self.state_populations
        return samplers

    @property
    def state_observer(self):
        """Return the radial observer, or ``None`` when disabled."""

        if self.radial_probability_interval is None:
            return None
        return self.observe_radial_probability

    @property
    def propagation_kwargs(self) -> dict:
        """Return sampler arguments for full-angular propagation."""

        options = {"samplers": self.samplers}
        if self.state_observer is not None:
            options["state_observer"] = self.state_observer
        return options

    @property
    def radial_probability_series(
        self,
    ) -> SphericalOneRadialRadialProbabilitySeries:
        """Return immutable access to collected radial samples."""

        radial_points = self.hamiltonian.r.copy()
        radial_weights = self.hamiltonian.weights.copy()
        radial_points.setflags(write=False)
        radial_weights.setflags(write=False)
        return SphericalOneRadialRadialProbabilitySeries(
            radial_points=radial_points,
            radial_weights=radial_weights,
            samples=tuple(self._radial_samples),
        )

    def reset_radial_probability_samples(self) -> None:
        """Discard samples retained by an earlier propagation call."""

        self._radial_samples.clear()

    @property
    def metadata(self) -> dict:
        """Return a JSON-compatible observable description."""

        return {
            "angular_representation": "full",
            "component_order": list(self._COMPONENTS),
            "origin": self.origin.tolist(),
            "normalization_policy": "raw_unnormalized_numerators",
            "initial_norm": self.initial_norm,
            "initial_total_dipole": self.initial_total_dipole.tolist(),
            "scalar_sample_interval": 1,
            "radial_probability_interval": (
                self.radial_probability_interval
            ),
            "radial_probability_includes_initial": (
                self.radial_probability_interval is not None
            ),
            "radial_probability_includes_final": (
                self.radial_probability_interval is not None
            ),
            "reference_state_labels": list(self.reference_labels),
            "canonical_momentum_sampled": True,
            "mechanical_momentum_sampled": (
                self.dynamics.field_gauge == "velocity"
            ),
            "mechanical_momentum_formula": (
                None
                if self.dynamics.field_gauge != "velocity"
                else "P_canonical - electron_charge * (0,0,A_z) * norm"
            ),
            "energy_operator": "physical_field_free_hamiltonian",
            "cap_excluded_from_energy": True,
        }

    def norm(self, state, time=None) -> float:
        """Return the physical, generally unnormalized state norm."""

        del time
        values = self._state(state)
        return self._real(
            self.hamiltonian.inner_product(values, values),
            "norm",
        )

    def energy_numerator(self, state, time=None) -> float:
        """Return ``Re <Psi|H0|Psi>`` without any CAP or mask."""

        del time
        values = self._state(state)
        return self._real(
            self.hamiltonian.inner_product(
                values,
                self.dynamics.field_free_action(values),
            ),
            "field-free energy numerator",
        )

    def electronic_dipole_numerator(self, state, time=None) -> np.ndarray:
        """Return the raw Cartesian electronic dipole numerator."""

        del time
        values = self._state(state)
        norm = self.norm(values)
        expectations = self._cartesian_expectations(
            values,
            ("apply_x", "apply_y", "apply_z"),
        )
        return self.system.electron_charge * (
            expectations - self.origin * norm
        )

    def nuclear_dipole_numerator(self, state, time=None) -> np.ndarray:
        """Return the raw clamped-nuclear dipole numerator."""

        del time
        values = self._state(state)
        return self.system.nuclear_dipole_about(self.origin) * self.norm(
            values
        )

    def total_dipole_numerator(self, state, time=None) -> np.ndarray:
        """Return the raw electronic plus nuclear dipole vector."""

        values = self._state(state)
        return self.electronic_dipole_numerator(
            values,
            time,
        ) + self.nuclear_dipole_numerator(values, time)

    def induced_dipole_numerator(self, state, time=None) -> np.ndarray:
        """Return the total dipole relative to its initial conditional value."""

        values = self._state(state)
        return self.total_dipole_numerator(values, time) - (
            self.initial_total_dipole * self.norm(values)
        )

    def canonical_momentum_numerator(
        self,
        state,
        time=None,
    ) -> np.ndarray:
        """Return the raw Cartesian canonical momentum numerator."""

        del time
        values = self._state(state)
        return self._cartesian_expectations(
            values,
            ("apply_px", "apply_py", "apply_pz"),
        )

    def mechanical_momentum_numerator(self, state, time) -> np.ndarray:
        """Return raw mechanical momentum for a laboratory-``z`` field."""

        if self.dynamics.field_gauge != "velocity":
            raise RuntimeError(
                "mechanical momentum sampling requires velocity gauge"
            )
        values = self._state(state)
        result = self.canonical_momentum_numerator(values).copy()
        result[2] -= (
            self.system.electron_charge
            * self._field_amplitude(time)
            * self.norm(values)
        )
        return result

    def state_populations(self, state, time=None) -> np.ndarray:
        """Return raw populations in configured normalized states."""

        del time
        if self._reference_states is None:
            raise RuntimeError("no reference states were configured")
        values = self._state(state)
        overlaps = self.hamiltonian.inner_products(
            self._reference_states,
            values,
        )
        return np.abs(overlaps) ** 2

    def radial_probability(self, state) -> np.ndarray:
        r"""Return raw ``P(r)=sum_lm |u_lm(r)|**2``."""

        values = self._state(state)
        return np.sum(np.abs(values) ** 2, axis=-1)

    def observe_radial_probability(
        self,
        state,
        accepted_step,
        time,
        diagnostics,
        final,
        checkpoint,
    ) -> None:
        """Collect initial, interval, checkpoint, and final radial samples."""

        del diagnostics
        if self.radial_probability_interval is None:
            return
        step = _nonnegative_integer(accepted_step, "accepted_step")
        included = (
            step == 0
            or step % self.radial_probability_interval == 0
            or bool(final)
            or bool(checkpoint)
        )
        if not included:
            return
        values = self._state(state)
        probability = np.asarray(
            self.radial_probability(values),
            dtype=float,
        ).copy()
        probability.setflags(write=False)
        sample = SphericalOneRadialRadialProbabilitySample(
            accepted_step=step,
            time=float(time),
            norm=self.norm(values),
            probability=probability,
            is_initial=step == 0,
            is_final=bool(final),
            is_checkpoint=bool(checkpoint),
        )
        if (
            self._radial_samples
            and self._radial_samples[-1].accepted_step == step
        ):
            previous = self._radial_samples[-1]
            sample = SphericalOneRadialRadialProbabilitySample(
                accepted_step=sample.accepted_step,
                time=sample.time,
                norm=sample.norm,
                probability=sample.probability,
                is_initial=sample.is_initial or previous.is_initial,
                is_final=sample.is_final or previous.is_final,
                is_checkpoint=sample.is_checkpoint or previous.is_checkpoint,
            )
            self._radial_samples[-1] = sample
        else:
            self._radial_samples.append(sample)

    def _cartesian_expectations(self, state, action_names) -> np.ndarray:
        values = np.asarray(
            [
                self.hamiltonian.inner_product(
                    state,
                    getattr(self.hamiltonian, action_name)(state),
                )
                for action_name in action_names
            ]
        )
        return self._real_vector(values, "Cartesian expectation")

    def _field_amplitude(self, time) -> float:
        interaction = self.dynamics.dipole_interaction
        if interaction is None:
            return 0.0
        return interaction.field_amplitude(time)

    def _state(self, state) -> np.ndarray:
        values = self.hamiltonian.validate_state(state)
        if values.ndim != 2:
            raise ValueError("state must contain exactly one full-angular state")
        return values

    def _references(self, references):
        if references is None:
            return (), None
        if not isinstance(references, Mapping):
            raise TypeError("reference_states must be a mapping or None")
        labels = []
        states = []
        for label, state in references.items():
            if not isinstance(label, str) or not label:
                raise ValueError(
                    "reference-state labels must be nonempty strings"
                )
            values = self._state(state)
            norm = self.norm(values)
            if norm == 0.0:
                raise ValueError(f"reference state {label!r} has zero norm")
            normalized = np.asarray(values, dtype=complex).copy()
            normalized /= np.sqrt(norm)
            normalized.setflags(write=False)
            labels.append(label)
            states.append(normalized)
        if not labels:
            return (), None
        state_array = np.asarray(states)
        state_array.setflags(write=False)
        return tuple(labels), state_array

    def _real(self, value, name: str) -> float:
        result = complex(value)
        scale = max(1.0, abs(result.real))
        if abs(result.imag) > self.reality_tolerance * scale:
            raise ValueError(f"{name} has a significant imaginary part")
        return float(result.real)

    def _real_vector(self, values, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=complex)
        scale = np.maximum(1.0, np.abs(result.real))
        if np.any(np.abs(result.imag) > self.reality_tolerance * scale):
            raise ValueError(f"{name} has a significant imaginary part")
        return np.asarray(result.real, dtype=float)


def _positive_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
