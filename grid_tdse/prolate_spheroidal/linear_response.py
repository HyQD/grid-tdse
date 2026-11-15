"""Axial delta-kick spectroscopy for one-electron prolate states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .absorbing_boundaries import ProlateSpheroidalAbsorbingBoundaries
from .hamiltonian import ProlateSpheroidalOneElectronHamiltonian


@dataclass(frozen=True)
class AxialDeltaKick:
    r"""An instantaneous axial electric-field impulse.

    ``integrated_field`` is ``K = integral E(t) dt``.  With the package's
    electronic length-gauge convention ``H(t) = H0 + E(t) z``, the exact
    state transformation is ``exp(-1j * K * z)``.
    """

    integrated_field: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integrated_field",
            _finite_real(self.integrated_field, "integrated_field"),
        )

    def apply(self, hamiltonian, state, m=0) -> np.ndarray:
        """Return the exactly phase-kicked nodal state."""

        hamiltonian = _hamiltonian(hamiltonian)
        m = hamiltonian.representation.validate_m(m)
        values = hamiltonian.representation.validate_fixed_m_state(state, m)
        phase = np.exp(
            -1j * self.integrated_field * hamiltonian.fixed_m_z(m)
        )
        return np.asarray(values, dtype=np.result_type(values, complex)) * phase

    @property
    def metadata(self) -> dict:
        """Return a JSON-compatible kick description."""

        return {
            "type": type(self).__name__,
            "direction": "molecular_axis_z",
            "gauge": "length",
            "integrated_field": self.integrated_field,
            "state_transformation": "exp(-1j * integrated_field * z)",
            "z_coordinate": "a * xi * eta",
        }


@dataclass(frozen=True)
class AreaNormalizedSineSquareImpulse:
    r"""Finite sine-square pulse with a prescribed electric-field area.

    On ``initial_time <= t <= initial_time + duration`` the field is

    ``2*K/duration * sin(pi*(t-initial_time)/duration)**2``,

    so its exact time integral is ``K = integrated_field``.
    """

    integrated_field: float
    duration: float
    initial_time: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integrated_field",
            _finite_real(self.integrated_field, "integrated_field"),
        )
        object.__setattr__(self, "duration", _positive_real(self.duration, "duration"))
        object.__setattr__(
            self,
            "initial_time",
            _finite_real(self.initial_time, "initial_time"),
        )

    @property
    def final_time(self) -> float:
        """Return the end of the finite impulse."""

        return self.initial_time + self.duration

    @property
    def gauge(self) -> str:
        """Return the field gauge expected by the dipole interaction."""

        return "length"

    def __call__(self, time: float) -> float:
        """Evaluate the area-normalized electric field."""

        time = _finite_real(time, "time")
        relative = time - self.initial_time
        if relative < 0.0 or relative > self.duration:
            return 0.0
        angle = np.pi * relative / self.duration
        return float(
            2.0
            * self.integrated_field
            / self.duration
            * np.sin(angle) ** 2
        )

    @property
    def metadata(self) -> dict:
        """Return a JSON-compatible finite-impulse description."""

        return {
            "type": type(self).__name__,
            "direction": "molecular_axis_z",
            "gauge": self.gauge,
            "integrated_field": self.integrated_field,
            "initial_time": self.initial_time,
            "duration": self.duration,
            "final_time": self.final_time,
            "envelope": "2*K/T * sin(pi*(t-t0)/T)**2",
        }


class ProlateSpheroidalDynamicsObservables:
    """Named real-time samplers for one fixed-``m`` prolate state."""

    def __init__(
        self,
        hamiltonian: ProlateSpheroidalOneElectronHamiltonian,
        *,
        m=0,
        absorbing_boundaries: ProlateSpheroidalAbsorbingBoundaries | None = None,
        reference_states: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.hamiltonian = _hamiltonian(hamiltonian)
        self.representation = hamiltonian.representation
        self.system = hamiltonian.system
        self.m = self.representation.validate_m(m)
        if absorbing_boundaries is not None:
            if not isinstance(
                absorbing_boundaries,
                ProlateSpheroidalAbsorbingBoundaries,
            ):
                raise TypeError(
                    "absorbing_boundaries must be a "
                    "ProlateSpheroidalAbsorbingBoundaries"
                )
            if absorbing_boundaries.representation is not self.representation:
                raise ValueError(
                    "absorbing boundaries belong to a different representation"
                )
        self.absorbing_boundaries = absorbing_boundaries
        self._reference_labels, self._reference_states = self._references(
            reference_states
        )
        self._absorber_region = self._build_absorber_region()

    @property
    def reference_labels(self) -> tuple[str, ...]:
        """Return the ordering of selected-state populations."""

        return self._reference_labels

    @property
    def samplers(self) -> dict:
        """Return samplers accepted by :class:`RealTimePropagator`."""

        samplers = {
            "electronic_dipole_numerator": self.electronic_dipole_numerator,
            "electronic_dipole": self.electronic_dipole,
            "total_dipole_numerator": self.total_dipole_numerator,
            "total_dipole": self.total_dipole,
        }
        if self._reference_states is not None:
            samplers["state_populations"] = self.state_populations
        if self._absorber_region is not None:
            samplers["absorber_region_probability"] = (
                self.absorber_region_probability
            )
        if (
            self.absorbing_boundaries is not None
            and self.absorbing_boundaries.has_cap
        ):
            samplers["cap_norm_loss_rate"] = self.cap_norm_loss_rate
        return samplers

    @property
    def metadata(self) -> dict:
        """Return a JSON-compatible description of sampled quantities."""

        return {
            "m": self.m,
            "dipole_origin": "bond_midpoint",
            "dipole_units": "electron_charge_bohr",
            "dipole_numerator_is_unnormalized": True,
            "state_population_labels": list(self.reference_labels),
            "absorber_diagnostics": {
                "region_probability": self._absorber_region is not None,
                "cap_norm_loss_rate": bool(
                    self.absorbing_boundaries is not None
                    and self.absorbing_boundaries.has_cap
                ),
            },
        }

    def norm(self, state) -> float:
        """Return the physical, generally unnormalized state norm."""

        values = self._state(state)
        return _real_value(
            self.hamiltonian.inner_product(values, values, self.m),
            "norm",
        )

    def electronic_dipole_numerator(self, state, time=None) -> float:
        r"""Return unnormalized ``<Psi|q_e z|Psi>``."""

        del time
        values = self._state(state)
        result = self.system.electron_charge * self.hamiltonian.inner_product(
            values,
            self.hamiltonian.apply_z_fixed_m(values, self.m),
            self.m,
        )
        return _real_value(result, "electronic dipole numerator")

    def electronic_dipole(self, state, time=None) -> float:
        """Return the normalized electronic axial dipole."""

        numerator = self.electronic_dipole_numerator(state, time)
        norm = self.norm(state)
        if norm == 0.0:
            raise ValueError("state norm must be nonzero")
        return numerator / norm

    def total_dipole_numerator(self, state, time=None) -> float:
        """Return unnormalized electronic plus clamped-nuclear dipole."""

        return self.electronic_dipole_numerator(state, time) + (
            float(self.system.nuclear_dipole[2]) * self.norm(state)
        )

    def total_dipole(self, state, time=None) -> float:
        """Return the normalized total axial dipole."""

        numerator = self.total_dipole_numerator(state, time)
        norm = self.norm(state)
        if norm == 0.0:
            raise ValueError("state norm must be nonzero")
        return numerator / norm

    def state_populations(self, state, time=None) -> np.ndarray:
        """Return raw populations in the selected normalized states."""

        del time
        if self._reference_states is None:
            raise RuntimeError("no reference states were configured")
        values = self._state(state)
        overlaps = np.asarray(
            [
                self.hamiltonian.inner_product(reference, values, self.m)
                for reference in self._reference_states
            ]
        )
        return np.abs(overlaps) ** 2

    def absorber_region_probability(self, state, time=None) -> float:
        """Return raw probability on nodes where the absorber is active."""

        del time
        if self._absorber_region is None:
            raise RuntimeError("no absorbing boundary was configured")
        values = self._state(state)
        weights = self.representation.metric_weights(self.m)
        result = np.sum(
            np.abs(values[self._absorber_region]) ** 2
            * weights[self._absorber_region]
        )
        return float(result)

    def cap_norm_loss_rate(self, state, time=None) -> float:
        r"""Return ``-2 Im <Psi|V_CAP|Psi>``, a nonnegative loss rate."""

        del time
        if (
            self.absorbing_boundaries is None
            or not self.absorbing_boundaries.has_cap
        ):
            raise RuntimeError("no complex absorbing potential was configured")
        values = self._state(state)
        cap_action = self.absorbing_boundaries.cap_action(values, self.m)
        expectation = self.hamiltonian.inner_product(
            values, cap_action, self.m
        )
        result = -2.0 * float(expectation.imag)
        tolerance = 1.0e-14 * max(1.0, abs(expectation))
        if result < -tolerance:
            raise ValueError("CAP norm-loss rate is significantly negative")
        return max(0.0, result)

    def _state(self, state) -> np.ndarray:
        return self.representation.validate_fixed_m_state(state, self.m)

    def _references(self, references):
        if references is None:
            return (), None
        if not isinstance(references, Mapping):
            raise TypeError("reference_states must be a mapping or None")
        labels = []
        states = []
        for label, state in references.items():
            if not isinstance(label, str) or not label:
                raise ValueError("reference-state labels must be nonempty strings")
            values = self._state(state)
            norm = self.norm(values)
            if norm == 0.0:
                raise ValueError(f"reference state {label!r} has zero norm")
            normalized = np.asarray(values, dtype=complex).copy() / np.sqrt(norm)
            normalized.setflags(write=False)
            labels.append(label)
            states.append(normalized)
        if not labels:
            return (), None
        return tuple(labels), tuple(states)

    def _build_absorber_region(self):
        if self.absorbing_boundaries is None:
            return None
        if self.absorbing_boundaries.has_cap:
            values = self.absorbing_boundaries.cap_values(self.m)
            region = np.abs(values) > 0.0
        else:
            values = self.absorbing_boundaries.mask_values(self.m)
            region = values < 1.0
        result = np.asarray(region, dtype=bool)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class PositiveFrequencySpectrum:
    """Windowed positive-frequency transform of a uniformly sampled signal."""

    angular_frequencies: np.ndarray
    transform: np.ndarray
    window: str
    initial_time: float
    final_time: float
    time_step: float
    remove_initial: bool

    @property
    def power(self) -> np.ndarray:
        """Return squared transform magnitudes."""

        return np.abs(self.transform) ** 2

    @property
    def metadata(self) -> dict:
        """Return a JSON-compatible transform description."""

        return {
            "frequency_units": "hartree_angular_frequency",
            "fourier_convention": "integral f(t)*exp(-1j*omega*(t-t0)) dt",
            "window": self.window,
            "initial_time": self.initial_time,
            "final_time": self.final_time,
            "duration": self.final_time - self.initial_time,
            "time_step": self.time_step,
            "remove_initial": self.remove_initial,
            "zero_frequency_omitted": True,
        }


def paired_kick_linear_response(
    positive_dipole,
    negative_dipole,
    kick_strength,
) -> np.ndarray:
    r"""Return ``(d(+K)-d(-K))/(2*K)`` for paired kicks."""

    strength = _nonzero_real(kick_strength, "kick_strength")
    positive = _finite_signal(positive_dipole, "positive_dipole")
    negative = _finite_signal(negative_dipole, "negative_dipole")
    if positive.shape != negative.shape:
        raise ValueError("positive and negative dipoles must have the same shape")
    return (positive - negative) / (2.0 * strength)


def windowed_positive_frequency_spectrum(
    times,
    signal,
    *,
    window: Literal["none", "hann"] = "hann",
    remove_initial: bool = True,
) -> PositiveFrequencySpectrum:
    """Return a windowed FFT on the positive angular-frequency grid.

    ``times`` must contain both ends of a uniform interval. The duplicated
    periodic endpoint is omitted from the FFT, making its frequency spacing
    exactly ``2*pi/(times[-1]-times[0])``.
    """

    times = _uniform_times(times)
    values = _finite_signal(signal, "signal")
    if values.ndim != 1 or values.shape != times.shape:
        raise ValueError("signal must be one-dimensional with one value per time")
    values = _real_signal(values)
    interval_count = times.size - 1
    sampled = values[:-1].copy()
    if remove_initial:
        sampled -= values[0]
    if window == "none":
        weights = np.ones(interval_count)
    elif window == "hann":
        phase = 2.0 * np.pi * np.arange(interval_count) / interval_count
        weights = 0.5 * (1.0 - np.cos(phase))
    else:
        raise ValueError("window must be 'none' or 'hann'")
    time_step = float(times[1] - times[0])
    transform = time_step * np.fft.rfft(sampled * weights)
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(
        interval_count, d=time_step
    )
    frequencies = _readonly(frequencies[1:])
    transform = _readonly(transform[1:])
    return PositiveFrequencySpectrum(
        angular_frequencies=frequencies,
        transform=transform,
        window=window,
        initial_time=float(times[0]),
        final_time=float(times[-1]),
        time_step=time_step,
        remove_initial=bool(remove_initial),
    )


def lorentzian_broadened_positive_frequency_spectrum(
    times,
    signal,
    *,
    fwhm,
    remove_initial: bool = True,
) -> PositiveFrequencySpectrum:
    r"""Return a transform with Lorentzian full width ``fwhm``.

    Multiplication by ``exp(-fwhm*(t-t0)/2)`` gives every infinitely long
    monochromatic component a Lorentzian line with the requested full width
    at half maximum in angular-frequency (energy) units.
    """

    width = _positive_real(fwhm, "fwhm")
    times = _uniform_times(times)
    values = _finite_signal(signal, "signal")
    if values.ndim != 1 or values.shape != times.shape:
        raise ValueError("signal must be one-dimensional with one value per time")
    values = _real_signal(values)
    interval_count = times.size - 1
    sampled = values[:-1].copy()
    if remove_initial:
        sampled -= values[0]
    relative_times = times[:-1] - times[0]
    weights = np.exp(-0.5 * width * relative_times)
    time_step = float(times[1] - times[0])
    transform = time_step * np.fft.rfft(sampled * weights)
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(
        interval_count, d=time_step
    )
    frequencies = _readonly(frequencies[1:])
    transform = _readonly(transform[1:])
    return PositiveFrequencySpectrum(
        angular_frequencies=frequencies,
        transform=transform,
        window=f"lorentzian_fwhm_{width:.16g}",
        initial_time=float(times[0]),
        final_time=float(times[-1]),
        time_step=time_step,
        remove_initial=bool(remove_initial),
    )


def _hamiltonian(value) -> ProlateSpheroidalOneElectronHamiltonian:
    if not isinstance(value, ProlateSpheroidalOneElectronHamiltonian):
        raise TypeError(
            "hamiltonian must be a ProlateSpheroidalOneElectronHamiltonian"
        )
    return value


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(value, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonzero_real(value, name: str) -> float:
    result = _finite_real(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return result


def _real_value(value, name: str) -> float:
    value = complex(value)
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > 1.0e-12 * scale:
        raise ValueError(f"{name} has a significant imaginary part")
    return float(value.real)


def _finite_signal(values, name: str) -> np.ndarray:
    result = np.asarray(values)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _real_signal(values) -> np.ndarray:
    if not np.iscomplexobj(values):
        return np.asarray(values, dtype=float)
    scale = max(1.0, float(np.max(np.abs(values.real))))
    if np.max(np.abs(values.imag)) > 1.0e-12 * scale:
        raise ValueError("signal must be real within numerical tolerance")
    return np.asarray(values.real, dtype=float)


def _uniform_times(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 3:
        raise ValueError("times must be one-dimensional with at least three values")
    if not np.all(np.isfinite(result)):
        raise ValueError("times must contain only finite values")
    differences = np.diff(result)
    if np.any(differences <= 0.0):
        raise ValueError("times must be strictly increasing")
    if not np.allclose(
        differences,
        differences[0],
        rtol=1.0e-11,
        atol=1.0e-13 * max(1.0, abs(differences[0])),
    ):
        raise ValueError("times must be uniformly spaced")
    return result


def _readonly(values) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result
