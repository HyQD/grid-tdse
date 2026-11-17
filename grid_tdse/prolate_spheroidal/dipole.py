r"""Axial field couplings for one-electron prolate states."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Real

import numpy as np

from .hamiltonian import ProlateSpheroidalOneElectronHamiltonian


class ProlateSpheroidalAxialLengthGaugeInteraction:
    r"""Apply a spatially uniform axial electric field in length gauge.

    For an axial origin ``z0`` measured from the bond midpoint and the
    midpoint coordinate ``z = a xi eta``, the electronic interaction is

    ``V_e(t) = -E_z(t) q_e (z - z0)``.

    The electron charge is supplied by the physical system and is ``-1`` in
    atomic units, so the electronic spatial action is ``+z``. Temporal field
    profiles remain separate from this field-independent action.

    A heteronuclear clamped-nuclei system also has the constant dipole
    ``d_n(z0) = a (Z_2 - Z_1) - (Z_1 + Z_2) z0``. Its interaction
    ``-E_z(t) d_n`` is a scalar and changes only the global phase. It is
    therefore omitted by default, but can be included explicitly with
    ``include_nuclear_scalar``. The default ``origin_z=0`` is the bond
    midpoint.
    """

    def __init__(
        self,
        hamiltonian: ProlateSpheroidalOneElectronHamiltonian,
        *,
        fields: Callable[[float], Real]
        | Iterable[Callable[[float], Real]],
        include_nuclear_scalar: bool = False,
        origin_z: Real = 0.0,
    ) -> None:
        if not isinstance(
            hamiltonian, ProlateSpheroidalOneElectronHamiltonian
        ):
            raise TypeError(
                "hamiltonian must be a "
                "ProlateSpheroidalOneElectronHamiltonian"
            )
        if not isinstance(include_nuclear_scalar, (bool, np.bool_)):
            raise TypeError("include_nuclear_scalar must be boolean")
        if isinstance(origin_z, (bool, np.bool_)) or not isinstance(
            origin_z, Real
        ):
            raise TypeError("origin_z must be a real scalar")
        if not np.isfinite(origin_z):
            raise ValueError("origin_z must be finite")
        self.hamiltonian = hamiltonian
        self.representation = hamiltonian.representation
        self.system = hamiltonian.system
        self.fields = _field_profiles(fields, gauge="length")
        self.include_nuclear_scalar = bool(include_nuclear_scalar)
        self.origin_z = float(origin_z)

    @property
    def nuclear_dipole_z(self) -> float:
        """Return the static nuclear dipole about the configured origin."""

        return float(
            self.system.nuclear_dipole_about(
                np.array([0.0, 0.0, self.origin_z])
            )[2]
        )

    def field_amplitude(self, time: float) -> float:
        """Evaluate and sum the scalar axial electric-field profiles."""

        return _summed_field_amplitude(self.fields, time)

    def spatial_action(self, state, m=0) -> np.ndarray:
        r"""Return the field-independent factor ``V(t) / E_z(t)``.

        By default this contains only ``-q_e z``. If requested at
        construction time, the clamped-nuclei scalar ``-d_n`` is added.
        """

        values = self.representation.validate_fixed_m_state(state, m)
        result = -self.system.electron_charge * (
            self.hamiltonian.apply_z_fixed_m(values, m)
            - self.origin_z * values
        )
        if self.include_nuclear_scalar and self.nuclear_dipole_z != 0.0:
            result = result - self.nuclear_dipole_z * values
        return result

    def apply(self, state, time: float, m=0) -> np.ndarray:
        """Return the complete time-dependent length-gauge action."""

        values = self.representation.validate_fixed_m_state(state, m)
        amplitude = self.field_amplitude(time)
        if amplitude == 0.0:
            return np.zeros_like(values)
        return amplitude * self.spatial_action(values, m)

    def electronic_dipole_expectation(self, state, m=0) -> float:
        r"""Return normalized ``<q_e (z-origin_z)>``."""

        values = self.representation.validate_fixed_m_state(state, m)
        norm = self.hamiltonian.inner_product(values, values, m)
        if norm == 0.0:
            raise ValueError("state norm must be nonzero")
        expectation_z = (
            self.hamiltonian.inner_product(
                values,
                self.hamiltonian.apply_z_fixed_m(values, m),
                m,
            )
            / norm
        )
        expectation = self.system.electron_charge * (
            expectation_z - self.origin_z
        )
        return _real_expectation(expectation, "electronic dipole")

    def total_dipole_expectation(self, state, m=0) -> float:
        """Return electronic plus clamped-nuclei axial dipole."""

        return self.electronic_dipole_expectation(
            state, m
        ) + self.nuclear_dipole_z

    def archive_metadata(self) -> dict:
        """Return JSON-compatible coupling and origin metadata."""

        return {
            "gauge": "length",
            "field_direction": "molecular_axis_z",
            "origin": (
                "bond_midpoint"
                if self.origin_z == 0.0
                else "axial_offset_from_bond_midpoint"
            ),
            "origin_z": self.origin_z,
            "electronic_coordinate": "z - origin_z = a * xi * eta - origin_z",
            "electron_charge": self.system.electron_charge,
            "nuclear_dipole_z": self.nuclear_dipole_z,
            "include_nuclear_scalar": self.include_nuclear_scalar,
            "nuclear_scalar_is_global_phase": True,
        }

    __call__ = apply


class ProlateSpheroidalAxialVelocityGaugeInteraction:
    r"""Apply a spatially uniform axial vector potential.

    With charge ``q_e`` and particle mass ``mu``, the linear part of minimal
    coupling is

    ``V(t) = -A_z(t) (q_e / mu) p_z``.

    The spatially constant ``q_e**2 A_z(t)**2 / (2 mu)`` term is omitted. It
    changes only the global phase, so norms, expectation values, final-state
    fidelities, and spectra are unaffected. Temporal profiles must use the
    ``velocity`` gauge and return ``A_z(t)``.
    """

    def __init__(
        self,
        hamiltonian: ProlateSpheroidalOneElectronHamiltonian,
        *,
        fields: Callable[[float], Real]
        | Iterable[Callable[[float], Real]],
    ) -> None:
        if not isinstance(
            hamiltonian, ProlateSpheroidalOneElectronHamiltonian
        ):
            raise TypeError(
                "hamiltonian must be a "
                "ProlateSpheroidalOneElectronHamiltonian"
            )
        self.hamiltonian = hamiltonian
        self.representation = hamiltonian.representation
        self.system = hamiltonian.system
        self.fields = _field_profiles(fields, gauge="velocity")

    @property
    def momentum_coupling(self) -> float:
        """Return the charge-to-mass factor ``q_e / mu``."""

        return float(
            self.system.electron_charge / self.system.particle_mass
        )

    def field_amplitude(self, time: float) -> float:
        """Evaluate and sum the scalar axial vector-potential profiles."""

        return _summed_field_amplitude(self.fields, time)

    def spatial_action(self, state, m=0) -> np.ndarray:
        r"""Return the field-independent factor ``V(t) / A_z(t)``."""

        values = self.representation.validate_fixed_m_state(state, m)
        return -self.momentum_coupling * (
            self.hamiltonian.apply_pz_fixed_m(values, m)
        )

    def apply(self, state, time: float, m=0) -> np.ndarray:
        """Return the complete linear velocity-gauge action."""

        values = self.representation.validate_fixed_m_state(state, m)
        amplitude = self.field_amplitude(time)
        if amplitude == 0.0:
            return np.zeros_like(values)
        return amplitude * self.spatial_action(values, m)

    def momentum_expectation(self, state, m=0) -> float:
        r"""Return normalized ``<p_z>``."""

        values = self.representation.validate_fixed_m_state(state, m)
        norm = self.hamiltonian.inner_product(values, values, m)
        if norm == 0.0:
            raise ValueError("state norm must be nonzero")
        expectation = self.hamiltonian.inner_product(
            values,
            self.hamiltonian.apply_pz_fixed_m(values, m),
            m,
        ) / norm
        return _real_expectation(expectation, "axial momentum")

    def archive_metadata(self) -> dict:
        """Return JSON-compatible velocity-gauge metadata."""

        return {
            "gauge": "velocity",
            "field_direction": "molecular_axis_z",
            "temporal_amplitude": "A_z(t)",
            "spatial_operator": "-(electron_charge / particle_mass) * p_z",
            "electron_charge": self.system.electron_charge,
            "particle_mass": self.system.particle_mass,
            "momentum_coupling": self.momentum_coupling,
            "diamagnetic_a_squared_term_included": False,
            "diamagnetic_a_squared_term_is_global_phase": True,
        }

    __call__ = apply


def _field_profiles(fields, *, gauge: str) -> tuple[Callable, ...]:
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
                f"field {type(field).__name__} uses gauge "
                f"{field_gauge!r}, but the interaction uses {gauge!r}"
            )
    return profiles


def _summed_field_amplitude(fields, time: float) -> float:
    amplitude = 0.0
    for index, field in enumerate(fields):
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


def _real_expectation(value, name: str) -> float:
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > 1.0e-12 * scale:
        raise ValueError(f"{name} has a significant imaginary part")
    return float(value.real)
