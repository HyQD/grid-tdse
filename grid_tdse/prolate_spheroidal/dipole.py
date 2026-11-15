r"""Axial length-gauge coupling for one-electron prolate states."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Real

import numpy as np

from .hamiltonian import ProlateSpheroidalOneElectronHamiltonian


class ProlateSpheroidalAxialLengthGaugeInteraction:
    r"""Apply a spatially uniform axial electric field in length gauge.

    For the midpoint origin and an electronic coordinate ``z = a xi eta``,
    the electronic interaction is

    ``V_e(t) = -E_z(t) q_e z``.

    The electron charge is supplied by the physical system and is ``-1`` in
    atomic units, so the electronic spatial action is ``+z``. Temporal field
    profiles remain separate from this field-independent action.

    A heteronuclear clamped-nuclei system also has the constant midpoint
    dipole ``d_n = a (Z_2 - Z_1)``. Its interaction ``-E_z(t) d_n`` is a
    scalar and changes only the global phase. It is therefore omitted by
    default, but can be included explicitly with ``include_nuclear_scalar``.
    """

    def __init__(
        self,
        hamiltonian: ProlateSpheroidalOneElectronHamiltonian,
        *,
        fields: Callable[[float], Real]
        | Iterable[Callable[[float], Real]],
        include_nuclear_scalar: bool = False,
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
        self.hamiltonian = hamiltonian
        self.representation = hamiltonian.representation
        self.system = hamiltonian.system
        self.fields = _field_profiles(fields)
        self.include_nuclear_scalar = bool(include_nuclear_scalar)

    @property
    def nuclear_dipole_z(self) -> float:
        """Return the static nuclear dipole about the bond midpoint."""

        return float(self.system.nuclear_dipole[2])

    def field_amplitude(self, time: float) -> float:
        """Evaluate and sum the scalar axial electric-field profiles."""

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

    def spatial_action(self, state, m=0) -> np.ndarray:
        r"""Return the field-independent factor ``V(t) / E_z(t)``.

        By default this contains only ``-q_e z``. If requested at
        construction time, the clamped-nuclei scalar ``-d_n`` is added.
        """

        values = self.representation.validate_fixed_m_state(state, m)
        result = -self.system.electron_charge * (
            self.hamiltonian.apply_z_fixed_m(values, m)
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
        r"""Return normalized ``<q_e z>`` about the midpoint origin."""

        values = self.representation.validate_fixed_m_state(state, m)
        norm = self.hamiltonian.inner_product(values, values, m)
        if norm == 0.0:
            raise ValueError("state norm must be nonzero")
        expectation = self.system.electron_charge * (
            self.hamiltonian.inner_product(
                values,
                self.hamiltonian.apply_z_fixed_m(values, m),
                m,
            )
            / norm
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
            "origin": "bond_midpoint",
            "electronic_coordinate": "z = a * xi * eta",
            "electron_charge": self.system.electron_charge,
            "nuclear_dipole_z": self.nuclear_dipole_z,
            "include_nuclear_scalar": self.include_nuclear_scalar,
            "nuclear_scalar_is_global_phase": True,
        }

    __call__ = apply


def _field_profiles(fields) -> tuple[Callable, ...]:
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
        field_gauge = getattr(field, "gauge", "length")
        if field_gauge != "length":
            raise ValueError(
                f"field {type(field).__name__} uses gauge "
                f"{field_gauge!r}, but the interaction uses 'length'"
            )
    return profiles


def _real_expectation(value, name: str) -> float:
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > 1.0e-12 * scale:
        raise ValueError(f"{name} has a significant imaginary part")
    return float(value.real)
