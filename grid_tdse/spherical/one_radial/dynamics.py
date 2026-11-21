"""Real-time Hamiltonian actions for spherical one-radial states."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Real
from typing import Literal

import numpy as np

from grid_tdse.real_time import RealTimePropagator

from .absorbing_boundaries import SphericalOneRadialAbsorbingBoundaries
from .dipole import (
    SphericalOneRadialAxialLengthGaugeInteraction,
    SphericalOneRadialAxialVelocityGaugeInteraction,
    SphericalOneRadialFullAngularAxialLengthGaugeInteraction,
    SphericalOneRadialFullAngularAxialVelocityGaugeInteraction,
    _cartesian_origin,
)
from .hamiltonian import OneElectronDiatomicHamiltonian
from .preconditioner import (
    SphericalOneRadialFullAngularKineticCrankNicolsonPreconditioner,
    SphericalOneRadialKineticCrankNicolsonPreconditioner,
)


class SphericalOneRadialFixedMTimeDependentHamiltonian:
    r"""Combine field-free, axial-field, and CAP fixed-``m`` actions."""

    def __init__(
        self,
        hamiltonian: OneElectronDiatomicHamiltonian,
        *,
        m=0,
        fields: (
            Callable[[float], Real]
            | Iterable[Callable[[float], Real]]
            | None
        ) = (),
        field_gauge: Literal["length", "velocity"] = "length",
        include_nuclear_scalar: bool = False,
        origin_z: Real = 0.0,
        absorbing_boundaries: (
            SphericalOneRadialAbsorbingBoundaries | None
        ) = None,
    ) -> None:
        if not isinstance(hamiltonian, OneElectronDiatomicHamiltonian):
            raise TypeError(
                "hamiltonian must be a OneElectronDiatomicHamiltonian"
            )
        self.field_free_hamiltonian = hamiltonian
        self.system = hamiltonian.system
        self.m = hamiltonian.validate_m(m)

        if absorbing_boundaries is not None:
            if not isinstance(
                absorbing_boundaries,
                SphericalOneRadialAbsorbingBoundaries,
            ):
                raise TypeError(
                    "absorbing_boundaries must be a "
                    "SphericalOneRadialAbsorbingBoundaries"
                )
            if absorbing_boundaries.hamiltonian is not hamiltonian:
                raise ValueError(
                    "absorbing boundaries belong to a different Hamiltonian"
                )
        self.absorbing_boundaries = absorbing_boundaries

        if field_gauge not in {"length", "velocity"}:
            raise ValueError("field_gauge must be 'length' or 'velocity'")
        self.field_gauge = field_gauge
        field_profiles = _optional_fields(fields)
        if include_nuclear_scalar and field_profiles is None:
            raise ValueError(
                "include_nuclear_scalar requires at least one field"
            )
        if field_gauge == "velocity" and include_nuclear_scalar:
            raise ValueError(
                "include_nuclear_scalar is a length-gauge phase option"
            )
        if field_gauge == "velocity" and origin_z != 0.0:
            raise ValueError("origin_z is a length-gauge option")

        if field_profiles is None:
            self.dipole_interaction = None
        elif field_gauge == "length":
            self.dipole_interaction = (
                SphericalOneRadialAxialLengthGaugeInteraction(
                    hamiltonian,
                    fields=field_profiles,
                    include_nuclear_scalar=include_nuclear_scalar,
                    origin_z=origin_z,
                )
            )
        else:
            self.dipole_interaction = (
                SphericalOneRadialAxialVelocityGaugeInteraction(
                    hamiltonian,
                    fields=field_profiles,
                )
            )

    @property
    def is_field_free(self) -> bool:
        """Return whether no external field is configured."""

        return self.dipole_interaction is None

    def field_free_action(self, state) -> np.ndarray:
        """Apply only the physical field-free Hamiltonian."""

        values = self.field_free_hamiltonian.validate_fixed_m_state(
            state, self.m
        )
        return self.field_free_hamiltonian.apply_fixed_m(values, self.m)

    def apply(self, state, time: float) -> np.ndarray:
        """Apply the complete electronic propagation Hamiltonian."""

        values = self.field_free_hamiltonian.validate_fixed_m_state(
            state, self.m
        )
        result = self.field_free_action(values)
        if self.dipole_interaction is not None:
            result = result + self.dipole_interaction(values, time, self.m)
        if (
            self.absorbing_boundaries is not None
            and self.absorbing_boundaries.has_cap
        ):
            result = result + self.absorbing_boundaries.cap_action(
                values, self.m
            )
        return result

    __call__ = apply

    def inner_product(self, left, right) -> complex:
        """Return the physical fixed-``m`` inner product."""

        self.field_free_hamiltonian.validate_fixed_m_state(left, self.m)
        self.field_free_hamiltonian.validate_fixed_m_state(right, self.m)
        return self.field_free_hamiltonian.inner_product(left, right)

    def inner_products(self, left, right) -> np.ndarray:
        """Return physical overlaps with several left fixed-``m`` states."""

        self.field_free_hamiltonian.validate_fixed_m_state(left, self.m)
        self.field_free_hamiltonian.validate_fixed_m_state(right, self.m)
        return self.field_free_hamiltonian.inner_products(left, right)

    def propagate(
        self,
        propagator: RealTimePropagator,
        initial_state,
        **kwargs,
    ):
        """Propagate while enforcing absorber/integrator compatibility."""

        if not isinstance(propagator, RealTimePropagator):
            raise TypeError("propagator must be a RealTimePropagator")
        self.field_free_hamiltonian.validate_fixed_m_state(
            initial_state, self.m
        )
        preconditioner = kwargs.get("preconditioner")
        if isinstance(
            preconditioner,
            SphericalOneRadialKineticCrankNicolsonPreconditioner,
        ):
            if preconditioner.hamiltonian is not self.field_free_hamiltonian:
                raise ValueError(
                    "preconditioner belongs to a different Hamiltonian"
                )
            if preconditioner.m != self.m:
                raise ValueError(
                    "preconditioner magnetic sector does not match dynamics"
                )
            preconditioner.validate_propagator(propagator)
            dynamics_metadata = dict(kwargs.get("dynamics_metadata") or {})
            dynamics_metadata.setdefault(
                "preconditioner", preconditioner.metadata
            )
            kwargs["dynamics_metadata"] = dynamics_metadata
        elif isinstance(
            preconditioner,
            SphericalOneRadialFullAngularKineticCrankNicolsonPreconditioner,
        ):
            raise ValueError(
                "full-angular preconditioner is incompatible with fixed-m "
                "dynamics"
            )
        if self.absorbing_boundaries is not None:
            self.absorbing_boundaries.validate_integrator(
                propagator.integrator
            )
            if self.absorbing_boundaries.has_mask:
                if kwargs.get("post_step") is not None:
                    raise ValueError(
                        "post_step cannot be supplied with a configured mask"
                    )
                kwargs["post_step"] = lambda state, time: (
                    self.absorbing_boundaries.apply_mask(
                        state, time, self.m
                    )
                )
        if propagator.integrator != "crank-nicolson":
            kwargs.setdefault("inner_products", self.inner_products)
        return propagator.propagate(
            initial_state,
            self,
            self.inner_product,
            **kwargs,
        )

    def archive_metadata(self) -> dict:
        """Return checkpoint-compatible system and dynamics metadata."""

        metadata = self.field_free_hamiltonian.archive_metadata()
        metadata.update(
            {
                "m": self.m,
                "field_gauge": self.field_gauge,
                "time_dependent_interaction": (
                    None
                    if self.dipole_interaction is None
                    else self.dipole_interaction.archive_metadata()
                ),
                "absorbing_boundaries": (
                    None
                    if self.absorbing_boundaries is None
                    else self.absorbing_boundaries.metadata
                ),
            }
        )
        return metadata


class SphericalOneRadialFullAngularTimeDependentHamiltonian:
    r"""Combine field-free, laboratory-``z``, and CAP full-state actions."""

    def __init__(
        self,
        hamiltonian: OneElectronDiatomicHamiltonian,
        *,
        fields: (
            Callable[[float], Real]
            | Iterable[Callable[[float], Real]]
            | None
        ) = (),
        field_gauge: Literal["length", "velocity"] = "length",
        include_nuclear_scalar: bool = False,
        origin=(0.0, 0.0, 0.0),
        absorbing_boundaries: (
            SphericalOneRadialAbsorbingBoundaries | None
        ) = None,
    ) -> None:
        if not isinstance(hamiltonian, OneElectronDiatomicHamiltonian):
            raise TypeError(
                "hamiltonian must be a OneElectronDiatomicHamiltonian"
            )
        self.field_free_hamiltonian = hamiltonian
        self.system = hamiltonian.system
        self.origin = _cartesian_origin(origin)

        if absorbing_boundaries is not None:
            if not isinstance(
                absorbing_boundaries,
                SphericalOneRadialAbsorbingBoundaries,
            ):
                raise TypeError(
                    "absorbing_boundaries must be a "
                    "SphericalOneRadialAbsorbingBoundaries"
                )
            if absorbing_boundaries.hamiltonian is not hamiltonian:
                raise ValueError(
                    "absorbing boundaries belong to a different Hamiltonian"
                )
        self.absorbing_boundaries = absorbing_boundaries

        if field_gauge not in {"length", "velocity"}:
            raise ValueError("field_gauge must be 'length' or 'velocity'")
        self.field_gauge = field_gauge
        field_profiles = _optional_fields(fields)
        if include_nuclear_scalar and field_profiles is None:
            raise ValueError(
                "include_nuclear_scalar requires at least one field"
            )
        if field_gauge == "velocity" and include_nuclear_scalar:
            raise ValueError(
                "include_nuclear_scalar is a length-gauge phase option"
            )

        if field_profiles is None:
            self.dipole_interaction = None
        elif field_gauge == "length":
            self.dipole_interaction = (
                SphericalOneRadialFullAngularAxialLengthGaugeInteraction(
                    hamiltonian,
                    fields=field_profiles,
                    include_nuclear_scalar=include_nuclear_scalar,
                    origin=self.origin,
                )
            )
        else:
            self.dipole_interaction = (
                SphericalOneRadialFullAngularAxialVelocityGaugeInteraction(
                    hamiltonian,
                    fields=field_profiles,
                )
            )

    @property
    def is_field_free(self) -> bool:
        """Return whether no external field is configured."""

        return self.dipole_interaction is None

    def field_free_action(self, state) -> np.ndarray:
        """Apply only the physical field-free Hamiltonian."""

        values = self.field_free_hamiltonian.validate_state(state)
        return self.field_free_hamiltonian.apply(values)

    def apply(self, state, time: float) -> np.ndarray:
        """Apply the complete electronic propagation Hamiltonian."""

        values = self.field_free_hamiltonian.validate_state(state)
        result = self.field_free_action(values)
        if self.dipole_interaction is not None:
            result = result + self.dipole_interaction(values, time)
        if (
            self.absorbing_boundaries is not None
            and self.absorbing_boundaries.has_cap
        ):
            result = result + self.absorbing_boundaries.cap_action_full(
                values
            )
        return result

    __call__ = apply

    def inner_product(self, left, right) -> complex:
        """Return the physical full-angular inner product."""

        self.field_free_hamiltonian.validate_state(left)
        self.field_free_hamiltonian.validate_state(right)
        return self.field_free_hamiltonian.inner_product(left, right)

    def inner_products(self, left, right) -> np.ndarray:
        """Return physical overlaps with several full-angular states."""

        self.field_free_hamiltonian.validate_state(left)
        self.field_free_hamiltonian.validate_state(right)
        return self.field_free_hamiltonian.inner_products(left, right)

    def propagate(
        self,
        propagator: RealTimePropagator,
        initial_state,
        **kwargs,
    ):
        """Propagate while enforcing full-angular absorber compatibility."""

        if not isinstance(propagator, RealTimePropagator):
            raise TypeError("propagator must be a RealTimePropagator")
        self.field_free_hamiltonian.validate_state(initial_state)
        preconditioner = kwargs.get("preconditioner")
        if isinstance(
            preconditioner,
            SphericalOneRadialFullAngularKineticCrankNicolsonPreconditioner,
        ):
            if preconditioner.hamiltonian is not self.field_free_hamiltonian:
                raise ValueError(
                    "preconditioner belongs to a different Hamiltonian"
                )
            preconditioner.validate_propagator(propagator)
            dynamics_metadata = dict(kwargs.get("dynamics_metadata") or {})
            dynamics_metadata.setdefault(
                "preconditioner",
                preconditioner.metadata,
            )
            kwargs["dynamics_metadata"] = dynamics_metadata
        elif isinstance(
            preconditioner,
            SphericalOneRadialKineticCrankNicolsonPreconditioner,
        ):
            raise ValueError(
                "fixed-m preconditioner is incompatible with full-angular "
                "dynamics"
            )
        if self.absorbing_boundaries is not None:
            self.absorbing_boundaries.validate_integrator(
                propagator.integrator
            )
            if self.absorbing_boundaries.has_mask:
                if kwargs.get("post_step") is not None:
                    raise ValueError(
                        "post_step cannot be supplied with a configured mask"
                    )
                boundaries = self.absorbing_boundaries
                kwargs["post_step"] = boundaries.apply_mask_full
        if propagator.integrator != "crank-nicolson":
            kwargs.setdefault("inner_products", self.inner_products)
        return propagator.propagate(
            initial_state,
            self,
            self.inner_product,
            **kwargs,
        )

    def archive_metadata(self) -> dict:
        """Return checkpoint-compatible full-angular dynamics metadata."""

        metadata = self.field_free_hamiltonian.archive_metadata()
        metadata.update(
            {
                "angular_representation": "full",
                "origin": self.origin.tolist(),
                "field_gauge": self.field_gauge,
                "time_dependent_interaction": (
                    None
                    if self.dipole_interaction is None
                    else self.dipole_interaction.archive_metadata()
                ),
                "absorbing_boundaries": (
                    None
                    if self.absorbing_boundaries is None
                    else self.absorbing_boundaries.metadata
                ),
            }
        )
        return metadata


def _optional_fields(fields):
    if fields is None:
        return None
    if callable(fields):
        return fields
    if isinstance(fields, (str, bytes)):
        return fields
    try:
        profiles = tuple(fields)
    except TypeError:
        return fields
    return None if not profiles else profiles
