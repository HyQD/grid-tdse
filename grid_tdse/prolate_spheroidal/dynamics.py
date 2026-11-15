"""Real-time Hamiltonian actions for one-electron prolate states."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Real

import numpy as np

from grid_tdse.real_time import RealTimePropagator

from .absorbing_boundaries import ProlateSpheroidalAbsorbingBoundaries
from .dipole import ProlateSpheroidalAxialLengthGaugeInteraction
from .hamiltonian import ProlateSpheroidalOneElectronHamiltonian


class ProlateSpheroidalFixedMTimeDependentHamiltonian:
    r"""Combine field-free and axial length-gauge fixed-``m`` actions.

    Instances satisfy the ``hamiltonian(state, time)`` contract of
    :class:`grid_tdse.real_time.RealTimePropagator`. The corresponding
    physical inner product is available as :meth:`inner_product`.

    An empty field sequence selects field-free propagation. Otherwise the
    supplied callables are interpreted as axial length-gauge electric fields
    and passed to
    :class:`ProlateSpheroidalAxialLengthGaugeInteraction`.
    """

    def __init__(
        self,
        hamiltonian: ProlateSpheroidalOneElectronHamiltonian,
        *,
        m=0,
        fields: (
            Callable[[float], Real]
            | Iterable[Callable[[float], Real]]
            | None
        ) = (),
        include_nuclear_scalar: bool = False,
        absorbing_boundaries: (
            ProlateSpheroidalAbsorbingBoundaries | None
        ) = None,
    ) -> None:
        if not isinstance(
            hamiltonian, ProlateSpheroidalOneElectronHamiltonian
        ):
            raise TypeError(
                "hamiltonian must be a "
                "ProlateSpheroidalOneElectronHamiltonian"
            )
        self.field_free_hamiltonian = hamiltonian
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
                    "absorbing boundaries belong to a different "
                    "representation"
                )
        self.absorbing_boundaries = absorbing_boundaries
        field_profiles = _optional_fields(fields)
        if include_nuclear_scalar and field_profiles is None:
            raise ValueError(
                "include_nuclear_scalar requires at least one field"
            )
        self.dipole_interaction = (
            None
            if field_profiles is None
            else ProlateSpheroidalAxialLengthGaugeInteraction(
                hamiltonian,
                fields=field_profiles,
                include_nuclear_scalar=include_nuclear_scalar,
            )
        )

    @property
    def is_field_free(self) -> bool:
        """Return whether no external field is configured."""

        return self.dipole_interaction is None

    def apply(self, state, time: float) -> np.ndarray:
        """Apply the complete electronic propagation Hamiltonian."""

        values = self.representation.validate_fixed_m_state(state, self.m)
        result = self.field_free_hamiltonian.apply_fixed_m(values, self.m)
        if self.dipole_interaction is not None:
            result = result + self.dipole_interaction(
                values, time, self.m
            )
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

        return self.field_free_hamiltonian.inner_product(
            left, right, self.m
        )

    def inner_products(self, left, right) -> np.ndarray:
        """Return physical overlaps with several left fixed-``m`` states."""

        return self.field_free_hamiltonian.inner_products(
            left, right, self.m
        )

    def propagate(
        self,
        propagator: RealTimePropagator,
        initial_state,
        **kwargs,
    ):
        """Propagate while enforcing absorber/integrator compatibility."""

        if not isinstance(propagator, RealTimePropagator):
            raise TypeError("propagator must be a RealTimePropagator")
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
