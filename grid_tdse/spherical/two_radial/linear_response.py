"""Instantaneous axial perturbations for two-radial states."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from integrator_lib import ShortIterativeLanczos


_EXPONENTIAL_OPTIONS = {
    "max_krylov_dimension",
    "rtol",
    "atol",
    "max_refinements",
    "reorthogonalize",
    "breakdown_tolerance",
}


@dataclass(frozen=True)
class AxialDeltaKick:
    r"""An instantaneous z-polarized electric-field impulse.

    ``integrated_field`` is ``K = integral E_z(t) dt``. For the two-radial
    length-gauge convention ``H_int(t) = -E_z(t) D_z``, the finite-space
    transformation is ``exp(+1j * K * P D_z P)``. The supplied interaction's
    field-independent action is ``W = -P D_z P``, so the implementation
    evaluates ``exp(-1j * K * W)`` without resolving a finite pulse in time.
    """

    integrated_field: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integrated_field",
            _finite_real(self.integrated_field, "integrated_field"),
        )

    def apply(
        self,
        interaction,
        state: np.ndarray,
        *,
        inner_product: Callable[[np.ndarray, np.ndarray], complex],
        inner_products: Callable[[tuple[np.ndarray, ...], np.ndarray], object]
        | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Return the kicked state in the interaction's finite representation."""

        if getattr(interaction, "gauge", None) != "length":
            raise ValueError("AxialDeltaKick requires a length-gauge interaction")
        spatial_action = getattr(interaction, "spatial_action", None)
        if not callable(spatial_action):
            raise TypeError("interaction must provide a spatial_action method")
        if not callable(inner_product):
            raise TypeError("inner_product must be callable")
        if inner_products is not None and not callable(inner_products):
            raise TypeError("inner_products must be callable or None")
        values = np.asarray(state)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("state must contain numeric values")
        if not np.isfinite(values).all():
            raise ValueError("state must contain only finite values")

        settings = resolve_axial_delta_kick_options(options)
        complex_state = np.asarray(values, dtype=np.result_type(values.dtype, complex))
        if self.integrated_field == 0.0:
            return complex_state.copy()

        sign = float(np.sign(self.integrated_field))

        def signed_spatial_action(current, time):
            del time
            return -1j * sign * spatial_action(current)

        exponential = ShortIterativeLanczos(
            inner_product=inner_product,
            inner_products=inner_products,
            **settings,
        )
        kicked = exponential(
            complex_state,
            0.0,
            abs(self.integrated_field),
            signed_spatial_action,
        )
        return np.asarray(kicked, dtype=np.result_type(kicked, complex))

    @property
    def metadata(self) -> dict[str, object]:
        """Return the physical, JSON-compatible kick description."""

        return {
            "type": type(self).__name__,
            "direction": "laboratory_z",
            "gauge_representation": "length",
            "integrated_field": self.integrated_field,
            "state_transformation": "exp(+1j * K * P D_z P)",
        }


def resolve_axial_delta_kick_options(
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return validated exponential-action options including defaults."""

    values = {} if options is None else dict(options)
    unknown = set(values).difference(_EXPONENTIAL_OPTIONS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown initial-kick options: {names}")
    defaults = {
        "max_krylov_dimension": 20,
        "rtol": 1.0e-12,
        "atol": 0.0,
        "max_refinements": 8,
        "reorthogonalize": True,
        "breakdown_tolerance": 1.0e-14,
    }
    defaults.update(values)
    return defaults


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


__all__ = ["AxialDeltaKick"]
