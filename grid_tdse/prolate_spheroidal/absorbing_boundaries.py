"""Absorbing boundaries for one-electron prolate-spheroidal states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from .representation import ProlateSpheroidalRepresentation


@dataclass(frozen=True)
class CartesianRadiusCosineMask:
    r"""Zhang et al. Cartesian-radius cosine mask.

    The mask is evaluated at the midpoint radius
    ``r = a * sqrt(xi**2 + eta**2 - 1)``. Its default exponent ``1/8`` is
    Eq. (42) of Zhang et al., Computer Physics Communications 194, 84 (2015).
    """

    r0: float

    def __post_init__(self) -> None:
        profile = RadialCosineMask(r0=self.r0, n=8.0)
        object.__setattr__(self, "r0", profile.r0)

    @property
    def n(self) -> float:
        """Return the fixed Eq. (42) denominator of the exponent."""

        return 8.0

    def values(self, representation) -> np.ndarray:
        """Return full-grid mask values."""

        grid = representation.grid
        radius = grid.cartesian_radius()
        r_max = grid.a * representation.xi_boundaries[-1]
        profile = RadialCosineMask(r0=self.r0, n=self.n)
        values = profile(radius.reshape(-1), r_max).reshape(grid.shape)
        values.setflags(write=False)
        return values

    def metadata(self, representation) -> dict:
        """Return JSON-compatible profile and geometry metadata."""

        return {
            "type": type(self).__name__,
            "coordinate": "cartesian_radius_from_bond_midpoint",
            "formula": "cos(pi*(r-r0)/(2*(r_max-r0)))**(1/n)",
            "r0": self.r0,
            "r_max": (
                representation.grid.a * representation.xi_boundaries[-1]
            ),
            "n": self.n,
            "reference": "Zhang et al. CPC 194, 84 (2015), Eq. (42)",
        }


@dataclass(frozen=True)
class ProlateSpheroidalCubicCAP:
    r"""Cubic CAP in the ellipsoidal length coordinate ``q = a xi``."""

    alpha: float
    eta: float = 1.0

    def __post_init__(self) -> None:
        profile = CubicCAP(alpha=self.alpha, eta=self.eta)
        object.__setattr__(self, "alpha", profile.alpha)
        object.__setattr__(self, "eta", profile.eta)

    def values(self, representation) -> np.ndarray:
        """Return full-grid diagonal CAP values."""

        grid = representation.grid
        q = grid.a * grid.xi
        q_max = grid.a * representation.xi_boundaries[-1]
        radial_values = CubicCAP(
            alpha=self.alpha, eta=self.eta
        )(q, q_max)
        values = np.broadcast_to(
            radial_values[:, None], grid.shape
        ).copy()
        values.setflags(write=False)
        return values

    def metadata(self, representation) -> dict:
        """Return JSON-compatible profile and geometry metadata."""

        q_max = representation.grid.a * representation.xi_boundaries[-1]
        return {
            "type": type(self).__name__,
            "coordinate": "q = a * xi",
            "formula": "-1j * eta * max(q-alpha*q_max, 0)**3",
            "alpha": self.alpha,
            "eta": self.eta,
            "q_onset": self.alpha * q_max,
            "q_max": q_max,
        }


class ProlateSpheroidalAbsorbingBoundaries:
    """Prepared post-step mask or propagation CAP on a prolate grid."""

    def __init__(
        self,
        representation: ProlateSpheroidalRepresentation,
        *,
        mask: CartesianRadiusCosineMask | None = None,
        cap: ProlateSpheroidalCubicCAP | None = None,
    ) -> None:
        if not isinstance(representation, ProlateSpheroidalRepresentation):
            raise TypeError(
                "representation must be a "
                "ProlateSpheroidalRepresentation"
            )
        if mask is not None and not isinstance(
            mask, CartesianRadiusCosineMask
        ):
            raise TypeError("mask must be a CartesianRadiusCosineMask")
        if cap is not None and not isinstance(
            cap, ProlateSpheroidalCubicCAP
        ):
            raise TypeError("cap must be a ProlateSpheroidalCubicCAP")
        if mask is not None and cap is not None:
            raise ValueError("mask and cap cannot both be configured")
        self.representation = representation
        self.mask_profile = mask
        self.cap_profile = cap
        self._full_mask = None if mask is None else mask.values(representation)
        self._full_cap = None if cap is None else cap.values(representation)
        self._active_masks: dict[int, np.ndarray] = {}
        self._active_caps: dict[int, np.ndarray] = {}

    @property
    def has_mask(self) -> bool:
        """Return whether a post-step mask is configured."""

        return self.mask_profile is not None

    @property
    def has_cap(self) -> bool:
        """Return whether a complex absorbing potential is configured."""

        return self.cap_profile is not None

    @property
    def metadata(self) -> dict:
        """Return JSON-compatible absorber metadata."""

        return {
            "mask": (
                None
                if self.mask_profile is None
                else self.mask_profile.metadata(self.representation)
            ),
            "cap": (
                None
                if self.cap_profile is None
                else self.cap_profile.metadata(self.representation)
            ),
        }

    def mask_values(self, m=0) -> np.ndarray | None:
        """Return the read-only active-grid mask for signed ``m``."""

        if self._full_mask is None:
            return None
        m = self.representation.validate_m(m)
        if m not in self._active_masks:
            values = self.representation.restrict_full_grid(
                self._full_mask, m
            ).copy()
            values.setflags(write=False)
            self._active_masks[m] = values
        return self._active_masks[m]

    def cap_values(self, m=0) -> np.ndarray | None:
        """Return the read-only active-grid CAP for signed ``m``."""

        if self._full_cap is None:
            return None
        m = self.representation.validate_m(m)
        if m not in self._active_caps:
            values = self.representation.restrict_full_grid(
                self._full_cap, m
            ).copy()
            values.setflags(write=False)
            self._active_caps[m] = values
        return self._active_caps[m]

    def apply_mask(self, state, time: float, m=0) -> np.ndarray:
        """Apply the configured mask once after an accepted step."""

        del time
        values = self.representation.validate_fixed_m_state(state, m)
        mask = self.mask_values(m)
        return values.copy() if mask is None else values * mask

    def cap_action(self, state, m=0) -> np.ndarray:
        """Return the diagonal complex-absorbing-potential action."""

        values = self.representation.validate_fixed_m_state(state, m)
        cap = self.cap_values(m)
        if cap is None:
            return np.zeros_like(values, dtype=np.result_type(values, complex))
        return values * cap

    def validate_integrator(self, integrator: str) -> None:
        """Require an Arnoldi exponential method when a CAP is active."""

        if self.has_cap and integrator not in {"sia", "cf4-sia"}:
            raise ValueError(
                "a prolate cubic CAP requires integrator 'sia' or "
                "'cf4-sia'"
            )
