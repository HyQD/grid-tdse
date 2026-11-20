"""Radial absorbing boundaries for spherical one-radial states."""

from __future__ import annotations

import numpy as np
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from .hamiltonian import OneElectronDiatomicHamiltonian


class SphericalOneRadialAbsorbingBoundaries:
    """Prepared radial cosine mask or cubic CAP on a spherical grid.

    A mask is a post-step transformation, whereas a CAP contributes to the
    propagation Hamiltonian.  Configuring both at once is therefore rejected
    explicitly instead of assigning an implicit precedence.
    """

    def __init__(
        self,
        hamiltonian: OneElectronDiatomicHamiltonian,
        *,
        mask: RadialCosineMask | None = None,
        cap: CubicCAP | None = None,
    ) -> None:
        if not isinstance(hamiltonian, OneElectronDiatomicHamiltonian):
            raise TypeError(
                "hamiltonian must be a OneElectronDiatomicHamiltonian"
            )
        if mask is not None and not isinstance(mask, RadialCosineMask):
            raise TypeError("mask must be a RadialCosineMask")
        if cap is not None and not isinstance(cap, CubicCAP):
            raise TypeError("cap must be a CubicCAP")
        if mask is not None and cap is not None:
            raise ValueError("mask and cap cannot both be configured")

        self.hamiltonian = hamiltonian
        self.mask_profile = mask
        self.cap_profile = cap
        self.r_max = float(hamiltonian.nodes[-1])
        self._mask_values = self._evaluate_mask(mask)
        self._cap_values = self._evaluate_cap(cap)

    @property
    def has_mask(self) -> bool:
        """Return whether a post-step radial mask is configured."""

        return self.mask_profile is not None

    @property
    def has_cap(self) -> bool:
        """Return whether a cubic complex absorbing potential is configured."""

        return self.cap_profile is not None

    @property
    def metadata(self) -> dict:
        """Return JSON-compatible absorber and radial-grid metadata."""

        return {
            "coordinate": "spherical_radius_r",
            "r_max": self.r_max,
            "mask": (
                None
                if self.mask_profile is None
                else {
                    "module": type(self.mask_profile).__module__,
                    "type": type(self.mask_profile).__name__,
                    "formula": (
                        "cos(pi*(r-r0)/(2*(r_max-r0)))**(1/n)"
                    ),
                    "r0": self.mask_profile.r0,
                    "n": self.mask_profile.n,
                    "outer_value": 0.0,
                }
            ),
            "cap": (
                None
                if self.cap_profile is None
                else {
                    "module": type(self.cap_profile).__module__,
                    "type": type(self.cap_profile).__name__,
                    "formula": (
                        "-1j * eta * max(r-alpha*r_max, 0)**3"
                    ),
                    "alpha": self.cap_profile.alpha,
                    "eta": self.cap_profile.eta,
                    "r_onset": self.cap_profile.alpha * self.r_max,
                }
            ),
        }

    def mask_values(self) -> np.ndarray | None:
        """Return the cached read-only mask on active radial nodes."""

        return self._mask_values

    def cap_values(self) -> np.ndarray | None:
        """Return the cached read-only CAP on active radial nodes."""

        return self._cap_values

    def apply_mask(self, state, time: float, m=0) -> np.ndarray:
        """Apply the configured mask after one accepted propagation step."""

        del time
        values = self.hamiltonian.validate_fixed_m_state(state, m)
        if self._mask_values is None:
            return values.copy()
        return values * _radial_broadcast(self._mask_values, values.ndim)

    def cap_action(self, state, m=0) -> np.ndarray:
        """Return the diagonal complex-absorbing-potential action."""

        values = self.hamiltonian.validate_fixed_m_state(state, m)
        if self._cap_values is None:
            return np.zeros_like(
                values, dtype=np.result_type(values.dtype, complex)
            )
        return values * _radial_broadcast(self._cap_values, values.ndim)

    def validate_integrator(self, integrator: str) -> None:
        """Enforce the initial non-Hermitian propagation contract."""

        if self.has_cap and integrator not in {"sia", "cf4-sia"}:
            raise ValueError(
                "a spherical one-radial cubic CAP requires integrator "
                "'sia' or 'cf4-sia'"
            )

    def _evaluate_mask(
        self, profile: RadialCosineMask | None
    ) -> np.ndarray | None:
        if profile is None:
            return None
        values = np.asarray(profile(self.hamiltonian.r, self.r_max))
        if values.shape != self.hamiltonian.r.shape:
            raise RuntimeError("grid-lib returned an incompatible mask shape")
        if np.iscomplexobj(values) or np.any((values < 0.0) | (values > 1.0)):
            raise RuntimeError("grid-lib returned invalid radial mask values")
        values = np.array(values, dtype=float, copy=True)
        values.setflags(write=False)
        return values

    def _evaluate_cap(self, profile: CubicCAP | None) -> np.ndarray | None:
        if profile is None:
            return None
        values = np.asarray(
            profile(self.hamiltonian.r, self.r_max), dtype=complex
        )
        if values.shape != self.hamiltonian.r.shape:
            raise RuntimeError("grid-lib returned an incompatible CAP shape")
        if np.any(values.imag > 0.0):
            raise RuntimeError("grid-lib returned an amplifying CAP")
        values = np.array(values, dtype=complex, copy=True)
        values.setflags(write=False)
        return values


def _radial_broadcast(values: np.ndarray, state_ndim: int) -> np.ndarray:
    return values.reshape((1,) * (state_ndim - 2) + (values.size, 1))
