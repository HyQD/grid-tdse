"""Crank--Nicolson preconditioning for spherical one-radial states."""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .hamiltonian import OneElectronDiatomicHamiltonian


class SphericalOneRadialKineticCrankNicolsonPreconditioner:
    r"""Invert the channel-diagonal kinetic part of the CN left-hand side.

    For every retained ``(l, m)`` channel, this preconditioner factorizes

    .. math::

        I + \frac{i\Delta t}{2}
        \left[-\frac{1}{2\mu}\frac{d^2}{dr^2}
        + \frac{l(l+1)}{2\mu r^2}\right].

    The nuclear potential, laser interaction, and absorber are omitted.  The
    approximation therefore preserves the current radial-first fixed-``m``
    representation while avoiding a dense factorization across all angular
    channels.  It is intended for Hermitian CN propagation, optionally
    followed by a radial mask; CN with a CAP remains a separate validation
    gate.
    """

    name = "kinetic-channel-lu"

    def __init__(
        self,
        hamiltonian: OneElectronDiatomicHamiltonian,
        *,
        m: int = 0,
        time_step: Real,
    ) -> None:
        if not isinstance(hamiltonian, OneElectronDiatomicHamiltonian):
            raise TypeError(
                "hamiltonian must be a OneElectronDiatomicHamiltonian"
            )
        if isinstance(time_step, (bool, np.bool_)) or not isinstance(
            time_step, Real
        ):
            raise TypeError("time_step must be a real number")
        time_step = float(time_step)
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be finite and positive")

        self.hamiltonian = hamiltonian
        self.m = hamiltonian.validate_m(m)
        self.time_step = time_step
        self._shape = hamiltonian.fixed_m_shape(self.m)
        self._l_values = tuple(range(abs(self.m), hamiltonian.l_max + 1))
        self._factors = self._factorize_channels()
        self.applications = 0

    def _factorize_channels(self):
        hamiltonian = self.hamiltonian
        mass = hamiltonian.system.particle_mass
        identity = np.eye(hamiltonian.r.size, dtype=complex)
        radial_kinetic = -0.5 / mass * hamiltonian.second_derivative
        factors = []
        for l_value in self._l_values:
            centrifugal = l_value * (l_value + 1) / (
                2.0 * mass * hamiltonian.r**2
            )
            kinetic = radial_kinetic + np.diag(centrifugal)
            left = identity + 0.5j * self.time_step * kinetic
            factors.append(lu_factor(left, check_finite=False))
        return tuple(factors)

    def __call__(self, state) -> np.ndarray:
        """Apply the channel-local approximate inverse to one or more states."""

        values = self.hamiltonian.validate_fixed_m_state(state, self.m)
        result = np.empty_like(values, dtype=np.result_type(values, complex))
        for angular_index, factor in enumerate(self._factors):
            channel = values[..., :, angular_index]
            radial_first = np.moveaxis(channel, -1, 0)
            solved = lu_solve(
                factor,
                radial_first.reshape((self._shape[0], -1)),
                check_finite=False,
            ).reshape(radial_first.shape)
            result[..., :, angular_index] = np.moveaxis(solved, 0, -1)
        self.applications += 1
        return result

    def validate_propagator(self, propagator) -> None:
        """Reject a preconditioner built for another CN step size."""

        if propagator.integrator != "crank-nicolson":
            raise ValueError(
                "the kinetic CN preconditioner requires integrator "
                "'crank-nicolson'"
            )
        if not np.isclose(
            self.time_step,
            propagator.time_step,
            rtol=0.0,
            atol=16.0
            * np.finfo(float).eps
            * max(1.0, abs(self.time_step)),
        ):
            raise ValueError(
                "preconditioner time_step does not match the effective "
                "propagator time step"
            )

    @property
    def metadata(self) -> dict:
        """Return JSON-compatible construction metadata."""

        return {
            "type": type(self).__name__,
            "name": self.name,
            "approximation": "radial_kinetic_plus_centrifugal",
            "omitted_terms": [
                "nuclear_potential",
                "field_interaction",
                "absorbing_boundary",
            ],
            "factorization": "per_channel_dense_lu",
            "m": self.m,
            "l_values": list(self._l_values),
            "time_step": self.time_step,
            "particle_mass": self.hamiltonian.system.particle_mass,
            "active_radial_points": int(self.hamiltonian.r.size),
            "state_axis_order": "radial_angular",
        }
