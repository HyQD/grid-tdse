"""Common diagnostics for direct ground-state eigenvalue solvers."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


EigenTerminationReason = Literal[
    "converged",
    "maximum_iterations",
    "maximum_hamiltonian_actions",
    "invariant_subspace",
    "numerical_breakdown",
]


@dataclass(frozen=True)
class GroundStateEigenIteration:
    """Diagnostics from one direct-eigensolver subspace iteration."""

    iteration: int
    eigenvalue: float
    eigenvalue_change: float
    state_change: float
    residual_norm: float
    subspace_dimension: int
    orthogonality_error: float
    restarted: bool = False
    hamiltonian_actions: int = 0
    preconditioner_applications: int = 0
    preconditioner_fallbacks: int = 0
    collective_reductions: int = 0
    reduced_values: int = 0
    wall_time: float = 0.0

    @property
    def energy(self) -> float:
        """Alias used by representation-level ground-state reporters."""

        return self.eigenvalue

    @property
    def energy_change(self) -> float:
        """Alias used by representation-level ground-state reporters."""

        return self.eigenvalue_change

    @property
    def norm(self) -> float:
        """Physical norm of the normalized Ritz vector."""

        return 1.0

    @property
    def step_wall_time(self) -> float:
        """Alias for the complete direct-eigensolver iteration time."""

        return self.wall_time


@dataclass(frozen=True)
class GroundStateEigenResult:
    """Final state and diagnostics from a direct ground-state eigensolver."""

    state: np.ndarray
    eigenvalue: float
    residual_norm: float
    converged: bool
    iterations: int
    restarts: int
    hamiltonian_actions: int
    preconditioner_applications: int
    termination_reason: EigenTerminationReason
    history: tuple[GroundStateEigenIteration, ...]
    solver: Literal["lanczos", "davidson"] = "lanczos"

    @property
    def energy(self) -> float:
        """Ground-state energy, matching the ITP result interface."""

        return self.eigenvalue

    @property
    def state_change(self) -> float:
        """Final phase-aligned state change."""

        if not self.history:
            return 0.0
        return self.history[-1].state_change

    @property
    def norm(self) -> float:
        """Physical norm of the returned Ritz vector."""

        return 1.0

    @property
    def energy_history(self) -> np.ndarray:
        """Lowest Ritz value after every recorded iteration."""

        return self.eigenvalue_history

    @property
    def eigenvalue_history(self) -> np.ndarray:
        """Lowest Ritz value after every recorded iteration."""

        return np.asarray([record.eigenvalue for record in self.history])

    @property
    def residual_history(self) -> np.ndarray:
        """Physical Ritz-residual norm after every recorded iteration."""

        return np.asarray([record.residual_norm for record in self.history])

    @property
    def state_change_history(self) -> np.ndarray:
        """Phase-aligned state change after every recorded iteration."""

        return np.asarray([record.state_change for record in self.history])

    @property
    def subspace_dimensions(self) -> np.ndarray:
        """Subspace dimension after every recorded iteration."""

        return np.asarray(
            [record.subspace_dimension for record in self.history],
            dtype=int,
        )

    @property
    def maximum_subspace_dimension(self) -> int:
        """Largest represented subspace dimension reached by the solver."""

        if not self.history:
            return 0
        return max(record.subspace_dimension for record in self.history)

    @property
    def final_subspace_dimension(self) -> int:
        """Subspace dimension used for the final recorded Ritz extraction."""

        if not self.history:
            return 0
        return self.history[-1].subspace_dimension

    @property
    def restart_iterations(self) -> np.ndarray:
        """Iterations after which a thick restart was performed."""

        return np.asarray(
            [record.iteration for record in self.history if record.restarted],
            dtype=int,
        )

    @property
    def preconditioner_fallbacks(self) -> int:
        """Number of rejected corrections replaced by raw residuals."""

        if not self.history:
            return 0
        return self.history[-1].preconditioner_fallbacks


__all__ = [
    "EigenTerminationReason",
    "GroundStateEigenIteration",
    "GroundStateEigenResult",
]
