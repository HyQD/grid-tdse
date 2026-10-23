"""Ground-state iteration by normalized imaginary-time propagation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from integrator_lib import CrankNicolson
from scipy.sparse.linalg import LinearOperator


@dataclass(frozen=True)
class GroundStateIteration:
    """Diagnostics after one normalized imaginary-time step."""

    iteration: int
    imaginary_time: float
    energy: float
    energy_change: float
    state_change: float
    residual_norm: float
    norm: float
    linear_iterations: int
    linear_residual_norm: float


@dataclass(frozen=True)
class GroundStateResult:
    """Final state and convergence diagnostics from imaginary-time iteration."""

    state: np.ndarray
    energy: float
    state_change: float
    residual_norm: float
    converged: bool
    iterations: int
    imaginary_time: float
    termination_reason: Literal["converged", "maximum_iterations"]
    history: tuple[GroundStateIteration, ...]

    @property
    def energy_history(self) -> np.ndarray:
        """Energy after every accepted imaginary-time step."""

        return np.asarray([record.energy for record in self.history])

    @property
    def residual_history(self) -> np.ndarray:
        """Normalized eigenstate residual after every accepted step."""

        return np.asarray([record.residual_norm for record in self.history])

    @property
    def state_change_history(self) -> np.ndarray:
        """Phase-invariant state change after every accepted step."""

        return np.asarray([record.state_change for record in self.history])

    @property
    def norm(self) -> float:
        """Physical norm of the returned state."""

        return self.history[-1].norm


class ImaginaryTimeGroundStateSolver:
    """Find the lowest state represented in an initial wavefunction.

    The solver knows nothing about the shape or spatial representation of the
    state. ``hamiltonian(state)`` supplies the time-independent Hamiltonian
    action and ``inner_product(left, right)`` supplies the representation's
    physical inner product.

    Each iteration advances ``d psi / d tau = -H psi`` by one matrix-free
    Crank--Nicolson step, normalizes the result, and evaluates the energy
    change, phase-invariant state change, and normalized eigenstate residual.
    """

    _LINEAR_SOLVERS = {"gmres", "bicgstab"}

    def __init__(
        self,
        *,
        time_step: float = 0.1,
        max_iterations: int = 1_000,
        energy_tolerance: float = 1.0e-10,
        state_tolerance: float = 1.0e-8,
        residual_tolerance: float = 1.0e-8,
        linear_solver: str = "gmres",
        linear_rtol: float = 1.0e-10,
        linear_atol: float = 0.0,
        linear_max_iterations: int | None = None,
        gmres_restart: int | None = None,
        reality_tolerance: float = 1.0e-10,
    ) -> None:
        self.time_step = _positive_real(time_step, "time_step")
        self.max_iterations = _positive_integer(max_iterations, "max_iterations")
        self.energy_tolerance = _positive_real(
            energy_tolerance, "energy_tolerance"
        )
        self.state_tolerance = _positive_real(state_tolerance, "state_tolerance")
        self.residual_tolerance = _positive_real(
            residual_tolerance, "residual_tolerance"
        )
        if linear_solver not in self._LINEAR_SOLVERS:
            choices = ", ".join(sorted(self._LINEAR_SOLVERS))
            raise ValueError(f"linear_solver must be one of: {choices}")
        self.linear_solver = linear_solver
        self.linear_rtol = _nonnegative_real(linear_rtol, "linear_rtol")
        self.linear_atol = _nonnegative_real(linear_atol, "linear_atol")
        if self.linear_rtol == self.linear_atol == 0.0:
            raise ValueError("at least one linear tolerance must be positive")
        self.linear_max_iterations = _optional_positive_integer(
            linear_max_iterations, "linear_max_iterations"
        )
        self.gmres_restart = _optional_positive_integer(
            gmres_restart, "gmres_restart"
        )
        self.reality_tolerance = _nonnegative_real(
            reality_tolerance, "reality_tolerance"
        )

    def solve(
        self,
        initial_state: np.ndarray,
        hamiltonian: Callable[[np.ndarray], np.ndarray],
        inner_product: Callable[[np.ndarray, np.ndarray], complex],
        *,
        preconditioner: (
            Callable[[np.ndarray], np.ndarray] | LinearOperator | None
        ) = None,
    ) -> GroundStateResult:
        """Run normalized imaginary-time propagation.

        Parameters
        ----------
        initial_state
            Nonzero initial wavefunction. It is copied and normalized.
        hamiltonian
            Linear, time-independent Hamiltonian action preserving the state
            shape.
        inner_product
            Physical inner product for the representation.
        preconditioner
            Optional shaped callable or SciPy ``LinearOperator`` forwarded to
            :class:`integrator_lib.CrankNicolson`.
        """

        if not callable(hamiltonian):
            raise TypeError("hamiltonian must be callable")
        if not callable(inner_product):
            raise TypeError("inner_product must be callable")

        state = _state_array(initial_state)
        shape = state.shape
        state = self._normalize(state, inner_product)
        initial_action = _hamiltonian_action(hamiltonian, state, shape)
        previous_energy = self._energy(state, initial_action, inner_product)

        integrator = CrankNicolson(
            solver=self.linear_solver,
            rtol=self.linear_rtol,
            atol=self.linear_atol,
            maxiter=self.linear_max_iterations,
            restart=self.gmres_restart,
            preconditioner=preconditioner,
        )

        def imaginary_time_rhs(values, time):
            del time
            return -_hamiltonian_action(hamiltonian, values, shape)

        history = []
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            imaginary_time = iteration * self.time_step
            previous_state = state
            state = integrator(
                state,
                imaginary_time - self.time_step,
                self.time_step,
                imaginary_time_rhs,
            )
            state = self._normalize(state, inner_product)
            state_change = self._state_change(
                previous_state, state, inner_product
            )
            action = _hamiltonian_action(hamiltonian, state, shape)
            energy = self._energy(state, action, inner_product)
            energy_change = abs(energy - previous_energy)
            residual = action - energy * state
            residual_norm = self._norm(residual, inner_product)
            norm = self._norm(state, inner_product)
            record = GroundStateIteration(
                iteration=iteration,
                imaginary_time=imaginary_time,
                energy=energy,
                energy_change=energy_change,
                state_change=state_change,
                residual_norm=residual_norm,
                norm=norm,
                linear_iterations=int(integrator.last_iterations),
                linear_residual_norm=float(integrator.last_residual_norm),
            )
            history.append(record)

            if (
                energy_change <= self.energy_tolerance
                and state_change <= self.state_tolerance
                and residual_norm <= self.residual_tolerance
            ):
                converged = True
                break
            previous_energy = energy

        final = history[-1]
        return GroundStateResult(
            state=state.copy(),
            energy=final.energy,
            state_change=final.state_change,
            residual_norm=final.residual_norm,
            converged=converged,
            iterations=final.iteration,
            imaginary_time=final.imaginary_time,
            termination_reason=("converged" if converged else "maximum_iterations"),
            history=tuple(history),
        )

    __call__ = solve

    def _normalize(self, state, inner_product):
        norm = self._norm(state, inner_product)
        if norm == 0.0:
            raise ValueError("state has zero norm")
        return state / norm

    def _norm(self, state, inner_product) -> float:
        norm_squared = _inner_product_value(inner_product, state, state)
        value = self._real_value(norm_squared, "squared norm")
        if value < 0.0:
            raise ValueError("inner product produced a negative squared norm")
        return float(np.sqrt(value))

    def _energy(self, state, action, inner_product) -> float:
        denominator = _inner_product_value(inner_product, state, state)
        norm_squared = self._real_value(denominator, "squared norm")
        if norm_squared <= 0.0:
            raise ValueError("state must have positive norm")
        numerator = _inner_product_value(inner_product, state, action)
        return self._real_value(numerator / denominator, "energy")

    def _state_change(self, previous, current, inner_product) -> float:
        overlap = _inner_product_value(inner_product, previous, current)
        if abs(overlap) == 0.0:
            aligned = current
        else:
            aligned = current * overlap.conjugate() / abs(overlap)
        return self._norm(aligned - previous, inner_product)

    def _real_value(self, value: complex, name: str) -> float:
        scale = max(1.0, abs(value.real))
        if abs(value.imag) > self.reality_tolerance * scale:
            raise ValueError(
                f"{name} has a significant imaginary part: {value.imag}"
            )
        return float(value.real)


def _state_array(state) -> np.ndarray:
    values = np.asarray(state)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("initial_state must be a nonempty array")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("initial_state must contain numeric values")
    if not np.isfinite(values).all():
        raise ValueError("initial_state must contain only finite values")
    return values.copy()


def _hamiltonian_action(hamiltonian, state, shape) -> np.ndarray:
    action = np.asarray(hamiltonian(state))
    if action.shape != shape:
        raise ValueError(
            f"hamiltonian returned shape {action.shape}; expected {shape}"
        )
    if not np.issubdtype(action.dtype, np.number):
        raise TypeError("hamiltonian must return numeric values")
    if not np.isfinite(action).all():
        raise ValueError("hamiltonian returned non-finite values")
    return action


def _inner_product_value(inner_product, left, right) -> complex:
    value = np.asarray(inner_product(left, right))
    if value.shape != ():
        raise ValueError("inner_product must return a scalar")
    result = complex(value)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("inner_product returned a non-finite value")
    return result


def _positive_real(value, name: str) -> float:
    result = _nonnegative_real(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)
