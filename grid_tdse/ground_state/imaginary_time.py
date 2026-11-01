"""Ground-state iteration by normalized imaginary-time propagation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from integrator_lib import (
    ArnoldiExponential,
    CrankNicolson,
    ExponentialEuler,
    LanczosPhi1,
    ShortIterativeLanczos,
)
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
    step_wall_time: float = 0.0
    linear_iterations: int = 0
    linear_residual_norm: float = 0.0
    krylov_operator_calls: int = 0
    krylov_accepted_substeps: int = 0
    krylov_rejected_substeps: int = 0
    krylov_refinement_depth: int = 0
    krylov_dimension: int = 0
    krylov_error_estimate: float = 0.0


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
    integrator: Literal[
        "crank-nicolson",
        "sil",
        "sia",
        "exponential-euler",
    ]
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

    @property
    def linear_iterations(self) -> np.ndarray:
        """Linear iterations used by every Crank--Nicolson step."""

        return np.asarray(
            [record.linear_iterations for record in self.history], dtype=int
        )

    @property
    def krylov_operator_calls(self) -> np.ndarray:
        """Hamiltonian calls made by the exponential integrator per step."""

        return np.asarray(
            [record.krylov_operator_calls for record in self.history], dtype=int
        )

    @property
    def krylov_dimensions(self) -> np.ndarray:
        """Largest accepted exponential Krylov dimension in every step."""

        return np.asarray(
            [record.krylov_dimension for record in self.history], dtype=int
        )

    @property
    def krylov_accepted_substeps(self) -> np.ndarray:
        """Accepted exponential substeps during every ITP step."""

        return np.asarray(
            [record.krylov_accepted_substeps for record in self.history],
            dtype=int,
        )

    @property
    def krylov_rejected_substeps(self) -> np.ndarray:
        """Rejected exponential substeps during every ITP step."""

        return np.asarray(
            [record.krylov_rejected_substeps for record in self.history],
            dtype=int,
        )

    @property
    def krylov_refinement_depths(self) -> np.ndarray:
        """Maximum adaptive refinement depth during every ITP step."""

        return np.asarray(
            [record.krylov_refinement_depth for record in self.history],
            dtype=int,
        )

    @property
    def krylov_error_estimates(self) -> np.ndarray:
        """Accumulated accepted-substep error estimate for every ITP step."""

        return np.asarray(
            [record.krylov_error_estimate for record in self.history],
            dtype=float,
        )


class ImaginaryTimeGroundStateSolver:
    """Find the lowest state represented in an initial wavefunction.

    The solver knows nothing about the shape or spatial representation of the
    state. ``hamiltonian(state)`` supplies the time-independent Hamiltonian
    action and ``inner_product(left, right)`` supplies the representation's
    physical inner product.

    Each iteration advances ``d psi / d tau = -H psi`` by one matrix-free
    Crank--Nicolson, short-iterative Lanczos (SIL), short-iterative Arnoldi
    (SIA), or exponential-Euler step. Exponential Euler applies
    ``h * phi_1(-h H)`` to the energy-shifted stationary residual
    ``-(H - E[state]) state``. Every method normalizes the result and
    evaluates the energy change, phase-invariant state change, and normalized
    eigenstate residual.
    """

    _INTEGRATORS = {
        "crank-nicolson",
        "sil",
        "sia",
        "exponential-euler",
    }
    _LINEAR_SOLVERS = {"gmres", "bicgstab"}

    def __init__(
        self,
        *,
        time_step: float = 0.1,
        max_iterations: int = 1_000,
        energy_tolerance: float = 1.0e-10,
        state_tolerance: float = 1.0e-8,
        residual_tolerance: float = 1.0e-8,
        integrator: Literal[
            "crank-nicolson",
            "sil",
            "sia",
            "exponential-euler",
        ] = "crank-nicolson",
        linear_solver: str = "gmres",
        linear_rtol: float = 1.0e-10,
        linear_atol: float = 0.0,
        linear_max_iterations: int | None = None,
        gmres_restart: int | None = None,
        max_krylov_dimension: int = 20,
        krylov_rtol: float = 1.0e-10,
        krylov_atol: float = 0.0,
        max_krylov_refinements: int = 8,
        krylov_reorthogonalize: bool = True,
        krylov_breakdown_tolerance: float = 1.0e-14,
        reality_tolerance: float = 1.0e-10,
        progress_interval: int | None = 10,
    ) -> None:
        self.time_step = _positive_real(time_step, "time_step")
        self.max_iterations = _positive_integer(max_iterations, "max_iterations")
        self.energy_tolerance = _positive_real(energy_tolerance, "energy_tolerance")
        self.state_tolerance = _positive_real(state_tolerance, "state_tolerance")
        self.residual_tolerance = _positive_real(
            residual_tolerance, "residual_tolerance"
        )
        if integrator not in self._INTEGRATORS:
            choices = ", ".join(sorted(self._INTEGRATORS))
            raise ValueError(f"integrator must be one of: {choices}")
        self.integrator = integrator
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
        self.max_krylov_dimension = _positive_integer(
            max_krylov_dimension, "max_krylov_dimension"
        )
        self.krylov_rtol = _nonnegative_real(krylov_rtol, "krylov_rtol")
        self.krylov_atol = _nonnegative_real(krylov_atol, "krylov_atol")
        if self.krylov_rtol == self.krylov_atol == 0.0:
            raise ValueError("at least one Krylov tolerance must be positive")
        self.max_krylov_refinements = _nonnegative_integer(
            max_krylov_refinements, "max_krylov_refinements"
        )
        if not isinstance(krylov_reorthogonalize, (bool, np.bool_)):
            raise TypeError("krylov_reorthogonalize must be boolean")
        self.krylov_reorthogonalize = bool(krylov_reorthogonalize)
        self.krylov_breakdown_tolerance = _positive_real(
            krylov_breakdown_tolerance,
            "krylov_breakdown_tolerance",
        )
        self.reality_tolerance = _nonnegative_real(
            reality_tolerance, "reality_tolerance"
        )
        self.progress_interval = _optional_positive_integer(
            progress_interval, "progress_interval"
        )

    def solve(
        self,
        initial_state: np.ndarray,
        hamiltonian: Callable[[np.ndarray], np.ndarray],
        inner_product: Callable[[np.ndarray, np.ndarray], complex],
        *,
        inner_products: Callable[[Sequence[np.ndarray], np.ndarray], np.ndarray]
        | None = None,
        preconditioner: (
            Callable[[np.ndarray], np.ndarray] | LinearOperator | None
        ) = None,
        iteration_observer: (
            Callable[[np.ndarray, GroundStateIteration], None] | None
        ) = None,
        observer_interval: int | None = None,
        print_progress: bool = True,
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
        inner_products
            Optional batched physical inner product. It receives a sequence
            of left states and one right state and is forwarded to SIL or SIA
            to reduce orthogonalization synchronization. It is also forwarded
            to the exponential-Euler phi-function action.
        preconditioner
            Optional shaped callable or SciPy ``LinearOperator`` forwarded to
            :class:`integrator_lib.CrankNicolson`. It is rejected for SIL,
            SIA, and exponential Euler.
        iteration_observer
            Optional callback receiving a read-only normalized state and its
            diagnostics at iteration zero, every ``observer_interval`` steps,
            and the final step.
        observer_interval
            Positive sampling interval. ``None`` requests only the initial and
            final states when an observer is supplied.
        print_progress
            Whether the solver prints its compact built-in progress line.
        """

        if not callable(hamiltonian):
            raise TypeError("hamiltonian must be callable")
        if not callable(inner_product):
            raise TypeError("inner_product must be callable")
        if inner_products is not None and not callable(inner_products):
            raise TypeError("inner_products must be callable or None")
        if iteration_observer is not None and not callable(iteration_observer):
            raise TypeError("iteration_observer must be callable or None")
        observer_interval = _optional_positive_integer(
            observer_interval, "observer_interval"
        )
        if not isinstance(print_progress, (bool, np.bool_)):
            raise TypeError("print_progress must be boolean")

        state = _state_array(initial_state)
        shape = state.shape
        state = self._normalize(state, inner_product)
        initial_action = _hamiltonian_action(hamiltonian, state, shape)
        previous_energy = self._energy(state, initial_action, inner_product)
        initial_residual = initial_action - previous_energy * state
        if iteration_observer is not None:
            initial_record = GroundStateIteration(
                iteration=0,
                imaginary_time=0.0,
                energy=previous_energy,
                energy_change=0.0,
                state_change=0.0,
                residual_norm=self._norm(initial_residual, inner_product),
                norm=self._norm(state, inner_product),
            )
            _notify_iteration_observer(
                iteration_observer,
                state,
                initial_record,
            )

        integrator = self._build_integrator(
            inner_product,
            inner_products,
            preconditioner,
        )

        def imaginary_time_rhs(values, time):
            del time
            return -_hamiltonian_action(hamiltonian, values, shape)

        def stationary_residual(values, time):
            del time
            action = _hamiltonian_action(hamiltonian, values, shape)
            energy = self._energy(values, action, inner_product)
            return -(action - energy * values)

        history = []
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            step_started = perf_counter()
            imaginary_time = iteration * self.time_step
            previous_state = state
            if self.integrator == "exponential-euler":
                state = integrator(
                    state,
                    imaginary_time - self.time_step,
                    self.time_step,
                    stationary_residual,
                    imaginary_time_rhs,
                )
            else:
                state = integrator(
                    state,
                    imaginary_time - self.time_step,
                    self.time_step,
                    imaginary_time_rhs,
                )
            state = self._normalize(state, inner_product)
            state_change = self._state_change(previous_state, state, inner_product)
            action = _hamiltonian_action(hamiltonian, state, shape)
            energy = self._energy(state, action, inner_product)
            energy_change = abs(energy - previous_energy)
            residual = action - energy * state
            residual_norm = self._norm(residual, inner_product)
            norm = self._norm(state, inner_product)
            step_wall_time = perf_counter() - step_started
            record = GroundStateIteration(
                iteration=iteration,
                imaginary_time=imaginary_time,
                energy=energy,
                energy_change=energy_change,
                state_change=state_change,
                residual_norm=residual_norm,
                norm=norm,
                step_wall_time=step_wall_time,
                **self._step_diagnostics(integrator),
            )
            history.append(record)

            converged = (
                energy_change <= self.energy_tolerance
                and state_change <= self.state_tolerance
                and residual_norm <= self.residual_tolerance
            )
            final_iteration = converged or iteration == self.max_iterations
            if print_progress and self.progress_interval is not None and (
                iteration % self.progress_interval == 0
                or final_iteration
            ):
                self._print_progress(record)
            if iteration_observer is not None and (
                final_iteration
                or (
                    observer_interval is not None
                    and iteration % observer_interval == 0
                )
            ):
                _notify_iteration_observer(iteration_observer, state, record)
            if converged:
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
            integrator=self.integrator,
            termination_reason=("converged" if converged else "maximum_iterations"),
            history=tuple(history),
        )

    __call__ = solve

    def _build_integrator(self, inner_product, inner_products, preconditioner):
        if self.integrator == "crank-nicolson":
            return CrankNicolson(
                solver=self.linear_solver,
                rtol=self.linear_rtol,
                atol=self.linear_atol,
                maxiter=self.linear_max_iterations,
                restart=self.gmres_restart,
                preconditioner=preconditioner,
            )
        if preconditioner is not None:
            raise ValueError(
                f"preconditioner is not supported by integrator "
                f"{self.integrator!r}"
            )
        common = {
            "max_krylov_dimension": self.max_krylov_dimension,
            "rtol": self.krylov_rtol,
            "atol": self.krylov_atol,
            "max_refinements": self.max_krylov_refinements,
            "reorthogonalize": self.krylov_reorthogonalize,
            "inner_product": inner_product,
            "inner_products": inner_products,
            "breakdown_tolerance": self.krylov_breakdown_tolerance,
        }
        if self.integrator == "sil":
            return ShortIterativeLanczos(generator_factor=-1.0, **common)
        if self.integrator == "sia":
            return ArnoldiExponential(**common)
        phi1_options = dict(common)
        phi1_options.pop("max_refinements")
        return ExponentialEuler(
            LanczosPhi1(generator_factor=-1.0, **phi1_options)
        )

    def _step_diagnostics(self, integrator) -> dict[str, int | float]:
        if self.integrator == "crank-nicolson":
            return {
                "linear_iterations": int(integrator.last_iterations),
                "linear_residual_norm": float(integrator.last_residual_norm),
            }
        if self.integrator == "exponential-euler":
            attempts = integrator.last_phi1_attempts
            error_estimate = integrator.last_phi1_error_estimate
            return {
                "krylov_operator_calls": int(
                    integrator.last_phi1_operator_calls
                ),
                "krylov_accepted_substeps": sum(
                    attempt.converged for attempt in attempts
                ),
                "krylov_rejected_substeps": sum(
                    not attempt.converged for attempt in attempts
                ),
                "krylov_refinement_depth": 0,
                "krylov_dimension": int(
                    integrator.last_phi1_krylov_dimension
                ),
                "krylov_error_estimate": (
                    0.0
                    if error_estimate is None
                    else float(error_estimate)
                ),
            }
        return {
            "krylov_operator_calls": int(integrator.last_operator_calls),
            "krylov_accepted_substeps": int(integrator.last_accepted_substeps),
            "krylov_rejected_substeps": int(integrator.last_rejected_substeps),
            "krylov_refinement_depth": int(integrator.last_refinement_depth),
            "krylov_dimension": int(integrator.last_krylov_dimension),
            "krylov_error_estimate": float(integrator.last_error_estimate),
        }

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
            raise ValueError(f"{name} has a significant imaginary part: {value.imag}")
        return float(value.real)

    @staticmethod
    def _print_progress(record: GroundStateIteration) -> None:
        print(
            f"ITP {record.iteration:6d}: "
            f"E={record.energy:.12f}  "
            f"|dE|={record.energy_change:.3e}  "
            f"dPsi={record.state_change:.3e}  "
            f"residual={record.residual_norm:.3e}"
        )


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
        raise ValueError(f"hamiltonian returned shape {action.shape}; expected {shape}")
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


def _notify_iteration_observer(observer, state, record) -> None:
    view = state.view()
    view.flags.writeable = False
    observer(view, record)


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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)
