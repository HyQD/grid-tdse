r"""Representation-independent real-time Schrödinger propagation."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from integrator_lib import CrankNicolson
from scipy.sparse.linalg import LinearOperator


@dataclass(frozen=True)
class RealTimePropagationStep:
    """Diagnostics after one accepted real-time step."""

    step: int
    time: float
    norm: float
    linear_iterations: int
    linear_residual_norm: float


@dataclass(frozen=True)
class RealTimePropagationResult:
    """Final state and diagnostics from a completed propagation interval."""

    state: np.ndarray
    initial_time: float
    final_time: float
    requested_time_step: float
    time_step: float
    initial_norm: float
    history: tuple[RealTimePropagationStep, ...]
    samples: dict[str, np.ndarray]

    @property
    def n_steps(self) -> int:
        """Number of accepted propagation steps."""

        return len(self.history)

    @property
    def norm(self) -> float:
        """Physical norm of the returned state."""

        return self.history[-1].norm

    @property
    def times(self) -> np.ndarray:
        """Initial time followed by every accepted-step time."""

        return np.asarray(
            [self.initial_time, *(record.time for record in self.history)]
        )

    @property
    def norm_history(self) -> np.ndarray:
        """Physical norm at the initial time and after every step."""

        return np.asarray(
            [self.initial_norm, *(record.norm for record in self.history)]
        )

    @property
    def linear_iterations(self) -> np.ndarray:
        """Krylov iteration count for every accepted step."""

        return np.asarray(
            [record.linear_iterations for record in self.history], dtype=int
        )

    @property
    def linear_residual_norms(self) -> np.ndarray:
        """Linear-system residual norm for every accepted step."""

        return np.asarray(
            [record.linear_residual_norm for record in self.history]
        )


class RealTimePropagator:
    r"""Advance a linear time-dependent Schrödinger equation.

    The propagator knows nothing about the shape or spatial representation of
    the state. ``hamiltonian(state, time)`` supplies the action of ``H(t)``,
    and ``inner_product(left, right)`` supplies the representation's physical
    inner product.

    The interval is divided into uniform steps no larger than the requested
    time step, with the step size adjusted so that the final time is reached
    exactly. Each step uses matrix-free implicit midpoint/Crank--Nicolson
    propagation from :mod:`integrator_lib`.

    States are never normalized during real-time propagation. No mask,
    complex absorbing potential, or other post-step state transformation is
    applied by this layer.
    """

    _LINEAR_SOLVERS = {"gmres", "bicgstab"}

    def __init__(
        self,
        *,
        initial_time: float = 0.0,
        final_time: float,
        time_step: float,
        linear_solver: str = "gmres",
        linear_rtol: float = 1.0e-10,
        linear_atol: float = 0.0,
        linear_max_iterations: int | None = None,
        gmres_restart: int | None = None,
        reality_tolerance: float = 1.0e-10,
        progress_interval: int | None = 10,
    ) -> None:
        self.initial_time = _finite_real(initial_time, "initial_time")
        self.final_time = _finite_real(final_time, "final_time")
        if self.final_time <= self.initial_time:
            raise ValueError("final_time must be larger than initial_time")
        self.requested_time_step = _positive_real(time_step, "time_step")
        duration = self.final_time - self.initial_time
        self.n_steps = int(np.ceil(duration / self.requested_time_step))
        self.time_step = duration / self.n_steps
        self.propagation_times = np.linspace(
            self.initial_time,
            self.final_time,
            self.n_steps + 1,
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
        self.progress_interval = _optional_positive_integer(
            progress_interval, "progress_interval"
        )

    def propagate(
        self,
        initial_state: np.ndarray,
        hamiltonian: Callable[[np.ndarray, float], np.ndarray],
        inner_product: Callable[[np.ndarray, np.ndarray], complex],
        *,
        preconditioner: (
            Callable[[np.ndarray], np.ndarray] | LinearOperator | None
        ) = None,
        samplers: Mapping[
            str, Callable[[np.ndarray, float], np.ndarray | complex | float]
        ]
        | None = None,
    ) -> RealTimePropagationResult:
        """Propagate ``initial_state`` over the configured time interval.

        Parameters
        ----------
        initial_state
            Nonzero wavefunction. It is copied but not normalized.
        hamiltonian
            Linear Hamiltonian action ``hamiltonian(state, time)`` preserving
            the state shape. The time is the midpoint of the current step.
        inner_product
            Physical inner product for the representation.
        preconditioner
            Optional shaped callable or SciPy ``LinearOperator`` forwarded to
            :class:`integrator_lib.CrankNicolson`.
        samplers
            Optional mapping from result names to ``sampler(state, time)``
            callables. Each sampler is evaluated at the initial time and after
            every accepted step. Samplers must not modify the read-only state
            view they receive.
        """

        if not callable(hamiltonian):
            raise TypeError("hamiltonian must be callable")
        if not callable(inner_product):
            raise TypeError("inner_product must be callable")

        state = _state_array(initial_state)
        shape = state.shape
        initial_norm = self._norm(state, inner_product)
        if initial_norm == 0.0:
            raise ValueError("initial_state has zero norm")
        sampler_functions = _sampler_functions(samplers)
        sample_storage = {name: [] for name in sampler_functions}
        _sample(state, self.initial_time, sampler_functions, sample_storage)

        integrator = CrankNicolson(
            solver=self.linear_solver,
            rtol=self.linear_rtol,
            atol=self.linear_atol,
            maxiter=self.linear_max_iterations,
            restart=self.gmres_restart,
            preconditioner=preconditioner,
        )

        def schrodinger_rhs(values, time):
            action = _hamiltonian_action(hamiltonian, values, time, shape)
            return -1j * action

        history = []
        for step, time in enumerate(self.propagation_times[:-1], start=1):
            state = integrator(
                state,
                time,
                self.time_step,
                schrodinger_rhs,
            )
            record = RealTimePropagationStep(
                step=step,
                time=float(self.propagation_times[step]),
                norm=self._norm(state, inner_product),
                linear_iterations=int(integrator.last_iterations),
                linear_residual_norm=float(integrator.last_residual_norm),
            )
            history.append(record)
            _sample(
                state,
                record.time,
                sampler_functions,
                sample_storage,
            )
            if self.progress_interval is not None and (
                step % self.progress_interval == 0 or step == self.n_steps
            ):
                self._print_progress(record, self.n_steps)

        return RealTimePropagationResult(
            state=state.copy(),
            initial_time=self.initial_time,
            final_time=self.final_time,
            requested_time_step=self.requested_time_step,
            time_step=self.time_step,
            initial_norm=initial_norm,
            history=tuple(history),
            samples={
                name: np.asarray(values)
                for name, values in sample_storage.items()
            },
        )

    __call__ = propagate

    def _norm(self, state, inner_product) -> float:
        norm_squared = _inner_product_value(inner_product, state, state)
        scale = max(1.0, abs(norm_squared.real))
        if abs(norm_squared.imag) > self.reality_tolerance * scale:
            raise ValueError(
                "squared norm has a significant imaginary part: "
                f"{norm_squared.imag}"
            )
        if norm_squared.real < 0.0:
            raise ValueError("inner product produced a negative squared norm")
        return float(norm_squared.real)

    @staticmethod
    def _print_progress(record: RealTimePropagationStep, n_steps: int) -> None:
        print(
            f"TDSE {record.step:6d}/{n_steps}: "
            f"t={record.time:.8f}  "
            f"norm={record.norm:.12f}  "
            f"linear_iterations={record.linear_iterations:d}  "
            f"linear_residual={record.linear_residual_norm:.3e}"
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


def _hamiltonian_action(hamiltonian, state, time, shape) -> np.ndarray:
    action = np.asarray(hamiltonian(state, time))
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


def _sampler_functions(samplers) -> dict[str, Callable]:
    if samplers is None:
        return {}
    if not isinstance(samplers, Mapping):
        raise TypeError("samplers must be a mapping from names to callables")
    result = {}
    for name, sampler in samplers.items():
        if not isinstance(name, str) or not name:
            raise ValueError("sampler names must be nonempty strings")
        if not callable(sampler):
            raise TypeError(f"sampler {name!r} must be callable")
        result[name] = sampler
    return result


def _sample(state, time, samplers, storage) -> None:
    read_only_state = state.view()
    read_only_state.flags.writeable = False
    for name, sampler in samplers.items():
        value = np.asarray(sampler(read_only_state, time))
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"sampler {name!r} must return numeric values")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"sampler {name!r} returned non-finite values")
        if storage[name] and value.shape != storage[name][0].shape:
            raise ValueError(f"sampler {name!r} changed its output shape")
        storage[name].append(value.copy())


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(value, name: str) -> float:
    result = _nonnegative_real(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value, name: str) -> float:
    result = _finite_real(value, name)
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


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)
