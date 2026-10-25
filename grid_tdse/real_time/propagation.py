r"""Representation-independent real-time Schrödinger propagation."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np

from integrator_lib import CrankNicolson
from scipy.sparse.linalg import LinearOperator

from .checkpoint import (
    RealTimeCheckpoint,
    RealTimeCheckpointSegment,
    load_real_time_checkpoint,
    save_real_time_checkpoint,
)


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

    States are never normalized during real-time propagation. An optional
    post-step transformation can apply a representation-specific operation,
    such as an absorbing mask, after each accepted integrator step.
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
        step_ratio = duration / self.requested_time_step
        nearest_step_count = round(step_ratio)
        if np.isclose(
            step_ratio,
            nearest_step_count,
            rtol=0.0,
            atol=16.0 * np.finfo(float).eps * max(1.0, abs(step_ratio)),
        ):
            self.n_steps = int(nearest_step_count)
        else:
            self.n_steps = int(np.ceil(step_ratio))
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
        post_step: Callable[[np.ndarray, float], np.ndarray] | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        compatibility_metadata: Mapping[str, Any] | None = None,
        dynamics_metadata: Mapping[str, Any] | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        checkpoint_overwrite: bool = False,
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
        post_step
            Optional ``post_step(state, time)`` transformation applied after
            each accepted step and before norms, samples, and checkpoints are
            recorded. It receives a read-only state view and must return a
            finite numeric array with the same shape. The initial state is not
            transformed.
        checkpoint_path
            Exact user path for an atomically replaced rolling checkpoint.
            When supplied, the final state is always checkpointed.
        checkpoint_interval
            Optional positive number of accepted steps between checkpoints.
            ``None`` saves only the final state.
        compatibility_metadata
            JSON-serializable physical-system and representation description
            used to validate resumed or continued calculations.
        dynamics_metadata
            JSON-serializable description of this propagation stage, such as
            its external field and gauge.
        checkpoint_metadata
            Additional JSON-serializable archive metadata.
        checkpoint_overwrite
            Permit replacing a checkpoint that exists before propagation.
        """

        return self._propagate(
            initial_state,
            hamiltonian,
            inner_product,
            preconditioner=preconditioner,
            samplers=samplers,
            post_step=post_step,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            compatibility_metadata=compatibility_metadata,
            dynamics_metadata=dynamics_metadata,
            checkpoint_metadata=checkpoint_metadata,
            checkpoint_overwrite=checkpoint_overwrite,
            previous_checkpoint=None,
            continuation_mode=None,
        )

    def resume(
        self,
        checkpoint: RealTimeCheckpoint | str | PathLike[str],
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
        post_step: Callable[[np.ndarray, float], np.ndarray] | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        compatibility_metadata: Mapping[str, Any] | None = None,
        dynamics_metadata: Mapping[str, Any] | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        checkpoint_overwrite: bool = False,
    ) -> RealTimePropagationResult:
        """Exactly resume an interrupted propagation segment.

        Exact recovery requires the original target time, time step,
        integrator settings, dynamics metadata, and sampler names. Use
        :meth:`continue_from` to start a new stage with a different field or
        propagation configuration.
        """

        stored = _checkpoint_value(checkpoint)
        if checkpoint_path is None:
            checkpoint_path = stored.source_path
        return self._propagate(
            stored.state,
            hamiltonian,
            inner_product,
            preconditioner=preconditioner,
            samplers=samplers,
            post_step=post_step,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            compatibility_metadata=compatibility_metadata,
            dynamics_metadata=dynamics_metadata,
            checkpoint_metadata=checkpoint_metadata,
            checkpoint_overwrite=checkpoint_overwrite,
            previous_checkpoint=stored,
            continuation_mode="exact",
        )

    def continue_from(
        self,
        checkpoint: RealTimeCheckpoint | str | PathLike[str],
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
        post_step: Callable[[np.ndarray, float], np.ndarray] | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        compatibility_metadata: Mapping[str, Any] | None = None,
        dynamics_metadata: Mapping[str, Any] | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        checkpoint_overwrite: bool = False,
    ) -> RealTimePropagationResult:
        """Start a new propagation segment from a checkpointed state.

        Only the compatibility metadata and start time must match. The new
        segment may use a different field, gauge, integrator configuration,
        final time, and set of samplers.
        """

        stored = _checkpoint_value(checkpoint)
        if checkpoint_path is None:
            checkpoint_path = stored.source_path
        return self._propagate(
            stored.state,
            hamiltonian,
            inner_product,
            preconditioner=preconditioner,
            samplers=samplers,
            post_step=post_step,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            compatibility_metadata=compatibility_metadata,
            dynamics_metadata=dynamics_metadata,
            checkpoint_metadata=checkpoint_metadata,
            checkpoint_overwrite=checkpoint_overwrite,
            previous_checkpoint=stored,
            continuation_mode="compatible",
        )

    def _propagate(
        self,
        initial_state,
        hamiltonian,
        inner_product,
        *,
        preconditioner,
        samplers,
        post_step,
        checkpoint_path,
        checkpoint_interval,
        compatibility_metadata,
        dynamics_metadata,
        checkpoint_metadata,
        checkpoint_overwrite,
        previous_checkpoint: RealTimeCheckpoint | None,
        continuation_mode: Literal["exact", "compatible"] | None,
    ) -> RealTimePropagationResult:
        """Run a new, resumed, or continued propagation segment."""

        if not callable(hamiltonian):
            raise TypeError("hamiltonian must be callable")
        if not callable(inner_product):
            raise TypeError("inner_product must be callable")
        if post_step is not None and not callable(post_step):
            raise TypeError("post_step must be callable or None")

        interval = _optional_positive_integer(
            checkpoint_interval,
            "checkpoint_interval",
        )
        if not isinstance(checkpoint_overwrite, (bool, np.bool_)):
            raise TypeError("checkpoint_overwrite must be boolean")
        compatibility = _mapping_or_empty(
            compatibility_metadata,
            "compatibility_metadata",
        )
        dynamics = _mapping_or_empty(dynamics_metadata, "dynamics_metadata")
        archive_metadata = _mapping_or_empty(
            checkpoint_metadata,
            "checkpoint_metadata",
        )
        sampler_functions = _sampler_functions(samplers)
        integrator_metadata = self._integrator_metadata()
        destination = _checkpoint_destination(
            checkpoint_path,
            interval,
            overwrite=bool(checkpoint_overwrite),
            previous_checkpoint=previous_checkpoint,
        )

        state = _state_array(initial_state)
        shape = state.shape
        if previous_checkpoint is None:
            initial_norm = self._norm(state, inner_product)
            if initial_norm == 0.0:
                raise ValueError("initial_state has zero norm")
            history = []
            sample_storage = {name: [] for name in sampler_functions}
            _sample(state, self.initial_time, sampler_functions, sample_storage)
            segment_initial_time = self.initial_time
            segment_initial_norm = initial_norm
            segment_prefix: tuple[RealTimeCheckpointSegment, ...] = ()
            previous_steps = 0
        else:
            self._validate_checkpoint_start(
                previous_checkpoint,
                compatibility,
            )
            if continuation_mode == "exact":
                (
                    history,
                    sample_storage,
                    segment_initial_time,
                    segment_initial_norm,
                    segment_prefix,
                    previous_steps,
                    archive_metadata,
                ) = self._resume_segment(
                    previous_checkpoint,
                    sampler_functions,
                    integrator_metadata,
                    dynamics,
                    archive_metadata,
                )
            else:
                initial_norm = self._norm(state, inner_product)
                if initial_norm == 0.0:
                    raise ValueError("checkpoint state has zero norm")
                history = []
                sample_storage = {name: [] for name in sampler_functions}
                _sample(
                    state,
                    self.initial_time,
                    sampler_functions,
                    sample_storage,
                )
                segment_initial_time = self.initial_time
                segment_initial_norm = initial_norm
                segment_prefix = previous_checkpoint.segments
                previous_steps = 0
                if checkpoint_metadata is None:
                    archive_metadata = previous_checkpoint.metadata
                else:
                    archive_metadata = {
                        **previous_checkpoint.metadata,
                        **archive_metadata,
                    }

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

        total_segment_steps = previous_steps + self.n_steps
        for local_step, time in enumerate(self.propagation_times[:-1], start=1):
            state = integrator(
                state,
                time,
                self.time_step,
                schrodinger_rhs,
            )
            accepted_time = float(self.propagation_times[local_step])
            if post_step is not None:
                state = _post_step_state(post_step, state, accepted_time, shape)
            step = previous_steps + local_step
            record = RealTimePropagationStep(
                step=step,
                time=accepted_time,
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
            if (
                destination is not None
                and interval is not None
                and step % interval == 0
            ):
                self._write_checkpoint(
                    destination,
                    state,
                    history,
                    sample_storage,
                    segment_initial_time,
                    segment_initial_norm,
                    segment_prefix,
                    compatibility,
                    dynamics,
                    integrator_metadata,
                    archive_metadata,
                    interval,
                )
            if self.progress_interval is not None and (
                step % self.progress_interval == 0 or step == total_segment_steps
            ):
                self._print_progress(record, total_segment_steps)

        result = RealTimePropagationResult(
            state=state.copy(),
            initial_time=segment_initial_time,
            final_time=self.final_time,
            requested_time_step=self.requested_time_step,
            time_step=self.time_step,
            initial_norm=segment_initial_norm,
            history=tuple(history),
            samples={
                name: np.asarray(values)
                for name, values in sample_storage.items()
            },
        )
        if destination is not None:
            self._write_checkpoint(
                destination,
                state,
                history,
                sample_storage,
                segment_initial_time,
                segment_initial_norm,
                segment_prefix,
                compatibility,
                dynamics,
                integrator_metadata,
                archive_metadata,
                interval,
            )
        return result

    __call__ = propagate

    def _integrator_metadata(self) -> dict[str, Any]:
        return {
            "method": "crank_nicolson",
            "linear_solver": self.linear_solver,
            "linear_rtol": self.linear_rtol,
            "linear_atol": self.linear_atol,
            "linear_max_iterations": self.linear_max_iterations,
            "gmres_restart": self.gmres_restart,
            "reality_tolerance": self.reality_tolerance,
        }

    def _validate_checkpoint_start(
        self,
        checkpoint: RealTimeCheckpoint,
        compatibility: Mapping[str, Any],
    ) -> None:
        if not np.isclose(
            self.initial_time,
            checkpoint.current_time,
            rtol=0.0,
            atol=1.0e-13,
        ):
            raise ValueError(
                "propagator initial_time must equal the checkpoint current time"
            )
        checkpoint.validate_compatibility(compatibility)

    def _resume_segment(
        self,
        checkpoint,
        sampler_functions,
        integrator_metadata,
        dynamics,
        archive_metadata,
    ):
        segment = checkpoint.latest_segment
        if segment.complete:
            raise ValueError(
                "the latest checkpoint segment is complete; use continue_from "
                "to start another propagation stage"
            )
        if not np.isclose(
            self.final_time,
            segment.target_final_time,
            rtol=0.0,
            atol=1.0e-13,
        ):
            raise ValueError("exact resume requires the original target final time")
        if not np.isclose(
            self.requested_time_step,
            segment.requested_time_step,
            rtol=0.0,
            atol=1.0e-15,
        ) or not np.isclose(
            self.time_step,
            segment.time_step,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("exact resume requires the original time step")
        if integrator_metadata != segment.integrator:
            raise ValueError("exact resume requires the original integrator settings")
        if dynamics != segment.dynamics:
            raise ValueError("exact resume requires matching dynamics metadata")
        if set(sampler_functions) != set(segment.samples):
            raise ValueError("exact resume requires the original sampler names")
        if archive_metadata and archive_metadata != checkpoint.metadata:
            raise ValueError("exact resume checkpoint metadata does not match")

        history = [
            RealTimePropagationStep(
                step=step,
                time=float(segment.times[step]),
                norm=float(segment.norms[step]),
                linear_iterations=int(segment.linear_iterations[step - 1]),
                linear_residual_norm=float(
                    segment.linear_residual_norms[step - 1]
                ),
            )
            for step in range(1, segment.times.size)
        ]
        sample_storage = {
            name: [value.copy() for value in values]
            for name, values in segment.samples.items()
        }
        return (
            history,
            sample_storage,
            segment.initial_time,
            segment.initial_norm,
            checkpoint.segments[:-1],
            segment.n_steps,
            checkpoint.metadata,
        )

    def _write_checkpoint(
        self,
        destination,
        state,
        history,
        sample_storage,
        segment_initial_time,
        segment_initial_norm,
        segment_prefix,
        compatibility,
        dynamics,
        integrator_metadata,
        archive_metadata,
        checkpoint_interval,
    ) -> None:
        times = np.asarray(
            [segment_initial_time, *(record.time for record in history)]
        )
        segment = RealTimeCheckpointSegment(
            initial_time=segment_initial_time,
            target_final_time=self.final_time,
            requested_time_step=self.requested_time_step,
            time_step=self.time_step,
            checkpoint_interval=checkpoint_interval,
            initial_norm=segment_initial_norm,
            times=times,
            norms=np.asarray(
                [segment_initial_norm, *(record.norm for record in history)]
            ),
            linear_iterations=np.asarray(
                [record.linear_iterations for record in history],
                dtype=int,
            ),
            linear_residual_norms=np.asarray(
                [record.linear_residual_norm for record in history]
            ),
            samples={
                name: np.asarray(values) for name, values in sample_storage.items()
            },
            integrator=integrator_metadata,
            dynamics=dynamics,
        )
        checkpoint = RealTimeCheckpoint(
            state=state,
            segments=(*segment_prefix, segment),
            compatibility=compatibility,
            metadata=archive_metadata,
            source_path=destination,
        )
        save_real_time_checkpoint(destination, checkpoint, overwrite=True)

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


def _post_step_state(post_step, state, time, shape) -> np.ndarray:
    read_only_state = state.view()
    read_only_state.flags.writeable = False
    values = np.asarray(post_step(read_only_state, time))
    if values.shape != shape:
        raise ValueError(
            f"post_step returned shape {values.shape}; expected {shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("post_step must return numeric values")
    if not np.isfinite(values).all():
        raise ValueError("post_step returned non-finite values")
    return values.copy()


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


def _mapping_or_empty(values, name: str) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        return json.loads(json.dumps(dict(values), default=_metadata_json_value))
    except TypeError as exc:
        raise TypeError(
            f"{name} must contain only JSON-serializable values"
        ) from exc


def _metadata_json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _checkpoint_value(
    checkpoint: RealTimeCheckpoint | str | PathLike[str],
) -> RealTimeCheckpoint:
    if isinstance(checkpoint, RealTimeCheckpoint):
        return checkpoint
    if isinstance(checkpoint, (str, PathLike)):
        return load_real_time_checkpoint(checkpoint)
    raise TypeError("checkpoint must be a RealTimeCheckpoint or path")


def _checkpoint_destination(
    path,
    interval,
    *,
    overwrite: bool,
    previous_checkpoint: RealTimeCheckpoint | None,
) -> Path | None:
    if path is None:
        if interval is not None:
            raise ValueError(
                "checkpoint_interval requires a checkpoint_path"
            )
        return None
    if not isinstance(path, (str, PathLike)):
        raise TypeError("checkpoint_path must be a path-like value")
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination.parent}"
        )
    same_as_source = (
        previous_checkpoint is not None
        and previous_checkpoint.source_path is not None
        and destination.absolute() == previous_checkpoint.source_path.absolute()
    )
    if destination.exists() and not overwrite and not same_as_source:
        raise FileExistsError(f"checkpoint already exists: {destination}")
    return destination
