"""Preconditioned Davidson ground-state eigensolver."""

from collections.abc import Callable, Sequence
from typing import Protocol
from time import perf_counter

import numpy as np

from .eigenproblem import (
    EigenTerminationReason,
    GroundStateEigenIteration,
    GroundStateEigenResult,
)
from .lanczos import LanczosGroundStateSolver
from .subspace import (
    BatchedInnerProducts,
    InnerProduct,
    basis_orthogonality_error,
    extend_projected_hermitian_matrix,
    normalize,
    orthonormalize,
    phase_aligned_distance,
    rayleigh_ritz,
)


class DavidsonPreconditioner(Protocol):
    """Protocol for an eigenvalue-dependent Davidson correction."""

    def __call__(
        self,
        residual: np.ndarray,
        ritz_value: float,
    ) -> np.ndarray: ...


class IdentityDavidsonPreconditioner:
    """Return the negative residual as the unpreconditioned correction."""

    name = "identity"

    def __call__(
        self,
        residual: np.ndarray,
        ritz_value: float,
    ) -> np.ndarray:
        del ritz_value
        return -np.asarray(residual).copy()


class DiagonalDavidsonPreconditioner:
    r"""Apply a safeguarded diagonal Davidson correction.

    For Hamiltonian diagonal ``D`` and Ritz value ``theta``, the correction is

    ``-residual / (D - theta)``.

    A sign-preserving level shift and denominator floor prevent singular or
    explosively large corrections near ``D == theta``.
    """

    name = "diagonal"

    def __init__(
        self,
        diagonal: np.ndarray,
        *,
        relative_floor: float = 1.0e-8,
        absolute_floor: float = 1.0e-12,
        level_shift: float = 0.0,
        reality_tolerance: float = 1.0e-12,
    ) -> None:
        values = _numeric_array(diagonal, "diagonal")
        self.reality_tolerance = _nonnegative_real(
            reality_tolerance,
            "reality_tolerance",
        )
        scale = np.maximum(1.0, np.abs(values.real))
        if np.any(np.abs(values.imag) > self.reality_tolerance * scale):
            raise ValueError("diagonal has a significant imaginary part")
        self.diagonal = np.asarray(values.real, dtype=float).copy()
        self.relative_floor = _nonnegative_real(
            relative_floor,
            "relative_floor",
        )
        self.absolute_floor = _positive_real(
            absolute_floor,
            "absolute_floor",
        )
        self.level_shift = _nonnegative_real(level_shift, "level_shift")
        self.last_regularized_entries = 0

    def __call__(
        self,
        residual: np.ndarray,
        ritz_value: float,
    ) -> np.ndarray:
        values = _numeric_array(residual, "residual")
        if values.shape != self.diagonal.shape:
            raise ValueError(
                f"residual has shape {values.shape}; expected "
                f"{self.diagonal.shape}"
            )
        theta = _finite_real(ritz_value, "ritz_value")
        denominator, regularized = safeguarded_davidson_denominator(
            self.diagonal - theta,
            spectral_scale=max(
                1.0,
                float(np.max(np.abs(self.diagonal))),
                abs(theta),
            ),
            relative_floor=self.relative_floor,
            absolute_floor=self.absolute_floor,
            level_shift=self.level_shift,
        )
        self.last_regularized_entries = regularized
        return -values / denominator


class DavidsonGroundStateSolver(LanczosGroundStateSolver):
    """Find the lowest eigenpair with preconditioned Davidson expansion.

    The solver shares the weighted Rayleigh--Ritz and thick-restart machinery
    of :class:`LanczosGroundStateSolver`.  Its expansion vector is the
    preconditioned lowest Ritz residual rather than the next Lanczos image.
    ``None`` selects the explicit identity preconditioner.
    """

    def solve(
        self,
        initial_state: np.ndarray,
        hamiltonian: Callable[[np.ndarray], np.ndarray],
        inner_product: InnerProduct,
        *,
        inner_products: BatchedInnerProducts | None = None,
        preconditioner: DavidsonPreconditioner | None = None,
        iteration_observer: (
            Callable[[np.ndarray, GroundStateEigenIteration], None] | None
        ) = None,
        observer_interval: int | None = None,
        print_progress: bool = True,
    ) -> GroundStateEigenResult:
        """Build a Davidson subspace and return its lowest Ritz pair."""

        if not callable(hamiltonian):
            raise TypeError("hamiltonian must be callable")
        if not callable(inner_product):
            raise TypeError("inner_product must be callable")
        if inner_products is not None and not callable(inner_products):
            raise TypeError("inner_products must be callable or None")
        if preconditioner is None:
            preconditioner = IdentityDavidsonPreconditioner()
        elif not callable(preconditioner):
            raise TypeError("preconditioner must be callable or None")
        if iteration_observer is not None and not callable(iteration_observer):
            raise TypeError("iteration_observer must be callable or None")
        observer_interval = _optional_positive_integer(
            observer_interval,
            "observer_interval",
        )
        if not isinstance(print_progress, (bool, np.bool_)):
            raise TypeError("print_progress must be boolean")

        state = _state_array(initial_state)
        shape = state.shape
        basis = [
            normalize(
                state,
                inner_product,
                breakdown_tolerance=0.0,
                reality_tolerance=self.reality_tolerance,
            )
        ]
        hamiltonian_images: list[np.ndarray] = []
        projected = np.empty((0, 0), dtype=float)
        history: list[GroundStateEigenIteration] = []
        previous_state = None
        previous_eigenvalue = None
        hamiltonian_actions = 0
        preconditioner_applications = 0
        preconditioner_fallbacks = 0
        restarts = 0

        final_state = basis[0]
        final_eigenvalue = np.nan
        final_residual_norm = np.inf
        termination_reason: EigenTerminationReason = "maximum_iterations"
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            iteration_started = perf_counter()
            image = _hamiltonian_action(hamiltonian, basis[-1], shape)
            hamiltonian_actions += 1
            hamiltonian_images.append(image)
            projected = extend_projected_hermitian_matrix(
                projected,
                basis,
                image,
                inner_product,
                inner_products=inner_products,
                hermiticity_tolerance=self.hermiticity_tolerance,
                reality_tolerance=self.reality_tolerance,
            )
            ritz = rayleigh_ritz(
                basis,
                hamiltonian_images,
                inner_product,
                number_of_roots=1,
                projected_matrix=projected,
                hermiticity_tolerance=self.hermiticity_tolerance,
                reality_tolerance=self.reality_tolerance,
            )
            final_state = ritz.vectors[0]
            final_eigenvalue = float(ritz.eigenvalues[0])
            final_residual_norm = float(ritz.residual_norms[0])
            ritz_subspace_dimension = len(basis)
            eigenvalue_change = (
                0.0
                if previous_eigenvalue is None
                else abs(final_eigenvalue - previous_eigenvalue)
            )
            state_change = (
                0.0
                if previous_state is None
                else phase_aligned_distance(
                    previous_state,
                    final_state,
                    inner_product,
                    reality_tolerance=self.reality_tolerance,
                )
            )
            orthogonality_error = (
                basis_orthogonality_error(
                    basis,
                    inner_product,
                    inner_products=inner_products,
                )
                if self.track_orthogonality
                else np.nan
            )
            converged = final_residual_norm <= self.residual_tolerance
            reached_iteration_limit = iteration == self.max_iterations
            restarted = False
            expansion_failed = False

            if not converged and not reached_iteration_limit:
                if self._restart_required(len(basis)):
                    retained = rayleigh_ritz(
                        basis,
                        hamiltonian_images,
                        inner_product,
                        number_of_roots=self.retained_ritz_vectors,
                        projected_matrix=projected,
                        hermiticity_tolerance=self.hermiticity_tolerance,
                        reality_tolerance=self.reality_tolerance,
                    )
                    retained_basis = [
                        vector.copy() for vector in retained.vectors
                    ]
                    retained_images = [
                        retained_image.copy()
                        for retained_image in retained.hamiltonian_images
                    ]
                    correction, used_fallback = self._correction(
                        retained.residuals[0],
                        float(retained.eigenvalues[0]),
                        retained_basis,
                        inner_product,
                        inner_products,
                        preconditioner,
                        shape,
                    )
                    preconditioner_applications += 1
                    preconditioner_fallbacks += int(used_fallback)
                    if correction is None:
                        expansion_failed = True
                        termination_reason = "invariant_subspace"
                    else:
                        basis = [*retained_basis, correction]
                        hamiltonian_images = retained_images
                        projected = np.diag(retained.eigenvalues)
                        restarts += 1
                        restarted = True
                else:
                    correction, used_fallback = self._correction(
                        ritz.residuals[0],
                        final_eigenvalue,
                        basis,
                        inner_product,
                        inner_products,
                        preconditioner,
                        shape,
                    )
                    preconditioner_applications += 1
                    preconditioner_fallbacks += int(used_fallback)
                    if correction is None:
                        expansion_failed = True
                        termination_reason = "invariant_subspace"
                    else:
                        basis.append(correction)

            record = GroundStateEigenIteration(
                iteration=iteration,
                eigenvalue=final_eigenvalue,
                eigenvalue_change=eigenvalue_change,
                state_change=state_change,
                residual_norm=final_residual_norm,
                subspace_dimension=ritz_subspace_dimension,
                orthogonality_error=orthogonality_error,
                restarted=restarted,
                hamiltonian_actions=hamiltonian_actions,
                preconditioner_applications=preconditioner_applications,
                preconditioner_fallbacks=preconditioner_fallbacks,
                wall_time=perf_counter() - iteration_started,
            )
            history.append(record)

            if converged:
                termination_reason = "converged"
            final_iteration = (
                converged or reached_iteration_limit or expansion_failed
            )
            if print_progress and self.progress_interval is not None and (
                iteration % self.progress_interval == 0 or final_iteration
            ):
                self._print_progress(record)
            if iteration_observer is not None and (
                iteration == 1
                or final_iteration
                or (
                    observer_interval is not None
                    and iteration % observer_interval == 0
                )
            ):
                _notify_iteration_observer(
                    iteration_observer,
                    final_state,
                    record,
                )
            if final_iteration:
                break
            previous_state = final_state
            previous_eigenvalue = final_eigenvalue

        return GroundStateEigenResult(
            state=final_state.copy(),
            eigenvalue=final_eigenvalue,
            residual_norm=final_residual_norm,
            converged=converged,
            iterations=len(history),
            restarts=restarts,
            hamiltonian_actions=hamiltonian_actions,
            preconditioner_applications=preconditioner_applications,
            termination_reason=termination_reason,
            history=tuple(history),
            solver="davidson",
        )

    __call__ = solve

    def _correction(
        self,
        residual: np.ndarray,
        ritz_value: float,
        basis: Sequence[np.ndarray],
        inner_product: InnerProduct,
        inner_products: BatchedInnerProducts | None,
        preconditioner: DavidsonPreconditioner,
        shape: tuple[int, ...],
    ) -> tuple[np.ndarray | None, bool]:
        candidate = _preconditioner_action(
            preconditioner,
            residual,
            ritz_value,
            shape,
        )
        correction = orthonormalize(
            candidate,
            basis,
            inner_product,
            inner_products=inner_products,
            reorthogonalization=self.reorthogonalization,
            reorthogonalization_threshold=(
                self.reorthogonalization_threshold
            ),
            breakdown_tolerance=self.breakdown_tolerance,
            reality_tolerance=self.reality_tolerance,
        )
        if correction.accepted:
            return correction.vector, False

        fallback = orthonormalize(
            residual,
            basis,
            inner_product,
            inner_products=inner_products,
            reorthogonalization=self.reorthogonalization,
            reorthogonalization_threshold=(
                self.reorthogonalization_threshold
            ),
            breakdown_tolerance=self.breakdown_tolerance,
            reality_tolerance=self.reality_tolerance,
        )
        if fallback.accepted:
            return fallback.vector, True
        return None, True

    @staticmethod
    def _print_progress(record: GroundStateEigenIteration) -> None:
        print(
            f"Davidson {record.iteration:4d}: "
            f"E={record.eigenvalue:.12f}  "
            f"|dE|={record.eigenvalue_change:.3e}  "
            f"dPsi={record.state_change:.3e}  "
            f"residual={record.residual_norm:.3e}  "
            f"subspace={record.subspace_dimension}"
        )


def _state_array(state) -> np.ndarray:
    values = _numeric_array(state, "initial_state")
    return values.copy()


def _hamiltonian_action(
    hamiltonian: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    action = _numeric_array(hamiltonian(state), "hamiltonian action")
    if action.shape != shape:
        raise ValueError(
            f"hamiltonian returned shape {action.shape}; expected {shape}"
        )
    return action.copy()


def _preconditioner_action(
    preconditioner: DavidsonPreconditioner,
    residual: np.ndarray,
    ritz_value: float,
    shape: tuple[int, ...],
) -> np.ndarray:
    action = _numeric_array(
        preconditioner(residual, ritz_value),
        "preconditioner action",
    )
    if action.shape != shape:
        raise ValueError(
            f"preconditioner returned shape {action.shape}; expected {shape}"
        )
    return action.copy()


def _notify_iteration_observer(
    observer: Callable[[np.ndarray, GroundStateEigenIteration], None],
    state: np.ndarray,
    record: GroundStateEigenIteration,
) -> None:
    view = state.view()
    view.flags.writeable = False
    observer(view, record)


def safeguarded_davidson_denominator(
    denominator: np.ndarray,
    *,
    spectral_scale: float,
    relative_floor: float,
    absolute_floor: float,
    level_shift: float,
) -> tuple[np.ndarray, int]:
    values = np.asarray(denominator, dtype=float).copy()
    if level_shift:
        values += np.where(values < 0.0, -level_shift, level_shift)
    floor = max(absolute_floor, relative_floor * spectral_scale)
    small = np.abs(values) < floor
    signs = np.where(values < 0.0, -1.0, 1.0)
    values[small] = signs[small] * floor
    return values, int(np.count_nonzero(small))


def _numeric_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_real(value, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_real(value, name: str) -> float:
    result = _nonnegative_real(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = [
    "DavidsonGroundStateSolver",
    "DavidsonPreconditioner",
    "DiagonalDavidsonPreconditioner",
    "IdentityDavidsonPreconditioner",
    "safeguarded_davidson_denominator",
]
