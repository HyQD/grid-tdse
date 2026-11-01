"""Hermitian Krylov ground-state solver with optional thick restart."""

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Literal

import numpy as np

from .eigenproblem import (
    EigenTerminationReason,
    GroundStateEigenIteration,
    GroundStateEigenResult,
)
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


class LanczosGroundStateSolver:
    """Find the lowest eigenstate represented by an initial vector.

    Full physical reorthogonalization makes the solver robust against loss of
    Lanczos orthogonality, while Rayleigh--Ritz extraction targets the lowest
    represented eigenpair after every Hamiltonian action.

    The solver is independent of the state shape and spatial representation.
    ``hamiltonian(state)`` supplies the matrix-free action and
    ``inner_product(left, right)`` supplies the physical inner product.  A
    batched callback can combine all basis overlaps for one new direction.

    By default the solver is unrestarted and ``max_iterations`` is also the
    maximum subspace dimension.  Setting ``maximum_subspace_dimension``
    enables thick restart: the requested lowest Ritz vectors and their cached
    Hamiltonian images are retained, the target Ritz residual becomes the
    next direction, and expansion continues in a general Hermitian projected
    subspace.  No tridiagonal projected-matrix assumption is made after a
    restart.
    """

    def __init__(
        self,
        *,
        max_iterations: int = 30,
        maximum_subspace_dimension: int | None = None,
        retained_ritz_vectors: int = 4,
        residual_tolerance: float = 1.0e-8,
        breakdown_tolerance: float = 1.0e-14,
        reorthogonalization: Literal["if-needed", "always"] = "always",
        reorthogonalization_threshold: float = 0.5,
        hermiticity_tolerance: float = 1.0e-10,
        reality_tolerance: float = 1.0e-12,
        track_orthogonality: bool = True,
        progress_interval: int | None = 10,
    ) -> None:
        self.max_iterations = _positive_integer(
            max_iterations,
            "max_iterations",
        )
        self.maximum_subspace_dimension = _optional_positive_integer(
            maximum_subspace_dimension,
            "maximum_subspace_dimension",
        )
        if (
            self.maximum_subspace_dimension is not None
            and self.maximum_subspace_dimension < 2
        ):
            raise ValueError(
                "maximum_subspace_dimension must be at least two"
            )
        self.retained_ritz_vectors = _positive_integer(
            retained_ritz_vectors,
            "retained_ritz_vectors",
        )
        if (
            self.maximum_subspace_dimension is not None
            and self.retained_ritz_vectors
            >= self.maximum_subspace_dimension
        ):
            raise ValueError(
                "retained_ritz_vectors must be smaller than "
                "maximum_subspace_dimension"
            )
        self.residual_tolerance = _positive_real(
            residual_tolerance,
            "residual_tolerance",
        )
        self.breakdown_tolerance = _positive_real(
            breakdown_tolerance,
            "breakdown_tolerance",
        )
        if reorthogonalization not in {"if-needed", "always"}:
            raise ValueError(
                "reorthogonalization must be 'if-needed' or 'always'"
            )
        self.reorthogonalization = reorthogonalization
        self.reorthogonalization_threshold = _unit_interval_real(
            reorthogonalization_threshold,
            "reorthogonalization_threshold",
        )
        self.hermiticity_tolerance = _nonnegative_real(
            hermiticity_tolerance,
            "hermiticity_tolerance",
        )
        self.reality_tolerance = _nonnegative_real(
            reality_tolerance,
            "reality_tolerance",
        )
        if not isinstance(track_orthogonality, (bool, np.bool_)):
            raise TypeError("track_orthogonality must be boolean")
        self.track_orthogonality = bool(track_orthogonality)
        self.progress_interval = _optional_positive_integer(
            progress_interval,
            "progress_interval",
        )

    def solve(
        self,
        initial_state: np.ndarray,
        hamiltonian: Callable[[np.ndarray], np.ndarray],
        inner_product: InnerProduct,
        *,
        inner_products: BatchedInnerProducts | None = None,
        iteration_observer: (
            Callable[[np.ndarray, GroundStateEigenIteration], None] | None
        ) = None,
        observer_interval: int | None = None,
        print_progress: bool = True,
    ) -> GroundStateEigenResult:
        """Build a Hermitian Krylov basis and return its lowest Ritz pair."""

        if not callable(hamiltonian):
            raise TypeError("hamiltonian must be callable")
        if not callable(inner_product):
            raise TypeError("inner_product must be callable")
        if inner_products is not None and not callable(inner_products):
            raise TypeError("inner_products must be callable or None")
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
        first_basis_vector = normalize(
            state,
            inner_product,
            breakdown_tolerance=0.0,
            reality_tolerance=self.reality_tolerance,
        )
        basis = [first_basis_vector]
        hamiltonian_images: list[np.ndarray] = []
        projected = np.empty((0, 0), dtype=float)
        history: list[GroundStateEigenIteration] = []
        previous_state = None
        previous_eigenvalue = None
        hamiltonian_actions = 0
        restarts = 0

        final_state = first_basis_vector
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
                    restarted_subspace = self._restart_subspace(
                        basis,
                        hamiltonian_images,
                        projected,
                        inner_product,
                        inner_products,
                    )
                    if restarted_subspace is None:
                        expansion_failed = True
                        termination_reason = "invariant_subspace"
                    else:
                        basis, hamiltonian_images, projected = (
                            restarted_subspace
                        )
                        restarts += 1
                        restarted = True
                else:
                    expansion = orthonormalize(
                        image,
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
                    if not expansion.accepted:
                        expansion_failed = True
                        termination_reason = "invariant_subspace"
                    else:
                        basis.append(expansion.vector)

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
            if converged or final_iteration:
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
            preconditioner_applications=0,
            termination_reason=termination_reason,
            history=tuple(history),
            solver="lanczos",
        )

    __call__ = solve

    def _restart_required(self, subspace_dimension: int) -> bool:
        return (
            self.maximum_subspace_dimension is not None
            and subspace_dimension >= self.maximum_subspace_dimension
        )

    def _restart_subspace(
        self,
        basis: Sequence[np.ndarray],
        hamiltonian_images: Sequence[np.ndarray],
        projected: np.ndarray,
        inner_product: InnerProduct,
        inner_products: BatchedInnerProducts | None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray] | None:
        retained = rayleigh_ritz(
            basis,
            hamiltonian_images,
            inner_product,
            number_of_roots=self.retained_ritz_vectors,
            projected_matrix=projected,
            hermiticity_tolerance=self.hermiticity_tolerance,
            reality_tolerance=self.reality_tolerance,
        )
        retained_basis = [vector.copy() for vector in retained.vectors]
        retained_images = [
            image.copy() for image in retained.hamiltonian_images
        ]
        retained_projected = np.diag(retained.eigenvalues)
        expansion = orthonormalize(
            retained.residuals[0],
            retained_basis,
            inner_product,
            inner_products=inner_products,
            reorthogonalization=self.reorthogonalization,
            reorthogonalization_threshold=(
                self.reorthogonalization_threshold
            ),
            breakdown_tolerance=self.breakdown_tolerance,
            reality_tolerance=self.reality_tolerance,
        )
        if not expansion.accepted:
            return None
        retained_basis.append(expansion.vector)
        return retained_basis, retained_images, retained_projected

    @staticmethod
    def _print_progress(record: GroundStateEigenIteration) -> None:
        print(
            f"Lanczos {record.iteration:4d}: "
            f"E={record.eigenvalue:.12f}  "
            f"|dE|={record.eigenvalue_change:.3e}  "
            f"dPsi={record.state_change:.3e}  "
            f"residual={record.residual_norm:.3e}  "
            f"subspace={record.subspace_dimension}"
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


def _hamiltonian_action(
    hamiltonian: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    action = np.asarray(hamiltonian(state))
    if action.shape != shape:
        raise ValueError(
            f"hamiltonian returned shape {action.shape}; expected {shape}"
        )
    if not np.issubdtype(action.dtype, np.number):
        raise TypeError("hamiltonian must return numeric values")
    if not np.isfinite(action).all():
        raise ValueError("hamiltonian returned non-finite values")
    return action.copy()


def _notify_iteration_observer(
    observer: Callable[[np.ndarray, GroundStateEigenIteration], None],
    state: np.ndarray,
    record: GroundStateEigenIteration,
) -> None:
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


def _unit_interval_real(value, name: str) -> float:
    result = _nonnegative_real(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return result


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
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


__all__ = ["LanczosGroundStateSolver"]
