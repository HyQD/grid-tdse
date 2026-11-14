"""Representation-independent block-Davidson eigensolver."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .davidson import (
    DavidsonPreconditioner,
    IdentityDavidsonPreconditioner,
)
from .lanczos import LanczosGroundStateSolver
from .subspace import (
    BatchedInnerProducts,
    InnerProduct,
    basis_orthogonality_error,
    orthonormalize,
    projected_hermitian_matrix,
    rayleigh_ritz,
)


@dataclass(frozen=True)
class BlockDavidsonIteration:
    """Diagnostics from one block-Davidson Ritz extraction."""

    iteration: int
    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    converged: np.ndarray
    subspace_dimension: int
    orthogonality_error: float
    restarted: bool
    hamiltonian_actions: int
    preconditioner_applications: int
    preconditioner_fallbacks: int
    regularized_denominators: int


@dataclass(frozen=True)
class BlockDavidsonResult:
    """Lowest roots and diagnostics from a block-Davidson solve."""

    states: np.ndarray
    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    root_converged: np.ndarray
    iterations: int
    restarts: int
    hamiltonian_actions: int
    preconditioner_applications: int
    preconditioner_fallbacks: int
    regularized_denominators: int
    termination_reason: Literal[
        "converged", "maximum_iterations", "invariant_subspace"
    ]
    history: tuple[BlockDavidsonIteration, ...]
    solver: str = "block_davidson"

    @property
    def converged(self) -> bool:
        """Return whether every requested root converged."""

        return bool(np.all(self.root_converged))


class BlockDavidsonSolver(LanczosGroundStateSolver):
    """Compute several lowest roots with metric block Davidson.

    Converged Ritz pairs are retained during expansion and thick restart;
    corrections are generated only for unconverged roots. This supplies
    root-wise residual tests and deflation without spatial assumptions.
    """

    def solve(
        self,
        initial_states: np.ndarray,
        hamiltonian: Callable[[np.ndarray], np.ndarray],
        inner_product: InnerProduct,
        *,
        number_of_roots: int,
        inner_products: BatchedInnerProducts | None = None,
        preconditioner: DavidsonPreconditioner | None = None,
        print_progress: bool = True,
    ) -> BlockDavidsonResult:
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
        if not isinstance(number_of_roots, (int, np.integer)) or isinstance(
            number_of_roots, (bool, np.bool_)
        ):
            raise TypeError("number_of_roots must be an integer")
        number_of_roots = int(number_of_roots)
        if number_of_roots < 1:
            raise ValueError("number_of_roots must be positive")
        if not isinstance(print_progress, (bool, np.bool_)):
            raise TypeError("print_progress must be boolean")

        guesses = _initial_block(initial_states)
        if guesses.shape[0] < number_of_roots:
            raise ValueError(
                "initial_states must contain at least number_of_roots states"
            )
        state_shape = guesses.shape[1:]
        if (
            self.maximum_subspace_dimension is not None
            and self.maximum_subspace_dimension <= number_of_roots
        ):
            raise ValueError(
                "maximum_subspace_dimension must exceed number_of_roots"
            )

        basis: list[np.ndarray] = []
        for guess in guesses:
            candidate = orthonormalize(
                guess,
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
            if candidate.accepted:
                basis.append(candidate.vector)
        if len(basis) < number_of_roots:
            raise ValueError("initial_states do not span the requested roots")

        images = [_action(hamiltonian, vector, state_shape) for vector in basis]
        hamiltonian_actions = len(images)
        preconditioner_applications = 0
        preconditioner_fallbacks = 0
        regularized_denominators = 0
        restarts = 0
        history: list[BlockDavidsonIteration] = []
        termination_reason = "maximum_iterations"

        for iteration in range(1, self.max_iterations + 1):
            projected = projected_hermitian_matrix(
                basis,
                images,
                inner_product,
                inner_products=inner_products,
                hermiticity_tolerance=self.hermiticity_tolerance,
            )
            ritz = rayleigh_ritz(
                basis,
                images,
                inner_product,
                number_of_roots=number_of_roots,
                projected_matrix=projected,
                hermiticity_tolerance=self.hermiticity_tolerance,
                reality_tolerance=self.reality_tolerance,
            )
            converged = ritz.residual_norms <= self.residual_tolerance
            restarted = False
            expansion_failed = False

            if not np.all(converged) and iteration < self.max_iterations:
                unconverged_count = int(np.count_nonzero(~converged))
                maximum = self.maximum_subspace_dimension
                if (
                    maximum is not None
                    and len(basis) + unconverged_count > maximum
                ):
                    retained_count = min(
                        len(basis),
                        max(number_of_roots, self.retained_ritz_vectors),
                    )
                    retained = rayleigh_ritz(
                        basis,
                        images,
                        inner_product,
                        number_of_roots=retained_count,
                        projected_matrix=projected,
                        hermiticity_tolerance=self.hermiticity_tolerance,
                        reality_tolerance=self.reality_tolerance,
                    )
                    basis = [vector.copy() for vector in retained.vectors]
                    images = [
                        image.copy() for image in retained.hamiltonian_images
                    ]
                    ritz = retained
                    converged = (
                        ritz.residual_norms[:number_of_roots]
                        <= self.residual_tolerance
                    )
                    restarts += 1
                    restarted = True

                accepted: list[np.ndarray] = []
                for root in range(number_of_roots):
                    if converged[root]:
                        continue
                    residual = ritz.residuals[root]
                    eigenvalue = float(ritz.eigenvalues[root])
                    correction = np.asarray(
                        preconditioner(residual, eigenvalue)
                    )
                    if correction.shape != state_shape:
                        raise ValueError(
                            "preconditioner returned an incompatible shape"
                        )
                    preconditioner_applications += 1
                    regularized_denominators += int(
                        getattr(preconditioner, "last_regularized_entries", 0)
                    )
                    orthogonal = orthonormalize(
                        correction,
                        [*basis, *accepted],
                        inner_product,
                        inner_products=inner_products,
                        reorthogonalization=self.reorthogonalization,
                        reorthogonalization_threshold=(
                            self.reorthogonalization_threshold
                        ),
                        breakdown_tolerance=self.breakdown_tolerance,
                        reality_tolerance=self.reality_tolerance,
                    )
                    if not orthogonal.accepted:
                        orthogonal = orthonormalize(
                            residual,
                            [*basis, *accepted],
                            inner_product,
                            inner_products=inner_products,
                            reorthogonalization=self.reorthogonalization,
                            reorthogonalization_threshold=(
                                self.reorthogonalization_threshold
                            ),
                            breakdown_tolerance=self.breakdown_tolerance,
                            reality_tolerance=self.reality_tolerance,
                        )
                        preconditioner_fallbacks += 1
                    if orthogonal.accepted:
                        accepted.append(orthogonal.vector)

                if not accepted:
                    expansion_failed = True
                    termination_reason = "invariant_subspace"
                else:
                    basis.extend(accepted)
                    new_images = [
                        _action(hamiltonian, vector, state_shape)
                        for vector in accepted
                    ]
                    images.extend(new_images)
                    hamiltonian_actions += len(new_images)

            orthogonality_error = (
                basis_orthogonality_error(
                    basis,
                    inner_product,
                    inner_products=inner_products,
                )
                if self.track_orthogonality
                else np.nan
            )
            record = BlockDavidsonIteration(
                iteration=iteration,
                eigenvalues=_readonly(ritz.eigenvalues[:number_of_roots]),
                residual_norms=_readonly(
                    ritz.residual_norms[:number_of_roots]
                ),
                converged=_readonly(converged, dtype=bool),
                subspace_dimension=len(ritz.projected_matrix),
                orthogonality_error=orthogonality_error,
                restarted=restarted,
                hamiltonian_actions=hamiltonian_actions,
                preconditioner_applications=preconditioner_applications,
                preconditioner_fallbacks=preconditioner_fallbacks,
                regularized_denominators=regularized_denominators,
            )
            history.append(record)

            final_iteration = (
                np.all(converged)
                or expansion_failed
                or iteration == self.max_iterations
            )
            if print_progress and self.progress_interval is not None and (
                iteration % self.progress_interval == 0 or final_iteration
            ):
                values = ", ".join(
                    f"{value:.12f}" for value in record.eigenvalues
                )
                residuals = ", ".join(
                    f"{value:.3e}" for value in record.residual_norms
                )
                print(
                    f"Block Davidson {iteration:4d}: E=[{values}]  "
                    f"residual=[{residuals}]  "
                    f"subspace={record.subspace_dimension}"
                )
            if final_iteration:
                if np.all(converged):
                    termination_reason = "converged"
                break

        states = np.stack(ritz.vectors[:number_of_roots])
        return BlockDavidsonResult(
            states=_readonly(states),
            eigenvalues=_readonly(ritz.eigenvalues[:number_of_roots]),
            residual_norms=_readonly(ritz.residual_norms[:number_of_roots]),
            root_converged=_readonly(converged, dtype=bool),
            iterations=len(history),
            restarts=restarts,
            hamiltonian_actions=hamiltonian_actions,
            preconditioner_applications=preconditioner_applications,
            preconditioner_fallbacks=preconditioner_fallbacks,
            regularized_denominators=regularized_denominators,
            termination_reason=termination_reason,
            history=tuple(history),
        )

    __call__ = solve


def _initial_block(values) -> np.ndarray:
    block = np.asarray(values)
    if block.ndim < 2 or block.shape[0] == 0:
        raise ValueError("initial_states must be a nonempty state block")
    if not np.issubdtype(block.dtype, np.number):
        raise TypeError("initial_states must contain numeric values")
    if not np.isfinite(block).all():
        raise ValueError("initial_states must contain only finite values")
    return block.copy()


def _action(hamiltonian, state, shape) -> np.ndarray:
    result = np.asarray(hamiltonian(state))
    if result.shape != shape:
        raise ValueError(
            f"hamiltonian returned shape {result.shape}; expected {shape}"
        )
    if (
        not np.issubdtype(result.dtype, np.number)
        or not np.isfinite(result).all()
    ):
        raise ValueError("hamiltonian returned an invalid state")
    return result.copy()


def _readonly(values, dtype=None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


__all__ = [
    "BlockDavidsonIteration",
    "BlockDavidsonResult",
    "BlockDavidsonSolver",
]
