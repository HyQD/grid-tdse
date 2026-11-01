"""Canonical solution of dense generalized Hermitian eigenproblems."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import eigh

from .assembly import BeckeDenseMatrices
from .metric import (
    CanonicalOrthogonalization,
    _relative_antihermitian_norm,
    _validated_hermiticity_tolerance,
    _validated_square_matrix,
    canonical_orthogonalization,
)


@dataclass(frozen=True)
class GeneralizedEigenDiagnostics:
    """Normalization, Hermiticity, and residual diagnostics."""

    hermiticity_tolerance: float
    hamiltonian_relative_antihermitian_norm: float
    orthogonal_hamiltonian_relative_antihermitian_norm: float
    orthonormality_error: float
    residual_norms: np.ndarray
    relative_residual_norms: np.ndarray
    projected_residual_norms: np.ndarray


@dataclass(frozen=True)
class GeneralizedEigenResult:
    """Eigenpairs in the original nonorthogonal basis."""

    energies: np.ndarray
    coefficients: np.ndarray
    orthogonal_coefficients: np.ndarray
    metric: CanonicalOrthogonalization
    diagnostics: GeneralizedEigenDiagnostics


def _validated_number_of_states(number_of_states, retained_rank: int) -> int:
    if number_of_states is None:
        return retained_rank
    if isinstance(number_of_states, (bool, np.bool_)) or not isinstance(
        number_of_states, Integral
    ):
        raise TypeError("number_of_states must be an integer or None")
    number_of_states = int(number_of_states)
    if not 1 <= number_of_states <= retained_rank:
        raise ValueError(
            "number_of_states must be between one and the retained overlap rank"
        )
    return number_of_states


def solve_generalized_hermitian(
    hamiltonian,
    overlap,
    *,
    relative_cutoff: float = 1.0e-10,
    number_of_states: int | None = None,
    hermiticity_tolerance: float = 1.0e-11,
) -> GeneralizedEigenResult:
    r"""Solve ``H B = S B E`` after canonical overlap orthogonalization.

    If overlap directions are discarded, the projected residual is the
    variational residual that must vanish.  The full residual is also reported
    because it can remain nonzero outside the retained metric subspace.
    """

    hamiltonian = _validated_square_matrix(hamiltonian, "hamiltonian")
    overlap = _validated_square_matrix(overlap, "overlap")
    if hamiltonian.shape != overlap.shape:
        raise ValueError("hamiltonian and overlap must have the same shape")
    hermiticity_tolerance = _validated_hermiticity_tolerance(
        hermiticity_tolerance
    )

    hamiltonian_relative_norm = _relative_antihermitian_norm(hamiltonian)
    if (
        not np.isfinite(hamiltonian_relative_norm)
        or hamiltonian_relative_norm > hermiticity_tolerance
    ):
        raise ValueError(
            "hamiltonian exceeds the relative anti-Hermitian tolerance: "
            f"{hamiltonian_relative_norm:.3e}"
        )

    metric = canonical_orthogonalization(
        overlap,
        relative_cutoff=relative_cutoff,
        hermiticity_tolerance=hermiticity_tolerance,
    )
    number_of_states = _validated_number_of_states(
        number_of_states, metric.retained_rank
    )
    hermitian_hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    hermitian_overlap = 0.5 * (overlap + overlap.conj().T)
    transformation = metric.transformation
    orthogonal_hamiltonian = (
        transformation.conj().T
        @ hermitian_hamiltonian
        @ transformation
    )
    orthogonal_relative_norm = _relative_antihermitian_norm(
        orthogonal_hamiltonian
    )
    if (
        not np.isfinite(orthogonal_relative_norm)
        or orthogonal_relative_norm > hermiticity_tolerance
    ):
        raise FloatingPointError(
            "orthogonal Hamiltonian exceeds the relative anti-Hermitian "
            f"tolerance: {orthogonal_relative_norm:.3e}"
        )
    orthogonal_hamiltonian = 0.5 * (
        orthogonal_hamiltonian + orthogonal_hamiltonian.conj().T
    )

    energies, orthogonal_coefficients = eigh(
        orthogonal_hamiltonian,
        subset_by_index=(0, number_of_states - 1),
        check_finite=False,
    )
    coefficients = transformation @ orthogonal_coefficients
    overlap_gram = coefficients.conj().T @ hermitian_overlap @ coefficients
    orthonormality_error = float(
        np.linalg.norm(overlap_gram - np.eye(number_of_states))
    )

    hamiltonian_action = hermitian_hamiltonian @ coefficients
    overlap_action = hermitian_overlap @ coefficients
    residuals = hamiltonian_action - overlap_action * energies[np.newaxis, :]
    residual_norms = np.linalg.norm(residuals, axis=0)
    denominators = np.linalg.norm(hamiltonian_action, axis=0) + (
        np.abs(energies) * np.linalg.norm(overlap_action, axis=0)
    )
    relative_residual_norms = np.divide(
        residual_norms,
        denominators,
        out=np.zeros_like(residual_norms),
        where=denominators > 0.0,
    )
    projected_residual_norms = np.linalg.norm(
        transformation.conj().T @ residuals, axis=0
    )

    energies = np.array(energies, copy=True)
    coefficients = np.array(coefficients, copy=True)
    orthogonal_coefficients = np.array(orthogonal_coefficients, copy=True)
    residual_norms = np.array(residual_norms, copy=True)
    relative_residual_norms = np.array(relative_residual_norms, copy=True)
    projected_residual_norms = np.array(projected_residual_norms, copy=True)
    for array in (
        energies,
        coefficients,
        orthogonal_coefficients,
        residual_norms,
        relative_residual_norms,
        projected_residual_norms,
    ):
        array.setflags(write=False)

    diagnostics = GeneralizedEigenDiagnostics(
        hermiticity_tolerance=hermiticity_tolerance,
        hamiltonian_relative_antihermitian_norm=hamiltonian_relative_norm,
        orthogonal_hamiltonian_relative_antihermitian_norm=(
            orthogonal_relative_norm
        ),
        orthonormality_error=orthonormality_error,
        residual_norms=residual_norms,
        relative_residual_norms=relative_residual_norms,
        projected_residual_norms=projected_residual_norms,
    )
    return GeneralizedEigenResult(
        energies=energies,
        coefficients=coefficients,
        orthogonal_coefficients=orthogonal_coefficients,
        metric=metric,
        diagnostics=diagnostics,
    )


def solve_becke_eigenproblem(
    matrices: BeckeDenseMatrices,
    *,
    relative_cutoff: float = 1.0e-10,
    number_of_states: int | None = None,
    hermiticity_tolerance: float = 1.0e-11,
) -> GeneralizedEigenResult:
    """Solve an assembled Becke-basis generalized eigenproblem."""

    if not isinstance(matrices, BeckeDenseMatrices):
        raise TypeError("matrices must be a BeckeDenseMatrices result")
    return solve_generalized_hermitian(
        matrices.hamiltonian,
        matrices.overlap,
        relative_cutoff=relative_cutoff,
        number_of_states=number_of_states,
        hermiticity_tolerance=hermiticity_tolerance,
    )
