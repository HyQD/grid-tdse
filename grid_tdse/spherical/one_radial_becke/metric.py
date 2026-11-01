"""Overlap conditioning and canonical orthogonalization."""

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class CanonicalOrthogonalization:
    """Rank decision and transformation for a nonorthogonal basis."""

    overlap_eigenvalues: np.ndarray
    retained_indices: np.ndarray
    discarded_indices: np.ndarray
    transformation: np.ndarray
    relative_cutoff: float
    absolute_cutoff: float
    retained_rank: int
    discarded_rank: int
    condition_number: float
    retained_condition_number: float
    hermiticity_tolerance: float
    relative_antihermitian_norm: float
    orthonormality_error: float


def _relative_antihermitian_norm(matrix: np.ndarray) -> float:
    denominator = float(np.linalg.norm(matrix))
    numerator = float(np.linalg.norm(matrix - matrix.conj().T))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else np.inf
    return numerator / denominator


def _validated_square_matrix(matrix, name: str) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must contain numbers")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(matrix, dtype=np.result_type(matrix.dtype, complex))


def _validated_relative_cutoff(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("relative_cutoff must be a real number")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("relative_cutoff must be finite and satisfy 0 <= tau < 1")
    return value


def _validated_hermiticity_tolerance(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("hermiticity_tolerance must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("hermiticity_tolerance must be finite and positive")
    return value


def canonical_orthogonalization(
    overlap,
    *,
    relative_cutoff: float = 1.0e-10,
    hermiticity_tolerance: float = 1.0e-11,
) -> CanonicalOrthogonalization:
    r"""Construct ``X`` such that ``X^dagger S X = I`` after rank selection.

    An overlap eigenvector is retained precisely when
    ``s_i > relative_cutoff * s_max``.  A negative eigenvalue whose magnitude
    exceeds the Hermiticity tolerance relative to ``s_max`` is treated as an
    invalid, non-positive-semidefinite overlap rather than silently removed.
    """

    overlap = _validated_square_matrix(overlap, "overlap")
    relative_cutoff = _validated_relative_cutoff(relative_cutoff)
    hermiticity_tolerance = _validated_hermiticity_tolerance(
        hermiticity_tolerance
    )
    relative_antihermitian_norm = _relative_antihermitian_norm(overlap)
    if (
        not np.isfinite(relative_antihermitian_norm)
        or relative_antihermitian_norm > hermiticity_tolerance
    ):
        raise ValueError(
            "overlap exceeds the relative anti-Hermitian tolerance: "
            f"{relative_antihermitian_norm:.3e}"
        )

    hermitian_overlap = 0.5 * (overlap + overlap.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_overlap)
    maximum_eigenvalue = float(eigenvalues[-1])
    if maximum_eigenvalue <= 0.0:
        raise np.linalg.LinAlgError("overlap has no positive eigenvalues")
    if eigenvalues[0] < -hermiticity_tolerance * maximum_eigenvalue:
        raise np.linalg.LinAlgError(
            "overlap has a significant negative eigenvalue: "
            f"{eigenvalues[0]:.3e}"
        )

    absolute_cutoff = relative_cutoff * maximum_eigenvalue
    retained_indices = np.flatnonzero(eigenvalues > absolute_cutoff)
    discarded_indices = np.flatnonzero(eigenvalues <= absolute_cutoff)
    retained_eigenvalues = eigenvalues[retained_indices]
    transformation = eigenvectors[:, retained_indices] / np.sqrt(
        retained_eigenvalues
    )[np.newaxis, :]

    identity = np.eye(retained_indices.size)
    orthonormality_error = float(
        np.linalg.norm(
            transformation.conj().T
            @ hermitian_overlap
            @ transformation
            - identity
        )
    )
    condition_number = (
        maximum_eigenvalue / float(eigenvalues[0])
        if eigenvalues[0] > 0.0
        else np.inf
    )
    retained_condition_number = maximum_eigenvalue / float(
        retained_eigenvalues[0]
    )

    overlap_eigenvalues = np.array(eigenvalues, copy=True)
    retained_indices = np.array(retained_indices, copy=True)
    discarded_indices = np.array(discarded_indices, copy=True)
    transformation = np.array(transformation, copy=True)
    for array in (
        overlap_eigenvalues,
        retained_indices,
        discarded_indices,
        transformation,
    ):
        array.setflags(write=False)

    return CanonicalOrthogonalization(
        overlap_eigenvalues=overlap_eigenvalues,
        retained_indices=retained_indices,
        discarded_indices=discarded_indices,
        transformation=transformation,
        relative_cutoff=relative_cutoff,
        absolute_cutoff=absolute_cutoff,
        retained_rank=retained_indices.size,
        discarded_rank=discarded_indices.size,
        condition_number=float(condition_number),
        retained_condition_number=float(retained_condition_number),
        hermiticity_tolerance=hermiticity_tolerance,
        relative_antihermitian_norm=relative_antihermitian_norm,
        orthonormality_error=orthonormality_error,
    )
