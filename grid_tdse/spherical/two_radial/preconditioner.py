"""One-body preconditioning for two-radial Crank--Nicolson systems."""

import numpy as np

from .system import TwoRadialSystem


class TwoRadialOneBodyPreconditioner:
    r"""Approximate the imaginary-time CN inverse with the one-body terms.

    The preconditioner applies

    ``[I + dt/2 (h1 + h2)]**-1``

    independently to every angular-channel pair.  The coordinate Hamiltonians
    are diagonalized in their quadrature-weighted Hermitian representations.
    The two-coordinate Coulomb interaction is deliberately omitted.
    """

    def __init__(self, system: TwoRadialSystem, time_step: float) -> None:
        if not isinstance(system, TwoRadialSystem):
            raise TypeError("system must be a TwoRadialSystem")
        if isinstance(time_step, (bool, np.bool_)) or not np.isscalar(time_step):
            raise TypeError("time_step must be a positive real scalar")
        if np.iscomplexobj(time_step):
            raise TypeError("time_step must be a positive real scalar")
        time_step = float(time_step)
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be positive and finite")

        self.system = system
        self.time_step = time_step
        self.alpha = 0.5 * time_step
        self._sqrt_w1 = np.sqrt(system.grid1.weights)
        self._sqrt_w2 = np.sqrt(system.grid2.weights)
        self._eigenvalues1, self._eigenvectors1 = _weighted_eigensystems(
            system.h1, self._sqrt_w1
        )
        self._eigenvalues2, self._eigenvectors2 = _weighted_eigensystems(
            system.h2, self._sqrt_w2
        )

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state)
        if state.shape != self.system.psi_shape:
            raise ValueError(
                f"state has shape {state.shape}; expected {self.system.psi_shape}"
            )

        result = np.empty_like(state, dtype=np.result_type(state.dtype, float))
        for i, (eps1, vectors1) in enumerate(
            zip(self._eigenvalues1, self._eigenvectors1, strict=True)
        ):
            for j, (eps2, vectors2) in enumerate(
                zip(self._eigenvalues2, self._eigenvectors2, strict=True)
            ):
                weighted = self._sqrt_w1[:, None] * state[i, j] * self._sqrt_w2[None, :]
                transformed = vectors1.T.conj() @ weighted @ vectors2
                denominator = 1.0 + self.alpha * (eps1[:, None] + eps2[None, :])
                if np.any(np.isclose(denominator, 0.0)):
                    raise np.linalg.LinAlgError(
                        "one-body preconditioner is singular for this time step"
                    )
                transformed /= denominator
                weighted_result = vectors1 @ transformed @ vectors2.T.conj()
                result[i, j] = (
                    weighted_result / self._sqrt_w1[:, None] / self._sqrt_w2[None, :]
                )
        return result


def lowest_weighted_eigenstate(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the lowest eigenpair of a quadrature-weighted Hermitian matrix."""

    eigenvalues, eigenstates = weighted_eigenstates(matrix, weights)
    return float(eigenvalues[0]), eigenstates[:, 0]


def weighted_eigenstates(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a matrix in its quadrature-weighted representation."""

    matrix = np.asarray(matrix)
    weights = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if weights.shape != (matrix.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("weights must be positive and match the matrix dimension")

    sqrt_weights = np.sqrt(weights)
    transformed = sqrt_weights[:, None] * matrix / sqrt_weights[None, :]
    transformed = 0.5 * (transformed + transformed.T.conj())
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    states = eigenvectors / sqrt_weights[:, None]
    for column in range(states.shape[1]):
        pivot = int(np.argmax(np.abs(states[:, column])))
        phase = states[pivot, column]
        if abs(phase) != 0.0:
            states[:, column] *= phase.conjugate() / abs(phase)
    return eigenvalues, np.real_if_close(states)


def _weighted_eigensystems(blocks: np.ndarray, sqrt_weights: np.ndarray):
    eigenvalues = []
    eigenvectors = []
    for block in blocks:
        transformed = sqrt_weights[:, None] * block / sqrt_weights[None, :]
        transformed = 0.5 * (transformed + transformed.T.conj())
        values, vectors = np.linalg.eigh(transformed)
        eigenvalues.append(values)
        eigenvectors.append(vectors)
    return np.asarray(eigenvalues), np.asarray(eigenvectors)
