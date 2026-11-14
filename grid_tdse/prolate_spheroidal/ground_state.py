"""Davidson stationary states for prolate one-electron systems."""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np
from scipy.linalg import eigh

from grid_tdse.ground_state import (
    BlockDavidsonSolver,
    DavidsonGroundStateSolver,
    DiagonalDavidsonPreconditioner,
)

from .hamiltonian import ProlateSpheroidalOneElectronHamiltonian


@dataclass(frozen=True)
class ProlateSpheroidalOneElectronEigenstates:
    """Lowest stationary states from one signed-``m`` sector."""

    hamiltonian: ProlateSpheroidalOneElectronHamiltonian
    m: int
    electronic_energies: np.ndarray
    orbitals: np.ndarray
    residual_norms: np.ndarray
    root_converged: np.ndarray
    termination_reasons: tuple[str, ...]
    iterations: int
    restarts: int
    hamiltonian_actions: int
    preconditioner_applications: int
    preconditioner_fallbacks: int
    regularized_denominators: int
    solver_result: Any
    solver: str

    @property
    def total_energies(self) -> np.ndarray:
        """Return electronic energies plus clamped nuclear repulsion."""

        return self.electronic_energies + (
            self.hamiltonian.system.nuclear_repulsion_energy
        )

    @property
    def electronic_energy(self) -> float:
        """Return the lowest requested electronic energy."""

        return float(self.electronic_energies[0])

    @property
    def total_energy(self) -> float:
        """Return the lowest requested clamped-nuclei total energy."""

        return float(self.total_energies[0])

    @property
    def orbital(self) -> np.ndarray:
        """Return the lowest requested nodal orbital."""

        return self.orbitals[0]

    @property
    def residual_norm(self) -> float:
        """Return the residual norm of the lowest requested root."""

        return float(self.residual_norms[0])

    @property
    def converged(self) -> bool:
        """Return whether every requested root converged."""

        return bool(np.all(self.root_converged))

    @property
    def termination_reason(self) -> str:
        """Return the termination status of the lowest requested root."""

        return self.termination_reasons[0]


def diagonalize_fixed_m(
    hamiltonian,
    *,
    m=0,
    n_states=1,
    solver: Literal["davidson", "direct"] = "davidson",
    residual_tolerance=1.0e-10,
    max_iterations=200,
    maximum_subspace_dimension=None,
    retained_ritz_vectors=None,
    initial_orbitals=None,
    print_progress=False,
):
    """Return the lowest field-free states in one fixed-``m`` sector.

    Davidson is the production default. ``solver="direct"`` constructs a
    dense Euclidean-Hermitian matrix and is intended only as a small-grid
    validation oracle.
    """

    if not isinstance(
        hamiltonian, ProlateSpheroidalOneElectronHamiltonian
    ):
        raise TypeError(
            "hamiltonian must be a "
            "ProlateSpheroidalOneElectronHamiltonian"
        )
    m = hamiltonian.representation.validate_m(m)
    shape = hamiltonian.fixed_m_shape(m)
    dimension = int(np.prod(shape))
    n_states = _root_count(n_states, dimension)
    if solver not in {"davidson", "direct"}:
        raise ValueError("solver must be 'davidson' or 'direct'")
    if not isinstance(print_progress, (bool, np.bool_)):
        raise TypeError("print_progress must be boolean")

    if solver == "direct":
        return _direct_states(hamiltonian, m, n_states)

    initial = _initial_orbitals(
        hamiltonian,
        m,
        n_states,
        initial_orbitals,
    )
    if maximum_subspace_dimension is None:
        maximum_subspace_dimension = min(
            dimension,
            max(24, 4 * n_states + 8),
        )
    if retained_ritz_vectors is None:
        retained_ritz_vectors = min(
            maximum_subspace_dimension - 1,
            max(n_states, min(8, 2 * n_states)),
        )
    if retained_ritz_vectors < n_states:
        raise ValueError("retained_ritz_vectors must be at least n_states")

    options = dict(
        max_iterations=max_iterations,
        maximum_subspace_dimension=maximum_subspace_dimension,
        retained_ritz_vectors=retained_ritz_vectors,
        residual_tolerance=residual_tolerance,
    )
    inner_product = lambda left, right: hamiltonian.inner_product(
        left, right, m
    )
    preconditioner = DiagonalDavidsonPreconditioner(
        hamiltonian.fixed_m_diagonal(m, representation="strong")
    )

    if n_states == 1:
        eigensolver = DavidsonGroundStateSolver(**options)
        result = eigensolver.solve(
            initial[0],
            lambda state: hamiltonian.apply_fixed_m(state, m),
            inner_product,
            preconditioner=preconditioner,
            print_progress=print_progress,
        )
        energies = _readonly([result.eigenvalue])
        orbitals = _readonly(result.state[None, ...])
        residual_norms = _readonly([result.residual_norm])
        root_converged = _readonly([result.converged], dtype=bool)
        termination_reasons = (result.termination_reason,)
        preconditioner_fallbacks = result.preconditioner_fallbacks
        regularized_denominators = result.regularized_denominators
    else:
        eigensolver = BlockDavidsonSolver(**options)
        result = eigensolver.solve(
            initial,
            lambda state: hamiltonian.apply_fixed_m(state, m),
            inner_product,
            number_of_roots=n_states,
            preconditioner=preconditioner,
            print_progress=print_progress,
        )
        energies = _readonly(result.eigenvalues)
        orbitals = _readonly(result.states)
        residual_norms = _readonly(result.residual_norms)
        root_converged = _readonly(result.root_converged, dtype=bool)
        termination_reasons = tuple(
            "converged" if value else result.termination_reason
            for value in result.root_converged
        )
        preconditioner_fallbacks = result.preconditioner_fallbacks
        regularized_denominators = result.regularized_denominators

    return ProlateSpheroidalOneElectronEigenstates(
        hamiltonian=hamiltonian,
        m=m,
        electronic_energies=energies,
        orbitals=orbitals,
        residual_norms=residual_norms,
        root_converged=root_converged,
        termination_reasons=termination_reasons,
        iterations=result.iterations,
        restarts=result.restarts,
        hamiltonian_actions=result.hamiltonian_actions,
        preconditioner_applications=result.preconditioner_applications,
        preconditioner_fallbacks=preconditioner_fallbacks,
        regularized_denominators=regularized_denominators,
        solver_result=result,
        solver="davidson" if n_states == 1 else "block_davidson",
    )


def _direct_states(hamiltonian, m, n_states):
    shape = hamiltonian.fixed_m_shape(m)
    matrix = hamiltonian.fixed_m_matrix(
        m, representation="symmetric"
    ).toarray()
    energies, vectors = eigh(
        matrix,
        subset_by_index=(0, n_states - 1),
        check_finite=False,
        driver="evr",
    )
    symmetric = vectors.T.reshape((n_states, *shape))
    orbitals = hamiltonian.from_symmetric_fixed_m(symmetric, m)
    residual_norms = []
    for index in range(n_states):
        norm = np.sqrt(
            hamiltonian.inner_product(
                orbitals[index], orbitals[index], m
            ).real
        )
        orbitals[index] /= norm
        residual = (
            hamiltonian.apply_fixed_m(orbitals[index], m)
            - energies[index] * orbitals[index]
        )
        residual_norms.append(
            np.sqrt(hamiltonian.inner_product(residual, residual, m).real)
        )
    return ProlateSpheroidalOneElectronEigenstates(
        hamiltonian=hamiltonian,
        m=m,
        electronic_energies=_readonly(energies),
        orbitals=_readonly(orbitals),
        residual_norms=_readonly(residual_norms),
        root_converged=_readonly([True] * n_states, dtype=bool),
        termination_reasons=("direct",) * n_states,
        iterations=0,
        restarts=0,
        hamiltonian_actions=0,
        preconditioner_applications=0,
        preconditioner_fallbacks=0,
        regularized_denominators=0,
        solver_result=None,
        solver="direct",
    )


def _initial_orbitals(hamiltonian, m, n_states, supplied):
    shape = hamiltonian.fixed_m_shape(m)
    if supplied is not None:
        values = np.asarray(supplied)
        expected = (n_states, *shape)
        if values.shape != expected:
            raise ValueError(
                f"initial_orbitals must have shape {expected}"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("initial_orbitals must contain numeric values")
        if not np.isfinite(values).all():
            raise ValueError("initial_orbitals must be finite")
        return values.copy()

    diagonal = hamiltonian.fixed_m_diagonal(
        m, representation="strong"
    ).reshape(-1)
    indices = np.argsort(diagonal)[:n_states]
    guesses = np.zeros((n_states, diagonal.size), dtype=float)
    guesses[np.arange(n_states), indices] = 1.0
    return guesses.reshape((n_states, *shape))


def _root_count(value, dimension):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("n_states must be an integer")
    result = int(value)
    if not 1 <= result <= dimension:
        raise ValueError(f"n_states must satisfy 1 <= n_states <= {dimension}")
    return result


def _readonly(values, dtype=None):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


__all__ = [
    "ProlateSpheroidalOneElectronEigenstates",
    "diagonalize_fixed_m",
]
