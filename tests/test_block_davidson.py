import numpy as np

from grid_tdse.ground_state import (
    BlockDavidsonSolver,
    DiagonalDavidsonPreconditioner,
)


def test_block_davidson_matches_dense_lowest_roots():
    generator = np.random.default_rng(829)
    matrix = generator.normal(size=(12, 12))
    matrix = matrix + matrix.T + np.diag(np.linspace(-15.0, 15.0, 12))
    reference, _ = np.linalg.eigh(matrix)
    guesses = np.eye(12)[:3]

    solver = BlockDavidsonSolver(
        max_iterations=80,
        maximum_subspace_dimension=9,
        retained_ritz_vectors=4,
        residual_tolerance=1.0e-10,
        progress_interval=None,
    )
    result = solver.solve(
        guesses,
        lambda state: matrix @ state,
        np.vdot,
        number_of_roots=3,
        preconditioner=DiagonalDavidsonPreconditioner(np.diag(matrix)),
        print_progress=False,
    )

    assert result.converged
    assert np.allclose(result.eigenvalues, reference[:3], atol=2.0e-10)
    assert np.all(result.residual_norms < 1.0e-10)
    assert result.hamiltonian_actions >= 3
