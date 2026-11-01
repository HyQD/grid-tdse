import numpy as np
import pytest

from grid_tdse.ground_state import (
    DavidsonGroundStateSolver,
    DiagonalDavidsonPreconditioner,
    IdentityDavidsonPreconditioner,
    LanczosGroundStateSolver,
)


def test_identity_davidson_matches_unrestarted_lanczos():
    hamiltonian = np.array(
        [
            [0.5, 0.2, 0.0, 0.0],
            [0.2, 1.5, 0.3, 0.0],
            [0.0, 0.3, 2.5, -0.1],
            [0.0, 0.0, -0.1, 4.0],
        ]
    )
    initial = np.ones(4)
    options = {
        "max_iterations": 4,
        "residual_tolerance": 1.0e-12,
        "progress_interval": None,
    }

    lanczos = LanczosGroundStateSolver(**options).solve(
        initial,
        lambda state: hamiltonian @ state,
        np.vdot,
    )
    davidson = DavidsonGroundStateSolver(**options).solve(
        initial,
        lambda state: hamiltonian @ state,
        np.vdot,
    )

    assert lanczos.converged
    assert davidson.converged
    assert davidson.eigenvalue == pytest.approx(lanczos.eigenvalue, abs=1.0e-13)
    assert davidson.residual_norm <= options["residual_tolerance"]
    assert davidson.preconditioner_applications == davidson.iterations - 1
    assert davidson.preconditioner_fallbacks == 0
    assert davidson.restarts == 0


def test_diagonal_davidson_reduces_actions_for_diagonally_dominant_problem():
    rng = np.random.default_rng(5)
    dimension = 20
    diagonal = np.linspace(1.0, 30.0, dimension)
    random = rng.normal(size=(dimension, dimension))
    off_diagonal = 0.02 * (random + random.T)
    np.fill_diagonal(off_diagonal, 0.0)
    hamiltonian = np.diag(diagonal) + off_diagonal
    options = {
        "max_iterations": 80,
        "maximum_subspace_dimension": 8,
        "retained_ritz_vectors": 2,
        "residual_tolerance": 1.0e-10,
        "track_orthogonality": False,
        "progress_interval": None,
    }
    initial = np.ones(dimension)

    identity = DavidsonGroundStateSolver(**options).solve(
        initial,
        lambda state: hamiltonian @ state,
        np.vdot,
    )
    preconditioned = DavidsonGroundStateSolver(**options).solve(
        initial,
        lambda state: hamiltonian @ state,
        np.vdot,
        preconditioner=DiagonalDavidsonPreconditioner(
            np.diag(hamiltonian)
        ),
    )
    expected, _ = np.linalg.eigh(hamiltonian)

    assert identity.converged
    assert preconditioned.converged
    assert preconditioned.eigenvalue == pytest.approx(expected[0], abs=1.0e-11)
    assert preconditioned.hamiltonian_actions < identity.hamiltonian_actions
    assert preconditioned.preconditioner_applications > 0
    assert preconditioned.restarts > 0


def test_diagonal_preconditioner_preserves_complex_shape_and_regularizes():
    preconditioner = DiagonalDavidsonPreconditioner(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        relative_floor=0.0,
        absolute_floor=0.25,
    )
    residual = np.array(
        [[1.0 + 0.5j, -2.0j], [0.25, 1.5 - 0.2j]]
    )

    correction = preconditioner(residual, 2.0)

    expected_denominator = np.array([[-1.0, 0.25], [1.0, 2.0]])
    np.testing.assert_allclose(correction, -residual / expected_denominator)
    assert correction.shape == residual.shape
    assert np.iscomplexobj(correction)
    assert preconditioner.last_regularized_entries == 1


def test_diagonal_preconditioner_level_shift_moves_denominators_from_zero():
    preconditioner = DiagonalDavidsonPreconditioner(
        np.array([1.0, 2.0, 3.0]),
        relative_floor=0.0,
        absolute_floor=1.0e-12,
        level_shift=0.5,
    )

    correction = preconditioner(np.ones(3), 2.0)

    np.testing.assert_allclose(correction, [2.0 / 3.0, -2.0, -2.0 / 3.0])
    assert preconditioner.last_regularized_entries == 0


def test_davidson_falls_back_when_preconditioner_returns_dependent_vector():
    hamiltonian = np.array(
        [[1.0, 0.2, 0.0], [0.2, 2.0, 0.3], [0.0, 0.3, 4.0]]
    )

    def unusable_preconditioner(residual, ritz_value):
        del residual, ritz_value
        return np.zeros(3)

    solver = DavidsonGroundStateSolver(
        max_iterations=3,
        residual_tolerance=1.0e-12,
        progress_interval=None,
    )
    result = solver.solve(
        np.ones(3),
        lambda state: hamiltonian @ state,
        np.vdot,
        preconditioner=unusable_preconditioner,
    )

    assert result.converged
    assert result.preconditioner_applications == result.iterations - 1
    assert result.preconditioner_fallbacks == result.iterations - 1


def test_restarted_davidson_tracks_preconditioner_and_restart_diagnostics():
    eigenvalues = np.linspace(0.5, 12.0, 12)
    solver = DavidsonGroundStateSolver(
        max_iterations=100,
        maximum_subspace_dimension=5,
        retained_ritz_vectors=2,
        residual_tolerance=1.0e-10,
        progress_interval=None,
    )

    result = solver.solve(
        np.ones(12),
        lambda state: eigenvalues * state,
        np.vdot,
        preconditioner=IdentityDavidsonPreconditioner(),
    )

    assert result.converged
    assert result.restarts > 0
    assert result.maximum_subspace_dimension == 5
    assert result.preconditioner_applications == result.iterations - 1
    assert result.history[-1].preconditioner_applications == (
        result.preconditioner_applications
    )


def test_davidson_observer_receives_read_only_states(capsys):
    observed = []

    def observer(state, record):
        assert not state.flags.writeable
        observed.append((state.copy(), record))

    solver = DavidsonGroundStateSolver(
        max_iterations=2,
        residual_tolerance=1.0e-14,
        progress_interval=1,
    )
    result = solver.solve(
        np.ones(2),
        lambda state: np.array([0.5, 2.0]) * state,
        np.vdot,
        iteration_observer=observer,
    )

    assert result.converged
    assert len(observed) == result.iterations
    assert "Davidson" in capsys.readouterr().out


def test_davidson_validates_preconditioner_and_its_action():
    solver = DavidsonGroundStateSolver(max_iterations=2)
    with pytest.raises(TypeError, match="preconditioner"):
        solver.solve(
            np.ones(2),
            lambda state: np.array([1.0, 2.0]) * state,
            np.vdot,
            preconditioner=1,
        )
    with pytest.raises(ValueError, match="shape"):
        solver.solve(
            np.ones(2),
            lambda state: np.array([1.0, 2.0]) * state,
            np.vdot,
            preconditioner=lambda residual, value: np.ones(3),
        )
    with pytest.raises(ValueError, match="finite"):
        solver.solve(
            np.ones(2),
            lambda state: np.array([1.0, 2.0]) * state,
            np.vdot,
            preconditioner=lambda residual, value: np.full(2, np.nan),
        )


def test_diagonal_preconditioner_validates_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        DiagonalDavidsonPreconditioner(np.array([]))
    with pytest.raises(ValueError, match="imaginary"):
        DiagonalDavidsonPreconditioner(np.array([1.0 + 1.0j]))
    with pytest.raises(ValueError, match="absolute_floor"):
        DiagonalDavidsonPreconditioner(np.ones(2), absolute_floor=0.0)

    preconditioner = DiagonalDavidsonPreconditioner(np.ones(2))
    with pytest.raises(ValueError, match="shape"):
        preconditioner(np.ones(3), 0.5)
    with pytest.raises(ValueError, match="finite"):
        preconditioner(np.ones(2), np.inf)
