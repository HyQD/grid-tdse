import numpy as np
import pytest

from grid_tdse.ground_state import LanczosGroundStateSolver
from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
)


def _weighted_inner_product(weights):
    weights = np.asarray(weights)

    def inner_product(left, right):
        return np.vdot(left, weights * right)

    return inner_product


def _weighted_operator(transformed_hamiltonian, weights):
    square_root = np.sqrt(weights)
    return (
        transformed_hamiltonian * square_root[np.newaxis, :]
    ) / square_root[:, np.newaxis]


def test_lanczos_converges_to_lowest_weighted_eigenpair():
    transformed = np.array(
        [
            [0.5, 0.2, 0.0, 0.0],
            [0.2, 1.5, 0.3, 0.0],
            [0.0, 0.3, 2.5, -0.1],
            [0.0, 0.0, -0.1, 4.0],
        ]
    )
    weights = np.array([2.0, 3.0, 4.0, 5.0])
    hamiltonian = _weighted_operator(transformed, weights)
    inner_product = _weighted_inner_product(weights)
    initial = np.ones(4)
    original = initial.copy()
    solver = LanczosGroundStateSolver(
        max_iterations=4,
        residual_tolerance=1.0e-12,
        progress_interval=None,
    )

    result = solver.solve(
        initial,
        lambda state: hamiltonian @ state,
        inner_product,
    )
    expected_values, expected_vectors = np.linalg.eigh(transformed)
    expected_state = expected_vectors[:, 0] / np.sqrt(weights)

    assert result.converged
    assert result.termination_reason == "converged"
    assert result.eigenvalue == pytest.approx(expected_values[0], abs=1.0e-13)
    assert result.residual_norm <= solver.residual_tolerance
    assert result.iterations <= 4
    assert result.hamiltonian_actions == result.iterations
    assert result.restarts == 0
    assert result.preconditioner_applications == 0
    assert inner_product(result.state, result.state) == pytest.approx(1.0)
    overlap = inner_product(expected_state, result.state)
    assert abs(overlap) == pytest.approx(1.0, abs=1.0e-13)
    np.testing.assert_array_equal(initial, original)
    assert np.max(
        [record.orthogonality_error for record in result.history]
    ) < 2.0e-14


def test_lanczos_supports_complex_hermitian_shaped_states():
    hamiltonian = np.array(
        [
            [0.4, 0.2j, 0.0, 0.1],
            [-0.2j, 1.0, 0.3, 0.0],
            [0.0, 0.3, 2.0, -0.1j],
            [0.1, 0.0, 0.1j, 3.0],
        ],
        dtype=complex,
    )
    initial = np.array([[1.0 + 0.1j, 0.5], [-0.3j, 0.7]])
    solver = LanczosGroundStateSolver(
        max_iterations=4,
        residual_tolerance=2.0e-13,
        progress_interval=None,
    )

    result = solver.solve(
        initial,
        lambda state: (hamiltonian @ state.ravel()).reshape(state.shape),
        lambda left, right: np.vdot(left, right),
    )
    expected, _ = np.linalg.eigh(hamiltonian)

    assert result.converged
    assert result.state.shape == initial.shape
    assert result.eigenvalue == pytest.approx(expected[0], abs=2.0e-13)
    assert result.residual_norm <= solver.residual_tolerance


def test_lanczos_accepts_an_exact_lowest_eigenstate_in_one_action():
    diagonal = np.array([0.5, 1.5, 3.0])
    solver = LanczosGroundStateSolver(
        max_iterations=3,
        residual_tolerance=1.0e-14,
        progress_interval=None,
    )

    result = solver.solve(
        np.array([1.0, 0.0, 0.0]),
        lambda state: diagonal * state,
        np.vdot,
    )

    assert result.converged
    assert result.iterations == 1
    assert result.hamiltonian_actions == 1
    assert result.eigenvalue == pytest.approx(0.5)
    np.testing.assert_allclose(result.state, [1.0, 0.0, 0.0])


def test_lanczos_normalizes_a_nonzero_small_scaled_initial_state():
    solver = LanczosGroundStateSolver(
        max_iterations=2,
        residual_tolerance=1.0e-14,
        progress_interval=None,
    )

    result = solver.solve(
        np.array([1.0e-30, 0.0]),
        lambda state: np.array([0.5, 2.0]) * state,
        np.vdot,
    )

    assert result.converged
    assert result.eigenvalue == pytest.approx(0.5)
    np.testing.assert_allclose(result.state, [1.0, 0.0])


def test_lanczos_finds_lowest_state_represented_by_initial_vector():
    diagonal = np.array([0.5, 1.5, 3.0])
    solver = LanczosGroundStateSolver(
        max_iterations=3,
        residual_tolerance=1.0e-13,
        progress_interval=None,
    )

    result = solver.solve(
        np.array([0.0, 1.0, 1.0]),
        lambda state: diagonal * state,
        np.vdot,
    )

    assert result.converged
    assert result.eigenvalue == pytest.approx(1.5)
    assert abs(result.state[0]) < 1.0e-15
    assert abs(result.state[2]) < 1.0e-14


def test_lanczos_reports_maximum_iterations():
    hamiltonian = np.array([[1.0, 0.4], [0.4, 3.0]])
    solver = LanczosGroundStateSolver(
        max_iterations=1,
        residual_tolerance=1.0e-14,
        progress_interval=None,
    )

    result = solver.solve(
        np.array([1.0, 0.0]),
        lambda state: hamiltonian @ state,
        np.vdot,
    )

    assert not result.converged
    assert result.termination_reason == "maximum_iterations"
    assert result.iterations == 1
    assert result.residual_norm == pytest.approx(0.4)


def test_thick_restarted_lanczos_matches_unrestarted_ground_pair():
    eigenvalues = np.linspace(0.5, 12.0, 12)
    initial = np.ones(eigenvalues.size)
    reference = LanczosGroundStateSolver(
        max_iterations=eigenvalues.size,
        residual_tolerance=1.0e-12,
        progress_interval=None,
    ).solve(
        initial,
        lambda state: eigenvalues * state,
        np.vdot,
    )
    solver = LanczosGroundStateSolver(
        max_iterations=80,
        maximum_subspace_dimension=5,
        retained_ritz_vectors=2,
        residual_tolerance=1.0e-10,
        progress_interval=None,
    )

    result = solver.solve(
        initial,
        lambda state: eigenvalues * state,
        np.vdot,
    )

    assert reference.converged
    assert result.converged
    assert result.eigenvalue == pytest.approx(reference.eigenvalue, abs=1.0e-12)
    assert result.residual_norm <= solver.residual_tolerance
    assert result.restarts > 0
    assert result.restarts == np.count_nonzero(
        [record.restarted for record in result.history]
    )
    assert result.maximum_subspace_dimension == 5
    assert np.max(result.subspace_dimensions) <= 5
    assert np.all(
        result.subspace_dimensions[result.restart_iterations - 1] == 5
    )
    assert result.final_subspace_dimension <= 5


def test_thick_restart_retains_clustered_low_ritz_subspace():
    rng = np.random.default_rng(1234)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    eigenvalues = np.array(
        [0.5, 0.5001, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    )
    hamiltonian = orthogonal @ np.diag(eigenvalues) @ orthogonal.T
    solver = LanczosGroundStateSolver(
        max_iterations=100,
        maximum_subspace_dimension=6,
        retained_ritz_vectors=2,
        residual_tolerance=1.0e-9,
        progress_interval=None,
    )

    result = solver.solve(
        np.ones(10),
        lambda state: hamiltonian @ state,
        np.vdot,
    )

    assert result.converged
    assert result.restarts > 0
    assert result.eigenvalue == pytest.approx(eigenvalues[0], abs=1.0e-11)
    assert result.residual_norm <= solver.residual_tolerance
    assert max(
        record.orthogonality_error for record in result.history
    ) < 2.0e-12


def test_lanczos_reports_nonconverged_invariant_subspace():
    hamiltonian = np.diag([1.0, 2.0])
    solver = LanczosGroundStateSolver(
        max_iterations=2,
        residual_tolerance=1.0e-16,
        breakdown_tolerance=1.0e-3,
        progress_interval=None,
    )

    result = solver.solve(
        np.array([1.0, 1.0e-8]),
        lambda state: hamiltonian @ state,
        np.vdot,
    )

    assert not result.converged
    assert result.termination_reason == "invariant_subspace"
    assert result.iterations == 1
    assert result.residual_norm > solver.residual_tolerance


def test_lanczos_forwards_batched_inner_products():
    hamiltonian = np.array(
        [[1.0, 0.2, 0.0], [0.2, 2.0, 0.3], [0.0, 0.3, 4.0]]
    )
    batches = []

    def inner_products(left, right):
        batches.append(len(left))
        return np.asarray([np.vdot(vector, right) for vector in left])

    solver = LanczosGroundStateSolver(
        max_iterations=3,
        residual_tolerance=1.0e-12,
        track_orthogonality=False,
        progress_interval=None,
    )
    result = solver.solve(
        np.ones(3),
        lambda state: hamiltonian @ state,
        np.vdot,
        inner_products=inner_products,
    )

    assert result.converged
    assert batches
    assert max(batches) == 3
    assert all(
        np.isnan(record.orthogonality_error) for record in result.history
    )


def test_lanczos_observer_receives_read_only_ritz_states(capsys):
    records = []

    def observer(state, record):
        records.append((state.copy(), record))
        assert not state.flags.writeable

    solver = LanczosGroundStateSolver(
        max_iterations=2,
        residual_tolerance=1.0e-14,
        progress_interval=1,
    )
    result = solver.solve(
        np.array([1.0, 1.0]),
        lambda state: np.array([0.5, 2.0]) * state,
        np.vdot,
        iteration_observer=observer,
    )

    assert result.converged
    assert len(records) == result.iterations
    assert "Lanczos" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("max_iterations", 0, ValueError),
        ("maximum_subspace_dimension", 1, ValueError),
        ("residual_tolerance", 0.0, ValueError),
        ("breakdown_tolerance", 0.0, ValueError),
        ("reorthogonalization", "never", ValueError),
        ("reorthogonalization_threshold", 1.1, ValueError),
        ("hermiticity_tolerance", -1.0, ValueError),
        ("reality_tolerance", -1.0, ValueError),
        ("track_orthogonality", 1, TypeError),
        ("progress_interval", 0, ValueError),
    ],
)
def test_lanczos_rejects_invalid_options(option, value, error):
    with pytest.raises(error):
        LanczosGroundStateSolver(**{option: value})


def test_lanczos_requires_fewer_retained_vectors_than_restart_dimension():
    with pytest.raises(ValueError, match="retained_ritz_vectors"):
        LanczosGroundStateSolver(
            maximum_subspace_dimension=4,
            retained_ritz_vectors=4,
        )


def test_lanczos_rejects_invalid_initial_state_and_operator_action():
    solver = LanczosGroundStateSolver(max_iterations=1)
    with pytest.raises(ValueError, match="norm"):
        solver.solve(np.zeros(2), lambda state: state, np.vdot)
    with pytest.raises(ValueError, match="shape"):
        solver.solve(
            np.ones(2),
            lambda state: np.ones(3),
            np.vdot,
        )
    with pytest.raises(ValueError, match="non-finite"):
        solver.solve(
            np.ones(2),
            lambda state: np.full(2, np.nan),
            np.vdot,
        )


def test_lanczos_converges_a_small_non_bo_diatomic_hamiltonian():
    electronic = FEMDVRGridSpec([0.0, 2.0, 5.0], [3, 3])
    internuclear = FEMDVRGridSpec([0.0, 1.5, 4.0], [3, 3])
    calculation = NonBornOppenheimerDiatomicIonGroundState(
        "H2+",
        electronic_grid=electronic,
        internuclear_grid=internuclear,
        inversion_parity=1,
        pair_exchange_parity=1,
        l_max=0,
        m_max=0,
        lambda_max=0,
        mu_max=0,
        M_tot=0,
        max_iterations=1,
        progress_interval=None,
        sample_interval=None,
        use_preconditioner=False,
    )
    initial_norm = calculation.representation.inner_product(
        calculation.initial_state,
        calculation.initial_state,
    )
    initial_energy = calculation.representation.inner_product(
        calculation.initial_state,
        calculation.hamiltonian(calculation.initial_state),
    ).real / initial_norm.real
    solver = LanczosGroundStateSolver(
        max_iterations=calculation.initial_state.size,
        residual_tolerance=1.0e-9,
        progress_interval=None,
    )

    result = solver.solve(
        calculation.initial_state,
        calculation.hamiltonian,
        calculation.representation.inner_product,
    )

    assert result.converged
    assert result.residual_norm <= solver.residual_tolerance
    assert result.eigenvalue < initial_energy


def test_thick_restarted_lanczos_converges_small_non_bo_diatomic_hamiltonian():
    electronic = FEMDVRGridSpec([0.0, 2.0, 5.0], [3, 3])
    internuclear = FEMDVRGridSpec([0.0, 1.5, 4.0], [3, 3])
    calculation = NonBornOppenheimerDiatomicIonGroundState(
        "H2+",
        electronic_grid=electronic,
        internuclear_grid=internuclear,
        inversion_parity=1,
        pair_exchange_parity=1,
        l_max=0,
        m_max=0,
        lambda_max=0,
        mu_max=0,
        M_tot=0,
        max_iterations=1,
        progress_interval=None,
        sample_interval=None,
        use_preconditioner=False,
    )
    reference = LanczosGroundStateSolver(
        max_iterations=calculation.initial_state.size,
        residual_tolerance=1.0e-10,
        progress_interval=None,
    ).solve(
        calculation.initial_state,
        calculation.hamiltonian,
        calculation.representation.inner_product,
    )
    solver = LanczosGroundStateSolver(
        max_iterations=80,
        maximum_subspace_dimension=5,
        retained_ritz_vectors=1,
        residual_tolerance=1.0e-9,
        progress_interval=None,
    )

    result = solver.solve(
        calculation.initial_state,
        calculation.hamiltonian,
        calculation.representation.inner_product,
    )

    assert reference.converged
    assert result.converged
    assert result.restarts > 0
    assert result.eigenvalue == pytest.approx(reference.eigenvalue, abs=1.0e-11)
    assert result.residual_norm <= solver.residual_tolerance
