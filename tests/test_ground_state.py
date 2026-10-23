import numpy as np
import pytest

from grid_tdse.ground_state import ImaginaryTimeGroundStateSolver


def _weighted_inner_product(weights):
    def inner_product(left, right):
        return np.sum(left.conj() * right * weights)

    return inner_product


@pytest.mark.parametrize("linear_solver", ["gmres", "bicgstab"])
def test_itp_converges_to_lowest_represented_eigenstate_with_weighted_norm(
    linear_solver,
):
    eigenvalues = np.array([0.5, 1.5, 3.0])
    weights = np.array([2.0, 3.0, 4.0])
    inner_product = _weighted_inner_product(weights)
    solver = ImaginaryTimeGroundStateSolver(
        time_step=0.4,
        max_iterations=200,
        energy_tolerance=1.0e-12,
        residual_tolerance=1.0e-10,
        linear_solver=linear_solver,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
    )

    result = solver.solve(
        np.ones(3),
        lambda state: eigenvalues * state,
        inner_product,
    )

    assert result.converged
    assert result.termination_reason == "converged"
    assert result.energy == pytest.approx(eigenvalues[0], abs=1.0e-12)
    assert result.state_change <= solver.state_tolerance
    assert result.residual_norm <= solver.residual_tolerance
    assert inner_product(result.state, result.state) == pytest.approx(1.0)
    assert abs(result.state[0]) == pytest.approx(1.0 / np.sqrt(weights[0]))
    assert np.linalg.norm(result.state[1:]) < 1.0e-10
    assert result.iterations == len(result.history)
    assert result.energy_history.shape == (result.iterations,)
    assert result.state_change_history.shape == (result.iterations,)
    assert result.residual_history.shape == (result.iterations,)


def test_one_iteration_matches_normalized_imaginary_time_cayley_step():
    hamiltonian = np.array([[0.8, 0.2], [0.2, 1.4]])
    initial = np.array([1.0, -0.3])
    original = initial.copy()
    time_step = 0.2
    solver = ImaginaryTimeGroundStateSolver(
        time_step=time_step,
        max_iterations=1,
        energy_tolerance=1.0e-30,
        residual_tolerance=1.0e-30,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
    )

    result = solver.solve(initial, lambda state: hamiltonian @ state, np.vdot)

    normalized = initial / np.linalg.norm(initial)
    identity = np.eye(2)
    expected = np.linalg.solve(
        identity + 0.5 * time_step * hamiltonian,
        (identity - 0.5 * time_step * hamiltonian) @ normalized,
    )
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result.state, expected, atol=2.0e-13)
    np.testing.assert_array_equal(initial, original)
    assert not result.converged
    assert result.termination_reason == "maximum_iterations"
    assert result.iterations == 1
    assert result.history[0].imaginary_time == pytest.approx(time_step)
    assert result.history[0].norm == pytest.approx(1.0)
    assert result.norm == pytest.approx(1.0)
    assert result.history[0].linear_iterations > 0
    assert result.history[0].linear_residual_norm < 1.0e-12


def test_itp_supports_complex_shaped_states_and_a_shaped_preconditioner():
    diagonal = np.array([[0.4, 0.9], [1.3, 2.0]])
    calls = []

    def preconditioner(state):
        calls.append(state.shape)
        return state

    solver = ImaginaryTimeGroundStateSolver(
        time_step=0.3,
        max_iterations=150,
        energy_tolerance=1.0e-11,
        residual_tolerance=1.0e-9,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
    )
    initial = np.array([[1.0 + 0.2j, 0.5j], [-0.3, 0.7 - 0.1j]])

    result = solver.solve(
        initial,
        lambda state: diagonal * state,
        np.vdot,
        preconditioner=preconditioner,
    )

    assert result.converged
    assert result.state.shape == initial.shape
    assert np.iscomplexobj(result.state)
    assert result.energy == pytest.approx(0.4, abs=1.0e-11)
    assert calls
    assert set(calls) == {initial.shape}


def test_convergence_requires_energy_state_and_residual_tolerances():
    solver = ImaginaryTimeGroundStateSolver(
        time_step=0.1,
        max_iterations=1,
        energy_tolerance=1.0,
        state_tolerance=1.0e-30,
        residual_tolerance=1.0,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
    )
    result = solver.solve(
        np.ones(2),
        lambda state: np.array([0.5, 2.0]) * state,
        np.vdot,
    )

    assert result.history[0].energy_change < solver.energy_tolerance
    assert result.history[0].state_change > solver.state_tolerance
    assert result.history[0].residual_norm < solver.residual_tolerance
    assert not result.converged


def test_state_change_is_invariant_to_a_global_sign_flip():
    solver = ImaginaryTimeGroundStateSolver(
        time_step=1.0,
        max_iterations=1,
        energy_tolerance=1.0e-14,
        state_tolerance=1.0e-14,
        residual_tolerance=1.0e-14,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
    )
    initial = np.array([1.0, -2.0])

    result = solver.solve(initial, lambda state: 3.0 * state, np.vdot)

    assert result.converged
    assert result.state_change == pytest.approx(0.0, abs=1.0e-14)
    normalized = initial / np.linalg.norm(initial)
    np.testing.assert_allclose(result.state, -normalized, atol=1.0e-13)


@pytest.mark.parametrize(
    ("initial_state", "hamiltonian", "inner_product", "message"),
    [
        (np.zeros(2), lambda state: state, np.vdot, "zero norm"),
        (np.ones(2), lambda state: np.ones(3), np.vdot, "shape"),
        (np.ones(2), lambda state: state, lambda left, right: np.ones(2), "scalar"),
        (
            np.ones(2),
            lambda state: state,
            lambda left, right: 1.0 + 1.0j,
            "imaginary part",
        ),
    ],
)
def test_itp_rejects_invalid_representation_operations(
    initial_state, hamiltonian, inner_product, message
):
    solver = ImaginaryTimeGroundStateSolver(max_iterations=1)

    with pytest.raises(ValueError, match=message):
        solver.solve(initial_state, hamiltonian, inner_product)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("time_step", 0.0, ValueError),
        ("max_iterations", 1.5, TypeError),
        ("energy_tolerance", -1.0, ValueError),
        ("state_tolerance", 0.0, ValueError),
        ("residual_tolerance", 0.0, ValueError),
        ("linear_solver", "invalid", ValueError),
        ("linear_rtol", 0.0, ValueError),
    ],
)
def test_itp_validates_solver_options(keyword, value, error):
    options = {keyword: value}
    if keyword == "linear_rtol":
        options["linear_atol"] = 0.0
    with pytest.raises(error):
        ImaginaryTimeGroundStateSolver(**options)
