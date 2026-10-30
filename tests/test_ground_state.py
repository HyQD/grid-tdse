import numpy as np
import pytest
from scipy.linalg import expm

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


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_one_iteration_matches_normalized_imaginary_time_exponential(integrator):
    hamiltonian = np.array([[0.8, 0.2], [0.2, 1.4]])
    initial = np.array([1.0, -0.3])
    time_step = 0.2
    solver = ImaginaryTimeGroundStateSolver(
        integrator=integrator,
        time_step=time_step,
        max_iterations=1,
        energy_tolerance=1.0e-30,
        state_tolerance=1.0e-30,
        residual_tolerance=1.0e-30,
        max_krylov_dimension=2,
        krylov_rtol=1.0e-13,
        krylov_atol=1.0e-15,
    )

    result = solver.solve(initial, lambda state: hamiltonian @ state, np.vdot)

    normalized = initial / np.linalg.norm(initial)
    expected = expm(-time_step * hamiltonian) @ normalized
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result.state, expected, atol=2.0e-13)
    assert result.integrator == integrator
    assert result.history[0].linear_iterations == 0
    assert result.history[0].krylov_operator_calls == 2
    assert result.history[0].krylov_dimension == 2
    assert result.history[0].krylov_error_estimate == 0.0


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_krylov_itp_forwards_batched_inner_products(integrator):
    hamiltonian = np.array([[0.8, 0.2], [0.2, 1.4]])
    batches = []

    def inner_products(left, right):
        batches.append(len(left))
        return np.asarray([np.vdot(vector, right) for vector in left])

    solver = ImaginaryTimeGroundStateSolver(
        integrator=integrator,
        time_step=0.2,
        max_iterations=1,
        max_krylov_dimension=2,
        krylov_rtol=1.0e-13,
        krylov_atol=1.0e-15,
    )

    solver.solve(
        np.array([1.0, -0.3]),
        lambda state: hamiltonian @ state,
        np.vdot,
        inner_products=inner_products,
    )

    assert batches
    assert max(batches) == 2


def test_itp_rejects_noncallable_batched_inner_products():
    solver = ImaginaryTimeGroundStateSolver(
        integrator="sil",
        max_iterations=1,
    )

    with pytest.raises(TypeError, match="inner_products must be callable"):
        solver.solve(
            np.ones(2),
            lambda state: state,
            np.vdot,
            inner_products=1,
        )


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_krylov_itp_converges_to_lowest_represented_eigenstate(integrator):
    eigenvalues = np.array([0.5, 1.5, 3.0])
    solver = ImaginaryTimeGroundStateSolver(
        integrator=integrator,
        time_step=0.4,
        max_iterations=200,
        energy_tolerance=1.0e-12,
        residual_tolerance=1.0e-10,
        max_krylov_dimension=3,
        krylov_rtol=1.0e-13,
        krylov_atol=1.0e-15,
    )

    result = solver.solve(
        np.ones(3),
        lambda state: eigenvalues * state,
        np.vdot,
    )

    assert result.converged
    assert result.energy == pytest.approx(eigenvalues[0], abs=1.0e-12)
    assert result.residual_norm <= solver.residual_tolerance
    assert np.all(result.linear_iterations == 0)
    assert np.all(result.krylov_operator_calls > 0)
    assert np.all(result.krylov_dimensions > 0)


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_krylov_itp_rejects_a_preconditioner(integrator):
    solver = ImaginaryTimeGroundStateSolver(
        integrator=integrator,
        max_iterations=1,
    )

    with pytest.raises(ValueError, match="preconditioner is not supported"):
        solver.solve(
            np.ones(2),
            lambda state: state,
            np.vdot,
            preconditioner=lambda state: state,
        )


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
        ("integrator", "invalid", ValueError),
        ("linear_solver", "invalid", ValueError),
        ("linear_rtol", 0.0, ValueError),
        ("max_krylov_dimension", 0, ValueError),
        ("krylov_rtol", 0.0, ValueError),
        ("max_krylov_refinements", -1, ValueError),
        ("krylov_reorthogonalize", 1, TypeError),
        ("progress_interval", 0, ValueError),
    ],
)
def test_itp_validates_solver_options(keyword, value, error):
    options = {keyword: value}
    if keyword == "linear_rtol":
        options["linear_atol"] = 0.0
    if keyword == "krylov_rtol":
        options["krylov_atol"] = 0.0
    with pytest.raises(error):
        ImaginaryTimeGroundStateSolver(**options)


def test_itp_reports_progress_at_requested_interval_and_final_step(capsys):
    assert ImaginaryTimeGroundStateSolver().progress_interval == 10

    solver = ImaginaryTimeGroundStateSolver(
        time_step=0.1,
        max_iterations=7,
        energy_tolerance=1.0e-30,
        state_tolerance=1.0e-30,
        residual_tolerance=1.0e-30,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=3,
    )
    solver.solve(
        np.ones(2),
        lambda state: np.array([0.5, 2.0]) * state,
        np.vdot,
    )

    lines = capsys.readouterr().out.splitlines()
    assert [int(line.split()[1].removesuffix(":")) for line in lines] == [3, 6, 7]
    assert all("E=" in line for line in lines)
    energy_fields = [line.split("E=")[1].split()[0] for line in lines]
    assert all("." in field for field in energy_fields)
    assert all("e" not in field.lower() for field in energy_fields)
    assert all("|dE|=" in line for line in lines)
    assert all("dPsi=" in line for line in lines)
    assert all("residual=" in line for line in lines)


def test_itp_progress_can_be_disabled(capsys):
    solver = ImaginaryTimeGroundStateSolver(
        max_iterations=1,
        progress_interval=None,
    )
    solver.solve(np.ones(2), lambda state: state, np.vdot)

    assert capsys.readouterr().out == ""


def test_itp_observer_receives_initial_interval_and_final_states(capsys):
    samples = []

    def observe(state, diagnostics):
        assert not state.flags.writeable
        samples.append((state.copy(), diagnostics))

    solver = ImaginaryTimeGroundStateSolver(
        time_step=0.1,
        max_iterations=5,
        energy_tolerance=1.0e-30,
        state_tolerance=1.0e-30,
        residual_tolerance=1.0e-30,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=1,
    )
    result = solver.solve(
        np.ones(2),
        lambda state: np.array([0.5, 2.0]) * state,
        np.vdot,
        iteration_observer=observe,
        observer_interval=2,
        print_progress=False,
    )

    assert [record.iteration for _, record in samples] == [0, 2, 4, 5]
    assert samples[0][1].imaginary_time == 0.0
    assert samples[0][1].step_wall_time == 0.0
    assert all(record.step_wall_time >= 0.0 for _, record in samples)
    np.testing.assert_array_equal(samples[-1][0], result.state)
    assert capsys.readouterr().out == ""


def test_itp_observer_can_request_only_initial_and_final_states():
    iterations = []
    solver = ImaginaryTimeGroundStateSolver(
        max_iterations=3,
        energy_tolerance=1.0e-30,
        state_tolerance=1.0e-30,
        residual_tolerance=1.0e-30,
        progress_interval=None,
    )

    solver.solve(
        np.ones(2),
        lambda state: np.array([0.5, 2.0]) * state,
        np.vdot,
        iteration_observer=lambda state, record: iterations.append(
            record.iteration
        ),
    )

    assert iterations == [0, 3]


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("iteration_observer", 1, TypeError),
        ("observer_interval", 0, ValueError),
        ("observer_interval", 1.5, TypeError),
        ("print_progress", 1, TypeError),
    ],
)
def test_itp_validates_observer_options(keyword, value, error):
    solver = ImaginaryTimeGroundStateSolver(max_iterations=1)

    with pytest.raises(error):
        solver.solve(
            np.ones(2),
            lambda state: state,
            np.vdot,
            **{keyword: value},
        )
