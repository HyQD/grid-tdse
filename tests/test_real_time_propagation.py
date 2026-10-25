import numpy as np
import pytest

from grid_tdse.real_time import RealTimePropagator


def _weighted_inner_product(weights):
    def inner_product(left, right):
        return np.sum(left.conj() * right * weights)

    return inner_product


@pytest.mark.parametrize("linear_solver", ["gmres", "bicgstab"])
def test_real_time_matches_crank_nicolson_cayley_factors(linear_solver):
    energies = np.array([[0.4, 0.9], [1.3, 2.0]])
    weights = np.array([[2.0, 3.0], [4.0, 5.0]])
    initial = np.array([[1.0, 0.2j], [-0.3, 0.7 - 0.1j]])
    original = initial.copy()
    propagator = RealTimePropagator(
        final_time=0.6,
        time_step=0.2,
        linear_solver=linear_solver,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=None,
    )

    result = propagator.propagate(
        initial,
        lambda state, time: energies * state,
        _weighted_inner_product(weights),
    )

    factor = (1.0 - 0.5j * result.time_step * energies) / (
        1.0 + 0.5j * result.time_step * energies
    )
    expected = initial * factor**result.n_steps
    np.testing.assert_allclose(result.state, expected, atol=2.0e-13)
    np.testing.assert_array_equal(initial, original)
    assert result.state.shape == initial.shape
    assert result.n_steps == 3
    assert result.norm == pytest.approx(result.initial_norm, abs=2.0e-13)
    np.testing.assert_allclose(
        result.norm_history,
        np.full(4, result.initial_norm),
        atol=2.0e-13,
    )
    assert result.linear_iterations.shape == (3,)
    assert np.all(result.linear_iterations > 0)
    assert np.max(result.linear_residual_norms) < 1.0e-12


def test_time_step_is_adjusted_to_reach_final_time_exactly():
    evaluated_times = []
    propagator = RealTimePropagator(
        initial_time=0.2,
        final_time=1.2,
        time_step=0.3,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=None,
    )

    def hamiltonian(state, time):
        evaluated_times.append(time)
        return time * state

    result = propagator(np.ones(2), hamiltonian, np.vdot)

    assert result.requested_time_step == pytest.approx(0.3)
    assert result.time_step == pytest.approx(0.25)
    assert result.n_steps == 4
    np.testing.assert_allclose(result.times, np.linspace(0.2, 1.2, 5))
    np.testing.assert_allclose(np.unique(evaluated_times), [0.325, 0.575, 0.825, 1.075])


def test_real_time_does_not_normalize_the_initial_state():
    initial = np.array([3.0, 4.0])
    propagator = RealTimePropagator(
        final_time=0.4,
        time_step=0.2,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=None,
    )

    result = propagator(initial, lambda state, time: 2.0 * state, np.vdot)

    assert result.initial_norm == pytest.approx(25.0)
    assert result.norm == pytest.approx(25.0, abs=1.0e-13)


def test_real_time_supports_a_shaped_preconditioner():
    calls = []

    def preconditioner(state):
        calls.append(state.shape)
        return state

    propagator = RealTimePropagator(
        final_time=0.2,
        time_step=0.2,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=None,
    )
    initial = np.ones((2, 3))

    result = propagator(
        initial,
        lambda state, time: state,
        np.vdot,
        preconditioner=preconditioner,
    )

    assert result.state.shape == initial.shape
    assert calls
    assert set(calls) == {initial.shape}


def test_real_time_samples_initial_state_and_every_accepted_step():
    propagator = RealTimePropagator(
        initial_time=0.2,
        final_time=0.5,
        time_step=0.1,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=None,
    )
    initial = np.array([1.0, 2.0])

    result = propagator(
        initial,
        lambda state, time: np.zeros_like(state),
        np.vdot,
        samplers={
            "sum": lambda state, time: np.sum(state),
            "time_and_norm": lambda state, time: np.array(
                [time, np.vdot(state, state).real]
            ),
        },
    )

    np.testing.assert_allclose(result.samples["sum"], np.full(4, 3.0))
    np.testing.assert_allclose(
        result.samples["time_and_norm"],
        np.column_stack((result.times, np.full(4, 5.0))),
    )


def test_post_step_transformation_precedes_diagnostics_and_sampling():
    propagator = RealTimePropagator(
        final_time=0.2,
        time_step=0.1,
        progress_interval=None,
    )
    initial = np.ones(2)

    result = propagator.propagate(
        initial,
        lambda state, time: np.zeros_like(state),
        np.vdot,
        post_step=lambda state, time: 0.5 * state,
        samplers={"amplitudes": lambda state, time: state.copy()},
    )

    np.testing.assert_allclose(result.state, 0.25 * initial)
    np.testing.assert_allclose(result.norm_history, [2.0, 0.5, 0.125])
    np.testing.assert_allclose(
        result.samples["amplitudes"],
        [initial, 0.5 * initial, 0.25 * initial],
    )


@pytest.mark.parametrize(
    ("post_step", "error", "message"),
    [
        (1.0, TypeError, "callable"),
        (lambda state, time: np.ones(3), ValueError, "shape"),
        (
            lambda state, time: np.array([np.nan, 0.0]),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_real_time_validates_post_step_transformations(
    post_step, error, message
):
    propagator = RealTimePropagator(final_time=0.1, time_step=0.1)

    with pytest.raises(error, match=message):
        propagator.propagate(
            np.ones(2),
            lambda state, time: np.zeros_like(state),
            np.vdot,
            post_step=post_step,
        )


def test_sampler_receives_a_read_only_state_view():
    propagator = RealTimePropagator(
        final_time=0.1,
        time_step=0.1,
        progress_interval=None,
    )

    def mutate(state, time):
        state[0] = time
        return 0.0

    with pytest.raises(ValueError, match="read-only"):
        propagator(
            np.ones(2),
            lambda state, time: state,
            np.vdot,
            samplers={"mutation": mutate},
        )


@pytest.mark.parametrize(
    ("samplers", "error", "message"),
    [
        ([lambda state, time: 0.0], TypeError, "mapping"),
        ({"": lambda state, time: 0.0}, ValueError, "nonempty"),
        ({"value": 1.0}, TypeError, "callable"),
        (
            {"value": lambda state, time: np.nan},
            ValueError,
            "non-finite",
        ),
    ],
)
def test_real_time_validates_samplers(samplers, error, message):
    propagator = RealTimePropagator(final_time=0.1, time_step=0.1)

    with pytest.raises(error, match=message):
        propagator(
            np.ones(2),
            lambda state, time: state,
            np.vdot,
            samplers=samplers,
        )


def test_sampler_output_shape_must_remain_constant():
    propagator = RealTimePropagator(
        final_time=0.2,
        time_step=0.1,
        progress_interval=None,
    )

    with pytest.raises(ValueError, match="changed its output shape"):
        propagator(
            np.ones(2),
            lambda state, time: state,
            np.vdot,
            samplers={
                "changing": lambda state, time: (
                    np.ones(1) if time == 0.0 else np.ones(2)
                )
            },
        )


@pytest.mark.parametrize(
    ("initial_state", "hamiltonian", "inner_product", "error", "message"),
    [
        (
            np.zeros(2),
            lambda state, time: state,
            np.vdot,
            ValueError,
            "zero norm",
        ),
        (
            np.ones(2),
            lambda state, time: np.ones(3),
            np.vdot,
            ValueError,
            "shape",
        ),
        (
            np.ones(2),
            lambda state, time: state,
            lambda left, right: np.ones(2),
            ValueError,
            "scalar",
        ),
        (
            np.ones(2),
            lambda state, time: state,
            lambda left, right: 1.0 + 1.0j,
            ValueError,
            "imaginary part",
        ),
        (
            np.ones(2),
            lambda state, time: np.array([np.nan, 1.0]),
            np.vdot,
            ValueError,
            "non-finite",
        ),
    ],
)
def test_real_time_rejects_invalid_representation_operations(
    initial_state,
    hamiltonian,
    inner_product,
    error,
    message,
):
    propagator = RealTimePropagator(final_time=0.1, time_step=0.1)

    with pytest.raises(error, match=message):
        propagator(initial_state, hamiltonian, inner_product)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("initial_time", np.inf, ValueError),
        ("final_time", 0.0, ValueError),
        ("time_step", 0.0, ValueError),
        ("linear_solver", "invalid", ValueError),
        ("linear_rtol", 0.0, ValueError),
        ("linear_max_iterations", 1.5, TypeError),
        ("gmres_restart", 0, ValueError),
        ("reality_tolerance", -1.0, ValueError),
        ("progress_interval", 0, ValueError),
    ],
)
def test_real_time_validates_propagator_options(keyword, value, error):
    options = {"final_time": 1.0, "time_step": 0.1, keyword: value}
    if keyword == "linear_rtol":
        options["linear_atol"] = 0.0
    with pytest.raises(error):
        RealTimePropagator(**options)


def test_real_time_reports_progress_at_requested_interval_and_final_step(capsys):
    assert RealTimePropagator(final_time=1.0, time_step=0.1).progress_interval == 10
    propagator = RealTimePropagator(
        final_time=0.7,
        time_step=0.1,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=3,
    )

    propagator(np.ones(2), lambda state, time: state, np.vdot)

    lines = capsys.readouterr().out.splitlines()
    assert [int(line.split()[1].split("/")[0]) for line in lines] == [3, 6, 7]
    assert all("t=" in line for line in lines)
    assert all("norm=" in line for line in lines)
    assert all("linear_iterations=" in line for line in lines)
    assert all("linear_residual=" in line for line in lines)


def test_real_time_progress_can_be_disabled(capsys):
    propagator = RealTimePropagator(
        final_time=0.1,
        time_step=0.1,
        progress_interval=None,
    )
    propagator(np.ones(2), lambda state, time: state, np.vdot)

    assert capsys.readouterr().out == ""
