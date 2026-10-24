import numpy as np
import pytest

from grid_tdse.real_time import (
    RealTimePropagator,
    load_real_time_checkpoint,
)


COMPATIBILITY = {
    "physical_system": {"name": "model"},
    "representation": {"basis": "two-level"},
}
DYNAMICS = {"field": {"name": "first", "amplitude": 0.2}}


def _propagator(initial_time=0.0, final_time=0.6, time_step=0.1):
    return RealTimePropagator(
        initial_time=initial_time,
        final_time=final_time,
        time_step=time_step,
        linear_rtol=1.0e-13,
        linear_atol=1.0e-15,
        progress_interval=None,
    )


def _hamiltonian(state, time):
    del time
    return np.array([0.4, 0.9]) * state


def _sampler(state, time):
    del time
    return np.sum(state)


def test_periodic_checkpoint_contains_state_observables_and_metadata(tmp_path):
    path = tmp_path / "propagation.chkpt"
    initial = np.array([1.0, 0.2j])
    result = _propagator(final_time=0.3).propagate(
        initial,
        _hamiltonian,
        np.vdot,
        samplers={"sum": _sampler},
        checkpoint_path=path,
        checkpoint_interval=2,
        compatibility_metadata=COMPATIBILITY,
        dynamics_metadata=DYNAMICS,
        checkpoint_metadata={"description": "checkpoint test"},
    )

    assert path.is_file()
    checkpoint = load_real_time_checkpoint(path)
    np.testing.assert_array_equal(checkpoint.state, result.state)
    assert checkpoint.current_time == pytest.approx(0.3)
    assert checkpoint.current_step == 3
    assert checkpoint.total_steps == 3
    assert checkpoint.compatibility == COMPATIBILITY
    assert checkpoint.metadata == {"description": "checkpoint test"}
    assert checkpoint.state.flags.writeable is False

    segment = checkpoint.latest_segment
    assert segment.complete
    assert segment.checkpoint_interval == 2
    np.testing.assert_allclose(segment.times, result.times)
    np.testing.assert_allclose(segment.norms, result.norm_history)
    np.testing.assert_array_equal(
        segment.linear_iterations,
        result.linear_iterations,
    )
    np.testing.assert_allclose(
        segment.linear_residual_norms,
        result.linear_residual_norms,
    )
    np.testing.assert_allclose(segment.samples["sum"], result.samples["sum"])
    assert segment.dynamics == DYNAMICS
    assert segment.integrator["method"] == "crank_nicolson"


def test_exact_resume_matches_uninterrupted_propagation(tmp_path):
    path = tmp_path / "interrupted.chkpt"
    initial = np.array([1.0, -0.3j])

    def interrupting_sampler(state, time):
        if time > 0.25:
            raise RuntimeError("simulated interruption")
        return np.sum(state)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _propagator().propagate(
            initial,
            _hamiltonian,
            np.vdot,
            samplers={"sum": interrupting_sampler},
            checkpoint_path=path,
            checkpoint_interval=2,
            compatibility_metadata=COMPATIBILITY,
            dynamics_metadata=DYNAMICS,
        )

    checkpoint = load_real_time_checkpoint(path)
    assert checkpoint.current_time == pytest.approx(0.2)
    assert not checkpoint.latest_segment.complete

    resumed = _propagator(initial_time=0.2).resume(
        checkpoint,
        _hamiltonian,
        np.vdot,
        samplers={"sum": _sampler},
        checkpoint_interval=2,
        compatibility_metadata=COMPATIBILITY,
        dynamics_metadata=DYNAMICS,
    )
    uninterrupted = _propagator().propagate(
        initial,
        _hamiltonian,
        np.vdot,
        samplers={"sum": _sampler},
    )

    np.testing.assert_allclose(resumed.state, uninterrupted.state, atol=1.0e-13)
    np.testing.assert_allclose(resumed.times, uninterrupted.times)
    np.testing.assert_allclose(resumed.norm_history, uninterrupted.norm_history)
    np.testing.assert_allclose(
        resumed.samples["sum"],
        uninterrupted.samples["sum"],
    )
    completed = load_real_time_checkpoint(path)
    assert len(completed.segments) == 1
    assert completed.latest_segment.complete
    assert completed.current_time == pytest.approx(0.6)


def test_compatible_continuation_starts_a_new_dynamics_segment(tmp_path):
    path = tmp_path / "two_fields.chkpt"
    first = _propagator(final_time=0.2).propagate(
        np.array([1.0, 0.5]),
        _hamiltonian,
        np.vdot,
        samplers={"sum": _sampler},
        checkpoint_path=path,
        checkpoint_interval=1,
        compatibility_metadata=COMPATIBILITY,
        dynamics_metadata=DYNAMICS,
        checkpoint_metadata={"experiment": "two pulses"},
    )
    first_checkpoint = load_real_time_checkpoint(path)
    second_dynamics = {"field": {"name": "second", "amplitude": 0.7}}

    def second_hamiltonian(state, time):
        del time
        return np.array([1.2, 0.1]) * state

    second = _propagator(initial_time=0.2, final_time=0.5).continue_from(
        first_checkpoint,
        second_hamiltonian,
        np.vdot,
        samplers={"population": lambda state, time: np.abs(state) ** 2},
        checkpoint_interval=2,
        compatibility_metadata=COMPATIBILITY,
        dynamics_metadata=second_dynamics,
        checkpoint_metadata={"second_stage": "different field"},
    )

    chained = load_real_time_checkpoint(path)
    assert len(chained.segments) == 2
    assert chained.total_steps == first.n_steps + second.n_steps
    assert chained.segments[0].dynamics == DYNAMICS
    assert chained.segments[1].dynamics == second_dynamics
    assert set(chained.segments[0].samples) == {"sum"}
    assert set(chained.segments[1].samples) == {"population"}
    assert chained.metadata == {
        "experiment": "two pulses",
        "second_stage": "different field",
    }
    np.testing.assert_array_equal(chained.state, second.state)
    np.testing.assert_array_equal(first_checkpoint.state, first.state)
    assert chained.segments[0].current_time == pytest.approx(
        chained.segments[1].initial_time
    )


def test_checkpoint_rejects_incompatible_continuation(tmp_path):
    path = tmp_path / "state.chkpt"
    _propagator(final_time=0.2).propagate(
        np.ones(2),
        _hamiltonian,
        np.vdot,
        checkpoint_path=path,
        compatibility_metadata=COMPATIBILITY,
    )
    checkpoint = load_real_time_checkpoint(path)

    with pytest.raises(ValueError, match="compatibility metadata"):
        _propagator(initial_time=0.2, final_time=0.3).continue_from(
            checkpoint,
            _hamiltonian,
            np.vdot,
            compatibility_metadata={"representation": "different"},
        )


def test_exact_resume_rejects_completed_segment_and_changed_dynamics(tmp_path):
    completed_path = tmp_path / "completed.chkpt"
    _propagator(final_time=0.2).propagate(
        np.ones(2),
        _hamiltonian,
        np.vdot,
        checkpoint_path=completed_path,
        compatibility_metadata=COMPATIBILITY,
        dynamics_metadata=DYNAMICS,
    )
    completed = load_real_time_checkpoint(completed_path)
    with pytest.raises(ValueError, match="complete"):
        _propagator(initial_time=0.2, final_time=0.3).resume(
            completed,
            _hamiltonian,
            np.vdot,
            compatibility_metadata=COMPATIBILITY,
            dynamics_metadata=DYNAMICS,
        )

    interrupted_path = tmp_path / "incomplete.chkpt"

    def interrupt(state, time):
        if time > 0.15:
            raise RuntimeError
        return state[0]

    with pytest.raises(RuntimeError):
        _propagator(final_time=0.4).propagate(
            np.ones(2),
            _hamiltonian,
            np.vdot,
            samplers={"value": interrupt},
            checkpoint_path=interrupted_path,
            checkpoint_interval=1,
            compatibility_metadata=COMPATIBILITY,
            dynamics_metadata=DYNAMICS,
        )
    interrupted = load_real_time_checkpoint(interrupted_path)
    with pytest.raises(ValueError, match="dynamics metadata"):
        _propagator(initial_time=0.1, final_time=0.4).resume(
            interrupted,
            _hamiltonian,
            np.vdot,
            samplers={"value": lambda state, time: state[0]},
            compatibility_metadata=COMPATIBILITY,
            dynamics_metadata={"field": "changed"},
        )


def test_new_checkpoint_protects_an_existing_path(tmp_path):
    path = tmp_path / "existing.chkpt"
    path.write_bytes(b"existing data")

    with pytest.raises(FileExistsError, match="already exists"):
        _propagator(final_time=0.1).propagate(
            np.ones(2),
            _hamiltonian,
            np.vdot,
            checkpoint_path=path,
        )
    assert path.read_bytes() == b"existing data"


def test_checkpoint_interval_requires_a_path():
    with pytest.raises(ValueError, match="requires a checkpoint_path"):
        _propagator(final_time=0.1).propagate(
            np.ones(2),
            _hamiltonian,
            np.vdot,
            checkpoint_interval=10,
        )


def test_checkpoint_detects_modified_state_data(tmp_path):
    path = tmp_path / "state.chkpt"
    _propagator(final_time=0.1).propagate(
        np.ones(2),
        _hamiltonian,
        np.vdot,
        checkpoint_path=path,
    )
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name].copy() for name in original.files}
    arrays["state"][0] += 1.0
    with path.open("wb") as stream:
        np.savez(stream, **arrays)

    with pytest.raises(ValueError, match="state checksum"):
        load_real_time_checkpoint(path)


def test_checkpoint_metadata_must_be_json_serializable(tmp_path):
    with pytest.raises(TypeError, match="JSON-serializable"):
        _propagator(final_time=0.1).propagate(
            np.ones(2),
            _hamiltonian,
            np.vdot,
            checkpoint_path=tmp_path / "state.chkpt",
            dynamics_metadata={"invalid": object()},
        )
