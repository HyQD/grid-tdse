import json
import os

import numpy as np
import pytest

from grid_tdse.real_time import (
    RealTimePropagator,
    RealTimeSnapshotSelector,
    RealTimeSnapshotWriter,
    RealTimeStateSnapshot,
    load_real_time_checkpoint,
    load_real_time_snapshot,
    real_time_state_data_id,
    save_real_time_snapshot,
)


def test_lightweight_snapshot_round_trip_and_checksum(tmp_path):
    path = tmp_path / "state.npz"
    state = np.array([1.0 + 0.5j, -0.2j])
    snapshot = RealTimeStateSnapshot(
        state=state,
        accepted_step=3,
        time=0.3,
        norm=float(np.vdot(state, state).real),
        compatibility={"representation": "test"},
        dynamics={"field": "pulse"},
        integrator={"method": "short_iterative_lanczos"},
        metadata={"code_provenance": {"commit": "abc123"}},
        parent_state_data_id=real_time_state_data_id(np.ones(2)),
    )

    save_real_time_snapshot(path, snapshot)
    stored = load_real_time_snapshot(path)

    np.testing.assert_array_equal(stored.state, state)
    assert not stored.state.flags.writeable
    assert stored.accepted_step == 3
    assert stored.time == pytest.approx(0.3)
    assert stored.norm == pytest.approx(np.vdot(state, state).real)
    assert stored.compatibility == {"representation": "test"}
    assert stored.dynamics == {"field": "pulse"}
    assert stored.integrator == {"method": "short_iterative_lanczos"}
    assert stored.metadata == {"code_provenance": {"commit": "abc123"}}
    assert stored.source_path == path
    assert stored.state_data_id == real_time_state_data_id(state)

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["state"][0] += 1.0
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_real_time_snapshot(path)


def test_snapshot_writer_reuses_coincident_checkpoints(tmp_path):
    snapshot_directory = tmp_path / "snapshots"
    snapshot_directory.mkdir()
    checkpoint_path = tmp_path / "rolling.chkpt"
    initial = np.array([1.0, 0.2j])
    compatibility = {"representation": {"name": "two-level"}}
    dynamics = {"field": {"name": "pulse"}}
    propagator = RealTimePropagator(
        final_time=0.4,
        time_step=0.1,
        integrator="sil",
        progress_interval=None,
    )
    writer = RealTimeSnapshotWriter(
        snapshot_directory,
        interval=2,
        include_initial=True,
        include_final=True,
        prefix="psi",
        compatibility_metadata=compatibility,
        dynamics_metadata=dynamics,
        integrator_metadata=propagator.integrator_metadata,
        metadata={"purpose": "test"},
        parent_state_data_id=real_time_state_data_id(initial),
        inner_product=np.vdot,
        checkpoint_path=checkpoint_path,
    )

    result = propagator.propagate(
        initial,
        lambda state, time: np.array([0.4, 0.9]) * state,
        np.vdot,
        state_observer=writer,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=2,
        compatibility_metadata=compatibility,
        dynamics_metadata=dynamics,
    )
    manifest_path = writer.finish()

    assert [record.accepted_step for record in writer.records] == [0, 2, 4]
    assert [record.storage for record in writer.records] == [
        "snapshot",
        "checkpoint",
        "checkpoint",
    ]
    initial_snapshot = load_real_time_snapshot(writer.records[0].path)
    np.testing.assert_array_equal(initial_snapshot.state, initial)

    midpoint_checkpoint = load_real_time_checkpoint(writer.records[1].path)
    final_checkpoint = load_real_time_checkpoint(writer.records[2].path)
    assert midpoint_checkpoint.current_time == pytest.approx(0.2)
    assert final_checkpoint.current_time == pytest.approx(0.4)
    np.testing.assert_array_equal(final_checkpoint.state, result.state)
    assert os.stat(writer.records[2].path).st_ino == os.stat(checkpoint_path).st_ino

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [record["accepted_step"] for record in manifest["records"]] == [0, 2, 4]
    assert manifest["records"][1]["storage"] == "checkpoint"


def test_snapshot_writer_requires_directory_and_checkpoint(tmp_path):
    state_id = real_time_state_data_id(np.ones(2))
    with pytest.raises(FileNotFoundError, match="snapshot directory"):
        RealTimeSnapshotWriter(
            tmp_path / "missing",
            parent_state_data_id=state_id,
        )

    writer = RealTimeSnapshotWriter(
        tmp_path,
        interval=1,
        include_initial=False,
        parent_state_data_id=state_id,
        inner_product=np.vdot,
    )
    with pytest.raises(RuntimeError, match="checkpoint path"):
        writer(
            np.ones(2),
            1,
            0.1,
            None,
            False,
            True,
        )


def test_snapshot_writer_unions_interval_steps_times_and_warns_unmatched(tmp_path):
    initial = np.array([1.0, 0.2j])
    propagator = RealTimePropagator(
        final_time=0.4,
        time_step=0.1,
        integrator="sil",
        progress_interval=None,
    )
    writer = RealTimeSnapshotWriter(
        tmp_path,
        interval=4,
        steps=[1, 1],
        times=[0.2, 0.35],
        prefix="selected",
        parent_state_data_id=real_time_state_data_id(initial),
        inner_product=np.vdot,
    )

    propagator.propagate(
        initial,
        lambda state, time: np.array([0.4, 0.9]) * state,
        np.vdot,
        state_observer=writer,
    )
    with pytest.warns(RuntimeWarning, match=r"times=\[0\.35\]"):
        manifest_path = writer.finish()

    assert [record.accepted_step for record in writer.records] == [0, 1, 2, 4]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection"] == {
        "interval": 4,
        "steps": [1],
        "times": [0.2, 0.35],
        "include_initial": True,
        "include_final": True,
        "unmatched_steps": [],
        "unmatched_times": [0.35],
    }


def test_snapshot_selector_uses_global_steps_and_absolute_times():
    selector = RealTimeSnapshotSelector(
        steps=[6],
        times=[1.25],
        include_initial=False,
        include_final=False,
    )

    assert not selector.selected(5, 0, 1.0)
    assert selector.selected(6, 1, 1.25)
    assert selector.as_dict()["unmatched_steps"] == []
    assert selector.as_dict()["unmatched_times"] == []
