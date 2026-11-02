import os
from pathlib import Path

import numpy as np
import pytest

from grid_tdse.real_time import load_real_time_checkpoint, load_real_time_snapshot
from grid_tdse.spherical.two_radial import (
    DiatomicIonDynamicsSamplingPolicy,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    phase_aligned_two_radial_state_difference,
)


GRID = FEMDVRGridSpec([0.0, 1.0, 2.0], [3, 3])
GROUND_STATE_OPTIONS = {
    "max_iterations": 100,
    "maximum_subspace_dimension": 12,
    "residual_tolerance": 1.0e-8,
    "progress_interval": None,
    "sample_interval": None,
}


class _SineSquaredPulse:
    gauge = "length"

    def __init__(self, amplitude=0.02, duration=0.04):
        self.amplitude = amplitude
        self.duration = duration
        self.evaluated_times = []

    def __call__(self, time):
        self.evaluated_times.append(float(time))
        if time < 0.0 or time > self.duration:
            return 0.0
        return self.amplitude * np.sin(np.pi * time / self.duration) ** 2


def _calculation(
    *,
    archive_path: Path | None,
    initial_state=None,
    fields=(),
    integrator="sil",
    time_step=0.01,
    initial_time=0.0,
    final_time=0.04,
    checkpoint_path=None,
    checkpoint_interval=None,
    checkpoint_overwrite=False,
    wavefunction_snapshot_directory=None,
    wavefunction_snapshot_prefix="wavefunction",
    mask_r=None,
    mask_R=None,
    cap_r=None,
    cap_R=None,
):
    return NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=initial_state,
        pair_exchange_parity=1,
        initial_state_policy=(
            "compute-if-missing" if initial_state is None else "require"
        ),
        ground_state_path=archive_path,
        ground_state_options=(
            GROUND_STATE_OPTIONS if initial_state is None else None
        ),
        fields=fields,
        field_gauge="length",
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        integrator=integrator,
        integrator_options={
            "max_krylov_dimension": 20,
            "rtol": 1.0e-11,
            "atol": 1.0e-13,
            "max_refinements": 8,
        },
        initial_time=initial_time,
        final_time=final_time,
        time_step=time_step,
        mask_r=mask_r,
        mask_R=mask_R,
        cap_r=cap_r,
        cap_R=cap_R,
        observables=None,
        dynamics_sampling_policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        checkpoint_overwrite=checkpoint_overwrite,
        checkpoint_metadata={"test": "phase9e"},
        wavefunction_snapshot_directory=wavefunction_snapshot_directory,
        wavefunction_snapshot_interval=(
            2 if wavefunction_snapshot_directory is not None else None
        ),
        wavefunction_snapshot_prefix=wavefunction_snapshot_prefix,
        wavefunction_snapshot_metadata={"code_provenance": {"source": "test"}},
        progress_interval=None,
        radial_operator_mode="element-local",
    )


def _state_difference(reference, candidate, calculation):
    return phase_aligned_two_radial_state_difference(
        reference,
        candidate,
        calculation.representation.inner_product,
    )


def _assert_radial_density_semantics(calculation):
    system = calculation.representation
    for sample in calculation.dynamics_time_series.samples:
        P_r = np.asarray(sample.snapshot.radial_densities.P_r)
        P_R = np.asarray(sample.snapshot.radial_densities.P_R)
        norm = sample.snapshot.norm
        assert np.dot(system.grid1.weights, P_r) == pytest.approx(
            norm,
            abs=2.0e-12,
        )
        assert np.dot(system.grid2.weights, P_R) == pytest.approx(
            norm,
            abs=2.0e-12,
        )
        assert np.dot(system.grid1.weights * system.grid1.r, P_r) / norm == (
            pytest.approx(sample.snapshot.conditional.r, abs=2.0e-12)
        )
        assert np.dot(system.grid2.weights * system.grid2.r, P_R) / norm == (
            pytest.approx(sample.snapshot.conditional.R, abs=2.0e-12)
        )


def test_phase9c_serial_field_free_and_length_gauge_validation(tmp_path):
    field_free = _calculation(archive_path=tmp_path / "ground.npz")
    archive = field_free.initialization.archive
    field_free_result = field_free.run(progress=False)

    assert field_free.initialization.source == "computed"
    assert max(abs(field_free_result.norm_history - 1.0)) < 2.0e-12
    assert (
        _state_difference(
            field_free.initial_state,
            field_free_result.state,
            field_free,
        )
        < 1.0e-10
    )
    assert max(
        abs(sample.conditional_energy_gain)
        for sample in field_free.dynamics_time_series.samples
    ) < 1.0e-10

    pulse_results = {}
    for name, integrator, time_step in (
        ("sil_coarse", "sil", 0.02),
        ("sil_fine", "sil", 0.01),
        ("sil_reference", "sil", 0.005),
        ("sia_fine", "sia", 0.01),
    ):
        pulse = _SineSquaredPulse()
        calculation = _calculation(
            archive_path=None,
            initial_state=archive,
            fields=pulse,
            integrator=integrator,
            time_step=time_step,
        )
        result = calculation.run(progress=False)
        pulse_results[name] = (calculation, result)
        midpoints = (
            calculation.propagator.propagation_times[:-1]
            + 0.5 * calculation.propagator.time_step
        )
        for midpoint in midpoints:
            assert any(
                np.isclose(midpoint, evaluated, rtol=0.0, atol=1.0e-15)
                for evaluated in pulse.evaluated_times
            )
        assert max(abs(result.norm_history - 1.0)) < 2.0e-12
        assert calculation.dynamics_time_series.samples[-1].is_final

    reference_calculation, reference_result = pulse_results["sil_reference"]
    coarse_error = _state_difference(
        reference_result.state,
        pulse_results["sil_coarse"][1].state,
        reference_calculation,
    )
    fine_error = _state_difference(
        reference_result.state,
        pulse_results["sil_fine"][1].state,
        reference_calculation,
    )
    sil_sia_difference = _state_difference(
        pulse_results["sil_fine"][1].state,
        pulse_results["sia_fine"][1].state,
        reference_calculation,
    )

    assert fine_error < 0.2 * coarse_error
    assert fine_error < 1.0e-8
    assert sil_sia_difference < 1.0e-11

    sil_series = pulse_results["sil_fine"][0].dynamics_time_series
    sia_series = pulse_results["sia_fine"][0].dynamics_time_series
    np.testing.assert_allclose(sil_series.times, sia_series.times)
    np.testing.assert_allclose(
        [sample.snapshot.conditional_energy for sample in sil_series.samples],
        [sample.snapshot.conditional_energy for sample in sia_series.samples],
        atol=1.0e-11,
        rtol=1.0e-11,
    )


def test_phase9e_serial_snapshots_checkpoint_and_continuation(tmp_path):
    seed = _calculation(archive_path=tmp_path / "ground.npz", final_time=0.01)
    archive = seed.initialization.archive
    pulse = _SineSquaredPulse(duration=0.04)

    uninterrupted = _calculation(
        archive_path=None,
        initial_state=archive,
        fields=pulse,
    )
    uninterrupted_result = uninterrupted.run(progress=False)

    exact_snapshots = tmp_path / "exact-snapshots"
    exact_snapshots.mkdir()
    exact_source = _calculation(
        archive_path=None,
        initial_state=archive,
        fields=_SineSquaredPulse(duration=0.04),
        checkpoint_path=tmp_path / "exact-source.chkpt",
        checkpoint_interval=2,
        wavefunction_snapshot_directory=exact_snapshots,
        wavefunction_snapshot_prefix="exact",
    )
    exact_source.run(progress=False)
    interrupted = load_real_time_checkpoint(exact_source.wavefunction_snapshots[1].path)
    assert interrupted.current_step == 2
    assert not interrupted.latest_segment.complete

    resumed = _calculation(
        archive_path=None,
        initial_state=interrupted.state_copy(),
        fields=_SineSquaredPulse(duration=0.04),
        initial_time=0.02,
        checkpoint_path=tmp_path / "exact-resumed.chkpt",
        checkpoint_interval=2,
    )
    resumed_result = resumed.resume(interrupted, progress=False)
    assert (
        _state_difference(
            uninterrupted_result.state,
            resumed_result.state,
            uninterrupted,
        )
        < 1.0e-12
    )
    assert [
        sample.accepted_step for sample in resumed.dynamics_time_series.samples
    ] == [2, 3, 4]

    checkpoint_path = tmp_path / "restart.chkpt"
    first_snapshots = tmp_path / "first-snapshots"
    first_snapshots.mkdir()
    first = _calculation(
        archive_path=None,
        initial_state=archive,
        fields=_SineSquaredPulse(duration=0.04),
        final_time=0.02,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=2,
        wavefunction_snapshot_directory=first_snapshots,
        wavefunction_snapshot_prefix="first",
    )
    first_result = first.run(progress=False)
    checkpoint = load_real_time_checkpoint(checkpoint_path)

    assert checkpoint.current_time == 0.02
    assert checkpoint.latest_segment.complete
    assert checkpoint.latest_segment.integrator["method"] == ("short_iterative_lanczos")
    assert checkpoint.latest_segment.dynamics["field_gauge"] == "length"
    assert checkpoint.metadata["test"] == "phase9e"
    assert set(
        checkpoint.metadata["dynamics_sampling_context"]["reference_values"]
    ) == {"norm", "energy_numerator", "conditional_energy"}
    assert [record.accepted_step for record in first.wavefunction_snapshots] == [
        0,
        2,
    ]
    assert first.wavefunction_snapshots[-1].storage == "checkpoint"
    assert first.wavefunction_snapshot_manifest.is_file()
    assert (
        os.stat(first.wavefunction_snapshots[-1].path).st_ino
        == os.stat(checkpoint_path).st_ino
    )
    initial_snapshot = load_real_time_snapshot(first.wavefunction_snapshots[0].path)
    assert (
        initial_snapshot.compatibility["physical_state"]["parameters"]["codata_set"]
        == "2022"
    )
    assert initial_snapshot.dynamics["field_gauge"] == "length"
    assert initial_snapshot.integrator["method"] == "short_iterative_lanczos"
    assert initial_snapshot.metadata == {"code_provenance": {"source": "test"}}

    second_snapshots = tmp_path / "second-snapshots"
    second_snapshots.mkdir()
    second = _calculation(
        archive_path=None,
        initial_state=checkpoint.state_copy(),
        fields=_SineSquaredPulse(duration=0.04),
        initial_time=0.02,
        final_time=0.04,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=2,
        checkpoint_overwrite=True,
        wavefunction_snapshot_directory=second_snapshots,
        wavefunction_snapshot_prefix="second",
    )
    second_result = second.continue_from(checkpoint, progress=False)

    assert [record.accepted_step for record in second.wavefunction_snapshots] == [
        2,
        4,
    ]

    assert (
        _state_difference(
            uninterrupted_result.state,
            second_result.state,
            uninterrupted,
        )
        < 1.0e-12
    )
    combined_samples = (
        first.dynamics_time_series.samples + second.dynamics_time_series.samples[1:]
    )
    reference_records = uninterrupted.dynamics_time_series.records()
    combined_records = type(uninterrupted.dynamics_time_series)(
        combined_samples
    ).records()
    assert len(combined_records) == len(reference_records)
    for combined, reference in zip(
        combined_records,
        reference_records,
        strict=True,
    ):
        for name in combined:
            if name.startswith("propagation.") or isinstance(
                combined[name], (bool, np.bool_)
            ):
                continue
            assert combined[name] == pytest.approx(reference[name], abs=2.0e-12)

    chained = load_real_time_checkpoint(checkpoint_path)
    assert len(chained.segments) == 2
    assert chained.total_steps == 4
    assert chained.current_time == 0.04
    np.testing.assert_allclose(chained.state, second_result.state)

    changed_field = _SineSquaredPulse(amplitude=0.03, duration=0.05)
    third = _calculation(
        archive_path=None,
        initial_state=chained.state_copy(),
        fields=changed_field,
        initial_time=0.04,
        final_time=0.05,
        checkpoint_path=checkpoint_path,
        checkpoint_overwrite=True,
    )
    third.continue_from(chained, progress=False)
    extended = load_real_time_checkpoint(checkpoint_path)
    assert len(extended.segments) == 3
    assert extended.segments[-1].dynamics != extended.segments[-2].dynamics


def test_phase9f_masks_caps_and_absorbed_restart(tmp_path):
    seed = _calculation(archive_path=tmp_path / "ground.npz", final_time=0.01)
    archive = seed.initialization.archive
    n_r = seed.representation.grid1.n_points
    n_R = seed.representation.grid2.n_points
    mask_r = np.linspace(0.99, 0.82, n_r)
    mask_R = np.linspace(0.98, 0.86, n_R)

    mask_run = _calculation(
        archive_path=None,
        initial_state=archive,
        final_time=0.04,
        mask_r=mask_r,
        mask_R=mask_R,
    )
    mask_result = mask_run.run(progress=False)
    assert np.all(np.diff(mask_result.norm_history) < 0.0)
    assert mask_result.norm < mask_result.initial_norm
    _assert_radial_density_semantics(mask_run)

    cap_r = -0.04j * np.ones(n_r)
    cap_R = -0.03j * np.ones(n_R)
    cap_run = _calculation(
        archive_path=None,
        initial_state=archive,
        integrator="sia",
        final_time=0.04,
        cap_r=cap_r,
        cap_R=cap_R,
    )
    cap_result = cap_run.run(progress=False)
    assert np.all(np.diff(cap_result.norm_history) < 0.0)
    assert cap_result.norm < cap_result.initial_norm
    final_sample = cap_run.dynamics_time_series.samples[-1]
    physical_energy = cap_run.representation.inner_product(
        cap_result.state,
        cap_run.field_free_hamiltonian(cap_result.state),
    ).real
    assert final_sample.snapshot.energy_numerator == pytest.approx(
        physical_energy,
        abs=2.0e-12,
    )
    _assert_radial_density_semantics(cap_run)

    snapshot_directory = tmp_path / "absorbed-snapshots"
    snapshot_directory.mkdir()
    checkpoint_source = _calculation(
        archive_path=None,
        initial_state=archive,
        final_time=0.04,
        mask_r=mask_r,
        mask_R=mask_R,
        checkpoint_path=tmp_path / "absorbed-source.chkpt",
        checkpoint_interval=2,
        wavefunction_snapshot_directory=snapshot_directory,
        wavefunction_snapshot_prefix="absorbed",
    )
    checkpoint_source_result = checkpoint_source.run(progress=False)
    interrupted = load_real_time_checkpoint(
        checkpoint_source.wavefunction_snapshots[1].path
    )
    assert interrupted.current_step == 2

    resumed = _calculation(
        archive_path=None,
        initial_state=interrupted.state_copy(),
        initial_time=0.02,
        final_time=0.04,
        mask_r=mask_r,
        mask_R=mask_R,
        checkpoint_path=tmp_path / "absorbed-resumed.chkpt",
        checkpoint_interval=2,
    )
    resumed_result = resumed.resume(interrupted, progress=False)
    assert (
        _state_difference(
            checkpoint_source_result.state,
            resumed_result.state,
            checkpoint_source,
        )
        < 2.0e-13
    )
    reference_samples = checkpoint_source.dynamics_time_series.samples[2:]
    for actual, reference in zip(
        resumed.dynamics_time_series.samples,
        reference_samples,
        strict=True,
    ):
        assert actual.accepted_step == reference.accepted_step
        assert actual.snapshot.norm == pytest.approx(
            reference.snapshot.norm,
            abs=2.0e-13,
        )
        np.testing.assert_allclose(
            actual.snapshot.radial_densities.P_r,
            reference.snapshot.radial_densities.P_r,
            atol=2.0e-13,
            rtol=2.0e-13,
        )
        np.testing.assert_allclose(
            actual.snapshot.radial_densities.P_R,
            reference.snapshot.radial_densities.P_R,
            atol=2.0e-13,
            rtol=2.0e-13,
        )
