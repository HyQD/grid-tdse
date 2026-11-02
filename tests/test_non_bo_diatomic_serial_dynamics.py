from pathlib import Path

import numpy as np

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
        final_time=0.04,
        time_step=time_step,
        observables=None,
        dynamics_sampling_policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
        progress_interval=None,
        radial_operator_mode="element-local",
    )


def _state_difference(reference, candidate, calculation):
    return phase_aligned_two_radial_state_difference(
        reference,
        candidate,
        calculation.representation.inner_product,
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
