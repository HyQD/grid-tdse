import json

import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.real_time import load_real_time_checkpoint
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    OneElectronDiatomicTDSE,
    SphericalOneRadialKineticCrankNicolsonPreconditioner,
    diagonalize_fixed_m,
)


@pytest.fixture(scope="module")
def prepared_state():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        particle_mass=1.25,
        name="mass-scaled H2+",
    )
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[6, 6, 6],
        l_max=3,
        m_max=0,
    )
    state = diagonalize_fixed_m(hamiltonian, m=0).orbital
    return system, state


def _field(time):
    return 0.01 * np.sin(np.pi * time / 0.02) ** 2


def _options(integrator):
    if integrator == "crank-nicolson":
        return {"rtol": 1.0e-12, "atol": 1.0e-14}
    return {
        "max_krylov_dimension": 30,
        "rtol": 1.0e-12,
        "atol": 1.0e-14,
        "max_refinements": 6,
    }


def _calculation(system, state, integrator, **overrides):
    arguments = dict(
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[6, 6, 6],
        l_max=3,
        m=0,
        initial_state=state,
        fields=_field,
        field_gauge="length",
        integrator=integrator,
        integrator_options=_options(integrator),
        final_time=0.02,
        time_step=0.01,
        radial_probability_interval=1,
        progress_interval=None,
    )
    arguments.update(overrides)
    return OneElectronDiatomicTDSE(system, **arguments)


def _distance(hamiltonian, left, right):
    difference = left - right
    return np.sqrt(hamiltonian.inner_product(difference, difference).real)


def test_same_setup_runs_with_cn_sil_and_sia(prepared_state):
    system, state = prepared_state
    original = state.copy()
    calculations = {
        integrator: _calculation(system, state, integrator)
        for integrator in ("crank-nicolson", "sil", "sia")
    }
    results = {
        integrator: calculation.run(progress=False)
        for integrator, calculation in calculations.items()
    }
    reference = results["sia"]

    np.testing.assert_array_equal(state, original)
    for integrator, result in results.items():
        calculation = calculations[integrator]
        assert result.norm == pytest.approx(result.initial_norm, abs=3.0e-12)
        assert _distance(
            calculation.hamiltonian, result.state, reference.state
        ) < 3.0e-6
        assert "field_free_energy_numerator" in result.samples
        assert "induced_dipole_numerator" in result.samples
        assert "canonical_momentum_numerator" in result.samples
        radial = calculation.radial_probability_series
        np.testing.assert_array_equal(radial.accepted_steps, [0, 1, 2])
        np.testing.assert_allclose(
            radial.probabilities @ radial.radial_weights,
            radial.norms,
            atol=2.0e-12,
        )

    assert isinstance(
        calculations["crank-nicolson"].preconditioner,
        SphericalOneRadialKineticCrankNicolsonPreconditioner,
    )
    assert calculations["sil"].preconditioner is None
    assert calculations["sia"].preconditioner is None


def test_custom_sampler_progress_and_metadata(prepared_state):
    system, state = prepared_state
    calculation = _calculation(
        system,
        state,
        "sia",
        samplers={"state_sum": lambda values, time: np.sum(values)},
        progress_interval=1,
        progress_bar=True,
    )

    result = calculation.propagate(progress=False)

    assert "state_sum" in result.samples
    assert calculation.propagator.progress_interval == 1
    assert calculation.propagator.progress_bar is True
    assert calculation.compatibility_metadata["particle_mass"] == pytest.approx(
        1.25
    )
    assert calculation.dynamics_metadata["initial_state"]["source"] == "provided"
    assert calculation.dynamics_metadata["observables"][
        "normalization_policy"
    ] == "raw_unnormalized_numerators"
    calculation.run(progress=False)
    assert calculation.radial_probability_series.accepted_steps.size == 3
    json.dumps(calculation.compatibility_metadata)
    json.dumps(calculation.dynamics_metadata)


def test_velocity_gauge_samples_mechanical_momentum(prepared_state):
    system, state = prepared_state

    class VectorPotential:
        gauge = "velocity"

        def __call__(self, time):
            return 0.02 * np.sin(np.pi * time / 0.02) ** 2

    calculation = _calculation(
        system,
        state,
        "sia",
        fields=VectorPotential(),
        field_gauge="velocity",
    )

    result = calculation.run(progress=False)

    assert "mechanical_momentum_numerator" in result.samples
    difference = (
        result.samples["mechanical_momentum_numerator"]
        - result.samples["canonical_momentum_numerator"]
    )
    vector_potential = np.asarray(
        [calculation.fields[0](time) for time in result.times]
    )
    np.testing.assert_allclose(
        difference,
        vector_potential * result.norm_history,
        atol=2.0e-12,
    )


def test_checkpoint_and_compatible_continuation_match_uninterrupted_run(
    prepared_state,
    tmp_path,
):
    system, state = prepared_state
    first_path = tmp_path / "first.chkpt"
    second_path = tmp_path / "second.chkpt"
    common = dict(
        fields=(),
        integrator_options=_options("sia"),
        radial_probability_interval=1,
    )
    first = _calculation(
        system,
        state,
        "sia",
        **common,
        final_time=0.01,
        checkpoint_path=first_path,
        checkpoint_interval=1,
        checkpoint_metadata={"purpose": "interface-test"},
    )
    first_result = first.run(progress=False)
    stored = load_real_time_checkpoint(first_path)

    assert stored.current_time == pytest.approx(0.01)
    assert stored.metadata == {"purpose": "interface-test"}
    assert stored.compatibility["particle_mass"] == pytest.approx(1.25)

    continuation = _calculation(
        system,
        state,
        "sia",
        **common,
        initial_time=0.01,
        final_time=0.02,
        checkpoint_path=second_path,
    )
    continued_result = continuation.continue_from(
        first_path, progress=False
    )
    uninterrupted = _calculation(
        system,
        state,
        "sia",
        **common,
        final_time=0.02,
    )
    uninterrupted_result = uninterrupted.run(progress=False)

    assert first_result.final_time == pytest.approx(0.01)
    np.testing.assert_allclose(
        continued_result.state,
        uninterrupted_result.state,
        atol=3.0e-12,
    )
    assert second_path.is_file()


def test_exact_resume_matches_uninterrupted_run(prepared_state, tmp_path):
    system, state = prepared_state
    path = tmp_path / "interrupted.chkpt"
    interrupted = _calculation(
        system,
        state,
        "sia",
        fields=(),
        final_time=0.02,
        checkpoint_path=path,
        checkpoint_interval=1,
    )
    arguments = interrupted._propagation_arguments()

    def stop_after_first_checkpoint(
        values,
        accepted_step,
        time,
        diagnostics,
        final,
        checkpoint,
    ):
        del values, time, diagnostics, final
        if accepted_step == 1 and checkpoint:
            raise RuntimeError("simulated interruption")

    arguments["state_observer"] = stop_after_first_checkpoint
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.dynamics.propagate(
            interrupted.propagator,
            interrupted.initial_state,
            **arguments,
        )

    partial = load_real_time_checkpoint(path)
    assert partial.current_time == pytest.approx(0.01)
    assert partial.latest_segment.complete is False

    resumed = _calculation(
        system,
        state,
        "sia",
        fields=(),
        initial_time=0.01,
        final_time=0.02,
        checkpoint_path=path,
    )
    resumed_result = resumed.resume(path, progress=False)
    uninterrupted = _calculation(
        system,
        state,
        "sia",
        fields=(),
        final_time=0.02,
    )
    reference = uninterrupted.run(progress=False)

    np.testing.assert_allclose(
        resumed_result.state,
        reference.state,
        atol=3.0e-12,
    )
    assert resumed_result.n_steps == 2


def test_masked_cn_uses_public_preconditioner_and_persists_metadata(
    prepared_state,
    tmp_path,
):
    system, state = prepared_state
    path = tmp_path / "cn.chkpt"
    calculation = _calculation(
        system,
        state,
        "crank-nicolson",
        mask=RadialCosineMask(r0=4.5, n=8),
        checkpoint_path=path,
    )

    result = calculation.run(progress=False)
    checkpoint = load_real_time_checkpoint(path)

    assert np.all(result.linear_iterations > 0)
    metadata = checkpoint.latest_segment.dynamics["preconditioner"]
    assert metadata["name"] == "kinetic-channel-lu"
    assert metadata["particle_mass"] == pytest.approx(1.25)


def test_high_level_cn_supports_cap_and_preconditioner(
    prepared_state,
    tmp_path,
):
    system, state = prepared_state
    path = tmp_path / "cn-cap.chkpt"
    calculation = _calculation(
        system,
        state,
        "crank-nicolson",
        cap=CubicCAP(alpha=0.5, eta=0.25),
        checkpoint_path=path,
    )

    result = calculation.run(progress=False)
    checkpoint = load_real_time_checkpoint(path)

    assert result.norm <= result.initial_norm
    assert np.all(np.diff(result.norm_history) <= 1.0e-13)
    assert np.all(result.linear_iterations > 0)
    assert calculation.preconditioner.applications > 0
    dynamics = checkpoint.latest_segment.dynamics
    assert dynamics["preconditioner"]["name"] == "kinetic-channel-lu"
    assert dynamics["absorbing_boundaries"]["cap"]["eta"] == pytest.approx(
        0.25
    )


def test_high_level_interface_rejects_cap_for_sil(
    prepared_state,
):
    system, state = prepared_state

    with pytest.raises(ValueError, match="requires integrator"):
        _calculation(
            system,
            state,
            "sil",
            cap=CubicCAP(alpha=0.5, eta=0.25),
        )


def test_high_level_interface_rejects_invalid_state_and_options(
    prepared_state,
):
    system, state = prepared_state

    with pytest.raises(ValueError, match="positive finite norm"):
        _calculation(system, np.zeros_like(state), "sia")
    with pytest.raises(ValueError, match="unknown sia integrator options"):
        _calculation(
            system,
            state,
            "sia",
            integrator_options={"preconditioner": False},
        )
    with pytest.raises(ValueError, match="reserved"):
        _calculation(
            system,
            state,
            "sia",
            samplers={"norm": lambda values, time: 0.0},
        )


def test_velocity_gauge_rejects_nonzero_initial_vector_potential(
    prepared_state,
):
    system, state = prepared_state

    class OffsetVectorPotential:
        gauge = "velocity"

        def __call__(self, time):
            return 0.1 + time

    with pytest.raises(ValueError, match=r"A\(initial_time\)=0"):
        _calculation(
            system,
            state,
            "sia",
            fields=OffsetVectorPotential(),
            field_gauge="velocity",
        )
