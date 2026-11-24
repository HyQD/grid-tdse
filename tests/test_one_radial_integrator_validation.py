import json

import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.real_time import (
    RealTimePropagator,
    load_real_time_checkpoint,
)
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    SphericalOneRadialAbsorbingBoundaries,
    SphericalOneRadialFixedMTimeDependentHamiltonian,
    SphericalOneRadialKineticCrankNicolsonPreconditioner,
    diagonalize_fixed_m,
)


@pytest.fixture(scope="module")
def development_problem():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[6, 6, 6],
        l_max=3,
        m_max=1,
    )
    eigenstates = diagonalize_fixed_m(hamiltonian, m=0, n_states=2)
    return hamiltonian, eigenstates


def _pulse(time):
    final_time = 0.2
    return (
        0.02
        * np.sin(np.pi * time / final_time) ** 2
        * np.sin(3.0 * time)
    )


def _propagator(
    integrator,
    time_step,
    *,
    linear_solver="gmres",
    linear_tolerance=1.0e-12,
):
    return RealTimePropagator(
        final_time=0.2,
        time_step=time_step,
        integrator=integrator,
        linear_solver=linear_solver,
        linear_rtol=linear_tolerance,
        linear_atol=1.0e-14,
        linear_max_iterations=500,
        max_krylov_dimension=30,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        max_krylov_refinements=6,
        progress_interval=None,
    )


def _state_distance(hamiltonian, left, right):
    difference = left - right
    return np.sqrt(hamiltonian.inner_product(difference, difference).real)


def _z_numerator(hamiltonian, state):
    return hamiltonian.inner_product(
        state, hamiltonian.apply_z_fixed_m(state, 0)
    ).real


def test_short_pulse_integrators_converge_to_fine_step_oracle(
    development_problem,
):
    hamiltonian, eigenstates = development_problem
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian, fields=_pulse
    )
    sampler = {
        "z_numerator": lambda state, time: _z_numerator(
            hamiltonian, state
        )
    }
    reference = dynamics.propagate(
        _propagator("sia", 0.001),
        eigenstates.orbital,
        samplers=sampler,
    )

    for integrator in ("crank-nicolson", "sil", "sia"):
        coarse = dynamics.propagate(
            _propagator(integrator, 0.02), eigenstates.orbital
        )
        fine = dynamics.propagate(
            _propagator(integrator, 0.01),
            eigenstates.orbital,
            samplers=sampler,
        )
        coarse_error = _state_distance(
            hamiltonian, coarse.state, reference.state
        )
        fine_error = _state_distance(
            hamiltonian, fine.state, reference.state
        )

        assert fine_error < coarse_error
        assert fine_error < 3.0e-6
        assert fine.norm == pytest.approx(1.0, abs=3.0e-12)
        np.testing.assert_allclose(
            fine.samples["z_numerator"],
            reference.samples["z_numerator"][::10],
            atol=3.0e-8,
            rtol=0.0,
        )


def test_all_integrators_agree_with_radial_mask(development_problem):
    hamiltonian, _ = development_problem
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian,
        mask=RadialCosineMask(r0=4.5, n=8),
    )
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=_pulse,
        absorbing_boundaries=boundaries,
    )
    initial = np.zeros(hamiltonian.fixed_m_shape(0), dtype=complex)
    initial[-3:, 0] = 1.0
    initial /= np.sqrt(hamiltonian.inner_product(initial, initial).real)

    results = {
        integrator: dynamics.propagate(
            _propagator(integrator, 0.005), initial
        )
        for integrator in ("crank-nicolson", "sil", "sia")
    }
    reference = results["sia"]

    assert results["sil"].norm == pytest.approx(reference.norm, abs=1.0e-12)
    assert results["crank-nicolson"].norm == pytest.approx(
        reference.norm, abs=1.0e-6
    )
    for integrator in ("crank-nicolson", "sil"):
        assert (
            _state_distance(
                hamiltonian, results[integrator].state, reference.state
            )
            < 3.0e-4
        )


def test_cn_and_sia_cap_loss_are_monotonic_and_sil_is_rejected(
    development_problem,
):
    hamiltonian, _ = development_problem
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian,
        cap=CubicCAP(alpha=0.5, eta=0.25),
    )
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=boundaries
    )
    initial = np.zeros(hamiltonian.fixed_m_shape(0), dtype=complex)
    initial[-3:, 0] = 1.0
    initial /= np.sqrt(hamiltonian.inner_product(initial, initial).real)

    results = {
        integrator: dynamics.propagate(
            _propagator(integrator, 0.005), initial
        )
        for integrator in ("crank-nicolson", "sia")
    }

    for result in results.values():
        assert result.norm < result.initial_norm
        assert np.all(np.diff(result.norm_history) <= 1.0e-13)
    assert (
        _state_distance(
            hamiltonian,
            results["crank-nicolson"].state,
            results["sia"].state,
        )
        < 3.0e-4
    )
    with pytest.raises(ValueError, match="requires integrator"):
        dynamics.propagate(_propagator("sil", 0.005), initial)


@pytest.mark.parametrize("linear_solver", ["gmres", "bicgstab"])
def test_kinetic_preconditioner_reduces_cn_iterations_with_mask(
    development_problem,
    linear_solver,
    tmp_path,
):
    hamiltonian, eigenstates = development_problem
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian,
        mask=RadialCosineMask(r0=4.5, n=8),
    )
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=_pulse,
        absorbing_boundaries=boundaries,
    )
    propagator = _propagator(
        "crank-nicolson",
        0.04,
        linear_solver=linear_solver,
        linear_tolerance=1.0e-10,
    )
    baseline = dynamics.propagate(propagator, eigenstates.orbital)
    preconditioner = SphericalOneRadialKineticCrankNicolsonPreconditioner(
        hamiltonian,
        m=0,
        time_step=propagator.time_step,
    )
    prepared = dynamics.propagate(
        propagator,
        eigenstates.orbital,
        preconditioner=preconditioner,
        checkpoint_path=tmp_path / f"{linear_solver}.chkpt",
    )

    assert np.all(prepared.linear_iterations > 0)
    assert np.sum(prepared.linear_iterations) < np.sum(
        baseline.linear_iterations
    )
    assert np.all(np.isfinite(prepared.linear_residual_norms))
    assert _state_distance(
        hamiltonian, baseline.state, prepared.state
    ) < 3.0e-9
    assert preconditioner.applications > 0
    assert preconditioner.metadata["particle_mass"] == pytest.approx(1.0)
    json.dumps(preconditioner.metadata)
    checkpoint = load_real_time_checkpoint(
        tmp_path / f"{linear_solver}.chkpt"
    )
    assert (
        checkpoint.latest_segment.dynamics["preconditioner"]
        == preconditioner.metadata
    )


def test_kinetic_preconditioner_supports_mass_and_state_batches():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        particle_mass=2.0,
    )
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=[0.0, 1.0, 3.0],
        n_points=[5, 5],
        l_max=2,
        m_max=0,
    )
    time_step = 0.03
    preconditioner = SphericalOneRadialKineticCrankNicolsonPreconditioner(
        hamiltonian,
        m=0,
        time_step=time_step,
    )
    rng = np.random.default_rng(2048)
    states = rng.normal(size=(2, *hamiltonian.fixed_m_shape(0)))
    states = states + 1j * rng.normal(size=states.shape)
    prepared = preconditioner(states)
    expected = np.empty_like(prepared)
    identity = np.eye(hamiltonian.r.size)

    for angular_index, l_value in enumerate(range(3)):
        kinetic = -0.5 / system.particle_mass * hamiltonian.second_derivative
        kinetic = kinetic + np.diag(
            l_value
            * (l_value + 1)
            / (2.0 * system.particle_mass * hamiltonian.r**2)
        )
        left = identity + 0.5j * time_step * kinetic
        for batch in range(states.shape[0]):
            expected[batch, :, angular_index] = np.linalg.solve(
                left, states[batch, :, angular_index]
            )

    np.testing.assert_allclose(prepared, expected)
    assert preconditioner.metadata["particle_mass"] == pytest.approx(2.0)


def test_preconditioner_rejects_incompatible_dynamics(development_problem):
    hamiltonian, eigenstates = development_problem
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(hamiltonian)
    propagator = _propagator("crank-nicolson", 0.04)
    wrong_step = SphericalOneRadialKineticCrankNicolsonPreconditioner(
        hamiltonian,
        m=0,
        time_step=0.02,
    )

    with pytest.raises(ValueError, match="effective propagator time step"):
        dynamics.propagate(
            propagator,
            eigenstates.orbital,
            preconditioner=wrong_step,
        )
