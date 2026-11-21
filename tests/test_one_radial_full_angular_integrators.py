import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.real_time import RealTimePropagator
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    SphericalOneRadialAbsorbingBoundaries,
    SphericalOneRadialFullAngularDynamicsObservables,
    SphericalOneRadialFullAngularTimeDependentHamiltonian,
    diagonalize_full_angular,
)


def _system():
    angle = np.pi / 4.0
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    positions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    positions = positions @ rotation.T
    return OneElectronDiatomicSystem.from_nuclear_positions(
        1.0,
        1.0,
        positions[0],
        positions[1],
        name="tilted H2+",
    )


@pytest.fixture(scope="module")
def development_problem():
    hamiltonian = OneElectronDiatomicHamiltonian(
        _system(),
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[5, 5, 5],
        l_max=2,
        m_max=2,
    )
    states = diagonalize_full_angular(
        hamiltonian,
        n_states=1,
        solver="direct",
    )
    return hamiltonian, states


def _pulse(time):
    return 0.04 * np.sin(np.pi * time / 0.08) ** 2 * np.sin(2.0 * time)


def _propagator(integrator, time_step, *, final_time=0.08):
    return RealTimePropagator(
        final_time=final_time,
        time_step=time_step,
        integrator=integrator,
        linear_solver="gmres",
        linear_rtol=1.0e-12,
        linear_atol=1.0e-14,
        max_krylov_dimension=30,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        max_krylov_refinements=6,
        progress_interval=None,
    )


def _run(hamiltonian, state, integrator, time_step, *, fields=()):
    dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
        hamiltonian,
        fields=fields,
    )
    observables = SphericalOneRadialFullAngularDynamicsObservables(
        dynamics,
        state,
        radial_probability_interval=None,
    )
    return dynamics.propagate(
        _propagator(integrator, time_step),
        state,
        samplers=observables.samplers,
    )


def _distance(hamiltonian, left, right):
    difference = left - right
    return np.sqrt(hamiltonian.inner_product(difference, difference).real)


def _vector_error(result, reference):
    differences = []
    for name in (
        "electronic_dipole_numerator",
        "canonical_momentum_numerator",
    ):
        differences.append(result.samples[name][-1] - reference.samples[name][-1])
    return np.linalg.norm(np.concatenate(differences))


@pytest.mark.parametrize("integrator", ["crank-nicolson", "sil", "sia"])
def test_field_free_integrators_preserve_norm_energy_and_phase(
    development_problem,
    integrator,
):
    hamiltonian, states = development_problem
    result = _run(
        hamiltonian,
        states.orbital,
        integrator,
        0.01,
    )
    if integrator == "crank-nicolson":
        energy = states.electronic_energy
        step_phase = (1.0 - 0.5j * 0.01 * energy) / (
            1.0 + 0.5j * 0.01 * energy
        )
        phase = step_phase**result.n_steps
    else:
        phase = np.exp(-1j * states.electronic_energy * result.final_time)
    expected = phase * states.orbital

    assert _distance(hamiltonian, result.state, expected) < 2.0e-9
    np.testing.assert_allclose(
        result.norm_history,
        result.initial_norm,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        result.samples["field_free_energy_numerator"],
        states.electronic_energy,
        atol=4.0e-11,
    )


def test_pulse_integrators_converge_to_common_state_and_vectors(
    development_problem,
):
    hamiltonian, states = development_problem
    reference = _run(
        hamiltonian,
        states.orbital,
        "sia",
        0.001,
        fields=_pulse,
    )

    for integrator in ("crank-nicolson", "sil", "sia"):
        coarse = _run(
            hamiltonian,
            states.orbital,
            integrator,
            0.02,
            fields=_pulse,
        )
        fine = _run(
            hamiltonian,
            states.orbital,
            integrator,
            0.01,
            fields=_pulse,
        )
        coarse_state_error = _distance(
            hamiltonian,
            coarse.state,
            reference.state,
        )
        fine_state_error = _distance(
            hamiltonian,
            fine.state,
            reference.state,
        )
        coarse_vector_error = _vector_error(coarse, reference)
        fine_vector_error = _vector_error(fine, reference)

        assert fine_state_error < coarse_state_error
        assert fine_state_error < 3.0e-6
        assert fine_vector_error < coarse_vector_error
        assert fine_vector_error < 2.0e-7
        assert fine.norm == pytest.approx(1.0, abs=4.0e-12)


def test_full_angular_mask_and_cap_contract(development_problem):
    hamiltonian, _ = development_problem
    state = np.zeros(hamiltonian.orbital_shape, dtype=complex)
    state[-3:, 0] = 1.0
    state /= np.sqrt(hamiltonian.inner_product(state, state).real)

    mask_dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
        hamiltonian,
        absorbing_boundaries=SphericalOneRadialAbsorbingBoundaries(
            hamiltonian,
            mask=RadialCosineMask(r0=4.5, n=8),
        ),
    )
    masked = {
        integrator: mask_dynamics.propagate(
            _propagator(integrator, 0.005, final_time=0.01),
            state,
        )
        for integrator in ("crank-nicolson", "sil", "sia")
    }
    for result in masked.values():
        assert result.norm < result.initial_norm

    cap_dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
        hamiltonian,
        absorbing_boundaries=SphericalOneRadialAbsorbingBoundaries(
            hamiltonian,
            cap=CubicCAP(alpha=0.5, eta=0.25),
        ),
    )
    cap_result = cap_dynamics.propagate(
        _propagator("sia", 0.005, final_time=0.02),
        state,
    )
    assert np.all(np.diff(cap_result.norm_history) <= 1.0e-13)
    with pytest.raises(ValueError, match="requires integrator"):
        cap_dynamics.propagate(
            _propagator("sil", 0.005, final_time=0.01),
            state,
        )
