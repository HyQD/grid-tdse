import json

import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.real_time import RealTimePropagator
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    SphericalOneRadialAbsorbingBoundaries,
    SphericalOneRadialFixedMTimeDependentHamiltonian,
    diagonalize_fixed_m,
)


@pytest.fixture(scope="module")
def stationary_problem():
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
    eigenstates = diagonalize_fixed_m(
        hamiltonian, m=0, n_states=2
    )
    return hamiltonian, eigenstates


def _propagator(integrator, final_time=0.2, time_step=0.05):
    return RealTimePropagator(
        final_time=final_time,
        time_step=time_step,
        integrator=integrator,
        linear_rtol=1.0e-12,
        linear_atol=1.0e-14,
        max_krylov_dimension=30,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        max_krylov_refinements=6,
        progress_interval=None,
    )


def test_complete_action_and_physical_field_free_action(stationary_problem):
    hamiltonian, eigenstates = stationary_problem
    cap = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, cap=CubicCAP(alpha=0.5, eta=0.25)
    )
    action = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=lambda time: 0.2 * time,
        absorbing_boundaries=cap,
    )
    state = eigenstates.orbital
    time = 0.3
    field_free = hamiltonian.apply_fixed_m(state, 0)
    expected = (
        field_free
        + action.dipole_interaction(state, time, 0)
        + cap.cap_action(state, 0)
    )

    np.testing.assert_allclose(action(state, time), expected)
    np.testing.assert_allclose(action.field_free_action(state), field_free)
    assert action.is_field_free is False


def test_batched_inner_products_match_individual_overlaps(
    stationary_problem,
):
    hamiltonian, eigenstates = stationary_problem
    action = SphericalOneRadialFixedMTimeDependentHamiltonian(hamiltonian)
    right = eigenstates.orbitals[0]

    expected = np.asarray(
        [
            action.inner_product(left, right)
            for left in eigenstates.orbitals
        ]
    )

    np.testing.assert_allclose(
        action.inner_products(eigenstates.orbitals, right), expected
    )


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_field_free_eigenstate_retains_phase_and_norm(
    stationary_problem, integrator
):
    hamiltonian, eigenstates = stationary_problem
    action = SphericalOneRadialFixedMTimeDependentHamiltonian(hamiltonian)
    final_time = 0.2

    result = action.propagate(
        _propagator(integrator, final_time), eigenstates.orbital
    )
    expected = (
        np.exp(-1j * eigenstates.electronic_energy * final_time)
        * eigenstates.orbital
    )

    np.testing.assert_allclose(result.state, expected, atol=3.0e-11)
    assert result.norm == pytest.approx(result.initial_norm, abs=3.0e-12)


def test_crank_nicolson_convenience_path_does_not_pass_batched_overlaps(
    stationary_problem,
):
    hamiltonian, eigenstates = stationary_problem
    action = SphericalOneRadialFixedMTimeDependentHamiltonian(hamiltonian)

    propagator = _propagator(
        "crank-nicolson", final_time=0.01, time_step=0.01
    )
    result = action.propagate(
        propagator,
        eigenstates.orbital,
    )

    assert result.norm == pytest.approx(result.initial_norm, abs=2.0e-12)
    energy = eigenstates.electronic_energy
    cayley_phase = (1.0 - 0.5j * propagator.time_step * energy) / (
        1.0 + 0.5j * propagator.time_step * energy
    )
    np.testing.assert_allclose(
        result.state,
        cayley_phase * eigenstates.orbital,
        atol=2.0e-12,
    )


def test_mask_is_installed_as_post_step_transformation(stationary_problem):
    hamiltonian, _ = stationary_problem
    state = np.zeros(hamiltonian.fixed_m_shape(0), dtype=complex)
    state[-2:, :] = 1.0
    norm = np.sqrt(hamiltonian.inner_product(state, state).real)
    state /= norm
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, mask=RadialCosineMask(r0=3.0)
    )
    action = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=boundaries
    )

    result = action.propagate(
        _propagator("sia", final_time=0.01, time_step=0.01), state
    )

    assert result.norm < result.initial_norm
    with pytest.raises(ValueError, match="post_step cannot"):
        action.propagate(
            _propagator("sia", final_time=0.01, time_step=0.01),
            state,
            post_step=lambda values, time: values,
        )


@pytest.mark.parametrize(
    "integrator", ["sil", "cf4-sil", "crank-nicolson"]
)
def test_invalid_cap_integrators_fail_before_propagation(
    stationary_problem, integrator
):
    hamiltonian, eigenstates = stationary_problem
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, cap=CubicCAP(alpha=0.5)
    )
    action = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=boundaries
    )

    with pytest.raises(ValueError, match="requires integrator"):
        action.propagate(
            _propagator(integrator, final_time=0.01, time_step=0.01),
            eigenstates.orbital,
        )


def test_configuration_validation_and_archive_metadata(stationary_problem):
    hamiltonian, _ = stationary_problem
    other = OneElectronDiatomicHamiltonian(
        hamiltonian.system,
        nodes=hamiltonian.nodes,
        n_points=hamiltonian.n_points,
        l_max=hamiltonian.l_max,
        m_max=hamiltonian.m_max,
    )
    foreign = SphericalOneRadialAbsorbingBoundaries(
        other, mask=RadialCosineMask(r0=3.0)
    )
    with pytest.raises(ValueError, match="different Hamiltonian"):
        SphericalOneRadialFixedMTimeDependentHamiltonian(
            hamiltonian, absorbing_boundaries=foreign
        )
    with pytest.raises(ValueError, match="requires at least one field"):
        SphericalOneRadialFixedMTimeDependentHamiltonian(
            hamiltonian, fields=(), include_nuclear_scalar=True
        )
    with pytest.raises(ValueError, match="length-gauge phase option"):
        SphericalOneRadialFixedMTimeDependentHamiltonian(
            hamiltonian,
            fields=lambda time: 0.0,
            field_gauge="velocity",
            include_nuclear_scalar=True,
        )

    action = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian, fields=[]
    )
    metadata = action.archive_metadata()
    assert action.is_field_free is True
    assert metadata["coordinate_system"] == "spherical_one_radial"
    assert metadata["m"] == 0
    assert metadata["time_dependent_interaction"] is None
    assert metadata["absorbing_boundaries"] is None
    json.dumps(metadata)
