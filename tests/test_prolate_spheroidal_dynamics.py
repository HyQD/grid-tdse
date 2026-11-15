import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalFixedMTimeDependentHamiltonian,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
)
from grid_tdse.real_time import RealTimePropagator
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian as SphericalHamiltonian,
)
from grid_tdse.spherical.one_radial import (
    diagonalize_fixed_m as diagonalize_spherical_fixed_m,
)


@pytest.fixture(scope="module")
def stationary_problem():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 4.0],
        xi_points_per_element=6,
        eta_degree=7,
        m_max=0,
    )
    eigenstates = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=1,
        solver="direct",
    )
    return hamiltonian, eigenstates


def _propagator(integrator, final_time, time_step):
    return RealTimePropagator(
        final_time=final_time,
        time_step=time_step,
        integrator=integrator,
        max_krylov_dimension=30,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        max_krylov_refinements=6,
        progress_interval=None,
    )


def test_time_dependent_action_matches_field_free_plus_dipole(
    stationary_problem,
):
    hamiltonian, eigenstates = stationary_problem
    pulse = SineSquareLaser(
        E0=1.0e-3,
        omega=4.0,
        ncycles=1,
        gauge="length",
    )
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian, m=0, fields=pulse
    )
    state = eigenstates.orbital
    time = 0.4

    expected = hamiltonian.apply_fixed_m(state) + (
        action.dipole_interaction(state, time)
    )

    np.testing.assert_allclose(action(state, time), expected)
    assert action.is_field_free is False
    assert action.archive_metadata()["m"] == 0


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_field_free_eigenstate_retains_phase_and_norm(
    stationary_problem, integrator
):
    hamiltonian, eigenstates = stationary_problem
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(hamiltonian)
    final_time = 0.3

    result = _propagator(integrator, final_time, 0.05).propagate(
        eigenstates.orbital,
        action,
        action.inner_product,
    )

    expected = np.exp(
        -1j * eigenstates.electronic_energy * final_time
    ) * eigenstates.orbital
    np.testing.assert_allclose(result.state, expected, atol=2.0e-11)
    assert result.norm == pytest.approx(result.initial_norm, abs=2.0e-12)


def test_weak_short_pulse_sil_and_sia_agree_and_conserve_norm(
    stationary_problem,
):
    hamiltonian, eigenstates = stationary_problem
    pulse = SineSquareLaser(
        E0=1.0e-3,
        omega=4.0,
        ncycles=1,
        gauge="length",
    )
    duration = 2.0 * np.pi / 4.0
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian, fields=pulse
    )

    results = {
        integrator: _propagator(integrator, duration, 0.05).propagate(
            eigenstates.orbital,
            action,
            action.inner_product,
        )
        for integrator in ("sil", "sia")
    }

    for result in results.values():
        assert result.norm == pytest.approx(
            result.initial_norm, abs=3.0e-11
        )
    difference = results["sil"].state - results["sia"].state
    difference_norm = np.sqrt(
        action.inner_product(difference, difference).real
    )
    assert difference_norm < 2.0e-10


def test_field_free_configuration_and_validation(stationary_problem):
    hamiltonian, eigenstates = stationary_problem
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian, fields=[]
    )

    assert action.is_field_free is True
    np.testing.assert_allclose(
        action(eigenstates.orbital, 0.2),
        hamiltonian.apply_fixed_m(eigenstates.orbital),
    )
    assert action.archive_metadata()["time_dependent_interaction"] is None

    with pytest.raises(ValueError, match="requires at least one field"):
        ProlateSpheroidalFixedMTimeDependentHamiltonian(
            hamiltonian,
            fields=(),
            include_nuclear_scalar=True,
        )


def _spherical_fixed_m_z(hamiltonian, state, m=0):
    """Independent one-radial ``z = r cos(theta)`` action for validation."""

    values = hamiltonian._validated_fixed_m_orbitals(state, m)
    l_values = np.arange(abs(m), hamiltonian.l_max + 1)
    angular = np.zeros((l_values.size, l_values.size))
    for input_index, l_value in enumerate(l_values[:-1]):
        coefficient = np.sqrt((l_value + 1) ** 2 - m**2) / np.sqrt(
            (2 * l_value + 1) * (2 * l_value + 3)
        )
        angular[input_index + 1, input_index] = coefficient
        angular[input_index, input_index + 1] = coefficient
    return hamiltonian.r[:, None] * np.einsum(
        "ij,...rj->...ri", angular, values, optimize=True
    )


def test_weak_pulse_agrees_with_spherical_one_radial_representation():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    prolate_hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 4.0, 8.0],
        xi_points_per_element=7,
        eta_degree=9,
        m_max=0,
    )
    prolate_ground_state = diagonalize_fixed_m(
        prolate_hamiltonian,
        m=0,
        n_states=1,
        solver="direct",
    )
    spherical_hamiltonian = SphericalHamiltonian(
        system,
        nodes=[0.0, 1.0, 2.0, 4.0, 8.0],
        n_points=[7, 7, 7, 7],
        l_max=8,
        m_max=0,
        nuclear_L_max=16,
        nuclear_method="analytic",
    )
    spherical_ground_state = diagonalize_spherical_fixed_m(
        spherical_hamiltonian, m=0, n_states=1
    )
    pulse = SineSquareLaser(
        E0=0.01,
        omega=2.0,
        ncycles=1,
        gauge="length",
    )
    duration = np.pi
    prolate_action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        prolate_hamiltonian, fields=pulse
    )

    def spherical_action(state, time):
        return spherical_hamiltonian.apply_fixed_m(
            state, 0
        ) + pulse(time) * _spherical_fixed_m_z(
            spherical_hamiltonian, state
        )

    propagator = _propagator("sia", duration, 0.1)
    prolate_result = propagator.propagate(
        prolate_ground_state.orbital,
        prolate_action,
        prolate_action.inner_product,
        samplers={
            "dipole": lambda state, time: -prolate_hamiltonian.inner_product(
                state,
                prolate_hamiltonian.apply_z_fixed_m(state),
            ).real
        },
    )
    spherical_result = propagator.propagate(
        spherical_ground_state.orbital,
        spherical_action,
        spherical_hamiltonian.inner_product,
        samplers={
            "dipole": lambda state, time: -spherical_hamiltonian.inner_product(
                state,
                _spherical_fixed_m_z(spherical_hamiltonian, state),
            ).real
        },
    )

    prolate_overlap = prolate_hamiltonian.inner_product(
        prolate_ground_state.orbital, prolate_result.state
    )
    spherical_overlap = spherical_hamiltonian.inner_product(
        spherical_ground_state.orbital, spherical_result.state
    )
    prolate_survival = abs(prolate_overlap) ** 2
    spherical_survival = abs(spherical_overlap) ** 2

    assert abs(
        prolate_ground_state.electronic_energy
        - spherical_ground_state.electronic_energy
    ) < 2.0e-3
    assert prolate_result.norm == pytest.approx(1.0, abs=2.0e-11)
    assert spherical_result.norm == pytest.approx(1.0, abs=2.0e-11)
    assert prolate_survival == pytest.approx(
        spherical_survival, abs=5.0e-8
    )
    np.testing.assert_allclose(
        prolate_result.samples["dipole"],
        spherical_result.samples["dipole"],
        atol=2.0e-5,
    )
