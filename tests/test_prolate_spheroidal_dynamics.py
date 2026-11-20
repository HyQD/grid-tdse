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
        n_states=2,
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


def test_axial_momentum_matches_field_free_commutator(stationary_problem):
    hamiltonian, eigenstates = stationary_problem
    ground = eigenstates.orbitals[0]
    excited = eigenstates.orbitals[1]
    z_matrix_element = hamiltonian.inner_product(
        ground,
        hamiltonian.apply_z_fixed_m(excited),
    )
    momentum_matrix_element = hamiltonian.inner_product(
        ground,
        hamiltonian.apply_pz_fixed_m(excited),
    )
    expected = (
        1j
        * hamiltonian.system.particle_mass
        * (
            eigenstates.electronic_energies[0]
            - eigenstates.electronic_energies[1]
        )
        * z_matrix_element
    )

    assert momentum_matrix_element == pytest.approx(
        expected, rel=2.0e-6, abs=2.0e-10
    )


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


def test_length_and_velocity_gauges_agree_for_weak_complete_pulse(
    stationary_problem,
):
    hamiltonian, eigenstates = stationary_problem
    pulse_parameters = {
        "E0": 1.0e-3,
        "omega": 4.0,
        "ncycles": 1,
        "phase": 0.0,
    }
    electric_field = SineSquareLaser(
        gauge="length", **pulse_parameters
    )
    vector_potential = SineSquareLaser(
        gauge="velocity", **pulse_parameters
    )
    length_action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=electric_field,
        field_gauge="length",
    )
    velocity_action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=vector_potential,
        field_gauge="velocity",
    )
    dipole_sampler = {
        "dipole": lambda state, time: -hamiltonian.inner_product(
            state,
            hamiltonian.apply_z_fixed_m(state),
        ).real
    }
    propagator = _propagator("sia", electric_field.tprime, 0.05)

    length_result = propagator.propagate(
        eigenstates.orbital,
        length_action,
        length_action.inner_product,
        samplers=dipole_sampler,
    )
    velocity_result = propagator.propagate(
        eigenstates.orbital,
        velocity_action,
        velocity_action.inner_product,
        samplers=dipole_sampler,
    )
    overlap = hamiltonian.inner_product(
        length_result.state, velocity_result.state
    )
    fidelity = abs(overlap) ** 2 / (
        length_result.norm * velocity_result.norm
    )

    assert vector_potential(0.0) == pytest.approx(0.0, abs=1.0e-15)
    assert vector_potential(vector_potential.tprime) == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert length_result.norm == pytest.approx(1.0, abs=2.0e-12)
    assert velocity_result.norm == pytest.approx(1.0, abs=2.0e-12)
    assert fidelity == pytest.approx(1.0, abs=2.0e-12)
    np.testing.assert_allclose(
        length_result.samples["dipole"],
        velocity_result.samples["dipole"],
        atol=2.0e-7,
    )
    assert velocity_action.archive_metadata()["field_gauge"] == "velocity"


def test_velocity_gauge_rejects_length_only_options(stationary_problem):
    hamiltonian, _ = stationary_problem
    pulse = SineSquareLaser(
        E0=0.01,
        omega=2.0,
        ncycles=1,
        gauge="velocity",
    )

    with pytest.raises(ValueError, match="length-gauge phase option"):
        ProlateSpheroidalFixedMTimeDependentHamiltonian(
            hamiltonian,
            fields=pulse,
            field_gauge="velocity",
            include_nuclear_scalar=True,
        )
    with pytest.raises(ValueError, match="length-gauge option"):
        ProlateSpheroidalFixedMTimeDependentHamiltonian(
            hamiltonian,
            fields=pulse,
            field_gauge="velocity",
            origin_z=0.2,
        )


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


def test_heteronuclear_origin_shift_changes_only_global_phase():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=3.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="LiH3+",
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 4.0],
        xi_points_per_element=6,
        eta_degree=7,
        m_max=0,
    )
    ground_state = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=1,
        solver="direct",
    ).orbital
    field_amplitude = 0.2
    origin_z = 0.4
    final_time = 0.3
    midpoint = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=lambda time: field_amplitude,
        include_nuclear_scalar=True,
    )
    shifted = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=lambda time: field_amplitude,
        include_nuclear_scalar=True,
        origin_z=origin_z,
    )
    propagator = _propagator("sia", final_time, 0.05)

    midpoint_result = midpoint.propagate(propagator, ground_state)
    shifted_result = shifted.propagate(propagator, ground_state)

    phase = np.exp(
        -1j
        * system.net_charge
        * origin_z
        * field_amplitude
        * final_time
    )
    np.testing.assert_allclose(
        shifted_result.state,
        phase * midpoint_result.state,
        atol=2.0e-11,
    )
    assert shifted_result.norm == pytest.approx(1.0, abs=2.0e-12)


@pytest.mark.parametrize(
    ("name", "nuclear_charge1", "nuclear_charge2", "parity_mixed"),
    [
        ("H2+", 1.0, 1.0, False),
        ("LiH3+", 3.0, 1.0, True),
        ("BH5+", 5.0, 1.0, True),
    ],
)
def test_target_family_field_free_and_short_dynamics(
    name,
    nuclear_charge1,
    nuclear_charge2,
    parity_mixed,
):
    system = OneElectronDiatomicSystem(
        nuclear_charge1=nuclear_charge1,
        nuclear_charge2=nuclear_charge2,
        half_bond_length=1.0,
        name=name,
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 4.0],
        xi_points_per_element=6,
        eta_degree=7,
        m_max=0,
    )
    ground_state = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=1,
        solver="direct",
    )
    pulse = SineSquareLaser(
        E0=0.01,
        omega=4.0,
        ncycles=1,
        gauge="length",
        phase=np.pi / 2.0,
    )
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=pulse,
        include_nuclear_scalar=True,
    )

    result = _propagator("sia", pulse.tprime, 0.1).propagate(
        ground_state.orbital,
        action,
        action.inner_product,
    )
    parity = hamiltonian.inversion_expectation(ground_state.orbital)

    assert ground_state.residual_norm < 1.0e-10
    assert result.norm == pytest.approx(1.0, abs=2.0e-11)
    if parity_mixed:
        assert abs(parity) < 0.99
    else:
        assert parity == pytest.approx(1.0, abs=1.0e-12)


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
    np.testing.assert_allclose(
        spherical_hamiltonian.apply_z_fixed_m(
            spherical_ground_state.orbital, 0
        ),
        _spherical_fixed_m_z(
            spherical_hamiltonian, spherical_ground_state.orbital, 0
        ),
        atol=2.0e-14,
        rtol=2.0e-14,
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
        ) + pulse(time) * spherical_hamiltonian.apply_z_fixed_m(
            state, 0
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
                spherical_hamiltonian.apply_z_fixed_m(state, 0),
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
