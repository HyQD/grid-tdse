from types import SimpleNamespace

import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser
from grid_lib.spherical_coordinates.angular_momentum import angular_matrix_element

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    AxialCoordinateDipole,
    AxialDipoleCouplings,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    FEMDVRGridSpec,
    TwoElectronAtomParameters,
    TwoRadialAxialDipoleInteraction,
    TwoRadialSystemSpec,
)


def _system(*, parity1=None, parity2=None, grid_family="femdvr"):
    if grid_family != "femdvr":
        from grid_tdse.spherical.two_radial import RadialSincDVRGridSpec

        grid1 = RadialSincDVRGridSpec(5.0, 8)
        grid2 = RadialSincDVRGridSpec(4.0, 7)
    else:
        grid1 = FEMDVRGridSpec([0.0, 2.0, 5.0], [5, 5])
        grid2 = FEMDVRGridSpec([0.0, 1.5, 4.0], [5, 5])
    basis1 = AngularBasisSpec(2, 2, coordinate_parity=parity1)
    basis2 = AngularBasisSpec(2, 2, coordinate_parity=parity2)
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(grid1, basis1, -0.5),
        coordinate2=CoordinateSpec(grid2, basis2, -0.25),
        multipoles=CoulombMultipoleSpec(0, 0),
        coulomb_terms=(CoulombTerm(1.0, 1.0),),
    ).build()


def _random_state(system, seed=18):
    rng = np.random.default_rng(seed)
    return rng.normal(size=system.psi_shape) + 1j * rng.normal(
        size=system.psi_shape
    )


def _quadrature_cos_theta(states):
    result = np.empty((len(states), len(states)), dtype=complex)
    for output, (l1, m1) in enumerate(states):
        for input_, (l2, m2) in enumerate(states):
            result[output, input_] = angular_matrix_element(
                l1,
                m1,
                l2,
                m2,
                lambda theta, phi: np.cos(theta),
                order=15,
            )
    return result


def _a_lm(l_value, m_value):
    return np.sqrt(
        (l_value + m_value + 1)
        * (l_value - m_value + 1)
        / ((2 * l_value + 1) * (2 * l_value + 3))
    )


def _analytic_z_angular_matrices(basis):
    cos_theta = np.zeros((basis.n_channels, basis.n_channels))
    sin_theta_dtheta = np.zeros_like(cos_theta)
    for input_, (l_value, m_value) in enumerate(basis.states):
        if l_value < basis.l_max:
            output = basis.index(l_value + 1, m_value)
            coefficient = _a_lm(l_value, m_value)
            cos_theta[output, input_] = coefficient
            sin_theta_dtheta[output, input_] = l_value * coefficient
        if l_value > abs(m_value):
            output = basis.index(l_value - 1, m_value)
            coefficient = _a_lm(l_value - 1, m_value)
            cos_theta[output, input_] = coefficient
            sin_theta_dtheta[output, input_] = -(l_value + 1) * coefficient
    return cos_theta, sin_theta_dtheta


def _direct_length_action(system, state, coordinate, coupling):
    if coordinate == 1:
        grid = system.grid1
        basis = system.basis1
        return -coupling * np.einsum(
            "a,IK,KJab->IJab",
            grid.r,
            _quadrature_cos_theta(basis.states),
            state,
            optimize=True,
        )
    grid = system.grid2
    basis = system.basis2
    return -coupling * np.einsum(
        "b,JL,ILab->IJab",
        grid.r,
        _quadrature_cos_theta(basis.states),
        state,
        optimize=True,
    )


def _direct_velocity_action(system, state, coordinate, coupling):
    if coordinate == 1:
        grid = system.grid1
        cos_theta, sin_theta_dtheta = _analytic_z_angular_matrices(system.basis1)
        d1_tilde = grid.first_derivative(0) - np.diag(1.0 / grid.r)
        derivative = np.einsum(
            "ac,IK,KJcb->IJab", d1_tilde, cos_theta, state, optimize=True
        )
        derivative -= np.einsum(
            "a,IK,KJab->IJab",
            1.0 / grid.r,
            sin_theta_dtheta,
            state,
            optimize=True,
        )
    else:
        grid = system.grid2
        cos_theta, sin_theta_dtheta = _analytic_z_angular_matrices(system.basis2)
        d1_tilde = grid.first_derivative(0) - np.diag(1.0 / grid.r)
        derivative = np.einsum(
            "bd,JL,ILad->IJab", d1_tilde, cos_theta, state, optimize=True
        )
        derivative -= np.einsum(
            "b,JL,ILab->IJab",
            1.0 / grid.r,
            sin_theta_dtheta,
            state,
            optimize=True,
        )
    return 1j * coupling * derivative


@pytest.mark.parametrize("coordinate", [1, 2])
def test_length_action_matches_independent_angular_quadrature(coordinate):
    system = _system()
    state = _random_state(system)
    coupling = -0.73
    operator = AxialCoordinateDipole(
        system, coordinate, gauge="length", coupling=coupling
    )

    actual = operator(state)
    expected = _direct_length_action(system, state, coordinate, coupling)

    np.testing.assert_allclose(actual, expected, atol=3.0e-14, rtol=3.0e-14)


@pytest.mark.parametrize("coordinate", [1, 2])
def test_velocity_action_matches_direct_reduced_radial_contraction(coordinate):
    system = _system()
    state = _random_state(system)
    coupling = -1.27
    operator = AxialCoordinateDipole(
        system, coordinate, gauge="velocity", coupling=coupling
    )

    actual = operator(state)
    expected = _direct_velocity_action(system, state, coordinate, coupling)

    np.testing.assert_allclose(actual, expected, atol=3.0e-14, rtol=3.0e-14)


def test_velocity_action_on_s_wave_matches_analytic_radial_derivative():
    system = _system()
    state = np.zeros(system.psi_shape)
    s1 = system.basis1.index(0, 0)
    s2 = system.basis2.index(0, 0)
    radial = system.grid1.r * (5.0 - system.grid1.r)
    state[s1, s2] = radial[:, None]
    coupling = -0.8
    operator = AxialCoordinateDipole(
        system, 1, gauge="velocity", coupling=coupling
    )

    reduced_derivative = 5.0 - 2.0 * system.grid1.r - radial / system.grid1.r
    p1 = system.basis1.index(1, 0)
    expected = np.zeros(system.psi_shape, dtype=complex)
    expected[p1, s2] = (
        1j
        * coupling
        * _a_lm(0, 0)
        * reduced_derivative[:, None]
    )

    np.testing.assert_allclose(operator(state), expected, atol=2.0e-13)


@pytest.mark.parametrize("gauge", ["length", "velocity"])
@pytest.mark.parametrize("coordinate", [1, 2])
def test_coordinate_actions_are_hermitian_in_two_radial_metric(gauge, coordinate):
    system = _system()
    left = _random_state(system, seed=2)
    right = _random_state(system, seed=3)
    operator = AxialCoordinateDipole(
        system, coordinate, gauge=gauge, coupling=0.61
    )

    lhs = system.inner_product(left, operator(right))
    rhs = system.inner_product(operator(left), right)

    np.testing.assert_allclose(lhs, rhs, atol=4.0e-11, rtol=3.0e-12)


def test_two_electron_factory_applies_equal_couplings_to_both_coordinates():
    system = _system()
    parameters = TwoElectronAtomParameters(
        "test", nuclear_charge=2.0, ionic_charge=0.0,
        electron_mass=2.0, electron_charge=-1.5,
    )
    interaction = TwoRadialAxialDipoleInteraction.for_two_electron_atom(
        system,
        parameters,
        gauge="velocity",
        fields=lambda time: 0.4 + time,
    )
    state = _random_state(system)
    time = 0.3
    coupling = parameters.electron_charge / parameters.electron_mass
    expected = (0.4 + time) * (
        _direct_velocity_action(system, state, 1, coupling)
        + _direct_velocity_action(system, state, 2, coupling)
    )

    assert interaction.couplings == AxialDipoleCouplings(coupling, coupling)
    np.testing.assert_allclose(interaction(state, time), expected, atol=5.0e-14)


def test_homonuclear_zero_coupling_allows_reduced_internuclear_basis():
    system = _system(parity2=1)
    parameters = SimpleNamespace(
        electronic_reduced_charge=-0.999,
        internuclear_reduced_charge=0.0,
        electronic_velocity_coupling=-1.001,
        internuclear_velocity_coupling=0.0,
    )
    interaction = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
        system,
        parameters,
        gauge="length",
        fields=lambda time: 2.0,
    )
    state = _random_state(system)

    assert [term.coordinate for term in interaction.coordinate_operators] == [1]
    expected = 2.0 * _direct_length_action(
        system, state, 1, parameters.electronic_reduced_charge
    )
    np.testing.assert_allclose(interaction(state, 0.0), expected, atol=5.0e-14)


def test_heteronuclear_factory_applies_distinct_velocity_couplings():
    system = _system()
    parameters = SimpleNamespace(
        electronic_reduced_charge=-0.91,
        internuclear_reduced_charge=0.14,
        electronic_velocity_coupling=-1.07,
        internuclear_velocity_coupling=0.03,
    )
    interaction = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
        system,
        parameters,
        gauge="velocity",
        fields=lambda time: 0.6,
    )
    state = _random_state(system)
    expected = 0.6 * (
        _direct_velocity_action(
            system, state, 1, parameters.electronic_velocity_coupling
        )
        + _direct_velocity_action(
            system, state, 2, parameters.internuclear_velocity_coupling
        )
    )

    assert interaction.couplings == AxialDipoleCouplings(-1.07, 0.03)
    np.testing.assert_allclose(interaction(state, 0.0), expected, atol=5.0e-14)


def test_nonzero_coupling_rejects_reduced_coordinate_basis():
    system = _system(parity2=1)
    with pytest.raises(ValueError, match="coordinate 2.*both even- and odd-l"):
        TwoRadialAxialDipoleInteraction(
            system,
            gauge="length",
            fields=lambda time: 1.0,
            couplings=AxialDipoleCouplings(-1.0, 0.2),
        )


def test_temporal_profiles_are_summed_and_input_is_not_modified():
    system = _system()
    state = _random_state(system)
    original = state.copy()
    interaction = TwoRadialAxialDipoleInteraction(
        system,
        gauge="length",
        fields=(lambda time: time, lambda time: 2.0 * time),
        couplings=AxialDipoleCouplings(-1.0, 0.0),
    )

    expected = 0.9 * _direct_length_action(system, state, 1, -1.0)
    np.testing.assert_allclose(interaction(state, 0.3), expected, atol=5.0e-14)
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize("gauge", ["length", "velocity"])
def test_electromagnetic_fields_profile_is_accepted(gauge):
    system = _system()
    pulse = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=2,
        gauge=gauge,
        phase=0.1,
    )
    interaction = TwoRadialAxialDipoleInteraction(
        system,
        gauge=gauge,
        fields=pulse,
        couplings=AxialDipoleCouplings(-1.0, -1.0),
    )

    assert interaction.field_amplitude(0.4) == pytest.approx(pulse(0.4))


def test_rejects_field_gauge_mismatch_and_invalid_field_values():
    system = _system()
    velocity_field = SineSquareLaser(
        E0=0.2, omega=0.7, ncycles=2, gauge="velocity"
    )
    with pytest.raises(ValueError, match="uses gauge 'velocity'"):
        TwoRadialAxialDipoleInteraction(
            system,
            gauge="length",
            fields=velocity_field,
            couplings=AxialDipoleCouplings(-1.0, -1.0),
        )

    interaction = TwoRadialAxialDipoleInteraction(
        system,
        gauge="length",
        fields=lambda time: np.ones(2),
        couplings=AxialDipoleCouplings(-1.0, 0.0),
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        interaction(np.ones(system.psi_shape), 0.0)


def test_rejects_sinc_grid_until_its_l_dependent_derivative_is_supported():
    system = _system(grid_family="sinc")
    with pytest.raises(ValueError, match="currently require FEM-DVR"):
        AxialCoordinateDipole(system, 1, gauge="velocity", coupling=-1.0)


def test_zero_field_returns_complex_zeros_in_velocity_gauge():
    system = _system()
    interaction = TwoRadialAxialDipoleInteraction(
        system,
        gauge="velocity",
        fields=lambda time: 0.0,
        couplings=AxialDipoleCouplings(-1.0, 0.0),
    )

    result = interaction(np.ones(system.psi_shape), 0.0)

    assert np.iscomplexobj(result)
    assert np.all(result == 0.0)
