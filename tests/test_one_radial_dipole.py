import json

import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser

from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    SphericalOneRadialAxialLengthGaugeInteraction,
    SphericalOneRadialAxialVelocityGaugeInteraction,
)


def _hamiltonian(*, homonuclear=True, particle_mass=1.0):
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0 if homonuclear else 1.5,
        nuclear_charge2=1.0 if homonuclear else 0.75,
        half_bond_length=1.0,
        particle_mass=particle_mass,
        name="test",
    )
    return OneElectronDiatomicHamiltonian(
        system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[6, 6, 6],
        l_max=3,
        m_max=1,
    )


def _random_state(hamiltonian, m=0, *, seed=1):
    generator = np.random.default_rng(seed)
    shape = hamiltonian.fixed_m_shape(m)
    return generator.normal(size=shape) + 1j * generator.normal(size=shape)


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_length_spatial_action_is_metric_hermitian(m):
    hamiltonian = _hamiltonian(homonuclear=False)
    interaction = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
    )
    left = _random_state(hamiltonian, m, seed=109 + m)
    right = _random_state(hamiltonian, m, seed=211 + m)

    lhs = hamiltonian.inner_product(
        left, interaction.spatial_action(right, m)
    )
    rhs = hamiltonian.inner_product(
        interaction.spatial_action(left, m), right
    )

    assert lhs == pytest.approx(rhs, abs=3.0e-13, rel=3.0e-13)


def test_electron_charge_sign_and_temporal_fields_are_separate():
    hamiltonian = _hamiltonian()
    interaction = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian,
        fields=(lambda time: time, lambda time: 2.0 * time),
    )
    states = np.ones((2, 3, *hamiltonian.fixed_m_shape(0)))

    np.testing.assert_allclose(
        interaction.spatial_action(states),
        hamiltonian.apply_z_fixed_m(states),
    )
    assert interaction.field_amplitude(0.3) == pytest.approx(0.9)
    np.testing.assert_allclose(
        interaction(states, 0.3),
        0.9 * interaction.spatial_action(states),
    )


def test_electromagnetic_fields_profiles_are_gauge_checked():
    hamiltonian = _hamiltonian()
    length_pulse = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=2,
        gauge="length",
    )
    velocity_pulse = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=2,
        gauge="velocity",
    )

    length = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian, fields=length_pulse
    )
    velocity = SphericalOneRadialAxialVelocityGaugeInteraction(
        hamiltonian, fields=velocity_pulse
    )

    assert length.field_amplitude(0.4) == pytest.approx(length_pulse(0.4))
    assert velocity.field_amplitude(0.4) == pytest.approx(
        velocity_pulse(0.4)
    )
    with pytest.raises(ValueError, match="uses gauge 'velocity'"):
        SphericalOneRadialAxialLengthGaugeInteraction(
            hamiltonian, fields=velocity_pulse
        )
    with pytest.raises(ValueError, match="uses gauge 'length'"):
        SphericalOneRadialAxialVelocityGaugeInteraction(
            hamiltonian, fields=length_pulse
        )


@pytest.mark.parametrize(
    "interaction_type",
    [
        SphericalOneRadialAxialLengthGaugeInteraction,
        SphericalOneRadialAxialVelocityGaugeInteraction,
    ],
)
def test_zero_field_produces_exactly_zero_action(interaction_type):
    hamiltonian = _hamiltonian()
    interaction = interaction_type(hamiltonian, fields=lambda time: 0.0)
    state = _random_state(hamiltonian, seed=312)

    interaction_action = interaction(state, 0.7)

    np.testing.assert_array_equal(
        interaction_action, np.zeros_like(state)
    )
    np.testing.assert_allclose(
        hamiltonian.apply_fixed_m(state, 0) + interaction_action,
        hamiltonian.apply_fixed_m(state, 0),
    )


def test_midpoint_h2plus_nuclear_scalar_vanishes():
    hamiltonian = _hamiltonian()
    electronic = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian, fields=lambda time: 1.0
    )
    total = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
    )
    state = _random_state(hamiltonian, seed=413)

    assert total.nuclear_dipole_z == pytest.approx(0.0)
    np.testing.assert_array_equal(
        total.spatial_action(state), electronic.spatial_action(state)
    )
    metadata = total.archive_metadata()
    assert metadata["nuclear_scalar_coefficient"] == pytest.approx(0.0)
    assert metadata["omitted_nuclear_scalar_coefficient"] == pytest.approx(
        0.0
    )


def test_heteronuclear_nuclear_scalar_is_explicit_and_optional():
    hamiltonian = _hamiltonian(homonuclear=False)
    electronic = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian, fields=lambda time: 1.0
    )
    total = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
    )
    state = _random_state(hamiltonian, seed=514)
    nuclear_dipole = -0.75

    assert electronic.nuclear_dipole_z == pytest.approx(nuclear_dipole)
    np.testing.assert_allclose(
        total.spatial_action(state) - electronic.spatial_action(state),
        -nuclear_dipole * state,
    )
    electronic_dipole = electronic.electronic_dipole_expectation(state)
    assert total.electronic_dipole_expectation(state) == pytest.approx(
        electronic_dipole
    )
    assert total.total_dipole_expectation(state) == pytest.approx(
        electronic_dipole + nuclear_dipole
    )
    metadata = total.archive_metadata()
    assert metadata["origin"] == "bond_midpoint"
    assert metadata["nuclear_dipole_z"] == pytest.approx(nuclear_dipole)
    assert metadata["nuclear_scalar_coefficient"] == pytest.approx(0.75)
    assert metadata["include_nuclear_scalar"] is True
    assert metadata["omitted_nuclear_scalar_coefficient"] == pytest.approx(
        0.0
    )


def test_total_dipole_and_interaction_transform_with_origin():
    hamiltonian = _hamiltonian(homonuclear=False)
    midpoint = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
    )
    origin_z = 0.4
    shifted = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
        origin_z=origin_z,
    )
    state = _random_state(hamiltonian, seed=615)
    net_charge = hamiltonian.system.net_charge

    assert shifted.nuclear_dipole_z == pytest.approx(
        midpoint.nuclear_dipole_z
        - (
            hamiltonian.system.nuclear_charge1
            + hamiltonian.system.nuclear_charge2
        )
        * origin_z
    )
    assert shifted.electronic_dipole_expectation(state) == pytest.approx(
        midpoint.electronic_dipole_expectation(state)
        - hamiltonian.system.electron_charge * origin_z
    )
    assert shifted.total_dipole_expectation(state) == pytest.approx(
        midpoint.total_dipole_expectation(state) - net_charge * origin_z
    )
    np.testing.assert_allclose(
        shifted.spatial_action(state) - midpoint.spatial_action(state),
        net_charge * origin_z * state,
    )


def test_velocity_gauge_uses_charge_to_mass_coupling_and_momentum():
    hamiltonian = _hamiltonian(particle_mass=2.0)
    interaction = SphericalOneRadialAxialVelocityGaugeInteraction(
        hamiltonian, fields=lambda time: 0.25
    )
    state = _random_state(hamiltonian, seed=716)
    momentum_action = hamiltonian.apply_pz_fixed_m(state)
    norm = hamiltonian.inner_product(state, state)
    expected_momentum = hamiltonian.inner_product(
        state, momentum_action
    ).real / norm.real

    assert interaction.momentum_coupling == pytest.approx(-0.5)
    np.testing.assert_allclose(
        interaction.spatial_action(state), 0.5 * momentum_action
    )
    np.testing.assert_allclose(
        interaction(state, 0.4), 0.125 * momentum_action
    )
    assert interaction.momentum_expectation(state) == pytest.approx(
        expected_momentum
    )
    metadata = interaction.archive_metadata()
    assert metadata["gauge"] == "velocity"
    assert metadata["particle_mass"] == pytest.approx(2.0)
    assert metadata["diamagnetic_a_squared_term_included"] is False


@pytest.mark.parametrize(
    ("origin_z", "error"),
    [(True, TypeError), ("0.2", TypeError), (np.inf, ValueError)],
)
def test_rejects_invalid_axial_origin(origin_z, error):
    with pytest.raises(error, match="origin_z"):
        SphericalOneRadialAxialLengthGaugeInteraction(
            _hamiltonian(),
            fields=lambda time: 1.0,
            origin_z=origin_z,
        )


def test_rejects_invalid_field_collection_and_field_values():
    hamiltonian = _hamiltonian()
    with pytest.raises(ValueError, match="at least one"):
        SphericalOneRadialAxialLengthGaugeInteraction(
            hamiltonian, fields=[]
        )
    with pytest.raises(TypeError, match="every field"):
        SphericalOneRadialAxialLengthGaugeInteraction(
            hamiltonian, fields=[lambda time: 1.0, 2.0]
        )

    interaction = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian, fields=lambda time: np.ones(2)
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        interaction(np.ones(hamiltonian.fixed_m_shape(0)), 0.0)


def test_zero_norm_expectations_are_rejected():
    hamiltonian = _hamiltonian()
    state = np.zeros(hamiltonian.fixed_m_shape(0))
    length = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian, fields=lambda time: 1.0
    )
    velocity = SphericalOneRadialAxialVelocityGaugeInteraction(
        hamiltonian, fields=lambda time: 1.0
    )

    with pytest.raises(ValueError, match="norm must be nonzero"):
        length.electronic_dipole_expectation(state)
    with pytest.raises(ValueError, match="norm must be nonzero"):
        velocity.momentum_expectation(state)


def test_archive_metadata_is_json_compatible():
    hamiltonian = _hamiltonian(homonuclear=False, particle_mass=1.5)
    interactions = (
        SphericalOneRadialAxialLengthGaugeInteraction(
            hamiltonian, fields=lambda time: 1.0, origin_z=0.2
        ),
        SphericalOneRadialAxialVelocityGaugeInteraction(
            hamiltonian, fields=lambda time: 1.0
        ),
    )

    for interaction in interactions:
        assert isinstance(json.dumps(interaction.archive_metadata()), str)
