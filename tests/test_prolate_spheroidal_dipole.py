import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalAxialLengthGaugeInteraction,
    ProlateSpheroidalAxialVelocityGaugeInteraction,
    ProlateSpheroidalOneElectronHamiltonian,
)


def _hamiltonian(*, homonuclear=True):
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0 if homonuclear else 1.5,
        nuclear_charge2=1.0 if homonuclear else 0.75,
        half_bond_length=1.0,
        name="test",
    )
    return ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 4.0],
        xi_points_per_element=5,
        eta_degree=5,
        m_max=1,
    )


@pytest.mark.parametrize("m", [0, 1, -1])
def test_fixed_m_z_is_local_and_preserves_m(m):
    hamiltonian = _hamiltonian()
    state = np.arange(np.prod(hamiltonian.fixed_m_shape(m))).reshape(
        hamiltonian.fixed_m_shape(m)
    )
    full_z = (
        hamiltonian.system.half_bond_length
        * hamiltonian.grid.xi[:, None]
        * hamiltonian.grid.eta[None, :]
    )
    expected_z = full_z.reshape(-1)[
        hamiltonian.representation.active_indices(m)
    ].reshape(hamiltonian.fixed_m_shape(m))

    z = hamiltonian.fixed_m_z(m)

    np.testing.assert_array_equal(z, expected_z)
    np.testing.assert_array_equal(
        hamiltonian.apply_z_fixed_m(state, m), expected_z * state
    )
    assert z.flags.writeable is False


def test_apply_z_preserves_leading_batch_dimensions():
    hamiltonian = _hamiltonian()
    states = np.ones((3, *hamiltonian.fixed_m_shape(0)))

    result = hamiltonian.apply_z_fixed_m(states)

    assert result.shape == states.shape
    np.testing.assert_array_equal(
        result, np.broadcast_to(hamiltonian.fixed_m_z(), states.shape)
    )


@pytest.mark.parametrize("m", [0, 1, -1])
def test_spatial_action_is_metric_hermitian(m):
    hamiltonian = _hamiltonian(homonuclear=False)
    interaction = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
    )
    generator = np.random.default_rng(109 + m)
    left = generator.normal(size=hamiltonian.fixed_m_shape(m))
    left = left + 1j * generator.normal(size=left.shape)
    right = generator.normal(size=hamiltonian.fixed_m_shape(m))
    right = right + 1j * generator.normal(size=right.shape)

    lhs = hamiltonian.inner_product(
        left, interaction.spatial_action(right, m), m
    )
    rhs = hamiltonian.inner_product(
        interaction.spatial_action(left, m), right, m
    )

    assert lhs == pytest.approx(rhs, abs=2.0e-12)


@pytest.mark.parametrize("m", [0, 1, -1])
def test_axial_z_flips_homonuclear_inversion_parity(m):
    hamiltonian = _hamiltonian()
    interaction = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian, fields=lambda time: 1.0
    )
    generator = np.random.default_rng(211 + m)
    seed = generator.normal(size=hamiltonian.fixed_m_shape(m))
    even = seed + hamiltonian.invert(seed, m)
    action = interaction.spatial_action(even, m)

    np.testing.assert_allclose(
        hamiltonian.invert(action, m), -action, atol=1.0e-14
    )
    assert interaction.electronic_dipole_expectation(
        even, m
    ) == pytest.approx(0.0, abs=2.0e-15)


def test_electron_charge_sign_and_temporal_field_are_separate():
    hamiltonian = _hamiltonian()
    interaction = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian,
        fields=(lambda time: time, lambda time: 2.0 * time),
    )
    state = np.ones(hamiltonian.fixed_m_shape())

    np.testing.assert_array_equal(
        interaction.spatial_action(state), hamiltonian.fixed_m_z()
    )
    assert interaction.field_amplitude(0.3) == pytest.approx(0.9)
    np.testing.assert_allclose(
        interaction(state, 0.3),
        0.9 * interaction.spatial_action(state),
    )


def test_electromagnetic_fields_length_profile_is_accepted():
    hamiltonian = _hamiltonian()
    pulse = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=2,
        gauge="length",
        phase=0.1,
    )
    interaction = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian, fields=pulse
    )

    assert interaction.field_amplitude(0.4) == pytest.approx(pulse(0.4))


def test_zero_field_reproduces_field_free_action():
    hamiltonian = _hamiltonian()
    interaction = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian, fields=lambda time: 0.0
    )
    generator = np.random.default_rng(312)
    state = generator.normal(size=hamiltonian.fixed_m_shape())
    state = state + 1j * generator.normal(size=state.shape)

    dipole_action = interaction(state, 0.7)

    np.testing.assert_array_equal(dipole_action, np.zeros_like(state))
    np.testing.assert_allclose(
        hamiltonian.apply_fixed_m(state) + dipole_action,
        hamiltonian.apply_fixed_m(state),
    )


def test_heteronuclear_nuclear_scalar_is_explicit_and_optional():
    hamiltonian = _hamiltonian(homonuclear=False)
    electronic = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian, fields=lambda time: 1.0
    )
    total = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
    )
    state = np.ones(hamiltonian.fixed_m_shape())
    nuclear_dipole = -0.75

    assert electronic.nuclear_dipole_z == pytest.approx(nuclear_dipole)
    np.testing.assert_allclose(
        total.spatial_action(state) - electronic.spatial_action(state),
        -nuclear_dipole * state,
    )
    assert total.electronic_dipole_expectation(state) == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert total.total_dipole_expectation(state) == pytest.approx(
        nuclear_dipole
    )
    metadata = total.archive_metadata()
    assert metadata["origin"] == "bond_midpoint"
    assert metadata["nuclear_dipole_z"] == pytest.approx(nuclear_dipole)
    assert metadata["include_nuclear_scalar"] is True


def test_total_dipole_and_interaction_transform_with_origin():
    hamiltonian = _hamiltonian(homonuclear=False)
    midpoint = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
    )
    origin_z = 0.4
    shifted = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 1.0,
        include_nuclear_scalar=True,
        origin_z=origin_z,
    )
    state = np.ones(hamiltonian.fixed_m_shape())
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
    metadata = shifted.archive_metadata()
    assert metadata["origin"] == "axial_offset_from_bond_midpoint"
    assert metadata["origin_z"] == pytest.approx(origin_z)


@pytest.mark.parametrize(
    ("origin_z", "error"),
    [(True, TypeError), ("0.2", TypeError), (np.inf, ValueError)],
)
def test_rejects_invalid_axial_origin(origin_z, error):
    hamiltonian = _hamiltonian(homonuclear=False)

    with pytest.raises(error, match="origin_z"):
        ProlateSpheroidalAxialLengthGaugeInteraction(
            hamiltonian,
            fields=lambda time: 1.0,
            origin_z=origin_z,
        )


def test_rejects_velocity_profile_and_invalid_field_value():
    hamiltonian = _hamiltonian()
    pulse = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=2,
        gauge="velocity",
    )
    with pytest.raises(ValueError, match="uses gauge 'velocity'"):
        ProlateSpheroidalAxialLengthGaugeInteraction(
            hamiltonian, fields=pulse
        )

    interaction = ProlateSpheroidalAxialLengthGaugeInteraction(
        hamiltonian, fields=lambda time: np.ones(2)
    )
    with pytest.raises(ValueError, match="finite real scalar"):
        interaction(np.ones(hamiltonian.fixed_m_shape()), 0.0)


@pytest.mark.parametrize("m", [0, 1, -1])
def test_axial_momentum_is_metric_hermitian_and_odd_under_inversion(m):
    hamiltonian = _hamiltonian()
    generator = np.random.default_rng(431 + m)
    left = generator.normal(size=hamiltonian.fixed_m_shape(m))
    left = left + 1j * generator.normal(size=left.shape)
    right = generator.normal(size=hamiltonian.fixed_m_shape(m))
    right = right + 1j * generator.normal(size=right.shape)

    lhs = hamiltonian.inner_product(
        left, hamiltonian.apply_pz_fixed_m(right, m), m
    )
    rhs = hamiltonian.inner_product(
        hamiltonian.apply_pz_fixed_m(left, m), right, m
    )
    assert lhs == pytest.approx(rhs, abs=3.0e-12)

    seed = generator.normal(size=hamiltonian.fixed_m_shape(m))
    even = seed + hamiltonian.invert(seed, m)
    action = hamiltonian.apply_pz_fixed_m(even, m)
    np.testing.assert_allclose(
        hamiltonian.invert(action, m), -action, atol=3.0e-12
    )


def test_velocity_gauge_uses_vector_potential_and_charge_to_mass_coupling():
    hamiltonian = _hamiltonian()
    pulse = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=2,
        gauge="velocity",
    )
    interaction = ProlateSpheroidalAxialVelocityGaugeInteraction(
        hamiltonian, fields=pulse
    )
    state = np.ones(hamiltonian.fixed_m_shape(), dtype=complex)

    assert interaction.momentum_coupling == pytest.approx(-1.0)
    np.testing.assert_allclose(
        interaction.spatial_action(state),
        hamiltonian.apply_pz_fixed_m(state),
    )
    assert interaction.field_amplitude(0.4) == pytest.approx(pulse(0.4))
    metadata = interaction.archive_metadata()
    assert metadata["gauge"] == "velocity"
    assert metadata["diamagnetic_a_squared_term_included"] is False

    length_pulse = SineSquareLaser(
        E0=0.2,
        omega=0.7,
        ncycles=2,
        gauge="length",
    )
    with pytest.raises(ValueError, match="uses gauge 'length'"):
        ProlateSpheroidalAxialVelocityGaugeInteraction(
            hamiltonian, fields=length_pulse
        )
