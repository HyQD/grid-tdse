import numpy as np
import pytest

from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    diagonalize_fixed_m,
)


def _system(**kwargs):
    parameters = dict(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    parameters.update(kwargs)
    return OneElectronDiatomicSystem(**parameters)


def _hamiltonian(
    *,
    points=6,
    l_max=2,
    m_max=1,
    nuclear_L_max=None,
    nuclear_method="analytic",
    system=None,
):
    return OneElectronDiatomicHamiltonian(
        _system() if system is None else system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[points, points, points],
        l_max=l_max,
        m_max=m_max,
        nuclear_L_max=nuclear_L_max,
        nuclear_method=nuclear_method,
    )


def test_hamiltonian_requires_nucleus_at_internal_element_boundary():
    with pytest.raises(ValueError, match="internal FEM element boundary"):
        OneElectronDiatomicHamiltonian(
            _system(half_bond_length=1.2),
            nodes=[0.0, 1.0, 3.0],
            n_points=[5, 5],
            l_max=2,
        )


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_fixed_m_action_matches_explicit_strong_matrix(m):
    hamiltonian = _hamiltonian()
    rng = np.random.default_rng(3 + m)
    shape = hamiltonian.fixed_m_shape(m)
    orbital = rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))

    expected = np.asarray(
        [
            (hamiltonian.fixed_m_matrix(m) @ values.ravel()).reshape(shape)
            for values in orbital
        ]
    )

    np.testing.assert_allclose(
        hamiltonian.apply_fixed_m(orbital, m),
        expected,
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_full_action_preserves_m_and_matches_fixed_m_action():
    hamiltonian = _hamiltonian()
    m = 1
    fixed = np.arange(np.prod(hamiltonian.fixed_m_shape(m)), dtype=float)
    fixed = fixed.reshape(hamiltonian.fixed_m_shape(m))
    full = np.zeros(hamiltonian.orbital_shape)
    indices = [
        index
        for index, (_, magnetic) in enumerate(hamiltonian.orbital_lm)
        if magnetic == m
    ]
    full[:, indices] = fixed

    action = hamiltonian.apply(full)

    np.testing.assert_allclose(
        action[:, indices], hamiltonian.apply_fixed_m(fixed, m)
    )
    complement = np.ones(action.shape[-1], dtype=bool)
    complement[indices] = False
    np.testing.assert_allclose(action[:, complement], 0.0, atol=1.0e-14)


def test_z_action_has_analytic_selection_rules():
    hamiltonian = _hamiltonian(l_max=3, m_max=2)
    radial_values = np.linspace(0.2, 1.1, hamiltonian.r.size)
    orbital = np.zeros(hamiltonian.orbital_shape)
    channel_index = {
        channel: index for index, channel in enumerate(hamiltonian.orbital_lm)
    }
    orbital[:, channel_index[(2, 1)]] = radial_values

    action = hamiltonian.apply_z(orbital)
    expected = np.zeros_like(action)
    expected[:, channel_index[(1, 1)]] = (
        np.sqrt(3.0 / 15.0) * hamiltonian.r * radial_values
    )
    expected[:, channel_index[(3, 1)]] = (
        np.sqrt(8.0 / 35.0) * hamiltonian.r * radial_values
    )

    np.testing.assert_allclose(action, expected, atol=2.0e-15)


@pytest.mark.parametrize("m", [-2, -1, 0, 1, 2])
def test_full_z_action_preserves_m_and_matches_fixed_m_action(m):
    hamiltonian = _hamiltonian(l_max=3, m_max=2)
    rng = np.random.default_rng(30 + m)
    shape = hamiltonian.fixed_m_shape(m)
    fixed = rng.normal(size=(2, 3, *shape)) + 1j * rng.normal(
        size=(2, 3, *shape)
    )
    full = np.zeros(
        (2, 3, *hamiltonian.orbital_shape), dtype=np.complex128
    )
    indices = [
        index
        for index, (_, magnetic) in enumerate(hamiltonian.orbital_lm)
        if magnetic == m
    ]
    full[..., indices] = fixed

    action = hamiltonian.apply_z(full)

    assert action.shape == full.shape
    np.testing.assert_allclose(
        action[..., indices],
        hamiltonian.apply_z_fixed_m(fixed, m),
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    complement = np.ones(action.shape[-1], dtype=bool)
    complement[indices] = False
    np.testing.assert_allclose(
        action[..., complement], 0.0, atol=2.0e-15
    )


def test_z_action_is_hermitian_in_the_femdvr_metric():
    hamiltonian = _hamiltonian(l_max=3, m_max=2)
    rng = np.random.default_rng(44)
    left = rng.normal(size=hamiltonian.orbital_shape) + 1j * rng.normal(
        size=hamiltonian.orbital_shape
    )
    right = rng.normal(size=hamiltonian.orbital_shape) + 1j * rng.normal(
        size=hamiltonian.orbital_shape
    )

    left_z_right = hamiltonian.inner_product(
        left, hamiltonian.apply_z(right)
    )
    z_left_right = hamiltonian.inner_product(
        hamiltonian.apply_z(left), right
    )

    assert left_z_right == pytest.approx(
        z_left_right, abs=2.0e-13, rel=2.0e-13
    )


@pytest.mark.parametrize("m", [-2, 0, 2])
def test_fixed_m_z_action_is_hermitian_in_the_femdvr_metric(m):
    hamiltonian = _hamiltonian(l_max=3, m_max=2)
    rng = np.random.default_rng(51 + m)
    shape = hamiltonian.fixed_m_shape(m)
    left = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    right = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    left_z_right = hamiltonian.inner_product(
        left, hamiltonian.apply_z_fixed_m(right, m)
    )
    z_left_right = hamiltonian.inner_product(
        hamiltonian.apply_z_fixed_m(left, m), right
    )

    assert left_z_right == pytest.approx(
        z_left_right, abs=2.0e-13, rel=2.0e-13
    )


def test_pz_action_has_analytic_selection_rules_and_radial_terms():
    hamiltonian = _hamiltonian(l_max=3, m_max=2)
    radial_values = hamiltonian.r * (6.0 - hamiltonian.r)
    radial_derivative = 6.0 - 2.0 * hamiltonian.r
    orbital = np.zeros(hamiltonian.orbital_shape)
    channel_index = {
        channel: index for index, channel in enumerate(hamiltonian.orbital_lm)
    }
    orbital[:, channel_index[(2, 1)]] = radial_values

    action = hamiltonian.apply_pz(orbital)
    expected = np.zeros_like(action, dtype=np.complex128)
    expected[:, channel_index[(1, 1)]] = -1j * np.sqrt(3.0 / 15.0) * (
        radial_derivative + 2.0 * radial_values / hamiltonian.r
    )
    expected[:, channel_index[(3, 1)]] = -1j * np.sqrt(8.0 / 35.0) * (
        radial_derivative - 3.0 * radial_values / hamiltonian.r
    )

    np.testing.assert_allclose(action, expected, atol=2.0e-13)


@pytest.mark.parametrize("m", [-2, -1, 0, 1, 2])
def test_full_pz_action_preserves_m_and_matches_fixed_m_action(m):
    hamiltonian = _hamiltonian(l_max=3, m_max=2)
    rng = np.random.default_rng(70 + m)
    shape = hamiltonian.fixed_m_shape(m)
    fixed = rng.normal(size=(2, 3, *shape)) + 1j * rng.normal(
        size=(2, 3, *shape)
    )
    full = np.zeros(
        (2, 3, *hamiltonian.orbital_shape), dtype=np.complex128
    )
    indices = [
        index
        for index, (_, magnetic) in enumerate(hamiltonian.orbital_lm)
        if magnetic == m
    ]
    full[..., indices] = fixed

    action = hamiltonian.apply_pz(full)

    assert action.shape == full.shape
    np.testing.assert_allclose(
        action[..., indices],
        hamiltonian.apply_pz_fixed_m(fixed, m),
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    complement = np.ones(action.shape[-1], dtype=bool)
    complement[indices] = False
    np.testing.assert_allclose(
        action[..., complement], 0.0, atol=2.0e-14
    )


def test_pz_action_is_hermitian_in_the_femdvr_metric():
    hamiltonian = _hamiltonian(l_max=3, m_max=2)
    rng = np.random.default_rng(84)
    left = rng.normal(size=hamiltonian.orbital_shape) + 1j * rng.normal(
        size=hamiltonian.orbital_shape
    )
    right = rng.normal(size=hamiltonian.orbital_shape) + 1j * rng.normal(
        size=hamiltonian.orbital_shape
    )

    left_pz_right = hamiltonian.inner_product(
        left, hamiltonian.apply_pz(right)
    )
    pz_left_right = hamiltonian.inner_product(
        hamiltonian.apply_pz(left), right
    )

    assert left_pz_right == pytest.approx(
        pz_left_right, abs=3.0e-13, rel=3.0e-13
    )


@pytest.mark.parametrize("m", [-2, 0, 2])
def test_fixed_m_pz_action_is_hermitian_in_the_femdvr_metric(m):
    hamiltonian = _hamiltonian(l_max=3, m_max=2)
    rng = np.random.default_rng(91 + m)
    shape = hamiltonian.fixed_m_shape(m)
    left = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    right = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    left_pz_right = hamiltonian.inner_product(
        left, hamiltonian.apply_pz_fixed_m(right, m)
    )
    pz_left_right = hamiltonian.inner_product(
        hamiltonian.apply_pz_fixed_m(left, m), right
    )

    assert left_pz_right == pytest.approx(
        pz_left_right, abs=3.0e-13, rel=3.0e-13
    )


def test_pz_converges_to_field_free_commutator_on_refined_grids():
    relative_errors = []
    for points, l_max in ((4, 2), (6, 4), (8, 6)):
        hamiltonian = _hamiltonian(
            points=points,
            l_max=l_max,
            m_max=0,
            nuclear_L_max=2 * l_max,
        )
        state = diagonalize_fixed_m(hamiltonian, m=0).orbital
        direct = hamiltonian.apply_pz_fixed_m(state, 0)
        commutator = 1j * hamiltonian.system.particle_mass * (
            hamiltonian.apply_fixed_m(
                hamiltonian.apply_z_fixed_m(state, 0), 0
            )
            - hamiltonian.apply_z_fixed_m(
                hamiltonian.apply_fixed_m(state, 0), 0
            )
        )
        difference = direct - commutator
        difference_norm = np.sqrt(
            hamiltonian.inner_product(difference, difference).real
        )
        direct_norm = np.sqrt(
            hamiltonian.inner_product(direct, direct).real
        )
        relative_errors.append(difference_norm / direct_norm)

    assert np.all(np.diff(relative_errors) < 0.0)
    assert relative_errors[-1] < 2.0e-2


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_fixed_m_matrix_is_hermitian_in_the_femdvr_metric(m):
    hamiltonian = _hamiltonian()
    strong = hamiltonian.fixed_m_matrix(m)
    weights = hamiltonian.fixed_m_metric_weights(m)
    symmetric = hamiltonian.fixed_m_matrix(m, representation="symmetric")

    np.testing.assert_allclose(
        strong.conj().T * weights[None, :],
        weights[:, None] * strong,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        symmetric,
        symmetric.conj().T,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    metric_sqrt = np.sqrt(weights)
    np.testing.assert_allclose(
        symmetric,
        metric_sqrt[:, None] * strong / metric_sqrt[None, :],
    )


def test_metric_transforms_round_trip_for_batched_orbitals():
    hamiltonian = _hamiltonian()
    rng = np.random.default_rng(9)
    nodal = rng.normal(size=(3, *hamiltonian.fixed_m_shape(0)))

    symmetric = hamiltonian.to_symmetric_fixed_m(nodal, 0)

    np.testing.assert_allclose(
        hamiltonian.from_symmetric_fixed_m(symmetric, 0), nodal
    )
    np.testing.assert_allclose(
        np.sum(np.abs(symmetric) ** 2),
        hamiltonian.inner_product(nodal, nodal),
    )


def test_magnetic_sign_blocks_are_degenerate_for_axial_potential():
    hamiltonian = _hamiltonian(l_max=3, m_max=2)

    np.testing.assert_allclose(
        hamiltonian.fixed_m_matrix(-2),
        hamiltonian.fixed_m_matrix(2),
        atol=2.0e-14,
        rtol=2.0e-14,
    )


def test_nuclear_multipoles_above_twice_lmax_do_not_change_matrix():
    complete = _hamiltonian(l_max=2, m_max=0, nuclear_L_max=4)
    extended = _hamiltonian(l_max=2, m_max=0, nuclear_L_max=7)

    np.testing.assert_allclose(
        complete.fixed_m_matrix(0),
        extended.fixed_m_matrix(0),
        atol=2.0e-14,
        rtol=2.0e-14,
    )


def test_poisson_energy_converges_to_analytic_energy_with_radial_order():
    errors = []
    for points in (6, 8):
        analytic = diagonalize_fixed_m(
            _hamiltonian(
                points=points,
                m_max=0,
                nuclear_method="analytic",
            )
        )
        poisson = diagonalize_fixed_m(
            _hamiltonian(
                points=points,
                m_max=0,
                nuclear_method="poisson",
            )
        )
        errors.append(abs(analytic.electronic_energy - poisson.electronic_energy))

    assert errors[1] < errors[0]
    assert errors[1] < 1.0e-6


def test_diagonalized_states_are_normalized_and_satisfy_eigenproblem():
    hamiltonian = _hamiltonian(points=8, m_max=0)
    result = diagonalize_fixed_m(hamiltonian, m=0, n_states=2)

    assert result.electronic_energies.shape == (2,)
    assert result.orbitals.shape == (2, *hamiltonian.fixed_m_shape(0))
    assert np.all(np.diff(result.electronic_energies) > 0.0)
    np.testing.assert_allclose(
        result.total_energies,
        result.electronic_energies + 0.5,
    )
    for energy, orbital in zip(result.electronic_energies, result.orbitals):
        assert hamiltonian.inner_product(orbital, orbital) == pytest.approx(1.0)
        residual = hamiltonian.apply_fixed_m(orbital, 0) - energy * orbital
        residual_norm = np.sqrt(
            hamiltonian.inner_product(residual, residual).real
        )
        assert residual_norm < 2.0e-11
    assert not result.electronic_energies.flags.writeable
    assert not result.orbitals.flags.writeable


def test_h2plus_energy_approaches_reference_with_angular_cutoff():
    reference_total_energy = -0.6026342144949465
    errors = []
    for l_max in (0, 2, 4):
        result = diagonalize_fixed_m(
            _hamiltonian(
                points=8,
                l_max=l_max,
                m_max=0,
                nuclear_L_max=2 * l_max,
            )
        )
        errors.append(abs(result.total_energy - reference_total_energy))

    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 1.0e-2
