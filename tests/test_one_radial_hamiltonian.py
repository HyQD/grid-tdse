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
