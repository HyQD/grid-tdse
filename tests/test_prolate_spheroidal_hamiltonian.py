import numpy as np
import pytest

from grid_lib.prolate_spheroidal_coordinates import kinetic_energy_matrix
from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalOneElectronHamiltonian,
)


@pytest.fixture
def hamiltonian():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.5,
        nuclear_charge2=0.75,
        half_bond_length=1.0,
        particle_mass=1.25,
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
def test_weak_strong_and_symmetric_actions_agree_with_matrices(
    hamiltonian, m
):
    generator = np.random.default_rng(22 + m)
    state = generator.normal(size=hamiltonian.fixed_m_shape(m))
    flat = state.reshape(-1)

    cases = [
        ("weak", hamiltonian.apply_weak_fixed_m),
        ("strong", hamiltonian.apply_fixed_m),
        ("symmetric", hamiltonian.apply_symmetric_fixed_m),
    ]
    for representation, action in cases:
        expected = hamiltonian.fixed_m_matrix(
            m, representation=representation
        ) @ flat
        assert np.allclose(action(state, m).reshape(-1), expected)


@pytest.mark.parametrize("m", [0, 1, -1])
def test_metric_and_euclidean_hermiticity(hamiltonian, m):
    weak = hamiltonian.fixed_m_matrix(m, representation="weak")
    symmetric = hamiltonian.fixed_m_matrix(
        m, representation="symmetric"
    )

    assert np.allclose(weak.toarray(), weak.toarray().conj().T, atol=1e-12)
    assert np.allclose(
        symmetric.toarray(), symmetric.toarray().conj().T, atol=1e-12
    )

    generator = np.random.default_rng(91 + m)
    left = generator.normal(size=hamiltonian.fixed_m_shape(m))
    left = left + 1j * generator.normal(size=left.shape)
    right = generator.normal(size=hamiltonian.fixed_m_shape(m))
    right = right + 1j * generator.normal(size=right.shape)
    lhs = hamiltonian.inner_product(
        left, hamiltonian.apply_fixed_m(right, m), m
    )
    rhs = hamiltonian.inner_product(
        hamiltonian.apply_fixed_m(left, m), right, m
    )
    assert lhs == pytest.approx(rhs, abs=2e-11)


def test_local_coulomb_weak_diagonal_matches_direct_quadrature(hamiltonian):
    m = 0
    indices = hamiltonian.representation.active_indices(m)
    kinetic = kinetic_energy_matrix(
        hamiltonian.grid,
        m=m,
        active_indices=indices,
    ) / hamiltonian.system.particle_mass
    weak = hamiltonian.fixed_m_matrix(m, representation="weak")
    local_diagonal = (weak - kinetic).diagonal()

    a = hamiltonian.system.half_bond_length
    xi = hamiltonian.grid.xi[:, None]
    eta = hamiltonian.grid.eta[None, :]
    potential = (
        -hamiltonian.system.nuclear_charge1 / (a * (xi + eta))
        - hamiltonian.system.nuclear_charge2 / (a * (xi - eta))
    ).reshape(-1)[indices]
    expected = hamiltonian.fixed_m_metric_weights(m) * potential

    assert np.allclose(local_diagonal, expected, rtol=2e-14, atol=2e-14)
