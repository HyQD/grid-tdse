import numpy as np
import pytest

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
)


@pytest.fixture
def hamiltonian():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        particle_mass=1.0,
        name="H2+",
    )
    return ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 4.0],
        xi_points_per_element=5,
        eta_degree=5,
        m_max=1,
    )


@pytest.mark.parametrize("n_states", [1, 2])
def test_davidson_matches_tiny_direct_oracle(hamiltonian, n_states):
    direct = diagonalize_fixed_m(
        hamiltonian, n_states=n_states, solver="direct"
    )
    davidson = diagonalize_fixed_m(
        hamiltonian,
        n_states=n_states,
        residual_tolerance=1.0e-9,
        max_iterations=100,
        maximum_subspace_dimension=20,
    )

    assert davidson.converged
    assert np.allclose(
        davidson.electronic_energies,
        direct.electronic_energies,
        atol=2.0e-9,
    )
    assert np.all(davidson.residual_norms < 1.0e-9)
    assert davidson.total_energy == pytest.approx(
        davidson.electronic_energy + 0.5
    )
    for orbital in davidson.orbitals:
        assert hamiltonian.inner_product(orbital, orbital).real == (
            pytest.approx(1.0, abs=2.0e-12)
        )


@pytest.mark.parametrize(
    ("m", "eta_parity", "expected_inversion"),
    [
        (0, 1, 1),
        (0, -1, -1),
        (1, 1, -1),
        (1, -1, 1),
    ],
)
def test_inversion_combines_eta_reflection_and_azimuthal_phase(
    hamiltonian, m, eta_parity, expected_inversion
):
    rng = np.random.default_rng(7129 + m + eta_parity)
    state = rng.standard_normal(hamiltonian.fixed_m_shape(m))
    state = state + eta_parity * state[:, ::-1]

    inverted = hamiltonian.invert(state, m)

    assert inverted == pytest.approx(expected_inversion * state)
    assert hamiltonian.inversion_expectation(state, m) == pytest.approx(
        expected_inversion, abs=2.0e-14
    )


def test_inversion_expectation_supports_state_batches(hamiltonian):
    rng = np.random.default_rng(90210)
    state = rng.standard_normal(hamiltonian.fixed_m_shape(0))
    even = state + state[:, ::-1]
    odd = state - state[:, ::-1]

    expectations = hamiltonian.inversion_expectation(
        np.stack((even, odd)), m=0
    )

    assert expectations == pytest.approx([1.0, -1.0], abs=2.0e-14)
