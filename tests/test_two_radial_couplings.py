import numpy as np
import pytest

from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    get_y,
    get_ybar,
)
from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoulombMultipoleSpec,
    RadialCoulombMultipoles,
    TwoRadialHamiltonian,
    build_coulomb_couplings,
)


@pytest.mark.parametrize("parity", [None, 1])
def test_sparse_coulomb_action_matches_dense_tensor_oracle(parity):
    basis1 = AngularBasisSpec(2, 2, coordinate_parity=parity)
    basis2 = AngularBasisSpec(2, 1, coordinate_parity=parity)
    multipoles = CoulombMultipoleSpec(3, 2)
    couplings = build_coulomb_couplings(basis1, basis2, multipoles)

    rng = np.random.default_rng(7)
    radial = rng.normal(size=(len(multipoles.lambdas), 2, 3))
    psi = rng.normal(size=(basis1.n_channels, basis2.n_channels, 2, 3))
    psi = psi + 1j * rng.normal(size=psi.shape)
    h1 = np.zeros((basis1.n_channels, 2, 2))
    h2 = np.zeros((basis2.n_channels, 3, 3))
    radial_data = RadialCoulombMultipoles(radial, multipoles.lambdas, "test")

    actual = TwoRadialHamiltonian(h1, h2, radial_data, couplings).apply_coulomb(psi)

    dense_ybar = get_ybar(basis1.l_max, basis1.m_max, 3, 2)
    dense_y = get_y(basis2.l_max, basis2.m_max, 3, 2)
    full_states1 = AngularBasisSpec(2, 2).states
    full_states2 = AngularBasisSpec(2, 1).states
    indices1 = [full_states1.index(state) for state in basis1.states]
    indices2 = [full_states2.index(state) for state in basis2.states]
    expected = np.zeros_like(psi)
    for lambda_, mu in multipoles.components:
        lm_index = LM_to_I(lambda_, mu, 3, 2)
        ybar = dense_ybar[lm_index][np.ix_(indices1, indices1)]
        y = dense_y[lm_index][np.ix_(indices2, indices2)]
        expected += np.einsum(
            "ik,jl,ab,klab->ijab",
            ybar,
            y,
            radial[multipoles.radial_index(lambda_)],
            psi,
            optimize=True,
        )

    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert couplings.n_pairs < dense_ybar.size * dense_y.size
