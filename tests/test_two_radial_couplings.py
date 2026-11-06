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


@pytest.mark.parametrize("parity", [None, 1])
def test_coulomb_R_slice_matches_selected_full_action(parity):
    basis1 = AngularBasisSpec(2, 2, coordinate_parity=parity)
    basis2 = AngularBasisSpec(2, 1, coordinate_parity=parity)
    multipoles = CoulombMultipoleSpec(3, 2)
    couplings = build_coulomb_couplings(basis1, basis2, multipoles)
    rng = np.random.default_rng(71)
    radial = rng.normal(size=(len(multipoles.lambdas), 3, 5))
    psi = rng.normal(size=(basis1.n_channels, basis2.n_channels, 3, 5))
    psi = psi + 1j * rng.normal(size=psi.shape)
    hamiltonian = TwoRadialHamiltonian(
        np.zeros((basis1.n_channels, 3, 3)),
        np.zeros((basis2.n_channels, 5, 5)),
        RadialCoulombMultipoles(radial, multipoles.lambdas, "test"),
        couplings,
    )
    indices = np.asarray([0, 2, 4])

    expected = hamiltonian.apply_coulomb(psi)[..., indices]
    actual = hamiltonian.apply_coulomb_R_slice(psi[..., indices], indices)

    np.testing.assert_allclose(actual, expected, atol=1.0e-13, rtol=1.0e-13)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("tile_size", [1, 4, 64])
def test_multipole_grouped_coulomb_matches_reference(dtype, tile_size):
    basis1 = AngularBasisSpec(2, 2)
    basis2 = AngularBasisSpec(2, 1)
    multipoles = CoulombMultipoleSpec(3, 2)
    couplings = build_coulomb_couplings(basis1, basis2, multipoles)
    rng = np.random.default_rng(1729)
    radial = rng.normal(size=(len(multipoles.lambdas), 2, 3))
    psi = rng.normal(size=(basis1.n_channels, basis2.n_channels, 2, 3))
    if np.issubdtype(dtype, np.complexfloating):
        psi = psi + 1j * rng.normal(size=psi.shape)
    psi = psi.astype(dtype)
    radial_data = RadialCoulombMultipoles(radial, multipoles.lambdas, "test")
    arguments = (
        np.zeros((basis1.n_channels, 2, 2)),
        np.zeros((basis2.n_channels, 3, 3)),
        radial_data,
        couplings,
    )
    reference = TwoRadialHamiltonian(
        *arguments,
        tile_size=tile_size,
        coulomb_kernel="reference",
    )
    candidate = TwoRadialHamiltonian(
        *arguments,
        tile_size=tile_size,
        coulomb_kernel="multipole-grouped",
    )
    initial = np.full(psi.shape, 0.125, dtype=dtype)

    expected = reference.apply_coulomb(psi, out=initial.copy())
    actual = candidate.apply_coulomb(psi, out=initial.copy())

    np.testing.assert_allclose(actual, expected, atol=2.0e-13, rtol=2.0e-13)


def test_rejects_unknown_coulomb_kernel():
    basis = AngularBasisSpec(0, 0)
    multipoles = CoulombMultipoleSpec(0, 0)
    couplings = build_coulomb_couplings(basis, basis, multipoles)
    radial = RadialCoulombMultipoles(
        np.ones((1, 2, 2)),
        multipoles.lambdas,
        "test",
    )

    with pytest.raises(ValueError, match="coulomb_kernel"):
        TwoRadialHamiltonian(
            np.zeros((1, 2, 2)),
            np.zeros((1, 2, 2)),
            radial,
            couplings,
            coulomb_kernel="unknown",
        )
