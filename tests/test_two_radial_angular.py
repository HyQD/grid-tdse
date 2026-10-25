import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    AngularPairBasis,
    CoulombMultipoleSpec,
)


def test_angular_basis_has_explicit_ordered_states():
    basis = AngularBasisSpec(l_max=2, m_max=1)

    assert basis.states == (
        (1, -1),
        (2, -1),
        (0, 0),
        (1, 0),
        (2, 0),
        (1, 1),
        (2, 1),
    )
    assert basis.n_channels == 7
    assert basis.index(2, 0) == 4


def test_coordinate_parity_reduces_basis_without_changing_l_cutoff():
    even = AngularBasisSpec(l_max=4, m_max=2, coordinate_parity=1)
    odd = AngularBasisSpec(l_max=4, m_max=2, coordinate_parity=-1)

    assert all(l % 2 == 0 for l, _ in even.states)
    assert all(l % 2 == 1 for l, _ in odd.states)
    assert even.l_max == odd.l_max == 4


def test_coulomb_multipoles_use_lambda_mu_names_and_filter_parity():
    spec = CoulombMultipoleSpec(lambda_max=4, mu_max=2, lambda_parity=1)

    assert spec.lambdas == (0, 2, 4)
    assert (2, -2) in spec.components
    assert (1, 0) not in spec.components
    assert spec.radial_index(4) == 2


def test_invalid_angular_limits_are_rejected():
    with pytest.raises(ValueError, match="m_max"):
        AngularBasisSpec(l_max=1, m_max=2)
    with pytest.raises(ValueError, match="mu_max"):
        CoulombMultipoleSpec(lambda_max=1, mu_max=2)
    with pytest.raises(ValueError, match="no basis states"):
        AngularBasisSpec(l_max=0, coordinate_parity=-1)
    with pytest.raises(ValueError, match="no Coulomb multipoles"):
        CoulombMultipoleSpec(lambda_max=0, lambda_parity=-1)


def test_angular_pair_basis_compacts_a_fixed_M_tot_sector():
    basis = AngularBasisSpec(l_max=2, m_max=2)

    full = AngularPairBasis(basis, basis)
    compact = AngularPairBasis(basis, basis, M_tot=0)

    assert full.n_pairs == basis.n_channels**2 == 81
    assert not full.is_compact
    assert compact.n_pairs == 19
    assert compact.is_compact
    assert all(
        basis.states[i][1] + basis.states[j][1] == 0
        for i, j in compact.pairs
    )
    assert compact.get(basis.index(1, 1), basis.index(1, -1)) is not None
    assert compact.get(basis.index(1, 1), basis.index(1, 0)) is None


def test_invalid_M_tot_selections_are_rejected():
    basis = AngularBasisSpec(l_max=1, m_max=1)

    with pytest.raises(TypeError, match="integer or None"):
        AngularPairBasis(basis, basis, M_tot=True)
    with pytest.raises(ValueError, match="no angular-channel pairs"):
        AngularPairBasis(basis, basis, M_tot=3)
