import pytest

from grid_tdse.spherical.two_radial import AngularBasisSpec, CoulombMultipoleSpec


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
