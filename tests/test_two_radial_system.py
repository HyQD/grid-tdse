import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    FEMDVRGridSpec,
    RadialSincDVRGridSpec,
    TwoRadialSystemSpec,
    radial_potential,
)


def _coordinate(grid, l_max=1):
    return CoordinateSpec(
        grid=grid,
        basis=AngularBasisSpec(l_max=l_max, m_max=0),
        kinetic_prefactor=-0.5,
        one_body_terms=(radial_potential(lambda r: -2.0 / r),),
    )


def test_sinc_system_uses_l_dependent_kinetic_blocks_and_analytic_coulomb():
    system = TwoRadialSystemSpec(
        coordinate1=_coordinate(RadialSincDVRGridSpec(6.0, 6)),
        coordinate2=_coordinate(RadialSincDVRGridSpec(4.0, 5)),
        multipoles=CoulombMultipoleSpec(2, 0),
        coulomb_terms=(CoulombTerm(1.0, 1.0),),
    ).build()

    assert system.psi_shape == (2, 2, 6, 5)
    assert system.radial_multipoles.representation == "analytic"
    assert not np.allclose(system.h1[0], system.h1[1])
    hamiltonian = system.hamiltonian()
    assert hamiltonian.psi_shape == system.psi_shape
    psi = np.ones(system.psi_shape, dtype=complex)
    result = hamiltonian(psi)
    assert result.shape == system.psi_shape
    assert np.isfinite(result).all()

    expected_norm_squared = (
        system.basis1.n_channels
        * system.basis2.n_channels
        * system.grid1.weights.sum()
        * system.grid2.weights.sum()
    )
    assert system.inner_product(psi, psi) == pytest.approx(expected_norm_squared)


def test_femdvr_system_uses_two_grid_poisson_coulomb():
    grid1 = FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4])
    grid2 = FEMDVRGridSpec([0.0, 1.5, 4.0], [4, 4])
    system = TwoRadialSystemSpec(
        coordinate1=_coordinate(grid1, l_max=0),
        coordinate2=_coordinate(grid2, l_max=0),
        multipoles=CoulombMultipoleSpec(0, 0),
        coulomb_terms=(CoulombTerm(1.0, 1.0),),
    ).build()

    assert system.radial_multipoles.representation == "femdvr-poisson"
    assert system.radial_multipoles.values.shape == (
        1,
        system.grid1.n_points,
        system.grid2.n_points,
    )
    assert np.isfinite(system.radial_multipoles.values).all()


def test_system_rejects_mixed_radial_grid_families():
    spec = TwoRadialSystemSpec(
        coordinate1=_coordinate(RadialSincDVRGridSpec(4.0, 4), l_max=0),
        coordinate2=_coordinate(FEMDVRGridSpec([0.0, 4.0], [5]), l_max=0),
        multipoles=CoulombMultipoleSpec(0, 0),
        coulomb_terms=(CoulombTerm(1.0, 1.0),),
    )

    with pytest.raises(ValueError, match="same grid family"):
        spec.build()
