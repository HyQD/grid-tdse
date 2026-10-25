import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    FEMDVRGridSpec,
    RadialSincDVRGridSpec,
    TwoRadialSystem,
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


def test_direct_legacy_system_construction_defaults_to_the_full_product_basis():
    prepared = TwoRadialSystemSpec(
        coordinate1=_coordinate(RadialSincDVRGridSpec(4.0, 4), l_max=0),
        coordinate2=_coordinate(RadialSincDVRGridSpec(4.0, 4), l_max=0),
        multipoles=CoulombMultipoleSpec(0, 0),
        coulomb_terms=(CoulombTerm(1.0, 1.0),),
    ).build()

    system = TwoRadialSystem(
        prepared.grid1,
        prepared.grid2,
        prepared.basis1,
        prepared.basis2,
        prepared.h1,
        prepared.h2,
        prepared.multipoles,
        prepared.radial_multipoles,
        prepared.couplings,
    )

    assert system.M_tot is None
    assert not system.uses_compact_angular_storage
    assert system.psi_shape == prepared.psi_shape


def test_system_rejects_mixed_radial_grid_families():
    spec = TwoRadialSystemSpec(
        coordinate1=_coordinate(RadialSincDVRGridSpec(4.0, 4), l_max=0),
        coordinate2=_coordinate(FEMDVRGridSpec([0.0, 4.0], [5]), l_max=0),
        multipoles=CoulombMultipoleSpec(0, 0),
        coulomb_terms=(CoulombTerm(1.0, 1.0),),
    )

    with pytest.raises(ValueError, match="same grid family"):
        spec.build()


def _magnetic_sector_system(M_tot):
    grid = RadialSincDVRGridSpec(5.0, 5)
    basis = AngularBasisSpec(l_max=2, m_max=2)
    coordinate = CoordinateSpec(
        grid=grid,
        basis=basis,
        kinetic_prefactor=-0.5,
        one_body_terms=(radial_potential(lambda r: -2.0 / r),),
    )
    return TwoRadialSystemSpec(
        coordinate1=coordinate,
        coordinate2=coordinate,
        multipoles=CoulombMultipoleSpec(4, 4),
        coulomb_terms=(CoulombTerm(1.0, 1.0),),
        M_tot=M_tot,
    ).build()


def test_compact_M_tot_hamiltonian_matches_full_representation():
    full = _magnetic_sector_system(None)
    compact = _magnetic_sector_system(0)
    rng = np.random.default_rng(902)
    compact_state = rng.normal(size=compact.psi_shape) + 1j * rng.normal(
        size=compact.psi_shape
    )
    full_state = compact.unpack_state(compact_state)

    assert full.psi_shape == (9, 9, 5, 5)
    assert compact.psi_shape == (19, 5, 5)
    assert compact.full_psi_shape == full.psi_shape
    assert compact.couplings.n_pairs < full.couplings.n_pairs
    np.testing.assert_array_equal(compact.pack_state(full_state), compact_state)
    np.testing.assert_array_equal(compact.unpack_state(compact_state), full_state)

    compact_action = compact.hamiltonian()(compact_state)
    full_action = full.hamiltonian()(full_state)
    np.testing.assert_allclose(
        compact_action,
        compact.pack_state(full_action),
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        compact.inner_product(compact_state, compact_action),
        full.inner_product(full_state, full_action),
        atol=2.0e-12,
        rtol=2.0e-13,
    )
