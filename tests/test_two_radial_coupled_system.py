import json

import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    CoupledAngularBasisSpec,
    CoupledTwoRadialSystem,
    FEMDVRGridSpec,
    TwoRadialRepresentation,
    TwoRadialSystemSpec,
    radial_potential,
)


def _system_spec(*, M_tot=0):
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=FEMDVRGridSpec([0.0, 1.5, 4.0], [4, 5]),
            basis=AngularBasisSpec(1),
            kinetic_prefactor=-0.5,
            one_body_terms=(radial_potential(lambda r: -2.0 / r),),
        ),
        coordinate2=CoordinateSpec(
            grid=FEMDVRGridSpec([0.0, 1.0, 3.0], [5, 4]),
            basis=AngularBasisSpec(1),
            kinetic_prefactor=-0.002,
            one_body_terms=(radial_potential(lambda r: 1.0 / r),),
        ),
        multipoles=CoulombMultipoleSpec(2, 2),
        coulomb_terms=(CoulombTerm(0.5, -1.0), CoulombTerm(-0.5, -1.0)),
        M_tot=M_tot,
    )


def _coupled_system():
    return _system_spec().build_coupled(
        CoupledAngularBasisSpec(
            1,
            1,
            2,
            M_tot=0,
            inversion_parity=1,
        )
    )


def test_build_coupled_prepares_radial_data_without_hamiltonian_arrays():
    specification = _system_spec()
    system = _coupled_system()
    uncoupled = specification.build(radial_operator_mode="element-local")

    assert isinstance(system, CoupledTwoRadialSystem)
    assert isinstance(system, TwoRadialRepresentation)
    assert system.M_tot == 0
    assert system.radial_shape == (
        system.grid1.n_points,
        system.grid2.n_points,
    )
    assert system.psi_shape == (
        system.angular_basis.n_channels,
        *system.radial_shape,
    )
    np.testing.assert_allclose(
        system.radial_multipoles.values,
        uncoupled.radial_multipoles.values,
    )
    assert system.radial_multipoles.representation == "femdvr-poisson"
    for forbidden in ("h1", "h2", "couplings", "angular_transform"):
        assert not hasattr(system, forbidden)


def test_coupled_state_allocation_validation_copy_and_norm():
    system = _coupled_system()
    zeros = system.zeros()

    assert zeros.shape == system.psi_shape
    assert zeros.dtype == np.complex128
    assert system.norm(zeros) == 0.0

    state = system.random_state(
        np.random.default_rng(2026),
        dtype=np.complex128,
        normalize=True,
    )
    assert system.norm(state) == pytest.approx(1.0, abs=3.0e-15)
    copied = system.copy_state(state)
    assert copied.flags.c_contiguous
    assert not np.shares_memory(copied, state)
    np.testing.assert_array_equal(copied, state)

    real = system.random_state(np.random.default_rng(7), dtype=np.float64)
    assert real.dtype == np.float64
    with pytest.raises(TypeError, match="real or complex floating"):
        system.random_state(dtype=np.int64)
    with pytest.raises(ValueError, match="expected"):
        system.validate_state(np.zeros((1, 2)))
    with pytest.raises(TypeError, match="numerical"):
        system.validate_state(np.full(system.psi_shape, "x"))


def test_inner_product_and_radial_marginals_use_both_quadratures():
    system = _coupled_system()
    state = system.random_state(np.random.default_rng(81))
    density1, density2 = system.radial_marginals(state)
    norm_squared = system.inner_product(state, state).real

    assert density1.shape == (system.grid1.n_points,)
    assert density2.shape == (system.grid2.n_points,)
    assert np.dot(density1, system.grid1.weights) == pytest.approx(norm_squared)
    assert np.dot(density2, system.grid2.weights) == pytest.approx(norm_squared)


def test_metadata_and_storage_estimate_are_deterministic_and_json_compatible():
    system = _coupled_system()
    metadata = system.to_metadata()

    assert metadata["type"] == "coupled-two-radial-system"
    assert metadata["angular_basis"] == system.angular_basis.to_metadata()
    assert metadata["psi_shape"] == list(system.psi_shape)
    assert metadata["radial_shape"] == list(system.radial_shape)
    assert json.loads(json.dumps(metadata)) == metadata
    assert system.state_nbytes() == np.prod(system.psi_shape) * 16
    assert system.state_nbytes(np.float64) == np.prod(system.psi_shape) * 8


@pytest.mark.parametrize(
    ("basis", "message"),
    [
        (CoupledAngularBasisSpec(0, 1, 1), "coordinate1 l_max"),
        (CoupledAngularBasisSpec(1, 0, 1), "coordinate2 l_max"),
        (CoupledAngularBasisSpec(1, 1, 2, M_tot=1), "M_tot"),
    ],
)
def test_build_coupled_rejects_inconsistent_angular_bases(basis, message):
    with pytest.raises(ValueError, match=message):
        _system_spec().build_coupled(basis)


def test_build_coupled_accepts_unspecified_recipe_M_tot():
    system = _system_spec(M_tot=None).build_coupled(
        CoupledAngularBasisSpec(1, 1, 2, M_tot=-1)
    )

    assert system.M_tot == -1


def test_build_coupled_requires_a_coupled_basis():
    with pytest.raises(TypeError, match="CoupledAngularBasisSpec"):
        _system_spec().build_coupled(object())
