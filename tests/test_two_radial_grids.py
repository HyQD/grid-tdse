import numpy as np
import pytest

from grid_tdse.spherical.two_radial import FEMDVRGridSpec, RadialSincDVRGridSpec
from grid_tdse.spherical.two_radial.grids import require_same_grid_family


def test_radial_sinc_grid_builds_l_adapted_derivatives():
    grid = RadialSincDVRGridSpec(r_max=8.0, n_points=8).build(l_max=2)

    assert grid.family == "sinc"
    assert grid.r.shape == (8,)
    assert grid.r[0] > 0.0
    assert not np.allclose(grid.second_derivative(0), grid.second_derivative(1))


def test_femdvr_grid_uses_inner_points_and_common_derivatives():
    grid = FEMDVRGridSpec(nodes=[0.0, 2.0, 5.0], n_points=[4, 5]).build(l_max=2)

    assert grid.family == "femdvr"
    assert np.all(grid.r > 0.0)
    assert np.all(grid.r < 5.0)
    assert grid.second_derivative(0).shape == (grid.n_points, grid.n_points)
    assert np.array_equal(grid.second_derivative(0), grid.second_derivative(2))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_femdvr_element_local_second_derivative_matches_global_matrix(axis):
    grid = FEMDVRGridSpec(nodes=[0.0, 0.7, 2.0, 5.0], n_points=[4, 5, 6]).build(
        l_max=2
    )
    rng = np.random.default_rng(981)
    shape = [2, 3, 4]
    shape[axis] = grid.n_points
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    moved = np.moveaxis(values, axis, -1)
    expected = np.einsum("ij,...j->...i", grid.second_derivative(0), moved)
    expected = np.moveaxis(expected, -1, axis)

    np.testing.assert_allclose(
        grid.apply_local_second_derivative(values, axis=axis),
        expected,
        atol=3.0e-13,
        rtol=3.0e-13,
    )


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_femdvr_element_local_first_derivative_matches_global_matrix(axis):
    grid = FEMDVRGridSpec(nodes=[0.0, 0.7, 2.0, 5.0], n_points=[4, 5, 6]).build(
        l_max=2
    )
    rng = np.random.default_rng(732)
    shape = [2, 3, 4]
    shape[axis] = grid.n_points
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    moved = np.moveaxis(values, axis, -1)
    expected = np.einsum("ij,...j->...i", grid.first_derivative(0), moved)
    expected = np.moveaxis(expected, -1, axis)

    np.testing.assert_allclose(
        grid.apply_local_first_derivative(values, axis=axis),
        expected,
        atol=3.0e-13,
        rtol=3.0e-13,
    )


def test_sinc_grid_rejects_element_local_second_derivative():
    grid = RadialSincDVRGridSpec(r_max=8.0, n_points=8).build(l_max=0)

    with pytest.raises(ValueError, match="require a FEM-DVR"):
        grid.apply_local_second_derivative(np.ones(grid.n_points))


def test_mixed_grid_families_are_rejected():
    sinc = RadialSincDVRGridSpec(4.0, 4).build(l_max=0)
    fem = FEMDVRGridSpec([0.0, 4.0], [5]).build(l_max=0)

    with pytest.raises(ValueError, match="same grid family"):
        require_same_grid_family(sinc, fem)
