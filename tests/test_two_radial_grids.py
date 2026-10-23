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


def test_mixed_grid_families_are_rejected():
    sinc = RadialSincDVRGridSpec(4.0, 4).build(l_max=0)
    fem = FEMDVRGridSpec([0.0, 4.0], [5]).build(l_max=0)

    with pytest.raises(ValueError, match="same grid family"):
        require_same_grid_family(sinc, fem)
