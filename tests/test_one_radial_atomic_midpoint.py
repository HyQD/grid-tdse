import numpy as np

from grid_tdse.spherical.one_radial_atomic import (
    solve_midpoint_h2plus_on_becke_grid,
)


def test_midpoint_basis_can_be_integrated_on_multicenter_becke_grid():
    result = solve_midpoint_h2plus_on_becke_grid(
        radial_nodes=(0.0, 1.0, 2.0),
        radial_n_points=3,
        l_max=0,
        integration_spacing=1.0,
        integration_radial_n_points=4,
        lebedev_order=9,
    )

    assert result.basis_size == 3
    assert result.quadrature_size > 0
    assert np.isfinite(result.structured_energy)
    assert np.isfinite(result.becke_energy)
    assert result.energy_difference == (
        result.becke_energy - result.structured_energy
    )
    for matrix in (result.overlap, result.kinetic, result.potential):
        assert matrix.shape == (result.basis_size, result.basis_size)
        assert not matrix.flags.writeable
