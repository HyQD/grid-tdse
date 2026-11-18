import numpy as np

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus
from grid_tdse.spherical.one_radial_atomic.h2plus import (
    _C1BSplineRadialBasis,
)


def _global_values(basis, element, radius):
    values, derivatives, indices = basis.evaluate_element(
        element, np.array([radius])
    )
    global_values = np.zeros(basis.size)
    global_derivatives = np.zeros(basis.size)
    global_values[indices] = values[0]
    global_derivatives[indices] = derivatives[0]
    return global_values, global_derivatives


def test_c1_bspline_values_and_derivatives_match_at_element_boundaries():
    basis = _C1BSplineRadialBasis([0.0, 1.0, 2.0, 4.0], degree=5)

    for element, radius in enumerate((1.0, 2.0)):
        left = _global_values(basis, element, radius)
        right = _global_values(basis, element + 1, radius)
        np.testing.assert_allclose(left[0], right[0], atol=1.0e-14)
        np.testing.assert_allclose(left[1], right[1], atol=1.0e-13)


def test_c1_bspline_enforces_zero_endpoint_values():
    basis = _C1BSplineRadialBasis([0.0, 1.0, 2.0], degree=3)

    left_values = _global_values(basis, 0, 0.0)[0]
    right_values = _global_values(basis, 1, 2.0)[0]

    np.testing.assert_allclose(left_values, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(right_values, 0.0, atol=1.0e-15)


def test_atomic_h2plus_accepts_c1_bspline_radial_basis():
    result = solve_atomic_h2plus(
        radial_nodes=(0.0, 1.0, 2.0, 4.0),
        radial_n_points=4,
        radial_basis="bspline-c1",
        radial_spline_degree=3,
        l_max=1,
        integration_spacing=1.0,
        integration_radial_n_points=4,
        lebedev_order=9,
    )

    assert np.isfinite(result.electronic_energy)
    assert result.matrices.basis_size == 24


def test_atomic_h2plus_rejects_unknown_radial_basis():
    try:
        solve_atomic_h2plus(radial_basis="unknown")
    except ValueError as error:
        assert "radial_basis" in str(error)
    else:
        raise AssertionError("unknown radial basis was accepted")
