import numpy as np
import pytest

pytest.importorskip("pyscf")

from grid_tdse.spherical.one_radial_atomic import (  # noqa: E402
    build_pyscf_becke_grid,
    solve_atomic_h2plus,
)


def test_radial_rule_owns_point_count_for_every_atom():
    radial_data = (
        (np.array([0.25, 1.0]), np.array([0.2, 0.4])),
        (np.array([0.2, 0.8, 2.0]), np.array([0.1, 0.3, 0.5])),
    )
    calls = []

    def radial_rule(atom_index, atomic_number):
        calls.append((atom_index, atomic_number))
        return radial_data[atom_index]

    grid = build_pyscf_becke_grid(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        [1, 1],
        radial_rule,
        angular_points=14,
        prune=None,
    )

    assert calls == [(0, 1), (1, 1)]
    assert grid.point_count == (2 + 3) * 14
    assert np.count_nonzero(grid.origin_atoms == 0) == 2 * 14
    assert np.count_nonzero(grid.origin_atoms == 1) == 3 * 14
    for array in (
        grid.coordinates,
        grid.weights,
        grid.origin_atoms,
        grid.quadrature_weights,
    ):
        assert not array.flags.writeable


def test_single_center_weights_preserve_custom_radial_quadrature():
    radii = np.array([0.2, 0.7, 1.5])
    radial_weights = np.array([0.1, 0.3, 0.6])
    grid = build_pyscf_becke_grid(
        [[0.0, 0.0, 0.0]],
        [1],
        lambda _atom_index, _atomic_number: (radii, radial_weights),
        angular_points=14,
        prune=None,
    )

    expected_volume = 4.0 * np.pi * np.dot(radii**2, radial_weights)
    np.testing.assert_allclose(np.sum(grid.weights), expected_volume, rtol=2e-15)
    np.testing.assert_allclose(grid.weights, grid.quadrature_weights, rtol=2e-15)


def test_pruned_pyscf_h2plus_grid_reaches_microhartree_accuracy():
    result = solve_atomic_h2plus(
        grid_backend="pyscf",
        integration_spacing=0.5,
        integration_radial_n_points=12,
        lebedev_order=47,
    )

    assert abs(result.electronic_energy - (-1.102634214)) < 1.0e-6
    assert result.matrices.quadrature_size < 300_000
