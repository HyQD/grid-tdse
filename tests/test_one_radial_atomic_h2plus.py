import numpy as np

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


def test_small_atomic_h2plus_problem_is_hermitian_and_normalized():
    result = solve_atomic_h2plus(
        radial_nodes=(0.0, 2.0, 4.0),
        radial_n_points=3,
        l_max=1,
        integration_spacing=2.0,
        integration_radial_n_points=3,
        lebedev_order=5,
    )

    matrices = result.matrices
    for matrix in (
        matrices.overlap,
        matrices.kinetic,
        matrices.potential,
        matrices.hamiltonian,
    ):
        np.testing.assert_allclose(matrix, matrix.T, atol=0.0)
        assert not matrix.flags.writeable
    coefficients = result.solution.coefficients
    np.testing.assert_allclose(
        coefficients.T @ matrices.overlap @ coefficients,
        np.eye(1),
        atol=2.0e-13,
    )
    assert matrices.basis_size == 12
    assert matrices.element_group_count > 1
    np.testing.assert_allclose(
        result.electronic_energy,
        -1.0390893805829138,
        atol=2.0e-13,
    )


def test_default_atomic_h2plus_reaches_microhartree_accuracy():
    result = solve_atomic_h2plus()

    assert abs(result.electronic_energy - (-1.102634214)) < 1.0e-6
    assert result.matrices.basis_size == 246
    assert result.solution.metric.retained_rank == 246
    assert result.matrices.overlap_condition_number < 5.0e7
