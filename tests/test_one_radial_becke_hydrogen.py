import numpy as np

from grid_tdse.spherical.one_radial_becke import (
    BeckeMulticenterQuadrature,
    BeckePartition,
    BeckeSphericalBasis,
    FEMDVRQuadratureSpec,
    assemble_becke_matrices,
    fixed_m_channels,
    full_lm_channels,
    solve_becke_eigenproblem,
)


RADIAL_NODES = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def _hydrogen_channel(ell, radial_order, *, number_of_states=1):
    n_elements = len(RADIAL_NODES) - 1
    basis = BeckeSphericalBasis(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec(
            RADIAL_NODES, [radial_order] * n_elements
        ),
        fixed_m_channels(ell, ell),
    )
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec(
            RADIAL_NODES, [2 * radial_order] * n_elements
        ),
        17,
    )
    matrices = assemble_becke_matrices(basis, quadrature, [1.0])
    solution = solve_becke_eigenproblem(
        matrices,
        relative_cutoff=1.0e-12,
        number_of_states=number_of_states,
    )
    return basis, matrices, solution


def test_single_center_orbital_and_integration_partitions_are_unity():
    points = np.array(
        [[0.2, -0.3, 0.4], [-1.1, 0.7, 2.3], [4.0, -2.0, 0.5]]
    )
    orbital_partition = BeckePartition([[0.0, 0.0, 0.0]])
    integration_partition = BeckePartition([[0.0, 0.0, 0.0]])

    for partition in (orbital_partition, integration_partition):
        evaluation = partition.evaluate(points)
        np.testing.assert_array_equal(evaluation.weights, np.ones((3, 1)))
        np.testing.assert_array_equal(
            evaluation.weight_gradients, np.zeros((3, 1, 3))
        )


def test_hydrogen_s_states_converge_with_radial_order():
    exact_energies = np.array([-0.5, -0.125])
    energy_errors = []
    final_matrices = None
    final_solution = None

    for radial_order in (4, 5, 6):
        _, matrices, solution = _hydrogen_channel(
            0, radial_order, number_of_states=2
        )
        energy_errors.append(np.abs(solution.energies - exact_energies))
        final_matrices = matrices
        final_solution = solution

    energy_errors = np.asarray(energy_errors)
    assert np.all(np.diff(energy_errors[:, 0]) < 0.0)
    assert np.all(np.diff(energy_errors[:, 1]) < 0.0)
    assert energy_errors[-1, 0] < 5.0e-8
    assert energy_errors[-1, 1] < 1.0e-7
    assert final_solution.metric.discarded_rank == 0
    np.testing.assert_allclose(
        final_solution.coefficients.conj().T
        @ final_matrices.overlap
        @ final_solution.coefficients,
        np.eye(2),
        atol=2.0e-14,
    )
    assert final_solution.diagnostics.orthonormality_error < 2.0e-14
    assert np.max(final_solution.diagnostics.relative_residual_norms) < 1.0e-12


def test_lowest_s_p_d_f_channels_have_hydrogenic_energies_and_positive_kinetic():
    for ell in range(4):
        _, matrices, solution = _hydrogen_channel(ell, 6)
        exact_energy = -0.5 / (ell + 1) ** 2

        assert abs(solution.energies[0] - exact_energy) < 1.0e-7
        assert np.linalg.eigvalsh(matrices.kinetic)[0] > -1.0e-12
        assert solution.diagnostics.orthonormality_error < 2.0e-14
        assert solution.diagnostics.relative_residual_norms[0] < 1.0e-11


def test_centered_hydrogen_matrices_obey_angular_selection_rules():
    basis = BeckeSphericalBasis(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec([0.0, 1.0, 3.0, 7.0], [4, 4, 4]),
        full_lm_channels(2),
    )
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec([0.0, 1.0, 3.0, 7.0], [8, 8, 8]),
        17,
    )
    matrices = assemble_becke_matrices(basis, quadrature, [1.0])
    channels = np.asarray(
        [(function.ell, function.m) for function in basis.functions]
    )
    off_channel = np.any(
        channels[:, np.newaxis, :] != channels[np.newaxis, :, :], axis=-1
    )

    assert matrices.diagnostics.geometry.classification == "single_center"
    assert matrices.diagnostics.angular_mode == "full_lm"
    assert matrices.diagnostics.overlap.cross_center_norm == 0.0
    for matrix in (
        matrices.overlap,
        matrices.kinetic,
        matrices.potential,
        matrices.hamiltonian,
    ):
        assert np.linalg.norm(matrix[off_channel]) < (
            5.0e-13 * np.linalg.norm(matrix)
        )

    for ell in range(3):
        multiplet_indices = [
            np.asarray(
                [
                    index
                    for index, function in enumerate(basis.functions)
                    if function.ell == ell and function.m == magnetic_number
                ]
            )
            for magnetic_number in range(-ell, ell + 1)
        ]
        for matrix in (
            matrices.overlap,
            matrices.kinetic,
            matrices.potential,
            matrices.hamiltonian,
        ):
            reference = matrix[
                np.ix_(multiplet_indices[0], multiplet_indices[0])
            ]
            for indices in multiplet_indices[1:]:
                np.testing.assert_allclose(
                    matrix[np.ix_(indices, indices)],
                    reference,
                    atol=2.0e-14,
                    rtol=2.0e-14,
                )
