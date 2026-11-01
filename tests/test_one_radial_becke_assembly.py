import numpy as np
import pytest

from grid_tdse.spherical.one_radial_becke import (
    BeckeMulticenterQuadrature,
    BeckePartition,
    BeckeSphericalBasis,
    FEMDVRQuadratureSpec,
    assemble_becke_matrices,
    classify_centers,
    fixed_m_channels,
    full_lm_channels,
)


def _two_center_problem(*, lebedev_order=17, charges=(1.0, 1.0)):
    centers = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    basis_specification = FEMDVRQuadratureSpec([0.0, 3.0], [4])
    quadrature_specification = FEMDVRQuadratureSpec([0.0, 3.0], [8])
    basis = BeckeSphericalBasis(
        BeckePartition(centers, switch_iterations=2),
        basis_specification,
        fixed_m_channels(0, 0),
    )
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers, switch_iterations=4),
        quadrature_specification,
        lebedev_order,
    )
    return (
        centers,
        basis,
        quadrature,
        assemble_becke_matrices(basis, quadrature, charges),
    )


def test_geometry_classification_and_deterministic_collinear_frame():
    collinear = classify_centers(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.2]]
    )
    noncollinear = classify_centers(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.8, 0.0]]
    )

    assert collinear.classification == "collinear"
    assert collinear.affine_rank == 1
    np.testing.assert_allclose(collinear.axis, [0.0, 0.0, 1.0], atol=0.0)
    frame = np.asarray(collinear.frame)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=2e-15)
    np.testing.assert_allclose(frame[:, 2], collinear.axis, atol=0.0)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    assert noncollinear.classification == "noncollinear"
    assert noncollinear.affine_rank == 2
    assert noncollinear.axis is None


def test_geometry_tolerance_does_not_silently_discard_near_collinearity():
    centers = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0e-8, 0.0]]

    assert classify_centers(centers).classification == "noncollinear"
    assert (
        classify_centers(centers, tolerance=1.0e-7).classification
        == "collinear"
    )


def test_dense_assembly_is_hermitian_and_reports_complete_diagnostics():
    centers, basis, quadrature, matrices = _two_center_problem()

    for matrix in (
        matrices.overlap,
        matrices.kinetic,
        matrices.potential,
        matrices.hamiltonian,
    ):
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=0.0)
        assert not matrix.flags.writeable
    np.testing.assert_allclose(
        matrices.hamiltonian,
        matrices.kinetic + matrices.potential,
        atol=2e-15,
    )

    diagnostics = matrices.diagnostics
    assert diagnostics.geometry.classification == "collinear"
    assert diagnostics.angular_mode == "fixed_m"
    assert diagnostics.fixed_m == 0
    assert diagnostics.basis_size == basis.size
    assert diagnostics.quadrature_size == quadrature.point_count
    assert diagnostics.overlap_min_eigenvalue > 0.0
    assert diagnostics.overlap_max_eigenvalue > diagnostics.overlap_min_eigenvalue
    assert diagnostics.overlap_condition_number > 1.0
    for report in (
        diagnostics.overlap,
        diagnostics.kinetic,
        diagnostics.potential,
        diagnostics.hamiltonian,
    ):
        assert report.relative_antihermitian_norm < 1.0e-14
        assert report.same_center_norm > 0.0
        assert report.cross_center_norm > 0.0

    parameters = diagnostics.parameters
    assert parameters.nuclear_positions == tuple(map(tuple, centers))
    assert parameters.orbital_switch_iterations == 2
    assert parameters.integration_switch_iterations == 4
    assert parameters.basis_radial_n_points == ((4,), (4,))
    assert parameters.quadrature_radial_n_points == ((8,), (8,))
    assert parameters.lebedev_order == 17
    assert parameters.overlap_relative_cutoff is None


def test_selected_elements_match_direct_high_order_shell_sums():
    centers, basis, quadrature, matrices = _two_center_problem(
        lebedev_order=29, charges=(1.0, 2.0)
    )
    charges = np.array([1.0, 2.0])
    row, column = 0, 2
    direct_overlap = 0.0j
    direct_kinetic = 0.0j
    direct_potential = 0.0j
    incorrectly_partitioned_potential = 0.0j

    for shell in quadrature.iter_shells():
        evaluation = basis.evaluate_shell(shell)
        row_value = evaluation.values[:, row]
        column_value = evaluation.values[:, column]
        product = row_value.conj() * column_value
        distances = np.linalg.norm(
            shell.points[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=-1,
        )
        complete_potential = -np.sum(charges / distances, axis=1)
        orbital_weights = basis.partition.weights(shell.points)
        wrong_potential = -np.sum(
            orbital_weights * charges / distances, axis=1
        )

        direct_overlap += np.sum(shell.weights * product)
        direct_potential += np.sum(
            shell.weights * complete_potential * product
        )
        incorrectly_partitioned_potential += np.sum(
            shell.weights * wrong_potential * product
        )
        for component in range(3):
            direct_kinetic += 0.5 * np.sum(
                shell.weights
                * evaluation.gradients[:, row, component].conj()
                * evaluation.gradients[:, column, component]
            )

    assert matrices.overlap[row, column] == pytest.approx(
        direct_overlap, abs=2e-15
    )
    assert matrices.kinetic[row, column] == pytest.approx(
        direct_kinetic, abs=2e-15
    )
    assert matrices.potential[row, column] == pytest.approx(
        direct_potential, abs=2e-15
    )
    assert abs(direct_potential - incorrectly_partitioned_potential) > 1.0e-2


def test_small_matrices_converge_with_lebedev_refinement():
    results = [
        _two_center_problem(lebedev_order=order)[-1]
        for order in (9, 17, 29)
    ]
    coarse_difference = np.linalg.norm(
        results[1].hamiltonian - results[0].hamiltonian
    )
    fine_difference = np.linalg.norm(
        results[2].hamiltonian - results[1].hamiltonian
    )

    assert fine_difference < 0.2 * coarse_difference


def test_small_matrices_converge_with_radial_order_refinement():
    centers = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
    basis = BeckeSphericalBasis(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 3.0], [4]),
        fixed_m_channels(0, 0),
    )
    results = []
    for radial_order in (5, 8, 11):
        quadrature = BeckeMulticenterQuadrature(
            BeckePartition(centers),
            FEMDVRQuadratureSpec([0.0, 3.0], [radial_order]),
            29,
        )
        results.append(
            assemble_becke_matrices(basis, quadrature, [1.0, 1.0])
        )

    coarse_difference = np.linalg.norm(
        results[1].hamiltonian - results[0].hamiltonian
    )
    fine_difference = np.linalg.norm(
        results[2].hamiltonian - results[1].hamiltonian
    )

    assert fine_difference < 0.2 * coarse_difference


def test_three_collinear_centers_assemble_one_signed_m_block():
    centers = np.array(
        [[0.0, 0.0, -1.1], [0.0, 0.0, 0.1], [0.0, 0.0, 1.4]]
    )
    partition = BeckePartition(centers)
    basis_specification = FEMDVRQuadratureSpec([0.0, 2.0], [3])
    quadrature_specification = FEMDVRQuadratureSpec([0.0, 2.0], [4])
    basis = BeckeSphericalBasis(
        partition, basis_specification, fixed_m_channels(2, -1)
    )
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers), quadrature_specification, 17
    )

    matrices = assemble_becke_matrices(basis, quadrature, [1.0, 2.0, 1.0])

    assert matrices.diagnostics.geometry.classification == "collinear"
    assert matrices.diagnostics.angular_mode == "fixed_m"
    assert matrices.diagnostics.fixed_m == -1
    assert matrices.overlap.shape == (6, 6)


def test_noncollinear_assembly_requires_and_couples_full_lm_channels():
    centers = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.2], [0.2, 1.0, -0.3]]
    )
    partition = BeckePartition(centers)
    basis_specification = FEMDVRQuadratureSpec([0.0, 2.0], [3])
    quadrature_specification = FEMDVRQuadratureSpec([0.0, 2.0], [4])
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers), quadrature_specification, 17
    )
    basis = BeckeSphericalBasis(
        partition, basis_specification, full_lm_channels(1)
    )

    matrices = assemble_becke_matrices(basis, quadrature, [1.0, 1.0, 1.0])
    magnetic_numbers = np.array([function.m for function in basis.functions])
    cross_m = magnetic_numbers[:, np.newaxis] != magnetic_numbers[np.newaxis, :]

    assert matrices.diagnostics.geometry.classification == "noncollinear"
    assert matrices.diagnostics.angular_mode == "full_lm"
    assert np.linalg.norm(matrices.hamiltonian[cross_m]) > 1.0e-2

    incomplete_basis = BeckeSphericalBasis(
        partition, basis_specification, fixed_m_channels(1, 0)
    )
    with pytest.raises(ValueError, match="noncollinear assembly requires"):
        assemble_becke_matrices(
            incomplete_basis, quadrature, [1.0, 1.0, 1.0]
        )


def test_assembly_validates_center_order_element_boundaries_and_parameters():
    centers = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    basis = BeckeSphericalBasis(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 2.0], [3]),
        fixed_m_channels(0, 0),
    )
    reversed_quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers[::-1]),
        FEMDVRQuadratureSpec([0.0, 2.0], [4]),
        17,
    )
    with pytest.raises(ValueError, match="same ordered centers"):
        assemble_becke_matrices(basis, reversed_quadrature, [1.0, 1.0])

    incompatible_quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 1.0, 2.0], [4, 4]),
        17,
    )
    with pytest.raises(ValueError, match="matching finite-element boundaries"):
        assemble_becke_matrices(basis, incompatible_quadrature, [1.0, 1.0])

    compatible_quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 2.0], [4]),
        17,
    )
    with pytest.raises(ValueError, match="one value per center"):
        assemble_becke_matrices(basis, compatible_quadrature, [1.0])
    with pytest.raises(ValueError, match="finite and positive"):
        assemble_becke_matrices(basis, compatible_quadrature, [1.0, 0.0])
    with pytest.raises(ValueError, match="particle_mass"):
        assemble_becke_matrices(
            basis, compatible_quadrature, [1.0, 1.0], particle_mass=-1.0
        )


def test_fixed_m_frame_must_follow_the_collinear_axis():
    centers = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]
    specification = FEMDVRQuadratureSpec([0.0, 2.0], [3])
    basis = BeckeSphericalBasis(
        BeckePartition(centers), specification, fixed_m_channels(1, 0)
    )
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers), specification, 17
    )

    with pytest.raises(ValueError, match="z axis along"):
        assemble_becke_matrices(basis, quadrature, [1.0, 1.0])
