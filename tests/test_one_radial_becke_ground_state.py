import numpy as np
import pytest
from scipy.linalg import eigh

from grid_tdse.spherical.one_radial_becke import (
    BeckeMulticenterQuadrature,
    BeckePartition,
    BeckeSphericalBasis,
    FEMDVRQuadratureSpec,
    assemble_becke_matrices,
    canonical_orthogonalization,
    fixed_m_channels,
    solve_becke_eigenproblem,
    solve_generalized_hermitian,
)


def _random_hermitian_problem(size=5):
    random = np.random.default_rng(8417)
    vectors = random.normal(size=(size, size)) + 1j * random.normal(
        size=(size, size)
    )
    unitary, _ = np.linalg.qr(vectors)
    overlap_eigenvalues = np.linspace(0.4, 1.7, size)
    overlap = (unitary * overlap_eigenvalues) @ unitary.conj().T
    raw_hamiltonian = random.normal(size=(size, size)) + 1j * random.normal(
        size=(size, size)
    )
    hamiltonian = 0.5 * (raw_hamiltonian + raw_hamiltonian.conj().T)
    return hamiltonian, overlap


def _small_becke_matrices():
    centers = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
    basis = BeckeSphericalBasis(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 3.0], [4]),
        fixed_m_channels(0, 0),
    )
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 3.0], [8]),
        17,
    )
    return assemble_becke_matrices(basis, quadrature, [1.0, 1.0])


def test_canonical_orthogonalization_reports_strict_rank_decision():
    overlap = np.diag([1.0e-14, 1.0e-8, 0.2, 1.0])

    metric = canonical_orthogonalization(overlap, relative_cutoff=1.0e-8)

    np.testing.assert_allclose(
        metric.overlap_eigenvalues, [1.0e-14, 1.0e-8, 0.2, 1.0]
    )
    np.testing.assert_array_equal(metric.retained_indices, [2, 3])
    np.testing.assert_array_equal(metric.discarded_indices, [0, 1])
    assert metric.absolute_cutoff == pytest.approx(1.0e-8)
    assert metric.retained_rank == 2
    assert metric.discarded_rank == 2
    assert metric.hermiticity_tolerance == 1.0e-11
    assert metric.condition_number == pytest.approx(1.0e14)
    assert metric.retained_condition_number == pytest.approx(5.0)
    assert metric.orthonormality_error < 1.0e-14
    np.testing.assert_allclose(
        metric.transformation.conj().T @ overlap @ metric.transformation,
        np.eye(2),
        atol=2.0e-15,
    )


def test_full_rank_solution_matches_scipy_generalized_eigh():
    hamiltonian, overlap = _random_hermitian_problem()
    expected_energies, _ = eigh(hamiltonian, overlap)

    result = solve_generalized_hermitian(
        hamiltonian, overlap, relative_cutoff=1.0e-12
    )

    np.testing.assert_allclose(result.energies, expected_energies, atol=3.0e-14)
    assert result.metric.retained_rank == overlap.shape[0]
    assert result.metric.discarded_rank == 0
    assert np.isrealobj(result.energies)
    np.testing.assert_allclose(
        result.coefficients.conj().T @ overlap @ result.coefficients,
        np.eye(overlap.shape[0]),
        atol=3.0e-15,
    )
    assert np.max(result.diagnostics.residual_norms) < 2.0e-14
    assert np.max(result.diagnostics.relative_residual_norms) < 2.0e-14
    assert np.max(result.diagnostics.projected_residual_norms) < 3.0e-14


def test_requested_lowest_states_have_small_generalized_residuals():
    hamiltonian, overlap = _random_hermitian_problem(size=7)

    result = solve_generalized_hermitian(
        hamiltonian,
        overlap,
        relative_cutoff=1.0e-10,
        number_of_states=3,
    )

    assert result.energies.shape == (3,)
    assert result.coefficients.shape == (7, 3)
    residual = (
        hamiltonian @ result.coefficients
        - overlap @ result.coefficients * result.energies[np.newaxis, :]
    )
    np.testing.assert_allclose(
        np.linalg.norm(residual, axis=0),
        result.diagnostics.residual_norms,
        atol=2.0e-15,
    )
    assert result.diagnostics.orthonormality_error < 4.0e-15


def test_energies_are_stable_over_required_overlap_cutoff_study():
    overlap = np.diag([1.0e-14, 0.4, 1.0])
    hamiltonian = np.diag([7.0e-14, -0.4, 0.5])
    cutoffs = (1.0e-8, 1.0e-10, 1.0e-12)

    results = [
        solve_generalized_hermitian(
            hamiltonian, overlap, relative_cutoff=cutoff
        )
        for cutoff in cutoffs
    ]

    for cutoff, result in zip(cutoffs, results):
        assert result.metric.relative_cutoff == cutoff
        assert result.metric.retained_rank == 2
        assert result.metric.discarded_rank == 1
        np.testing.assert_allclose(result.energies, [-1.0, 0.5], atol=0.0)
        assert np.max(result.diagnostics.residual_norms) < 1.0e-15
    np.testing.assert_allclose(results[0].energies, results[-1].energies, atol=0.0)


def test_becke_assembly_solver_preserves_metric_normalization():
    matrices = _small_becke_matrices()
    results = [
        solve_becke_eigenproblem(
            matrices, relative_cutoff=cutoff, number_of_states=2
        )
        for cutoff in (1.0e-8, 1.0e-10, 1.0e-12)
    ]
    result = results[1]

    assert result.energies.shape == (2,)
    assert np.all(np.diff(result.energies) >= 0.0)
    assert result.metric.retained_rank == matrices.overlap.shape[0]
    for candidate in results:
        np.testing.assert_allclose(
            candidate.energies, result.energies, atol=2.0e-15
        )
    np.testing.assert_allclose(
        result.coefficients.conj().T
        @ matrices.overlap
        @ result.coefficients,
        np.eye(2),
        atol=2.0e-15,
    )
    assert np.max(result.diagnostics.relative_residual_norms) < 2.0e-15


def test_overlap_rejects_nonhermitian_and_indefinite_inputs():
    with pytest.raises(ValueError, match="anti-Hermitian tolerance"):
        canonical_orthogonalization([[1.0, 0.2], [0.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError, match="negative eigenvalue"):
        canonical_orthogonalization(np.diag([-1.0e-4, 1.0]))
    with pytest.raises(np.linalg.LinAlgError, match="no positive"):
        canonical_orthogonalization(-np.eye(2))


@pytest.mark.parametrize("relative_cutoff", [-1.0, 1.0, np.inf])
def test_overlap_cutoff_validation(relative_cutoff):
    with pytest.raises(ValueError, match="0 <= tau < 1"):
        canonical_orthogonalization(
            np.eye(2), relative_cutoff=relative_cutoff
        )


def test_eigensolver_validates_hamiltonian_and_state_count():
    with pytest.raises(ValueError, match="same shape"):
        solve_generalized_hermitian(np.eye(3), np.eye(2))
    with pytest.raises(ValueError, match="anti-Hermitian tolerance"):
        solve_generalized_hermitian(
            np.array([[1.0, 1.0], [0.0, 2.0]]), np.eye(2)
        )
    with pytest.raises(ValueError, match="between one"):
        solve_generalized_hermitian(
            np.eye(2), np.eye(2), number_of_states=3
        )
    with pytest.raises(TypeError, match="integer or None"):
        solve_generalized_hermitian(
            np.eye(2), np.eye(2), number_of_states=1.0
        )


def test_result_arrays_are_read_only():
    result = solve_generalized_hermitian(
        np.diag([-1.0, 0.5]), np.eye(2)
    )

    for array in (
        result.energies,
        result.coefficients,
        result.orthogonal_coefficients,
        result.metric.overlap_eigenvalues,
        result.metric.retained_indices,
        result.metric.discarded_indices,
        result.metric.transformation,
        result.diagnostics.residual_norms,
        result.diagnostics.relative_residual_norms,
        result.diagnostics.projected_residual_norms,
    ):
        assert not array.flags.writeable
