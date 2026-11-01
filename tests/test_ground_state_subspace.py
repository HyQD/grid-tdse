import numpy as np
import pytest

from grid_tdse.ground_state import (
    GroundStateEigenIteration,
    GroundStateEigenResult,
    basis_orthogonality_error,
    basis_overlaps,
    extend_projected_hermitian_matrix,
    normalize,
    orthonormalize,
    phase_align,
    phase_aligned_distance,
    physical_norm,
    projected_hermitian_matrix,
    rayleigh_ritz,
)


def _weighted_inner_product(weights):
    weights = np.asarray(weights)

    def inner_product(left, right):
        return np.vdot(left, weights * right)

    return inner_product


def _weighted_problem():
    weights = np.array([1.5, 2.0, 3.0, 4.0])
    transformed_hamiltonian = np.array(
        [
            [0.9, 0.2 - 0.1j, 0.0, 0.1],
            [0.2 + 0.1j, 1.7, -0.3j, 0.0],
            [0.0, 0.3j, 2.4, 0.25],
            [0.1, 0.0, 0.25, 3.1],
        ],
        dtype=complex,
    )
    square_root = np.sqrt(weights)
    hamiltonian = (
        transformed_hamiltonian * square_root[np.newaxis, :]
    ) / square_root[:, np.newaxis]
    inner_product = _weighted_inner_product(weights)
    return transformed_hamiltonian, hamiltonian, weights, inner_product


def test_physical_norm_and_normalize_use_weighted_inner_product_without_mutation():
    weights = np.array([[2.0, 3.0], [4.0, 5.0]])
    inner_product = _weighted_inner_product(weights)
    vector = np.array([[1.0 + 2.0j, -0.5j], [0.25, -2.0]])
    original = vector.copy()

    expected_norm = np.sqrt(np.sum(weights * abs(vector) ** 2))
    assert physical_norm(vector, inner_product) == pytest.approx(expected_norm)

    normalized = normalize(vector, inner_product)

    np.testing.assert_array_equal(vector, original)
    assert physical_norm(normalized, inner_product) == pytest.approx(1.0)


def test_basis_overlaps_uses_one_batched_call():
    basis = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    vector = np.array([2.0 + 1.0j, -3.0])
    calls = []

    def inner_products(left, right):
        calls.append((len(left), right.copy()))
        return np.asarray([np.vdot(item, right) for item in left])

    overlaps = basis_overlaps(
        basis,
        vector,
        np.vdot,
        inner_products=inner_products,
    )

    np.testing.assert_allclose(overlaps, vector)
    assert len(calls) == 1
    assert calls[0][0] == len(basis)


def test_orthonormalize_returns_weighted_orthogonal_unit_direction():
    weights = np.array([2.0, 3.0, 5.0])
    inner_product = _weighted_inner_product(weights)
    basis = (
        np.array([1.0 / np.sqrt(weights[0]), 0.0, 0.0]),
        np.array([0.0, 1.0 / np.sqrt(weights[1]), 0.0]),
    )
    vector = np.array([3.0, -2.0j, 4.0 + 0.5j])

    result = orthonormalize(
        vector,
        basis,
        inner_product,
        reorthogonalization="always",
    )

    assert result.accepted
    assert np.iscomplexobj(result.vector)
    assert result.passes == 2
    assert result.norm > 0.0
    assert physical_norm(result.vector, inner_product) == pytest.approx(1.0)
    np.testing.assert_allclose(
        basis_overlaps(basis, result.vector, inner_product),
        0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.overlaps,
        basis_overlaps(basis, vector, inner_product),
    )


def test_orthonormalize_rejects_a_dependent_direction():
    basis = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    result = orthonormalize(
        np.array([2.0, -3.0]),
        basis,
        np.vdot,
        reorthogonalization="always",
    )

    assert not result.accepted
    assert result.passes == 2
    assert result.norm == pytest.approx(0.0)
    assert not np.iscomplexobj(result.vector)
    np.testing.assert_allclose(result.vector, 0.0)


def test_if_needed_reorthogonalization_uses_dgks_norm_loss_criterion():
    basis = (np.array([1.0, 0.0]),)
    batches = []

    def inner_products(left, right):
        batches.append(len(left))
        return np.asarray([np.vdot(item, right) for item in left])

    mostly_dependent = orthonormalize(
        np.array([1.0, 1.0e-8]),
        basis,
        np.vdot,
        inner_products=inner_products,
    )
    independent = orthonormalize(
        np.array([0.1, 1.0]),
        basis,
        np.vdot,
        inner_products=inner_products,
    )

    assert mostly_dependent.passes == 2
    assert independent.passes == 1
    assert batches == [1, 1, 1]


def test_basis_orthogonality_error_detects_a_weighted_defect():
    weights = np.array([2.0, 3.0])
    inner_product = _weighted_inner_product(weights)
    orthonormal = (
        np.array([1.0 / np.sqrt(2.0), 0.0]),
        np.array([0.0, 1.0 / np.sqrt(3.0)]),
    )
    defective = (orthonormal[0], orthonormal[0] + orthonormal[1])

    assert basis_orthogonality_error(
        orthonormal,
        inner_product,
    ) == pytest.approx(0.0, abs=3.0e-16)
    assert basis_orthogonality_error(defective, inner_product) > 0.9


def test_projected_matrix_rejects_significant_non_hermiticity():
    basis = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    images = (np.array([1.0, 0.0]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="not Hermitian"):
        projected_hermitian_matrix(basis, images, np.vdot)


def test_incremental_projected_matrix_matches_full_construction_with_batches():
    transformed, hamiltonian, weights, inner_product = _weighted_problem()
    basis = tuple(
        np.eye(4, dtype=complex)[:, index] / np.sqrt(weights[index])
        for index in range(4)
    )
    images = tuple(hamiltonian @ vector for vector in basis)
    batches = []

    def inner_products(left, right):
        batches.append(len(left))
        return np.asarray([inner_product(vector, right) for vector in left])

    incremental = np.empty((0, 0))
    for dimension in range(1, len(basis) + 1):
        incremental = extend_projected_hermitian_matrix(
            incremental,
            basis[:dimension],
            images[dimension - 1],
            inner_product,
            inner_products=inner_products,
        )

    np.testing.assert_allclose(incremental, transformed, atol=2.0e-15)
    np.testing.assert_allclose(
        incremental,
        projected_hermitian_matrix(basis, images, inner_product),
    )
    assert batches == [1, 2, 3, 4]


def test_rayleigh_ritz_recovers_weighted_complex_hermitian_eigenpairs():
    transformed, hamiltonian, weights, inner_product = _weighted_problem()
    basis = tuple(
        np.eye(4, dtype=complex)[:, index] / np.sqrt(weights[index])
        for index in range(4)
    )
    images = tuple(hamiltonian @ vector for vector in basis)

    result = rayleigh_ritz(
        basis,
        images,
        inner_product,
        number_of_roots=2,
    )
    expected_values, _ = np.linalg.eigh(transformed)

    np.testing.assert_allclose(result.eigenvalues, expected_values[:2])
    np.testing.assert_allclose(result.residual_norms, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        result.projected_matrix,
        transformed,
        atol=2.0e-15,
    )
    assert result.coefficients.shape == (4, 2)
    assert basis_orthogonality_error(
        result.vectors,
        inner_product,
    ) == pytest.approx(0.0, abs=2.0e-15)
    np.testing.assert_allclose(
        projected_hermitian_matrix(
            result.vectors,
            result.hamiltonian_images,
            inner_product,
        ),
        np.diag(expected_values[:2]),
        atol=2.0e-15,
    )
    for index, vector in enumerate(result.vectors):
        assert physical_norm(vector, inner_product) == pytest.approx(1.0)
        np.testing.assert_allclose(
            result.hamiltonian_images[index],
            hamiltonian @ vector,
            atol=2.0e-15,
        )


def test_rayleigh_ritz_returns_residual_for_an_incomplete_subspace():
    hamiltonian = np.array([[1.0, 0.4], [0.4, 3.0]])
    basis = (np.array([1.0, 0.0]),)
    images = (hamiltonian @ basis[0],)

    result = rayleigh_ritz(basis, images, np.vdot)

    np.testing.assert_allclose(result.eigenvalues, [1.0])
    np.testing.assert_allclose(result.residuals[0], [0.0, 0.4])
    np.testing.assert_allclose(result.residual_norms, [0.4])


def test_rayleigh_ritz_accepts_a_precomputed_projected_matrix():
    hamiltonian = np.array([[1.0, 0.4], [0.4, 3.0]])
    basis = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    images = tuple(hamiltonian @ vector for vector in basis)

    result = rayleigh_ritz(
        basis,
        images,
        np.vdot,
        number_of_roots=1,
        projected_matrix=hamiltonian,
    )

    expected, _ = np.linalg.eigh(hamiltonian)
    np.testing.assert_allclose(result.eigenvalues, expected[:1])
    np.testing.assert_allclose(result.residual_norms, 0.0, atol=5.0e-16)


def test_phase_alignment_removes_a_complex_global_phase():
    weights = np.array([2.0, 3.0, 4.0])
    inner_product = _weighted_inner_product(weights)
    reference = normalize(
        np.array([1.0 + 0.5j, -0.2j, 0.3]),
        inner_product,
    )
    candidate = np.exp(0.73j) * reference

    aligned = phase_align(reference, candidate, inner_product)

    np.testing.assert_allclose(aligned, reference, atol=2.0e-15)
    assert phase_aligned_distance(
        reference,
        candidate,
        inner_product,
    ) == pytest.approx(0.0, abs=2.0e-15)


def test_subspace_primitives_validate_callback_outputs_and_dimensions():
    with pytest.raises(ValueError, match="non-finite"):
        physical_norm(np.ones(2), lambda left, right: np.nan)
    with pytest.raises(ValueError, match="significant imaginary"):
        physical_norm(np.ones(2), lambda left, right: 1.0 + 1.0j)
    with pytest.raises(ValueError, match="negative squared norm"):
        physical_norm(np.ones(2), lambda left, right: -1.0)
    with pytest.raises(ValueError, match="one value per left vector"):
        basis_overlaps(
            (np.ones(2),),
            np.ones(2),
            np.vdot,
            inner_products=lambda left, right: np.ones(2),
        )
    with pytest.raises(ValueError, match="same length"):
        projected_hermitian_matrix(
            (np.array([1.0, 0.0]),),
            (),
            np.vdot,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        rayleigh_ritz(
            (np.array([1.0, 0.0]),),
            (np.array([1.0, 0.0]),),
            np.vdot,
            number_of_roots=2,
        )


def test_direct_eigensolver_result_exposes_history_arrays():
    history = (
        GroundStateEigenIteration(
            iteration=1,
            eigenvalue=-0.5,
            eigenvalue_change=0.1,
            state_change=0.2,
            residual_norm=0.3,
            subspace_dimension=2,
            orthogonality_error=1.0e-15,
        ),
        GroundStateEigenIteration(
            iteration=2,
            eigenvalue=-0.6,
            eigenvalue_change=0.1,
            state_change=0.05,
            residual_norm=0.02,
            subspace_dimension=3,
            orthogonality_error=2.0e-15,
            restarted=True,
        ),
    )
    result = GroundStateEigenResult(
        state=np.array([1.0, 0.0]),
        eigenvalue=-0.6,
        residual_norm=0.02,
        converged=False,
        iterations=2,
        restarts=1,
        hamiltonian_actions=3,
        preconditioner_applications=0,
        termination_reason="maximum_iterations",
        history=history,
    )

    np.testing.assert_allclose(result.eigenvalue_history, [-0.5, -0.6])
    np.testing.assert_allclose(result.residual_history, [0.3, 0.02])
    np.testing.assert_allclose(result.state_change_history, [0.2, 0.05])
    np.testing.assert_array_equal(result.subspace_dimensions, [2, 3])
    assert result.maximum_subspace_dimension == 3
    assert result.final_subspace_dimension == 3
    np.testing.assert_array_equal(result.restart_iterations, [2])
