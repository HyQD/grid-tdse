import numpy as np
import pytest

from grid_tdse.spherical.one_radial_becke import (
    KineticIntegrationSettings,
    validate_h2plus_ground_state,
)


@pytest.fixture(scope="module")
def h2plus_validation():
    return validate_h2plus_ground_state()


def test_h2plus_matrices_are_hermitian_positive_and_read_only(
    h2plus_validation,
):
    for result in (
        h2plus_validation.coarse,
        h2plus_validation.refined,
        h2plus_validation.reference,
    ):
        matrices = result.matrices
        for matrix in (
            matrices.overlap,
            matrices.kinetic,
            matrices.potential,
            matrices.hamiltonian,
        ):
            np.testing.assert_allclose(matrix, matrix.conj().T)
            assert not matrix.flags.writeable
        assert max(matrices.relative_antihermitian_norms) < 1.0e-14
        assert matrices.overlap_minimum_eigenvalue > 0.0
        assert matrices.kinetic_minimum_eigenvalue > -1.0e-12
        assert matrices.basis_size == 20


def test_h2plus_generalized_solutions_are_normalized_and_stable(
    h2plus_validation,
):
    result = h2plus_validation.reference

    assert h2plus_validation.overlap_cutoffs == (
        1.0e-8,
        1.0e-10,
        1.0e-12,
    )
    assert result.retained_ranks == (20, 20, 20)
    assert result.discarded_ranks == (0, 0, 0)
    assert h2plus_validation.cutoff_energy_spread < 1.0e-13
    for solution in result.solutions:
        coefficients = solution.coefficients
        np.testing.assert_allclose(
            coefficients.conj().T
            @ result.matrices.overlap
            @ coefficients,
            np.eye(1),
            atol=3.0e-14,
        )
        assert solution.diagnostics.orthonormality_error < 3.0e-14
        assert solution.diagnostics.relative_residual_norms[0] < 3.0e-14


def test_h2plus_energy_converges_under_quadrature_refinement(
    h2plus_validation,
):
    assert h2plus_validation.refined_to_reference_energy_error < 0.02 * (
        h2plus_validation.coarse_to_reference_energy_error
    )
    assert h2plus_validation.refined_to_reference_energy_error < 1.0e-5


def test_h2plus_reports_matched_extent_midpoint_comparison(
    h2plus_validation,
):
    reference = h2plus_validation.reference
    midpoint = h2plus_validation.midpoint

    assert reference.matrices.integration_radius == pytest.approx(6.0)
    assert midpoint.radial_extent == pytest.approx(5.0)
    assert midpoint.radial_nodes == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert midpoint.basis_size == 594
    assert reference.total_energies[0] == pytest.approx(
        reference.electronic_energies[0]
        + h2plus_validation.nuclear_repulsion_energy
    )
    assert midpoint.total_energy == pytest.approx(
        midpoint.electronic_energy
        + h2plus_validation.nuclear_repulsion_energy
    )
    assert 0.02 < h2plus_validation.becke_to_midpoint_energy_error < 0.04


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("internuclear_distance", 0.0, ValueError, "finite and positive"),
        ("particle_mass", "1", TypeError, "real number"),
        ("radial_n_points", True, TypeError, "contain integers"),
        ("m", 1, ValueError, "requires m = 0"),
        ("overlap_cutoffs", (), ValueError, "must not be empty"),
        ("overlap_cutoffs", (1.0,), ValueError, "0 <= tau < 1"),
        (
            "reference_settings",
            (0.25, 14, 41),
            TypeError,
            "KineticIntegrationSettings",
        ),
    ],
)
def test_h2plus_validation_rejects_invalid_controls(
    keyword, value, error, message
):
    with pytest.raises(error, match=message):
        validate_h2plus_ground_state(**{keyword: value})


def test_h2plus_validation_caches_identical_integration_settings():
    settings = KineticIntegrationSettings(1.0, 4, 9)
    result = validate_h2plus_ground_state(
        coarse_settings=settings,
        refined_settings=settings,
        reference_settings=settings,
        midpoint_radial_n_points=4,
        midpoint_l_max=1,
    )

    assert result.coarse.matrices is result.refined.matrices
    assert result.refined.matrices is result.reference.matrices
    assert result.coarse_to_reference_energy_error == 0.0
    assert result.refined_to_reference_energy_error == 0.0
