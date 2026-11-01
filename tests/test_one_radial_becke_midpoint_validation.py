import numpy as np
import pytest

from grid_tdse.spherical.one_radial_becke import validate_midpoint_l0


@pytest.fixture(scope="module")
def midpoint_validations():
    coarse = validate_midpoint_l0(
        integration_radial_spacing=1.0,
        integration_n_points=6,
        lebedev_order=17,
    )
    refined = validate_midpoint_l0()
    return coarse, refined


def test_partitioned_midpoint_matrices_converge_to_continuum(
    midpoint_validations,
):
    coarse, refined = midpoint_validations
    coarse_errors = coarse.partitioned_to_continuum
    refined_errors = refined.partitioned_to_continuum

    for name in ("overlap", "kinetic", "potential", "hamiltonian"):
        assert getattr(refined_errors, name) < 0.1 * getattr(
            coarse_errors, name
        )
    assert refined_errors.overlap < 5.0e-5
    assert refined_errors.kinetic < 5.0e-4
    assert refined_errors.potential < 5.0e-5
    assert refined_errors.hamiltonian < 5.0e-4


def test_structured_midpoint_comparison_identifies_dvr_quadrature_gap(
    midpoint_validations,
):
    _, validation = midpoint_validations
    errors = validation.structured_to_continuum

    assert errors.kinetic < 2.0e-14
    assert 0.25 < errors.overlap < 0.35
    assert 0.30 < errors.potential < 0.40
    assert 0.15 < errors.hamiltonian < 0.25

    for name in ("overlap", "kinetic", "potential", "hamiltonian"):
        structured = getattr(validation.structured, name)
        continuum = getattr(validation.continuum, name)
        partitioned = getattr(validation.partitioned, name)
        np.testing.assert_allclose(
            partitioned - structured,
            continuum - structured,
            atol=7.0e-4,
            rtol=0.0,
        )


def test_midpoint_validation_matrices_are_hermitian_and_read_only(
    midpoint_validations,
):
    _, validation = midpoint_validations

    assert validation.midpoint_nodes == (0.0, 1.0, 3.0)
    assert validation.midpoint_n_points == (3, 3)
    assert validation.partitioned_quadrature_size == 225380
    for matrix_set in (
        validation.structured,
        validation.continuum,
        validation.partitioned,
    ):
        for matrix in (
            matrix_set.overlap,
            matrix_set.kinetic,
            matrix_set.potential,
            matrix_set.hamiltonian,
        ):
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=0.0)
            assert matrix.shape == (3, 3)
            assert not matrix.flags.writeable


def test_midpoint_validation_rejects_invalid_controls():
    with pytest.raises(ValueError, match="finite and positive"):
        validate_midpoint_l0(integration_radial_spacing=0.0)
    with pytest.raises(TypeError, match="real number"):
        validate_midpoint_l0(integration_radial_spacing="0.5")
