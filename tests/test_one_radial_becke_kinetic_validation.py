import numpy as np
import pytest

from grid_tdse.spherical.one_radial_becke import (
    KineticIntegrationSettings,
    validate_two_center_kinetic,
)


@pytest.fixture(scope="module")
def kinetic_validation():
    return validate_two_center_kinetic()


def test_two_center_kinetic_is_hermitian_positive_and_read_only(
    kinetic_validation,
):
    for result in (
        kinetic_validation.coarse,
        kinetic_validation.refined,
        kinetic_validation.reference,
    ):
        np.testing.assert_allclose(result.kinetic, result.kinetic.conj().T)
        np.testing.assert_allclose(
            result.eigenvalues, np.linalg.eigvalsh(result.kinetic)
        )
        assert result.relative_antihermitian_norm < 1.0e-14
        assert result.minimum_eigenvalue > -1.0e-12 * (
            result.maximum_eigenvalue
        )
        assert result.basis_size == 20
        assert not result.kinetic.flags.writeable
        assert not result.eigenvalues.flags.writeable


def test_refinement_converges_toward_direct_high_order_integration(
    kinetic_validation,
):
    assert kinetic_validation.refined_relative_matrix_error < 0.15 * (
        kinetic_validation.coarse_relative_matrix_error
    )
    assert kinetic_validation.refined_relative_matrix_error < 1.0e-2
    assert max(
        comparison.relative_error
        for comparison in kinetic_validation.selected_elements
    ) < 2.0e-3

    pairs = {
        (comparison.row, comparison.column)
        for comparison in kinetic_validation.selected_elements
    }
    assert (0, 0) in pairs
    assert (0, 2) in pairs
    assert (0, 10) in pairs
    assert (1, 11) in pairs


def test_kinetic_integration_covers_the_complete_basis_union(
    kinetic_validation,
):
    expected_radius = (
        kinetic_validation.radial_nodes[-1]
        + kinetic_validation.internuclear_distance
    )
    for result in (
        kinetic_validation.coarse,
        kinetic_validation.refined,
        kinetic_validation.reference,
    ):
        assert result.integration_radius == pytest.approx(expected_radius)
        assert result.maximum_radial_spacing <= (
            result.integration_settings.radial_spacing + 1.0e-14
        )


def test_kinetic_scales_inversely_with_particle_mass():
    settings = KineticIntegrationSettings(1.0, 5, 17)
    mass_one = validate_two_center_kinetic(
        particle_mass=1.0,
        coarse_settings=settings,
        refined_settings=settings,
        reference_settings=settings,
    ).reference
    mass_two = validate_two_center_kinetic(
        particle_mass=2.0,
        coarse_settings=settings,
        refined_settings=settings,
        reference_settings=settings,
    ).reference

    np.testing.assert_allclose(mass_two.kinetic, 0.5 * mass_one.kinetic)


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("internuclear_distance", 0.0, ValueError, "finite and positive"),
        ("particle_mass", "1", TypeError, "real number"),
        ("radial_n_points", True, TypeError, "contain integers"),
        (
            "coarse_settings",
            (1.0, 6, 17),
            TypeError,
            "KineticIntegrationSettings",
        ),
        ("selected_elements", ((0, 20),), ValueError, "out of range"),
    ],
)
def test_two_center_kinetic_rejects_invalid_controls(
    keyword, value, error, message
):
    with pytest.raises(error, match=message):
        validate_two_center_kinetic(**{keyword: value})


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ((0.0, 5, 17), ValueError, "finite and positive"),
        ((1.0, 1, 17), ValueError, "at least two"),
        ((1.0, 5.0, 17), TypeError, "integer"),
        ((1.0, 5, 17.0), TypeError, "integer"),
    ],
)
def test_kinetic_integration_settings_validate_inputs(
    arguments, error, message
):
    with pytest.raises(error, match=message):
        KineticIntegrationSettings(*arguments)
