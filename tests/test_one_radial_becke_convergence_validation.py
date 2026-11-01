import numpy as np
import pytest

from grid_tdse.spherical.one_radial_becke import (
    KineticIntegrationSettings,
    run_h2plus_convergence_study,
)


@pytest.fixture(scope="module")
def convergence_study():
    return run_h2plus_convergence_study(
        radial_orders=(3, 4, 5),
        element_layouts=(
            (0.0, 2.0, 4.0),
            (0.0, 1.0, 2.0, 4.0),
        ),
        radial_extents=(2.0, 4.0, 6.0),
        lebedev_orders=(17, 29, 41),
        angular_cutoffs=(0, 1, 2),
        becke_iterations=(2, 3, 4),
        midpoint_reference_atomic_extent=6.0,
        midpoint_radial_n_points=8,
        midpoint_l_max=6,
    )


def _energies(points):
    return np.asarray([point.electronic_energy for point in points])


def test_radial_and_angular_basis_refinement_lowers_energy(
    convergence_study,
):
    assert np.all(np.diff(_energies(convergence_study.radial_order)) < 0.0)
    assert np.all(np.diff(_energies(convergence_study.element_layout)) < 0.0)
    assert np.all(np.diff(_energies(convergence_study.radial_extent)) < 0.0)
    assert np.all(np.diff(_energies(convergence_study.angular_cutoff)) < 0.0)


def test_lebedev_energy_converges_independently(convergence_study):
    energies = _energies(convergence_study.lebedev_order)
    coarse_change = abs(energies[1] - energies[0])
    fine_change = abs(energies[2] - energies[1])

    assert fine_change < 0.03 * coarse_change
    assert fine_change < 2.0e-6


def test_becke_sweep_keeps_integration_partition_fixed(convergence_study):
    assert tuple(
        point.orbital_switch_iterations
        for point in convergence_study.becke_iterations
    ) == (2, 3, 4)
    assert all(
        point.integration_switch_iterations == 3
        for point in convergence_study.becke_iterations
    )
    condition_numbers = np.asarray(
        [
            point.overlap_condition_number
            for point in convergence_study.becke_iterations
        ]
    )
    assert np.all(np.diff(condition_numbers) < 0.0)


def test_overlap_cutoff_stability_detects_high_l_rank_loss(
    convergence_study,
):
    baseline = convergence_study.overlap_cutoff
    near_dependent = convergence_study.near_dependent_overlap_cutoff

    assert all(point.discarded_rank == 0 for point in baseline)
    np.testing.assert_allclose(_energies(baseline), baseline[0].electronic_energy)
    assert tuple(point.retained_rank for point in near_dependent) == (
        119,
        120,
        120,
    )
    assert tuple(point.discarded_rank for point in near_dependent) == (1, 0, 0)
    assert abs(
        near_dependent[0].electronic_energy
        - near_dependent[-1].electronic_energy
    ) > 1.0e-4
    assert near_dependent[1].electronic_energy == pytest.approx(
        near_dependent[2].electronic_energy, abs=1.0e-13
    )


def test_convergence_points_report_small_residuals_and_complete_metadata(
    convergence_study,
):
    groups = (
        convergence_study.radial_order,
        convergence_study.element_layout,
        convergence_study.radial_extent,
        convergence_study.lebedev_order,
        convergence_study.angular_cutoff,
        convergence_study.becke_iterations,
        convergence_study.overlap_cutoff,
        convergence_study.near_dependent_overlap_cutoff,
    )
    for point in (item for group in groups for item in group):
        assert point.projected_residual_norm < 1.0e-10
        if point.discarded_rank == 0:
            assert point.relative_residual_norm < 1.0e-10
        assert point.basis_size == point.retained_rank + point.discarded_rank
        assert point.quadrature_size > point.basis_size
        assert point.radial_nodes[0] == 0.0


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("internuclear_distance", 0.0, ValueError, "finite and positive"),
        ("radial_orders", (), ValueError, "must not be empty"),
        ("element_layouts", None, TypeError, "must be a sequence"),
        (
            "integration_settings",
            (1.0, 8, 29),
            TypeError,
            "KineticIntegrationSettings",
        ),
        (
            "include_near_dependent_cutoff_study",
            1,
            TypeError,
            "must be boolean",
        ),
    ],
)
def test_convergence_study_rejects_invalid_controls(
    keyword, value, error, message
):
    with pytest.raises(error, match=message):
        run_h2plus_convergence_study(**{keyword: value})


def test_convergence_study_can_skip_expensive_near_dependent_case():
    settings = KineticIntegrationSettings(1.0, 4, 17)
    study = run_h2plus_convergence_study(
        integration_settings=settings,
        radial_orders=(4,),
        element_layouts=((0.0, 2.0, 4.0),),
        radial_extents=(4.0,),
        lebedev_orders=(17,),
        angular_cutoffs=(1,),
        becke_iterations=(3,),
        include_near_dependent_cutoff_study=False,
        midpoint_reference_atomic_extent=4.0,
        midpoint_radial_n_points=4,
        midpoint_l_max=1,
    )

    assert study.near_dependent_overlap_cutoff == ()
