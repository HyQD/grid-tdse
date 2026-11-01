import numpy as np
import pytest

from grid_tdse.spherical.one_radial_becke import (
    validate_two_center_overlap,
)


@pytest.fixture(scope="module")
def overlap_cases():
    common = {"lebedev_order": 29}
    return {
        "baseline": validate_two_center_overlap(**common),
        "near": validate_two_center_overlap(
            internuclear_distance=1.0, **common
        ),
        "far": validate_two_center_overlap(
            internuclear_distance=6.0, **common
        ),
        "short": validate_two_center_overlap(
            radial_nodes=(0.0, 1.5, 3.0), **common
        ),
        "long": validate_two_center_overlap(
            radial_nodes=(0.0, 4.0, 8.0), **common
        ),
        "low_order": validate_two_center_overlap(
            radial_n_points=3, **common
        ),
        "high_order": validate_two_center_overlap(
            radial_n_points=6, **common
        ),
        "fine_layout": validate_two_center_overlap(
            radial_nodes=(0.0, 0.5, 1.5, 2.5, 4.0), **common
        ),
        "l0": validate_two_center_overlap(l_max=0, **common),
        "l3": validate_two_center_overlap(l_max=3, **common),
        "l11_29": validate_two_center_overlap(l_max=11, **common),
        "l11_41": validate_two_center_overlap(
            l_max=11, lebedev_order=41
        ),
        "becke1": validate_two_center_overlap(
            orbital_switch_iterations=1,
            integration_switch_iterations=1,
            **common,
        ),
        "becke4": validate_two_center_overlap(
            orbital_switch_iterations=4,
            integration_switch_iterations=4,
            **common,
        ),
    }


def test_two_center_overlap_is_hermitian_positive_and_read_only(
    overlap_cases,
):
    result = overlap_cases["baseline"]

    np.testing.assert_allclose(result.overlap, result.overlap.conj().T)
    np.testing.assert_allclose(
        result.eigenvalues, np.linalg.eigvalsh(result.overlap)
    )
    assert result.minimum_eigenvalue > 0.0
    assert result.relative_antihermitian_norm < 1.0e-14
    assert result.basis_size == 20
    assert result.eigenvalues.shape == (result.basis_size,)
    assert not result.overlap.flags.writeable
    assert not result.eigenvalues.flags.writeable
    assert tuple(
        decision.relative_cutoff for decision in result.rank_decisions
    ) == (1.0e-8, 1.0e-10, 1.0e-12)
    assert all(
        decision.retained_rank == result.basis_size
        and decision.discarded_rank == 0
        for decision in result.rank_decisions
    )


def test_integration_cells_cover_complete_two_center_basis_support(
    overlap_cases,
):
    for result in overlap_cases.values():
        assert result.integration_radius == pytest.approx(
            result.radial_nodes[-1] + result.internuclear_distance
        )
        n_elements = int(
            np.ceil(
                result.integration_radius
                / result.integration_radial_spacing
            )
        )
        assert result.integration_radial_spacing == pytest.approx(
            result.integration_radius / n_elements
        )


def test_overlap_conditioning_trends_identify_redundancy(overlap_cases):
    assert (
        overlap_cases["near"].condition_number
        > overlap_cases["far"].condition_number
    )
    assert (
        overlap_cases["long"].condition_number
        > overlap_cases["short"].condition_number
    )
    assert (
        overlap_cases["high_order"].condition_number
        > overlap_cases["low_order"].condition_number
    )
    assert (
        overlap_cases["fine_layout"].condition_number
        > overlap_cases["baseline"].condition_number
    )
    assert (
        overlap_cases["l3"].condition_number
        > 1000.0 * overlap_cases["l0"].condition_number
    )
    assert (
        overlap_cases["becke1"].condition_number
        > overlap_cases["becke4"].condition_number
    )


def test_high_l_rank_loss_is_stable_to_lebedev_refinement(overlap_cases):
    order29 = overlap_cases["l11_29"]
    order41 = overlap_cases["l11_41"]

    assert order29.basis_size == 120
    assert tuple(
        decision.retained_rank for decision in order29.rank_decisions
    ) == (119, 120, 120)
    assert tuple(
        decision.retained_rank for decision in order41.rank_decisions
    ) == (119, 120, 120)
    assert order29.condition_number > 1.0e8
    assert order41.condition_number > 1.0e8
    assert order29.minimum_eigenvalue == pytest.approx(
        order41.minimum_eigenvalue, rel=0.01
    )


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("internuclear_distance", 0.0, ValueError, "finite and positive"),
        ("internuclear_distance", "2", TypeError, "real number"),
        ("integration_radial_spacing", 0.0, ValueError, "finite and positive"),
        ("radial_n_points", True, TypeError, "contain integers"),
        ("rank_cutoffs", (), ValueError, "must not be empty"),
        ("rank_cutoffs", (1.0,), ValueError, "0 <= tau < 1"),
    ],
)
def test_two_center_overlap_rejects_invalid_controls(
    keyword, value, error, message
):
    with pytest.raises(error, match=message):
        validate_two_center_overlap(**{keyword: value})
