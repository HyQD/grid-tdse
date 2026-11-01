import numpy as np
import pytest
from scipy.special import erf

from grid_tdse.spherical.one_radial_becke import (
    BeckeMulticenterQuadrature,
    BeckePartition,
    FEMDVRQuadratureSpec,
)


COLLINEAR_CENTERS = np.array(
    [[0.0, 0.0, -1.4], [0.0, 0.0, 0.1], [0.0, 0.0, 2.0]]
)
NONCOLLINEAR_CENTERS = np.array(
    [[-0.8, 0.1, 0.2], [0.7, -0.4, 0.3], [0.2, 1.1, -0.5]]
)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _gaussian(center, alpha):
    center = np.asarray(center)

    def evaluate(points):
        return np.exp(-alpha * np.sum((points - center) ** 2, axis=-1))

    return evaluate


def _full_gaussian_integral(alpha):
    return (np.pi / alpha) ** 1.5


def _centered_gaussian_ball_integral(alpha, radius):
    return (
        np.pi**1.5 / alpha**1.5 * erf(np.sqrt(alpha) * radius)
        - 2.0 * np.pi * radius * np.exp(-alpha * radius**2) / alpha
    )


def _uniform_spec(r_max=8.0, n_elements=8, n_points=10):
    return FEMDVRQuadratureSpec(
        np.linspace(0.0, r_max, n_elements + 1),
        [n_points] * n_elements,
    )


def test_radial_specification_preserves_element_local_shell_count():
    specification = FEMDVRQuadratureSpec([0.0, 1.0, 2.5], [3, 4])

    assert specification.nodes == (0.0, 1.0, 2.5)
    assert specification.n_points == (3, 4)
    assert specification.n_elements == 2
    assert specification.r_max == 2.5
    assert specification.nonzero_shell_count == 6


@pytest.mark.parametrize(
    ("nodes", "n_points", "error", "message"),
    [
        ([0.0], [], ValueError, "at least two"),
        ([1.0, 2.0], [3], ValueError, "start at zero"),
        ([0.0, 2.0, 1.0], [3, 3], ValueError, "strictly increasing"),
        ([0.0, np.inf], [3], ValueError, "finite"),
        ([0.0, 1.0], [3, 3], ValueError, "one entry"),
        ([0.0, 1.0], [2.5], TypeError, "integers"),
        ([0.0, 1.0], [1], ValueError, "at least two"),
    ],
)
def test_invalid_radial_specifications_are_rejected(
    nodes, n_points, error, message
):
    with pytest.raises(error, match=message):
        FEMDVRQuadratureSpec(nodes, n_points)


def test_shells_retain_bridge_nodes_and_omit_the_origin():
    specification = FEMDVRQuadratureSpec([0.0, 1.0, 2.0], [3, 4])
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, 0.0]]), specification, 5
    )
    shells = list(quadrature.iter_shells())
    radii = np.array([shell.radius for shell in shells])

    assert len(shells) == specification.nonzero_shell_count
    assert np.count_nonzero(radii == 1.0) == 2
    assert np.all(radii > 0.0)
    assert quadrature.point_count == len(shells) * 14
    for shell in shells:
        np.testing.assert_allclose(shell.partition_weights, 1.0)
        np.testing.assert_allclose(shell.weights, shell.base_volume_weights)
        assert np.all(
            np.linalg.norm(shell.points, axis=-1)
            > 100.0 * np.finfo(float).eps
        )


def test_shell_weight_formula_and_solid_angle_normalization():
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
        FEMDVRQuadratureSpec([0.0, 2.0], [5]),
        9,
    )
    shell = next(quadrature.iter_shells(center=0))

    np.testing.assert_allclose(np.sum(shell.angular_weights), 4.0 * np.pi)
    np.testing.assert_allclose(
        shell.base_volume_weights,
        shell.radial_weight * shell.radius**2 * shell.angular_weights,
    )
    np.testing.assert_allclose(
        shell.weights, shell.base_volume_weights * shell.partition_weights
    )


def test_constant_and_vector_integrals_over_one_sphere():
    radius = 2.3
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec([0.0, radius], [5]),
        5,
    )

    volume = quadrature.integrate(lambda points: 1.0)
    vector = quadrature.integrate(
        lambda points: np.column_stack((np.ones(points.shape[0]), points))
    )

    np.testing.assert_allclose(volume, 4.0 * np.pi * radius**3 / 3.0)
    np.testing.assert_allclose(vector[0], volume)
    np.testing.assert_allclose(vector[1:], 0.0, atol=2e-15)


def test_finite_box_error_is_separate_from_radial_quadrature_error():
    alpha = 0.8
    radius = 2.5
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec([0.0, radius], [16]),
        3,
    )
    numerical = quadrature.integrate(_gaussian([0.0, 0.0, 0.0], alpha))
    finite_box = _centered_gaussian_ball_integral(alpha, radius)
    full_space = _full_gaussian_integral(alpha)

    assert abs(numerical - finite_box) < 1e-13
    assert full_space - finite_box > 0.1


def test_radial_order_converges_independently():
    alpha = 0.8
    radius = 2.5
    finite_box = _centered_gaussian_ball_integral(alpha, radius)
    errors = []
    for n_points in [4, 6, 8, 10]:
        quadrature = BeckeMulticenterQuadrature(
            BeckePartition([[0.0, 0.0, 0.0]]),
            FEMDVRQuadratureSpec([0.0, radius], [n_points]),
            3,
        )
        numerical = quadrature.integrate(
            _gaussian([0.0, 0.0, 0.0], alpha)
        )
        errors.append(abs(numerical - finite_box))

    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 2e-8


def test_lebedev_order_converges_independently_for_three_centers():
    alpha = 1.1
    gaussian = _gaussian([0.3, -0.2, 0.4], alpha)
    exact = _full_gaussian_integral(alpha)
    specification = _uniform_spec(r_max=8.0, n_elements=8, n_points=12)
    errors = []
    for order in [5, 13, 29, 47]:
        quadrature = BeckeMulticenterQuadrature(
            BeckePartition(NONCOLLINEAR_CENTERS), specification, order
        )
        errors.append(abs(quadrature.integrate(gaussian) - exact))

    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 5e-9


@pytest.mark.parametrize("centers", [COLLINEAR_CENTERS, NONCOLLINEAR_CENTERS])
def test_three_center_displaced_gaussian_integral(centers):
    alpha = 1.1
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers),
        _uniform_spec(r_max=8.0, n_elements=8, n_points=12),
        47,
    )
    numerical = quadrature.integrate(_gaussian([0.3, -0.2, 0.4], alpha))

    np.testing.assert_allclose(
        numerical, _full_gaussian_integral(alpha), atol=5e-9, rtol=0.0
    )


def test_converged_integral_is_invariant_under_lebedev_rotation():
    specification = _uniform_spec(r_max=8.0, n_elements=8, n_points=10)
    partition = BeckePartition(NONCOLLINEAR_CENTERS)
    gaussian = _gaussian([0.3, -0.2, 0.4], 1.1)
    rotations = [
        _rotation([1.0, 2.0, 3.0], 0.437),
        _rotation([-0.3, 0.7, 1.1], 0.91),
    ]
    values = [
        BeckeMulticenterQuadrature(
            partition, specification, 47, rotation=rotation
        ).integrate(gaussian)
        for rotation in rotations
    ]

    np.testing.assert_allclose(values[0], values[1], atol=3e-9, rtol=0.0)


def test_two_center_exchange_symmetry_of_partitioned_integrals():
    centers = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers), _uniform_spec(), 29
    )
    contributions = quadrature.integrate_by_center(
        _gaussian([0.0, 0.0, 0.0], 1.0)
    )

    np.testing.assert_allclose(
        contributions[0], contributions[1], atol=2e-15, rtol=0.0
    )


def test_three_center_permutation_permutates_center_contributions():
    specifications = (
        _uniform_spec(r_max=6.0, n_elements=6, n_points=7),
        _uniform_spec(r_max=7.0, n_elements=7, n_points=8),
        _uniform_spec(r_max=8.0, n_elements=8, n_points=9),
    )
    permutation = np.array([2, 0, 1])
    gaussian = _gaussian([0.2, -0.1, 0.3], 1.2)
    reference = BeckeMulticenterQuadrature(
        BeckePartition(NONCOLLINEAR_CENTERS), specifications, 17
    ).integrate_by_center(gaussian)
    permuted = BeckeMulticenterQuadrature(
        BeckePartition(NONCOLLINEAR_CENTERS[permutation]),
        tuple(specifications[index] for index in permutation),
        17,
    ).integrate_by_center(gaussian)

    np.testing.assert_allclose(permuted, reference[permutation], atol=2e-15)


def test_quadrature_points_do_not_coincide_with_any_center():
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(COLLINEAR_CENTERS),
        FEMDVRQuadratureSpec([0.0, 0.7, 1.5, 3.4], [4, 5, 6]),
        17,
    )

    for shell in quadrature.iter_shells():
        distances = np.linalg.norm(
            shell.points[:, np.newaxis, :]
            - COLLINEAR_CENTERS[np.newaxis, :, :],
            axis=-1,
        )
        assert np.all(distances > 1e-12)


def test_center_specific_radial_specs_and_center_selection():
    specifications = (
        FEMDVRQuadratureSpec([0.0, 1.0], [3]),
        FEMDVRQuadratureSpec([0.0, 1.0, 2.0], [3, 4]),
    )
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]),
        specifications,
        5,
    )

    assert len(list(quadrature.iter_shells(0))) == 2
    assert len(list(quadrature.iter_shells(1))) == 6
    with pytest.raises(ValueError, match="out of range"):
        list(quadrature.iter_shells(2))
    with pytest.raises(TypeError, match="integer"):
        list(quadrature.iter_shells(0.0))


def test_quadrature_constructor_validation():
    partition = BeckePartition([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    specification = FEMDVRQuadratureSpec([0.0, 2.0], [5])

    with pytest.raises(TypeError, match="BeckePartition"):
        BeckeMulticenterQuadrature(object(), specification, 5)
    with pytest.raises(ValueError, match="one entry per center"):
        BeckeMulticenterQuadrature(partition, [specification], 5)
    with pytest.raises(TypeError, match="FEM-DVR"):
        BeckeMulticenterQuadrature(partition, [specification, object()], 5)
    with pytest.raises(TypeError, match="integer"):
        BeckeMulticenterQuadrature(partition, specification, 5.0)
    with pytest.raises(ValueError, match="unsupported"):
        BeckeMulticenterQuadrature(partition, specification, 4)
    with pytest.raises(ValueError, match="shape"):
        BeckeMulticenterQuadrature(
            partition, specification, 5, rotation=np.eye(2)
        )
    with pytest.raises(ValueError, match="orthogonal"):
        BeckeMulticenterQuadrature(
            partition, specification, 5, rotation=np.ones((3, 3))
        )
    with pytest.raises(ValueError, match="proper"):
        BeckeMulticenterQuadrature(
            partition, specification, 5, rotation=np.diag([1.0, 1.0, -1.0])
        )
    with pytest.raises(ValueError, match="spherical pole"):
        BeckeMulticenterQuadrature(
            partition, specification, 5, rotation=np.eye(3)
        )


def test_all_intercenter_axes_are_checked_against_lebedev_directions():
    specification = FEMDVRQuadratureSpec([0.0, 2.0], [5])
    one_center = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, 0.0]]), specification, 5
    )
    aligned_center = one_center.directions[0]

    with pytest.raises(ValueError, match="intercenter axis"):
        BeckeMulticenterQuadrature(
            BeckePartition([[0.0, 0.0, 0.0], aligned_center]),
            specification,
            5,
        )


def test_integrand_validation():
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec([0.0, 1.0], [3]),
        5,
    )

    with pytest.raises(TypeError, match="callable"):
        quadrature.integrate(None)
    with pytest.raises(ValueError, match="angular point axis"):
        quadrature.integrate(lambda points: np.ones(points.shape[0] + 1))
