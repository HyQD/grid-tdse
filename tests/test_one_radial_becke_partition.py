import numpy as np
import pytest

from grid_tdse.spherical.one_radial_becke import BeckePartition


COLLINEAR_CENTERS = np.array(
    [[0.0, 0.0, -1.4], [0.0, 0.0, 0.1], [0.0, 0.0, 2.0]]
)
NONCOLLINEAR_CENTERS = np.array(
    [[-0.8, 0.1, 0.2], [0.7, -0.4, 0.3], [0.2, 1.1, -0.5]]
)
SAMPLE_POINTS = np.array(
    [
        [-1.2, 0.4, 0.7],
        [0.3, -0.8, 1.4],
        [1.7, 0.6, -0.9],
        [-0.1, 1.8, 0.5],
    ]
)


@pytest.mark.parametrize(
    "centers",
    [
        np.array([[0.2, -0.3, 0.4]]),
        np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
        COLLINEAR_CENTERS,
        NONCOLLINEAR_CENTERS,
    ],
)
def test_partition_and_gradient_sum_identities(centers):
    evaluation = BeckePartition(centers).evaluate(SAMPLE_POINTS)

    np.testing.assert_allclose(
        np.sum(evaluation.weights, axis=-1), 1.0, atol=2e-15, rtol=0.0
    )
    np.testing.assert_allclose(
        np.sum(evaluation.weight_gradients, axis=-2),
        0.0,
        atol=2e-15,
        rtol=0.0,
    )
    assert np.all(evaluation.weights >= 0.0)
    assert np.all(evaluation.weights <= 1.0)


@pytest.mark.parametrize("centers", [COLLINEAR_CENTERS, NONCOLLINEAR_CENTERS])
def test_each_center_has_its_partition_limit(centers):
    evaluation = BeckePartition(centers).evaluate(centers)

    np.testing.assert_allclose(evaluation.weights, np.eye(3), atol=0.0, rtol=0.0)
    assert np.all(np.isfinite(evaluation.weight_gradients))
    np.testing.assert_allclose(evaluation.weight_gradients, 0.0, atol=0.0, rtol=0.0)


def test_pair_data_use_ordered_center_convention():
    evaluation = BeckePartition(NONCOLLINEAR_CENTERS).evaluate(SAMPLE_POINTS)
    diagonal = np.arange(3)

    np.testing.assert_allclose(
        evaluation.pair_variables
        + np.swapaxes(evaluation.pair_variables, -1, -2),
        0.0,
        atol=2e-15,
        rtol=0.0,
    )
    switch_sum = evaluation.pair_switches + np.swapaxes(
        evaluation.pair_switches, -1, -2
    )
    switch_sum[..., diagonal, diagonal] = 1.0
    np.testing.assert_allclose(switch_sum, 1.0, atol=2e-15, rtol=0.0)
    np.testing.assert_array_equal(
        evaluation.pair_switches[..., diagonal, diagonal], 1.0
    )


@pytest.mark.parametrize("centers", [COLLINEAR_CENTERS, NONCOLLINEAR_CENTERS])
def test_analytic_gradients_agree_with_cartesian_finite_differences(centers):
    partition = BeckePartition(centers)
    step = 2.0e-6
    numerical = np.empty((SAMPLE_POINTS.shape[0], 3, 3))

    for component in range(3):
        displacement = np.zeros(3)
        displacement[component] = step
        plus = partition.weights(SAMPLE_POINTS + displacement)
        minus = partition.weights(SAMPLE_POINTS - displacement)
        numerical[..., component] = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(
        partition.gradients(SAMPLE_POINTS), numerical, atol=2e-10, rtol=2e-8
    )


def test_scalar_and_vectorized_evaluations_agree():
    partition = BeckePartition(NONCOLLINEAR_CENTERS)
    vectorized = partition.evaluate(SAMPLE_POINTS)

    for point_index, point in enumerate(SAMPLE_POINTS):
        scalar = partition.evaluate(point)
        for field in scalar.__dataclass_fields__:
            np.testing.assert_allclose(
                getattr(scalar, field),
                getattr(vectorized, field)[point_index],
                atol=0.0,
                rtol=0.0,
            )


def test_arbitrary_leading_point_dimensions_are_preserved():
    points = SAMPLE_POINTS.reshape(2, 2, 3)
    evaluation = BeckePartition(NONCOLLINEAR_CENTERS).evaluate(points)

    assert evaluation.center_displacements.shape == (2, 2, 3, 3)
    assert evaluation.pair_variables.shape == (2, 2, 3, 3)
    assert evaluation.pair_variable_gradients.shape == (2, 2, 3, 3, 3)
    assert evaluation.weights.shape == (2, 2, 3)
    assert evaluation.weight_gradients.shape == (2, 2, 3, 3)


def test_center_permutation_only_permutates_center_axes():
    permutation = np.array([2, 0, 1])
    reference = BeckePartition(NONCOLLINEAR_CENTERS).evaluate(SAMPLE_POINTS)
    permuted = BeckePartition(NONCOLLINEAR_CENTERS[permutation]).evaluate(
        SAMPLE_POINTS
    )

    np.testing.assert_allclose(permuted.weights, reference.weights[..., permutation])
    np.testing.assert_allclose(
        permuted.weight_gradients,
        reference.weight_gradients[..., permutation, :],
    )
    np.testing.assert_allclose(
        permuted.pair_variables,
        reference.pair_variables[..., permutation, :][..., :, permutation],
    )
    np.testing.assert_allclose(
        permuted.pair_switches,
        reference.pair_switches[..., permutation, :][..., :, permutation],
    )


def test_rigid_translation_and_rotation_covariance():
    angle = 0.63
    axis = np.array([1.0, -2.0, 0.7])
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    translation = np.array([0.9, -1.7, 0.4])
    transformed_centers = NONCOLLINEAR_CENTERS @ rotation.T + translation
    transformed_points = SAMPLE_POINTS @ rotation.T + translation

    reference = BeckePartition(NONCOLLINEAR_CENTERS).evaluate(SAMPLE_POINTS)
    transformed = BeckePartition(transformed_centers).evaluate(transformed_points)

    np.testing.assert_allclose(transformed.weights, reference.weights, atol=3e-15)
    np.testing.assert_allclose(
        transformed.weight_gradients,
        reference.weight_gradients @ rotation.T,
        atol=3e-15,
    )


def test_homonuclear_two_center_exchange_symmetry():
    centers = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.2, -0.4, -0.7], [-0.1, 0.8, 1.3]])
    reflected = points.copy()
    reflected[:, 2] *= -1.0
    partition = BeckePartition(centers)

    np.testing.assert_allclose(
        partition.weights(reflected), partition.weights(points)[..., ::-1]
    )


def test_single_center_is_constant_everywhere():
    partition = BeckePartition([[0.5, -0.4, 0.3]])
    evaluation = partition.evaluate(SAMPLE_POINTS)

    np.testing.assert_array_equal(evaluation.weights, np.ones((4, 1)))
    np.testing.assert_array_equal(
        evaluation.weight_gradients, np.zeros((4, 1, 3))
    )


@pytest.mark.parametrize(
    ("centers", "error", "message"),
    [
        ([], ValueError, "shape"),
        (np.empty((0, 3)), ValueError, "at least one"),
        ([[0.0, 0.0]], ValueError, "shape"),
        ([[0.0, 0.0, np.nan]], ValueError, "finite"),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], ValueError, "distinct"),
    ],
)
def test_invalid_center_arrays_are_rejected(centers, error, message):
    with pytest.raises(error, match=message):
        BeckePartition(centers)


@pytest.mark.parametrize("value", [True, 1.5, "3"])
def test_switch_iterations_must_be_an_integer(value):
    with pytest.raises(TypeError, match="integer"):
        BeckePartition([[0.0, 0.0, 0.0]], switch_iterations=value)


def test_switch_iterations_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        BeckePartition([[0.0, 0.0, 0.0]], switch_iterations=0)


@pytest.mark.parametrize(
    "points",
    [np.array(1.0), np.zeros(2), np.zeros((2, 2)), [[0.0, np.inf, 0.0]]],
)
def test_invalid_points_are_rejected(points):
    with pytest.raises(ValueError):
        BeckePartition(NONCOLLINEAR_CENTERS).evaluate(points)


def test_constructor_arrays_are_read_only_and_input_is_copied():
    centers = NONCOLLINEAR_CENTERS.copy()
    partition = BeckePartition(centers)
    centers[:] = 0.0

    np.testing.assert_array_equal(partition.centers, NONCOLLINEAR_CENTERS)
    assert not partition.centers.flags.writeable
    assert not partition.pair_distances.flags.writeable
