import numpy as np
import pytest
from scipy.special import sph_harm_y

from grid_tdse.spherical.one_radial_becke import (
    BeckeMulticenterQuadrature,
    BeckePartition,
    BeckeSphericalBasis,
    FEMDVRQuadratureSpec,
    fixed_m_channels,
    full_lm_channels,
)
from grid_tdse.spherical.one_radial_becke.basis import (
    _FEMDVRCardinalBasis,
    _spherical_harmonics_and_gradients,
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


def test_channel_helpers_define_expected_orderings():
    assert full_lm_channels(2) == (
        (0, 0),
        (1, -1),
        (1, 0),
        (1, 1),
        (2, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (2, 2),
    )
    assert fixed_m_channels(4, -2) == ((2, -2), (3, -2), (4, -2))


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_channel_cutoffs_must_be_integers(value):
    with pytest.raises(TypeError, match="integer"):
        full_lm_channels(value)


def test_fixed_m_channel_validation():
    with pytest.raises(ValueError, match="non-negative"):
        full_lm_channels(-1)
    with pytest.raises(TypeError, match="integer"):
        fixed_m_channels(2, 1.0)
    with pytest.raises(ValueError, match="must not exceed"):
        fixed_m_channels(2, 3)


def test_radial_cardinals_are_identity_at_interior_nodes():
    radial_basis = _FEMDVRCardinalBasis(
        FEMDVRQuadratureSpec([0.0, 0.8, 2.0], [5, 6])
    )
    values, _ = radial_basis.evaluate(radial_basis.nodes)

    np.testing.assert_allclose(values, np.eye(radial_basis.size), atol=2e-14)


def test_radial_values_agree_with_grid_lib_interpolation():
    radial_basis = _FEMDVRCardinalBasis(
        FEMDVRQuadratureSpec([0.0, 0.8, 2.0], [5, 6])
    )
    points = np.array([0.13, 0.61, 0.8, 1.17, 1.93])
    values, _ = radial_basis.evaluate(points)
    reference = radial_basis.dvr.interpolation_matrix(points)[:, 1:-1]

    np.testing.assert_allclose(values, reference, atol=3e-15, rtol=0.0)


def test_radial_cardinal_values_and_derivatives_reproduce_polynomial():
    radial_basis = _FEMDVRCardinalBasis(
        FEMDVRQuadratureSpec([0.0, 1.0, 2.0], [6, 6])
    )
    coefficients = radial_basis.nodes * (2.0 - radial_basis.nodes) * (
        1.0 + 0.3 * radial_basis.nodes
    )
    points = np.array([0.13, 0.47, 0.91, 1.18, 1.63, 1.94])
    values, derivatives = radial_basis.evaluate(points)
    interpolated = values @ coefficients
    differentiated = derivatives @ coefficients
    exact = points * (2.0 - points) * (1.0 + 0.3 * points)
    exact_derivative = 2.0 - 0.8 * points - 0.9 * points**2

    np.testing.assert_allclose(interpolated, exact, atol=2e-14)
    np.testing.assert_allclose(differentiated, exact_derivative, atol=8e-14)


def test_element_hints_retain_distinct_bridge_derivative_traces():
    radial_basis = _FEMDVRCardinalBasis(
        FEMDVRQuadratureSpec([0.0, 1.0, 2.0], [4, 5])
    )
    left_values, left_derivatives = radial_basis.evaluate(
        [1.0], element_hints=[0]
    )
    right_values, right_derivatives = radial_basis.evaluate(
        [1.0], element_hints=[1]
    )

    np.testing.assert_allclose(left_values, right_values, atol=0.0)
    assert not np.allclose(left_derivatives, right_derivatives)
    with pytest.raises(ValueError, match="does not contain"):
        radial_basis.evaluate([1.5], element_hints=[0])


def test_radial_cardinals_vanish_outside_and_at_removed_endpoints():
    radial_basis = _FEMDVRCardinalBasis(
        FEMDVRQuadratureSpec([0.0, 1.0, 2.0], [4, 5])
    )
    values, _ = radial_basis.evaluate([-0.1, 0.0, 2.0, 2.1])

    np.testing.assert_allclose(values, 0.0, atol=0.0)


@pytest.mark.parametrize(
    "unit_vectors",
    [
        np.array([[0.3, -0.4, np.sqrt(0.75)]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
    ],
)
def test_cartesian_angular_gradients_agree_with_finite_differences(unit_vectors):
    channels = full_lm_channels(3)
    values, gradients = _spherical_harmonics_and_gradients(
        unit_vectors, channels
    )
    step = 2e-7
    numerical = np.empty_like(gradients)
    for component in range(3):
        displacement = np.zeros(3)
        displacement[component] = step
        plus_vectors = unit_vectors + displacement
        plus_vectors /= np.linalg.norm(plus_vectors, axis=-1)[:, np.newaxis]
        minus_vectors = unit_vectors - displacement
        minus_vectors /= np.linalg.norm(minus_vectors, axis=-1)[:, np.newaxis]
        plus = np.column_stack(
            [
                sph_harm_y(
                    ell,
                    m,
                    np.arctan2(
                        np.hypot(plus_vectors[:, 0], plus_vectors[:, 1]),
                        plus_vectors[:, 2],
                    ),
                    np.mod(
                        np.arctan2(plus_vectors[:, 1], plus_vectors[:, 0]),
                        2 * np.pi,
                    ),
                )
                for ell, m in channels
            ]
        )
        minus = np.column_stack(
            [
                sph_harm_y(
                    ell,
                    m,
                    np.arctan2(
                        np.hypot(minus_vectors[:, 0], minus_vectors[:, 1]),
                        minus_vectors[:, 2],
                    ),
                    np.mod(
                        np.arctan2(minus_vectors[:, 1], minus_vectors[:, 0]),
                        2 * np.pi,
                    ),
                )
                for ell, m in channels
            ]
        )
        numerical[..., component] = (plus - minus) / (2 * step)

    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))
    np.testing.assert_allclose(gradients, numerical, atol=2e-8, rtol=2e-7)


def test_basis_metadata_shapes_and_center_specific_channels():
    partition = BeckePartition(
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
    )
    specifications = (
        FEMDVRQuadratureSpec([0.0, 2.0], [5]),
        FEMDVRQuadratureSpec([0.0, 1.0, 2.5], [4, 5]),
    )
    channels = (fixed_m_channels(2, 0), full_lm_channels(1))
    basis = BeckeSphericalBasis(partition, specifications, channels)
    points = np.array([[0.3, 0.2, 0.1], [-0.4, 0.5, -0.2]])
    evaluation = basis.evaluate(points)
    expected_size = 3 * 3 + 6 * 4

    assert basis.size == expected_size
    assert len(basis.functions) == expected_size
    assert evaluation.values.shape == (2, expected_size)
    assert evaluation.gradients.shape == (2, expected_size, 3)
    assert basis.functions[0].center_index == 0
    assert basis.functions[-1].center_index == 1


def test_weighted_basis_gradients_agree_with_cartesian_finite_differences():
    centers = np.array([[-0.6, 0.1, -0.4], [0.7, -0.2, 0.5]])
    basis = BeckeSphericalBasis(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 1.5, 3.5], [5, 6]),
        full_lm_channels(1),
    )
    points = np.array([[0.2, 0.3, -0.1], [-0.3, 0.6, 0.4]])
    analytic = basis.evaluate(points)
    numerical = np.empty_like(analytic.gradients)
    step = 1e-6
    for component in range(3):
        displacement = np.zeros(3)
        displacement[component] = step
        plus = basis.evaluate(points + displacement).values
        minus = basis.evaluate(points - displacement).values
        numerical[..., component] = (plus - minus) / (2 * step)

    np.testing.assert_allclose(
        analytic.gradients, numerical, atol=4e-9, rtol=4e-7
    )


def test_s_wave_basis_has_a_finite_direction_independent_origin_limit():
    basis = BeckeSphericalBasis(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec([0.0, 1.0], [6]),
        ((0, 0),),
    )
    epsilon = 1e-7
    values = basis.evaluate(
        [[epsilon, 0.0, 0.0], [0.0, epsilon, 0.0], [0.0, 0.0, epsilon]]
    ).values

    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], values[1], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(values[0], values[2], atol=2e-6, rtol=2e-6)


def test_partition_suppresses_other_center_basis_at_a_nuclear_limit():
    centers = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    basis = BeckeSphericalBasis(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 2.5], [7]),
        ((0, 0),),
    )
    point = centers[0] + np.array([1e-6, 0.0, 0.0])
    values = basis.evaluate(point).values
    center_zero = np.array(
        [function.center_index == 0 for function in basis.functions]
    )

    assert np.linalg.norm(values[center_zero]) > 0.0
    assert np.linalg.norm(values[~center_zero]) < 1e-25


def test_basis_values_satisfy_outer_radial_boundary_condition():
    centers = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    basis = BeckeSphericalBasis(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 1.0, 2.0], [4, 5]),
        full_lm_channels(1),
    )
    points = np.array(
        [centers[0] + np.array([2.0, 0.0, 0.0]), centers[1] + np.array([2.0, 0.0, 0.0])]
    )
    values = basis.evaluate(points).values

    center_zero_columns = [
        index
        for index, function in enumerate(basis.functions)
        if function.center_index == 0
    ]
    center_one_columns = [
        index
        for index, function in enumerate(basis.functions)
        if function.center_index == 1
    ]
    np.testing.assert_allclose(values[0, center_zero_columns], 0.0, atol=2e-15)
    np.testing.assert_allclose(values[1, center_one_columns], 0.0, atol=2e-15)


def test_common_frame_rotates_covariantly_with_geometry():
    centers = np.array([[-0.6, 0.1, -0.4], [0.7, -0.2, 0.5]])
    points = np.array([[0.2, 0.3, -0.1], [-0.3, 0.6, 0.4]])
    rotation = _rotation([0.4, -0.7, 1.2], 0.73)
    specification = FEMDVRQuadratureSpec([0.0, 1.5, 3.5], [5, 6])
    reference = BeckeSphericalBasis(
        BeckePartition(centers), specification, full_lm_channels(2)
    ).evaluate(points)
    transformed = BeckeSphericalBasis(
        BeckePartition(centers @ rotation.T),
        specification,
        full_lm_channels(2),
        frame=rotation,
    ).evaluate(points @ rotation.T)

    np.testing.assert_allclose(transformed.values, reference.values, atol=2e-14)
    np.testing.assert_allclose(
        transformed.gradients, reference.gradients @ rotation.T, atol=5e-14
    )


def test_noncollinear_three_center_full_lm_evaluation():
    centers = np.array(
        [[-0.8, 0.1, 0.2], [0.7, -0.4, 0.3], [0.2, 1.1, -0.5]]
    )
    basis = BeckeSphericalBasis(
        BeckePartition(centers),
        FEMDVRQuadratureSpec([0.0, 1.5, 3.0], [4, 5]),
        full_lm_channels(2),
    )
    evaluation = basis.evaluate([[0.1, 0.2, 0.7]])

    assert basis.angular_channels == (full_lm_channels(2),) * 3
    assert np.all(np.isfinite(evaluation.values))
    assert np.all(np.isfinite(evaluation.gradients))


def test_shell_evaluation_uses_each_element_side_of_a_bridge():
    partition = BeckePartition([[0.0, 0.0, 0.0]])
    specification = FEMDVRQuadratureSpec([0.0, 1.0, 2.0], [4, 5])
    quadrature = BeckeMulticenterQuadrature(partition, specification, 5)
    basis = BeckeSphericalBasis(partition, specification, ((0, 0),))
    bridge_shells = [
        shell for shell in quadrature.iter_shells() if shell.radius == 1.0
    ]
    left = basis.evaluate_shell(bridge_shells[0])
    right = basis.evaluate_shell(bridge_shells[1])

    np.testing.assert_allclose(left.values, right.values, atol=0.0)
    assert not np.allclose(left.gradients, right.gradients)


def test_basis_input_validation():
    partition = BeckePartition([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    specification = FEMDVRQuadratureSpec([0.0, 2.0], [5])

    with pytest.raises(TypeError, match="BeckePartition"):
        BeckeSphericalBasis(object(), specification, ((0, 0),))
    with pytest.raises(ValueError, match="one entry per center"):
        BeckeSphericalBasis(partition, [specification], ((0, 0),))
    with pytest.raises(ValueError, match="at least one channel"):
        BeckeSphericalBasis(partition, specification, [[], []])
    with pytest.raises(ValueError, match="must satisfy"):
        BeckeSphericalBasis(partition, specification, ((1, 2),))
    with pytest.raises(ValueError, match="duplicates"):
        BeckeSphericalBasis(partition, specification, ((0, 0), (0, 0)))
    with pytest.raises(ValueError, match="right-handed"):
        BeckeSphericalBasis(
            partition,
            specification,
            ((0, 0),),
            frame=np.diag([1.0, 1.0, -1.0]),
        )

    basis = BeckeSphericalBasis(partition, specification, ((0, 0),))
    with pytest.raises(ValueError, match="must not coincide"):
        basis.evaluate(partition.centers[0])
    with pytest.raises(ValueError, match="element_hints"):
        basis.evaluate([[0.1, 0.2, 0.3]], element_hints=np.zeros((1, 1), int))
