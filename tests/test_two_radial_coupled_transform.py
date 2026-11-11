import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    AngularPairBasis,
    CoupledAngularBasisSpec,
    CoupledAngularTransform,
)


def _expectation(values, eigenvalues):
    return np.einsum(
        "c...,c,c...->",
        values.conj(),
        eigenvalues,
        values,
        optimize=True,
    )


def test_complete_fixed_M_transform_is_unitary():
    transform = CoupledAngularTransform(
        CoupledAngularBasisSpec(
            l1_max=2,
            l2_max=1,
            L_max=3,
            M_tot=0,
        )
    )

    assert transform.is_complete
    assert transform.n_coupled == transform.n_uncoupled
    np.testing.assert_allclose(
        transform.coefficients.T @ transform.coefficients,
        np.eye(transform.n_coupled),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        transform.coefficients @ transform.coefficients.T,
        np.eye(transform.n_uncoupled),
        atol=2.0e-15,
    )


def test_transform_round_trip_supports_arbitrary_trailing_axes():
    transform = CoupledAngularTransform(CoupledAngularBasisSpec(2, 2, 4, M_tot=1))
    rng = np.random.default_rng(2051)
    coupled = rng.normal(size=(transform.n_coupled, 2, 3, 4)) + 1j * rng.normal(
        size=(transform.n_coupled, 2, 3, 4)
    )

    uncoupled = transform.to_uncoupled(coupled)
    restored = transform.to_coupled(uncoupled)

    assert uncoupled.shape == (transform.n_uncoupled, 2, 3, 4)
    np.testing.assert_allclose(restored, coupled, atol=3.0e-15, rtol=3.0e-15)


def test_uncoupled_order_matches_existing_compact_pair_storage():
    transform = CoupledAngularTransform(CoupledAngularBasisSpec(2, 1, 3, M_tot=1))
    basis1 = AngularBasisSpec(2)
    basis2 = AngularBasisSpec(1)
    pair_basis = AngularPairBasis(basis1, basis2, M_tot=1)
    expected = tuple(
        (*basis1.states[channel1], *basis2.states[channel2])
        for channel1, channel2 in pair_basis.pairs
    )

    assert transform.uncoupled_states == expected


def test_complete_transform_preserves_inner_products_and_radial_marginals():
    transform = CoupledAngularTransform(CoupledAngularBasisSpec(2, 1, 3))
    rng = np.random.default_rng(321)
    left = rng.normal(size=(transform.n_coupled, 3, 5)) + 1j * rng.normal(
        size=(transform.n_coupled, 3, 5)
    )
    right = rng.normal(size=(transform.n_coupled, 3, 5)) + 1j * rng.normal(
        size=(transform.n_coupled, 3, 5)
    )

    left_uncoupled = transform.to_uncoupled(left)
    right_uncoupled = transform.to_uncoupled(right)

    np.testing.assert_allclose(
        np.vdot(left_uncoupled, right_uncoupled),
        np.vdot(left, right),
        atol=2.0e-13,
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.sum(abs(left_uncoupled) ** 2, axis=0),
        np.sum(abs(left) ** 2, axis=0),
        atol=2.0e-14,
        rtol=2.0e-15,
    )


def test_transform_preserves_inversion_and_pair_exchange_expectations():
    transform = CoupledAngularTransform(CoupledAngularBasisSpec(2, 2, 4))
    rng = np.random.default_rng(711)
    coupled = rng.normal(size=(transform.n_coupled, 4)) + 1j * rng.normal(
        size=(transform.n_coupled, 4)
    )
    uncoupled = transform.to_uncoupled(coupled)

    coupled_inversion = np.asarray(
        [(-1) ** (l1 + l2) for l1, l2, _, _ in transform.basis.states]
    )
    uncoupled_inversion = np.asarray(
        [(-1) ** (l1 + l2) for l1, _, l2, _ in transform.uncoupled_states]
    )
    coupled_exchange = np.asarray(
        [(-1) ** l2 for _, l2, _, _ in transform.basis.states]
    )
    uncoupled_exchange = np.asarray(
        [(-1) ** l2 for _, _, l2, _ in transform.uncoupled_states]
    )

    np.testing.assert_allclose(
        _expectation(uncoupled, uncoupled_inversion),
        _expectation(coupled, coupled_inversion),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        _expectation(uncoupled, uncoupled_exchange),
        _expectation(coupled, coupled_exchange),
        atol=2.0e-13,
    )


def test_symmetry_filtered_transform_is_complete_in_selected_sector():
    transform = CoupledAngularTransform(
        CoupledAngularBasisSpec(
            3,
            2,
            5,
            inversion_parity=-1,
            pair_exchange_parity=1,
        )
    )

    assert transform.is_complete
    assert all(
        (-1) ** (l1 + l2) == -1 and (-1) ** l2 == 1
        for l1, _, l2, _ in transform.uncoupled_states
    )
    np.testing.assert_allclose(
        transform.coefficients @ transform.coefficients.T,
        np.eye(transform.n_uncoupled),
        atol=3.0e-15,
    )


def test_truncated_L_projection_reports_discarded_norm():
    transform = CoupledAngularTransform(CoupledAngularBasisSpec(2, 2, 1))
    rng = np.random.default_rng(919)
    uncoupled = rng.normal(size=(transform.n_uncoupled, 2, 3)) + 1j * rng.normal(
        size=(transform.n_uncoupled, 2, 3)
    )

    projection = transform.project(uncoupled)

    assert not transform.is_complete
    assert projection.coefficients.shape == (transform.n_coupled, 2, 3)
    assert projection.discarded_norm_squared > 0.0
    assert 0.0 < projection.discarded_fraction < 1.0
    assert projection.input_norm_squared == pytest.approx(
        projection.retained_norm_squared + projection.discarded_norm_squared,
        rel=3.0e-15,
        abs=3.0e-14,
    )


def test_transform_validates_basis_and_angular_axis():
    with pytest.raises(TypeError, match="CoupledAngularBasisSpec"):
        CoupledAngularTransform(object())

    transform = CoupledAngularTransform(CoupledAngularBasisSpec(1, 1, 2))
    with pytest.raises(ValueError, match="first axis"):
        transform.to_uncoupled(np.zeros((transform.n_coupled + 1, 2)))
    with pytest.raises(ValueError, match="first axis"):
        transform.to_coupled(np.zeros((transform.n_uncoupled + 1, 2)))
    with pytest.raises(ValueError, match="not retained"):
        transform.uncoupled_index(1, 1, 1, 1)

    with pytest.raises(ValueError):
        transform.coefficients[0, 0] = 1.0
