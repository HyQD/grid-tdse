import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    TwoElectronAtomGroundState,
    TwoRadialAbsorbingBoundaries,
)


def _representation(M_tot):
    calculation = TwoElectronAtomGroundState(
        "He",
        grid=FEMDVRGridSpec([0.0, 2.0], [4]),
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=M_tot,
        max_iterations=1,
        use_preconditioner=False,
    )
    return calculation.representation


@pytest.mark.parametrize("M_tot", [None, 0])
def test_masks_multiply_both_radial_axes_for_full_and_compact_storage(M_tot):
    representation = _representation(M_tot)
    mask1 = np.array([0.8, 0.5])
    mask2 = np.array([0.7, 0.4])
    boundaries = TwoRadialAbsorbingBoundaries(
        representation,
        mask1=mask1,
        mask2=mask2,
    )
    state = np.ones(representation.psi_shape, dtype=complex)

    result = boundaries.apply_mask(state, time=1.0)
    expected_radial = mask1[:, None] * mask2[None, :]

    np.testing.assert_allclose(
        result,
        np.broadcast_to(expected_radial, representation.psi_shape),
    )
    np.testing.assert_array_equal(state, np.ones(representation.psi_shape))
    assert boundaries.has_mask
    assert not boundaries.has_cap


@pytest.mark.parametrize("M_tot", [None, 0])
def test_caps_add_actions_on_both_radial_axes(M_tot):
    representation = _representation(M_tot)
    cap1 = -1j * np.array([0.1, 0.2])
    cap2 = -1j * np.array([0.3, 0.4])
    boundaries = TwoRadialAbsorbingBoundaries(
        representation,
        cap1=cap1,
        cap2=cap2,
    )
    state = np.ones(representation.psi_shape, dtype=complex)

    result = boundaries.cap_action(state)

    expected_radial = cap1[:, None] + cap2[None, :]
    np.testing.assert_allclose(
        result,
        np.broadcast_to(expected_radial, representation.psi_shape),
    )
    assert boundaries.has_cap
    assert not boundaries.has_mask


def test_grid_lib_profiles_are_evaluated_on_realized_grid():
    representation = _representation(M_tot=0)
    boundaries = TwoRadialAbsorbingBoundaries(
        representation,
        mask1=RadialCosineMask(r0=1.0),
        cap2=CubicCAP(alpha=0.5, eta=0.25),
    )

    assert boundaries.mask1.shape == representation.grid1.r.shape
    assert boundaries.cap2.shape == representation.grid2.r.shape
    assert np.all((0.0 <= boundaries.mask1) & (boundaries.mask1 <= 1.0))
    assert np.all(boundaries.cap2.imag <= 0.0)
    assert boundaries.metadata["mask1"]["type"] == "RadialCosineMask"
    assert boundaries.metadata["cap2"]["type"] == "CubicCAP"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"mask1": np.ones(2), "cap1": -1j * np.ones(2)}, "cannot both"),
        ({"mask1": np.ones(3)}, "shape"),
        ({"mask1": np.array([1.0, 1.1])}, r"\[0, 1\]"),
        ({"cap1": 1j * np.ones(2)}, "non-positive"),
    ],
)
def test_absorbing_boundaries_validate_profiles(options, message):
    with pytest.raises(ValueError, match=message):
        TwoRadialAbsorbingBoundaries(_representation(M_tot=0), **options)
