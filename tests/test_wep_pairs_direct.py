import numpy as np

from grid_lib.spherical_coordinates.angular_momentum import get_y, get_ybar
from grid_tdse.spherical.two_radial.hamiltonian import (
    wep_pairs_ij,
    wep_pairs_ij_direct,
)


def _assert_pairs_equal(pairs_ref, pairs_direct):
    gI_ref, gJ_ref, off_ref, Kp_ref, Lp_ref, Mp_ref, Cp_ref = pairs_ref
    gI_dir, gJ_dir, off_dir, Kp_dir, Lp_dir, Mp_dir, Cp_dir = pairs_direct

    assert np.array_equal(gI_ref, gI_dir)
    assert np.array_equal(gJ_ref, gJ_dir)
    assert np.array_equal(off_ref, off_dir)
    assert np.array_equal(Kp_ref, Kp_dir)
    assert np.array_equal(Lp_ref, Lp_dir)
    assert np.array_equal(Mp_ref, Mp_dir)
    assert np.allclose(Cp_ref, Cp_dir, atol=1e-12, rtol=1e-12)


def test_wep_pairs_ij_direct_matches_wep_pairs_ij():
    cases = [
        (2, 2, 2, 2, 4, 4),
        (3, 2, 2, 2, 4, 4),
        (3, 1, 2, 1, 4, 2),
    ]

    for l1_max, m1_max, l2_max, m2_max, L_max, M_max in cases:
        y_bar = get_ybar(l1_max, m1_max, L_max, M_max)
        y = get_y(l2_max, m2_max, L_max, M_max)

        pairs_ref = wep_pairs_ij(y, y_bar)
        pairs_direct = wep_pairs_ij_direct(
            l1_max, m1_max, l2_max, m2_max, L_max, M_max
        )

        _assert_pairs_equal(pairs_ref, pairs_direct)
