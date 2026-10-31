import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    get_size_NBO_Psi,
)


ELECTRONIC_GRID = FEMDVRGridSpec([0, 4, 8, 12, 16], [11] * 4)
INTERNUCLEAR_GRID = ([0, 2, 4, 6, 8], 21)


def test_h2p_storage_matches_production_channel_and_radial_counts():
    sizes = get_size_NBO_Psi(
        "H2+",
        ELECTRONIC_GRID,
        INTERNUCLEAR_GRID,
        l_max=2,
        m_max=2,
    )

    assert sizes.radial_shape == (39, 79)
    assert sizes.ground_state.shape == (8, 39, 79)
    assert sizes.ground_state.n_coefficients == 24_648
    assert sizes.ground_state.dtype == "float64"
    assert sizes.ground_state.n_bytes == 24_648 * 8

    assert sizes.time_dependent_state.shape == (12, 39, 79)
    assert sizes.time_dependent_state.n_coefficients == 36_972
    assert sizes.time_dependent_state.dtype == "complex128"
    assert sizes.time_dependent_state.n_bytes == 36_972 * 16

    assert sizes.unrestricted_real_state.shape == (9, 9, 39, 79)
    assert sizes.unrestricted_real_state.n_coefficients == 249_561
    assert sizes.unrestricted_complex_state.n_bytes == 249_561 * 16


def test_hd_has_no_pair_exchange_reduction():
    sizes = get_size_NBO_Psi(
        "HD+",
        ([0, 2], 4),
        ([0, 3], [5]),
        l_max=2,
    )

    assert sizes.radial_shape == (2, 3)
    assert sizes.pair_exchange_parity is None
    assert sizes.ground_state.n_angular_pairs == 19
    assert sizes.time_dependent_state.n_angular_pairs == 19
    assert sizes.ground_state.n_coefficients == 19 * 2 * 3


@pytest.mark.parametrize("symmetry_adapted_basis", [False, None])
def test_disabled_symmetry_adaptation_retains_full_ground_coordinate_bases(
    symmetry_adapted_basis,
):
    sizes = get_size_NBO_Psi(
        "H2+",
        ([0, 2], 4),
        ([0, 2], 4),
        l_max=2,
        symmetry_adapted_basis=symmetry_adapted_basis,
    )

    assert sizes.symmetry_adapted_basis is False
    assert sizes.ground_state.n_angular_pairs == 19
    # The TDSE still exploits nuclear pair exchange, while adding both
    # electronic parities for a z-polarized field.
    assert sizes.time_dependent_state.n_angular_pairs == 12
    assert sizes.unrestricted_real_state.n_angular_pairs == 81


def test_none_M_tot_uses_cartesian_storage_for_selected_bases():
    sizes = get_size_NBO_Psi(
        "H2+",
        ([0, 2], 4),
        ([0, 2], 4),
        l_max=2,
        M_tot=None,
    )

    assert sizes.ground_state.shape == (6, 6, 2, 2)
    assert sizes.time_dependent_state.shape == (9, 6, 2, 2)
    assert sizes.unrestricted_real_state.shape == (9, 9, 2, 2)


def test_sizes_report_decimal_and_binary_units_and_format_cleanly():
    sizes = get_size_NBO_Psi(
        "D2+",
        ([0, 2], 4),
        ([0, 2], 4),
        l_max=1,
    )

    storage = sizes.ground_state
    assert storage.gigabytes == pytest.approx(storage.n_bytes / 1e9)
    assert storage.gibibytes == pytest.approx(storage.n_bytes / 1024**3)
    rendered = str(sizes)
    assert "Real ground state" in rendered
    assert "Complex time-dependent state" in rendered
    assert "GB" in rendered
    assert "GiB" in rendered


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"system": "LiH+"}, ValueError, "system"),
        ({"m_max": 3}, ValueError, "m_max"),
        ({"M_tot": True}, TypeError, "M_tot"),
        ({"inversion_parity": 0}, ValueError, "inversion_parity"),
        (
            {"system": "HD+", "pair_exchange_parity": 1},
            ValueError,
            "not defined",
        ),
    ],
)
def test_storage_input_validation(kwargs, error, message):
    arguments = {
        "system": "H2+",
        "electronic_grid": ([0, 2], 4),
        "internuclear_grid": ([0, 2], 4),
        "l_max": 2,
    }
    arguments.update(kwargs)

    with pytest.raises(error, match=message):
        get_size_NBO_Psi(**arguments)


def test_nonzero_M_tot_counts_only_the_requested_sector():
    sizes = get_size_NBO_Psi(
        "HD+",
        ([0, 2], 4),
        ([0, 2], 4),
        l_max=2,
        M_tot=1,
    )

    assert sizes.ground_state.n_angular_pairs == 16
    assert np.prod(sizes.ground_state.shape) == sizes.ground_state.n_coefficients
