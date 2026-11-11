import json
from dataclasses import FrozenInstanceError

import pytest

from grid_tdse.spherical.two_radial import CoupledAngularBasisSpec


def test_coupled_basis_has_deterministic_ordered_states():
    basis = CoupledAngularBasisSpec(
        l1_max=1,
        l2_max=1,
        L_max=2,
        M_tot=0,
    )

    assert basis.states == (
        (0, 0, 0, 0),
        (0, 1, 1, 0),
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (1, 1, 1, 0),
        (1, 1, 2, 0),
    )
    assert basis.n_channels == 6
    assert basis.index(1, 1, 2, 0) == 5
    assert basis.get(0, 1, 1, 0) == 1
    assert basis.get(0, 1, 0, 0) is None


def test_fixed_projection_and_triangle_rules_filter_channels():
    basis = CoupledAngularBasisSpec(
        l1_max=2,
        l2_max=3,
        L_max=4,
        M_tot=-2,
    )

    assert all(M == -2 for _, _, _, M in basis.states)
    assert all(abs(l1 - l2) <= L <= l1 + l2 for l1, l2, L, _ in basis.states)
    assert all(2 <= L <= 4 for _, _, L, _ in basis.states)


@pytest.mark.parametrize("inversion_parity", [-1, 1])
def test_inversion_parity_filters_l1_plus_l2(inversion_parity):
    basis = CoupledAngularBasisSpec(
        l1_max=3,
        l2_max=3,
        L_max=3,
        inversion_parity=inversion_parity,
    )

    assert all(
        (-1 if (l1 + l2) % 2 else 1) == inversion_parity
        for l1, l2, _, _ in basis.states
    )


@pytest.mark.parametrize("pair_exchange_parity", [-1, 1])
def test_pair_exchange_parity_filters_l2(pair_exchange_parity):
    basis = CoupledAngularBasisSpec(
        l1_max=3,
        l2_max=3,
        L_max=3,
        pair_exchange_parity=pair_exchange_parity,
    )

    assert all(
        (-1 if l2 % 2 else 1) == pair_exchange_parity for _, l2, _, _ in basis.states
    )


def test_inversion_and_exchange_sectors_can_be_combined():
    basis = CoupledAngularBasisSpec(
        l1_max=2,
        l2_max=2,
        L_max=2,
        inversion_parity=-1,
        pair_exchange_parity=1,
    )

    assert all(l1 % 2 == 1 for l1, _, _, _ in basis.states)
    assert all(l2 % 2 == 0 for _, l2, _, _ in basis.states)


def test_diatomic_factory_validates_nuclear_exchange_symmetry():
    homonuclear = CoupledAngularBasisSpec.for_diatomic_ion(
        l1_max=2,
        l2_max=2,
        L_max=2,
        has_identical_pair=True,
        pair_exchange_parity=1,
    )
    heteronuclear = CoupledAngularBasisSpec.for_diatomic_ion(
        l1_max=2,
        l2_max=2,
        L_max=2,
        has_identical_pair=False,
    )

    assert homonuclear.pair_exchange_parity == 1
    assert heteronuclear.pair_exchange_parity is None

    with pytest.raises(ValueError, match="distinguishable nuclei"):
        CoupledAngularBasisSpec.for_diatomic_ion(
            l1_max=2,
            l2_max=2,
            L_max=2,
            has_identical_pair=False,
            pair_exchange_parity=1,
        )
    with pytest.raises(TypeError, match="has_identical_pair"):
        CoupledAngularBasisSpec.for_diatomic_ion(
            l1_max=2,
            l2_max=2,
            L_max=2,
            has_identical_pair=1,
        )


def test_coupled_basis_metadata_round_trip_is_json_compatible():
    basis = CoupledAngularBasisSpec(
        l1_max=3,
        l2_max=2,
        L_max=4,
        M_tot=1,
        inversion_parity=-1,
        pair_exchange_parity=1,
    )

    serialized = json.loads(json.dumps(basis.to_metadata(), sort_keys=True))
    restored = CoupledAngularBasisSpec.from_metadata(serialized)

    assert restored == basis
    assert restored.states == basis.states


def test_coupled_basis_public_state_is_immutable():
    basis = CoupledAngularBasisSpec(1, 1, 2)

    with pytest.raises(FrozenInstanceError):
        basis.L_max = 1
    with pytest.raises(TypeError):
        basis.states[0] = (0, 0, 0, 0)


def test_metadata_rejects_tampered_channel_table():
    basis = CoupledAngularBasisSpec(1, 1, 2)
    metadata = basis.to_metadata()
    metadata["states"] = metadata["states"][:-1]

    with pytest.raises(ValueError, match="states disagree"):
        CoupledAngularBasisSpec.from_metadata(metadata)


def test_metadata_requires_supported_type_version_and_fields():
    metadata = CoupledAngularBasisSpec(1, 1, 2).to_metadata()

    with pytest.raises(TypeError, match="mapping"):
        CoupledAngularBasisSpec.from_metadata([])

    wrong_type = metadata | {"type": "uncoupled"}
    with pytest.raises(ValueError, match="does not describe"):
        CoupledAngularBasisSpec.from_metadata(wrong_type)

    wrong_version = metadata | {"version": 2}
    with pytest.raises(ValueError, match="version"):
        CoupledAngularBasisSpec.from_metadata(wrong_version)

    missing_cutoff = dict(metadata)
    del missing_cutoff["l1_max"]
    with pytest.raises(ValueError, match="missing l1_max"):
        CoupledAngularBasisSpec.from_metadata(missing_cutoff)


@pytest.mark.parametrize(
    "states, error, message",
    [
        ("not-a-sequence", TypeError, "sequence"),
        ([[0, 0, 0]], ValueError, "four integers"),
        ([[0, 0, 0, True]], TypeError, "must be integers"),
    ],
)
def test_metadata_rejects_malformed_states(states, error, message):
    metadata = CoupledAngularBasisSpec(0, 0, 0).to_metadata()
    metadata["states"] = states

    with pytest.raises(error, match=message):
        CoupledAngularBasisSpec.from_metadata(metadata)


@pytest.mark.parametrize("name", ["l1_max", "l2_max", "L_max"])
def test_angular_cutoffs_must_be_nonnegative_integers(name):
    arguments = {"l1_max": 1, "l2_max": 1, "L_max": 1}

    arguments[name] = -1
    with pytest.raises(ValueError, match=name):
        CoupledAngularBasisSpec(**arguments)

    arguments[name] = 1.5
    with pytest.raises(TypeError, match=name):
        CoupledAngularBasisSpec(**arguments)

    arguments[name] = True
    with pytest.raises(TypeError, match=name):
        CoupledAngularBasisSpec(**arguments)


def test_invalid_total_angular_limits_are_rejected():
    with pytest.raises(ValueError, match=r"l1_max \+ l2_max"):
        CoupledAngularBasisSpec(1, 1, 3)
    with pytest.raises(ValueError, match=r"abs\(M_tot\)"):
        CoupledAngularBasisSpec(1, 1, 2, M_tot=3)
    with pytest.raises(TypeError, match="M_tot"):
        CoupledAngularBasisSpec(1, 1, 2, M_tot=True)


@pytest.mark.parametrize(
    "keyword",
    ["inversion_parity", "pair_exchange_parity"],
)
def test_invalid_parities_are_rejected(keyword):
    with pytest.raises(ValueError, match=keyword):
        CoupledAngularBasisSpec(1, 1, 2, **{keyword: 0})
    with pytest.raises(ValueError, match=keyword):
        CoupledAngularBasisSpec(1, 1, 2, **{keyword: True})


def test_empty_symmetry_sector_is_rejected():
    with pytest.raises(ValueError, match="no coupled channels"):
        CoupledAngularBasisSpec(
            l1_max=0,
            l2_max=0,
            L_max=0,
            inversion_parity=-1,
        )


def test_lookup_rejects_a_channel_outside_the_basis():
    basis = CoupledAngularBasisSpec(1, 1, 2)

    with pytest.raises(ValueError, match="not retained"):
        basis.index(1, 1, 2, 1)
