import json

import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    FEMDVRGridSpec,
    TwoRadialSystemSpec,
    load_two_radial_state,
    save_two_radial_state,
    two_radial_compatibility_metadata,
)


def _system(*, parity=None, l_max=2, second_nodes=(0.0, 1.5, 4.0)):
    grid1 = FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4])
    grid2 = FEMDVRGridSpec(second_nodes, [4, 4])
    basis = AngularBasisSpec(l_max, l_max, coordinate_parity=parity)
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(grid1, basis, -0.5),
        coordinate2=CoordinateSpec(grid2, basis, -0.25),
        multipoles=CoulombMultipoleSpec(0, 0),
        coulomb_terms=(CoulombTerm(1.0, 1.0),),
    ).build()


def _physical_state():
    return {
        "system_family": "test_system",
        "system": "test",
        "state_index": 0,
        "parameters": {"charge": 1.0},
        "symmetry": {"inversion_parity": 1},
    }


def test_state_archive_round_trip_uses_exact_user_path(tmp_path):
    system = _system(parity=1)
    rng = np.random.default_rng(17)
    state = rng.normal(size=system.psi_shape) + 1j * rng.normal(
        size=system.psi_shape
    )
    path = tmp_path / "filename-chosen-by-user"

    returned = save_two_radial_state(
        path,
        state,
        system,
        physical_state=_physical_state(),
        calculation={"energy": -1.25, "converged": True},
    )
    loaded = load_two_radial_state(path)

    assert returned == path
    assert path.is_file()
    assert not (tmp_path / "filename-chosen-by-user.npz").exists()
    np.testing.assert_array_equal(loaded.coefficients, state)
    np.testing.assert_array_equal(loaded.radial_points1, system.grid1.r)
    np.testing.assert_array_equal(loaded.radial_weights2, system.grid2.weights)
    assert tuple(map(tuple, loaded.angular_states1)) == system.basis1.states
    assert loaded.physical_state == _physical_state()
    assert loaded.calculation["energy"] == -1.25
    assert loaded.grid1_specification == system.grid1.specification
    assert loaded.grid2_specification == system.grid2.specification
    assert loaded.basis1 == system.basis1
    assert loaded.basis2 == system.basis2
    assert loaded.multipoles == system.multipoles
    assert len(loaded.representation_id) == 64
    assert len(loaded.state_data_id) == 64
    loaded.validate_representation(system)
    np.testing.assert_array_equal(loaded.state_for(system), state)


def test_state_archive_requires_explicit_overwrite(tmp_path):
    system = _system(parity=1)
    state = np.zeros(system.psi_shape)
    path = tmp_path / "state"
    save_two_radial_state(
        path, state, system, physical_state=_physical_state()
    )

    with pytest.raises(FileExistsError, match="already exists"):
        save_two_radial_state(
            path, state + 1.0, system, physical_state=_physical_state()
        )

    save_two_radial_state(
        path,
        state + 1.0,
        system,
        physical_state=_physical_state(),
        overwrite=True,
    )
    np.testing.assert_array_equal(
        load_two_radial_state(path).coefficients, state + 1.0
    )


def test_reduced_state_embeds_into_full_angular_basis(tmp_path):
    source = _system(parity=1)
    target = _system(parity=None)
    rng = np.random.default_rng(31)
    state = rng.normal(size=source.psi_shape)
    path = tmp_path / "reduced-state"
    save_two_radial_state(
        path, state, source, physical_state=_physical_state()
    )
    loaded = load_two_radial_state(path)

    with pytest.raises(ValueError, match="angular state table does not match"):
        loaded.validate_representation(target)

    embedded = loaded.embed_into(target)
    for source1, angular1 in enumerate(source.basis1.states):
        target1 = target.basis1.index(*angular1)
        for source2, angular2 in enumerate(source.basis2.states):
            target2 = target.basis2.index(*angular2)
            np.testing.assert_array_equal(
                embedded[target1, target2], state[source1, source2]
            )
    odd_channels = [
        index for index, (l_value, _) in enumerate(target.basis1.states) if l_value % 2
    ]
    assert np.all(embedded[odd_channels] == 0.0)
    assert target.inner_product(embedded, embedded) == pytest.approx(
        source.inner_product(state, state)
    )


def test_archive_rejects_different_radial_grid(tmp_path):
    source = _system(parity=1)
    target = _system(parity=1, second_nodes=(0.0, 1.6, 4.0))
    path = tmp_path / "state"
    save_two_radial_state(
        path,
        np.ones(source.psi_shape),
        source,
        physical_state=_physical_state(),
    )

    with pytest.raises(ValueError, match="coordinate 2 radial grid specification"):
        load_two_radial_state(path).embed_into(target)


def test_archive_rejects_target_missing_source_angular_channels(tmp_path):
    source = _system(parity=None, l_max=2)
    target = _system(parity=None, l_max=1)
    path = tmp_path / "state"
    save_two_radial_state(
        path,
        np.ones(source.psi_shape),
        source,
        physical_state=_physical_state(),
    )

    with pytest.raises(ValueError, match="missing stored channels"):
        load_two_radial_state(path).embed_into(target)


def test_archive_detects_modified_coefficients(tmp_path):
    system = _system(parity=1)
    path = tmp_path / "state"
    save_two_radial_state(
        path,
        np.ones(system.psi_shape),
        system,
        physical_state=_physical_state(),
    )
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name].copy() for name in original.files}
    arrays["state"].flat[0] = 2.0
    with path.open("wb") as stream:
        np.savez(stream, **arrays)

    with pytest.raises(ValueError, match="coefficient checksum"):
        load_two_radial_state(path)


def test_archive_rejects_inconsistent_angular_metadata(tmp_path):
    system = _system(parity=1)
    path = tmp_path / "state"
    save_two_radial_state(
        path,
        np.ones(system.psi_shape),
        system,
        physical_state=_physical_state(),
    )
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: original[name].copy() for name in original.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["representation"]["coordinate1"]["angular_basis"]["states"][0] = [2, 0]
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    with path.open("wb") as stream:
        np.savez(stream, **arrays)

    with pytest.raises(ValueError, match="angular state table is inconsistent"):
        load_two_radial_state(path)


def test_two_radial_checkpoint_metadata_identifies_representation():
    system = _system(parity=1)
    metadata = two_radial_compatibility_metadata(
        system,
        physical_state=_physical_state(),
    )

    assert metadata["physical_state"] == _physical_state()
    assert metadata["state_shape"] == list(system.psi_shape)
    assert len(metadata["representation_id"]) == 64
    assert metadata["representation"]["coordinate1"]["grid"] == {
        "family": "femdvr",
        "nodes": [0.0, 2.0, 5.0],
        "n_points": [4, 4],
    }
    assert metadata == two_radial_compatibility_metadata(
        system,
        physical_state=_physical_state(),
    )
    assert metadata != two_radial_compatibility_metadata(
        _system(parity=1, second_nodes=(0.0, 1.6, 4.0)),
        physical_state=_physical_state(),
    )
