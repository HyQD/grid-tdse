import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    DistributedTwoRadialState,
    FEMDVRGridSpec,
    RadialSincDVRGridSpec,
    TwoRadialDistributedLayout,
    TwoRadialSystemSpec,
    gather_two_radial_state,
    load_distributed_two_radial_state,
    load_two_radial_state,
    save_distributed_two_radial_state,
    save_two_radial_state,
    scatter_two_radial_state,
)


def _system(*, M_tot=0, sinc=False):
    if sinc:
        grid1 = RadialSincDVRGridSpec(5.0, 7)
        grid2 = RadialSincDVRGridSpec(8.0, 9)
    else:
        grid1 = FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4])
        grid2 = FEMDVRGridSpec(
            [0.0, 1.0, 3.0, 6.0, 10.0],
            [4, 5, 3, 4],
        )
    basis = AngularBasisSpec(1, 1)
    return TwoRadialSystemSpec(
        CoordinateSpec(grid1, basis, -0.5),
        CoordinateSpec(grid2, basis, -0.25),
        CoulombMultipoleSpec(0, 0),
        (CoulombTerm(1.0, 1.0),),
        M_tot=M_tot,
    ).build()


def _state(system):
    values = np.arange(np.prod(system.psi_shape), dtype=float).reshape(
        system.psi_shape
    )
    return values + 1j * (values + 0.25)


def _physical_state():
    return {
        "system_family": "non_bo_diatomic_ion",
        "system": "H2+",
        "symmetry": {"M_tot": 0},
    }


class _SerialCommunicator:
    rank = 0
    size = 1

    def scatter(self, values, root=0):
        assert root == 0
        return values[0]

    def gather(self, value, root=0):
        assert root == 0
        return [value]

    def bcast(self, value, root=0):
        assert root == 0
        return value


class _ObjectCollectiveGroup:
    def __init__(self, size):
        self.size = size
        self.broadcast = None
        self.scatter_values = None
        self.gather_values = [None] * size

    def communicator(self, rank):
        return _ObjectCommunicator(self, rank)


class _ObjectCommunicator:
    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.size = group.size

    def scatter(self, values, root=0):
        if self.rank == root:
            self.group.scatter_values = values
        return self.group.scatter_values[self.rank]

    def gather(self, value, root=0):
        self.group.gather_values[self.rank] = value
        if self.rank == root:
            return list(self.group.gather_values)
        return None

    def bcast(self, value, root=0):
        if self.rank == root:
            self.group.broadcast = value
        return self.group.broadcast


def test_layout_assigns_element_aligned_owned_and_halo_ranges():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 3)

    assert layout.global_shape == system.psi_shape
    assert layout.n_R == system.grid2.n_points
    assert len(layout.layout_id) == 64
    assert [
        (partition.element_start, partition.element_stop)
        for partition in layout.partitions
    ] == [(0, 1), (1, 2), (2, 4)]
    np.testing.assert_array_equal(layout.partition(0).owned_indices, [0, 1, 2])
    np.testing.assert_array_equal(layout.partition(1).owned_indices, [3, 4, 5, 6])
    np.testing.assert_array_equal(layout.partition(2).owned_indices, [7, 8, 9, 10])
    np.testing.assert_array_equal(layout.partition(0).halo_indices, [])
    np.testing.assert_array_equal(layout.partition(1).halo_indices, [2])
    np.testing.assert_array_equal(layout.partition(2).halo_indices, [6])
    np.testing.assert_array_equal(layout.bridge_indices, [2, 6, 8])
    np.testing.assert_array_equal(
        layout.R_owner_ranks,
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    )
    np.testing.assert_array_equal(layout.partition(0).owned_bridge_indices, [2])
    np.testing.assert_array_equal(layout.partition(1).owned_bridge_indices, [6])
    np.testing.assert_array_equal(layout.partition(2).owned_bridge_indices, [8])
    np.testing.assert_array_equal(layout.partition(1).halo_owner_ranks, [0])
    np.testing.assert_array_equal(layout.partition(2).halo_owner_ranks, [1])
    assert layout.partition(0).neighbor_ranks == (1,)
    assert layout.partition(1).neighbor_ranks == (0, 2)
    assert layout.partition(2).neighbor_ranks == (1,)
    assert layout.partition(1).global_owned_slice == slice(3, 7)
    assert layout.partition(1).global_local_slice == slice(2, 7)

    owned = np.concatenate(
        [partition.owned_indices for partition in layout.partitions]
    )
    np.testing.assert_array_equal(np.sort(owned), np.arange(system.grid2.n_points))
    assert np.unique(owned).size == owned.size
    with pytest.raises(ValueError):
        layout.partition(3)
    with pytest.raises(ValueError, match="read-only"):
        layout.partition(0).owned_indices[0] = 9


@pytest.mark.parametrize("M_tot", [None, 0])
def test_distribute_and_assemble_preserve_full_and_compact_states(M_tot):
    system = _system(M_tot=M_tot)
    layout = TwoRadialDistributedLayout(system, 3)
    state = _state(system)

    distributed = layout.distribute(state)

    assert [local.rank for local in distributed] == [0, 1, 2]
    for rank, local in enumerate(distributed):
        assert local.coefficients.shape == layout.owned_shape(rank)
        assert layout.local_shape(rank)[-1] == layout.partition(rank).n_local
        np.testing.assert_array_equal(
            local.coefficients,
            state[..., layout.partition(rank).owned_indices],
        )
        np.testing.assert_array_equal(
            local.owned_R,
            system.grid2.r[layout.partition(rank).owned_indices],
        )
    np.testing.assert_array_equal(layout.assemble(distributed), state)


def test_layout_element_maps_address_owned_plus_halo_work_array():
    layout = TwoRadialDistributedLayout(_system(), 3)
    partition = layout.partition(2)
    global_values = np.arange(layout.n_R)
    local_values = global_values[partition.local_indices]

    element_values = [
        local_values[local_map] for local_map in partition.element_local_dof_maps
    ]

    np.testing.assert_array_equal(element_values[0], [6, 7, 8])
    np.testing.assert_array_equal(element_values[1], [8, 9, 10])
    np.testing.assert_array_equal(
        partition.local_indices[partition.owned_local_indices],
        partition.owned_indices,
    )
    np.testing.assert_array_equal(
        partition.local_indices[partition.halo_local_indices],
        partition.halo_indices,
    )


def test_archived_state_can_be_distributed_without_conversion(tmp_path):
    system = _system()
    layout = TwoRadialDistributedLayout(system, 2)
    state = _state(system)
    archive_path = tmp_path / "state.npz"
    save_two_radial_state(
        archive_path,
        state,
        system,
        physical_state=_physical_state(),
    )

    distributed = layout.distribute(load_two_radial_state(archive_path))

    np.testing.assert_array_equal(layout.assemble(distributed), state)


def test_serial_collectives_and_archive_round_trip(tmp_path):
    system = _system()
    layout = TwoRadialDistributedLayout(system, 1)
    communicator = _SerialCommunicator()
    state = _state(system)

    local = scatter_two_radial_state(layout, communicator, state)
    gathered = gather_two_radial_state(local, communicator)
    np.testing.assert_array_equal(gathered, state)

    archive_path = tmp_path / "distributed-state.npz"
    returned = save_distributed_two_radial_state(
        archive_path,
        local,
        communicator,
        physical_state=_physical_state(),
        calculation={"time": 2.0},
    )
    assert returned == archive_path

    loaded = load_distributed_two_radial_state(
        archive_path,
        layout,
        communicator,
    )
    np.testing.assert_array_equal(loaded.coefficients, state)
    assert load_two_radial_state(archive_path).calculation["time"] == 2.0


def test_object_collectives_scatter_and_gather_multiple_ranks():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 3)
    state = _state(system)
    group = _ObjectCollectiveGroup(3)
    communicators = [group.communicator(rank) for rank in range(3)]

    local_states = [
        scatter_two_radial_state(layout, communicators[0], state),
        scatter_two_radial_state(layout, communicators[1]),
        scatter_two_radial_state(layout, communicators[2]),
    ]

    assert gather_two_radial_state(local_states[1], communicators[1]) is None
    assert gather_two_radial_state(local_states[2], communicators[2]) is None
    gathered = gather_two_radial_state(local_states[0], communicators[0])
    np.testing.assert_array_equal(gathered, state)


def test_distributed_state_validation_and_copy():
    layout = TwoRadialDistributedLayout(_system(), 2)
    state = layout.distribute(_state(layout.representation))[0]

    copied = state.copy()

    np.testing.assert_array_equal(copied.coefficients, state.coefficients)
    assert copied.coefficients is not state.coefficients
    with pytest.raises(ValueError, match="coefficients have shape"):
        DistributedTwoRadialState(layout, 0, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="communicator has"):
        scatter_two_radial_state(layout, _SerialCommunicator(), None)


def test_layout_rejects_unsupported_or_invalid_decompositions():
    system = _system()

    with pytest.raises(ValueError, match="requires FEM-DVR grid2"):
        TwoRadialDistributedLayout(_system(sinc=True), 1)
    with pytest.raises(ValueError, match="must be positive"):
        TwoRadialDistributedLayout(system, 0)
    with pytest.raises(ValueError, match="exceeds the 4 FEM-DVR elements"):
        TwoRadialDistributedLayout(system, 5)
    with pytest.raises(TypeError, match="must be an integer"):
        TwoRadialDistributedLayout(system, True)


def test_assemble_rejects_missing_duplicate_and_incompatible_ranks():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 2)
    states = layout.distribute(_state(system))

    with pytest.raises(ValueError, match="expected 2"):
        layout.assemble(states[:1])
    with pytest.raises(ValueError, match="duplicate"):
        layout.assemble((states[0], states[0]))

    different = TwoRadialDistributedLayout(system, 1).distribute(_state(system))[0]
    with pytest.raises(ValueError, match="different distributed layout"):
        layout.assemble((states[0], different))
