import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    DistributedTwoRadial2DState,
    FEMDVRGridSpec,
    RadialSincDVRGridSpec,
    TwoRadialDistributed2DLayout,
    TwoRadialSystemSpec,
    gather_two_radial_2d_state,
    load_distributed_two_radial_2d_state,
    load_two_radial_state,
    phase_align_two_radial_state,
    phase_aligned_two_radial_state_difference,
    save_distributed_two_radial_2d_state,
    scatter_two_radial_2d_state,
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


def test_explicit_process_grid_assigns_cartesian_tiles_and_neighbors():
    system = _system()
    layout = TwoRadialDistributed2DLayout(system, 4, process_grid=(2, 2))

    assert layout.global_shape == system.psi_shape
    assert layout.process_grid == (2, 2)
    assert len(layout.layout_id) == 64
    assert len(layout.representation_id) == 64
    assert layout.archive_metadata["decomposition"] == "cartesian-2d"
    assert layout.archive_metadata["process_grid"] == (2, 2)
    assert [part.process_coordinates for part in layout.partitions] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert layout.rank_at((1, 0)) == 2
    assert layout.process_coordinates(3) == (1, 1)
    assert layout.coordinate1_line_ranks(0) == (0, 2)
    assert layout.coordinate2_line_ranks(0) == (0, 1)

    rank0 = layout.partition(0)
    assert rank0.coordinate1_neighbor_ranks == (2,)
    assert rank0.coordinate2_neighbor_ranks == (1,)
    assert rank0.neighbor_ranks == (2, 1)
    assert rank0.element_tiles == tuple(
        (element1, element2)
        for element1 in rank0.element_range1
        for element2 in rank0.element_range2
    )
    assert layout.owned_shape(0)[-2:] == (
        rank0.coordinate1.n_owned,
        rank0.coordinate2.n_owned,
    )
    assert layout.local_shape(0)[-2:] == (
        rank0.coordinate1.n_local,
        rank0.coordinate2.n_local,
    )

    coverage = np.zeros(
        (system.grid1.n_points, system.grid2.n_points),
        dtype=int,
    )
    for partition in layout.partitions:
        coverage[
            partition.owned_indices1[:, None],
            partition.owned_indices2[None, :],
        ] += 1
    np.testing.assert_array_equal(coverage, 1)


def test_bridge_and_halo_ownership_are_available_in_both_coordinates():
    layout = TwoRadialDistributed2DLayout(
        _system(),
        4,
        process_grid=(2, 2),
    )

    for coordinate_layout in (
        layout.coordinate1_layout,
        layout.coordinate2_layout,
    ):
        owned = np.concatenate(
            [partition.owned_indices for partition in coordinate_layout.partitions]
        )
        bridges = np.concatenate(
            [
                partition.owned_bridge_indices
                for partition in coordinate_layout.partitions
            ]
        )
        np.testing.assert_array_equal(
            np.sort(owned),
            np.arange(coordinate_layout.partitioned_grid.n_points),
        )
        np.testing.assert_array_equal(
            np.sort(bridges),
            coordinate_layout.bridge_indices,
        )
        for partition in coordinate_layout.partitions:
            assert set(partition.halo_owner_ranks).issubset(
                partition.neighbor_ranks
            )

    rank3 = layout.partition(3)
    np.testing.assert_array_equal(
        rank3.halo_indices1,
        rank3.coordinate1.halo_indices,
    )
    np.testing.assert_array_equal(
        rank3.halo_indices2,
        rank3.coordinate2.halo_indices,
    )
    assert set(rank3.halo_owner_ranks1).issubset(
        rank3.coordinate1_neighbor_ranks
    )
    assert set(rank3.halo_owner_ranks2).issubset(
        rank3.coordinate2_neighbor_ranks
    )
    with pytest.raises(ValueError, match="read-only"):
        rank3.halo_owner_ranks1[0] = 0


def test_automatic_process_grid_handles_unequal_grids_and_rank_counts():
    system = _system()
    prime_layout = TwoRadialDistributed2DLayout(system, 3)
    automatic = TwoRadialDistributed2DLayout(system, 4)
    alternatives = [
        TwoRadialDistributed2DLayout(system, 4, process_grid=process_grid)
        for process_grid in ((1, 4), (2, 2))
    ]

    assert prime_layout.process_grid == (1, 3)
    assert automatic.process_grid in ((1, 4), (2, 2))
    assert np.max(automatic.estimated_rank_costs) == min(
        np.max(layout.estimated_rank_costs) for layout in alternatives
    )
    assert not automatic.estimated_rank_costs.flags.writeable


@pytest.mark.parametrize("M_tot", [None, 0])
def test_distribute_and_assemble_preserve_full_and_compact_states(M_tot):
    system = _system(M_tot=M_tot)
    layout = TwoRadialDistributed2DLayout(system, 4, process_grid=(2, 2))
    state = _state(system)

    distributed = layout.distribute(state)

    assert [local.rank for local in distributed] == [0, 1, 2, 3]
    for rank, local in enumerate(distributed):
        partition = layout.partition(rank)
        expected = np.take(state, partition.owned_indices1, axis=-2)
        expected = np.take(expected, partition.owned_indices2, axis=-1)
        np.testing.assert_array_equal(local.coefficients, expected)
        np.testing.assert_array_equal(
            local.owned_r,
            system.grid1.r[partition.owned_indices1],
        )
        np.testing.assert_array_equal(
            local.owned_R,
            system.grid2.r[partition.owned_indices2],
        )
    np.testing.assert_array_equal(layout.assemble(distributed), state)


def test_in_memory_redistribution_between_process_grids_is_exact():
    system = _system()
    source = TwoRadialDistributed2DLayout(system, 4, process_grid=(1, 4))
    target = TwoRadialDistributed2DLayout(system, 4, process_grid=(2, 2))
    state = _state(system)

    redistributed = source.redistribute(source.distribute(state), target)

    np.testing.assert_array_equal(target.assemble(redistributed), state)
    assert all(local.layout.layout_id == target.layout_id for local in redistributed)


def test_object_collectives_scatter_and_gather_four_cartesian_ranks():
    system = _system()
    layout = TwoRadialDistributed2DLayout(system, 4, process_grid=(2, 2))
    state = _state(system)
    group = _ObjectCollectiveGroup(4)
    communicators = [group.communicator(rank) for rank in range(4)]

    local_states = [
        scatter_two_radial_2d_state(layout, communicators[0], state),
        *[
            scatter_two_radial_2d_state(layout, communicators[rank])
            for rank in range(1, 4)
        ],
    ]

    for rank in range(1, 4):
        assert gather_two_radial_2d_state(
            local_states[rank],
            communicators[rank],
        ) is None
    gathered = gather_two_radial_2d_state(local_states[0], communicators[0])
    np.testing.assert_array_equal(gathered, state)


def test_serial_collectives_preserve_existing_archive_format(tmp_path):
    system = _system()
    layout = TwoRadialDistributed2DLayout(system, 1, process_grid=(1, 1))
    communicator = _SerialCommunicator()
    state = _state(system)
    local = scatter_two_radial_2d_state(layout, communicator, state)
    archive_path = tmp_path / "cartesian-state.npz"

    returned = save_distributed_two_radial_2d_state(
        archive_path,
        local,
        communicator,
        physical_state=_physical_state(),
        calculation={"time": 2.0},
    )
    loaded = load_distributed_two_radial_2d_state(
        archive_path,
        layout,
        communicator,
    )

    assert returned == archive_path
    np.testing.assert_array_equal(loaded.coefficients, state)
    archived = load_two_radial_state(archive_path)
    assert archived.coefficients.shape == system.psi_shape
    assert archived.calculation["time"] == 2.0
    assert archived.calculation["distributed_layout"] == {
        **layout.archive_metadata,
        "process_grid": [1, 1],
    }


def test_phase_aligned_equivalence_helpers_remove_global_phase():
    reference = _state(_system())
    phase = np.exp(0.37j)
    candidate = reference * phase

    aligned = phase_align_two_radial_state(reference, candidate)

    np.testing.assert_allclose(aligned, reference, rtol=1.0e-14, atol=1.0e-12)
    assert phase_aligned_two_radial_state_difference(
        reference,
        candidate,
    ) < 1.0e-10


def test_layout_and_state_validation_reject_invalid_inputs():
    system = _system()

    with pytest.raises(ValueError, match="requires FEM-DVR grid1"):
        TwoRadialDistributed2DLayout(_system(sinc=True), 1)
    with pytest.raises(TypeError, match="n_ranks must be an integer"):
        TwoRadialDistributed2DLayout(system, True)
    with pytest.raises(ValueError, match="n_ranks must be positive"):
        TwoRadialDistributed2DLayout(system, 0)
    with pytest.raises(ValueError, match="product"):
        TwoRadialDistributed2DLayout(system, 4, process_grid=(1, 2))
    with pytest.raises(ValueError, match="grid1 elements"):
        TwoRadialDistributed2DLayout(system, 4, process_grid=(4, 1))
    with pytest.raises(ValueError, match="cannot form a process grid"):
        TwoRadialDistributed2DLayout(system, 11)
    with pytest.raises(ValueError, match="process_coordinates"):
        TwoRadialDistributed2DLayout(
            system,
            4,
            process_grid=(2, 2),
        ).rank_at((2, 0))

    layout = TwoRadialDistributed2DLayout(system, 4, process_grid=(2, 2))
    states = layout.distribute(_state(system))
    with pytest.raises(ValueError, match="expected 4"):
        layout.assemble(states[:3])
    with pytest.raises(ValueError, match="duplicate"):
        layout.assemble((states[0], states[0], states[2], states[3]))
    with pytest.raises(ValueError, match="coefficients have shape"):
        DistributedTwoRadial2DState(layout, 0, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="state shapes differ"):
        phase_aligned_two_radial_state_difference(
            np.zeros(2),
            np.zeros(3),
        )
