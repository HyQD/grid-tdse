import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    TwoRadialDistributed2DLayout,
    prepare_distributed_non_bo_diatomic_dynamics_initial_state,
)


GRID = FEMDVRGridSpec([0.0, 1.0, 2.0], [3, 3])
GROUND_STATE_OPTIONS = {
    "max_iterations": 100,
    "maximum_subspace_dimension": 12,
    "residual_tolerance": 1.0e-8,
    "progress_interval": None,
    "sample_interval": None,
}


def _calculation(
    path,
    *,
    policy="compute-if-missing",
    initial_state=None,
    fields=(),
    field_gauge="length",
):
    automatic = initial_state is None
    return NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=initial_state,
        pair_exchange_parity=1,
        initial_state_policy=policy if automatic else "require",
        ground_state_path=path if automatic else None,
        ground_state_options=GROUND_STATE_OPTIONS if automatic else None,
        fields=fields,
        field_gauge=field_gauge,
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        final_time=0.01,
        time_step=0.01,
        integrator_options={"preconditioner": False},
    )


def test_compute_if_missing_and_require_produce_identical_dynamics_state(
    tmp_path,
):
    path = tmp_path / "states" / "h2p.npz"

    computed = _calculation(path)
    loaded = _calculation(path, policy="require")

    assert path.is_file()
    assert computed.initialization.source == "computed"
    assert loaded.initialization.source == "archive"
    assert computed.representation.M_tot == 0
    assert computed.representation.basis1.coordinate_parity is None
    assert computed.representation.basis2.coordinate_parity == 1
    assert computed.initialization.ground_state_wall_time > 0.0
    assert loaded.initialization.ground_state_wall_time is None
    assert computed.dynamics_metadata["initial_state"]["source"] == "computed"
    np.testing.assert_array_equal(computed.initial_state, loaded.initial_state)

    for pair, (index1, _) in enumerate(
        computed.representation.angular_pairs.pairs
    ):
        l_e, _ = computed.representation.basis1.states[index1]
        if l_e % 2:
            assert np.all(computed.initial_state[pair] == 0.0)


@pytest.mark.parametrize(
    ("system", "pair_exchange_parity", "symmetry_adapted_basis"),
    [("HD+", None, False), ("D2+", 1, True)],
)
def test_automatic_initialization_preserves_isotopologue_symmetry(
    tmp_path,
    system,
    pair_exchange_parity,
    symmetry_adapted_basis,
):
    calculation = NonBornOppenheimerDiatomicIonTDSE(
        system,
        electronic_grid=GRID,
        internuclear_grid=GRID,
        pair_exchange_parity=pair_exchange_parity,
        initial_state_policy="compute-if-missing",
        ground_state_path=tmp_path / f"{system}.npz",
        ground_state_options=GROUND_STATE_OPTIONS,
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        final_time=0.01,
        time_step=0.01,
        integrator_options={"preconditioner": False},
    )

    symmetry = calculation.initialization.archive.physical_state["symmetry"]
    assert calculation.representation.basis1.coordinate_parity is None
    assert (
        calculation.representation.basis2.coordinate_parity
        == pair_exchange_parity
    )
    assert symmetry["pair_exchange_parity"] == pair_exchange_parity
    assert symmetry["symmetry_adapted_basis"] is symmetry_adapted_basis


def test_require_rejects_missing_archive_and_recompute_overwrites(tmp_path):
    path = tmp_path / "h2p.npz"
    with pytest.raises(FileNotFoundError, match="required ground-state"):
        _calculation(path, policy="require")

    first = _calculation(path)
    replacement = _calculation(path, policy="recompute")

    assert first.initialization.source == "computed"
    assert replacement.initialization.source == "computed"
    assert replacement.initialization.ground_state_wall_time > 0.0
    np.testing.assert_allclose(
        replacement.initial_state,
        first.initial_state,
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_automatic_initialization_rejects_incompatible_codata(tmp_path):
    path = tmp_path / "h2p.npz"
    _calculation(path)

    with pytest.raises(ValueError, match="CODATA set"):
        NonBornOppenheimerDiatomicIonTDSE(
            "H2+",
            electronic_grid=GRID,
            internuclear_grid=GRID,
            pair_exchange_parity=1,
            ground_state_path=path,
            initial_state_policy="require",
            codata_set="2018",
            l_max=1,
            M_tot=0,
            final_time=0.01,
            time_step=0.01,
            integrator_options={"preconditioner": False},
        )


def test_velocity_gauge_requires_zero_initial_vector_potential(tmp_path):
    archived = _calculation(tmp_path / "h2p.npz").initialization.archive

    class VectorPotential:
        gauge = "velocity"

        def __init__(self, offset):
            self.offset = offset

        def __call__(self, time):
            return self.offset + time

    with pytest.raises(ValueError, match=r"A\(initial_time\)=0"):
        _calculation(
            None,
            initial_state=archived,
            fields=VectorPotential(0.1),
            field_gauge="velocity",
        )

    calculation = _calculation(
        None,
        initial_state=archived,
        fields=VectorPotential(0.0),
        field_gauge="velocity",
    )
    assert calculation.field_gauge == "velocity"


def test_field_rejects_basis_without_odd_electronic_channels(tmp_path):
    archived = _calculation(tmp_path / "h2p.npz").initialization.archive

    class Field:
        gauge = "length"

        def __call__(self, time):
            return 0.1

    with pytest.raises(ValueError, match="both even and odd"):
        NonBornOppenheimerDiatomicIonTDSE(
            "H2+",
            electronic_grid=GRID,
            internuclear_grid=GRID,
            initial_state=archived,
            pair_exchange_parity=1,
            fields=Field(),
            l_max=0,
            M_tot=0,
            final_time=0.01,
            time_step=0.01,
            integrator_options={"preconditioner": False},
        )


class _CollectiveGroup:
    def __init__(self, size):
        self.size = size
        self.broadcasts = []
        self.broadcast_positions = [0] * size
        self.scatter_values = None
        self.barriers = [0] * size

    def communicator(self, rank):
        return _Communicator(self, rank)


class _Communicator:
    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.size = group.size

    def bcast(self, value, root=0):
        position = self.group.broadcast_positions[self.rank]
        if self.rank == root:
            self.group.broadcasts.append(value)
        result = self.group.broadcasts[position]
        self.group.broadcast_positions[self.rank] += 1
        return result

    def scatter(self, values, root=0):
        if self.rank == root:
            self.group.scatter_values = values
        return self.group.scatter_values[self.rank]

    def Barrier(self):
        self.group.barriers[self.rank] += 1


def test_cartesian_initialization_is_prepared_on_root_and_scattered(tmp_path):
    calculation = _calculation(tmp_path / "h2p.npz")
    layout = TwoRadialDistributed2DLayout(
        calculation.representation,
        2,
        process_grid=(1, 2),
    )
    group = _CollectiveGroup(2)
    options = {
        "system": "H2+",
        "electronic_grid": GRID,
        "internuclear_grid": GRID,
        "pair_exchange_parity": 1,
        "archive_path": calculation.initialization.archive_path,
        "policy": "require",
    }

    rank0 = prepare_distributed_non_bo_diatomic_dynamics_initial_state(
        layout,
        group.communicator(0),
        **options,
    )
    rank1 = prepare_distributed_non_bo_diatomic_dynamics_initial_state(
        layout,
        group.communicator(1),
        **options,
    )

    assert rank0.source == rank1.source == "archive"
    assert group.barriers == [1, 1]
    np.testing.assert_array_equal(
        layout.assemble((rank0.state, rank1.state)),
        calculation.initial_state,
    )
