from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialAbsorbingBoundaries,
    TwoRadialDistributed2DHamiltonian,
    TwoRadialDistributed2DLayout,
    TwoRadialSystemSpec,
    radial_potential,
)


def _system(*, M_tot=0):
    grid1 = FEMDVRGridSpec([0.0, 0.8, 2.0, 5.0], [4, 5, 6])
    grid2 = FEMDVRGridSpec([0.0, 0.5, 1.5, 4.0], [5, 4, 6])
    basis = AngularBasisSpec(l_max=2, m_max=2)
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=grid1,
            basis=basis,
            kinetic_prefactor=-0.7,
            one_body_terms=(radial_potential(lambda r: -2.0 / r),),
        ),
        coordinate2=CoordinateSpec(
            grid=grid2,
            basis=basis,
            kinetic_prefactor=-0.003,
            one_body_terms=(radial_potential(lambda r: 1.0 / r),),
        ),
        multipoles=CoulombMultipoleSpec(4, 4),
        coulomb_terms=(CoulombTerm(1.0, -1.0),),
        M_tot=M_tot,
    ).build()


def _complex_state(system, seed=947):
    rng = np.random.default_rng(seed)
    return rng.normal(size=system.psi_shape) + 1j * rng.normal(
        size=system.psi_shape
    )


class _CompletedRequest:
    def Wait(self):
        return None


class _ReceiveRequest:
    def __init__(self, group, key, buffer):
        self.group = group
        self.key = key
        self.buffer = buffer

    def Wait(self):
        with self.group.condition:
            self.group.condition.wait_for(lambda: self.key in self.group.messages)
            values = self.group.messages.pop(self.key)
        np.copyto(self.buffer, values)


class _ThreadCommunicatorGroup:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.messages = {}

    def communicator(self, rank):
        return _ThreadCommunicator(self, rank)


class _ThreadCommunicator:
    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.size = group.size

    def Isend(self, buffer, dest, tag):
        key = (self.rank, dest, tag)
        with self.group.condition:
            self.group.messages[key] = np.asarray(buffer).copy()
            self.group.condition.notify_all()
        return _CompletedRequest()

    def Irecv(self, buffer, source, tag):
        return _ReceiveRequest(
            self.group,
            (source, self.rank, tag),
            buffer,
        )


@pytest.mark.parametrize("M_tot", [None, 0])
@pytest.mark.parametrize(
    "process_grid",
    [(1, 1), (1, 2), (2, 1), (2, 2)],
)
def test_emulated_cartesian_components_match_serial(M_tot, process_grid):
    system = _system(M_tot=M_tot)
    n_ranks = process_grid[0] * process_grid[1]
    layout = TwoRadialDistributed2DLayout(
        system,
        n_ranks,
        process_grid=process_grid,
    )
    distributed = TwoRadialDistributed2DHamiltonian(layout)
    serial = system.hamiltonian(radial_operator_mode="element-local")
    state = _complex_state(system)
    local = layout.distribute(state)

    coordinate1, coordinate2, coulomb = distributed.apply_components_all(local)

    np.testing.assert_allclose(
        layout.assemble(coordinate1),
        serial.apply_one_body_coordinate1(state),
        atol=1.0e-11,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        layout.assemble(coordinate2),
        serial.apply_one_body_coordinate2(state),
        atol=1.0e-11,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        layout.assemble(coulomb),
        serial.apply_coulomb(state),
        atol=1.0e-11,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        layout.assemble(distributed.apply_all(local)),
        serial(state),
        atol=2.0e-11,
        rtol=2.0e-12,
    )


def test_cartesian_mpi_edge_exchanges_match_serial_components():
    system = _system()
    layout = TwoRadialDistributed2DLayout(system, 4, process_grid=(2, 2))
    state = _complex_state(system)
    local = layout.distribute(state)
    group = _ThreadCommunicatorGroup(4)
    hamiltonians = [
        TwoRadialDistributed2DHamiltonian(
            layout,
            group.communicator(rank),
            profile=True,
        )
        for rank in range(4)
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        actions = tuple(
            executor.map(
                lambda pair: pair[0](pair[1]),
                zip(hamiltonians, local, strict=True),
            )
        )

    expected = system.hamiltonian(radial_operator_mode="element-local")(state)
    np.testing.assert_allclose(
        layout.assemble(actions),
        expected,
        atol=2.0e-11,
        rtol=2.0e-12,
    )
    for hamiltonian in hamiltonians:
        profile = hamiltonian.profile
        assert profile.hamiltonian_calls == 1
        assert profile.coordinate1_calls == 1
        assert profile.coordinate2_calls == 1
        assert profile.coulomb_calls == 1
        assert profile.coordinate1_halo_calls == 1
        assert profile.coordinate2_halo_calls == 1
        assert profile.coordinate1_weak_calls == 1
        assert profile.coordinate2_weak_calls == 1


@pytest.mark.parametrize("name", ["H2+", "HD+", "D2+"])
@pytest.mark.parametrize("process_grid", [(1, 2), (2, 1), (2, 2)])
def test_non_bo_diatomic_ions_match_serial(name, process_grid):
    electronic = FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4])
    internuclear = FEMDVRGridSpec([0.0, 1.5, 4.0], [4, 4])
    identical = name != "HD+"
    calculation = NonBornOppenheimerDiatomicIonGroundState(
        name,
        electronic_grid=electronic,
        internuclear_grid=internuclear,
        inversion_parity=1,
        pair_exchange_parity=1 if identical else None,
        symmetry_adapted_basis=identical,
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        radial_coulomb_method="analytic",
        codata_set="2018",
        max_iterations=1,
        use_preconditioner=False,
    )
    system = calculation.representation
    n_ranks = process_grid[0] * process_grid[1]
    layout = TwoRadialDistributed2DLayout(
        system,
        n_ranks,
        process_grid=process_grid,
    )
    distributed = TwoRadialDistributed2DHamiltonian(layout)
    state = _complex_state(system, seed=951)

    local = layout.distribute(state)
    coordinate1, coordinate2, coulomb = distributed.apply_components_all(local)
    serial = system.hamiltonian(radial_operator_mode="element-local")

    np.testing.assert_allclose(
        layout.assemble(coordinate1),
        serial.apply_one_body_coordinate1(state),
        atol=2.0e-11,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        layout.assemble(coordinate2),
        serial.apply_one_body_coordinate2(state),
        atol=2.0e-11,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        layout.assemble(coulomb),
        serial.apply_coulomb(state),
        atol=2.0e-11,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        layout.assemble(distributed.apply_all(local)),
        serial(state),
        atol=2.0e-11,
        rtol=2.0e-12,
    )


def test_masks_and_caps_are_applied_on_owned_tiles():
    system = _system()
    mask1 = np.linspace(1.0, 0.4, system.grid1.n_points)
    cap2 = -0.05j * np.linspace(0.0, 1.0, system.grid2.n_points) ** 2
    boundaries = TwoRadialAbsorbingBoundaries(
        system,
        mask1=mask1,
        cap2=cap2,
    )
    layout = TwoRadialDistributed2DLayout(system, 4, process_grid=(2, 2))
    distributed = TwoRadialDistributed2DHamiltonian(
        layout,
        absorbing_boundaries=boundaries,
    )
    state = _complex_state(system)
    local = layout.distribute(state)

    masked = tuple(distributed.apply_mask(part) for part in local)
    cap = tuple(distributed.cap_action(part) for part in local)
    action = layout.assemble(distributed.apply_all(local))

    np.testing.assert_allclose(
        layout.assemble(masked),
        boundaries.apply_mask(state, 0.0),
    )
    np.testing.assert_allclose(
        layout.assemble(cap),
        boundaries.cap_action(state),
    )
    np.testing.assert_allclose(
        action,
        system.hamiltonian(radial_operator_mode="element-local")(state)
        + boundaries.cap_action(state),
        atol=2.0e-11,
        rtol=2.0e-12,
    )


def test_multi_rank_direct_action_requires_communicator():
    system = _system()
    layout = TwoRadialDistributed2DLayout(system, 4, process_grid=(2, 2))
    distributed = TwoRadialDistributed2DHamiltonian(layout)
    state = layout.distribute(_complex_state(system))[0]

    with pytest.raises(RuntimeError, match="communicator is required"):
        distributed.apply_one_body_coordinate1(state)
