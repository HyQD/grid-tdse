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
    TwoRadialDistributedHamiltonian,
    TwoRadialDistributedLayout,
    TwoRadialDistributedReductions,
    TwoRadialSystemSpec,
    load_two_radial_state,
    radial_potential,
    save_two_radial_state,
)


def _system(*, M_tot=0):
    grid1 = FEMDVRGridSpec([0.0, 0.8, 2.0, 5.0], [4, 5, 6])
    grid2 = FEMDVRGridSpec([0.0, 0.5, 1.5, 4.0, 7.0], [5, 4, 6, 4])
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


def _complex_state(system, seed=731):
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
        self.reduction_values = {}
        self.reduction_result = None

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

    def allreduce(self, value):
        with self.group.condition:
            self.group.reduction_values[self.rank] = value
            if len(self.group.reduction_values) == self.size:
                self.group.reduction_result = sum(
                    self.group.reduction_values.values()
                )
                self.group.condition.notify_all()
            else:
                self.group.condition.wait_for(
                    lambda: self.group.reduction_result is not None
                )
            return self.group.reduction_result

    def Allreduce(self, send_buffer, receive_buffer):
        np.copyto(receive_buffer, self.allreduce(np.asarray(send_buffer).copy()))


@pytest.mark.parametrize("M_tot", [None, 0])
@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
def test_emulated_distributed_hamiltonian_matches_serial(M_tot, n_ranks):
    system = _system(M_tot=M_tot)
    layout = TwoRadialDistributedLayout(system, n_ranks)
    distributed = TwoRadialDistributedHamiltonian(layout)
    state = _complex_state(system)
    local_states = layout.distribute(state)

    local_action = distributed.apply_all(local_states)
    actual = layout.assemble(local_action)
    expected = system.hamiltonian(radial_operator_mode="element-local")(state)

    np.testing.assert_allclose(actual, expected, atol=1.0e-11, rtol=1.0e-12)


@pytest.mark.parametrize("M_tot", [None, 0])
@pytest.mark.parametrize("n_ranks", [1, 2, 3])
def test_coordinate1_distributed_hamiltonian_matches_serial(M_tot, n_ranks):
    system = _system(M_tot=M_tot)
    layout = TwoRadialDistributedLayout(
        system,
        n_ranks,
        partitioned_coordinate=1,
    )
    distributed = TwoRadialDistributedHamiltonian(layout)
    state = _complex_state(system)

    actual = layout.assemble(distributed.apply_all(layout.distribute(state)))
    expected = system.hamiltonian(radial_operator_mode="element-local")(state)

    np.testing.assert_allclose(actual, expected, atol=1.0e-11, rtol=1.0e-12)


@pytest.mark.parametrize("M_tot", [None, 0])
def test_single_rank_components_match_serial_components(M_tot):
    system = _system(M_tot=M_tot)
    layout = TwoRadialDistributedLayout(system, 1)
    distributed = TwoRadialDistributedHamiltonian(layout)
    serial = system.hamiltonian(radial_operator_mode="element-local")
    state = _complex_state(system)
    local = layout.distribute(state)[0]

    components = distributed.components(local)

    np.testing.assert_allclose(
        components.coordinate1.coefficients,
        serial.apply_one_body_coordinate1(state),
        atol=5.0e-12,
        rtol=5.0e-13,
    )
    np.testing.assert_allclose(
        components.coordinate2.coefficients,
        serial.apply_one_body_coordinate2(state),
        atol=5.0e-12,
        rtol=5.0e-13,
    )
    np.testing.assert_allclose(
        components.coulomb.coefficients,
        serial.apply_coulomb(state),
        atol=5.0e-12,
        rtol=5.0e-13,
    )
    np.testing.assert_allclose(
        components.total.coefficients,
        serial(state),
        atol=1.0e-11,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize("M_tot", [None, 0])
def test_distributed_physical_reductions_match_serial(M_tot):
    system = _system(M_tot=M_tot)
    layout = TwoRadialDistributedLayout(system, 3)
    distributed = TwoRadialDistributedHamiltonian(layout)
    reductions = TwoRadialDistributedReductions(layout)
    state = _complex_state(system)
    other = _complex_state(system, seed=811)
    local_state = layout.distribute(state)
    local_other = layout.distribute(other)
    local_action = distributed.apply_all(local_state)

    np.testing.assert_allclose(
        reductions.inner_product_all(local_state, local_other),
        system.inner_product(state, other),
        atol=2.0e-12,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        reductions.inner_product_all(local_state, local_action),
        system.inner_product(
            state,
            system.hamiltonian(radial_operator_mode="element-local")(state),
        ),
        atol=2.0e-11,
        rtol=2.0e-12,
    )
    local_norm_squared = sum(
        reductions.local_inner_product(part, part).real
        for part in local_state
    )
    assert local_norm_squared == pytest.approx(
        system.inner_product(state, state).real,
        rel=2.0e-13,
        abs=2.0e-12,
    )


def test_nonblocking_bridge_exchange_matches_serial_coordinate2_action():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 3)
    state = _complex_state(system)
    local_states = layout.distribute(state)
    group = _ThreadCommunicatorGroup(3)
    hamiltonians = [
        TwoRadialDistributedHamiltonian(layout, group.communicator(rank))
        for rank in range(3)
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        local_actions = tuple(
            executor.map(
                lambda pair: pair[0].apply_one_body_coordinate2(pair[1]),
                zip(hamiltonians, local_states),
            )
        )

    actual = layout.assemble(local_actions)
    expected = system.hamiltonian(
        radial_operator_mode="element-local"
    ).apply_one_body_coordinate2(state)
    np.testing.assert_allclose(actual, expected, atol=5.0e-12, rtol=5.0e-13)


def test_nonblocking_bridge_exchange_matches_serial_coordinate1_action():
    system = _system()
    layout = TwoRadialDistributedLayout(
        system,
        3,
        partitioned_coordinate=1,
    )
    state = _complex_state(system)
    local_states = layout.distribute(state)
    group = _ThreadCommunicatorGroup(3)
    hamiltonians = [
        TwoRadialDistributedHamiltonian(layout, group.communicator(rank))
        for rank in range(3)
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        local_actions = tuple(
            executor.map(
                lambda pair: pair[0].apply_one_body_coordinate1(pair[1]),
                zip(hamiltonians, local_states),
            )
        )

    actual = layout.assemble(local_actions)
    expected = system.hamiltonian(
        radial_operator_mode="element-local"
    ).apply_one_body_coordinate1(state)
    np.testing.assert_allclose(actual, expected, atol=5.0e-12, rtol=5.0e-13)


def test_coordinate1_distributed_reductions_match_serial():
    system = _system()
    layout = TwoRadialDistributedLayout(
        system,
        3,
        partitioned_coordinate=1,
    )
    reductions = TwoRadialDistributedReductions(layout)
    state = _complex_state(system)
    other = _complex_state(system, seed=812)

    actual = reductions.inner_product_all(
        layout.distribute(state),
        layout.distribute(other),
    )

    assert actual == pytest.approx(
        system.inner_product(state, other),
        rel=2.0e-13,
        abs=2.0e-12,
    )


def test_collective_inner_product_matches_serial_result():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 3)
    state = _complex_state(system)
    other = _complex_state(system, seed=812)
    local_state = layout.distribute(state)
    local_other = layout.distribute(other)
    group = _ThreadCommunicatorGroup(3)
    reductions = [
        TwoRadialDistributedReductions(layout, group.communicator(rank))
        for rank in range(3)
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = tuple(
            executor.map(
                lambda values: values[0].inner_product(values[1], values[2]),
                zip(reductions, local_state, local_other),
            )
        )

    expected = system.inner_product(state, other)
    for result in results:
        assert result == pytest.approx(expected, rel=2.0e-13, abs=2.0e-12)


def test_collective_inner_products_batch_one_reduction():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 3)
    states = [_complex_state(system, seed=seed) for seed in (811, 812, 813)]
    right = _complex_state(system, seed=814)
    local_left = [layout.distribute(state) for state in states]
    local_right = layout.distribute(right)
    group = _ThreadCommunicatorGroup(3)
    reductions = [
        TwoRadialDistributedReductions(
            layout,
            group.communicator(rank),
            profile=True,
        )
        for rank in range(3)
    ]

    def apply(rank):
        return reductions[rank].inner_products(
            [parts[rank] for parts in local_left],
            local_right[rank],
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = tuple(executor.map(apply, range(3)))

    expected = np.asarray(
        [system.inner_product(state, right) for state in states]
    )
    for result, reduction in zip(results, reductions, strict=True):
        np.testing.assert_allclose(result, expected, atol=2.0e-12, rtol=2.0e-13)
        assert reduction.profile.local_inner_product_calls == 3
        assert reduction.profile.collective_calls == 1
        assert reduction.profile.collective_values == 3


def test_one_rank_communicator_bypasses_collective():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 1)
    state = layout.distribute(_complex_state(system))[0]
    group = _ThreadCommunicatorGroup(1)
    communicator = group.communicator(0)
    reductions = TwoRadialDistributedReductions(
        layout,
        communicator,
        profile=True,
    )

    result = reductions.inner_product(state, state)

    assert result == pytest.approx(system.inner_product(
        state.coefficients,
        state.coefficients,
    ))
    assert group.reduction_values == {}
    assert reductions.profile.collective_calls == 0


def test_single_rank_energy_and_residual_reductions():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 1)
    distributed = TwoRadialDistributedHamiltonian(layout)
    reductions = TwoRadialDistributedReductions(layout)
    state = layout.distribute(_complex_state(system))[0]
    action = distributed(state)

    expectation = reductions.expectation(state, action)
    quotient = reductions.rayleigh_quotient(state, action)
    residual = reductions.residual_norm(state, action, quotient)
    serial_state = state.coefficients
    serial_action = action.coefficients
    serial_norm_squared = system.inner_product(serial_state, serial_state).real

    assert reductions.norm_squared(state) == pytest.approx(serial_norm_squared)
    assert reductions.norm(state) == pytest.approx(np.sqrt(serial_norm_squared))
    assert expectation == pytest.approx(
        system.inner_product(serial_state, serial_action)
    )
    assert quotient == pytest.approx(expectation / serial_norm_squared)
    expected_residual = serial_action - quotient * serial_state
    assert residual == pytest.approx(
        np.sqrt(system.inner_product(expected_residual, expected_residual).real)
    )


def test_single_rank_profiling_records_local_component_work():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 1)
    hamiltonian = TwoRadialDistributedHamiltonian(layout, profile=True)
    reductions = TwoRadialDistributedReductions(layout, profile=True)
    state = layout.distribute(_complex_state(system))[0]

    action = hamiltonian(state)
    reductions.inner_product(state, action)

    hamiltonian_profile = hamiltonian.profile
    assert hamiltonian_profile.hamiltonian_calls == 1
    assert hamiltonian_profile.coordinate1_calls == 1
    assert hamiltonian_profile.coordinate2_calls == 1
    assert hamiltonian_profile.coulomb_calls == 1
    assert hamiltonian_profile.halo_exchange_calls == 0
    assert hamiltonian_profile.weak_exchange_calls == 0
    assert hamiltonian_profile.coordinate1_seconds >= 0.0
    assert hamiltonian_profile.coordinate2_local_seconds >= 0.0
    assert hamiltonian_profile.coulomb_seconds >= 0.0

    reduction_profile = reductions.profile
    assert reduction_profile.local_inner_product_calls == 1
    assert reduction_profile.collective_calls == 0
    assert reduction_profile.local_inner_product_seconds >= 0.0

    hamiltonian.reset_profile()
    reductions.reset_profile()
    assert hamiltonian.profile.hamiltonian_calls == 0
    assert reductions.profile.local_inner_product_calls == 0


def test_multi_rank_profiling_records_bridge_exchanges_and_collectives():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 2)
    local_states = layout.distribute(_complex_state(system))
    group = _ThreadCommunicatorGroup(2)
    hamiltonians = [
        TwoRadialDistributedHamiltonian(
            layout,
            group.communicator(rank),
            profile=True,
        )
        for rank in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(
            executor.map(
                lambda pair: pair[0](pair[1]),
                zip(hamiltonians, local_states),
            )
        )

    for hamiltonian in hamiltonians:
        profile = hamiltonian.profile
        assert profile.hamiltonian_calls == 1
        assert profile.halo_exchange_calls == 1
        assert profile.weak_exchange_calls == 1
        assert profile.halo_exchange_seconds >= 0.0
        assert profile.weak_exchange_seconds >= 0.0

    reduction_group = _ThreadCommunicatorGroup(2)
    reductions = [
        TwoRadialDistributedReductions(
            layout,
            reduction_group.communicator(rank),
            profile=True,
        )
        for rank in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(
            executor.map(
                lambda values: values[0].inner_product(values[1], values[1]),
                zip(reductions, local_states),
            )
        )

    for reduction in reductions:
        profile = reduction.profile
        assert profile.local_inner_product_calls == 1
        assert profile.collective_calls == 1
        assert profile.collective_values == 1
        assert profile.collective_seconds >= 0.0


def test_archived_state_uses_same_distributed_hamiltonian_path(tmp_path):
    system = _system()
    state = _complex_state(system)
    norm = np.sqrt(system.inner_product(state, state).real)
    state /= norm
    path = tmp_path / "archived-ground-state.npz"
    save_two_radial_state(
        path,
        state,
        system,
        physical_state={
            "system_family": "non_bo_diatomic_ion",
            "system": "H2+",
            "symmetry": {"M_tot": 0},
        },
    )
    layout = TwoRadialDistributedLayout(system, 3)
    distributed = TwoRadialDistributedHamiltonian(layout)

    local_state = layout.distribute(load_two_radial_state(path))
    actual = layout.assemble(distributed.apply_all(local_state))
    expected = system.hamiltonian(radial_operator_mode="element-local")(state)

    np.testing.assert_allclose(actual, expected, atol=1.0e-11, rtol=1.0e-12)


def test_multi_rank_direct_action_requires_communicator():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 2)
    distributed = TwoRadialDistributedHamiltonian(layout)
    state = layout.distribute(_complex_state(system))[0]

    with pytest.raises(RuntimeError, match="communicator is required"):
        distributed.apply_one_body_coordinate2(state)


@pytest.mark.parametrize(
    "constructor",
    [TwoRadialDistributedHamiltonian, TwoRadialDistributedReductions],
)
def test_distributed_profiling_flag_must_be_boolean(constructor):
    system = _system()
    layout = TwoRadialDistributedLayout(system, 1)

    with pytest.raises(TypeError, match="profile must be boolean"):
        constructor(layout, profile="yes")


def test_reductions_do_not_normalize_expectations_implicitly():
    system = _system()
    layout = TwoRadialDistributedLayout(system, 1)
    reductions = TwoRadialDistributedReductions(layout)
    state = layout.distribute(2.0 * _complex_state(system))[0]

    expectation = reductions.expectation(state, state)

    assert expectation.real == pytest.approx(reductions.norm_squared(state))
    assert reductions.rayleigh_quotient(state, state) == pytest.approx(1.0)
