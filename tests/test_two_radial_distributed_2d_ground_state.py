from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from grid_tdse.ground_state import ImaginaryTimeGroundStateSolver
from grid_tdse.spherical.two_radial import (
    DiatomicIonDynamicsObservableEvaluator,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributed2DGroundStateSolver,
    TwoRadialDistributed2DLayout,
    TwoRadialDistributed2DReductions,
)


def _calculation(name="H2+"):
    identical = name != "HD+"
    return NonBornOppenheimerDiatomicIonGroundState(
        name,
        electronic_grid=FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4]),
        internuclear_grid=FEMDVRGridSpec([0.0, 1.5, 5.0], [4, 4]),
        inversion_parity=1,
        pair_exchange_parity=1 if identical else None,
        symmetry_adapted_basis=identical,
        l_max=0,
        m_max=0,
        lambda_max=0,
        mu_max=0,
        M_tot=0,
        radial_coulomb_method="poisson",
        radial_operator_mode="element-local",
        codata_set="2022",
        initial_state="product",
        integrator="sil",
        use_preconditioner=False,
        max_iterations=1,
        progress_interval=None,
    )


def _solver_options(integrator="sil", max_iterations=2):
    return {
        "time_step": 0.05,
        "max_iterations": max_iterations,
        "energy_tolerance": 1.0e-30,
        "state_tolerance": 1.0e-30,
        "residual_tolerance": 1.0e-30,
        "integrator": integrator,
        "max_krylov_dimension": 8,
        "krylov_rtol": 1.0e-9,
        "krylov_atol": 1.0e-12,
        "max_krylov_refinements": 4,
        "progress_interval": None,
    }


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
            self.group.condition.wait_for(lambda: bool(self.group.messages[self.key]))
            values = self.group.messages[self.key].popleft()
        np.copyto(self.buffer, values)


class _ThreadCommunicatorGroup:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.messages = defaultdict(deque)
        self.reduction_values = defaultdict(dict)
        self.reduction_results = {}
        self.gather_values = defaultdict(dict)
        self.gather_results = {}

    def communicator(self, rank):
        return _ThreadCommunicator(self, rank)


class _ThreadCommunicator:
    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.size = group.size
        self.reduction_round = 0
        self.gather_round = 0

    def Isend(self, buffer, dest, tag):
        key = (self.rank, dest, tag)
        with self.group.condition:
            self.group.messages[key].append(np.asarray(buffer).copy())
            self.group.condition.notify_all()
        return _CompletedRequest()

    def Irecv(self, buffer, source, tag):
        return _ReceiveRequest(
            self.group,
            (source, self.rank, tag),
            buffer,
        )

    def allreduce(self, value):
        reduction_round = self.reduction_round
        self.reduction_round += 1
        with self.group.condition:
            values = self.group.reduction_values[reduction_round]
            values[self.rank] = value
            if len(values) == self.size:
                self.group.reduction_results[reduction_round] = sum(values.values())
                self.group.condition.notify_all()
            else:
                self.group.condition.wait_for(
                    lambda: reduction_round in self.group.reduction_results
                )
            return self.group.reduction_results[reduction_round]

    def Allreduce(self, send_buffer, receive_buffer):
        np.copyto(receive_buffer, self.allreduce(np.asarray(send_buffer).copy()))

    def allgather(self, value):
        gather_round = self.gather_round
        self.gather_round += 1
        with self.group.condition:
            values = self.group.gather_values[gather_round]
            values[self.rank] = value
            if len(values) == self.size:
                self.group.gather_results[gather_round] = [
                    values[rank] for rank in range(self.size)
                ]
                self.group.condition.notify_all()
            else:
                self.group.condition.wait_for(
                    lambda: gather_round in self.group.gather_results
                )
            return self.group.gather_results[gather_round]


def _serial_result(calculation, options):
    solver = ImaginaryTimeGroundStateSolver(**options)
    return solver.solve(
        calculation.initial_state,
        calculation.hamiltonian,
        calculation.representation.inner_product,
    )


def _distributed_results(
    calculation,
    process_grid,
    options,
    *,
    observable_evaluator=None,
    sample_interval=10,
    log_path=None,
    profile=False,
):
    n_ranks = process_grid[0] * process_grid[1]
    layout = TwoRadialDistributed2DLayout(
        calculation.representation,
        n_ranks,
        process_grid=process_grid,
    )
    local_initial = layout.distribute(calculation.initial_state)
    group = _ThreadCommunicatorGroup(n_ranks)
    solvers = [
        TwoRadialDistributed2DGroundStateSolver(
            layout,
            group.communicator(rank),
            observable_evaluator=observable_evaluator,
            sample_interval=sample_interval,
            log_path=log_path,
            profile=profile,
            **options,
        )
        for rank in range(n_ranks)
    ]
    with ThreadPoolExecutor(max_workers=n_ranks) as executor:
        futures = [
            executor.submit(solver.solve, local)
            for solver, local in zip(solvers, local_initial, strict=True)
        ]
        results = tuple(future.result(timeout=60.0) for future in futures)
    return layout, results


@pytest.mark.parametrize(
    ("integrator", "process_grid"),
    [
        ("sil", (1, 1)),
        ("sil", (1, 2)),
        ("sil", (2, 1)),
        ("sil", (2, 2)),
        ("sia", (1, 1)),
        ("sia", (2, 2)),
        ("exponential-euler", (1, 1)),
        ("exponential-euler", (2, 2)),
    ],
)
def test_cartesian_krylov_itp_matches_serial_iterations(
    integrator,
    process_grid,
):
    calculation = _calculation()
    calculation.initial_state = (1.0 + 0.2j) * calculation.initial_state
    options = _solver_options(integrator)
    serial = _serial_result(calculation, options)
    layout, distributed = _distributed_results(
        calculation,
        process_grid,
        options,
    )
    state = layout.assemble([result.state for result in distributed])

    np.testing.assert_allclose(state, serial.state, atol=3.0e-11, rtol=3.0e-11)
    for result in distributed:
        assert result.integrator == integrator
        assert result.energy == pytest.approx(serial.energy, abs=3.0e-12)
        assert result.state_change == pytest.approx(
            serial.state_change,
            abs=3.0e-12,
        )
        assert result.residual_norm == pytest.approx(
            serial.residual_norm,
            abs=3.0e-12,
        )
        assert result.norm == pytest.approx(1.0, abs=3.0e-13)
        np.testing.assert_allclose(
            result.energy_history,
            serial.energy_history,
            atol=3.0e-12,
            rtol=3.0e-12,
        )
        np.testing.assert_array_equal(
            result.krylov_dimensions,
            serial.krylov_dimensions,
        )


@pytest.mark.parametrize("name", ["H2+", "HD+", "D2+"])
def test_cartesian_sil_iteration_matches_serial_for_diatomic_ions(name):
    calculation = _calculation(name)
    options = _solver_options(max_iterations=1)
    serial = _serial_result(calculation, options)
    layout, distributed = _distributed_results(
        calculation,
        (2, 2),
        options,
    )

    np.testing.assert_allclose(
        layout.assemble([result.state for result in distributed]),
        serial.state,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    assert distributed[0].energy == pytest.approx(serial.energy, abs=3.0e-12)
    assert distributed[0].residual_norm == pytest.approx(
        serial.residual_norm,
        abs=3.0e-12,
    )


@pytest.mark.parametrize("process_grid", [(1, 2), (2, 1), (2, 2)])
def test_cartesian_reductions_match_serial_inner_product(process_grid):
    calculation = _calculation()
    layout = TwoRadialDistributed2DLayout(
        calculation.representation,
        process_grid[0] * process_grid[1],
        process_grid=process_grid,
    )
    rng = np.random.default_rng(26117)
    left = rng.normal(size=layout.global_shape) + 1j * rng.normal(
        size=layout.global_shape
    )
    right = rng.normal(size=layout.global_shape) + 1j * rng.normal(
        size=layout.global_shape
    )
    reductions = TwoRadialDistributed2DReductions(layout)

    actual = reductions.inner_product_all(
        layout.distribute(left),
        layout.distribute(right),
    )
    expected = calculation.representation.inner_product(left, right)

    assert actual == pytest.approx(expected, abs=2.0e-12)


def test_cartesian_solver_profiles_and_batches_collectives():
    calculation = _calculation()
    _, results = _distributed_results(
        calculation,
        (2, 2),
        _solver_options(max_iterations=1),
        profile=True,
    )

    for result in results:
        assert result.hamiltonian_profile.hamiltonian_calls > 0
        assert result.hamiltonian_profile.coordinate1_calls > 0
        assert result.hamiltonian_profile.coordinate2_calls > 0
        assert result.hamiltonian_profile.coulomb_calls > 0
        reductions = result.reduction_profile
        assert reductions.collective_values == reductions.local_inner_product_calls
        assert reductions.collective_calls < reductions.collective_values


def test_cartesian_sampling_matches_serial_observables_and_logs_once(
    tmp_path,
    capsys,
):
    calculation = _calculation()
    options = _solver_options(max_iterations=2)
    serial = _serial_result(calculation, options)
    expected_initial = calculation.observable_evaluator(calculation.initial_state)
    expected_final = calculation.observable_evaluator(serial.state)
    log_path = tmp_path / "cartesian-ground-state.log"

    _, distributed = _distributed_results(
        calculation,
        (2, 2),
        options,
        observable_evaluator=calculation.observable_evaluator,
        sample_interval=1,
        log_path=log_path,
    )

    for result in distributed:
        assert [sample.iteration for sample in result.samples] == [0, 1, 2]
        initial = result.samples[0].observables
        final = result.samples[-1].observables
        assert initial.r == pytest.approx(expected_initial.r, abs=3.0e-13)
        assert initial.R == pytest.approx(expected_initial.R, abs=3.0e-13)
        assert final.total_l_squared == pytest.approx(
            expected_final.total_l_squared,
            abs=3.0e-13,
        )
        assert final.inversion == pytest.approx(expected_final.inversion)
        assert final.pair_exchange == pytest.approx(expected_final.pair_exchange)
        np.testing.assert_allclose(
            final.dipole_total,
            expected_final.dipole_total,
            atol=3.0e-13,
        )
    terminal = capsys.readouterr().out
    logfile = log_path.read_text(encoding="utf-8")
    for iteration in (0, 1, 2):
        marker = f"ITP sample {iteration} "
        assert terminal.count(marker) == 1
        assert logfile.count(marker) == 1


def test_cartesian_dynamics_snapshot_matches_serial_below_unit_norm():
    calculation = _calculation()
    state = 0.43 * calculation.initial_state
    layout = TwoRadialDistributed2DLayout(
        calculation.representation,
        4,
        process_grid=(2, 2),
    )
    local_states = layout.distribute(state)
    group = _ThreadCommunicatorGroup(4)
    reductions = [
        TwoRadialDistributed2DReductions(layout, group.communicator(rank))
        for rank in range(4)
    ]

    def scaled(values):
        if isinstance(values, np.ndarray):
            return 1.75 * values
        return type(values)(
            values.layout,
            values.rank,
            1.75 * values.coefficients,
        )

    evaluator = DiatomicIonDynamicsObservableEvaluator(
        calculation.representation,
        calculation.parameters,
        scaled,
        gauge="length",
    )
    expected = evaluator(state, 0.5)
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(
            executor.map(
                lambda pair: evaluator.distributed_2d(*pair, 0.5),
                zip(local_states, reductions, strict=True),
            )
        )

    for snapshot in actual:
        assert snapshot.norm == pytest.approx(expected.norm, abs=3.0e-13)
        assert snapshot.energy_numerator == pytest.approx(
            expected.energy_numerator, abs=3.0e-13
        )
        assert snapshot.numerators.R_squared == pytest.approx(
            expected.numerators.R_squared, abs=3.0e-13
        )
        assert snapshot.numerators.r_dot_R == pytest.approx(
            expected.numerators.r_dot_R, abs=3.0e-13
        )
        np.testing.assert_allclose(
            snapshot.conditional.dipole_total,
            expected.conditional.dipole_total,
            atol=3.0e-13,
        )
        assert snapshot.electron_center_distances.partner_rms == pytest.approx(
            expected.electron_center_distances.partner_rms,
            abs=3.0e-13,
        )


def test_cartesian_solver_requires_a_communicator_for_multiple_ranks():
    calculation = _calculation()
    layout = TwoRadialDistributed2DLayout(
        calculation.representation,
        4,
        process_grid=(2, 2),
    )

    with pytest.raises(ValueError, match="communicator is required"):
        TwoRadialDistributed2DGroundStateSolver(layout)


def test_cartesian_solver_rejects_cn_and_wrong_local_rank():
    calculation = _calculation()
    layout = TwoRadialDistributed2DLayout(
        calculation.representation,
        2,
        process_grid=(1, 2),
    )
    group = _ThreadCommunicatorGroup(2)

    with pytest.raises(ValueError, match="sil.*sia.*exponential-euler"):
        TwoRadialDistributed2DGroundStateSolver(
            layout,
            group.communicator(0),
            integrator="crank-nicolson",
        )

    solver = TwoRadialDistributed2DGroundStateSolver(
        layout,
        group.communicator(0),
        **_solver_options(max_iterations=1),
    )
    with pytest.raises(ValueError, match="belongs to rank 1"):
        solver.solve(layout.distribute(calculation.initial_state)[1])
