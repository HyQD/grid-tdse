from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from grid_tdse.ground_state import ImaginaryTimeGroundStateSolver
from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributedGroundStateSolver,
    TwoRadialDistributedLayout,
)


def _calculation():
    return NonBornOppenheimerDiatomicIonGroundState(
        "H2+",
        electronic_grid=FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4]),
        internuclear_grid=FEMDVRGridSpec(
            [0.0, 1.0, 2.5, 5.0],
            [4, 4, 4],
        ),
        inversion_parity=1,
        pair_exchange_parity=1,
        symmetry_adapted_basis=True,
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


def _solver_options(max_iterations=3):
    return {
        "time_step": 0.05,
        "max_iterations": max_iterations,
        "energy_tolerance": 1.0e-30,
        "state_tolerance": 1.0e-30,
        "residual_tolerance": 1.0e-30,
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
            self.group.condition.wait_for(
                lambda: bool(self.group.messages[self.key])
            )
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
                self.group.reduction_results[reduction_round] = sum(
                    values.values()
                )
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
    solver = ImaginaryTimeGroundStateSolver(
        integrator="sil",
        **options,
    )
    return solver.solve(
        calculation.initial_state,
        calculation.hamiltonian,
        calculation.representation.inner_product,
    )


def _distributed_results(
    calculation,
    n_ranks,
    options,
    *,
    partitioned_coordinate=2,
    observable_evaluator=None,
    sample_interval=10,
    log_path=None,
):
    representation = calculation.representation
    layout = TwoRadialDistributedLayout(
        representation,
        n_ranks,
        partitioned_coordinate=partitioned_coordinate,
    )
    local_initial = layout.distribute(calculation.initial_state)
    group = _ThreadCommunicatorGroup(n_ranks)
    solvers = [
        TwoRadialDistributedGroundStateSolver(
            layout,
            group.communicator(rank),
            observable_evaluator=observable_evaluator,
            sample_interval=sample_interval,
            log_path=log_path,
            **options,
        )
        for rank in range(n_ranks)
    ]
    with ThreadPoolExecutor(max_workers=n_ranks) as executor:
        futures = [
            executor.submit(solver.solve, local)
            for solver, local in zip(solvers, local_initial, strict=True)
        ]
        results = tuple(future.result(timeout=30.0) for future in futures)
    return layout, results


@pytest.mark.parametrize("n_ranks", [1, 2, 3])
def test_distributed_lanczos_itp_matches_serial_iterations(n_ranks):
    calculation = _calculation()
    calculation.initial_state = (1.0 + 0.2j) * calculation.initial_state
    options = _solver_options()
    serial = _serial_result(calculation, options)
    layout, distributed = _distributed_results(
        calculation,
        n_ranks,
        options,
    )
    state = layout.assemble([result.state for result in distributed])

    np.testing.assert_allclose(state, serial.state, atol=2.0e-11, rtol=2.0e-11)
    for result in distributed:
        assert result.energy == pytest.approx(serial.energy, abs=2.0e-12)
        assert result.state_change == pytest.approx(
            serial.state_change,
            abs=2.0e-12,
        )
        assert result.residual_norm == pytest.approx(
            serial.residual_norm,
            abs=2.0e-12,
        )
        assert result.norm == pytest.approx(1.0, abs=2.0e-13)
        np.testing.assert_allclose(
            result.energy_history,
            serial.energy_history,
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        np.testing.assert_array_equal(
            result.krylov_dimensions,
            serial.krylov_dimensions,
        )
        np.testing.assert_array_equal(
            result.krylov_accepted_substeps,
            serial.krylov_accepted_substeps,
        )
        np.testing.assert_array_equal(
            result.krylov_rejected_substeps,
            serial.krylov_rejected_substeps,
        )
        np.testing.assert_array_equal(
            result.krylov_refinement_depths,
            serial.krylov_refinement_depths,
        )
        np.testing.assert_allclose(
            result.krylov_error_estimates,
            serial.krylov_error_estimates,
            atol=1.0e-24,
            rtol=2.0e-14,
        )


@pytest.mark.parametrize("partitioned_coordinate", [1, 2])
def test_both_radial_layouts_match_the_same_serial_ground_state(
    partitioned_coordinate,
):
    calculation = _calculation()
    calculation.initial_state = (1.0 + 0.2j) * calculation.initial_state
    options = _solver_options(max_iterations=2)
    serial = _serial_result(calculation, options)

    layout, distributed = _distributed_results(
        calculation,
        2,
        options,
        partitioned_coordinate=partitioned_coordinate,
    )
    state = layout.assemble([result.state for result in distributed])

    np.testing.assert_allclose(state, serial.state, atol=2.0e-11, rtol=2.0e-11)
    assert distributed[0].energy == pytest.approx(serial.energy, abs=2.0e-12)
    assert distributed[0].residual_norm == pytest.approx(
        serial.residual_norm,
        abs=2.0e-12,
    )


def test_one_rank_distributed_solver_converges_from_an_eigenstate():
    calculation = _calculation()
    representation = calculation.representation
    hamiltonian = calculation.hamiltonian
    shape = representation.psi_shape
    dimension = int(np.prod(shape))
    matrix = np.column_stack(
        [
            hamiltonian(np.eye(dimension)[:, column].reshape(shape)).ravel()
            for column in range(dimension)
        ]
    )
    weights = np.multiply.outer(
        representation.grid1.weights,
        representation.grid2.weights,
    ).ravel()
    square_root = np.sqrt(weights)
    transformed = square_root[:, None] * matrix / square_root[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    initial = (eigenvectors[:, 0] / square_root).reshape(shape)
    layout = TwoRadialDistributedLayout(representation, 1)
    solver = TwoRadialDistributedGroundStateSolver(
        layout,
        **(_solver_options(max_iterations=2) | {
            "energy_tolerance": 1.0e-11,
            "state_tolerance": 1.0e-10,
            "residual_tolerance": 1.0e-9,
        }),
    )

    result = solver.solve(layout.distribute(initial)[0])

    assert result.converged
    assert result.energy == pytest.approx(eigenvalues[0], abs=2.0e-10)
    assert result.residual_norm < 1.0e-9


def test_distributed_solver_requires_a_communicator_for_multiple_ranks():
    calculation = _calculation()
    layout = TwoRadialDistributedLayout(calculation.representation, 2)

    with pytest.raises(ValueError, match="communicator is required"):
        TwoRadialDistributedGroundStateSolver(layout)


def test_distributed_solver_validates_local_state_rank():
    calculation = _calculation()
    layout = TwoRadialDistributedLayout(calculation.representation, 2)
    group = _ThreadCommunicatorGroup(2)
    solver = TwoRadialDistributedGroundStateSolver(
        layout,
        group.communicator(0),
        **_solver_options(max_iterations=1),
    )
    rank_one_state = layout.distribute(calculation.initial_state)[1]

    with pytest.raises(ValueError, match="belongs to rank 1"):
        solver.solve(rank_one_state)


@pytest.mark.parametrize("partitioned_coordinate", [1, 2])
def test_distributed_solver_returns_rank_local_profiles(partitioned_coordinate):
    calculation = _calculation()
    layout = TwoRadialDistributedLayout(
        calculation.representation,
        1,
        partitioned_coordinate=partitioned_coordinate,
    )
    solver = TwoRadialDistributedGroundStateSolver(
        layout,
        profile=True,
        **_solver_options(max_iterations=1),
    )

    result = solver.solve(layout.distribute(calculation.initial_state)[0])

    assert result.hamiltonian_profile.hamiltonian_calls > 0
    assert result.hamiltonian_profile.coordinate1_calls > 0
    assert result.hamiltonian_profile.coordinate2_calls > 0
    assert result.hamiltonian_profile.coulomb_calls > 0
    assert result.reduction_profile.local_inner_product_calls > 0
    assert result.reduction_profile.collective_calls == 0


def test_distributed_solver_batches_krylov_collectives():
    calculation = _calculation()
    layout = TwoRadialDistributedLayout(calculation.representation, 2)
    local_initial = layout.distribute(calculation.initial_state)
    group = _ThreadCommunicatorGroup(2)
    solvers = [
        TwoRadialDistributedGroundStateSolver(
            layout,
            group.communicator(rank),
            profile=True,
            **_solver_options(max_iterations=1),
        )
        for rank in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda pair: pair[0].solve(pair[1]),
                zip(solvers, local_initial, strict=True),
            )
        )

    for result in results:
        profile = result.reduction_profile
        assert profile.collective_values == profile.local_inner_product_calls
        assert profile.collective_calls < profile.collective_values


def test_only_root_enables_progress_output():
    calculation = _calculation()
    layout = TwoRadialDistributedLayout(calculation.representation, 2)
    group = _ThreadCommunicatorGroup(2)

    root_solver = TwoRadialDistributedGroundStateSolver(
        layout,
        group.communicator(0),
        progress_interval=7,
    )
    other_solver = TwoRadialDistributedGroundStateSolver(
        layout,
        group.communicator(1),
        progress_interval=7,
    )

    assert root_solver.solver.progress_interval == 7
    assert other_solver.solver.progress_interval is None


@pytest.mark.parametrize("partitioned_coordinate", [1, 2])
def test_distributed_sampling_matches_serial_observables_and_logs_once(
    partitioned_coordinate,
    tmp_path,
    capsys,
):
    calculation = _calculation()
    options = _solver_options(max_iterations=3)
    serial = _serial_result(calculation, options)
    expected_initial = calculation.observable_evaluator(
        calculation.initial_state
    )
    expected_final = calculation.observable_evaluator(serial.state)
    log_path = tmp_path / f"coordinate-{partitioned_coordinate}.log"

    _, distributed = _distributed_results(
        calculation,
        2,
        options,
        partitioned_coordinate=partitioned_coordinate,
        observable_evaluator=calculation.observable_evaluator,
        sample_interval=2,
        log_path=log_path,
    )

    for result in distributed:
        assert [sample.iteration for sample in result.samples] == [0, 2, 3]
        initial = result.samples[0].observables
        assert initial.r == pytest.approx(expected_initial.r, abs=2.0e-13)
        assert initial.R == pytest.approx(expected_initial.R, abs=2.0e-13)
        assert initial.cos2_theta_R == pytest.approx(
            expected_initial.cos2_theta_R,
            abs=2.0e-13,
        )
        assert initial.inversion == pytest.approx(expected_initial.inversion)
        assert initial.pair_exchange == pytest.approx(
            expected_initial.pair_exchange
        )
        actual = result.samples[-1].observables
        assert actual.r == pytest.approx(expected_final.r, abs=2.0e-13)
        assert actual.R == pytest.approx(expected_final.R, abs=2.0e-13)
        assert actual.total_l_squared == pytest.approx(
            expected_final.total_l_squared,
            abs=2.0e-13,
        )
        assert actual.cos2_theta_R == pytest.approx(
            expected_final.cos2_theta_R,
            abs=2.0e-13,
        )
        np.testing.assert_allclose(
            actual.dipole_total,
            expected_final.dipole_total,
            atol=2.0e-13,
        )
        assert all(
            sample.observable_wall_time >= 0.0 for sample in result.samples
        )
    terminal = capsys.readouterr().out
    logfile = log_path.read_text(encoding="utf-8")
    for iteration in (0, 2, 3):
        marker = f"ITP sample {iteration} "
        assert terminal.count(marker) == 1
        assert logfile.count(marker) == 1
