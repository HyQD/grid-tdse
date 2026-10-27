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

    def communicator(self, rank):
        return _ThreadCommunicator(self, rank)


class _ThreadCommunicator:
    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.size = group.size
        self.reduction_round = 0

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


def _distributed_results(calculation, n_ranks, options):
    representation = calculation.representation
    layout = TwoRadialDistributedLayout(representation, n_ranks)
    local_initial = layout.distribute(calculation.initial_state)
    group = _ThreadCommunicatorGroup(n_ranks)
    solvers = [
        TwoRadialDistributedGroundStateSolver(
            layout,
            group.communicator(rank),
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
