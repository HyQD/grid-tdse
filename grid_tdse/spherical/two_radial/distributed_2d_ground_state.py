"""SIL/SIA imaginary-time propagation on Cartesian two-radial layouts."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Literal, Protocol

import numpy as np

from grid_tdse.ground_state import (
    GroundStateIteration,
    ImaginaryTimeGroundStateSolver,
)

from .distributed_2d_hamiltonian import (
    Distributed2DHamiltonianProfile,
    TwoRadialDistributed2DHamiltonian,
)
from .distributed_2d_reductions import TwoRadialDistributed2DReductions
from .distributed_2d_state import (
    DistributedTwoRadial2DState,
    TwoRadialDistributed2DLayout,
)
from .distributed_hamiltonian import DistributedReductionProfile
from .ground_state_sampling import (
    DiatomicIonGroundStateReporter,
    DiatomicIonGroundStateSample,
)
from .observables import DiatomicIonObservableEvaluator


class CartesianGroundStateCommunicator(Protocol):
    """MPI-compatible operations required by the Cartesian solver."""

    rank: int
    size: int

    def Irecv(self, buffer, source: int, tag: int): ...

    def Isend(self, buffer, dest: int, tag: int): ...

    def allreduce(self, value): ...

    def Allreduce(self, send_buffer, receive_buffer): ...

    def allgather(self, value): ...


@dataclass(frozen=True)
class Distributed2DGroundStateResult:
    """Rank-local state and global diagnostics from Cartesian ITP."""

    state: DistributedTwoRadial2DState
    energy: float
    state_change: float
    residual_norm: float
    converged: bool
    iterations: int
    imaginary_time: float
    integrator: Literal["sil", "sia"]
    termination_reason: Literal["converged", "maximum_iterations"]
    history: tuple[GroundStateIteration, ...]
    samples: tuple[DiatomicIonGroundStateSample, ...] = ()
    hamiltonian_profile: Distributed2DHamiltonianProfile = (
        Distributed2DHamiltonianProfile()
    )
    reduction_profile: DistributedReductionProfile = DistributedReductionProfile()

    @property
    def energy_history(self) -> np.ndarray:
        return np.asarray([record.energy for record in self.history])

    @property
    def residual_history(self) -> np.ndarray:
        return np.asarray([record.residual_norm for record in self.history])

    @property
    def state_change_history(self) -> np.ndarray:
        return np.asarray([record.state_change for record in self.history])

    @property
    def norm(self) -> float:
        return self.history[-1].norm

    @property
    def krylov_operator_calls(self) -> np.ndarray:
        return np.asarray(
            [record.krylov_operator_calls for record in self.history],
            dtype=int,
        )

    @property
    def krylov_dimensions(self) -> np.ndarray:
        return np.asarray(
            [record.krylov_dimension for record in self.history],
            dtype=int,
        )

    @property
    def krylov_accepted_substeps(self) -> np.ndarray:
        return np.asarray(
            [record.krylov_accepted_substeps for record in self.history],
            dtype=int,
        )

    @property
    def krylov_rejected_substeps(self) -> np.ndarray:
        return np.asarray(
            [record.krylov_rejected_substeps for record in self.history],
            dtype=int,
        )

    @property
    def krylov_refinement_depths(self) -> np.ndarray:
        return np.asarray(
            [record.krylov_refinement_depth for record in self.history],
            dtype=int,
        )

    @property
    def krylov_error_estimates(self) -> np.ndarray:
        return np.asarray(
            [record.krylov_error_estimate for record in self.history],
            dtype=float,
        )


class TwoRadialDistributed2DGroundStateSolver:
    """Find a two-radial ground state with Cartesian SIL or SIA ITP.

    Every rank advances only its permanently owned Cartesian radial tile. The
    Hamiltonian exchanges edge bridge values independently along both process
    grid directions, and physical inner products use batched collective
    reductions. Ground-state CAPs are intentionally unsupported.
    """

    _INTEGRATORS = {"sil", "sia"}

    def __init__(
        self,
        layout: TwoRadialDistributed2DLayout,
        communicator: CartesianGroundStateCommunicator | None = None,
        *,
        root: int = 0,
        time_step: float = 0.25,
        max_iterations: int = 1_000,
        energy_tolerance: float = 1.0e-10,
        state_tolerance: float = 1.0e-8,
        residual_tolerance: float = 1.0e-8,
        integrator: Literal["sil", "sia"] = "sil",
        max_krylov_dimension: int = 20,
        krylov_rtol: float = 1.0e-10,
        krylov_atol: float = 0.0,
        max_krylov_refinements: int = 8,
        krylov_reorthogonalize: bool = True,
        krylov_breakdown_tolerance: float = 1.0e-14,
        reality_tolerance: float = 1.0e-10,
        progress_interval: int | None = 10,
        observable_evaluator: DiatomicIonObservableEvaluator | None = None,
        sample_interval: int | None = 10,
        log_path: str | PathLike[str] | None = None,
        log_overwrite: bool = False,
        tile_size: int = 2048,
        profile: bool = False,
    ) -> None:
        if not isinstance(layout, TwoRadialDistributed2DLayout):
            raise TypeError("layout must be a TwoRadialDistributed2DLayout")
        if integrator not in self._INTEGRATORS:
            raise ValueError("integrator must be 'sil' or 'sia'")
        rank, size = _rank_and_size(layout, communicator)
        root = _validated_root(root, size)
        if observable_evaluator is not None and not isinstance(
            observable_evaluator, DiatomicIonObservableEvaluator
        ):
            raise TypeError(
                "observable_evaluator must be a "
                "DiatomicIonObservableEvaluator or None"
            )
        if (
            observable_evaluator is not None
            and observable_evaluator.system is not layout.representation
        ):
            raise ValueError("observable_evaluator uses a different representation")
        if log_path is not None and observable_evaluator is None:
            raise ValueError("log_path requires observable_evaluator")
        if not isinstance(log_overwrite, (bool, np.bool_)):
            raise TypeError("log_overwrite must be boolean")

        self.layout = layout
        self.communicator = communicator
        self.rank = rank
        self.root = root
        self.integrator = integrator
        self.observable_evaluator = observable_evaluator
        self.sample_interval = _optional_positive_integer(
            sample_interval, "sample_interval"
        )
        self.log_path = log_path
        self.log_overwrite = bool(log_overwrite)
        self.hamiltonian = TwoRadialDistributed2DHamiltonian(
            layout,
            communicator,
            tile_size=tile_size,
            profile=profile,
        )
        self.reductions = TwoRadialDistributed2DReductions(
            layout,
            communicator,
            profile=profile,
        )
        self.solver = ImaginaryTimeGroundStateSolver(
            time_step=time_step,
            max_iterations=max_iterations,
            energy_tolerance=energy_tolerance,
            state_tolerance=state_tolerance,
            residual_tolerance=residual_tolerance,
            integrator=integrator,
            max_krylov_dimension=max_krylov_dimension,
            krylov_rtol=krylov_rtol,
            krylov_atol=krylov_atol,
            max_krylov_refinements=max_krylov_refinements,
            krylov_reorthogonalize=krylov_reorthogonalize,
            krylov_breakdown_tolerance=krylov_breakdown_tolerance,
            reality_tolerance=reality_tolerance,
            progress_interval=progress_interval,
        )
        if rank != root:
            self.solver.progress_interval = None

    def solve(
        self,
        initial_state: DistributedTwoRadial2DState,
    ) -> Distributed2DGroundStateResult:
        """Run collective normalized imaginary-time propagation."""

        self._validate_state(initial_state)
        self.hamiltonian.reset_profile()
        self.reductions.reset_profile()

        def local_state(coefficients):
            return DistributedTwoRadial2DState(
                self.layout,
                self.rank,
                coefficients,
            )

        def hamiltonian_action(coefficients):
            return self.hamiltonian(local_state(coefficients)).coefficients

        def inner_product(left, right):
            return self.reductions.inner_product(
                local_state(left),
                local_state(right),
            )

        def inner_products(left, right):
            return self.reductions.inner_products(
                tuple(local_state(values) for values in left),
                local_state(right),
            )

        reporter = self._reporter()
        try:
            result = self.solver.solve(
                initial_state.coefficients,
                hamiltonian_action,
                inner_product,
                inner_products=inner_products,
                iteration_observer=(
                    reporter.observe if reporter is not None else None
                ),
                observer_interval=self.sample_interval,
                print_progress=reporter is None and self.rank == self.root,
            )
        finally:
            if reporter is not None:
                reporter.close()

        return Distributed2DGroundStateResult(
            state=local_state(result.state),
            energy=result.energy,
            state_change=result.state_change,
            residual_norm=result.residual_norm,
            converged=result.converged,
            iterations=result.iterations,
            imaginary_time=result.imaginary_time,
            integrator=self.integrator,
            termination_reason=result.termination_reason,
            history=result.history,
            samples=() if reporter is None else tuple(reporter.samples),
            hamiltonian_profile=self.hamiltonian.profile,
            reduction_profile=self.reductions.profile,
        )

    __call__ = solve

    def _reporter(self):
        if self.observable_evaluator is None:
            return None

        def evaluate(coefficients):
            return self.observable_evaluator.distributed_2d(
                DistributedTwoRadial2DState(
                    self.layout,
                    self.rank,
                    coefficients,
                ),
                self.reductions,
            )

        def reduce_max(value):
            if self.layout.n_ranks == 1:
                return float(value)
            return max(float(item) for item in self.communicator.allgather(value))

        def synchronize_error(error):
            if self.layout.n_ranks == 1:
                return error
            errors = self.communicator.allgather(error)
            return next((item for item in errors if item is not None), None)

        return DiatomicIonGroundStateReporter(
            evaluate,
            log_path=self.log_path,
            log_overwrite=self.log_overwrite,
            is_root=self.rank == self.root,
            reduce_max=reduce_max,
            synchronize_error=synchronize_error,
        )

    def _validate_state(self, state: DistributedTwoRadial2DState) -> None:
        if not isinstance(state, DistributedTwoRadial2DState):
            raise TypeError(
                "initial_state must be a DistributedTwoRadial2DState"
            )
        if state.layout.layout_id != self.layout.layout_id:
            raise ValueError("initial_state belongs to a different layout")
        if state.rank != self.rank:
            raise ValueError(
                f"initial_state belongs to rank {state.rank}; solver belongs "
                f"to rank {self.rank}"
            )


def _rank_and_size(layout, communicator) -> tuple[int, int]:
    if communicator is None:
        if layout.n_ranks != 1:
            raise ValueError(
                "a communicator is required when layout.n_ranks is greater "
                "than one"
            )
        return 0, 1
    try:
        rank, size = communicator.rank, communicator.size
    except AttributeError:
        try:
            rank, size = communicator.Get_rank(), communicator.Get_size()
        except AttributeError as exc:
            raise TypeError(
                "communicator must expose rank/size or Get_rank/Get_size"
            ) from exc
    rank = int(rank)
    size = int(size)
    if size != layout.n_ranks:
        raise ValueError(
            f"communicator has {size} ranks; layout expects {layout.n_ranks}"
        )
    if not 0 <= rank < size:
        raise ValueError(f"communicator rank {rank} is invalid for size {size}")
    return rank, size


def _validated_root(root, size: int) -> int:
    if isinstance(root, (bool, np.bool_)) or not isinstance(
        root, (int, np.integer)
    ):
        raise TypeError("root must be an integer")
    root = int(root)
    if not 0 <= root < size:
        raise ValueError(f"root must be between 0 and {size - 1}")
    return root


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer or None")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
