"""MPI-compatible Lanczos imaginary-time propagation for two-radial states."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Literal

import numpy as np

from grid_tdse.ground_state import (
    GroundStateIteration,
    ImaginaryTimeGroundStateSolver,
)

from .distributed_hamiltonian import (
    DistributedHamiltonianProfile,
    DistributedOperatorCommunicator,
    DistributedReductionProfile,
    TwoRadialDistributedHamiltonian,
    TwoRadialDistributedReductions,
)
from .distributed_state import (
    DistributedTwoRadialState,
    TwoRadialDistributedLayout,
)
from .ground_state_sampling import (
    DiatomicIonGroundStateReporter,
    DiatomicIonGroundStateSample,
)
from .observables import DiatomicIonObservableEvaluator


@dataclass(frozen=True)
class DistributedGroundStateResult:
    """Rank-local state and global diagnostics from distributed ITP."""

    state: DistributedTwoRadialState
    energy: float
    state_change: float
    residual_norm: float
    converged: bool
    iterations: int
    imaginary_time: float
    integrator: Literal["sil"]
    termination_reason: Literal["converged", "maximum_iterations"]
    history: tuple[GroundStateIteration, ...]
    samples: tuple[DiatomicIonGroundStateSample, ...] = ()
    hamiltonian_profile: DistributedHamiltonianProfile = DistributedHamiltonianProfile()
    reduction_profile: DistributedReductionProfile = DistributedReductionProfile()

    @property
    def energy_history(self) -> np.ndarray:
        """Global energy after every accepted imaginary-time step."""

        return np.asarray([record.energy for record in self.history])

    @property
    def residual_history(self) -> np.ndarray:
        """Global eigenstate residual after every accepted step."""

        return np.asarray([record.residual_norm for record in self.history])

    @property
    def state_change_history(self) -> np.ndarray:
        """Global phase-invariant state change after every accepted step."""

        return np.asarray([record.state_change for record in self.history])

    @property
    def norm(self) -> float:
        """Global physical norm of the returned distributed state."""

        return self.history[-1].norm

    @property
    def krylov_operator_calls(self) -> np.ndarray:
        """Hamiltonian calls made by Lanczos during every ITP step."""

        return np.asarray(
            [record.krylov_operator_calls for record in self.history],
            dtype=int,
        )

    @property
    def krylov_dimensions(self) -> np.ndarray:
        """Largest accepted Lanczos dimension during every ITP step."""

        return np.asarray(
            [record.krylov_dimension for record in self.history],
            dtype=int,
        )

    @property
    def krylov_accepted_substeps(self) -> np.ndarray:
        """Accepted exponential substeps during every ITP step."""

        return np.asarray(
            [record.krylov_accepted_substeps for record in self.history],
            dtype=int,
        )

    @property
    def krylov_rejected_substeps(self) -> np.ndarray:
        """Rejected exponential substeps during every ITP step."""

        return np.asarray(
            [record.krylov_rejected_substeps for record in self.history],
            dtype=int,
        )

    @property
    def krylov_refinement_depths(self) -> np.ndarray:
        """Maximum adaptive refinement depth during every ITP step."""

        return np.asarray(
            [record.krylov_refinement_depth for record in self.history],
            dtype=int,
        )

    @property
    def krylov_error_estimates(self) -> np.ndarray:
        """Accumulated accepted-substep error estimate for every ITP step."""

        return np.asarray(
            [record.krylov_error_estimate for record in self.history],
            dtype=float,
        )


class TwoRadialDistributedGroundStateSolver:
    """Find a two-radial ground state with distributed Lanczos ITP.

    Each MPI rank advances its permanently owned coefficient array. Hamiltonian
    actions exchange only neighboring coordinate-2 bridge data, while every
    physical inner product is a collective reduction. The generic
    representation-independent imaginary-time solver supplies the Lanczos
    iteration, normalization, convergence measures, and diagnostics.

    The first distributed implementation intentionally supports only SIL. A
    communicator is optional for a one-rank layout and required otherwise.
    Progress is printed only by ``root``.
    """

    def __init__(
        self,
        layout: TwoRadialDistributedLayout,
        communicator: DistributedOperatorCommunicator | None = None,
        *,
        root: int = 0,
        time_step: float = 0.25,
        max_iterations: int = 1_000,
        energy_tolerance: float = 1.0e-10,
        state_tolerance: float = 1.0e-8,
        residual_tolerance: float = 1.0e-8,
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
        if not isinstance(layout, TwoRadialDistributedLayout):
            raise TypeError("layout must be a TwoRadialDistributedLayout")
        rank, size = _rank_and_size(layout, communicator)
        root = _validated_root(root, size)

        self.layout = layout
        self.communicator = communicator
        self.rank = rank
        self.root = root
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
        self.observable_evaluator = observable_evaluator
        self.sample_interval = _optional_positive_integer(
            sample_interval, "sample_interval"
        )
        self.log_path = log_path
        self.log_overwrite = bool(log_overwrite)
        self.hamiltonian = TwoRadialDistributedHamiltonian(
            layout,
            communicator,
            tile_size=tile_size,
            profile=profile,
        )
        self.reductions = TwoRadialDistributedReductions(
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
            integrator="sil",
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
        initial_state: DistributedTwoRadialState,
    ) -> DistributedGroundStateResult:
        """Run collective normalized Lanczos imaginary-time propagation."""

        self._validate_state(initial_state)
        self.hamiltonian.reset_profile()
        self.reductions.reset_profile()

        def hamiltonian_action(coefficients):
            local_state = DistributedTwoRadialState(
                self.layout,
                self.rank,
                coefficients,
            )
            return self.hamiltonian(local_state).coefficients

        def inner_product(left, right):
            left_state = DistributedTwoRadialState(
                self.layout,
                self.rank,
                left,
            )
            right_state = DistributedTwoRadialState(
                self.layout,
                self.rank,
                right,
            )
            return self.reductions.inner_product(left_state, right_state)

        def inner_products(left, right):
            left_states = tuple(
                DistributedTwoRadialState(
                    self.layout,
                    self.rank,
                    values,
                )
                for values in left
            )
            right_state = DistributedTwoRadialState(
                self.layout,
                self.rank,
                right,
            )
            return self.reductions.inner_products(left_states, right_state)

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
        state = DistributedTwoRadialState(
            self.layout,
            self.rank,
            result.state,
        )
        return DistributedGroundStateResult(
            state=state,
            energy=result.energy,
            state_change=result.state_change,
            residual_norm=result.residual_norm,
            converged=result.converged,
            iterations=result.iterations,
            imaginary_time=result.imaginary_time,
            integrator="sil",
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
            state = DistributedTwoRadialState(
                self.layout,
                self.rank,
                coefficients,
            )
            return self.observable_evaluator.distributed(state, self.reductions)

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

    def _validate_state(self, state: DistributedTwoRadialState) -> None:
        if not isinstance(state, DistributedTwoRadialState):
            raise TypeError("initial_state must be a DistributedTwoRadialState")
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
        root,
        (int, np.integer),
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
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a positive integer or None")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
