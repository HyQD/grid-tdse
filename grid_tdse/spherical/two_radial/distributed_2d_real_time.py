r"""Cartesian-MPI SIL/SIA propagation for two-radial dynamics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Protocol

import numpy as np

from grid_tdse.real_time import RealTimePropagationResult, RealTimePropagator

from .absorbing_boundaries import TwoRadialAbsorbingBoundaries
from .diatomic import DiatomicIonParameters
from .dipole import DipoleGauge
from .distributed_2d_dipole import (
    TwoRadialDistributed2DLengthGaugeInteraction,
    TwoRadialDistributed2DVelocityGaugeInteraction,
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
from .dynamics_sampling import (
    DiatomicIonDynamicsObservableEvaluator,
    DiatomicIonDynamicsSampler,
    DiatomicIonDynamicsSamplingPolicy,
    DiatomicIonDynamicsTimeSeries,
)


class CartesianRealTimeCommunicator(Protocol):
    """MPI-compatible operations required by Cartesian real-time dynamics."""

    rank: int
    size: int

    def Irecv(self, buffer, source: int, tag: int): ...

    def Isend(self, buffer, dest: int, tag: int): ...

    def allreduce(self, value): ...

    def Allreduce(self, send_buffer, receive_buffer): ...


@dataclass(frozen=True)
class Distributed2DRealTimeResult:
    """Rank-local final state and global real-time diagnostics."""

    state: DistributedTwoRadial2DState
    propagation: RealTimePropagationResult
    time_series: DiatomicIonDynamicsTimeSeries
    hamiltonian_profile: Distributed2DHamiltonianProfile = (
        Distributed2DHamiltonianProfile()
    )
    reduction_profile: DistributedReductionProfile = DistributedReductionProfile()

    @property
    def n_steps(self) -> int:
        return self.propagation.n_steps

    @property
    def norm(self) -> float:
        return self.propagation.norm

    @property
    def history(self):
        return self.propagation.history

    @property
    def times(self) -> np.ndarray:
        return self.propagation.times

    @property
    def norm_history(self) -> np.ndarray:
        return self.propagation.norm_history


Distributed2DStateObserver = Callable[
    [DistributedTwoRadial2DState, int, float, object | None, bool],
    None,
]


class TwoRadialDistributed2DRealTimeSolver:
    """Propagate a Cartesian two-radial state with collective SIL or SIA.

    Every rank retains only its permanently owned radial tile.  The
    field-free Hamiltonian exchanges FEM-DVR bridge values along the two
    Cartesian process directions.  The z-polarized length-gauge interaction
    is point-local, while the velocity-gauge interaction performs the bridge
    exchanges required by its first radial derivatives.  Krylov inner
    products and observable samples use batched physical MPI reductions.
    """

    def __init__(
        self,
        layout: TwoRadialDistributed2DLayout,
        parameters: DiatomicIonParameters,
        communicator: CartesianRealTimeCommunicator | None = None,
        *,
        fields: Callable[[float], float] | Iterable[Callable[[float], float]] = (),
        field_gauge: DipoleGauge = "length",
        initial_time: float = 0.0,
        final_time: float,
        time_step: float,
        integrator: Literal["sil", "sia"] = "sil",
        max_krylov_dimension: int = 20,
        krylov_rtol: float = 1.0e-10,
        krylov_atol: float = 0.0,
        max_krylov_refinements: int = 8,
        krylov_reorthogonalize: bool = True,
        krylov_breakdown_tolerance: float = 1.0e-14,
        reality_tolerance: float = 1.0e-10,
        sampling_policy: DiatomicIonDynamicsSamplingPolicy | None = None,
        absorbing_boundaries: TwoRadialAbsorbingBoundaries | None = None,
        root: int = 0,
        progress_interval: int | None = 10,
        tile_size: int = 2048,
        profile: bool = False,
    ) -> None:
        if not isinstance(layout, TwoRadialDistributed2DLayout):
            raise TypeError("layout must be a TwoRadialDistributed2DLayout")
        if not isinstance(parameters, DiatomicIonParameters):
            raise TypeError("parameters must be a DiatomicIonParameters")
        if integrator not in ("sil", "sia"):
            raise ValueError("integrator must be 'sil' or 'sia'")
        if absorbing_boundaries is not None:
            if not isinstance(absorbing_boundaries, TwoRadialAbsorbingBoundaries):
                raise TypeError(
                    "absorbing_boundaries must be a TwoRadialAbsorbingBoundaries"
                )
            if absorbing_boundaries.representation is not layout.representation:
                raise ValueError(
                    "absorbing boundaries belong to a different representation"
                )
            if integrator == "sil" and absorbing_boundaries.has_cap:
                raise ValueError(
                    "integrator 'sil' requires a Hermitian Hamiltonian and "
                    "cannot be used with a CAP; use 'sia'"
                )
        if not isinstance(profile, (bool, np.bool_)):
            raise TypeError("profile must be boolean")
        self.layout = layout
        self.representation = layout.representation
        self.parameters = parameters
        self.communicator = communicator
        self.rank, size = _communicator_rank_size(communicator, layout.n_ranks)
        if size != layout.n_ranks:
            raise ValueError(
                f"communicator has {size} ranks; layout requires {layout.n_ranks}"
            )
        self.root = _validated_root(root, size)
        self.integrator = integrator
        if field_gauge not in ("length", "velocity"):
            raise ValueError("field_gauge must be 'length' or 'velocity'")
        self.field_gauge = field_gauge
        self.fields = _field_profiles(fields, field_gauge)
        _validate_velocity_gauge_initial_vector_potential(
            self.fields,
            gauge=field_gauge,
            initial_time=initial_time,
        )
        if sampling_policy is None:
            sampling_policy = DiatomicIonDynamicsSamplingPolicy()
        if not isinstance(sampling_policy, DiatomicIonDynamicsSamplingPolicy):
            raise TypeError(
                "sampling_policy must be a DiatomicIonDynamicsSamplingPolicy"
            )
        self.sampling_policy = sampling_policy
        self.absorbing_boundaries = absorbing_boundaries
        self.hamiltonian = TwoRadialDistributed2DHamiltonian(
            layout,
            communicator,
            tile_size=tile_size,
            absorbing_boundaries=absorbing_boundaries,
            profile=profile,
        )
        self.reductions = TwoRadialDistributed2DReductions(
            layout,
            communicator,
            profile=profile,
        )
        if field_gauge == "length":
            self.dipole_interaction = (
                None
                if not self.fields
                else TwoRadialDistributed2DLengthGaugeInteraction(
                    layout,
                    parameters,
                    fields=self.fields,
                )
            )
        else:
            interaction_fields = self.fields or (lambda time: 0.0,)
            self.dipole_interaction = (
                TwoRadialDistributed2DVelocityGaugeInteraction(
                    layout,
                    parameters,
                    self.hamiltonian,
                    fields=interaction_fields,
                )
            )
        velocity_action = (
            self.dipole_interaction.spatial_action
            if field_gauge == "velocity"
            else None
        )
        self.observable_evaluator = DiatomicIonDynamicsObservableEvaluator(
            self.representation,
            parameters,
            self.hamiltonian.physical_action,
            gauge=field_gauge,
            fields=self.fields,
            velocity_action=velocity_action,
            radial_operator_mode="element-local",
            reality_tolerance=reality_tolerance,
        )
        self.propagator = RealTimePropagator(
            initial_time=initial_time,
            final_time=final_time,
            time_step=time_step,
            integrator=integrator,
            max_krylov_dimension=max_krylov_dimension,
            krylov_rtol=krylov_rtol,
            krylov_atol=krylov_atol,
            max_krylov_refinements=max_krylov_refinements,
            krylov_reorthogonalize=krylov_reorthogonalize,
            krylov_breakdown_tolerance=krylov_breakdown_tolerance,
            reality_tolerance=reality_tolerance,
            progress_interval=(progress_interval if self.rank == self.root else None),
        )

    def field_action(
        self,
        state: DistributedTwoRadial2DState,
        time: float,
    ) -> DistributedTwoRadial2DState:
        """Return only the configured gauge interaction action."""

        self._validate_state(state)
        if self.dipole_interaction is None:
            return DistributedTwoRadial2DState(
                self.layout,
                self.rank,
                np.zeros_like(state.coefficients),
            )
        return self.dipole_interaction(state, time)

    def total_action(
        self,
        state: DistributedTwoRadial2DState,
        time: float,
    ) -> DistributedTwoRadial2DState:
        """Return the field-free plus configured-gauge Hamiltonian action."""

        self._validate_state(state)
        field_free = self.hamiltonian(state)
        if self.dipole_interaction is None:
            return field_free
        field = self.dipole_interaction(state, time)
        return DistributedTwoRadial2DState(
            self.layout,
            self.rank,
            field_free.coefficients + field.coefficients,
        )

    def run(
        self,
        initial_state: DistributedTwoRadial2DState,
        *,
        state_observer: Distributed2DStateObserver | None = None,
    ) -> Distributed2DRealTimeResult:
        """Collectively propagate one rank-local initial state."""

        self._validate_state(initial_state)
        if state_observer is not None and not callable(state_observer):
            raise TypeError("state_observer must be callable or None")
        self.hamiltonian.reset_profile()
        self.reductions.reset_profile()
        sampler = DiatomicIonDynamicsSampler(
            self.observable_evaluator,
            policy=self.sampling_policy,
        )

        def local_state(coefficients):
            return DistributedTwoRadial2DState(
                self.layout,
                self.rank,
                coefficients,
            )

        def hamiltonian_action(coefficients, time):
            return self.total_action(local_state(coefficients), time).coefficients

        def inner_product(left, right):
            return self.reductions.inner_product(
                local_state(left),
                local_state(right),
            )

        def observe(coefficients, step, time, diagnostics, final, checkpoint):
            state = local_state(coefficients)
            sampler.sample_distributed_2d(
                state,
                self.reductions,
                step,
                time,
                diagnostics=diagnostics,
                final=final,
                checkpoint=checkpoint,
            )
            if state_observer is not None:
                state_observer(state, step, time, diagnostics, final)

        def post_step(coefficients, time):
            state = self.hamiltonian.apply_mask(local_state(coefficients), time)
            return state.coefficients

        propagation = self.propagator.propagate(
            initial_state.coefficients,
            hamiltonian_action,
            inner_product,
            state_observer=observe,
            post_step=(
                post_step
                if self.absorbing_boundaries is not None
                and self.absorbing_boundaries.has_mask
                else None
            ),
        )
        return Distributed2DRealTimeResult(
            state=local_state(propagation.state),
            propagation=propagation,
            time_series=sampler.time_series,
            hamiltonian_profile=self.hamiltonian.profile,
            reduction_profile=self.reductions.profile,
        )

    __call__ = run

    def _validate_state(self, state: DistributedTwoRadial2DState) -> None:
        if not isinstance(state, DistributedTwoRadial2DState):
            raise TypeError("state must be a DistributedTwoRadial2DState")
        if state.layout.layout_id != self.layout.layout_id:
            raise ValueError("state belongs to a different distributed layout")
        if state.rank != self.rank:
            raise ValueError(
                f"state belongs to rank {state.rank}; solver belongs to rank "
                f"{self.rank}"
            )


def _field_profiles(
    fields,
    gauge: DipoleGauge,
) -> tuple[Callable[[float], float], ...]:
    if callable(fields):
        profiles = (fields,)
    elif isinstance(fields, (str, bytes)):
        raise TypeError("fields must contain callables")
    else:
        profiles = tuple(fields)
    for field in profiles:
        if not callable(field):
            raise TypeError("every field must be callable")
        field_gauge = getattr(field, "gauge", gauge)
        if field_gauge != gauge:
            raise ValueError(
                f"field uses gauge {field_gauge!r}, expected {gauge!r}"
            )
    return profiles


def _validate_velocity_gauge_initial_vector_potential(
    fields,
    *,
    gauge: DipoleGauge,
    initial_time: float,
) -> None:
    if gauge != "velocity" or not fields:
        return
    values = []
    for field in fields:
        value = np.asarray(field(initial_time))
        if value.shape != () or not np.issubdtype(value.dtype, np.number):
            raise TypeError("velocity-gauge fields must return scalar numeric values")
        value = complex(value.item())
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError("velocity-gauge fields must return finite values")
        values.append(value)
    if abs(sum(values, 0.0j)) > 1.0e-14:
        raise ValueError(
            "velocity-gauge dynamics currently require A(initial_time)=0; "
            "a nonzero vector potential requires an explicit gauge "
            "transformation of the field-free ground state"
        )


def _communicator_rank_size(communicator, expected_size):
    if communicator is None:
        if expected_size != 1:
            raise ValueError("a communicator is required when layout.n_ranks > 1")
        return 0, 1
    rank = getattr(communicator, "rank", None)
    size = getattr(communicator, "size", None)
    if rank is None:
        rank = communicator.Get_rank()
    if size is None:
        size = communicator.Get_size()
    return int(rank), int(size)


def _validated_root(value, size):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("root must be an integer")
    root = int(value)
    if not 0 <= root < size:
        raise ValueError(f"root must be in [0, {size})")
    return root
