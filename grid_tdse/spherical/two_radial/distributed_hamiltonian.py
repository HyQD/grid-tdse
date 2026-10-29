"""Distributed field-free Hamiltonian actions and physical reductions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np

from .distributed_state import (
    DistributedTwoRadialState,
    TwoRadialDistributedLayout,
)


_STATE_HALO_TAG = 27101
_WEAK_ACTION_TAG = 27102


class DistributedOperatorCommunicator(Protocol):
    """Communicator operations required by the distributed Hamiltonian."""

    rank: int
    size: int

    def Irecv(self, buffer, source: int, tag: int): ...

    def Isend(self, buffer, dest: int, tag: int): ...

    def allreduce(self, value): ...


@dataclass(frozen=True)
class DistributedHamiltonianComponents:
    """Rank-local component actions, each in permanent owned storage."""

    coordinate1: DistributedTwoRadialState
    coordinate2: DistributedTwoRadialState
    coulomb: DistributedTwoRadialState

    @property
    def total(self) -> DistributedTwoRadialState:
        values = (
            self.coordinate1.coefficients
            + self.coordinate2.coefficients
            + self.coulomb.coefficients
        )
        return DistributedTwoRadialState(
            self.coordinate1.layout,
            self.coordinate1.rank,
            values,
        )


@dataclass(frozen=True)
class DistributedHamiltonianProfile:
    """Accumulated rank-local timings for distributed Hamiltonian actions.

    Communication timings include buffer preparation and request completion.
    ``coordinate2_local_seconds`` contains only the remaining local work, so
    it can be compared independently with the two bridge-exchange phases.
    """

    hamiltonian_calls: int = 0
    coordinate1_calls: int = 0
    coordinate1_seconds: float = 0.0
    coordinate2_calls: int = 0
    coordinate2_local_seconds: float = 0.0
    coulomb_calls: int = 0
    coulomb_seconds: float = 0.0
    halo_exchange_calls: int = 0
    halo_exchange_seconds: float = 0.0
    weak_exchange_calls: int = 0
    weak_exchange_seconds: float = 0.0


@dataclass(frozen=True)
class DistributedReductionProfile:
    """Accumulated rank-local timings for physical inner products."""

    local_inner_product_calls: int = 0
    local_inner_product_seconds: float = 0.0
    collective_calls: int = 0
    collective_seconds: float = 0.0


class TwoRadialDistributedHamiltonian:
    """Field-free two-radial Hamiltonian distributed over coordinate 2.

    All angular channels and coordinate 1 remain local to every rank.  The
    coordinate-2 FEM-DVR derivative exchanges bridge values with the left
    neighbor and returns the neighboring weak-form bridge contribution to the
    bridge owner before applying the inverse assembled mass.
    """

    def __init__(
        self,
        layout: TwoRadialDistributedLayout,
        communicator: DistributedOperatorCommunicator | None = None,
        *,
        tile_size: int = 2048,
        profile: bool = False,
    ) -> None:
        if not isinstance(layout, TwoRadialDistributedLayout):
            raise TypeError("layout must be a TwoRadialDistributedLayout")
        self.layout = layout
        self.representation = layout.representation
        self.communicator = communicator
        if not isinstance(profile, (bool, np.bool_)):
            raise TypeError("profile must be boolean")
        self.profile_enabled = bool(profile)
        self.reset_profile()
        self.serial_hamiltonian = self.representation.hamiltonian(
            tile_size=tile_size,
            radial_operator_mode="element-local",
        )
        if communicator is not None:
            rank, size = _communicator_rank_size(communicator)
            if size != layout.n_ranks:
                raise ValueError(
                    f"communicator has {size} ranks; layout expects "
                    f"{layout.n_ranks}"
                )
            self.rank = rank
        elif layout.n_ranks == 1:
            self.rank = 0
        else:
            self.rank = None

        if self.representation.uses_compact_angular_storage:
            pairs = self.representation.angular_pairs.pairs
            channels1 = np.asarray([first for first, _ in pairs], dtype=int)
            channels2 = np.asarray([second for _, second in pairs], dtype=int)
            self._h1_diagonal = np.ascontiguousarray(
                self.representation.h1_diagonal[channels1]
            )
            self._h2_diagonal = np.ascontiguousarray(
                self.representation.h2_diagonal[channels2]
            )
        else:
            self._h1_diagonal = self.representation.h1_diagonal
            self._h2_diagonal = self.representation.h2_diagonal

        self._coordinate2_element_matrices = _coordinate2_element_matrices(
            self.representation.grid2
        )

    def __call__(
        self, state: DistributedTwoRadialState
    ) -> DistributedTwoRadialState:
        """Apply the complete field-free Hamiltonian on one rank."""

        if self.profile_enabled:
            self._profile_counts["hamiltonian_calls"] += 1
        coordinate1 = self.apply_one_body_coordinate1(state)
        coordinate2 = self.apply_one_body_coordinate2(state)
        coulomb = self.apply_coulomb(state)
        values = coordinate1.coefficients
        values = values + coordinate2.coefficients + coulomb.coefficients
        return DistributedTwoRadialState(self.layout, state.rank, values)

    def components(
        self, state: DistributedTwoRadialState
    ) -> DistributedHamiltonianComponents:
        """Return separately evaluated coordinate and Coulomb actions."""

        return DistributedHamiltonianComponents(
            coordinate1=self.apply_one_body_coordinate1(state),
            coordinate2=self.apply_one_body_coordinate2(state),
            coulomb=self.apply_coulomb(state),
        )

    def apply_one_body_coordinate1(
        self, state: DistributedTwoRadialState
    ) -> DistributedTwoRadialState:
        """Apply the rank-local electronic-coordinate one-body term."""

        started = self._profile_start()
        self._validate_local_state(state)
        values = state.coefficients
        result = (
            self.representation.kinetic_prefactor1
            * self.representation.grid1.apply_local_second_derivative(
                values,
                axis=-2,
            )
        )
        if self.representation.uses_compact_angular_storage:
            result += self._h1_diagonal[:, :, None] * values
        else:
            result += self._h1_diagonal[:, None, :, None] * values
        self._profile_finish("coordinate1", started)
        return DistributedTwoRadialState(self.layout, state.rank, result)

    def apply_one_body_coordinate2(
        self, state: DistributedTwoRadialState
    ) -> DistributedTwoRadialState:
        """Apply the distributed internuclear-coordinate one-body term."""

        if self.profile_enabled:
            self._profile_counts["coordinate2_calls"] += 1
        started = self._profile_start()
        self._validate_local_state(state)
        self._profile_finish("coordinate2_local", started, count=False)

        halo_values = self._exchange_state_halo(state)
        started = self._profile_start()
        working = self._working_values(state, halo_values)
        self._profile_finish("coordinate2_local", started, count=False)

        started = self._profile_start()
        weak_owned, outgoing = self._local_coordinate2_weak(state.rank, working)
        self._profile_finish("coordinate2_local", started, count=False)
        incoming = self._exchange_weak_contributions(
            state.rank,
            outgoing,
            state.coefficients.dtype,
        )

        started = self._profile_start()
        result = self._finish_coordinate2(state, weak_owned, incoming)
        self._profile_finish("coordinate2_local", started, count=False)
        return DistributedTwoRadialState(self.layout, state.rank, result)

    def apply_coulomb(
        self, state: DistributedTwoRadialState
    ) -> DistributedTwoRadialState:
        """Apply point-local radial multipoles and angular couplings."""

        started = self._profile_start()
        self._validate_local_state(state)
        values = self.serial_hamiltonian.apply_coulomb_R_slice(
            state.coefficients,
            state.partition.owned_indices,
        )
        self._profile_finish("coulomb", started)
        return DistributedTwoRadialState(self.layout, state.rank, values)

    @property
    def profile(self) -> DistributedHamiltonianProfile:
        """Return an immutable snapshot of the rank-local profile."""

        return DistributedHamiltonianProfile(
            hamiltonian_calls=self._profile_counts["hamiltonian_calls"],
            coordinate1_calls=self._profile_counts["coordinate1_calls"],
            coordinate1_seconds=self._profile_seconds["coordinate1"],
            coordinate2_calls=self._profile_counts["coordinate2_calls"],
            coordinate2_local_seconds=self._profile_seconds[
                "coordinate2_local"
            ],
            coulomb_calls=self._profile_counts["coulomb_calls"],
            coulomb_seconds=self._profile_seconds["coulomb"],
            halo_exchange_calls=self._profile_counts["halo_exchange_calls"],
            halo_exchange_seconds=self._profile_seconds["halo_exchange"],
            weak_exchange_calls=self._profile_counts["weak_exchange_calls"],
            weak_exchange_seconds=self._profile_seconds["weak_exchange"],
        )

    def reset_profile(self) -> None:
        """Clear all accumulated rank-local profiling counters and timings."""

        self._profile_counts = {
            "hamiltonian_calls": 0,
            "coordinate1_calls": 0,
            "coordinate2_calls": 0,
            "coulomb_calls": 0,
            "halo_exchange_calls": 0,
            "weak_exchange_calls": 0,
        }
        self._profile_seconds = {
            "coordinate1": 0.0,
            "coordinate2_local": 0.0,
            "coulomb": 0.0,
            "halo_exchange": 0.0,
            "weak_exchange": 0.0,
        }

    def apply_all(
        self,
        states: Sequence[DistributedTwoRadialState],
    ) -> tuple[DistributedTwoRadialState, ...]:
        """Emulate the multi-rank action locally without a communicator.

        This validation path executes the same rank-local element actions and
        bridge assembly as the MPI path.  It does not assemble a global input
        state before applying the Hamiltonian.
        """

        states = self._rank_ordered_states(states)
        workings = []
        for rank, state in enumerate(states):
            partition = self.layout.partition(rank)
            halo = None
            if partition.n_halo:
                halo = _owned_global_values(
                    states[rank - 1],
                    partition.halo_indices,
                )
            workings.append(self._working_values(state, halo))

        weak = []
        outgoing = []
        for rank, working in enumerate(workings):
            owned, sent = self._local_coordinate2_weak(rank, working)
            weak.append(owned)
            outgoing.append(sent)

        results = []
        for rank, state in enumerate(states):
            incoming = None
            if rank + 1 < self.layout.n_ranks:
                incoming = (
                    self.layout.partition(rank + 1).halo_indices,
                    outgoing[rank + 1],
                )
            coordinate2 = self._finish_coordinate2(state, weak[rank], incoming)
            coordinate1 = self.apply_one_body_coordinate1(state).coefficients
            coulomb = self.apply_coulomb(state).coefficients
            results.append(
                DistributedTwoRadialState(
                    self.layout,
                    rank,
                    coordinate1 + coordinate2 + coulomb,
                )
            )
        return tuple(results)

    def _working_values(
        self,
        state: DistributedTwoRadialState,
        halo_values: np.ndarray | None,
    ) -> np.ndarray:
        partition = state.partition
        working = np.empty(
            self.layout.local_shape(state.rank),
            dtype=state.coefficients.dtype,
        )
        working[..., partition.owned_local_indices] = state.coefficients
        if partition.n_halo:
            if halo_values is None:
                raise ValueError("halo values are required for this rank")
            halo_values = np.asarray(halo_values)
            expected = (*state.coefficients.shape[:-1], partition.n_halo)
            if halo_values.shape != expected:
                raise ValueError(
                    f"halo values have shape {halo_values.shape}; expected "
                    f"{expected}"
                )
            working[..., partition.halo_local_indices] = halo_values
        return working

    def _local_coordinate2_weak(
        self,
        rank: int,
        working: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        partition = self.layout.partition(rank)
        weak = np.zeros_like(working)
        for offset, element in enumerate(partition.element_indices):
            local_map = partition.element_local_dof_maps[offset]
            matrix = self._coordinate2_element_matrices[element]
            local_values = working[..., local_map]
            local_action = np.einsum(
                "ij,...j->...i",
                matrix,
                local_values,
                optimize=True,
            )
            weak[..., local_map] += local_action
        owned = np.ascontiguousarray(weak[..., partition.owned_local_indices])
        outgoing = None
        if partition.n_halo:
            outgoing = np.ascontiguousarray(
                weak[..., partition.halo_local_indices]
            )
        return owned, outgoing

    def _finish_coordinate2(
        self,
        state: DistributedTwoRadialState,
        weak_owned: np.ndarray,
        incoming: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        weak_owned = weak_owned.copy()
        if incoming is not None:
            indices, values = incoming
            positions = np.searchsorted(state.partition.owned_indices, indices)
            if np.any(positions >= state.partition.n_owned) or not np.array_equal(
                state.partition.owned_indices[positions],
                indices,
            ):
                raise ValueError("received weak contribution for an unowned index")
            weak_owned[..., positions] += values

        scale = (
            self.representation.kinetic_prefactor2
            / state.owned_R_weights
        )
        result = weak_owned * scale.reshape(
            (1,) * (weak_owned.ndim - 1) + (scale.size,)
        )
        indices = state.partition.owned_indices
        if self.representation.uses_compact_angular_storage:
            result += self._h2_diagonal[:, None, indices] * state.coefficients
        else:
            result += (
                self._h2_diagonal[None, :, None, indices]
                * state.coefficients
            )
        return result

    def _exchange_state_halo(
        self,
        state: DistributedTwoRadialState,
    ) -> np.ndarray | None:
        partition = state.partition
        if self.layout.n_ranks == 1:
            return None
        started = self._profile_start()
        communicator = self._required_communicator(state.rank)
        requests = []
        buffers = []
        received = None
        if partition.n_halo:
            received = np.empty(
                (*state.coefficients.shape[:-1], partition.n_halo),
                dtype=state.coefficients.dtype,
            )
            requests.append(
                communicator.Irecv(
                    received,
                    source=state.rank - 1,
                    tag=_STATE_HALO_TAG,
                )
            )
        if state.rank + 1 < self.layout.n_ranks:
            requested = self.layout.partition(state.rank + 1).halo_indices
            sent = np.ascontiguousarray(_owned_global_values(state, requested))
            buffers.append(sent)
            requests.append(
                communicator.Isend(
                    sent,
                    dest=state.rank + 1,
                    tag=_STATE_HALO_TAG,
                )
            )
        _wait_all(requests)
        self._profile_finish("halo_exchange", started)
        return received

    def _exchange_weak_contributions(
        self,
        rank: int,
        outgoing: np.ndarray | None,
        dtype: np.dtype,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.layout.n_ranks == 1:
            return None
        started = self._profile_start()
        communicator = self._required_communicator(rank)
        requests = []
        buffers = []
        received = None
        received_indices = None
        if rank + 1 < self.layout.n_ranks:
            received_indices = self.layout.partition(rank + 1).halo_indices
            received = np.empty(
                (*self.layout.owned_shape(rank)[:-1], received_indices.size),
                dtype=dtype,
            )
        if rank > 0:
            if outgoing is None:
                raise ValueError("left-neighbor weak contribution is missing")
            sent = np.ascontiguousarray(outgoing)
            buffers.append(sent)
            requests.append(
                communicator.Isend(
                    sent,
                    dest=rank - 1,
                    tag=_WEAK_ACTION_TAG,
                )
            )
        if received is not None:
            requests.insert(
                0,
                communicator.Irecv(
                    received,
                    source=rank + 1,
                    tag=_WEAK_ACTION_TAG,
                ),
            )
        _wait_all(requests)
        self._profile_finish("weak_exchange", started)
        if received is None:
            return None
        return received_indices, received

    def _profile_start(self) -> float | None:
        if not self.profile_enabled:
            return None
        return perf_counter()

    def _profile_finish(
        self,
        name: str,
        started: float | None,
        *,
        count: bool = True,
    ) -> None:
        if started is None:
            return
        self._profile_seconds[name] += perf_counter() - started
        if count:
            self._profile_counts[f"{name}_calls"] += 1

    def _required_communicator(self, rank: int):
        if self.communicator is None:
            raise RuntimeError(
                "a communicator is required for a multi-rank Hamiltonian action"
            )
        communicator_rank, size = _communicator_rank_size(self.communicator)
        if size != self.layout.n_ranks or communicator_rank != rank:
            raise ValueError("communicator and local distributed state do not match")
        return self.communicator

    def _validate_local_state(self, state: DistributedTwoRadialState) -> None:
        if not isinstance(state, DistributedTwoRadialState):
            raise TypeError("state must be a DistributedTwoRadialState")
        if state.layout.layout_id != self.layout.layout_id:
            raise ValueError("state belongs to a different distributed layout")
        if self.rank is not None and state.rank != self.rank:
            raise ValueError(
                f"state belongs to rank {state.rank}; Hamiltonian belongs to "
                f"rank {self.rank}"
            )

    def _rank_ordered_states(
        self,
        states: Sequence[DistributedTwoRadialState],
    ) -> tuple[DistributedTwoRadialState, ...]:
        if len(states) != self.layout.n_ranks:
            raise ValueError(
                f"received {len(states)} local states; expected "
                f"{self.layout.n_ranks}"
            )
        ordered = [None] * self.layout.n_ranks
        for state in states:
            if not isinstance(state, DistributedTwoRadialState):
                raise TypeError(
                    "states must contain DistributedTwoRadialState objects"
                )
            if state.layout.layout_id != self.layout.layout_id:
                raise ValueError("state belongs to a different distributed layout")
            if ordered[state.rank] is not None:
                raise ValueError(f"duplicate state for rank {state.rank}")
            ordered[state.rank] = state
        if any(state is None for state in ordered):
            raise ValueError("one state is required for every rank")
        return tuple(ordered)


class TwoRadialDistributedReductions:
    """Quadrature-weighted local contributions and MPI sum reductions."""

    def __init__(
        self,
        layout: TwoRadialDistributedLayout,
        communicator: DistributedOperatorCommunicator | None = None,
        *,
        profile: bool = False,
    ) -> None:
        if not isinstance(layout, TwoRadialDistributedLayout):
            raise TypeError("layout must be a TwoRadialDistributedLayout")
        self.layout = layout
        self.communicator = communicator
        if not isinstance(profile, (bool, np.bool_)):
            raise TypeError("profile must be boolean")
        self.profile_enabled = bool(profile)
        self.reset_profile()
        if communicator is not None:
            rank, size = _communicator_rank_size(communicator)
            if size != layout.n_ranks:
                raise ValueError(
                    f"communicator has {size} ranks; layout expects "
                    f"{layout.n_ranks}"
                )
            self.rank = rank
        elif layout.n_ranks == 1:
            self.rank = 0
        else:
            self.rank = None

    def local_inner_product(
        self,
        left: DistributedTwoRadialState,
        right: DistributedTwoRadialState,
    ) -> complex:
        """Return one rank's nonoverlapping physical-inner-product term."""

        started = perf_counter() if self.profile_enabled else None
        self._validate_pair(left, right)
        subscripts = (
            "pab,pab,a,b->"
            if self.layout.representation.uses_compact_angular_storage
            else "ijab,ijab,a,b->"
        )
        value = np.einsum(
            subscripts,
            left.coefficients.conj(),
            right.coefficients,
            self.layout.representation.grid1.weights,
            left.owned_R_weights,
            optimize=True,
        )
        if started is not None:
            self._local_inner_product_calls += 1
            self._local_inner_product_seconds += perf_counter() - started
        return value

    @property
    def profile(self) -> DistributedReductionProfile:
        """Return an immutable snapshot of the rank-local profile."""

        return DistributedReductionProfile(
            local_inner_product_calls=self._local_inner_product_calls,
            local_inner_product_seconds=self._local_inner_product_seconds,
            collective_calls=self._collective_calls,
            collective_seconds=self._collective_seconds,
        )

    def reset_profile(self) -> None:
        """Clear all accumulated rank-local reduction timings."""

        self._local_inner_product_calls = 0
        self._local_inner_product_seconds = 0.0
        self._collective_calls = 0
        self._collective_seconds = 0.0

    def inner_product(
        self,
        left: DistributedTwoRadialState,
        right: DistributedTwoRadialState,
    ) -> complex:
        """Collectively return the global physical inner product."""

        local = self.local_inner_product(left, right)
        return self._allreduce(local, left.rank)

    def inner_product_all(
        self,
        left: Sequence[DistributedTwoRadialState],
        right: Sequence[DistributedTwoRadialState],
    ) -> complex:
        """Emulate a global sum from complete rank-local state collections."""

        left = _rank_ordered(self.layout, left)
        right = _rank_ordered(self.layout, right)
        return sum(
            (self.local_inner_product(a, b) for a, b in zip(left, right)),
            start=0.0j,
        )

    def norm_squared(self, state: DistributedTwoRadialState) -> float:
        value = self.inner_product(state, state).real
        return float(value)

    def norm(self, state: DistributedTwoRadialState) -> float:
        value = self.norm_squared(state)
        if value < 0.0:
            raise ValueError("distributed norm squared is negative")
        return float(np.sqrt(value))

    def expectation(
        self,
        state: DistributedTwoRadialState,
        operator_state: DistributedTwoRadialState,
    ) -> complex:
        """Return the unnormalized expectation ``<state|operator_state>``."""

        return self.inner_product(state, operator_state)

    def rayleigh_quotient(
        self,
        state: DistributedTwoRadialState,
        operator_state: DistributedTwoRadialState,
    ) -> complex:
        """Return the explicitly normalized operator expectation."""

        norm_squared = self.norm_squared(state)
        if norm_squared <= 0.0:
            raise ValueError("state must have positive norm")
        return self.expectation(state, operator_state) / norm_squared

    def residual_norm(
        self,
        state: DistributedTwoRadialState,
        operator_state: DistributedTwoRadialState,
        eigenvalue: complex,
    ) -> float:
        """Return the global norm of ``operator_state - eigenvalue * state``."""

        self._validate_pair(state, operator_state)
        residual = DistributedTwoRadialState(
            self.layout,
            state.rank,
            operator_state.coefficients - eigenvalue * state.coefficients,
        )
        return self.norm(residual)

    def _allreduce(self, value, rank: int):
        if self.layout.n_ranks == 1 and self.communicator is None:
            return value
        if self.communicator is None:
            raise RuntimeError("a communicator is required for a global reduction")
        communicator_rank, size = _communicator_rank_size(self.communicator)
        if communicator_rank != rank or size != self.layout.n_ranks:
            raise ValueError("communicator and local distributed state do not match")
        started = perf_counter() if self.profile_enabled else None
        result = self.communicator.allreduce(value)
        if started is not None:
            self._collective_calls += 1
            self._collective_seconds += perf_counter() - started
        return result

    def _validate_pair(
        self,
        left: DistributedTwoRadialState,
        right: DistributedTwoRadialState,
    ) -> None:
        for state in (left, right):
            if not isinstance(state, DistributedTwoRadialState):
                raise TypeError("states must be DistributedTwoRadialState objects")
            if state.layout.layout_id != self.layout.layout_id:
                raise ValueError("state belongs to a different distributed layout")
        if left.rank != right.rank:
            raise ValueError("local states must belong to the same rank")
        if self.rank is not None and left.rank != self.rank:
            raise ValueError(
                f"states belong to rank {left.rank}; reductions belong to "
                f"rank {self.rank}"
            )


def _coordinate2_element_matrices(grid) -> tuple[np.ndarray, ...]:
    dvr = grid.dvr
    matrices = []
    for element, (full_map, stiffness) in enumerate(
        zip(dvr.element_dof_maps, dvr.local_stiffness_matrices)
    ):
        del element
        retained = (full_map > 0) & (full_map < dvr.r.size - 1)
        matrices.append(
            np.ascontiguousarray(-stiffness[np.ix_(retained, retained)])
        )
    return tuple(matrices)


def _owned_global_values(
    state: DistributedTwoRadialState,
    global_indices: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(state.partition.owned_indices, global_indices)
    if np.any(positions >= state.partition.n_owned) or not np.array_equal(
        state.partition.owned_indices[positions],
        global_indices,
    ):
        raise ValueError("requested halo value is not owned by the source rank")
    return np.ascontiguousarray(state.coefficients[..., positions])


def _rank_ordered(
    layout: TwoRadialDistributedLayout,
    states: Sequence[DistributedTwoRadialState],
) -> tuple[DistributedTwoRadialState, ...]:
    if len(states) != layout.n_ranks:
        raise ValueError(
            f"received {len(states)} local states; expected {layout.n_ranks}"
        )
    ordered = [None] * layout.n_ranks
    for state in states:
        if state.layout.layout_id != layout.layout_id:
            raise ValueError("state belongs to a different distributed layout")
        if ordered[state.rank] is not None:
            raise ValueError(f"duplicate state for rank {state.rank}")
        ordered[state.rank] = state
    if any(state is None for state in ordered):
        raise ValueError("one state is required for every rank")
    return tuple(ordered)


def _communicator_rank_size(communicator) -> tuple[int, int]:
    try:
        return int(communicator.rank), int(communicator.size)
    except AttributeError:
        try:
            return int(communicator.Get_rank()), int(communicator.Get_size())
        except AttributeError as exc:
            raise TypeError(
                "communicator must expose rank/size or Get_rank/Get_size"
            ) from exc


def _wait_all(requests) -> None:
    for request in requests:
        try:
            request.Wait()
        except AttributeError:
            request.wait()
