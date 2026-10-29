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

    def Allreduce(self, send_buffer, receive_buffer): ...


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
    collective_values: int = 0
    collective_seconds: float = 0.0


@dataclass
class _PendingExchange:
    received: np.ndarray | None
    received_indices: np.ndarray | None
    requests: list
    setup_seconds: float | None


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

        self._working_buffers = {}
        self._weak_buffers = {}
        self._owned_weak_buffers = {}
        self._halo_receive_buffers = {}
        self._halo_send_buffers = {}
        self._weak_receive_buffers = {}
        self._weak_send_buffers = {}

        ranks = range(layout.n_ranks) if self.rank is None else (self.rank,)
        radial_values = self.serial_hamiltonian.radial_multipoles.values
        self._local_radial_coulomb = {
            rank: np.ascontiguousarray(
                radial_values[..., layout.partition(rank).owned_indices].reshape(
                    radial_values.shape[0],
                    -1,
                )
            )
            for rank in ranks
        }

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
            self._profile_counts["coordinate2_calls"] += 1
        self._validate_local_state(state)
        pending_halo = self._start_state_halo_exchange(state)

        coordinate1 = self.apply_one_body_coordinate1(state)
        values = coordinate1.coefficients
        self._apply_coulomb_values(state, out=values)

        partition = state.partition
        started = self._profile_start()
        working = self._working_values(state, None, require_halo=False)
        weak = self._zero_weak_buffer(state.rank, state.coefficients.dtype)
        interior = range(1, len(partition.element_local_dof_maps))
        if not partition.n_halo:
            interior = range(len(partition.element_local_dof_maps))
        self._accumulate_coordinate2_elements(
            state.rank,
            working,
            weak,
            interior,
        )
        self._profile_finish("coordinate2_local", started, count=False)

        halo_values = self._finish_state_halo_exchange(pending_halo)
        if partition.n_halo:
            started = self._profile_start()
            self._set_working_halo(state, working, halo_values)
            self._accumulate_coordinate2_elements(
                state.rank,
                working,
                weak,
                (0,),
            )
            self._profile_finish("coordinate2_local", started, count=False)

        weak_owned, outgoing = self._owned_weak_values(state.rank, weak)
        pending_weak = self._start_weak_exchange(
            state.rank,
            outgoing,
            state.coefficients.dtype,
        )
        started = self._profile_start()
        self._finish_coordinate2(state, weak_owned, None, out=values)
        self._profile_finish("coordinate2_local", started, count=False)
        incoming = self._finish_weak_exchange(pending_weak)
        if incoming is not None:
            started = self._profile_start()
            self._add_coordinate2_incoming(state, values, incoming)
            self._profile_finish("coordinate2_local", started, count=False)
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
        self._validate_local_state(state)
        pending_halo = self._start_state_halo_exchange(state)
        started = self._profile_start()
        partition = state.partition
        working = self._working_values(state, None, require_halo=False)
        weak = self._zero_weak_buffer(state.rank, state.coefficients.dtype)
        interior = range(1, len(partition.element_local_dof_maps))
        if not partition.n_halo:
            interior = range(len(partition.element_local_dof_maps))
        self._accumulate_coordinate2_elements(
            state.rank,
            working,
            weak,
            interior,
        )
        self._profile_finish("coordinate2_local", started, count=False)

        halo_values = self._finish_state_halo_exchange(pending_halo)
        if partition.n_halo:
            started = self._profile_start()
            self._set_working_halo(state, working, halo_values)
            self._accumulate_coordinate2_elements(
                state.rank,
                working,
                weak,
                (0,),
            )
            self._profile_finish("coordinate2_local", started, count=False)

        weak_owned, outgoing = self._owned_weak_values(state.rank, weak)
        pending_weak = self._start_weak_exchange(
            state.rank,
            outgoing,
            state.coefficients.dtype,
        )
        started = self._profile_start()
        result = self._finish_coordinate2(state, weak_owned, None)
        self._profile_finish("coordinate2_local", started, count=False)
        incoming = self._finish_weak_exchange(pending_weak)
        if incoming is not None:
            started = self._profile_start()
            self._add_coordinate2_incoming(state, result, incoming)
            self._profile_finish("coordinate2_local", started, count=False)
        return DistributedTwoRadialState(self.layout, state.rank, result)

    def apply_coulomb(
        self, state: DistributedTwoRadialState
    ) -> DistributedTwoRadialState:
        """Apply point-local radial multipoles and angular couplings."""

        started = self._profile_start()
        self._validate_local_state(state)
        values = self._apply_coulomb_values(state, started=started)
        return DistributedTwoRadialState(self.layout, state.rank, values)

    def _apply_coulomb_values(
        self,
        state: DistributedTwoRadialState,
        *,
        out: np.ndarray | None = None,
        started: float | None = None,
    ) -> np.ndarray:
        if started is None:
            started = self._profile_start()
            self._validate_local_state(state)
        radial = self._local_radial_coulomb[state.rank]
        values = self.serial_hamiltonian._apply_coulomb_values(
            state.coefficients,
            radial,
            out=out,
        )
        self._profile_finish("coulomb", started)
        return values

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
        *,
        require_halo: bool = True,
    ) -> np.ndarray:
        partition = state.partition
        working = self._cached_buffer(
            self._working_buffers,
            (state.rank, state.coefficients.dtype.str),
            self.layout.local_shape(state.rank),
            state.coefficients.dtype,
        )
        working[..., partition.owned_local_indices] = state.coefficients
        if partition.n_halo and (require_halo or halo_values is not None):
            if halo_values is None and require_halo:
                raise ValueError("halo values are required for this rank")
            self._set_working_halo(state, working, halo_values)
        return working

    def _set_working_halo(
        self,
        state: DistributedTwoRadialState,
        working: np.ndarray,
        halo_values: np.ndarray | None,
    ) -> None:
        partition = state.partition
        if not partition.n_halo:
            return
        if halo_values is None:
            raise ValueError("halo values are required for this rank")
        halo_values = np.asarray(halo_values)
        expected = (*state.coefficients.shape[:-1], partition.n_halo)
        if halo_values.shape != expected:
            raise ValueError(
                f"halo values have shape {halo_values.shape}; expected {expected}"
            )
        working[..., partition.halo_local_indices] = halo_values

    def _local_coordinate2_weak(
        self,
        rank: int,
        working: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        partition = self.layout.partition(rank)
        weak = self._zero_weak_buffer(rank, working.dtype)
        self._accumulate_coordinate2_elements(
            rank,
            working,
            weak,
            range(len(partition.element_local_dof_maps)),
        )
        return self._owned_weak_values(rank, weak)

    def _zero_weak_buffer(self, rank: int, dtype: np.dtype) -> np.ndarray:
        weak = self._cached_buffer(
            self._weak_buffers,
            (rank, np.dtype(dtype).str),
            self.layout.local_shape(rank),
            dtype,
        )
        weak.fill(0)
        return weak

    def _accumulate_coordinate2_elements(
        self,
        rank: int,
        working: np.ndarray,
        weak: np.ndarray,
        offsets,
    ) -> None:
        partition = self.layout.partition(rank)
        elements = tuple(partition.element_indices)
        for offset in offsets:
            element = elements[offset]
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

    def _owned_weak_values(
        self,
        rank: int,
        weak: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        partition = self.layout.partition(rank)
        owned = self._cached_buffer(
            self._owned_weak_buffers,
            (rank, weak.dtype.str),
            self.layout.owned_shape(rank),
            weak.dtype,
        )
        np.copyto(owned, weak[..., partition.owned_local_indices])
        outgoing = None
        if partition.n_halo:
            shape = (*self.layout.owned_shape(rank)[:-1], partition.n_halo)
            outgoing = self._cached_buffer(
                self._weak_send_buffers,
                (rank, weak.dtype.str),
                shape,
                weak.dtype,
            )
            np.copyto(outgoing, weak[..., partition.halo_local_indices])
        return owned, outgoing

    def _finish_coordinate2(
        self,
        state: DistributedTwoRadialState,
        weak_owned: np.ndarray,
        incoming: tuple[np.ndarray, np.ndarray] | None,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        scale = (
            self.representation.kinetic_prefactor2
            / state.owned_R_weights
        )
        shaped_scale = scale.reshape((1,) * (weak_owned.ndim - 1) + (scale.size,))
        if out is None:
            result = weak_owned * shaped_scale
        else:
            result = out
            result += weak_owned * shaped_scale
        indices = state.partition.owned_indices
        if self.representation.uses_compact_angular_storage:
            result += self._h2_diagonal[:, None, indices] * state.coefficients
        else:
            result += (
                self._h2_diagonal[None, :, None, indices]
                * state.coefficients
            )
        if incoming is not None:
            self._add_coordinate2_incoming(state, result, incoming)
        return result

    def _add_coordinate2_incoming(
        self,
        state: DistributedTwoRadialState,
        result: np.ndarray,
        incoming: tuple[np.ndarray, np.ndarray],
    ) -> None:
        indices, values = incoming
        positions = np.searchsorted(state.partition.owned_indices, indices)
        if np.any(positions >= state.partition.n_owned) or not np.array_equal(
            state.partition.owned_indices[positions],
            indices,
        ):
            raise ValueError("received weak contribution for an unowned index")
        scale = (
            self.representation.kinetic_prefactor2
            / state.owned_R_weights[positions]
        )
        shaped_scale = scale.reshape((1,) * (values.ndim - 1) + (scale.size,))
        result[..., positions] += values * shaped_scale

    def _exchange_state_halo(
        self,
        state: DistributedTwoRadialState,
    ) -> np.ndarray | None:
        return self._finish_state_halo_exchange(
            self._start_state_halo_exchange(state)
        )

    def _start_state_halo_exchange(
        self,
        state: DistributedTwoRadialState,
    ) -> _PendingExchange | None:
        if self.layout.n_ranks == 1:
            return None
        partition = state.partition
        profile_started = self._profile_start()
        communicator = self._required_communicator(state.rank)
        requests = []
        received = None
        if partition.n_halo:
            shape = (*state.coefficients.shape[:-1], partition.n_halo)
            received = self._cached_buffer(
                self._halo_receive_buffers,
                (state.rank, state.coefficients.dtype.str),
                shape,
                state.coefficients.dtype,
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
            shape = (*state.coefficients.shape[:-1], requested.size)
            sent = self._cached_buffer(
                self._halo_send_buffers,
                (state.rank, state.coefficients.dtype.str),
                shape,
                state.coefficients.dtype,
            )
            np.copyto(sent, _owned_global_values(state, requested))
            requests.append(
                communicator.Isend(
                    sent,
                    dest=state.rank + 1,
                    tag=_STATE_HALO_TAG,
                )
            )
        setup_seconds = None
        if profile_started is not None:
            setup_seconds = perf_counter() - profile_started
        return _PendingExchange(received, None, requests, setup_seconds)

    def _finish_state_halo_exchange(
        self,
        pending: _PendingExchange | None,
    ) -> np.ndarray | None:
        if pending is None:
            return None
        wait_started = self._profile_start()
        _wait_all(pending.requests)
        self._finish_exchange_profile(
            "halo_exchange",
            pending.setup_seconds,
            wait_started,
        )
        return pending.received

    def _exchange_weak_contributions(
        self,
        rank: int,
        outgoing: np.ndarray | None,
        dtype: np.dtype,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        return self._finish_weak_exchange(
            self._start_weak_exchange(rank, outgoing, dtype)
        )

    def _start_weak_exchange(
        self,
        rank: int,
        outgoing: np.ndarray | None,
        dtype: np.dtype,
    ) -> _PendingExchange | None:
        if self.layout.n_ranks == 1:
            return None
        profile_started = self._profile_start()
        communicator = self._required_communicator(rank)
        requests = []
        received = None
        received_indices = None
        if rank + 1 < self.layout.n_ranks:
            received_indices = self.layout.partition(rank + 1).halo_indices
            shape = (*self.layout.owned_shape(rank)[:-1], received_indices.size)
            received = self._cached_buffer(
                self._weak_receive_buffers,
                (rank, np.dtype(dtype).str),
                shape,
                dtype,
            )
        if rank > 0:
            if outgoing is None:
                raise ValueError("left-neighbor weak contribution is missing")
            requests.append(
                communicator.Isend(
                    outgoing,
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
        setup_seconds = None
        if profile_started is not None:
            setup_seconds = perf_counter() - profile_started
        return _PendingExchange(
            received,
            received_indices,
            requests,
            setup_seconds,
        )

    def _finish_weak_exchange(
        self,
        pending: _PendingExchange | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if pending is None:
            return None
        wait_started = self._profile_start()
        _wait_all(pending.requests)
        self._finish_exchange_profile(
            "weak_exchange",
            pending.setup_seconds,
            wait_started,
        )
        if pending.received is None:
            return None
        return pending.received_indices, pending.received

    def _finish_exchange_profile(
        self,
        name: str,
        setup_seconds: float | None,
        wait_started: float | None,
    ) -> None:
        if wait_started is None:
            return
        self._profile_seconds[name] += (
            (0.0 if setup_seconds is None else setup_seconds)
            + perf_counter()
            - wait_started
        )
        self._profile_counts[f"{name}_calls"] += 1

    @staticmethod
    def _cached_buffer(cache, key, shape, dtype) -> np.ndarray:
        try:
            return cache[key]
        except KeyError:
            buffer = np.empty(shape, dtype=dtype)
            cache[key] = buffer
            return buffer

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
        self._quadrature_weights = tuple(
            np.multiply.outer(
                layout.representation.grid1.weights,
                layout.representation.grid2.weights[partition.owned_indices],
            )
            for partition in layout.partitions
        )
        self._weighted_right_buffers = {}
        self._collective_buffers = {}
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

        return self.local_inner_products((left,), right)[0]

    def local_inner_products(
        self,
        left: Sequence[DistributedTwoRadialState],
        right: DistributedTwoRadialState,
    ) -> np.ndarray:
        """Return several rank-local inner products without a collective."""

        left = tuple(left)
        if not left:
            raise ValueError("left must contain at least one distributed state")
        started = perf_counter() if self.profile_enabled else None
        for state in left:
            self._validate_pair(state, right)
        weighted_right = self._weighted_right(right)
        values = np.asarray(
            [
                np.vdot(state.coefficients, weighted_right)
                for state in left
            ]
        )
        if started is not None:
            self._local_inner_product_calls += len(left)
            self._local_inner_product_seconds += perf_counter() - started
        return values

    @property
    def profile(self) -> DistributedReductionProfile:
        """Return an immutable snapshot of the rank-local profile."""

        return DistributedReductionProfile(
            local_inner_product_calls=self._local_inner_product_calls,
            local_inner_product_seconds=self._local_inner_product_seconds,
            collective_calls=self._collective_calls,
            collective_values=self._collective_values,
            collective_seconds=self._collective_seconds,
        )

    def reset_profile(self) -> None:
        """Clear all accumulated rank-local reduction timings."""

        self._local_inner_product_calls = 0
        self._local_inner_product_seconds = 0.0
        self._collective_calls = 0
        self._collective_values = 0
        self._collective_seconds = 0.0

    def inner_product(
        self,
        left: DistributedTwoRadialState,
        right: DistributedTwoRadialState,
    ) -> complex:
        """Collectively return the global physical inner product."""

        local = self.local_inner_product(left, right)
        return self._allreduce(local, left.rank)

    def inner_products(
        self,
        left: Sequence[DistributedTwoRadialState],
        right: DistributedTwoRadialState,
    ) -> np.ndarray:
        """Collectively return several inner products in one reduction."""

        local = self.local_inner_products(left, right)
        return self._allreduce(local, right.rank)

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
        if self.layout.n_ranks == 1:
            return value
        if self.communicator is None:
            raise RuntimeError("a communicator is required for a global reduction")
        communicator_rank, size = _communicator_rank_size(self.communicator)
        if communicator_rank != rank or size != self.layout.n_ranks:
            raise ValueError("communicator and local distributed state do not match")
        started = perf_counter() if self.profile_enabled else None
        array = np.asarray(value)
        allreduce = getattr(self.communicator, "Allreduce", None)
        if callable(allreduce):
            scalar = array.ndim == 0
            shape = (1,) if scalar else array.shape
            key = (array.dtype.str, shape)
            try:
                send, receive = self._collective_buffers[key]
            except KeyError:
                send = np.empty(shape, dtype=array.dtype)
                receive = np.empty(shape, dtype=array.dtype)
                self._collective_buffers[key] = send, receive
            np.copyto(send, array.reshape(shape))
            allreduce(send, receive)
            result = receive[0].item() if scalar else receive.copy()
        else:
            result = self.communicator.allreduce(value)
        if started is not None:
            self._collective_calls += 1
            self._collective_values += int(array.size)
            self._collective_seconds += perf_counter() - started
        return result

    def _weighted_right(self, state: DistributedTwoRadialState) -> np.ndarray:
        weights = self._quadrature_weights[state.rank]
        dtype = np.result_type(state.coefficients.dtype, weights.dtype)
        key = (state.rank, np.dtype(dtype).str)
        try:
            buffer = self._weighted_right_buffers[key]
        except KeyError:
            buffer = np.empty(state.coefficients.shape, dtype=dtype)
            self._weighted_right_buffers[key] = buffer
        np.multiply(state.coefficients, weights, out=buffer)
        return buffer

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
