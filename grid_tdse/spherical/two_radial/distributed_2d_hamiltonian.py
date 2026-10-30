"""Hamiltonian actions on Cartesian two-radial MPI layouts."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np

from .absorbing_boundaries import TwoRadialAbsorbingBoundaries
from .distributed_2d_state import (
    DistributedTwoRadial2DState,
    TwoRadialDistributed2DLayout,
)


_STATE_HALO_TAG = {1: 28201, 2: 28202}
_WEAK_ACTION_TAG = {1: 28203, 2: 28204}


class CartesianOperatorCommunicator(Protocol):
    """Point-to-point communicator used by the Cartesian Hamiltonian."""

    rank: int
    size: int

    def Irecv(self, buffer, source: int, tag: int): ...

    def Isend(self, buffer, dest: int, tag: int): ...


@dataclass(frozen=True)
class Distributed2DHamiltonianComponents:
    """Rank-local component actions in permanent Cartesian storage."""

    coordinate1: DistributedTwoRadial2DState
    coordinate2: DistributedTwoRadial2DState
    coulomb: DistributedTwoRadial2DState
    cap: DistributedTwoRadial2DState | None = None

    @property
    def total(self) -> DistributedTwoRadial2DState:
        values = (
            self.coordinate1.coefficients
            + self.coordinate2.coefficients
            + self.coulomb.coefficients
        )
        if self.cap is not None:
            values = values + self.cap.coefficients
        return DistributedTwoRadial2DState(
            self.coordinate1.layout,
            self.coordinate1.rank,
            values,
        )


@dataclass(frozen=True)
class Distributed2DHamiltonianProfile:
    """Accumulated rank-local work and edge-exchange timings."""

    hamiltonian_calls: int = 0
    coordinate1_calls: int = 0
    coordinate1_seconds: float = 0.0
    coordinate2_calls: int = 0
    coordinate2_seconds: float = 0.0
    coulomb_calls: int = 0
    coulomb_seconds: float = 0.0
    cap_calls: int = 0
    cap_seconds: float = 0.0
    coordinate1_halo_calls: int = 0
    coordinate1_halo_seconds: float = 0.0
    coordinate2_halo_calls: int = 0
    coordinate2_halo_seconds: float = 0.0
    coordinate1_weak_calls: int = 0
    coordinate1_weak_seconds: float = 0.0
    coordinate2_weak_calls: int = 0
    coordinate2_weak_seconds: float = 0.0


@dataclass
class _PendingExchange:
    received: np.ndarray | None
    received_indices: np.ndarray | None
    requests: list
    setup_seconds: float | None


class TwoRadialDistributed2DHamiltonian:
    """Apply a two-radial Hamiltonian over a Cartesian process grid.

    Both radial coordinates are partitioned.  Each derivative exchanges only
    the bridge values on its two edge-neighbor ranks and returns the weak-form
    bridge contribution to the unique owner on the same Cartesian line.
    Point-local diagonal, Coulomb, mask, and CAP actions require no MPI
    communication.
    """

    def __init__(
        self,
        layout: TwoRadialDistributed2DLayout,
        communicator: CartesianOperatorCommunicator | None = None,
        *,
        tile_size: int = 2048,
        absorbing_boundaries: TwoRadialAbsorbingBoundaries | None = None,
        profile: bool = False,
    ) -> None:
        if not isinstance(layout, TwoRadialDistributed2DLayout):
            raise TypeError("layout must be a TwoRadialDistributed2DLayout")
        if not isinstance(profile, (bool, np.bool_)):
            raise TypeError("profile must be boolean")
        if absorbing_boundaries is not None:
            if not isinstance(absorbing_boundaries, TwoRadialAbsorbingBoundaries):
                raise TypeError(
                    "absorbing_boundaries must be a TwoRadialAbsorbingBoundaries"
                )
            if absorbing_boundaries.representation is not layout.representation:
                raise ValueError(
                    "absorbing boundaries belong to a different representation"
                )

        self.layout = layout
        self.representation = layout.representation
        self.communicator = communicator
        self.absorbing_boundaries = absorbing_boundaries
        self.profile_enabled = bool(profile)
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

        self._element_matrices = {
            1: _coordinate_element_matrices(self.representation.grid1),
            2: _coordinate_element_matrices(self.representation.grid2),
        }
        self._working_buffers = {}
        self._weak_buffers = {}
        self._owned_weak_buffers = {}
        self._halo_receive_buffers = {}
        self._halo_send_buffers = {}
        self._weak_receive_buffers = {}
        self._weak_send_buffers = {}

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

        ranks = range(layout.n_ranks) if self.rank is None else (self.rank,)
        radial_values = self.serial_hamiltonian.radial_multipoles.values
        self._local_radial_coulomb = {}
        self._local_masks = {}
        self._local_caps = {}
        for rank in ranks:
            partition = layout.partition(rank)
            radial = np.take(radial_values, partition.owned_indices1, axis=1)
            radial = np.take(radial, partition.owned_indices2, axis=2)
            self._local_radial_coulomb[rank] = np.ascontiguousarray(
                radial.reshape(radial.shape[0], -1)
            )
            self._prepare_local_absorbers(rank)
        self.reset_profile()

    def __call__(
        self,
        state: DistributedTwoRadial2DState,
    ) -> DistributedTwoRadial2DState:
        """Apply both radial terms, Coulomb interaction, and optional CAP."""

        self._validate_local_state(state)
        if self.profile_enabled:
            self._profile_counts["hamiltonian_calls"] += 1
        values = self.apply_one_body_coordinate1(state).coefficients.copy()
        values += self.apply_one_body_coordinate2(state).coefficients
        self._apply_coulomb_values(state, out=values)
        if self._local_caps[state.rank] is not None:
            values += self.cap_action(state).coefficients
        return DistributedTwoRadial2DState(self.layout, state.rank, values)

    def components(
        self,
        state: DistributedTwoRadial2DState,
    ) -> Distributed2DHamiltonianComponents:
        """Return separately evaluated Hamiltonian components."""

        cap = None
        if self._local_caps[state.rank] is not None:
            cap = self.cap_action(state)
        return Distributed2DHamiltonianComponents(
            coordinate1=self.apply_one_body_coordinate1(state),
            coordinate2=self.apply_one_body_coordinate2(state),
            coulomb=self.apply_coulomb(state),
            cap=cap,
        )

    def apply_one_body_coordinate1(
        self,
        state: DistributedTwoRadial2DState,
    ) -> DistributedTwoRadial2DState:
        """Apply the coordinate-1 derivative and diagonal one-body term."""

        return self._apply_coordinate(state, 1)

    def apply_one_body_coordinate2(
        self,
        state: DistributedTwoRadial2DState,
    ) -> DistributedTwoRadial2DState:
        """Apply the coordinate-2 derivative and diagonal one-body term."""

        return self._apply_coordinate(state, 2)

    def _apply_coordinate(
        self,
        state: DistributedTwoRadial2DState,
        coordinate: int,
    ) -> DistributedTwoRadial2DState:
        self._validate_local_state(state)
        if self.profile_enabled:
            self._profile_counts[f"coordinate{coordinate}_calls"] += 1
        pending_halo = self._start_state_halo_exchange(state, coordinate)
        partition = _coordinate_partition(state.partition, coordinate)
        started = self._profile_start()
        working = self._working_values(
            state,
            coordinate,
            None,
            require_halo=False,
        )
        weak = self._zero_weak_buffer(
            state.rank,
            coordinate,
            state.coefficients.dtype,
        )
        interior = range(1, len(partition.element_local_dof_maps))
        if not partition.n_halo:
            interior = range(len(partition.element_local_dof_maps))
        self._accumulate_elements(
            state.rank,
            coordinate,
            working,
            weak,
            interior,
        )
        self._profile_finish(f"coordinate{coordinate}", started, count=False)

        halo = self._finish_exchange(pending_halo, coordinate, "halo")
        if partition.n_halo:
            started = self._profile_start()
            self._set_working_halo(state, coordinate, working, halo)
            self._accumulate_elements(
                state.rank,
                coordinate,
                working,
                weak,
                (0,),
            )
            self._profile_finish(f"coordinate{coordinate}", started, count=False)

        weak_owned, outgoing = self._owned_weak_values(
            state.rank,
            coordinate,
            weak,
        )
        pending_weak = self._start_weak_exchange(
            state.rank,
            coordinate,
            outgoing,
            state.coefficients.dtype,
        )
        started = self._profile_start()
        result = self._finish_coordinate(state, coordinate, weak_owned)
        self._profile_finish(f"coordinate{coordinate}", started, count=False)
        incoming = self._finish_exchange(pending_weak, coordinate, "weak")
        if incoming is not None:
            started = self._profile_start()
            self._add_incoming(state, coordinate, result, incoming)
            self._profile_finish(f"coordinate{coordinate}", started, count=False)
        return DistributedTwoRadial2DState(self.layout, state.rank, result)

    def apply_coulomb(
        self,
        state: DistributedTwoRadial2DState,
    ) -> DistributedTwoRadial2DState:
        """Apply tile-local radial multipoles and angular couplings."""

        started = self._profile_start()
        self._validate_local_state(state)
        values = self._apply_coulomb_values(state, started=started)
        return DistributedTwoRadial2DState(self.layout, state.rank, values)

    def _apply_coulomb_values(
        self,
        state: DistributedTwoRadial2DState,
        *,
        out: np.ndarray | None = None,
        started: float | None = None,
    ) -> np.ndarray:
        if started is None:
            started = self._profile_start()
            self._validate_local_state(state)
        values = self.serial_hamiltonian._apply_coulomb_values(
            state.coefficients,
            self._local_radial_coulomb[state.rank],
            out=out,
        )
        self._profile_finish("coulomb", started)
        return values

    def apply_mask(
        self,
        state: DistributedTwoRadial2DState,
        time: float = 0.0,
    ) -> DistributedTwoRadial2DState:
        """Apply configured radial masks locally after an accepted step."""

        del time
        self._validate_local_state(state)
        mask = self._local_masks[state.rank]
        values = state.coefficients.copy()
        if mask is not None:
            values *= _radial_broadcast(mask, values.ndim)
        return DistributedTwoRadial2DState(self.layout, state.rank, values)

    def cap_action(
        self,
        state: DistributedTwoRadial2DState,
    ) -> DistributedTwoRadial2DState:
        """Return the tile-local complex-absorbing-potential action."""

        started = self._profile_start()
        self._validate_local_state(state)
        cap = self._local_caps[state.rank]
        if cap is None:
            values = np.zeros_like(state.coefficients)
        else:
            values = state.coefficients * _radial_broadcast(
                cap,
                state.coefficients.ndim,
            )
        self._profile_finish("cap", started)
        return DistributedTwoRadial2DState(self.layout, state.rank, values)

    def apply_all(
        self,
        states: Sequence[DistributedTwoRadial2DState],
    ) -> tuple[DistributedTwoRadial2DState, ...]:
        """Emulate all Cartesian ranks without constructing global actions."""

        states = self._rank_ordered_states(states)
        coordinate1 = self._apply_coordinate_all(states, 1)
        coordinate2 = self._apply_coordinate_all(states, 2)
        results = []
        for rank, state in enumerate(states):
            values = coordinate1[rank].coefficients.copy()
            values += coordinate2[rank].coefficients
            self._apply_coulomb_values(state, out=values)
            cap = self._local_caps[rank]
            if cap is not None:
                values += state.coefficients * _radial_broadcast(
                    cap,
                    state.coefficients.ndim,
                )
            results.append(DistributedTwoRadial2DState(self.layout, rank, values))
        return tuple(results)

    def apply_components_all(
        self,
        states: Sequence[DistributedTwoRadial2DState],
    ) -> tuple[tuple[DistributedTwoRadial2DState, ...], ...]:
        """Return rank-ordered coordinate and Coulomb component actions."""

        states = self._rank_ordered_states(states)
        coordinate1 = self._apply_coordinate_all(states, 1)
        coordinate2 = self._apply_coordinate_all(states, 2)
        coulomb = tuple(self.apply_coulomb(state) for state in states)
        return coordinate1, coordinate2, coulomb

    def _apply_coordinate_all(
        self,
        states: tuple[DistributedTwoRadial2DState, ...],
        coordinate: int,
    ) -> tuple[DistributedTwoRadial2DState, ...]:
        workings = []
        for state in states:
            lower = self._neighbor_rank(state.rank, coordinate, -1)
            partition = _coordinate_partition(state.partition, coordinate)
            halo = None
            if partition.n_halo:
                if lower is None:
                    raise RuntimeError("halo storage has no lower edge neighbor")
                halo = _owned_global_axis_values(
                    states[lower],
                    coordinate,
                    partition.halo_indices,
                )
            workings.append(self._working_values(state, coordinate, halo))

        weak_owned = []
        outgoing = []
        for rank, working in enumerate(workings):
            weak = self._zero_weak_buffer(rank, coordinate, working.dtype)
            partition = _coordinate_partition(
                self.layout.partition(rank),
                coordinate,
            )
            self._accumulate_elements(
                rank,
                coordinate,
                working,
                weak,
                range(len(partition.element_local_dof_maps)),
            )
            owned, sent = self._owned_weak_values(rank, coordinate, weak)
            weak_owned.append(owned)
            outgoing.append(sent)

        results = []
        for rank, state in enumerate(states):
            result = self._finish_coordinate(
                state,
                coordinate,
                weak_owned[rank],
            )
            upper = self._neighbor_rank(rank, coordinate, 1)
            if upper is not None:
                incoming = outgoing[upper]
                if incoming is None:
                    raise RuntimeError("upper edge weak contribution is missing")
                indices = _coordinate_partition(
                    self.layout.partition(upper),
                    coordinate,
                ).halo_indices
                self._add_incoming(
                    state,
                    coordinate,
                    result,
                    (indices, incoming),
                )
            results.append(
                DistributedTwoRadial2DState(self.layout, rank, result)
            )
        return tuple(results)

    @property
    def profile(self) -> Distributed2DHamiltonianProfile:
        """Return an immutable profiling snapshot."""

        return Distributed2DHamiltonianProfile(
            hamiltonian_calls=self._profile_counts["hamiltonian_calls"],
            coordinate1_calls=self._profile_counts["coordinate1_calls"],
            coordinate1_seconds=self._profile_seconds["coordinate1"],
            coordinate2_calls=self._profile_counts["coordinate2_calls"],
            coordinate2_seconds=self._profile_seconds["coordinate2"],
            coulomb_calls=self._profile_counts["coulomb_calls"],
            coulomb_seconds=self._profile_seconds["coulomb"],
            cap_calls=self._profile_counts["cap_calls"],
            cap_seconds=self._profile_seconds["cap"],
            coordinate1_halo_calls=self._profile_counts[
                "coordinate1_halo_calls"
            ],
            coordinate1_halo_seconds=self._profile_seconds[
                "coordinate1_halo"
            ],
            coordinate2_halo_calls=self._profile_counts[
                "coordinate2_halo_calls"
            ],
            coordinate2_halo_seconds=self._profile_seconds[
                "coordinate2_halo"
            ],
            coordinate1_weak_calls=self._profile_counts[
                "coordinate1_weak_calls"
            ],
            coordinate1_weak_seconds=self._profile_seconds[
                "coordinate1_weak"
            ],
            coordinate2_weak_calls=self._profile_counts[
                "coordinate2_weak_calls"
            ],
            coordinate2_weak_seconds=self._profile_seconds[
                "coordinate2_weak"
            ],
        )

    def reset_profile(self) -> None:
        """Clear all accumulated profiling counters and timings."""

        self._profile_counts = {
            "hamiltonian_calls": 0,
            "coordinate1_calls": 0,
            "coordinate2_calls": 0,
            "coulomb_calls": 0,
            "cap_calls": 0,
            "coordinate1_halo_calls": 0,
            "coordinate2_halo_calls": 0,
            "coordinate1_weak_calls": 0,
            "coordinate2_weak_calls": 0,
        }
        self._profile_seconds = {
            "coordinate1": 0.0,
            "coordinate2": 0.0,
            "coulomb": 0.0,
            "cap": 0.0,
            "coordinate1_halo": 0.0,
            "coordinate2_halo": 0.0,
            "coordinate1_weak": 0.0,
            "coordinate2_weak": 0.0,
        }

    def _working_values(
        self,
        state: DistributedTwoRadial2DState,
        coordinate: int,
        halo_values: np.ndarray | None,
        *,
        require_halo: bool = True,
    ) -> np.ndarray:
        axis = _coordinate_axis(coordinate)
        partition = _coordinate_partition(state.partition, coordinate)
        shape = list(state.coefficients.shape)
        shape[axis] = partition.n_local
        working = self._cached_buffer(
            self._working_buffers,
            (state.rank, coordinate, state.coefficients.dtype.str),
            tuple(shape),
            state.coefficients.dtype,
        )
        _assign_axis(
            working,
            axis,
            partition.owned_local_indices,
            state.coefficients,
        )
        if partition.n_halo and (require_halo or halo_values is not None):
            if halo_values is None:
                raise ValueError("halo values are required for this rank")
            self._set_working_halo(
                state,
                coordinate,
                working,
                halo_values,
            )
        return working

    def _set_working_halo(
        self,
        state: DistributedTwoRadial2DState,
        coordinate: int,
        working: np.ndarray,
        halo_values: np.ndarray | None,
    ) -> None:
        axis = _coordinate_axis(coordinate)
        partition = _coordinate_partition(state.partition, coordinate)
        if not partition.n_halo:
            return
        if halo_values is None:
            raise ValueError("halo values are required for this rank")
        expected = list(state.coefficients.shape)
        expected[axis] = partition.n_halo
        halo_values = np.asarray(halo_values)
        if halo_values.shape != tuple(expected):
            raise ValueError(
                f"halo values have shape {halo_values.shape}; expected "
                f"{tuple(expected)}"
            )
        _assign_axis(
            working,
            axis,
            partition.halo_local_indices,
            halo_values,
        )

    def _zero_weak_buffer(
        self,
        rank: int,
        coordinate: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        partition = _coordinate_partition(self.layout.partition(rank), coordinate)
        shape = list(self.layout.owned_shape(rank))
        shape[_coordinate_axis(coordinate)] = partition.n_local
        weak = self._cached_buffer(
            self._weak_buffers,
            (rank, coordinate, np.dtype(dtype).str),
            tuple(shape),
            dtype,
        )
        weak.fill(0)
        return weak

    def _accumulate_elements(
        self,
        rank: int,
        coordinate: int,
        working: np.ndarray,
        weak: np.ndarray,
        offsets,
    ) -> None:
        axis = _coordinate_axis(coordinate)
        partition = _coordinate_partition(self.layout.partition(rank), coordinate)
        elements = tuple(partition.element_indices)
        for offset in offsets:
            element = elements[offset]
            local_map = partition.element_local_dof_maps[offset]
            local_values = np.take(working, local_map, axis=axis)
            local_values = np.moveaxis(local_values, axis, -1)
            local_action = np.einsum(
                "ij,...j->...i",
                self._element_matrices[coordinate][element],
                local_values,
                optimize=True,
            )
            local_action = np.moveaxis(local_action, -1, axis)
            _add_axis(weak, axis, local_map, local_action)

    def _owned_weak_values(
        self,
        rank: int,
        coordinate: int,
        weak: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        axis = _coordinate_axis(coordinate)
        partition = _coordinate_partition(self.layout.partition(rank), coordinate)
        owned = self._cached_buffer(
            self._owned_weak_buffers,
            (rank, coordinate, weak.dtype.str),
            self.layout.owned_shape(rank),
            weak.dtype,
        )
        np.copyto(owned, np.take(weak, partition.owned_local_indices, axis=axis))
        outgoing = None
        if partition.n_halo:
            shape = list(self.layout.owned_shape(rank))
            shape[axis] = partition.n_halo
            outgoing = self._cached_buffer(
                self._weak_send_buffers,
                (rank, coordinate, weak.dtype.str),
                tuple(shape),
                weak.dtype,
            )
            np.copyto(
                outgoing,
                np.take(weak, partition.halo_local_indices, axis=axis),
            )
        return owned, outgoing

    def _finish_coordinate(
        self,
        state: DistributedTwoRadial2DState,
        coordinate: int,
        weak_owned: np.ndarray,
    ) -> np.ndarray:
        axis = _coordinate_axis(coordinate)
        partition = _coordinate_partition(state.partition, coordinate)
        weights = (
            self.representation.grid1.weights[partition.owned_indices]
            if coordinate == 1
            else self.representation.grid2.weights[partition.owned_indices]
        )
        prefactor = (
            self.representation.kinetic_prefactor1
            if coordinate == 1
            else self.representation.kinetic_prefactor2
        )
        result = weak_owned * _reshape_for_axis(
            prefactor / weights,
            weak_owned.ndim,
            axis,
        )
        indices = partition.owned_indices
        if coordinate == 1:
            diagonal = (
                self._h1_diagonal[:, indices, None]
                if self.representation.uses_compact_angular_storage
                else self._h1_diagonal[:, None, indices, None]
            )
        else:
            diagonal = (
                self._h2_diagonal[:, None, indices]
                if self.representation.uses_compact_angular_storage
                else self._h2_diagonal[None, :, None, indices]
            )
        result += diagonal * state.coefficients
        return result

    def _add_incoming(
        self,
        state: DistributedTwoRadial2DState,
        coordinate: int,
        result: np.ndarray,
        incoming: tuple[np.ndarray, np.ndarray],
    ) -> None:
        axis = _coordinate_axis(coordinate)
        partition = _coordinate_partition(state.partition, coordinate)
        indices, values = incoming
        positions = np.searchsorted(partition.owned_indices, indices)
        if np.any(positions >= partition.n_owned) or not np.array_equal(
            partition.owned_indices[positions],
            indices,
        ):
            raise ValueError("received weak contribution for an unowned index")
        weights = (
            self.representation.grid1.weights[partition.owned_indices[positions]]
            if coordinate == 1
            else self.representation.grid2.weights[partition.owned_indices[positions]]
        )
        prefactor = (
            self.representation.kinetic_prefactor1
            if coordinate == 1
            else self.representation.kinetic_prefactor2
        )
        _add_axis(
            result,
            axis,
            positions,
            values * _reshape_for_axis(prefactor / weights, values.ndim, axis),
        )

    def _start_state_halo_exchange(
        self,
        state: DistributedTwoRadial2DState,
        coordinate: int,
    ) -> _PendingExchange | None:
        if self.layout.process_grid[coordinate - 1] == 1:
            return None
        profile_started = self._profile_start()
        communicator = self._required_communicator(state.rank)
        partition = _coordinate_partition(state.partition, coordinate)
        axis = _coordinate_axis(coordinate)
        requests = []
        received = None
        lower = self._neighbor_rank(state.rank, coordinate, -1)
        upper = self._neighbor_rank(state.rank, coordinate, 1)
        if partition.n_halo:
            if lower is None:
                raise RuntimeError("halo storage has no lower edge neighbor")
            shape = list(state.coefficients.shape)
            shape[axis] = partition.n_halo
            received = self._cached_buffer(
                self._halo_receive_buffers,
                (state.rank, coordinate, state.coefficients.dtype.str),
                tuple(shape),
                state.coefficients.dtype,
            )
            requests.append(
                communicator.Irecv(
                    received,
                    source=lower,
                    tag=_STATE_HALO_TAG[coordinate],
                )
            )
        if upper is not None:
            requested = _coordinate_partition(
                self.layout.partition(upper),
                coordinate,
            ).halo_indices
            values = _owned_global_axis_values(state, coordinate, requested)
            sent = self._cached_buffer(
                self._halo_send_buffers,
                (state.rank, coordinate, state.coefficients.dtype.str),
                values.shape,
                state.coefficients.dtype,
            )
            np.copyto(sent, values)
            requests.append(
                communicator.Isend(
                    sent,
                    dest=upper,
                    tag=_STATE_HALO_TAG[coordinate],
                )
            )
        setup_seconds = None
        if profile_started is not None:
            setup_seconds = perf_counter() - profile_started
        return _PendingExchange(received, None, requests, setup_seconds)

    def _start_weak_exchange(
        self,
        rank: int,
        coordinate: int,
        outgoing: np.ndarray | None,
        dtype: np.dtype,
    ) -> _PendingExchange | None:
        if self.layout.process_grid[coordinate - 1] == 1:
            return None
        profile_started = self._profile_start()
        communicator = self._required_communicator(rank)
        lower = self._neighbor_rank(rank, coordinate, -1)
        upper = self._neighbor_rank(rank, coordinate, 1)
        axis = _coordinate_axis(coordinate)
        requests = []
        received = None
        received_indices = None
        if upper is not None:
            received_indices = _coordinate_partition(
                self.layout.partition(upper),
                coordinate,
            ).halo_indices
            shape = list(self.layout.owned_shape(rank))
            shape[axis] = received_indices.size
            received = self._cached_buffer(
                self._weak_receive_buffers,
                (rank, coordinate, np.dtype(dtype).str),
                tuple(shape),
                dtype,
            )
            requests.append(
                communicator.Irecv(
                    received,
                    source=upper,
                    tag=_WEAK_ACTION_TAG[coordinate],
                )
            )
        if lower is not None:
            if outgoing is None:
                raise ValueError("lower-neighbor weak contribution is missing")
            requests.append(
                communicator.Isend(
                    outgoing,
                    dest=lower,
                    tag=_WEAK_ACTION_TAG[coordinate],
                )
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

    def _finish_exchange(
        self,
        pending: _PendingExchange | None,
        coordinate: int,
        exchange: str,
    ):
        if pending is None:
            return None
        wait_started = self._profile_start()
        _wait_all(pending.requests)
        if wait_started is not None:
            name = f"coordinate{coordinate}_{exchange}"
            self._profile_seconds[name] += (
                (0.0 if pending.setup_seconds is None else pending.setup_seconds)
                + perf_counter()
                - wait_started
            )
            self._profile_counts[f"{name}_calls"] += 1
        if exchange == "halo":
            return pending.received
        if pending.received is None:
            return None
        return pending.received_indices, pending.received

    def _neighbor_rank(
        self,
        rank: int,
        coordinate: int,
        direction: int,
    ) -> int | None:
        rank1, rank2 = self.layout.process_coordinates(rank)
        coordinates = [rank1, rank2]
        coordinates[coordinate - 1] += direction
        if not 0 <= coordinates[coordinate - 1] < self.layout.process_grid[
            coordinate - 1
        ]:
            return None
        return self.layout.rank_at(tuple(coordinates))

    def _prepare_local_absorbers(self, rank: int) -> None:
        boundaries = self.absorbing_boundaries
        if boundaries is None:
            self._local_masks[rank] = None
            self._local_caps[rank] = None
            return
        partition = self.layout.partition(rank)
        mask = np.ones((partition.coordinate1.n_owned, partition.coordinate2.n_owned))
        has_mask = False
        if boundaries.mask1 is not None:
            mask *= boundaries.mask1[partition.owned_indices1, None]
            has_mask = True
        if boundaries.mask2 is not None:
            mask *= boundaries.mask2[None, partition.owned_indices2]
            has_mask = True
        cap = np.zeros(mask.shape, dtype=complex)
        has_cap = False
        if boundaries.cap1 is not None:
            cap += boundaries.cap1[partition.owned_indices1, None]
            has_cap = True
        if boundaries.cap2 is not None:
            cap += boundaries.cap2[None, partition.owned_indices2]
            has_cap = True
        self._local_masks[rank] = np.ascontiguousarray(mask) if has_mask else None
        self._local_caps[rank] = np.ascontiguousarray(cap) if has_cap else None

    def _required_communicator(self, rank: int):
        if self.communicator is None:
            raise RuntimeError(
                "a communicator is required for a multi-rank Hamiltonian action"
            )
        communicator_rank, size = _communicator_rank_size(self.communicator)
        if size != self.layout.n_ranks or communicator_rank != rank:
            raise ValueError("communicator and local distributed state do not match")
        return self.communicator

    def _validate_local_state(self, state: DistributedTwoRadial2DState) -> None:
        if not isinstance(state, DistributedTwoRadial2DState):
            raise TypeError("state must be a DistributedTwoRadial2DState")
        if state.layout.layout_id != self.layout.layout_id:
            raise ValueError("state belongs to a different distributed layout")
        if self.rank is not None and state.rank != self.rank:
            raise ValueError(
                f"state belongs to rank {state.rank}; Hamiltonian belongs to "
                f"rank {self.rank}"
            )

    def _rank_ordered_states(
        self,
        states: Sequence[DistributedTwoRadial2DState],
    ) -> tuple[DistributedTwoRadial2DState, ...]:
        if len(states) != self.layout.n_ranks:
            raise ValueError(
                f"received {len(states)} local states; expected "
                f"{self.layout.n_ranks}"
            )
        ordered = [None] * self.layout.n_ranks
        for state in states:
            if not isinstance(state, DistributedTwoRadial2DState):
                raise TypeError(
                    "states must contain DistributedTwoRadial2DState objects"
                )
            if state.layout.layout_id != self.layout.layout_id:
                raise ValueError("state belongs to a different distributed layout")
            if ordered[state.rank] is not None:
                raise ValueError(f"duplicate state for rank {state.rank}")
            ordered[state.rank] = state
        if any(state is None for state in ordered):
            raise ValueError("one state is required for every rank")
        return tuple(ordered)

    @staticmethod
    def _cached_buffer(cache, key, shape, dtype) -> np.ndarray:
        try:
            buffer = cache[key]
        except KeyError:
            buffer = np.empty(shape, dtype=dtype)
            cache[key] = buffer
        return buffer

    def _profile_start(self) -> float | None:
        return perf_counter() if self.profile_enabled else None

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


def _coordinate_partition(partition, coordinate: int):
    return partition.coordinate1 if coordinate == 1 else partition.coordinate2


def _coordinate_axis(coordinate: int) -> int:
    return -2 if coordinate == 1 else -1


def _coordinate_element_matrices(grid) -> tuple[np.ndarray, ...]:
    dvr = grid.dvr
    matrices = []
    for full_map, stiffness in zip(
        dvr.element_dof_maps,
        dvr.local_stiffness_matrices,
        strict=True,
    ):
        retained = (full_map > 0) & (full_map < dvr.r.size - 1)
        matrices.append(
            np.ascontiguousarray(-stiffness[np.ix_(retained, retained)])
        )
    return tuple(matrices)


def _owned_global_axis_values(
    state: DistributedTwoRadial2DState,
    coordinate: int,
    global_indices: np.ndarray,
) -> np.ndarray:
    partition = _coordinate_partition(state.partition, coordinate)
    positions = np.searchsorted(partition.owned_indices, global_indices)
    if np.any(positions >= partition.n_owned) or not np.array_equal(
        partition.owned_indices[positions],
        global_indices,
    ):
        raise ValueError("requested halo value is not owned by the source rank")
    return np.ascontiguousarray(
        np.take(state.coefficients, positions, axis=_coordinate_axis(coordinate))
    )


def _reshape_for_axis(values: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = values.size
    return values.reshape(shape)


def _assign_axis(
    destination: np.ndarray,
    axis: int,
    indices: np.ndarray,
    values: np.ndarray,
) -> None:
    selection = [slice(None)] * destination.ndim
    selection[axis] = indices
    destination[tuple(selection)] = values


def _add_axis(
    destination: np.ndarray,
    axis: int,
    indices: np.ndarray,
    values: np.ndarray,
) -> None:
    selection = [slice(None)] * destination.ndim
    selection[axis] = indices
    destination[tuple(selection)] += values


def _radial_broadcast(values: np.ndarray, ndim: int) -> np.ndarray:
    return values.reshape((1,) * (ndim - 2) + values.shape)


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
