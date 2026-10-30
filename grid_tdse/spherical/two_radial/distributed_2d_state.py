"""Cartesian two-coordinate state layouts for two-radial FEM-DVR systems.

This module complements the production one-coordinate decomposition in
``distributed_state``.  A ``P_r x P_R`` process grid partitions both radial
coordinates while retaining every angular channel on each rank.  The module
contains no direct MPI dependency; collective helpers accept mpi4py-compatible
communicators or lightweight test doubles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .distributed_state import (
    TwoRadialDistributedLayout,
    TwoRadialRankLayout,
)
from .state_archive import (
    TwoRadialState,
    load_two_radial_state,
    save_two_radial_state,
)
from .system import TwoRadialSystem


class CartesianCollectiveCommunicator(Protocol):
    """Object-collective communicator interface used by this module."""

    rank: int
    size: int

    def scatter(self, values, root: int = 0): ...

    def gather(self, value, root: int = 0): ...

    def bcast(self, value, root: int = 0): ...


@dataclass(frozen=True, eq=False)
class TwoRadial2DRankLayout:
    """Cartesian element-tile and radial-DOF ownership for one rank."""

    rank: int
    process_coordinates: tuple[int, int]
    coordinate1: TwoRadialRankLayout
    coordinate2: TwoRadialRankLayout
    coordinate1_neighbor_ranks: tuple[int, ...]
    coordinate2_neighbor_ranks: tuple[int, ...]
    halo_owner_ranks1: np.ndarray
    halo_owner_ranks2: np.ndarray

    def __post_init__(self) -> None:
        for name in ("halo_owner_ranks1", "halo_owner_ranks2"):
            values = np.ascontiguousarray(getattr(self, name), dtype=int)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def neighbor_ranks(self) -> tuple[int, ...]:
        """Unique edge-neighbor ranks in coordinate order."""

        return tuple(
            dict.fromkeys(
                self.coordinate1_neighbor_ranks
                + self.coordinate2_neighbor_ranks
            )
        )

    @property
    def element_range1(self) -> range:
        return self.coordinate1.element_indices

    @property
    def element_range2(self) -> range:
        return self.coordinate2.element_indices

    @property
    def element_tiles(self) -> tuple[tuple[int, int], ...]:
        """Connected rectangular group of FEM-DVR element tiles."""

        return tuple(
            (element1, element2)
            for element1 in self.element_range1
            for element2 in self.element_range2
        )

    @property
    def owned_indices1(self) -> np.ndarray:
        return self.coordinate1.owned_indices

    @property
    def owned_indices2(self) -> np.ndarray:
        return self.coordinate2.owned_indices

    @property
    def local_indices1(self) -> np.ndarray:
        return self.coordinate1.local_indices

    @property
    def local_indices2(self) -> np.ndarray:
        return self.coordinate2.local_indices

    @property
    def halo_indices1(self) -> np.ndarray:
        return self.coordinate1.halo_indices

    @property
    def halo_indices2(self) -> np.ndarray:
        return self.coordinate2.halo_indices

    @property
    def owned_bridge_indices1(self) -> np.ndarray:
        return self.coordinate1.owned_bridge_indices

    @property
    def owned_bridge_indices2(self) -> np.ndarray:
        return self.coordinate2.owned_bridge_indices

    @property
    def n_owned(self) -> int:
        """Number of permanently owned radial coefficient pairs."""

        return self.coordinate1.n_owned * self.coordinate2.n_owned

    @property
    def n_local(self) -> int:
        """Size of the rectangular owned-plus-halo radial work area."""

        return self.coordinate1.n_local * self.coordinate2.n_local

    @property
    def global_owned_slices(self) -> tuple[slice, slice]:
        return (
            self.coordinate1.global_owned_slice,
            self.coordinate2.global_owned_slice,
        )

    @property
    def global_local_slices(self) -> tuple[slice, slice]:
        return (
            self.coordinate1.global_local_slice,
            self.coordinate2.global_local_slice,
        )


@dataclass(frozen=True, eq=False)
class TwoRadialDistributed2DLayout:
    """Permanent Cartesian decomposition of both radial coordinates.

    Ranks are ordered with coordinate 2 varying fastest:
    ``rank = rank1 * P_R + rank2``.  If ``process_grid`` is omitted, all valid
    factor pairs of ``n_ranks`` are evaluated with a lightweight estimate of
    derivative and Coulomb work, and the least expensive maximum-rank layout
    is selected.
    """

    representation: TwoRadialSystem
    n_ranks: int
    process_grid: tuple[int, int] | None = None
    coordinate1_layout: TwoRadialDistributedLayout = field(
        init=False,
        repr=False,
    )
    coordinate2_layout: TwoRadialDistributedLayout = field(
        init=False,
        repr=False,
    )
    partitions: tuple[TwoRadial2DRankLayout, ...] = field(
        init=False,
        repr=False,
    )
    estimated_rank_costs: np.ndarray = field(init=False, repr=False)
    representation_id: str = field(init=False)
    layout_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.representation, TwoRadialSystem):
            raise TypeError("representation must be a TwoRadialSystem")
        n_ranks = _positive_integer(self.n_ranks, name="n_ranks")
        if self.representation.grid1.family != "femdvr":
            raise ValueError("two-dimensional distribution requires FEM-DVR grid1")
        if self.representation.grid2.family != "femdvr":
            raise ValueError("two-dimensional distribution requires FEM-DVR grid2")

        process_grid = self.process_grid
        if process_grid is None:
            process_grid = _select_process_grid(self.representation, n_ranks)
        else:
            process_grid = _validated_process_grid(
                process_grid,
                n_ranks=n_ranks,
                n_elements1=int(self.representation.grid1.dvr.n_intervals),
                n_elements2=int(self.representation.grid2.dvr.n_intervals),
            )

        layout1 = TwoRadialDistributedLayout(
            self.representation,
            process_grid[0],
            partitioned_coordinate=1,
        )
        layout2 = TwoRadialDistributedLayout(
            self.representation,
            process_grid[1],
            partitioned_coordinate=2,
        )
        partitions = tuple(
            _cartesian_rank_layout(
                rank1,
                rank2,
                layout1,
                layout2,
                process_grid,
            )
            for rank1 in range(process_grid[0])
            for rank2 in range(process_grid[1])
        )
        costs = _estimated_costs(self.representation, layout1, layout2)
        costs.setflags(write=False)
        representation_id = _representation_id(self.representation)

        object.__setattr__(self, "n_ranks", n_ranks)
        object.__setattr__(self, "process_grid", process_grid)
        object.__setattr__(self, "coordinate1_layout", layout1)
        object.__setattr__(self, "coordinate2_layout", layout2)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "estimated_rank_costs", costs)
        object.__setattr__(self, "representation_id", representation_id)
        object.__setattr__(
            self,
            "layout_id",
            _cartesian_layout_id(
                representation_id,
                process_grid,
                layout1,
                layout2,
            ),
        )

    @property
    def global_shape(self) -> tuple[int, ...]:
        return self.representation.psi_shape

    @property
    def radial_axes(self) -> tuple[int, int]:
        return len(self.global_shape) - 2, len(self.global_shape) - 1

    @property
    def archive_metadata(self) -> dict[str, Any]:
        return {
            "decomposition": "cartesian-2d",
            "process_grid": self.process_grid,
            "n_ranks": self.n_ranks,
            "layout_id": self.layout_id,
        }

    def rank_at(self, process_coordinates: tuple[int, int]) -> int:
        """Return the row-major rank at ``(rank1, rank2)``."""

        rank1, rank2 = _validated_process_coordinates(
            process_coordinates,
            self.process_grid,
        )
        return rank1 * self.process_grid[1] + rank2

    def process_coordinates(self, rank: int) -> tuple[int, int]:
        """Return Cartesian coordinates for a validated rank."""

        rank = _validated_rank(rank, self.n_ranks)
        return divmod(rank, self.process_grid[1])

    def coordinate1_line_ranks(self, rank: int) -> tuple[int, ...]:
        """Ranks sharing coordinate-2 process position with ``rank``."""

        _, rank2 = self.process_coordinates(rank)
        return tuple(
            rank1 * self.process_grid[1] + rank2
            for rank1 in range(self.process_grid[0])
        )

    def coordinate2_line_ranks(self, rank: int) -> tuple[int, ...]:
        """Ranks sharing coordinate-1 process position with ``rank``."""

        rank1, _ = self.process_coordinates(rank)
        return tuple(
            rank1 * self.process_grid[1] + rank2
            for rank2 in range(self.process_grid[1])
        )

    def partition(self, rank: int) -> TwoRadial2DRankLayout:
        return self.partitions[_validated_rank(rank, self.n_ranks)]

    def owned_shape(self, rank: int) -> tuple[int, ...]:
        partition = self.partition(rank)
        return (
            *self.global_shape[:-2],
            partition.coordinate1.n_owned,
            partition.coordinate2.n_owned,
        )

    def local_shape(self, rank: int) -> tuple[int, ...]:
        partition = self.partition(rank)
        return (
            *self.global_shape[:-2],
            partition.coordinate1.n_local,
            partition.coordinate2.n_local,
        )

    def extract_owned(self, state: np.ndarray, rank: int) -> np.ndarray:
        """Copy one rank's permanently owned rectangular coefficient tile."""

        values = self._global_coefficients(state)
        partition = self.partition(rank)
        result = np.take(values, partition.owned_indices1, axis=-2)
        result = np.take(result, partition.owned_indices2, axis=-1)
        return np.ascontiguousarray(result)

    def distribute(
        self,
        state: np.ndarray | TwoRadialState,
    ) -> tuple["DistributedTwoRadial2DState", ...]:
        values = self._global_coefficients(state)
        return tuple(
            DistributedTwoRadial2DState(
                self,
                rank,
                self.extract_owned(values, rank),
            )
            for rank in range(self.n_ranks)
        )

    def assemble(
        self,
        states: Sequence["DistributedTwoRadial2DState"],
    ) -> np.ndarray:
        if len(states) != self.n_ranks:
            raise ValueError(
                f"received {len(states)} local states; expected {self.n_ranks}"
            )
        by_rank: list[DistributedTwoRadial2DState | None] = [None] * self.n_ranks
        for state in states:
            if not isinstance(state, DistributedTwoRadial2DState):
                raise TypeError(
                    "states must contain DistributedTwoRadial2DState objects"
                )
            if state.layout.layout_id != self.layout_id:
                raise ValueError("local state belongs to a different 2D layout")
            if by_rank[state.rank] is not None:
                raise ValueError(
                    f"received duplicate local state for rank {state.rank}"
                )
            by_rank[state.rank] = state
        if any(state is None for state in by_rank):
            raise ValueError("one local state is required for every rank")
        return self.assemble_coefficients(
            [state.coefficients for state in by_rank if state is not None]
        )

    def assemble_coefficients(self, values: Sequence[np.ndarray]) -> np.ndarray:
        if len(values) != self.n_ranks:
            raise ValueError(
                f"received {len(values)} local arrays; expected {self.n_ranks}"
            )
        dtype = np.result_type(*(np.asarray(value).dtype for value in values))
        result = np.empty(self.global_shape, dtype=dtype)
        for rank, (value, partition) in enumerate(zip(values, self.partitions)):
            value = np.asarray(value)
            expected = self.owned_shape(rank)
            if value.shape != expected:
                raise ValueError(
                    f"rank {rank} state has shape {value.shape}; expected {expected}"
                )
            _assign_radial_tile(
                result,
                partition.owned_indices1,
                partition.owned_indices2,
                value,
            )
        return result

    def redistribute(
        self,
        states: Sequence["DistributedTwoRadial2DState"],
        target: "TwoRadialDistributed2DLayout",
    ) -> tuple["DistributedTwoRadial2DState", ...]:
        """Exactly redistribute in-memory states to another process grid."""

        if not isinstance(target, TwoRadialDistributed2DLayout):
            raise TypeError("target must be a TwoRadialDistributed2DLayout")
        if target.representation_id != self.representation_id:
            raise ValueError("target uses an incompatible two-radial representation")
        return target.distribute(self.assemble(states))

    def _global_coefficients(
        self,
        state: np.ndarray | TwoRadialState,
    ) -> np.ndarray:
        if isinstance(state, TwoRadialState):
            values = state.state_for(self.representation)
        else:
            values = np.asarray(state)
        if values.shape != self.global_shape:
            raise ValueError(
                f"state has shape {values.shape}; expected {self.global_shape}"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("state must contain numeric values")
        if not np.isfinite(values).all():
            raise ValueError("state must contain only finite values")
        return values


@dataclass(frozen=True, eq=False)
class DistributedTwoRadial2DState:
    """Permanently owned coefficients for one Cartesian rank."""

    layout: TwoRadialDistributed2DLayout
    rank: int
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.layout, TwoRadialDistributed2DLayout):
            raise TypeError("layout must be a TwoRadialDistributed2DLayout")
        rank = _validated_rank(self.rank, self.layout.n_ranks)
        coefficients = np.asarray(self.coefficients)
        expected = self.layout.owned_shape(rank)
        if coefficients.shape != expected:
            raise ValueError(
                f"rank {rank} coefficients have shape {coefficients.shape}; "
                f"expected {expected}"
            )
        if not np.issubdtype(coefficients.dtype, np.number):
            raise TypeError("coefficients must contain numeric values")
        if not np.isfinite(coefficients).all():
            raise ValueError("coefficients must contain only finite values")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "coefficients", np.ascontiguousarray(coefficients))

    @property
    def partition(self) -> TwoRadial2DRankLayout:
        return self.layout.partition(self.rank)

    @property
    def process_coordinates(self) -> tuple[int, int]:
        return self.partition.process_coordinates

    @property
    def owned_r(self) -> np.ndarray:
        return self.layout.representation.grid1.r[self.partition.owned_indices1]

    @property
    def owned_R(self) -> np.ndarray:
        return self.layout.representation.grid2.r[self.partition.owned_indices2]

    @property
    def owned_r_weights(self) -> np.ndarray:
        return self.layout.representation.grid1.weights[
            self.partition.owned_indices1
        ]

    @property
    def owned_R_weights(self) -> np.ndarray:
        return self.layout.representation.grid2.weights[
            self.partition.owned_indices2
        ]

    def copy(self) -> "DistributedTwoRadial2DState":
        return DistributedTwoRadial2DState(
            self.layout,
            self.rank,
            self.coefficients.copy(),
        )


def scatter_two_radial_2d_state(
    layout: TwoRadialDistributed2DLayout,
    communicator: CartesianCollectiveCommunicator,
    state: np.ndarray | TwoRadialState | None = None,
    *,
    root: int = 0,
) -> DistributedTwoRadial2DState:
    """Collectively scatter a global state over a Cartesian process grid."""

    rank, size = _communicator_rank_size(communicator)
    _validate_collective(layout, size, root)
    payload = None
    original_error: Exception | None = None
    error = None
    if rank == root:
        try:
            if state is None:
                raise ValueError("root must provide a state to scatter")
            payload = [part.coefficients for part in layout.distribute(state)]
        except Exception as exc:  # synchronize setup failures across ranks
            original_error = exc
            error = (type(exc).__name__, str(exc))
    error = communicator.bcast(error, root=root)
    if error is not None:
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"rank {root} could not prepare state scatter: {error[1]}")
    coefficients = communicator.scatter(payload, root=root)
    return DistributedTwoRadial2DState(layout, rank, coefficients)


def gather_two_radial_2d_state(
    state: DistributedTwoRadial2DState,
    communicator: CartesianCollectiveCommunicator,
    *,
    root: int = 0,
) -> np.ndarray | None:
    """Collectively gather Cartesian state storage onto ``root``."""

    if not isinstance(state, DistributedTwoRadial2DState):
        raise TypeError("state must be a DistributedTwoRadial2DState")
    rank, size = _communicator_rank_size(communicator)
    _validate_collective(state.layout, size, root)
    if state.rank != rank:
        raise ValueError(
            f"local state belongs to rank {state.rank}, communicator rank is {rank}"
        )
    gathered = communicator.gather(state.coefficients, root=root)
    if rank != root:
        return None
    return state.layout.assemble_coefficients(gathered)


def redistribute_two_radial_2d_state(
    state: DistributedTwoRadial2DState,
    target: TwoRadialDistributed2DLayout,
    communicator: CartesianCollectiveCommunicator,
    *,
    root: int = 0,
) -> DistributedTwoRadial2DState:
    """Collectively redistribute a state to another same-size process grid."""

    if not isinstance(state, DistributedTwoRadial2DState):
        raise TypeError("state must be a DistributedTwoRadial2DState")
    if not isinstance(target, TwoRadialDistributed2DLayout):
        raise TypeError("target must be a TwoRadialDistributed2DLayout")
    if target.representation_id != state.layout.representation_id:
        raise ValueError("target uses an incompatible two-radial representation")
    if target.n_ranks != state.layout.n_ranks:
        raise ValueError("collective redistribution requires the same rank count")
    rank, _ = _communicator_rank_size(communicator)
    global_state = gather_two_radial_2d_state(state, communicator, root=root)
    return scatter_two_radial_2d_state(
        target,
        communicator,
        global_state if rank == root else None,
        root=root,
    )


def load_distributed_two_radial_2d_state(
    path: str | Path,
    layout: TwoRadialDistributed2DLayout,
    communicator: CartesianCollectiveCommunicator,
    *,
    root: int = 0,
) -> DistributedTwoRadial2DState:
    """Load the existing serial archive on ``root`` and scatter it."""

    rank, size = _communicator_rank_size(communicator)
    _validate_collective(layout, size, root)
    archived = None
    original_error: Exception | None = None
    error = None
    if rank == root:
        try:
            archived = load_two_radial_state(path)
            archived.validate_representation(layout.representation)
        except Exception as exc:
            original_error = exc
            error = (type(exc).__name__, str(exc))
    error = communicator.bcast(error, root=root)
    if error is not None:
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"rank {root} could not load state archive: {error[1]}")
    return scatter_two_radial_2d_state(
        layout,
        communicator,
        archived if rank == root else None,
        root=root,
    )


def save_distributed_two_radial_2d_state(
    path: str | Path,
    state: DistributedTwoRadial2DState,
    communicator: CartesianCollectiveCommunicator,
    *,
    physical_state: Mapping[str, Any],
    calculation: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    root: int = 0,
) -> Path | None:
    """Gather a small Cartesian state and save the serial archive format."""

    rank, _ = _communicator_rank_size(communicator)
    coefficients = gather_two_radial_2d_state(state, communicator, root=root)
    destination = None
    original_error: Exception | None = None
    error = None
    if rank == root:
        try:
            archive_calculation = {} if calculation is None else dict(calculation)
            archive_calculation["distributed_layout"] = state.layout.archive_metadata
            destination = save_two_radial_state(
                path,
                coefficients,
                state.layout.representation,
                physical_state=physical_state,
                calculation=archive_calculation,
                overwrite=overwrite,
            )
        except Exception as exc:
            original_error = exc
            error = (type(exc).__name__, str(exc))
    error = communicator.bcast(error, root=root)
    if error is not None:
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"rank {root} could not save state archive: {error[1]}")
    return destination if rank == root else None


def phase_align_two_radial_state(
    reference: np.ndarray,
    candidate: np.ndarray,
    inner_product: Callable[[np.ndarray, np.ndarray], complex] | None = None,
) -> np.ndarray:
    """Return ``candidate`` aligned to the global phase of ``reference``."""

    reference, candidate = _validated_state_pair(reference, candidate)
    product = np.vdot if inner_product is None else inner_product
    overlap = complex(product(reference, candidate))
    if abs(overlap) == 0.0:
        return candidate.copy()
    return np.ascontiguousarray(candidate * overlap.conjugate() / abs(overlap))


def phase_aligned_two_radial_state_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
    inner_product: Callable[[np.ndarray, np.ndarray], complex] | None = None,
) -> float:
    """Return the norm of the global-phase-aligned state difference."""

    reference, candidate = _validated_state_pair(reference, candidate)
    product = np.vdot if inner_product is None else inner_product
    aligned = phase_align_two_radial_state(reference, candidate, product)
    difference = aligned - reference
    norm_squared = float(complex(product(difference, difference)).real)
    tolerance = 100.0 * np.finfo(float).eps * max(
        1.0,
        float(abs(complex(product(reference, reference)))),
    )
    if norm_squared < -tolerance:
        raise ValueError("inner product produced a negative squared norm")
    return float(np.sqrt(max(0.0, norm_squared)))


def _select_process_grid(
    representation: TwoRadialSystem,
    n_ranks: int,
) -> tuple[int, int]:
    n_elements1 = int(representation.grid1.dvr.n_intervals)
    n_elements2 = int(representation.grid2.dvr.n_intervals)
    candidates = [
        (parts1, n_ranks // parts1)
        for parts1 in range(1, n_ranks + 1)
        if n_ranks % parts1 == 0
        and parts1 <= n_elements1
        and n_ranks // parts1 <= n_elements2
    ]
    if not candidates:
        raise ValueError(
            f"n_ranks={n_ranks} cannot form a process grid within "
            f"{n_elements1} x {n_elements2} FEM-DVR elements"
        )

    def score(process_grid: tuple[int, int]) -> tuple[float, float, float, int]:
        layout1 = TwoRadialDistributedLayout(
            representation,
            process_grid[0],
            partitioned_coordinate=1,
        )
        layout2 = TwoRadialDistributedLayout(
            representation,
            process_grid[1],
            partitioned_coordinate=2,
        )
        costs = _estimated_costs(representation, layout1, layout2)
        communication = max(
            part1.n_halo * part2.n_owned + part1.n_owned * part2.n_halo
            for part1 in layout1.partitions
            for part2 in layout2.partitions
        )
        owned = max(
            part1.n_owned * part2.n_owned
            for part1 in layout1.partitions
            for part2 in layout2.partitions
        )
        return (
            float(np.max(costs)),
            float(communication),
            float(owned),
            abs(process_grid[0] - process_grid[1]),
        )

    return min(candidates, key=score)


def _estimated_costs(
    representation: TwoRadialSystem,
    layout1: TwoRadialDistributedLayout,
    layout2: TwoRadialDistributedLayout,
) -> np.ndarray:
    angular_blocks = int(representation.angular_pairs.n_pairs)
    coulomb_terms = max(angular_blocks, int(representation.couplings.n_pairs))
    costs = []
    for part1 in layout1.partitions:
        for part2 in layout2.partitions:
            derivative_work = angular_blocks * (
                part1.n_local * part2.n_owned
                + part1.n_owned * part2.n_local
            )
            coulomb_work = coulomb_terms * part1.n_owned * part2.n_owned
            costs.append(float(derivative_work + coulomb_work))
    return np.asarray(costs, dtype=float)


def _validated_process_grid(
    process_grid,
    *,
    n_ranks: int,
    n_elements1: int,
    n_elements2: int,
) -> tuple[int, int]:
    if isinstance(process_grid, (str, bytes)):
        raise TypeError("process_grid must contain two positive integers")
    try:
        values = tuple(process_grid)
    except TypeError as exc:
        raise TypeError("process_grid must contain two positive integers") from exc
    if len(values) != 2:
        raise ValueError("process_grid must contain exactly two entries")
    parts1 = _positive_integer(values[0], name="process_grid[0]")
    parts2 = _positive_integer(values[1], name="process_grid[1]")
    if parts1 * parts2 != n_ranks:
        raise ValueError(
            f"process_grid product is {parts1 * parts2}; expected {n_ranks}"
        )
    if parts1 > n_elements1:
        raise ValueError(
            f"process_grid[0]={parts1} exceeds the {n_elements1} grid1 elements"
        )
    if parts2 > n_elements2:
        raise ValueError(
            f"process_grid[1]={parts2} exceeds the {n_elements2} grid2 elements"
        )
    return parts1, parts2


def _cartesian_rank_layout(
    rank1: int,
    rank2: int,
    layout1: TwoRadialDistributedLayout,
    layout2: TwoRadialDistributedLayout,
    process_grid: tuple[int, int],
) -> TwoRadial2DRankLayout:
    rank = rank1 * process_grid[1] + rank2
    coordinate1_neighbors = tuple(
        neighbor1 * process_grid[1] + rank2
        for neighbor1 in layout1.partition(rank1).neighbor_ranks
    )
    coordinate2_neighbors = tuple(
        rank1 * process_grid[1] + neighbor2
        for neighbor2 in layout2.partition(rank2).neighbor_ranks
    )
    return TwoRadial2DRankLayout(
        rank=rank,
        process_coordinates=(rank1, rank2),
        coordinate1=layout1.partition(rank1),
        coordinate2=layout2.partition(rank2),
        coordinate1_neighbor_ranks=coordinate1_neighbors,
        coordinate2_neighbor_ranks=coordinate2_neighbors,
        halo_owner_ranks1=(
            layout1.partition(rank1).halo_owner_ranks * process_grid[1]
            + rank2
        ),
        halo_owner_ranks2=(
            rank1 * process_grid[1]
            + layout2.partition(rank2).halo_owner_ranks
        ),
    )


def _representation_id(representation: TwoRadialSystem) -> str:
    digest = hashlib.sha256()
    digest.update(repr(representation.psi_shape).encode())
    digest.update(repr(representation.M_tot).encode())
    for values in (
        representation.grid1.r,
        representation.grid1.weights,
        representation.grid2.r,
        representation.grid2.weights,
        np.asarray(representation.basis1.states, dtype=np.int64),
        np.asarray(representation.basis2.states, dtype=np.int64),
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.dtype.str.encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _cartesian_layout_id(
    representation_id: str,
    process_grid: tuple[int, int],
    layout1: TwoRadialDistributedLayout,
    layout2: TwoRadialDistributedLayout,
) -> str:
    digest = hashlib.sha256()
    digest.update(representation_id.encode())
    digest.update(repr(process_grid).encode())
    digest.update(layout1.layout_id.encode())
    digest.update(layout2.layout_id.encode())
    return digest.hexdigest()


def _assign_radial_tile(
    destination: np.ndarray,
    indices1: np.ndarray,
    indices2: np.ndarray,
    values: np.ndarray,
) -> None:
    destination[..., indices1[:, None], indices2[None, :]] = values


def _validated_state_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"state shapes differ: {reference.shape} and {candidate.shape}"
        )
    for name, values in (("reference", reference), ("candidate", candidate)):
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
    return reference, candidate


def _validated_process_coordinates(
    coordinates,
    process_grid: tuple[int, int],
) -> tuple[int, int]:
    if isinstance(coordinates, (str, bytes)):
        raise TypeError("process_coordinates must contain two integers")
    try:
        values = tuple(coordinates)
    except TypeError as exc:
        raise TypeError("process_coordinates must contain two integers") from exc
    if len(values) != 2:
        raise ValueError("process_coordinates must contain exactly two entries")
    result = []
    for axis, (value, size) in enumerate(zip(values, process_grid)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"process_coordinates[{axis}] must be an integer")
        value = int(value)
        if not 0 <= value < size:
            raise ValueError(
                f"process_coordinates[{axis}] must satisfy 0 <= value < {size}"
            )
        result.append(value)
    return result[0], result[1]


def _communicator_rank_size(communicator) -> tuple[int, int]:
    try:
        rank = communicator.rank
        size = communicator.size
    except AttributeError:
        try:
            rank = communicator.Get_rank()
            size = communicator.Get_size()
        except AttributeError as exc:
            raise TypeError(
                "communicator must expose rank/size or Get_rank/Get_size"
            ) from exc
    return int(rank), int(size)


def _validate_collective(
    layout: TwoRadialDistributed2DLayout,
    size: int,
    root: int,
) -> None:
    if not isinstance(layout, TwoRadialDistributed2DLayout):
        raise TypeError("layout must be a TwoRadialDistributed2DLayout")
    if size != layout.n_ranks:
        raise ValueError(
            f"communicator has {size} ranks; layout expects {layout.n_ranks}"
        )
    _validated_rank(root, size, name="root")


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _validated_rank(value: int, size: int, *, name: str = "rank") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if not 0 <= value < size:
        raise ValueError(f"{name} must satisfy 0 <= {name} < {size}; got {value}")
    return value
