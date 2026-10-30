"""Distributed radial-state layout for two-radial FEM-DVR calculations.

The decomposition retains all angular channels and one complete radial
coordinate on every rank.  Either radial coordinate can be divided into
contiguous element-aligned blocks; coordinate 2 remains the default.
No MPI package is imported here: collective helpers accept any communicator
with the standard ``rank``, ``size``, ``scatter``, ``gather``, and ``bcast``
interface provided by :mod:`mpi4py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .state_archive import (
    TwoRadialState,
    load_two_radial_state,
    save_two_radial_state,
)
from .system import TwoRadialSystem


class CollectiveCommunicator(Protocol):
    """Subset of an MPI communicator used by the distributed-state helpers."""

    rank: int
    size: int

    def scatter(self, values, root: int = 0): ...

    def gather(self, value, root: int = 0): ...

    def bcast(self, value, root: int = 0): ...


@dataclass(frozen=True, eq=False)
class TwoRadialRankLayout:
    """Element and radial-degree-of-freedom ownership for one rank."""

    rank: int
    element_start: int
    element_stop: int
    owned_indices: np.ndarray
    owned_bridge_indices: np.ndarray
    local_indices: np.ndarray
    halo_indices: np.ndarray
    owned_local_indices: np.ndarray
    halo_local_indices: np.ndarray
    halo_owner_ranks: np.ndarray
    element_local_dof_maps: tuple[np.ndarray, ...]
    neighbor_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "owned_indices",
            "owned_bridge_indices",
            "local_indices",
            "halo_indices",
            "owned_local_indices",
            "halo_local_indices",
            "halo_owner_ranks",
        ):
            object.__setattr__(self, name, _readonly_int_array(getattr(self, name)))
        object.__setattr__(
            self,
            "element_local_dof_maps",
            tuple(
                _readonly_int_array(values)
                for values in self.element_local_dof_maps
            ),
        )

    @property
    def element_indices(self) -> range:
        """Contiguous FEM-DVR elements assigned to this rank."""

        return range(self.element_start, self.element_stop)

    @property
    def n_owned(self) -> int:
        return int(self.owned_indices.size)

    @property
    def n_local(self) -> int:
        """Number of owned plus halo radial degrees of freedom."""

        return int(self.local_indices.size)

    @property
    def n_halo(self) -> int:
        return int(self.halo_indices.size)

    @property
    def global_owned_slice(self) -> slice:
        """Contiguous global slice containing the permanently owned values."""

        return slice(int(self.owned_indices[0]), int(self.owned_indices[-1]) + 1)

    @property
    def global_local_slice(self) -> slice:
        """Contiguous global slice containing owned and halo values."""

        return slice(int(self.local_indices[0]), int(self.local_indices[-1]) + 1)


@dataclass(frozen=True, eq=False)
class TwoRadialDistributedLayout:
    """Permanent one-coordinate decomposition of a two-radial representation.

    ``partitioned_coordinate`` selects coordinate 1 or 2.  Every rank stores
    all angular axes, the complete unpartitioned radial coordinate, and only
    its owned indices along the selected coordinate.
    Halo indices describe additional input values needed by the elements local
    to that rank; halos are not part of the permanent state storage.
    """

    representation: TwoRadialSystem
    n_ranks: int
    partitioned_coordinate: int = 2
    partitions: tuple[TwoRadialRankLayout, ...] = field(init=False, repr=False)
    bridge_indices: np.ndarray = field(init=False, repr=False)
    owner_ranks: np.ndarray = field(init=False, repr=False)
    layout_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.representation, TwoRadialSystem):
            raise TypeError("representation must be a TwoRadialSystem")
        coordinate = self.partitioned_coordinate
        if isinstance(coordinate, (bool, np.bool_)) or not isinstance(
            coordinate, (int, np.integer)
        ):
            raise TypeError("partitioned_coordinate must be the integer 1 or 2")
        coordinate = int(coordinate)
        if coordinate not in (1, 2):
            raise ValueError("partitioned_coordinate must be 1 or 2")
        object.__setattr__(self, "partitioned_coordinate", coordinate)
        if self.partitioned_grid.family != "femdvr":
            raise ValueError(
                "distributed radial storage requires FEM-DVR "
                f"grid{coordinate}"
            )
        if isinstance(self.n_ranks, (bool, np.bool_)) or not isinstance(
            self.n_ranks, (int, np.integer)
        ):
            raise TypeError("n_ranks must be an integer")
        n_ranks = int(self.n_ranks)
        if n_ranks < 1:
            raise ValueError("n_ranks must be positive")

        dvr = self.partitioned_grid.dvr
        n_elements = int(dvr.n_intervals)
        if n_ranks > n_elements:
            raise ValueError(
                f"n_ranks={n_ranks} exceeds the {n_elements} FEM-DVR elements"
            )
        object.__setattr__(self, "n_ranks", n_ranks)

        owned_by_element = _interior_owned_indices(dvr)
        element_ranges = _balanced_element_ranges(
            np.asarray([values.size for values in owned_by_element], dtype=int),
            n_ranks,
        )
        element_owner = np.empty(n_elements, dtype=int)
        for rank, (start, stop) in enumerate(element_ranges):
            element_owner[start:stop] = rank
        bridge_indices = np.asarray(dvr.bridge_indices, dtype=int) - 1
        owner_ranks = element_owner[np.asarray(dvr.dof_owner[1:-1], dtype=int)]

        partitions = tuple(
            _rank_layout(
                rank,
                start,
                stop,
                dvr,
                owned_by_element,
                element_owner,
                bridge_indices,
                n_ranks,
            )
            for rank, (start, stop) in enumerate(element_ranges)
        )
        if any(partition.n_owned == 0 for partition in partitions):
            raise ValueError(
                "the requested decomposition gives a rank no interior radial "
                "degrees of freedom; use fewer ranks or more points per element"
            )

        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(
            self,
            "bridge_indices",
            _readonly_int_array(bridge_indices),
        )
        object.__setattr__(
            self,
            "owner_ranks",
            _readonly_int_array(owner_ranks),
        )
        object.__setattr__(self, "layout_id", _layout_id(self, element_ranges))

    @property
    def global_shape(self) -> tuple[int, ...]:
        return self.representation.psi_shape

    @property
    def n_R(self) -> int:
        return self.representation.grid2.n_points

    @property
    def partitioned_axis(self) -> int:
        """Positive array axis corresponding to the partitioned coordinate."""

        return len(self.global_shape) - (3 - self.partitioned_coordinate)

    @property
    def partitioned_grid(self):
        """Prepared radial grid selected for distribution."""

        if self.partitioned_coordinate == 1:
            return self.representation.grid1
        return self.representation.grid2

    @property
    def R_owner_ranks(self) -> np.ndarray:
        """Backward-compatible owner array for the default coordinate-2 layout."""

        return self.owner_ranks

    @property
    def archive_metadata(self) -> dict[str, Any]:
        """JSON-compatible description stored with distributed archives."""

        return {
            "partitioned_coordinate": self.partitioned_coordinate,
            "n_ranks": self.n_ranks,
            "layout_id": self.layout_id,
        }

    def partition(self, rank: int) -> TwoRadialRankLayout:
        """Return validated ownership metadata for ``rank``."""

        rank = _validated_rank(rank, self.n_ranks)
        return self.partitions[rank]

    def owned_shape(self, rank: int) -> tuple[int, ...]:
        """Permanent local state shape for ``rank``."""

        return _shape_with_axis(
            self.global_shape,
            self.partitioned_axis,
            self.partition(rank).n_owned,
        )

    def local_shape(self, rank: int) -> tuple[int, ...]:
        """Working shape including the halo values for ``rank``."""

        return _shape_with_axis(
            self.global_shape,
            self.partitioned_axis,
            self.partition(rank).n_local,
        )

    def extract_owned(self, state: np.ndarray, rank: int) -> np.ndarray:
        """Copy one rank's permanently owned values from a global state."""

        values = self._global_coefficients(state)
        indices = self.partition(rank).owned_indices
        return np.ascontiguousarray(
            np.take(values, indices, axis=self.partitioned_axis)
        )

    def distribute(
        self, state: np.ndarray | TwoRadialState
    ) -> tuple["DistributedTwoRadialState", ...]:
        """Split a global array or compatible archive state over all ranks."""

        values = self._global_coefficients(state)
        return tuple(
            DistributedTwoRadialState(
                self,
                rank,
                np.take(values, part.owned_indices, axis=self.partitioned_axis),
            )
            for rank, part in enumerate(self.partitions)
        )

    def assemble(
        self, states: Sequence["DistributedTwoRadialState"]
    ) -> np.ndarray:
        """Assemble one local state per rank into the serial representation."""

        if len(states) != self.n_ranks:
            raise ValueError(
                f"received {len(states)} local states; expected {self.n_ranks}"
            )
        by_rank: list[DistributedTwoRadialState | None] = [None] * self.n_ranks
        for state in states:
            if not isinstance(state, DistributedTwoRadialState):
                raise TypeError("states must contain DistributedTwoRadialState objects")
            if state.layout.layout_id != self.layout_id:
                raise ValueError(
                    "local state belongs to a different distributed layout"
                )
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
        """Assemble rank-ordered owned coefficient arrays."""

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
            _assign_axis(
                result,
                self.partitioned_axis,
                partition.owned_indices,
                value,
            )
        return result

    def _global_coefficients(
        self, state: np.ndarray | TwoRadialState
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
class DistributedTwoRadialState:
    """Permanently owned coefficients for one distributed rank."""

    layout: TwoRadialDistributedLayout
    rank: int
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.layout, TwoRadialDistributedLayout):
            raise TypeError("layout must be a TwoRadialDistributedLayout")
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
    def partition(self) -> TwoRadialRankLayout:
        return self.layout.partition(self.rank)

    @property
    def global_shape(self) -> tuple[int, ...]:
        return self.layout.global_shape

    @property
    def owned_R(self) -> np.ndarray:
        """Global ``R`` coordinates owned by this rank."""

        if self.layout.partitioned_coordinate == 1:
            return self.layout.representation.grid2.r
        return self.layout.representation.grid2.r[self.partition.owned_indices]

    @property
    def owned_R_weights(self) -> np.ndarray:
        if self.layout.partitioned_coordinate == 1:
            return self.layout.representation.grid2.weights
        return self.layout.representation.grid2.weights[self.partition.owned_indices]

    @property
    def owned_r(self) -> np.ndarray:
        """Global coordinate-1 points locally owned by this rank."""

        if self.layout.partitioned_coordinate == 2:
            return self.layout.representation.grid1.r
        return self.layout.representation.grid1.r[self.partition.owned_indices]

    @property
    def owned_r_weights(self) -> np.ndarray:
        if self.layout.partitioned_coordinate == 2:
            return self.layout.representation.grid1.weights
        return self.layout.representation.grid1.weights[self.partition.owned_indices]

    @property
    def owned_partitioned_coordinates(self) -> np.ndarray:
        """Points owned along the selected distributed coordinate."""

        return self.layout.partitioned_grid.r[self.partition.owned_indices]

    @property
    def owned_partitioned_weights(self) -> np.ndarray:
        return self.layout.partitioned_grid.weights[self.partition.owned_indices]

    def copy(self) -> "DistributedTwoRadialState":
        return DistributedTwoRadialState(
            self.layout,
            self.rank,
            self.coefficients.copy(),
        )


def scatter_two_radial_state(
    layout: TwoRadialDistributedLayout,
    communicator: CollectiveCommunicator,
    state: np.ndarray | TwoRadialState | None = None,
    *,
    root: int = 0,
) -> DistributedTwoRadialState:
    """Collectively scatter a global or archived state from ``root``."""

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
    return DistributedTwoRadialState(layout, rank, coefficients)


def gather_two_radial_state(
    state: DistributedTwoRadialState,
    communicator: CollectiveCommunicator,
    *,
    root: int = 0,
) -> np.ndarray | None:
    """Collectively gather permanent state storage onto ``root``."""

    if not isinstance(state, DistributedTwoRadialState):
        raise TypeError("state must be a DistributedTwoRadialState")
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


def load_distributed_two_radial_state(
    path: str | Path,
    layout: TwoRadialDistributedLayout,
    communicator: CollectiveCommunicator,
    *,
    root: int = 0,
) -> DistributedTwoRadialState:
    """Collectively load a serial archive on ``root`` and scatter its state."""

    rank, size = _communicator_rank_size(communicator)
    _validate_collective(layout, size, root)
    archived = None
    original_error: Exception | None = None
    error = None
    if rank == root:
        try:
            archived = load_two_radial_state(path)
            archived.validate_representation(layout.representation)
        except Exception as exc:  # keep all ranks on the same collective path
            original_error = exc
            error = (type(exc).__name__, str(exc))
    error = communicator.bcast(error, root=root)
    if error is not None:
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"rank {root} could not load state archive: {error[1]}")
    return scatter_two_radial_state(
        layout,
        communicator,
        archived if rank == root else None,
        root=root,
    )


def save_distributed_two_radial_state(
    path: str | Path,
    state: DistributedTwoRadialState,
    communicator: CollectiveCommunicator,
    *,
    physical_state: Mapping[str, Any],
    calculation: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    root: int = 0,
) -> Path | None:
    """Collectively gather and save through the existing serial archive format."""

    rank, _ = _communicator_rank_size(communicator)
    coefficients = gather_two_radial_state(state, communicator, root=root)
    destination = None
    original_error: Exception | None = None
    error = None
    if rank == root:
        try:
            destination = save_two_radial_state(
                path,
                coefficients,
                state.layout.representation,
                physical_state=physical_state,
                calculation=_archive_calculation(state.layout, calculation),
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


def _interior_owned_indices(dvr) -> tuple[np.ndarray, ...]:
    n_full = int(dvr.r.size)
    result = []
    for element in range(int(dvr.n_intervals)):
        full = np.flatnonzero(np.asarray(dvr.dof_owner) == element)
        result.append(full[(full > 0) & (full < n_full - 1)] - 1)
    return tuple(result)


def _interior_element_indices(dvr, element: int) -> np.ndarray:
    values = np.asarray(dvr.element_dof_maps[element], dtype=int)
    values = values[(values > 0) & (values < dvr.r.size - 1)]
    return values - 1


def _balanced_element_ranges(
    weights: np.ndarray, n_parts: int
) -> tuple[tuple[int, int], ...]:
    """Split ordered elements into nonempty, approximately balanced ranges."""

    n_elements = int(weights.size)
    prefix = np.concatenate(([0], np.cumsum(weights, dtype=np.int64)))
    ranges = []
    start = 0
    for part in range(n_parts - 1):
        remaining_parts = n_parts - part
        maximum_stop = n_elements - (remaining_parts - 1)
        target = prefix[start] + (prefix[-1] - prefix[start]) / remaining_parts
        insertion = int(np.searchsorted(prefix, target, side="left"))
        candidates = {
            min(max(start + 1, insertion - 1), maximum_stop),
            min(max(start + 1, insertion), maximum_stop),
        }
        stop = min(
            candidates,
            key=lambda value: (abs(prefix[value] - target), value),
        )
        ranges.append((start, stop))
        start = stop
    ranges.append((start, n_elements))
    return tuple(ranges)


def _rank_layout(
    rank: int,
    start: int,
    stop: int,
    dvr,
    owned_by_element: tuple[np.ndarray, ...],
    element_owner: np.ndarray,
    bridge_indices: np.ndarray,
    n_ranks: int,
) -> TwoRadialRankLayout:
    owned = np.concatenate(owned_by_element[start:stop])
    local = np.unique(
        np.concatenate(
            [_interior_element_indices(dvr, element) for element in range(start, stop)]
        )
    )
    halo = np.setdiff1d(local, owned, assume_unique=True)
    owned_local = np.searchsorted(local, owned)
    halo_local = np.searchsorted(local, halo)

    full_owner = np.asarray(dvr.dof_owner, dtype=int)
    halo_owner_elements = full_owner[halo + 1]
    halo_owner_ranks = element_owner[halo_owner_elements]
    element_maps = tuple(
        np.searchsorted(local, _interior_element_indices(dvr, element))
        for element in range(start, stop)
    )
    neighbors = tuple(
        neighbor
        for neighbor in (rank - 1, rank + 1)
        if 0 <= neighbor < n_ranks
    )
    return TwoRadialRankLayout(
        rank=rank,
        element_start=start,
        element_stop=stop,
        owned_indices=owned,
        owned_bridge_indices=np.intersect1d(
            owned,
            bridge_indices,
            assume_unique=True,
        ),
        local_indices=local,
        halo_indices=halo,
        owned_local_indices=owned_local,
        halo_local_indices=halo_local,
        halo_owner_ranks=halo_owner_ranks,
        element_local_dof_maps=element_maps,
        neighbor_ranks=neighbors,
    )


def _layout_id(
    layout: TwoRadialDistributedLayout,
    element_ranges: tuple[tuple[int, int], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(repr(layout.global_shape).encode())
    digest.update(repr(layout.representation.M_tot).encode())
    digest.update(repr(layout.partitioned_coordinate).encode())
    digest.update(repr(element_ranges).encode())
    for values in (
        layout.representation.grid1.r,
        layout.representation.grid1.weights,
        layout.representation.grid2.r,
        layout.representation.grid2.weights,
        np.asarray(layout.representation.basis1.states, dtype=np.int64),
        np.asarray(layout.representation.basis2.states, dtype=np.int64),
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.dtype.str.encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


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
    layout: TwoRadialDistributedLayout, size: int, root: int
) -> None:
    if size != layout.n_ranks:
        raise ValueError(
            f"communicator has {size} ranks; layout expects {layout.n_ranks}"
        )
    _validated_rank(root, size, name="root")


def _validated_rank(value: int, size: int, *, name: str = "rank") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if not 0 <= value < size:
        raise ValueError(f"{name} must satisfy 0 <= {name} < {size}; got {value}")
    return value


def _readonly_int_array(values) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=int)
    result.setflags(write=False)
    return result


def _shape_with_axis(
    shape: tuple[int, ...], axis: int, size: int
) -> tuple[int, ...]:
    result = list(shape)
    result[axis] = int(size)
    return tuple(result)


def _assign_axis(
    destination: np.ndarray,
    axis: int,
    indices: np.ndarray,
    values: np.ndarray,
) -> None:
    selection = [slice(None)] * destination.ndim
    selection[axis] = indices
    destination[tuple(selection)] = values


def _archive_calculation(
    layout: TwoRadialDistributedLayout,
    calculation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = {} if calculation is None else dict(calculation)
    result["distributed_layout"] = layout.archive_metadata
    return result
