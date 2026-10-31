"""Physical reductions for Cartesian two-radial MPI layouts."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

import numpy as np

from .distributed_2d_state import (
    DistributedTwoRadial2DState,
    TwoRadialDistributed2DLayout,
)
from .distributed_hamiltonian import DistributedReductionProfile


class CartesianReductionCommunicator(Protocol):
    """Collective communicator operations required by 2D reductions."""

    rank: int
    size: int

    def allreduce(self, value): ...

    def Allreduce(self, send_buffer, receive_buffer): ...


class TwoRadialDistributed2DReductions:
    """Quadrature-weighted reductions over Cartesian radial tiles.

    Permanent ownership is nonoverlapping, so every rank contributes its
    owned coefficient tile exactly once. Batched inner products use one
    buffer-based collective when the communicator provides ``Allreduce``.
    """

    def __init__(
        self,
        layout: TwoRadialDistributed2DLayout,
        communicator: CartesianReductionCommunicator | None = None,
        *,
        profile: bool = False,
    ) -> None:
        if not isinstance(layout, TwoRadialDistributed2DLayout):
            raise TypeError("layout must be a TwoRadialDistributed2DLayout")
        if not isinstance(profile, (bool, np.bool_)):
            raise TypeError("profile must be boolean")
        self.layout = layout
        self.communicator = communicator
        self.profile_enabled = bool(profile)
        self._quadrature_weights = tuple(
            np.multiply.outer(
                layout.representation.grid1.weights[partition.owned_indices1],
                layout.representation.grid2.weights[partition.owned_indices2],
            )
            for partition in layout.partitions
        )
        self._weighted_right_buffers: dict[tuple[int, str], np.ndarray] = {}
        self._collective_buffers: dict[
            tuple[str, tuple[int, ...]], tuple[np.ndarray, np.ndarray]
        ] = {}
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
        self.reset_profile()

    def local_inner_product(
        self,
        left: DistributedTwoRadial2DState,
        right: DistributedTwoRadial2DState,
    ) -> complex:
        """Return one rank's nonoverlapping physical-inner-product term."""

        return self.local_inner_products((left,), right)[0]

    def local_inner_products(
        self,
        left: Sequence[DistributedTwoRadial2DState],
        right: DistributedTwoRadial2DState,
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
            [np.vdot(state.coefficients, weighted_right) for state in left]
        )
        if started is not None:
            self._local_inner_product_calls += len(left)
            self._local_inner_product_seconds += perf_counter() - started
        return values

    def inner_product(
        self,
        left: DistributedTwoRadial2DState,
        right: DistributedTwoRadial2DState,
    ) -> complex:
        """Collectively return the global physical inner product."""

        return self._allreduce(self.local_inner_product(left, right), left.rank)

    def inner_products(
        self,
        left: Sequence[DistributedTwoRadial2DState],
        right: DistributedTwoRadial2DState,
    ) -> np.ndarray:
        """Collectively return several inner products in one reduction."""

        return self._allreduce(self.local_inner_products(left, right), right.rank)

    def global_sum(self, values, *, rank: int):
        """Collectively sum a scalar or array of rank-local observables."""

        rank = _validated_rank(rank, self.layout.n_ranks)
        values = np.asarray(values)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("values must be numeric")
        if not np.isfinite(values).all():
            raise ValueError("values must be finite")
        return self._allreduce(values, rank)

    def inner_product_all(
        self,
        left: Sequence[DistributedTwoRadial2DState],
        right: Sequence[DistributedTwoRadial2DState],
    ) -> complex:
        """Emulate a global sum from complete rank-local collections."""

        left = _rank_ordered(self.layout, left)
        right = _rank_ordered(self.layout, right)
        return sum(
            (self.local_inner_product(a, b) for a, b in zip(left, right)),
            start=0.0j,
        )

    def norm_squared(self, state: DistributedTwoRadial2DState) -> float:
        """Return the global physical norm squared."""

        return float(self.inner_product(state, state).real)

    def norm(self, state: DistributedTwoRadial2DState) -> float:
        """Return the global physical norm."""

        value = self.norm_squared(state)
        if value < 0.0:
            raise ValueError("distributed norm squared is negative")
        return float(np.sqrt(value))

    def expectation(
        self,
        state: DistributedTwoRadial2DState,
        operator_state: DistributedTwoRadial2DState,
    ) -> complex:
        """Return the unnormalized expectation ``<state|operator_state>``."""

        return self.inner_product(state, operator_state)

    def rayleigh_quotient(
        self,
        state: DistributedTwoRadial2DState,
        operator_state: DistributedTwoRadial2DState,
    ) -> complex:
        """Return the normalized operator expectation."""

        norm_squared = self.norm_squared(state)
        if norm_squared <= 0.0:
            raise ValueError("state must have positive norm")
        return self.expectation(state, operator_state) / norm_squared

    def residual_norm(
        self,
        state: DistributedTwoRadial2DState,
        operator_state: DistributedTwoRadial2DState,
        eigenvalue: complex,
    ) -> float:
        """Return the global norm of ``operator_state - eigenvalue * state``."""

        self._validate_pair(state, operator_state)
        residual = DistributedTwoRadial2DState(
            self.layout,
            state.rank,
            operator_state.coefficients - eigenvalue * state.coefficients,
        )
        return self.norm(residual)

    @property
    def profile(self) -> DistributedReductionProfile:
        """Return an immutable snapshot of rank-local reduction timings."""

        return DistributedReductionProfile(
            local_inner_product_calls=self._local_inner_product_calls,
            local_inner_product_seconds=self._local_inner_product_seconds,
            collective_calls=self._collective_calls,
            collective_values=self._collective_values,
            collective_seconds=self._collective_seconds,
        )

    def reset_profile(self) -> None:
        """Clear accumulated rank-local reduction timings."""

        self._local_inner_product_calls = 0
        self._local_inner_product_seconds = 0.0
        self._collective_calls = 0
        self._collective_values = 0
        self._collective_seconds = 0.0

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

    def _weighted_right(self, state: DistributedTwoRadial2DState) -> np.ndarray:
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
        left: DistributedTwoRadial2DState,
        right: DistributedTwoRadial2DState,
    ) -> None:
        for state in (left, right):
            if not isinstance(state, DistributedTwoRadial2DState):
                raise TypeError(
                    "states must be DistributedTwoRadial2DState objects"
                )
            if state.layout.layout_id != self.layout.layout_id:
                raise ValueError("state belongs to a different distributed layout")
        if left.rank != right.rank:
            raise ValueError("local states must belong to the same rank")
        if self.rank is not None and left.rank != self.rank:
            raise ValueError(
                f"states belong to rank {left.rank}; reductions belong to "
                f"rank {self.rank}"
            )


def _rank_ordered(layout, states) -> tuple[DistributedTwoRadial2DState, ...]:
    if len(states) != layout.n_ranks:
        raise ValueError("one state is required for every rank")
    ordered = [None] * layout.n_ranks
    for state in states:
        if not isinstance(state, DistributedTwoRadial2DState):
            raise TypeError("states must contain DistributedTwoRadial2DState objects")
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
        rank, size = communicator.rank, communicator.size
    except AttributeError:
        try:
            rank, size = communicator.Get_rank(), communicator.Get_size()
        except AttributeError as exc:
            raise TypeError(
                "communicator must expose rank/size or Get_rank/Get_size"
            ) from exc
    return int(rank), int(size)


def _validated_rank(rank, size: int) -> int:
    if isinstance(rank, (bool, np.bool_)) or not isinstance(
        rank, (int, np.integer)
    ):
        raise TypeError("rank must be an integer")
    rank = int(rank)
    if not 0 <= rank < size:
        raise ValueError(f"rank must satisfy 0 <= rank < {size}")
    return rank
