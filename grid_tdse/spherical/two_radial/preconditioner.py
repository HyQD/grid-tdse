"""One-body preconditioning for two-radial Crank--Nicolson systems."""

from collections.abc import Mapping
from typing import Literal

import numpy as np

from .system import TwoRadialSystem


RealTimePreconditionerName = Literal[
    "spectral",
    "diagonal",
    "element-block-jacobi",
    "additive-schwarz",
]


class _TwoRadialOneBodyInverse:
    """Apply a shifted one-body inverse in weighted eigenbases."""

    def __init__(self, system: TwoRadialSystem, shift: complex) -> None:
        if not isinstance(system, TwoRadialSystem):
            raise TypeError("system must be a TwoRadialSystem")
        self.system = system
        self.shift = shift
        self._sqrt_w1 = np.sqrt(system.grid1.weights)
        self._sqrt_w2 = np.sqrt(system.grid2.weights)
        self._eigenvalues1, self._eigenvectors1 = _weighted_eigensystems(
            system.h1, self._sqrt_w1
        )
        self._eigenvalues2, self._eigenvectors2 = _weighted_eigensystems(
            system.h2, self._sqrt_w2
        )

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state)
        if state.shape != self.system.psi_shape:
            raise ValueError(
                f"state has shape {state.shape}; expected {self.system.psi_shape}"
            )

        result = np.empty_like(
            state,
            dtype=np.result_type(state.dtype, self.shift),
        )
        for pair, (i, j) in enumerate(self.system.angular_pairs.pairs):
            eps1 = self._eigenvalues1[i]
            vectors1 = self._eigenvectors1[i]
            eps2 = self._eigenvalues2[j]
            vectors2 = self._eigenvectors2[j]
            block = (
                state[pair]
                if self.system.uses_compact_angular_storage
                else state[i, j]
            )
            weighted = self._sqrt_w1[:, None] * block * self._sqrt_w2[None, :]
            transformed = vectors1.T.conj() @ weighted @ vectors2
            denominator = 1.0 + self.shift * (eps1[:, None] + eps2[None, :])
            if np.any(np.isclose(denominator, 0.0)):
                raise np.linalg.LinAlgError(
                    "one-body preconditioner is singular for this time step"
                )
            transformed = transformed / denominator
            weighted_result = vectors1 @ transformed @ vectors2.T.conj()
            block_result = (
                weighted_result / self._sqrt_w1[:, None] / self._sqrt_w2[None, :]
            )
            if self.system.uses_compact_angular_storage:
                result[pair] = block_result
            else:
                result[i, j] = block_result
        return result


class TwoRadialOneBodyPreconditioner(_TwoRadialOneBodyInverse):
    r"""Approximate the imaginary-time CN inverse with the one-body terms.

    The preconditioner applies

    ``[I + dt/2 (h1 + h2)]**-1``

    independently to every angular-channel pair.  The coordinate Hamiltonians
    are diagonalized in their quadrature-weighted Hermitian representations.
    The two-coordinate Coulomb interaction is deliberately omitted.
    """

    def __init__(self, system: TwoRadialSystem, time_step: float) -> None:
        time_step = _positive_time_step(time_step)
        self.time_step = time_step
        self.alpha = 0.5 * time_step
        super().__init__(system, self.alpha)


class TwoRadialRealTimePreconditioner(_TwoRadialOneBodyInverse):
    r"""Approximate the real-time CN inverse with the one-body terms.

    The preconditioner applies

    ``[I + i dt/2 (h1 + h2)]**-1``

    independently to every angular-channel pair. The two-coordinate Coulomb
    interaction and explicitly time-dependent interactions are omitted.
    """

    name = "spectral"

    def __init__(self, system: TwoRadialSystem, time_step: float) -> None:
        time_step = _positive_time_step(time_step)
        self.time_step = time_step
        self.alpha = 0.5j * time_step
        super().__init__(system, self.alpha)


class TwoRadialDiagonalRealTimePreconditioner:
    r"""Apply the diagonal/Jacobi approximation to the real-time CN inverse.

    Only the diagonal of the separable one-body Hamiltonian is retained.  The
    Coulomb interaction, external field, and absorbing potential are omitted.
    The action is point-local and therefore requires no communication in a
    distributed radial representation.
    """

    name = "diagonal"

    def __init__(self, system: TwoRadialSystem, time_step: float) -> None:
        _validate_system(system)
        self.system = system
        self.time_step = _positive_time_step(time_step)
        self.alpha = 0.5j * self.time_step
        self._diagonal1 = np.diagonal(system.h1, axis1=-2, axis2=-1)
        self._diagonal2 = np.diagonal(system.h2, axis1=-2, axis2=-1)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state = _validate_state(self.system, state)
        result = np.empty_like(state, dtype=np.result_type(state.dtype, complex))
        for pair, (i, j) in enumerate(self.system.angular_pairs.pairs):
            block = _angular_block(self.system, state, pair, i, j)
            denominator = 1.0 + self.alpha * (
                self._diagonal1[i, :, None] + self._diagonal2[j, None, :]
            )
            if np.any(np.isclose(denominator, 0.0)):
                raise np.linalg.LinAlgError(
                    "diagonal real-time preconditioner is singular"
                )
            _set_angular_block(
                self.system,
                result,
                pair,
                i,
                j,
                block / denominator,
            )
        return result


class _TwoRadialElementPreconditioner:
    """Shared separable element-subdomain inverse along one radial axis."""

    def __init__(
        self,
        system: TwoRadialSystem,
        time_step: float,
        *,
        partition_coordinate: int,
        subdomain_indices: tuple[np.ndarray, ...],
        subdomain_weights: tuple[np.ndarray, ...],
    ) -> None:
        _validate_system(system)
        _validate_partition_coordinate(partition_coordinate)
        self.system = system
        self.time_step = _positive_time_step(time_step)
        self.alpha = 0.5j * self.time_step
        self.partition_coordinate = partition_coordinate
        self.subdomain_indices = subdomain_indices
        self.subdomain_weights = subdomain_weights

        if partition_coordinate == 2:
            global_blocks = system.h1
            global_weights = system.grid1.weights
            partitioned_blocks = system.h2
            partitioned_weights = system.grid2.weights
        else:
            global_blocks = system.h2
            global_weights = system.grid2.weights
            partitioned_blocks = system.h1
            partitioned_weights = system.grid1.weights

        self._sqrt_global_weights = np.sqrt(global_weights)
        self._global_eigenvalues, self._global_eigenvectors = (
            _weighted_eigensystems(global_blocks, self._sqrt_global_weights)
        )
        self._local_eigensystems = {}
        for channel, matrix in enumerate(partitioned_blocks):
            for subdomain, indices in enumerate(self.subdomain_indices):
                sqrt_weights = np.sqrt(partitioned_weights[indices])
                local_matrix = matrix[np.ix_(indices, indices)]
                values, vectors = _weighted_eigensystem(
                    local_matrix, sqrt_weights
                )
                self._local_eigensystems[channel, subdomain] = (
                    sqrt_weights,
                    values,
                    vectors,
                )

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state = _validate_state(self.system, state)
        result = np.zeros_like(state, dtype=np.result_type(state.dtype, complex))
        for pair, (i, j) in enumerate(self.system.angular_pairs.pairs):
            block = _angular_block(self.system, state, pair, i, j)
            if self.partition_coordinate == 2:
                oriented = block
                global_channel, partitioned_channel = i, j
            else:
                oriented = block.T
                global_channel, partitioned_channel = j, i

            block_result = np.zeros_like(
                oriented, dtype=np.result_type(oriented.dtype, complex)
            )
            global_values = self._global_eigenvalues[global_channel]
            global_vectors = self._global_eigenvectors[global_channel]
            for subdomain, (indices, scatter_weights) in enumerate(
                zip(self.subdomain_indices, self.subdomain_weights)
            ):
                sqrt_local, local_values, local_vectors = (
                    self._local_eigensystems[
                        partitioned_channel, subdomain
                    ]
                )
                local = oriented[:, indices]
                weighted = (
                    self._sqrt_global_weights[:, None]
                    * local
                    * sqrt_local[None, :]
                )
                transformed = global_vectors.T.conj() @ weighted @ local_vectors
                denominator = 1.0 + self.alpha * (
                    global_values[:, None] + local_values[None, :]
                )
                if np.any(np.isclose(denominator, 0.0)):
                    raise np.linalg.LinAlgError(
                        "element real-time preconditioner is singular"
                    )
                transformed /= denominator
                weighted_result = (
                    global_vectors @ transformed @ local_vectors.T.conj()
                )
                local_result = (
                    weighted_result
                    / self._sqrt_global_weights[:, None]
                    / sqrt_local[None, :]
                )
                block_result[:, indices] += (
                    local_result * scatter_weights[None, :]
                )

            if self.partition_coordinate == 1:
                block_result = block_result.T
            _set_angular_block(
                self.system, result, pair, i, j, block_result
            )
        return result


class TwoRadialElementBlockJacobiRealTimePreconditioner(
    _TwoRadialElementPreconditioner
):
    r"""Apply non-overlapping FEM-DVR element-block Jacobi along one axis.

    By default the second coordinate is partitioned.  For a non-BO diatomic
    system this is the internuclear coordinate ``R``, matching the planned
    distributed ``R``-slab state layout.  Every block keeps the full other
    radial coordinate and all work is independent between owned elements.
    """

    name = "element-block-jacobi"

    def __init__(
        self,
        system: TwoRadialSystem,
        time_step: float,
        *,
        partition_coordinate: int = 2,
    ) -> None:
        grid = _partition_grid(system, partition_coordinate)
        subdomains = _owned_element_subdomains(grid)
        weights = tuple(np.ones(indices.size) for indices in subdomains)
        super().__init__(
            system,
            time_step,
            partition_coordinate=partition_coordinate,
            subdomain_indices=subdomains,
            subdomain_weights=weights,
        )


class TwoRadialAdditiveSchwarzRealTimePreconditioner(
    _TwoRadialElementPreconditioner
):
    r"""Apply a weighted overlapping additive-Schwarz CN preconditioner.

    A subdomain is centered on each FEM-DVR element and includes the requested
    number of neighboring elements on either side.  Shared outputs are scaled
    by inverse multiplicity, giving a partition of unity.  The default
    partitions the second coordinate with one neighboring-element layer.
    """

    name = "additive-schwarz"

    def __init__(
        self,
        system: TwoRadialSystem,
        time_step: float,
        *,
        partition_coordinate: int = 2,
        overlap_elements: int = 1,
    ) -> None:
        if isinstance(overlap_elements, (bool, np.bool_)) or not isinstance(
            overlap_elements, (int, np.integer)
        ):
            raise TypeError("overlap_elements must be a nonnegative integer")
        if overlap_elements < 0:
            raise ValueError("overlap_elements must be nonnegative")
        self.overlap_elements = int(overlap_elements)
        grid = _partition_grid(system, partition_coordinate)
        subdomains = _overlapping_element_subdomains(
            grid, self.overlap_elements
        )
        multiplicity = np.zeros(grid.n_points, dtype=int)
        for indices in subdomains:
            multiplicity[indices] += 1
        if np.any(multiplicity == 0):
            raise RuntimeError("additive-Schwarz subdomains do not cover the grid")
        weights = tuple(1.0 / multiplicity[indices] for indices in subdomains)
        super().__init__(
            system,
            time_step,
            partition_coordinate=partition_coordinate,
            subdomain_indices=subdomains,
            subdomain_weights=weights,
        )


def build_two_radial_real_time_preconditioner(
    system: TwoRadialSystem,
    time_step: float,
    preconditioner: bool | RealTimePreconditionerName | None = True,
    *,
    options: Mapping | None = None,
):
    """Build a real-time CN preconditioner from a public selector.

    ``True`` retains the historical global spectral preconditioner and
    ``False`` or ``None`` disables preconditioning.
    """

    if isinstance(preconditioner, (bool, np.bool_)):
        name = "spectral" if preconditioner else None
    elif preconditioner is None:
        name = None
    elif isinstance(preconditioner, str):
        name = preconditioner
    else:
        raise TypeError("preconditioner must be boolean, a string, or None")

    choices = {
        "spectral",
        "diagonal",
        "element-block-jacobi",
        "additive-schwarz",
    }
    if name is not None and name not in choices:
        values = ", ".join(sorted(choices))
        raise ValueError(f"preconditioner must be one of: {values}, or disabled")

    supplied_options = {} if options is None else dict(options)
    allowed_options = {
        "spectral": set(),
        "diagonal": set(),
        "element-block-jacobi": {"partition_coordinate"},
        "additive-schwarz": {"partition_coordinate", "overlap_elements"},
        None: set(),
    }
    unknown = set(supplied_options).difference(allowed_options[name])
    if unknown:
        values = ", ".join(sorted(unknown))
        raise ValueError(
            f"unknown {name or 'disabled'} preconditioner options: {values}"
        )
    if name is None:
        return None
    classes = {
        "spectral": TwoRadialRealTimePreconditioner,
        "diagonal": TwoRadialDiagonalRealTimePreconditioner,
        "element-block-jacobi": (
            TwoRadialElementBlockJacobiRealTimePreconditioner
        ),
        "additive-schwarz": TwoRadialAdditiveSchwarzRealTimePreconditioner,
    }
    return classes[name](system, time_step, **supplied_options)


def lowest_weighted_eigenstate(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the lowest eigenpair of a quadrature-weighted Hermitian matrix."""

    eigenvalues, eigenstates = weighted_eigenstates(matrix, weights)
    return float(eigenvalues[0]), eigenstates[:, 0]


def weighted_eigenstates(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a matrix in its quadrature-weighted representation."""

    matrix = np.asarray(matrix)
    weights = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if weights.shape != (matrix.shape[0],) or np.any(weights <= 0.0):
        raise ValueError("weights must be positive and match the matrix dimension")

    sqrt_weights = np.sqrt(weights)
    transformed = sqrt_weights[:, None] * matrix / sqrt_weights[None, :]
    transformed = 0.5 * (transformed + transformed.T.conj())
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    states = eigenvectors / sqrt_weights[:, None]
    for column in range(states.shape[1]):
        pivot = int(np.argmax(np.abs(states[:, column])))
        phase = states[pivot, column]
        if abs(phase) != 0.0:
            states[:, column] *= phase.conjugate() / abs(phase)
    return eigenvalues, np.real_if_close(states)


def _weighted_eigensystems(blocks: np.ndarray, sqrt_weights: np.ndarray):
    eigenvalues = []
    eigenvectors = []
    for block in blocks:
        transformed = sqrt_weights[:, None] * block / sqrt_weights[None, :]
        transformed = 0.5 * (transformed + transformed.T.conj())
        values, vectors = np.linalg.eigh(transformed)
        eigenvalues.append(values)
        eigenvectors.append(vectors)
    return np.asarray(eigenvalues), np.asarray(eigenvectors)


def _weighted_eigensystem(matrix: np.ndarray, sqrt_weights: np.ndarray):
    transformed = sqrt_weights[:, None] * matrix / sqrt_weights[None, :]
    transformed = 0.5 * (transformed + transformed.T.conj())
    return np.linalg.eigh(transformed)


def _validate_system(system: TwoRadialSystem) -> None:
    if not isinstance(system, TwoRadialSystem):
        raise TypeError("system must be a TwoRadialSystem")


def _validate_state(system: TwoRadialSystem, state: np.ndarray) -> np.ndarray:
    state = np.asarray(state)
    if state.shape != system.psi_shape:
        raise ValueError(f"state has shape {state.shape}; expected {system.psi_shape}")
    return state


def _angular_block(system, state, pair, channel1, channel2):
    if system.uses_compact_angular_storage:
        return state[pair]
    return state[channel1, channel2]


def _set_angular_block(system, state, pair, channel1, channel2, values):
    if system.uses_compact_angular_storage:
        state[pair] = values
    else:
        state[channel1, channel2] = values


def _partition_grid(system: TwoRadialSystem, coordinate: int):
    _validate_system(system)
    _validate_partition_coordinate(coordinate)
    grid = system.grid1 if coordinate == 1 else system.grid2
    if grid.family != "femdvr":
        raise ValueError("element preconditioners require a FEM-DVR grid")
    return grid


def _validate_partition_coordinate(coordinate: int) -> None:
    if isinstance(coordinate, (bool, np.bool_)) or not isinstance(
        coordinate, (int, np.integer)
    ):
        raise TypeError("partition_coordinate must be the integer 1 or 2")
    if coordinate not in (1, 2):
        raise ValueError("partition_coordinate must be 1 or 2")


def _interior_indices(full_indices: np.ndarray, full_size: int) -> np.ndarray:
    full_indices = np.asarray(full_indices, dtype=int)
    retained = full_indices[(full_indices > 0) & (full_indices < full_size - 1)]
    return retained - 1


def _owned_element_subdomains(grid) -> tuple[np.ndarray, ...]:
    full_size = grid.dvr.r.size
    subdomains = tuple(
        _interior_indices(indices, full_size)
        for indices in grid.dvr.element_owned_dof_maps
    )
    subdomains = tuple(indices for indices in subdomains if indices.size)
    _validate_partition(subdomains, grid.n_points)
    return subdomains


def _overlapping_element_subdomains(
    grid, overlap_elements: int
) -> tuple[np.ndarray, ...]:
    full_size = grid.dvr.r.size
    element_maps = grid.dvr.element_dof_maps
    subdomains = []
    for center in range(len(element_maps)):
        first = max(0, center - overlap_elements)
        stop = min(len(element_maps), center + overlap_elements + 1)
        indices = np.unique(np.concatenate(element_maps[first:stop]))
        indices = _interior_indices(indices, full_size)
        if indices.size:
            subdomains.append(indices)
    return tuple(subdomains)


def _validate_partition(subdomains, size):
    indices = np.concatenate(subdomains) if subdomains else np.asarray([], dtype=int)
    if not np.array_equal(np.sort(indices), np.arange(size)):
        raise RuntimeError("owned FEM-DVR element blocks do not partition the grid")


def _positive_time_step(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("time_step must be a positive real scalar")
    if np.iscomplexobj(value):
        raise TypeError("time_step must be a positive real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("time_step must be positive and finite")
    return result
