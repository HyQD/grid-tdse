"""Optimized Hamiltonian action for two-radial wavefunctions."""

import numpy as np
from numba import njit, prange

from .angular import AngularPairBasis
from .couplings import CoulombCouplings
from .coulomb import RadialCoulombMultipoles
from .grids import RadialGrid
from .system import CoulombKernelMode, RadialKernelMode, RadialOperatorMode


class TwoRadialHamiltonian:
    """Apply one-body blocks and sparse multipole couplings to a wavefunction."""

    def __init__(
        self,
        h1: np.ndarray | None,
        h2: np.ndarray | None,
        radial_multipoles: RadialCoulombMultipoles,
        couplings: CoulombCouplings,
        *,
        angular_pairs: AngularPairBasis | None = None,
        grid1: RadialGrid | None = None,
        grid2: RadialGrid | None = None,
        kinetic_prefactor1: float | None = None,
        kinetic_prefactor2: float | None = None,
        h1_diagonal: np.ndarray | None = None,
        h2_diagonal: np.ndarray | None = None,
        radial_operator_mode: RadialOperatorMode = "global",
        tile_size: int = 2048,
        coulomb_kernel: CoulombKernelMode = "reference",
        radial_kernel: RadialKernelMode = "reference",
    ) -> None:
        self.h1 = None if h1 is None else np.ascontiguousarray(h1)
        self.h2 = None if h2 is None else np.ascontiguousarray(h2)
        self.h2_transpose = (
            None
            if self.h2 is None or radial_operator_mode != "global"
            else np.ascontiguousarray(self.h2.transpose(0, 2, 1))[None]
        )
        self.radial_multipoles = radial_multipoles
        self.couplings = couplings
        self.angular_pairs = angular_pairs
        self.grid1 = grid1
        self.grid2 = grid2
        self.kinetic_prefactor1 = kinetic_prefactor1
        self.kinetic_prefactor2 = kinetic_prefactor2
        self.h1_diagonal = (
            None if h1_diagonal is None else np.ascontiguousarray(h1_diagonal)
        )
        self.h2_diagonal = (
            None if h2_diagonal is None else np.ascontiguousarray(h2_diagonal)
        )
        self.radial_operator_mode = radial_operator_mode
        self.tile_size = int(tile_size)
        if self.tile_size < 1:
            raise ValueError("tile_size must be positive")
        if coulomb_kernel not in {"reference", "multipole-grouped"}:
            raise ValueError(
                "coulomb_kernel must be 'reference' or 'multipole-grouped'"
            )
        self.coulomb_kernel = coulomb_kernel
        if radial_kernel not in {"reference", "row-wise"}:
            raise ValueError("radial_kernel must be 'reference' or 'row-wise'")
        self.radial_kernel = radial_kernel
        self._validate_shapes()
        self._radial_flat = np.ascontiguousarray(
            radial_multipoles.values.reshape(radial_multipoles.values.shape[0], -1)
        )
        if (
            self.radial_operator_mode == "element-local"
            and self.radial_kernel == "row-wise"
        ):
            self._local_radial_arrays = (
                _element_local_radial_arrays(self.grid1),
                _element_local_radial_arrays(self.grid2),
            )
        if self._uses_compact_storage:
            self._pair_channels1 = np.asarray(
                [i for i, _ in self.angular_pairs.pairs], dtype=np.int32
            )
            self._pair_channels2 = np.asarray(
                [j for _, j in self.angular_pairs.pairs], dtype=np.int32
            )
            self._compact_couplings = _compact_coupling_arrays(
                couplings,
                self.angular_pairs,
            )
            if self.coulomb_kernel == "multipole-grouped":
                self._compact_multipole_blocks = _multipole_block_arrays(
                    self._compact_couplings[1],
                    self._compact_couplings[3],
                )
            if self.radial_operator_mode == "global":
                self._h1_pairs = np.ascontiguousarray(
                    self.h1[self._pair_channels1]
                )
                self._h2_pairs_transpose = np.ascontiguousarray(
                    self.h2[self._pair_channels2].transpose(0, 2, 1)
                )
            if self.h1_diagonal is not None:
                self._h1_diagonal_pairs = np.ascontiguousarray(
                    self.h1_diagonal[self._pair_channels1]
                )
            if self.h2_diagonal is not None:
                self._h2_diagonal_pairs = np.ascontiguousarray(
                    self.h2_diagonal[self._pair_channels2]
                )
        elif self.coulomb_kernel == "multipole-grouped":
            self._multipole_blocks = _multipole_block_arrays(
                couplings.offsets,
                couplings.radial_index,
            )

    @property
    def _uses_compact_storage(self) -> bool:
        return self.angular_pairs is not None and self.angular_pairs.is_compact

    @property
    def psi_shape(self) -> tuple[int, ...]:
        n_radial1, n_radial2 = self.radial_multipoles.values.shape[1:]
        if self._uses_compact_storage:
            return (
                self.angular_pairs.n_pairs,
                n_radial1,
                n_radial2,
            )
        return (
            self.couplings.n_channels_1,
            self.couplings.n_channels_2,
            n_radial1,
            n_radial2,
        )

    def __call__(self, psi: np.ndarray) -> np.ndarray:
        psi = self._validated_state(psi)
        result = self.apply_one_body_coordinate1(psi)
        result += self.apply_one_body_coordinate2(psi)
        return self.apply_coulomb(psi, out=result)

    def apply_one_body_coordinate1(self, psi: np.ndarray) -> np.ndarray:
        """Apply the first-coordinate one-body Hamiltonian."""

        psi = self._validated_state(psi)
        if self.radial_operator_mode == "global":
            if self._uses_compact_storage:
                return np.matmul(self._h1_pairs, psi)
            return np.matmul(self.h1[:, None], psi)

        if self.radial_kernel == "reference":
            result = self.kinetic_prefactor1 * (
                self.grid1.apply_local_second_derivative(psi, axis=-2)
            )
        else:
            matrices, maps, sizes = self._local_radial_arrays[0]
            result = np.empty(
                psi.shape,
                dtype=np.result_type(psi.dtype, matrices.dtype, float),
            )
            flattened = psi.reshape((-1, psi.shape[-2], psi.shape[-1]))
            flattened_result = result.reshape(flattened.shape)
            _apply_local_radial_coordinate1_kernel(
                flattened_result,
                flattened,
                matrices,
                maps,
                sizes,
                self.grid1.weights,
                self.kinetic_prefactor1,
            )
        if self._uses_compact_storage:
            result += self._h1_diagonal_pairs[:, :, np.newaxis] * psi
        else:
            result += self.h1_diagonal[:, np.newaxis, :, np.newaxis] * psi
        return result

    def apply_one_body_coordinate2(self, psi: np.ndarray) -> np.ndarray:
        """Apply the second-coordinate one-body Hamiltonian."""

        psi = self._validated_state(psi)
        if self.radial_operator_mode == "global":
            if self._uses_compact_storage:
                return np.matmul(psi, self._h2_pairs_transpose)
            return np.matmul(psi, self.h2_transpose)

        if self.radial_kernel == "reference":
            result = self.kinetic_prefactor2 * (
                self.grid2.apply_local_second_derivative(psi, axis=-1)
            )
        else:
            matrices, maps, sizes = self._local_radial_arrays[1]
            result = np.empty(
                psi.shape,
                dtype=np.result_type(psi.dtype, matrices.dtype, float),
            )
            flattened = psi.reshape((-1, psi.shape[-2], psi.shape[-1]))
            flattened_result = result.reshape(flattened.shape)
            _apply_local_radial_coordinate2_kernel(
                flattened_result,
                flattened,
                matrices,
                maps,
                sizes,
                self.grid2.weights,
                self.kinetic_prefactor2,
            )
        if self._uses_compact_storage:
            result += self._h2_diagonal_pairs[:, np.newaxis, :] * psi
        else:
            result += self.h2_diagonal[np.newaxis, :, np.newaxis, :] * psi
        return result

    def apply_coulomb(
        self, psi: np.ndarray, *, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Apply the optimized, output-blocked Coulomb action.

        This is the production path descended from the Bodenstein kernel. The
        dense ``y`` and ``y_bar`` tensors are never constructed.
        """

        psi = self._validated_state(psi)
        return self._apply_coulomb_values(psi, self._radial_flat, out=out)

    def apply_coulomb_R_slice(
        self,
        psi: np.ndarray,
        R_indices: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the point-local Coulomb action on selected coordinate-2 points.

        The angular and first-radial axes remain complete.  This method is the
        distributed-state path and does not construct a full coordinate-2
        state or output array.
        """

        indices = np.asarray(R_indices)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("R_indices must be a nonempty one-dimensional array")
        if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
            raise TypeError("R_indices must contain integers")
        indices = np.asarray(indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= self.psi_shape[-1]):
            raise ValueError("R_indices contains an out-of-range index")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("R_indices must be strictly increasing")

        expected_shape = (*self.psi_shape[:-1], indices.size)
        psi = self._validated_shape(psi, expected_shape)
        radial = np.ascontiguousarray(
            self.radial_multipoles.values[..., indices].reshape(
                self.radial_multipoles.values.shape[0],
                -1,
            )
        )
        return self._apply_coulomb_values(psi, radial, out=out)

    def _apply_coulomb_values(
        self,
        psi: np.ndarray,
        radial_flat: np.ndarray,
        *,
        out: np.ndarray | None,
    ) -> np.ndarray:
        expected_shape = psi.shape
        dtype = np.result_type(psi.dtype, radial_flat.dtype)
        if out is None:
            out = np.zeros(expected_shape, dtype=dtype)
        elif out.shape != expected_shape:
            raise ValueError(f"out has shape {out.shape}; expected {expected_shape}")
        elif not np.can_cast(dtype, out.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot accumulate {dtype} Coulomb action in {out.dtype} output"
            )
        elif not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")

        if self._uses_compact_storage:
            output = out.reshape(expected_shape[0], -1)
            wavefunction = psi.reshape(expected_shape[0], -1)
            output_pair, offsets, input_pair, radial_index, coefficient = (
                self._compact_couplings
            )
            if self.coulomb_kernel == "reference":
                _apply_compact_coulomb_kernel(
                    output,
                    wavefunction,
                    radial_flat,
                    output_pair,
                    offsets,
                    input_pair,
                    radial_index,
                    coefficient,
                    self.tile_size,
                )
            else:
                group_block_offsets, block_pair_offsets, block_radial_index = (
                    self._compact_multipole_blocks
                )
                if np.iscomplexobj(wavefunction):
                    _apply_compact_multipole_grouped_complex_kernel(
                        output,
                        wavefunction,
                        radial_flat,
                        output_pair,
                        group_block_offsets,
                        block_pair_offsets,
                        block_radial_index,
                        input_pair,
                        coefficient,
                        self.tile_size,
                    )
                else:
                    _apply_compact_multipole_grouped_real_kernel(
                        output,
                        wavefunction,
                        radial_flat,
                        output_pair,
                        group_block_offsets,
                        block_pair_offsets,
                        block_radial_index,
                        input_pair,
                        coefficient,
                        self.tile_size,
                    )
        else:
            output = out.reshape(expected_shape[0], expected_shape[1], -1)
            wavefunction = psi.reshape(expected_shape[0], expected_shape[1], -1)
            c = self.couplings
            if self.coulomb_kernel == "reference":
                _apply_coulomb_kernel(
                    output,
                    wavefunction,
                    radial_flat,
                    c.output_1,
                    c.output_2,
                    c.offsets,
                    c.input_1,
                    c.input_2,
                    c.radial_index,
                    c.coefficient,
                    self.tile_size,
                )
            else:
                group_block_offsets, block_pair_offsets, block_radial_index = (
                    self._multipole_blocks
                )
                if np.iscomplexobj(wavefunction):
                    _apply_multipole_grouped_complex_kernel(
                        output,
                        wavefunction,
                        radial_flat,
                        c.output_1,
                        c.output_2,
                        group_block_offsets,
                        block_pair_offsets,
                        block_radial_index,
                        c.input_1,
                        c.input_2,
                        c.coefficient,
                        self.tile_size,
                    )
                else:
                    _apply_multipole_grouped_real_kernel(
                        output,
                        wavefunction,
                        radial_flat,
                        c.output_1,
                        c.output_2,
                        group_block_offsets,
                        block_pair_offsets,
                        block_radial_index,
                        c.input_1,
                        c.input_2,
                        c.coefficient,
                        self.tile_size,
                    )
        return output.reshape(expected_shape)

    def _validate_shapes(self) -> None:
        if self.radial_operator_mode not in {"global", "element-local"}:
            raise ValueError(
                "radial_operator_mode must be 'global' or 'element-local'"
            )
        c = self.couplings
        n_radial1 = self.radial_multipoles.values.shape[1]
        n_radial2 = self.radial_multipoles.values.shape[2]
        expected = (
            len(self.radial_multipoles.lambdas),
            n_radial1,
            n_radial2,
        )
        if self.radial_multipoles.values.shape != expected:
            raise ValueError(
                "radial multipoles have shape "
                f"{self.radial_multipoles.values.shape}; expected {expected}"
            )
        if c.n_pairs and c.radial_index.max() >= expected[0]:
            raise ValueError("Coulomb coupling references a missing radial multipole")
        if self.coulomb_kernel == "multipole-grouped":
            if np.iscomplexobj(self.radial_multipoles.values):
                raise TypeError(
                    "multipole-grouped Coulomb kernel requires real radial "
                    "multipoles"
                )
            if np.iscomplexobj(c.coefficient):
                raise TypeError(
                    "multipole-grouped Coulomb kernel requires real coupling "
                    "coefficients"
                )
        if self.radial_operator_mode == "global":
            if self.h1 is None or self.h2 is None:
                raise ValueError(
                    "global radial operators require both dense h1 and h2 arrays"
                )
            if self.h1.ndim != 3 or self.h1.shape[1:] != (
                n_radial1,
                n_radial1,
            ):
                raise ValueError(
                    "h1 must have shape (channels1, r1, r1)"
                )
            if self.h2.ndim != 3 or self.h2.shape[1:] != (
                n_radial2,
                n_radial2,
            ):
                raise ValueError(
                    "h2 must have shape (channels2, r2, r2)"
                )
            if (
                self.h1.shape[0] != c.n_channels_1
                or self.h2.shape[0] != c.n_channels_2
            ):
                raise ValueError(
                    "one-body channel counts do not match Coulomb couplings"
                )
        else:
            self._validate_element_local_data()

    def _validate_element_local_data(self) -> None:
        if self.grid1 is None or self.grid2 is None:
            raise ValueError(
                "element-local radial operators require both prepared radial grids"
            )
        if self.grid1.family != "femdvr" or self.grid2.family != "femdvr":
            raise ValueError("element-local radial operators require FEM-DVR grids")
        if self.kinetic_prefactor1 is None or self.kinetic_prefactor2 is None:
            raise ValueError(
                "element-local radial operators require both kinetic prefactors"
            )
        if not np.isfinite(self.kinetic_prefactor1) or not np.isfinite(
            self.kinetic_prefactor2
        ):
            raise ValueError("kinetic prefactors must be finite")
        expected1 = (self.couplings.n_channels_1, self.grid1.n_points)
        expected2 = (self.couplings.n_channels_2, self.grid2.n_points)
        if self.h1_diagonal is None or self.h1_diagonal.shape != expected1:
            actual = None if self.h1_diagonal is None else self.h1_diagonal.shape
            raise ValueError(
                f"h1_diagonal has shape {actual}; expected {expected1}"
            )
        if self.h2_diagonal is None or self.h2_diagonal.shape != expected2:
            actual = None if self.h2_diagonal is None else self.h2_diagonal.shape
            raise ValueError(
                f"h2_diagonal has shape {actual}; expected {expected2}"
            )

    def _validated_state(self, psi: np.ndarray) -> np.ndarray:
        return self._validated_shape(psi, self.psi_shape)

    @staticmethod
    def _validated_shape(psi: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
        psi = np.asarray(psi)
        if psi.shape != expected_shape:
            raise ValueError(f"psi has shape {psi.shape}; expected {expected_shape}")
        return np.ascontiguousarray(psi)


def _compact_coupling_arrays(
    couplings: CoulombCouplings,
    angular_pairs: AngularPairBasis,
):
    output_pair = []
    offsets = [0]
    input_pair = []
    radial_index = []
    coefficient = []
    for group, (i, j) in enumerate(
        zip(couplings.output_1, couplings.output_2, strict=True)
    ):
        output_pair.append(angular_pairs.index(int(i), int(j)))
        for pair in range(couplings.offsets[group], couplings.offsets[group + 1]):
            input_pair.append(
                angular_pairs.index(
                    int(couplings.input_1[pair]),
                    int(couplings.input_2[pair]),
                )
            )
            radial_index.append(couplings.radial_index[pair])
            coefficient.append(couplings.coefficient[pair])
        offsets.append(len(input_pair))
    return (
        np.ascontiguousarray(output_pair, dtype=np.int32),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(input_pair, dtype=np.int32),
        np.ascontiguousarray(radial_index, dtype=np.int32),
        np.ascontiguousarray(coefficient),
    )


def _element_local_radial_arrays(grid):
    dvr = grid.dvr
    retained_maps = []
    retained_matrices = []
    for full_map, stiffness in zip(
        dvr.element_dof_maps,
        dvr.local_stiffness_matrices,
        strict=True,
    ):
        retained = (full_map > 0) & (full_map < dvr.r.size - 1)
        retained_maps.append(np.asarray(full_map[retained] - 1, dtype=np.int32))
        retained_matrices.append(
            np.ascontiguousarray(-stiffness[np.ix_(retained, retained)])
        )
    maximum_size = max(map_.size for map_ in retained_maps)
    matrices = np.zeros(
        (len(retained_maps), maximum_size, maximum_size),
        dtype=np.result_type(*[matrix.dtype for matrix in retained_matrices]),
    )
    maps = np.full((len(retained_maps), maximum_size), -1, dtype=np.int32)
    sizes = np.empty(len(retained_maps), dtype=np.int32)
    for element, (map_, matrix) in enumerate(
        zip(retained_maps, retained_matrices, strict=True)
    ):
        size = map_.size
        maps[element, :size] = map_
        matrices[element, :size, :size] = matrix
        sizes[element] = size
    return (
        np.ascontiguousarray(matrices),
        np.ascontiguousarray(maps),
        np.ascontiguousarray(sizes),
    )


def _multipole_block_arrays(offsets, radial_index):
    """Group each output block's contiguous couplings by radial multipole."""

    group_block_offsets = [0]
    block_pair_offsets = [0]
    block_radial_index = []
    for group in range(len(offsets) - 1):
        start = int(offsets[group])
        stop = int(offsets[group + 1])
        if start < stop:
            block_radial_index.append(radial_index[start])
            for pair in range(start + 1, stop):
                if radial_index[pair] != radial_index[pair - 1]:
                    block_pair_offsets.append(pair)
                    block_radial_index.append(radial_index[pair])
            block_pair_offsets.append(stop)
        group_block_offsets.append(len(block_radial_index))
    return (
        np.ascontiguousarray(group_block_offsets, dtype=np.int64),
        np.ascontiguousarray(block_pair_offsets, dtype=np.int64),
        np.ascontiguousarray(block_radial_index, dtype=np.int32),
    )


@njit(cache=True, fastmath=True, parallel=True)
def _apply_local_radial_coordinate1_kernel(
    out,
    values,
    element_matrices,
    element_maps,
    element_sizes,
    weights,
    prefactor,
):
    n_channels, n_radial1, n_radial2 = values.shape
    for row in prange(n_channels * n_radial2):
        channel = row // n_radial2
        radial2 = row - channel * n_radial2
        for radial1 in range(n_radial1):
            out[channel, radial1, radial2] = 0
        for element in range(element_sizes.size):
            size = element_sizes[element]
            for local_output in range(size):
                radial1 = element_maps[element, local_output]
                action = 0
                for local_input in range(size):
                    input_radial1 = element_maps[element, local_input]
                    action += (
                        element_matrices[element, local_output, local_input]
                        * values[channel, input_radial1, radial2]
                    )
                out[channel, radial1, radial2] += action
        for radial1 in range(n_radial1):
            out[channel, radial1, radial2] *= prefactor / weights[radial1]


@njit(cache=True, fastmath=True, parallel=True)
def _apply_local_radial_coordinate2_kernel(
    out,
    values,
    element_matrices,
    element_maps,
    element_sizes,
    weights,
    prefactor,
):
    n_channels, n_radial1, n_radial2 = values.shape
    for row in prange(n_channels * n_radial1):
        channel = row // n_radial1
        radial1 = row - channel * n_radial1
        for radial2 in range(n_radial2):
            out[channel, radial1, radial2] = 0
        for element in range(element_sizes.size):
            size = element_sizes[element]
            for local_output in range(size):
                radial2 = element_maps[element, local_output]
                action = 0
                for local_input in range(size):
                    input_radial2 = element_maps[element, local_input]
                    action += (
                        element_matrices[element, local_output, local_input]
                        * values[channel, radial1, input_radial2]
                    )
                out[channel, radial1, radial2] += action
        for radial2 in range(n_radial2):
            out[channel, radial1, radial2] *= prefactor / weights[radial2]


@njit(cache=True, fastmath=True, parallel=True)
def _apply_coulomb_kernel(
    out,
    psi,
    radial,
    output_1,
    output_2,
    offsets,
    input_1,
    input_2,
    radial_index,
    coefficient,
    tile_size,
):
    n_radial = psi.shape[2]
    for group in prange(output_1.size):
        i = output_1[group]
        j = output_2[group]
        scratch = np.zeros(tile_size, dtype=out.dtype)
        for start in range(0, n_radial, tile_size):
            stop = min(start + tile_size, n_radial)
            length = stop - start
            for point in range(length):
                scratch[point] = 0
            for pair in range(offsets[group], offsets[group + 1]):
                k = input_1[pair]
                l = input_2[pair]
                radial_row = radial_index[pair]
                value = coefficient[pair]
                for point in range(length):
                    flat_index = start + point
                    scratch[point] += (
                        value
                        * radial[radial_row, flat_index]
                        * psi[k, l, flat_index]
                    )
            for point in range(length):
                out[i, j, start + point] += scratch[point]


@njit(cache=True, fastmath=True, parallel=True)
def _apply_compact_coulomb_kernel(
    out,
    psi,
    radial,
    output_pair,
    offsets,
    input_pair,
    radial_index,
    coefficient,
    tile_size,
):
    n_radial = psi.shape[1]
    for group in prange(output_pair.size):
        output = output_pair[group]
        scratch = np.zeros(tile_size, dtype=out.dtype)
        for start in range(0, n_radial, tile_size):
            stop = min(start + tile_size, n_radial)
            length = stop - start
            for point in range(length):
                scratch[point] = 0
            for pair in range(offsets[group], offsets[group + 1]):
                input_ = input_pair[pair]
                radial_row = radial_index[pair]
                value = coefficient[pair]
                for point in range(length):
                    flat_index = start + point
                    scratch[point] += (
                        value
                        * radial[radial_row, flat_index]
                        * psi[input_, flat_index]
                    )
            for point in range(length):
                out[output, start + point] += scratch[point]


@njit(cache=True, fastmath=True, parallel=True)
def _apply_multipole_grouped_real_kernel(
    out,
    psi,
    radial,
    output_1,
    output_2,
    group_block_offsets,
    block_pair_offsets,
    block_radial_index,
    input_1,
    input_2,
    coefficient,
    tile_size,
):
    n_radial = psi.shape[2]
    for group in prange(output_1.size):
        i = output_1[group]
        j = output_2[group]
        total = np.empty(tile_size, dtype=out.dtype)
        angular = np.empty(tile_size, dtype=out.dtype)
        for start in range(0, n_radial, tile_size):
            stop = min(start + tile_size, n_radial)
            length = stop - start
            for point in range(length):
                total[point] = 0
            for block in range(
                group_block_offsets[group], group_block_offsets[group + 1]
            ):
                pair_start = block_pair_offsets[block]
                pair_stop = block_pair_offsets[block + 1]
                k = input_1[pair_start]
                l = input_2[pair_start]
                value = coefficient[pair_start]
                for point in range(length):
                    angular[point] = value * psi[k, l, start + point]
                for pair in range(pair_start + 1, pair_stop):
                    k = input_1[pair]
                    l = input_2[pair]
                    value = coefficient[pair]
                    for point in range(length):
                        angular[point] += value * psi[k, l, start + point]
                radial_row = block_radial_index[block]
                for point in range(length):
                    flat_index = start + point
                    total[point] += radial[radial_row, flat_index] * angular[point]
            for point in range(length):
                out[i, j, start + point] += total[point]


@njit(cache=True, fastmath=True, parallel=True)
def _apply_multipole_grouped_complex_kernel(
    out,
    psi,
    radial,
    output_1,
    output_2,
    group_block_offsets,
    block_pair_offsets,
    block_radial_index,
    input_1,
    input_2,
    coefficient,
    tile_size,
):
    n_radial = psi.shape[2]
    for group in prange(output_1.size):
        i = output_1[group]
        j = output_2[group]
        total_real = np.empty(tile_size, dtype=out.real.dtype)
        total_imag = np.empty(tile_size, dtype=out.real.dtype)
        angular_real = np.empty(tile_size, dtype=out.real.dtype)
        angular_imag = np.empty(tile_size, dtype=out.real.dtype)
        for start in range(0, n_radial, tile_size):
            stop = min(start + tile_size, n_radial)
            length = stop - start
            for point in range(length):
                total_real[point] = 0
                total_imag[point] = 0
            for block in range(
                group_block_offsets[group], group_block_offsets[group + 1]
            ):
                pair_start = block_pair_offsets[block]
                pair_stop = block_pair_offsets[block + 1]
                k = input_1[pair_start]
                l = input_2[pair_start]
                value = coefficient[pair_start]
                for point in range(length):
                    state = psi[k, l, start + point]
                    angular_real[point] = value * state.real
                    angular_imag[point] = value * state.imag
                for pair in range(pair_start + 1, pair_stop):
                    k = input_1[pair]
                    l = input_2[pair]
                    value = coefficient[pair]
                    for point in range(length):
                        state = psi[k, l, start + point]
                        angular_real[point] += value * state.real
                        angular_imag[point] += value * state.imag
                radial_row = block_radial_index[block]
                for point in range(length):
                    flat_index = start + point
                    radial_value = radial[radial_row, flat_index]
                    total_real[point] += radial_value * angular_real[point]
                    total_imag[point] += radial_value * angular_imag[point]
            for point in range(length):
                out[i, j, start + point] += (
                    total_real[point] + 1j * total_imag[point]
                )


@njit(cache=True, fastmath=True, parallel=True)
def _apply_compact_multipole_grouped_real_kernel(
    out,
    psi,
    radial,
    output_pair,
    group_block_offsets,
    block_pair_offsets,
    block_radial_index,
    input_pair,
    coefficient,
    tile_size,
):
    n_radial = psi.shape[1]
    for group in prange(output_pair.size):
        output = output_pair[group]
        total = np.empty(tile_size, dtype=out.dtype)
        angular = np.empty(tile_size, dtype=out.dtype)
        for start in range(0, n_radial, tile_size):
            stop = min(start + tile_size, n_radial)
            length = stop - start
            for point in range(length):
                total[point] = 0
            for block in range(
                group_block_offsets[group], group_block_offsets[group + 1]
            ):
                pair_start = block_pair_offsets[block]
                pair_stop = block_pair_offsets[block + 1]
                input_ = input_pair[pair_start]
                value = coefficient[pair_start]
                for point in range(length):
                    angular[point] = value * psi[input_, start + point]
                for pair in range(pair_start + 1, pair_stop):
                    input_ = input_pair[pair]
                    value = coefficient[pair]
                    for point in range(length):
                        angular[point] += value * psi[input_, start + point]
                radial_row = block_radial_index[block]
                for point in range(length):
                    flat_index = start + point
                    total[point] += radial[radial_row, flat_index] * angular[point]
            for point in range(length):
                out[output, start + point] += total[point]


@njit(cache=True, fastmath=True, parallel=True)
def _apply_compact_multipole_grouped_complex_kernel(
    out,
    psi,
    radial,
    output_pair,
    group_block_offsets,
    block_pair_offsets,
    block_radial_index,
    input_pair,
    coefficient,
    tile_size,
):
    n_radial = psi.shape[1]
    for group in prange(output_pair.size):
        output = output_pair[group]
        total_real = np.empty(tile_size, dtype=out.real.dtype)
        total_imag = np.empty(tile_size, dtype=out.real.dtype)
        angular_real = np.empty(tile_size, dtype=out.real.dtype)
        angular_imag = np.empty(tile_size, dtype=out.real.dtype)
        for start in range(0, n_radial, tile_size):
            stop = min(start + tile_size, n_radial)
            length = stop - start
            for point in range(length):
                total_real[point] = 0
                total_imag[point] = 0
            for block in range(
                group_block_offsets[group], group_block_offsets[group + 1]
            ):
                pair_start = block_pair_offsets[block]
                pair_stop = block_pair_offsets[block + 1]
                input_ = input_pair[pair_start]
                value = coefficient[pair_start]
                for point in range(length):
                    state = psi[input_, start + point]
                    angular_real[point] = value * state.real
                    angular_imag[point] = value * state.imag
                for pair in range(pair_start + 1, pair_stop):
                    input_ = input_pair[pair]
                    value = coefficient[pair]
                    for point in range(length):
                        state = psi[input_, start + point]
                        angular_real[point] += value * state.real
                        angular_imag[point] += value * state.imag
                radial_row = block_radial_index[block]
                for point in range(length):
                    flat_index = start + point
                    radial_value = radial[radial_row, flat_index]
                    total_real[point] += radial_value * angular_real[point]
                    total_imag[point] += radial_value * angular_imag[point]
            for point in range(length):
                out[output, start + point] += (
                    total_real[point] + 1j * total_imag[point]
                )
