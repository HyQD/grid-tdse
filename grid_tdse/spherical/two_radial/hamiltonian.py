"""Optimized Hamiltonian action for two-radial wavefunctions."""

import numpy as np
from numba import njit, prange

from .angular import AngularPairBasis
from .couplings import CoulombCouplings
from .coulomb import RadialCoulombMultipoles
from .grids import RadialGrid
from .system import RadialOperatorMode


class TwoRadialHamiltonian:
    """Apply one-body blocks and sparse multipole couplings to a wavefunction."""

    def __init__(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
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
    ) -> None:
        self.h1 = np.ascontiguousarray(h1)
        self.h2 = np.ascontiguousarray(h2)
        self.h2_transpose = np.ascontiguousarray(h2.transpose(0, 2, 1))[None]
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
        self._validate_shapes()
        self._radial_flat = np.ascontiguousarray(
            radial_multipoles.values.reshape(radial_multipoles.values.shape[0], -1)
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
            self._h1_pairs = np.ascontiguousarray(self.h1[self._pair_channels1])
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

    @property
    def _uses_compact_storage(self) -> bool:
        return self.angular_pairs is not None and self.angular_pairs.is_compact

    @property
    def psi_shape(self) -> tuple[int, ...]:
        if self._uses_compact_storage:
            return (
                self.angular_pairs.n_pairs,
                self.h1.shape[1],
                self.h2.shape[1],
            )
        return (
            self.h1.shape[0],
            self.h2.shape[0],
            self.h1.shape[1],
            self.h2.shape[1],
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

        result = self.kinetic_prefactor1 * self.grid1.apply_local_second_derivative(
            psi, axis=-2
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

        result = self.kinetic_prefactor2 * self.grid2.apply_local_second_derivative(
            psi, axis=-1
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
        dtype = np.result_type(psi.dtype, self._radial_flat.dtype)
        if out is None:
            out = np.zeros(self.psi_shape, dtype=dtype)
        elif out.shape != self.psi_shape:
            raise ValueError(f"out has shape {out.shape}; expected {self.psi_shape}")
        elif not np.can_cast(dtype, out.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot accumulate {dtype} Coulomb action in {out.dtype} output"
            )
        elif not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")

        if self._uses_compact_storage:
            output = out.reshape(self.psi_shape[0], -1)
            wavefunction = psi.reshape(self.psi_shape[0], -1)
            output_pair, offsets, input_pair, radial_index, coefficient = (
                self._compact_couplings
            )
            _apply_compact_coulomb_kernel(
                output,
                wavefunction,
                self._radial_flat,
                output_pair,
                offsets,
                input_pair,
                radial_index,
                coefficient,
                self.tile_size,
            )
        else:
            output = out.reshape(self.psi_shape[0], self.psi_shape[1], -1)
            wavefunction = psi.reshape(self.psi_shape[0], self.psi_shape[1], -1)
            c = self.couplings
            _apply_coulomb_kernel(
                output,
                wavefunction,
                self._radial_flat,
                c.output_1,
                c.output_2,
                c.offsets,
                c.input_1,
                c.input_2,
                c.radial_index,
                c.coefficient,
                self.tile_size,
            )
        return output.reshape(self.psi_shape)

    def _validate_shapes(self) -> None:
        if self.radial_operator_mode not in {"global", "element-local"}:
            raise ValueError(
                "radial_operator_mode must be 'global' or 'element-local'"
            )
        if self.h1.ndim != 3 or self.h1.shape[1] != self.h1.shape[2]:
            raise ValueError("h1 must have shape (channels1, r1, r1)")
        if self.h2.ndim != 3 or self.h2.shape[1] != self.h2.shape[2]:
            raise ValueError("h2 must have shape (channels2, r2, r2)")
        c = self.couplings
        if self.h1.shape[0] != c.n_channels_1 or self.h2.shape[0] != c.n_channels_2:
            raise ValueError("one-body channel counts do not match Coulomb couplings")
        expected = (
            len(self.radial_multipoles.lambdas),
            self.h1.shape[1],
            self.h2.shape[1],
        )
        if self.radial_multipoles.values.shape != expected:
            raise ValueError(
                "radial multipoles have shape "
                f"{self.radial_multipoles.values.shape}; expected {expected}"
            )
        if c.n_pairs and c.radial_index.max() >= expected[0]:
            raise ValueError("Coulomb coupling references a missing radial multipole")
        if self.radial_operator_mode == "element-local":
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
        expected1 = (self.h1.shape[0], self.h1.shape[1])
        expected2 = (self.h2.shape[0], self.h2.shape[1])
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
        psi = np.asarray(psi)
        if psi.shape != self.psi_shape:
            raise ValueError(f"psi has shape {psi.shape}; expected {self.psi_shape}")
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
