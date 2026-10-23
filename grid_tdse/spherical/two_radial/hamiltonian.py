"""Optimized Hamiltonian action for two-radial wavefunctions."""

import numpy as np
from numba import njit, prange

from .couplings import CoulombCouplings
from .coulomb import RadialCoulombMultipoles


class TwoRadialHamiltonian:
    """Apply one-body blocks and sparse multipole couplings to a wavefunction."""

    def __init__(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        radial_multipoles: RadialCoulombMultipoles,
        couplings: CoulombCouplings,
        *,
        tile_size: int = 2048,
    ) -> None:
        self.h1 = np.ascontiguousarray(h1)
        self.h2 = np.ascontiguousarray(h2)
        self.h2_transpose = np.ascontiguousarray(h2.transpose(0, 2, 1))[None]
        self.radial_multipoles = radial_multipoles
        self.couplings = couplings
        self.tile_size = int(tile_size)
        if self.tile_size < 1:
            raise ValueError("tile_size must be positive")
        self._validate_shapes()
        self._radial_flat = np.ascontiguousarray(
            radial_multipoles.values.reshape(radial_multipoles.values.shape[0], -1)
        )

    @property
    def psi_shape(self) -> tuple[int, int, int, int]:
        return (
            self.h1.shape[0],
            self.h2.shape[0],
            self.h1.shape[1],
            self.h2.shape[1],
        )

    def __call__(self, psi: np.ndarray) -> np.ndarray:
        psi = np.asarray(psi)
        if psi.shape != self.psi_shape:
            raise ValueError(f"psi has shape {psi.shape}; expected {self.psi_shape}")
        psi = np.ascontiguousarray(psi)
        result = np.matmul(self.h1[:, None], psi)
        result += np.matmul(psi, self.h2_transpose)
        return self.apply_coulomb(psi, out=result)

    def apply_coulomb(
        self, psi: np.ndarray, *, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Apply the optimized, output-blocked Coulomb action.

        This is the production path descended from the Bodenstein kernel. The
        dense ``y`` and ``y_bar`` tensors are never constructed.
        """

        psi = np.asarray(psi)
        if psi.shape != self.psi_shape:
            raise ValueError(f"psi has shape {psi.shape}; expected {self.psi_shape}")
        psi = np.ascontiguousarray(psi)
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
