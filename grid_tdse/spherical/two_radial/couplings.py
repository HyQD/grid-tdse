"""Sparse angular couplings for a multipole-expanded Coulomb interaction."""

from dataclasses import dataclass

import numpy as np

from grid_lib.spherical_coordinates.angular_momentum import (
    get_y_value,
    get_ybar_value,
)

from .angular import AngularBasisSpec, AngularPairBasis, CoulombMultipoleSpec


@dataclass(frozen=True)
class CoulombCouplings:
    """Compact couplings grouped by output angular-channel pair."""

    output_1: np.ndarray
    output_2: np.ndarray
    offsets: np.ndarray
    input_1: np.ndarray
    input_2: np.ndarray
    radial_index: np.ndarray
    coefficient: np.ndarray
    n_channels_1: int
    n_channels_2: int

    def __post_init__(self) -> None:
        n_groups = self.output_1.size
        n_pairs = self.coefficient.size
        if self.output_2.size != n_groups or self.offsets.size != n_groups + 1:
            raise ValueError("invalid Coulomb output-group arrays")
        pair_sizes = (
            self.input_1.size,
            self.input_2.size,
            self.radial_index.size,
        )
        if pair_sizes != (n_pairs, n_pairs, n_pairs):
            raise ValueError("invalid Coulomb pair arrays")
        if self.offsets[0] != 0 or self.offsets[-1] != n_pairs:
            raise ValueError("Coulomb coupling offsets do not span all pairs")

    @property
    def n_pairs(self) -> int:
        return self.coefficient.size


def build_coulomb_couplings(
    basis1: AngularBasisSpec,
    basis2: AngularBasisSpec,
    multipoles: CoulombMultipoleSpec,
    *,
    angular_pairs: AngularPairBasis | None = None,
    tolerance: float = 1.0e-12,
    max_pairs: int = 20_000_000,
) -> CoulombCouplings:
    """Build nonzero Gaunt-coefficient products without dense angular tensors."""

    if angular_pairs is None:
        angular_pairs = AngularPairBasis(basis1, basis2)
    elif angular_pairs.basis1 != basis1 or angular_pairs.basis2 != basis2:
        raise ValueError("angular_pairs does not match the coordinate bases")

    retained = set(angular_pairs.pairs)
    rows: list[tuple[int, int, int, int, int, float]] = []
    lambda_indices = {value: i for i, value in enumerate(multipoles.lambdas)}

    for lambda_, mu in multipoles.components:
        left = []
        for output, (l_out, m_out) in enumerate(basis1.states):
            m_in = m_out + mu
            for input_, (l_in, candidate_m) in enumerate(basis1.states):
                if candidate_m != m_in:
                    continue
                value = get_ybar_value(
                    lambda_, mu, l_out, m_out, l_in, candidate_m
                )
                if abs(value) > tolerance:
                    left.append((output, input_, value))

        right = []
        for output, (l_out, m_out) in enumerate(basis2.states):
            m_in = m_out - mu
            for input_, (l_in, candidate_m) in enumerate(basis2.states):
                if candidate_m != m_in:
                    continue
                value = get_y_value(
                    lambda_, mu, l_out, m_out, l_in, candidate_m
                )
                if abs(value) > tolerance:
                    right.append((output, input_, value))

        radial_index = lambda_indices[lambda_]
        for out1, in1, value1 in left:
            for out2, in2, value2 in right:
                if (out1, out2) not in retained or (in1, in2) not in retained:
                    continue
                rows.append((out1, out2, in1, in2, radial_index, value1 * value2))
                if len(rows) > max_pairs:
                    raise ValueError(
                        f"more than {max_pairs} Coulomb coupling pairs; "
                        "reduce the angular truncation or raise max_pairs"
                    )

    if not rows:
        empty = np.empty(0, dtype=np.int32)
        return CoulombCouplings(
            empty,
            empty,
            np.array([0], dtype=np.int64),
            empty,
            empty,
            empty,
            np.empty(0),
            basis1.n_channels,
            basis2.n_channels,
        )

    rows.sort(key=lambda row: (row[0], row[1], row[4]))
    values = np.asarray(rows)
    out1 = values[:, 0].astype(np.int32)
    out2 = values[:, 1].astype(np.int32)
    keys = out1.astype(np.int64) * basis2.n_channels + out2
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])

    return CoulombCouplings(
        output_1=np.ascontiguousarray(out1[starts]),
        output_2=np.ascontiguousarray(out2[starts]),
        offsets=np.ascontiguousarray(np.r_[starts, len(rows)].astype(np.int64)),
        input_1=np.ascontiguousarray(values[:, 2].astype(np.int32)),
        input_2=np.ascontiguousarray(values[:, 3].astype(np.int32)),
        radial_index=np.ascontiguousarray(values[:, 4].astype(np.int32)),
        coefficient=np.ascontiguousarray(values[:, 5]),
        n_channels_1=basis1.n_channels,
        n_channels_2=basis2.n_channels,
    )
