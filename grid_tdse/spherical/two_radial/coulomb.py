"""Radial multipoles for Coulomb interactions between two coordinates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from grid_lib.spherical_coordinates.radial_Coulomb import (
    radial_Coulomb_femdvr_two_grid,
)

from .angular import CoulombMultipoleSpec
from .grids import RadialGrid, require_same_grid_family


RadialCoulombMethod = Literal["poisson", "analytic"]


@dataclass(frozen=True)
class CoulombTerm:
    r"""One term ``charge_product / |r1 - radial_scale * r2|``."""

    radial_scale: float
    charge_product: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.radial_scale):
            raise ValueError("radial_scale must be finite")
        if not np.isfinite(self.charge_product):
            raise ValueError("charge_product must be finite")


@dataclass(frozen=True)
class RadialCoulombMultipoles:
    """Physical, pointwise radial multipoles indexed by retained ``lambda``."""

    values: np.ndarray
    lambdas: tuple[int, ...]
    representation: str

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("radial multipole values must be three-dimensional")
        if self.values.shape[0] != len(self.lambdas):
            raise ValueError("one radial multipole row is required per lambda")
        if len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("retained lambdas must be unique")


def build_radial_coulomb_multipoles(
    grid1: RadialGrid,
    grid2: RadialGrid,
    multipoles: CoulombMultipoleSpec,
    terms: tuple[CoulombTerm, ...],
    *,
    method: RadialCoulombMethod | None = None,
) -> RadialCoulombMultipoles:
    """Build radial multipoles with a Poisson or analytic kernel.

    FEM-DVR defaults to the Poisson construction. Sinc-DVR defaults to the
    analytic multipole expression and does not support the Poisson path. Both
    methods return physical pointwise values: source quadrature weights are
    not stored in the returned array.
    """

    require_same_grid_family(grid1, grid2)
    if not terms:
        raise ValueError("at least one CoulombTerm is required")
    if method is None:
        method = "poisson" if grid1.family == "femdvr" else "analytic"
    if method not in ("poisson", "analytic"):
        raise ValueError("method must be 'poisson' or 'analytic'")

    if method == "poisson":
        if grid1.family != "femdvr":
            raise ValueError("the Poisson Coulomb method requires FEM-DVR")
        weighted = radial_Coulomb_femdvr_two_grid(
            grid1.dvr,
            grid2.dvr,
            multipoles.lambda_max + 1,
            [(term.radial_scale, term.charge_product, 1.0) for term in terms],
        )
        rows = np.asarray(multipoles.lambdas)
        values = weighted[rows] / grid2.weights[None, None, :]
        representation = "femdvr-poisson"
    else:
        values = _analytic_multipoles(grid1.r, grid2.r, multipoles.lambdas, terms)
        representation = "analytic"

    return RadialCoulombMultipoles(
        values=np.ascontiguousarray(values),
        lambdas=multipoles.lambdas,
        representation=representation,
    )


def _analytic_multipoles(
    r1: np.ndarray,
    r2: np.ndarray,
    lambdas: tuple[int, ...],
    terms: tuple[CoulombTerm, ...],
) -> np.ndarray:
    result = np.zeros((len(lambdas), r1.size, r2.size))
    target = r1[:, None]

    for row, lambda_ in enumerate(lambdas):
        prefactor = 4.0 * np.pi / (2 * lambda_ + 1)
        for term in terms:
            scale = term.radial_scale
            if scale == 0.0:
                if lambda_ == 0:
                    result[row] += term.charge_product * prefactor / target
                continue
            source = abs(scale) * r2[None, :]
            r_less = np.minimum(target, source)
            r_greater = np.maximum(target, source)
            parity = -1.0 if scale < 0.0 and lambda_ % 2 else 1.0
            result[row] += (
                term.charge_product
                * parity
                * prefactor
                * r_less**lambda_
                / r_greater ** (lambda_ + 1)
            )

    return result
