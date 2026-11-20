"""Gauge transformations for spherical one-radial states."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import numpy as np
from scipy.special import gammaln, lpmv

from .hamiltonian import OneElectronDiatomicHamiltonian


FieldGauge = Literal["length", "velocity"]


def transform_axial_gauge_state(
    hamiltonian: OneElectronDiatomicHamiltonian,
    state,
    vector_potential: Real,
    *,
    m: int = 0,
    source_gauge: FieldGauge,
    target_gauge: FieldGauge,
    quadrature_order: int | None = None,
) -> np.ndarray:
    r"""Project an axial length/velocity gauge transformation.

    In the dipole approximation, the spatial gauge relation is

    .. math::

        \Psi_L = \exp[-i q_e A_z(t) z] \Psi_V.

    This routine applies the phase on a Gauss--Legendre angular quadrature
    and projects the result back into the finite fixed-``m`` spherical-
    harmonic space. The projection is deliberate: a finite angular space is
    not closed under the exact gauge transformation, so the transformed norm
    need not be preserved. That norm defect is itself a useful angular-
    truncation diagnostic.

    The velocity-gauge ``A_z**2`` term omitted by the dynamics contributes
    only a spatially uniform phase. This routine applies the spatial gauge
    phase and does not reconstruct that global phase.
    """

    if not isinstance(hamiltonian, OneElectronDiatomicHamiltonian):
        raise TypeError(
            "hamiltonian must be a OneElectronDiatomicHamiltonian"
        )
    values = hamiltonian.validate_fixed_m_state(state, m)
    source = _gauge(source_gauge, "source_gauge")
    target = _gauge(target_gauge, "target_gauge")
    amplitude = _finite_real(vector_potential, "vector_potential")
    if source == target or amplitude == 0.0:
        return np.array(values, dtype=np.result_type(values, complex), copy=True)

    n_channels = values.shape[-1]
    order = _quadrature_order(quadrature_order, n_channels)
    cosine, weights = np.polynomial.legendre.leggauss(order)
    polar_harmonics = _normalized_associated_legendre(
        cosine,
        abs(int(m)),
        hamiltonian.l_max,
    )

    direction = 1.0 if (source, target) == ("velocity", "length") else -1.0
    coefficient = (
        -direction * hamiltonian.system.electron_charge * amplitude
    )
    phases = np.exp(
        1j * coefficient * hamiltonian.r[:, None] * cosine[None, :]
    )
    matrices = np.einsum(
        "lx,rx,mx,x->rlm",
        polar_harmonics,
        phases,
        polar_harmonics,
        weights,
        optimize=True,
    )
    return np.einsum("rij,...rj->...ri", matrices, values, optimize=True)


def _gauge(value, name: str) -> FieldGauge:
    if value not in {"length", "velocity"}:
        raise ValueError(f"{name} must be 'length' or 'velocity'")
    return value


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quadrature_order(value, n_channels: int) -> int:
    minimum = n_channels
    if value is None:
        return max(64, 4 * n_channels)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("quadrature_order must be an integer or None")
    result = int(value)
    if result < minimum:
        raise ValueError(
            f"quadrature_order must be at least {minimum} for this m block"
        )
    return result


def _normalized_associated_legendre(cosine, m_abs: int, l_max: int):
    l_values = np.arange(m_abs, l_max + 1)
    logarithmic_norm = 0.5 * (
        np.log(2.0 * l_values + 1.0)
        - np.log(2.0)
        + gammaln(l_values - m_abs + 1.0)
        - gammaln(l_values + m_abs + 1.0)
    )
    return np.asarray(
        [
            np.exp(norm) * lpmv(m_abs, int(l_value), cosine)
            for l_value, norm in zip(l_values, logarithmic_norm)
        ]
    )


__all__ = ["transform_axial_gauge_state"]
