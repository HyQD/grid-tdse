"""Observables shared by two-radial spherical representations."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .system import TwoRadialSystem


@dataclass(frozen=True)
class AngularMomentumExpectations:
    """Total orbital-angular-momentum expectations in atomic units."""

    total_l_squared: float
    total_l_z: float


def angular_momentum_expectations(
    system: TwoRadialSystem, state: np.ndarray
) -> AngularMomentumExpectations:
    """Return ``<L_total**2>`` and ``<L_total,z>`` for a state."""

    state = validated_state(system, state)
    norm_squared = system.inner_product(state, state).real
    if norm_squared <= 0.0:
        raise ValueError("state has zero norm")

    action = np.zeros_like(state, dtype=np.result_type(state.dtype, float))
    basis1 = system.basis1
    basis2 = system.basis2
    for i, (l1, m1) in enumerate(basis1.states):
        for j, (l2, m2) in enumerate(basis2.states):
            action[i, j] += (l1 * (l1 + 1) + l2 * (l2 + 1) + 2 * m1 * m2) * state[i, j]

            if m1 < l1 and m2 > -l2:
                out_i = basis1.index(l1, m1 + 1)
                out_j = basis2.index(l2, m2 - 1)
                coefficient = np.sqrt(l1 * (l1 + 1) - m1 * (m1 + 1)) * np.sqrt(
                    l2 * (l2 + 1) - m2 * (m2 - 1)
                )
                action[out_i, out_j] += coefficient * state[i, j]

            if m1 > -l1 and m2 < l2:
                out_i = basis1.index(l1, m1 - 1)
                out_j = basis2.index(l2, m2 + 1)
                coefficient = np.sqrt(l1 * (l1 + 1) - m1 * (m1 - 1)) * np.sqrt(
                    l2 * (l2 + 1) - m2 * (m2 + 1)
                )
                action[out_i, out_j] += coefficient * state[i, j]

    total_l_squared = system.inner_product(state, action).real / norm_squared
    total_l_z = diagonal_angular_expectation(
        system,
        state,
        lambda l1, l2, m1, m2: m1 + m2,
    )
    return AngularMomentumExpectations(
        total_l_squared=float(total_l_squared),
        total_l_z=float(total_l_z),
    )


def diagonal_angular_expectation(
    system: TwoRadialSystem,
    state: np.ndarray,
    value: Callable[[int, int, int, int], float],
) -> float:
    """Evaluate an angular operator diagonal in the uncoupled product basis."""

    state = validated_state(system, state)
    angular_values = np.empty((system.basis1.n_channels, system.basis2.n_channels))
    for i, (l1, m1) in enumerate(system.basis1.states):
        for j, (l2, m2) in enumerate(system.basis2.states):
            angular_values[i, j] = value(l1, l2, m1, m2)
    norm_squared = system.inner_product(state, state).real
    if norm_squared <= 0.0:
        raise ValueError("state has zero norm")
    action = angular_values[:, :, None, None] * state
    expectation = system.inner_product(state, action)
    return float(expectation.real / norm_squared)


def validated_state(system: TwoRadialSystem, state: np.ndarray) -> np.ndarray:
    """Validate and return a two-radial state array without copying it."""

    state = np.asarray(state)
    if state.shape != system.psi_shape:
        raise ValueError(f"state has shape {state.shape}; expected {system.psi_shape}")
    if not np.issubdtype(state.dtype, np.number):
        raise TypeError("state must contain numeric values")
    if not np.isfinite(state).all():
        raise ValueError("state must contain only finite values")
    return state
