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


@dataclass(frozen=True)
class DiatomicIonObservables:
    r"""Normalized non-BO diatomic observables in atomic units.

    The two coordinate dipoles are the charge-weighted Jacobi-coordinate
    contributions

    ``dipole_r = xi_e <r_vector>`` and
    ``dipole_R = xi_R <R_vector>``.

    ``dipole_total`` is their vector sum.  ``pair_exchange`` is ``None`` when
    the particles defining the internuclear coordinate are distinguishable.
    """

    r: float
    R: float
    total_l_squared: float
    total_l_z: float
    l_r_squared: float
    l_R_squared: float
    l_r_z: float
    l_R_z: float
    cos_theta_R: float
    cos2_theta_R: float
    dipole_r: tuple[float, float, float]
    dipole_R: tuple[float, float, float]
    dipole_total: tuple[float, float, float]
    inversion: float
    pair_exchange: float | None


class DiatomicIonObservableEvaluator:
    """Evaluate serial or distributed non-BO diatomic observables."""

    _NAMES = (
        "norm",
        "r",
        "R",
        "total_l_squared",
        "total_l_z",
        "l_r_squared",
        "l_R_squared",
        "l_r_z",
        "l_R_z",
        "cos_theta_R",
        "cos2_theta_R",
        "r_x",
        "r_y",
        "r_z",
        "R_x",
        "R_y",
        "R_z",
        "inversion",
        "pair_exchange",
    )

    def __init__(self, system: TwoRadialSystem, parameters) -> None:
        if not isinstance(system, TwoRadialSystem):
            raise TypeError("system must be a TwoRadialSystem")
        for name in (
            "electronic_reduced_charge",
            "internuclear_reduced_charge",
            "has_identical_pair",
        ):
            if not hasattr(parameters, name):
                raise TypeError(f"parameters is missing {name!r}")
        self.system = system
        self.parameters = parameters
        self._directions1 = _direction_matrices(system.basis1.states)
        self._directions2 = _direction_matrices(system.basis2.states)
        self._cos2_R = _cos2_matrix(system.basis2.states)

    def __call__(self, state: np.ndarray) -> DiatomicIonObservables:
        """Return observables for a serial state."""

        values = validated_state(self.system, state)
        local = self._local_values(
            values,
            self.system.grid1.r,
            self.system.grid1.weights,
            self.system.grid2.r,
            self.system.grid2.weights,
        )
        return self._normalized(local)

    def distributed(self, state, reductions) -> DiatomicIonObservables:
        """Collectively return observables for a one-coordinate MPI state."""

        from .distributed_hamiltonian import TwoRadialDistributedReductions
        from .distributed_state import DistributedTwoRadialState

        if not isinstance(state, DistributedTwoRadialState):
            raise TypeError("state must be a DistributedTwoRadialState")
        if state.layout.representation is not self.system:
            raise ValueError("state uses a different two-radial representation")
        if not isinstance(reductions, TwoRadialDistributedReductions):
            raise TypeError("reductions must be TwoRadialDistributedReductions")
        local = self._local_values(
            state.coefficients,
            state.owned_r,
            state.owned_r_weights,
            state.owned_R,
            state.owned_R_weights,
        )
        global_values = reductions.global_sum(local, rank=state.rank)
        return self._normalized(global_values)

    def _local_values(self, state, r, w_r, R, w_R) -> np.ndarray:
        values = np.asarray(state)
        r = np.asarray(r)
        R = np.asarray(R)
        w_r = np.asarray(w_r)
        w_R = np.asarray(w_R)
        expected_angular = (
            (self.system.angular_pairs.n_pairs,)
            if self.system.uses_compact_angular_storage
            else (
                self.system.basis1.n_channels,
                self.system.basis2.n_channels,
            )
        )
        expected = expected_angular + (r.size, R.size)
        if values.shape != expected:
            raise ValueError(f"state has shape {values.shape}; expected {expected}")

        population = _angular_population(
            self.system,
            values,
            w_r,
            w_R,
        )
        pair_values = _pair_diagonals(self.system)
        norm = np.sum(population)
        results = [
            norm,
            _radial_local_expectation(values, r, 1, w_r, w_R),
            _radial_local_expectation(values, R, 2, w_r, w_R),
            _local_inner_product(
                values,
                _total_l_squared_action(self.system, values),
                w_r,
                w_R,
            ),
            np.sum(population * pair_values["total_l_z"]),
            np.sum(population * pair_values["l_r_squared"]),
            np.sum(population * pair_values["l_R_squared"]),
            np.sum(population * pair_values["l_r_z"]),
            np.sum(population * pair_values["l_R_z"]),
            _local_inner_product(
                values,
                _coordinate_angular_action(
                    self.system,
                    values,
                    2,
                    self._directions2[2],
                ),
                w_r,
                w_R,
            ),
            _local_inner_product(
                values,
                _coordinate_angular_action(
                    self.system,
                    values,
                    2,
                    self._cos2_R,
                ),
                w_r,
                w_R,
            ),
        ]
        for coordinate, radial, matrices in (
            (1, r, self._directions1),
            (2, R, self._directions2),
        ):
            for matrix in matrices:
                action = _coordinate_angular_action(
                    self.system,
                    values,
                    coordinate,
                    matrix,
                )
                if coordinate == 1:
                    action *= radial.reshape((1,) * (action.ndim - 2) + (-1, 1))
                else:
                    action *= radial.reshape((1,) * (action.ndim - 1) + (-1,))
                results.append(
                    _local_inner_product(values, action, w_r, w_R)
                )
        results.extend(
            (
                np.sum(population * pair_values["inversion"]),
                np.sum(population * pair_values["pair_exchange"]),
            )
        )
        return np.asarray(results, dtype=complex)

    def _normalized(self, values) -> DiatomicIonObservables:
        values = np.asarray(values)
        if values.shape != (len(self._NAMES),):
            raise ValueError("observable reduction returned an invalid shape")
        norm = _real_scalar(values[0], "norm")
        if norm <= 0.0:
            raise ValueError("state has zero norm")
        normalized = {
            name: _real_scalar(value / norm, name)
            for name, value in zip(self._NAMES[1:], values[1:])
        }
        xi_r = float(self.parameters.electronic_reduced_charge)
        xi_R = float(self.parameters.internuclear_reduced_charge)
        dipole_r = tuple(xi_r * normalized[f"r_{axis}"] for axis in "xyz")
        dipole_R = tuple(xi_R * normalized[f"R_{axis}"] for axis in "xyz")
        dipole_total = tuple(
            electronic + internuclear
            for electronic, internuclear in zip(dipole_r, dipole_R)
        )
        return DiatomicIonObservables(
            r=normalized["r"],
            R=normalized["R"],
            total_l_squared=normalized["total_l_squared"],
            total_l_z=normalized["total_l_z"],
            l_r_squared=normalized["l_r_squared"],
            l_R_squared=normalized["l_R_squared"],
            l_r_z=normalized["l_r_z"],
            l_R_z=normalized["l_R_z"],
            cos_theta_R=normalized["cos_theta_R"],
            cos2_theta_R=normalized["cos2_theta_R"],
            dipole_r=dipole_r,
            dipole_R=dipole_R,
            dipole_total=dipole_total,
            inversion=normalized["inversion"],
            pair_exchange=(
                normalized["pair_exchange"]
                if self.parameters.has_identical_pair
                else None
            ),
        )


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
    for pair, (i, j) in enumerate(system.angular_pairs.pairs):
        l1, m1 = basis1.states[i]
        l2, m2 = basis2.states[j]
        block = (
            state[pair] if system.uses_compact_angular_storage else state[i, j]
        )
        system.add_to_angular_block(
            action,
            i,
            j,
            (l1 * (l1 + 1) + l2 * (l2 + 1) + 2 * m1 * m2) * block,
        )

        if m1 < l1 and m2 > -l2:
            out_i = basis1.index(l1, m1 + 1)
            out_j = basis2.index(l2, m2 - 1)
            coefficient = np.sqrt(l1 * (l1 + 1) - m1 * (m1 + 1)) * np.sqrt(
                l2 * (l2 + 1) - m2 * (m2 - 1)
            )
            system.add_to_angular_block(
                action, out_i, out_j, coefficient * block
            )

        if m1 > -l1 and m2 < l2:
            out_i = basis1.index(l1, m1 - 1)
            out_j = basis2.index(l2, m2 + 1)
            coefficient = np.sqrt(l1 * (l1 + 1) - m1 * (m1 - 1)) * np.sqrt(
                l2 * (l2 + 1) - m2 * (m2 + 1)
            )
            system.add_to_angular_block(
                action, out_i, out_j, coefficient * block
            )

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
    norm_squared = system.inner_product(state, state).real
    if norm_squared <= 0.0:
        raise ValueError("state has zero norm")
    if system.uses_compact_angular_storage:
        angular_values = np.asarray(
            [
                value(
                    system.basis1.states[i][0],
                    system.basis2.states[j][0],
                    system.basis1.states[i][1],
                    system.basis2.states[j][1],
                )
                for i, j in system.angular_pairs.pairs
            ]
        )
        action = angular_values[:, None, None] * state
    else:
        angular_values = np.empty(
            (system.basis1.n_channels, system.basis2.n_channels)
        )
        for i, (l1, m1) in enumerate(system.basis1.states):
            for j, (l2, m2) in enumerate(system.basis2.states):
                angular_values[i, j] = value(l1, l2, m1, m2)
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


def _angular_population(system, state, w1, w2):
    if system.uses_compact_angular_storage:
        return np.einsum(
            "pab,pab,a,b->p",
            state.conj(),
            state,
            w1,
            w2,
            optimize=True,
        )
    return np.einsum(
        "ijab,ijab,a,b->ij",
        state.conj(),
        state,
        w1,
        w2,
        optimize=True,
    )


def _pair_diagonals(system):
    values = {
        name: np.empty(
            (system.angular_pairs.n_pairs,)
            if system.uses_compact_angular_storage
            else (system.basis1.n_channels, system.basis2.n_channels),
            dtype=float,
        )
        for name in (
            "total_l_z",
            "l_r_squared",
            "l_R_squared",
            "l_r_z",
            "l_R_z",
            "inversion",
            "pair_exchange",
        )
    }
    for pair, (i, j) in enumerate(system.angular_pairs.pairs):
        l_r, m_r = system.basis1.states[i]
        l_R, m_R = system.basis2.states[j]
        index = pair if system.uses_compact_angular_storage else (i, j)
        values["total_l_z"][index] = m_r + m_R
        values["l_r_squared"][index] = l_r * (l_r + 1)
        values["l_R_squared"][index] = l_R * (l_R + 1)
        values["l_r_z"][index] = m_r
        values["l_R_z"][index] = m_R
        values["inversion"][index] = -1.0 if (l_r + l_R) % 2 else 1.0
        values["pair_exchange"][index] = -1.0 if l_R % 2 else 1.0
    return values


def _radial_local_expectation(state, radial, coordinate, w1, w2):
    shape = [1] * state.ndim
    shape[-2 if coordinate == 1 else -1] = radial.size
    return _local_inner_product(
        state,
        state * radial.reshape(shape),
        w1,
        w2,
    )


def _local_inner_product(left, right, w1, w2):
    return np.einsum(
        "...ab,...ab,a,b->",
        left.conj(),
        right,
        w1,
        w2,
        optimize=True,
    )


def _coordinate_angular_action(system, state, coordinate, matrix):
    result = np.zeros_like(state, dtype=np.result_type(state.dtype, matrix.dtype))
    for input_pair, (i, j) in enumerate(system.angular_pairs.pairs):
        input_channel = i if coordinate == 1 else j
        block = state[input_pair] if system.uses_compact_angular_storage else state[i, j]
        for output_channel in np.flatnonzero(matrix[:, input_channel]):
            if coordinate == 1:
                output_i, output_j = int(output_channel), j
            else:
                output_i, output_j = i, int(output_channel)
            coefficient = matrix[output_channel, input_channel]
            if system.uses_compact_angular_storage:
                output_pair = system.angular_pairs.get(output_i, output_j)
                if output_pair is not None:
                    result[output_pair] += coefficient * block
            else:
                result[output_i, output_j] += coefficient * block
    return result


def _total_l_squared_action(system, state):
    result = np.zeros_like(state, dtype=np.result_type(state.dtype, float))
    indices1 = {
        angular_state: index
        for index, angular_state in enumerate(system.basis1.states)
    }
    indices2 = {
        angular_state: index
        for index, angular_state in enumerate(system.basis2.states)
    }
    for pair, (i, j) in enumerate(system.angular_pairs.pairs):
        l1, m1 = system.basis1.states[i]
        l2, m2 = system.basis2.states[j]
        block = state[pair] if system.uses_compact_angular_storage else state[i, j]
        _add_angular_block(
            system,
            result,
            i,
            j,
            (l1 * (l1 + 1) + l2 * (l2 + 1) + 2 * m1 * m2) * block,
        )
        if m1 < l1 and m2 > -l2:
            output_i = indices1.get((l1, m1 + 1))
            output_j = indices2.get((l2, m2 - 1))
            if output_i is not None and output_j is not None:
                coefficient = np.sqrt(
                    l1 * (l1 + 1) - m1 * (m1 + 1)
                ) * np.sqrt(l2 * (l2 + 1) - m2 * (m2 - 1))
                _add_angular_block(
                    system, result, output_i, output_j, coefficient * block
                )
        if m1 > -l1 and m2 < l2:
            output_i = indices1.get((l1, m1 - 1))
            output_j = indices2.get((l2, m2 + 1))
            if output_i is not None and output_j is not None:
                coefficient = np.sqrt(
                    l1 * (l1 + 1) - m1 * (m1 - 1)
                ) * np.sqrt(l2 * (l2 + 1) - m2 * (m2 + 1))
                _add_angular_block(
                    system, result, output_i, output_j, coefficient * block
                )
    return result


def _add_angular_block(system, result, i, j, values):
    if system.uses_compact_angular_storage:
        pair = system.angular_pairs.get(i, j)
        if pair is not None:
            result[pair] += values
    else:
        result[i, j] += values


def _direction_matrices(states):
    size = len(states)
    matrices = [np.zeros((size, size), dtype=complex) for _ in range(3)]
    indices = {state: index for index, state in enumerate(states)}
    for input_, (l_value, m_value) in enumerate(states):
        for output_, (l_out, m_out) in enumerate(states):
            matrices[0][output_, input_] = _sin_theta_cos_phi(
                l_out, m_out, l_value, m_value
            )
            matrices[1][output_, input_] = _sin_theta_sin_phi(
                l_out, m_out, l_value, m_value
            )
            matrices[2][output_, input_] = _a_lm(l_value, m_value) * (
                indices.get((l_value + 1, m_value)) == output_
            ) + _a_lm(l_value - 1, m_value) * (
                indices.get((l_value - 1, m_value)) == output_
            )
    return tuple(matrices)


def _cos2_matrix(states):
    result = np.zeros((len(states), len(states)), dtype=float)
    indices = {state: index for index, state in enumerate(states)}
    for input_, (l_value, m_value) in enumerate(states):
        coefficients = {
            (l_value + 2, m_value): _a_lm(l_value, m_value)
            * _a_lm(l_value + 1, m_value),
            (l_value, m_value): _a_lm(l_value, m_value) ** 2
            + _a_lm(l_value - 1, m_value) ** 2,
            (l_value - 2, m_value): _a_lm(l_value - 1, m_value)
            * _a_lm(l_value - 2, m_value),
        }
        for state, coefficient in coefficients.items():
            output = indices.get(state)
            if output is not None:
                result[output, input_] += coefficient
    return result


def _a_lm(l_value, m_value):
    if l_value < 0 or abs(m_value) > l_value:
        return 0.0
    return np.sqrt(
        ((l_value + 1) ** 2 - m_value**2)
        / ((2 * l_value + 1) * (2 * l_value + 3))
    )


def _b_lm(l_value, m_value):
    if l_value < 0 or abs(m_value) > l_value:
        return 0.0
    return np.sqrt(
        ((l_value + m_value + 1) * (l_value + m_value + 2))
        / ((2 * l_value + 1) * (2 * l_value + 3))
    )


def _sin_theta_cos_phi(l1, m1, l2, m2):
    value = _b_lm(l2 - 1, -m2 - 1) * (l1 == l2 - 1) * (m1 == m2 + 1)
    value -= _b_lm(l2, m2) * (l1 == l2 + 1) * (m1 == m2 + 1)
    value -= _b_lm(l2 - 1, m2 - 1) * (l1 == l2 - 1) * (m1 == m2 - 1)
    value += _b_lm(l2, -m2) * (l1 == l2 + 1) * (m1 == m2 - 1)
    return 0.5 * value


def _sin_theta_sin_phi(l1, m1, l2, m2):
    value = _b_lm(l2 - 1, -m2 - 1) * (l1 == l2 - 1) * (m1 == m2 + 1)
    value -= _b_lm(l2, m2) * (l1 == l2 + 1) * (m1 == m2 + 1)
    value += _b_lm(l2 - 1, m2 - 1) * (l1 == l2 - 1) * (m1 == m2 - 1)
    value -= _b_lm(l2, -m2) * (l1 == l2 + 1) * (m1 == m2 - 1)
    return -0.5j * value


def _real_scalar(value, name):
    value = complex(value)
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > 1.0e-10 * scale:
        raise ValueError(f"{name} has a significant imaginary part: {value.imag}")
    return float(value.real)
