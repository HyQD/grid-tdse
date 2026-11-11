"""Serial observables for coupled-angular two-radial states."""

from __future__ import annotations

import numpy as np

from .coupled_angular import CoupledAngularTransform
from .observables import (
    DiatomicIonObservableEvaluator,
    _cos_power_matrix,
    _direction_matrices,
)
from .system import CoupledTwoRadialSystem


class CoupledDiatomicIonObservableEvaluator(DiatomicIonObservableEvaluator):
    """Evaluate non-BO diatomic observables in a coupled angular basis.

    The radial integrations act directly on coupled states.  Small dense
    angular matrices are constructed once by transforming exact projected
    product-basis operators.  No wavefunction transformation occurs while
    sampling a trajectory.
    """

    def __init__(self, system: CoupledTwoRadialSystem, parameters) -> None:
        if not isinstance(system, CoupledTwoRadialSystem):
            raise TypeError("system must be a CoupledTwoRadialSystem")
        for name in (
            "electronic_reduced_charge",
            "internuclear_reduced_charge",
            "has_identical_pair",
        ):
            if not hasattr(parameters, name):
                raise TypeError(f"parameters is missing {name!r}")
        self.system = system
        self.parameters = parameters
        self.transform = CoupledAngularTransform(system.angular_basis)
        self._matrices = self._build_angular_matrices()

    def __call__(self, state: np.ndarray):
        """Return norm-divided observables for a serial coupled state."""

        values = self.system.validate_state(state)
        return self._normalized(self._local_values(
            values,
            self.system.grid1.r,
            self.system.grid1.weights,
            self.system.grid2.r,
            self.system.grid2.weights,
        ))

    def numerators(self, state: np.ndarray):
        """Return raw observable numerators for a serial coupled state."""

        values = self.system.validate_state(state)
        return self._numerators(self._local_values(
            values,
            self.system.grid1.r,
            self.system.grid1.weights,
            self.system.grid2.r,
            self.system.grid2.weights,
        ))

    def angular_populations(self, state: np.ndarray):
        """Return raw populations grouped by ``L``, ``l_1``, and ``l_2``."""

        from .dynamics_sampling import DiatomicIonAngularPopulations

        values = self.system.validate_state(state)
        weights = np.multiply.outer(
            self.system.grid1.weights,
            self.system.grid2.weights,
        )
        channel = np.einsum(
            "cab,cab,ab->c",
            values.conj(),
            values,
            weights,
            optimize=True,
        ).real
        basis = self.system.angular_basis
        by_L = np.zeros(basis.L_max + 1)
        by_l1 = np.zeros(basis.l1_max + 1)
        by_l2 = np.zeros(basis.l2_max + 1)
        for population, (l1, l2, L, _) in zip(
            channel,
            basis.states,
            strict=True,
        ):
            by_L[L] += population
            by_l1[l1] += population
            by_l2[l2] += population
        return DiatomicIonAngularPopulations(
            L=tuple(by_L),
            l1=tuple(by_l1),
            l2=tuple(by_l2),
        )

    def _local_values(self, state, r, w_r, R, w_R) -> np.ndarray:
        values = np.asarray(state)
        expected = (self.system.angular_basis.n_channels, len(r), len(R))
        if values.shape != expected:
            raise ValueError(f"state has shape {values.shape}; expected {expected}")
        population = np.einsum(
            "cab,cab,a,b->c",
            values.conj(),
            values,
            w_r,
            w_R,
            optimize=True,
        )
        norm = np.sum(population)
        cos2_action = self._angular_action(values, self._matrices["cos2_R"])
        results = [
            norm,
            self._radial_expectation(values, r, 1, w_r, w_R),
            self._radial_expectation(values, R, 2, w_r, w_R),
            self._radial_expectation(values, r**2, 1, w_r, w_R),
            self._radial_expectation(values, R**2, 2, w_r, w_R),
            self._inner(
                values,
                self._angular_action(values, self._matrices["rhat_dot_Rhat"])
                * np.multiply.outer(r, R)[None, :, :],
                w_r,
                w_R,
            ),
        ]
        for name in (
            "L_squared",
            "L_z",
            "l1_squared",
            "l2_squared",
            "l1_z",
            "l2_z",
            "cos_R",
        ):
            results.append(
                self._inner(
                    values,
                    self._angular_action(values, self._matrices[name]),
                    w_r,
                    w_R,
                )
            )
        results.append(self._inner(values, cos2_action, w_r, w_R))
        for power in (4, 6, 8):
            results.append(
                self._inner(
                    values,
                    self._angular_action(
                        values,
                        self._matrices[f"cos{power}_R"],
                    ),
                    w_r,
                    w_R,
                )
            )
        results.append(
            self._inner(values, cos2_action * R[None, None, :], w_r, w_R)
        )
        for coordinate, radial in (("r", r), ("R", R)):
            for axis in "xyz":
                action = self._angular_action(
                    values,
                    self._matrices[f"{coordinate}_{axis}"],
                )
                if coordinate == "r":
                    action *= radial[None, :, None]
                else:
                    action *= radial[None, None, :]
                results.append(self._inner(values, action, w_r, w_R))
        for name in ("inversion", "pair_exchange"):
            results.append(
                self._inner(
                    values,
                    self._angular_action(values, self._matrices[name]),
                    w_r,
                    w_R,
                )
            )
        return np.asarray(results, dtype=complex)

    def _build_angular_matrices(self) -> dict[str, np.ndarray]:
        transform = self.transform
        U = transform.coefficients
        states = transform.uncoupled_states
        l1_states = tuple((l1, m1) for l1, m1, _, _ in states)
        l2_states = tuple((l2, m2) for _, _, l2, m2 in states)
        unique1 = tuple(dict.fromkeys(l1_states))
        unique2 = tuple(dict.fromkeys(l2_states))
        index1 = {state: index for index, state in enumerate(unique1)}
        index2 = {state: index for index, state in enumerate(unique2)}
        directions1 = _direction_matrices(unique1)
        directions2 = _direction_matrices(unique2)

        def transformed(matrix):
            result = U.conj().T @ matrix @ U
            result[np.abs(result) < 1.0e-15] = 0.0
            return result

        def diagonal(values):
            return transformed(np.diag(np.asarray(values, dtype=float)))

        coupled_states = self.system.angular_basis.states
        matrices = {
            "L_squared": np.diag([L * (L + 1) for _, _, L, _ in coupled_states]),
            "L_z": np.diag([M for _, _, _, M in coupled_states]),
            "l1_squared": diagonal([l1 * (l1 + 1) for l1, _, _, _ in states]),
            "l2_squared": diagonal([l2 * (l2 + 1) for _, _, l2, _ in states]),
            "l1_z": diagonal([m1 for _, m1, _, _ in states]),
            "l2_z": diagonal([m2 for _, _, _, m2 in states]),
            "inversion": diagonal(
                [(-1.0) ** (l1 + l2) for l1, _, l2, _ in states]
            ),
            "pair_exchange": diagonal([(-1.0) ** l2 for _, _, l2, _ in states]),
        }

        for label, indices, directions, offset in (
            ("r", index1, directions1, 0),
            ("R", index2, directions2, 2),
        ):
            for axis, direction in zip("xyz", directions, strict=True):
                matrix = np.zeros((len(states), len(states)), dtype=complex)
                for output, output_state in enumerate(states):
                    output_coordinate = output_state[offset : offset + 2]
                    output_spectator = output_state[2:4] if offset == 0 else output_state[0:2]
                    for input_, input_state in enumerate(states):
                        input_spectator = input_state[2:4] if offset == 0 else input_state[0:2]
                        if output_spectator != input_spectator:
                            continue
                        matrix[output, input_] = direction[
                            indices[output_coordinate],
                            indices[input_state[offset : offset + 2]],
                        ]
                matrices[f"{label}_{axis}"] = transformed(matrix)

        dot = np.zeros((len(states), len(states)), dtype=complex)
        for output, (l1_out, m1_out, l2_out, m2_out) in enumerate(states):
            output1 = index1[(l1_out, m1_out)]
            output2 = index2[(l2_out, m2_out)]
            for input_, (l1_in, m1_in, l2_in, m2_in) in enumerate(states):
                input1 = index1[(l1_in, m1_in)]
                input2 = index2[(l2_in, m2_in)]
                dot[output, input_] = sum(
                    direction1[output1, input1]
                    * direction2[output2, input2]
                    for direction1, direction2 in zip(
                        directions1,
                        directions2,
                        strict=True,
                    )
                )
        matrices["rhat_dot_Rhat"] = transformed(dot)

        for power in (1, 2, 4, 6, 8):
            coordinate = _cos_power_matrix(unique2, power)
            matrix = np.zeros((len(states), len(states)), dtype=float)
            for output, (l1_out, m1_out, l2_out, m2_out) in enumerate(states):
                for input_, (l1_in, m1_in, l2_in, m2_in) in enumerate(states):
                    if (l1_out, m1_out) != (l1_in, m1_in):
                        continue
                    matrix[output, input_] = coordinate[
                        index2[(l2_out, m2_out)],
                        index2[(l2_in, m2_in)],
                    ]
            name = "cos_R" if power == 1 else f"cos{power}_R"
            matrices[name] = transformed(matrix)
        return matrices

    @staticmethod
    def _angular_action(state, matrix):
        return np.einsum("dc,cab->dab", matrix, state, optimize=True)

    @staticmethod
    def _inner(left, right, w1, w2):
        return np.einsum(
            "cab,cab,a,b->",
            left.conj(),
            right,
            w1,
            w2,
            optimize=True,
        )

    @classmethod
    def _radial_expectation(cls, state, radial, coordinate, w1, w2):
        factor = radial[None, :, None] if coordinate == 1 else radial[None, None, :]
        return cls._inner(state, state * factor, w1, w2)
