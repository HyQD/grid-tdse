"""Field-free one-body actions for coupled two-radial representations."""

from __future__ import annotations

import numpy as np

from .system import (
    CoupledTwoRadialSystem,
    RadialOperatorMode,
    RadialPotential,
)


class CoupledTwoRadialOneBodyHamiltonian:
    r"""Apply radial kinetic and angular-diagonal field-free terms.

    The production ``"element-local"`` path uses the FEMDVR derivative action
    owned by each radial grid.  The ``"global"`` path retains dense matrices
    only on this validation object and serves as a small-system oracle.

    The action is diagonal in coupled channels.  Its centrifugal factors are
    determined by ``l_1`` and ``l_2`` and are independent of ``L`` and ``M``.
    """

    def __init__(
        self,
        representation: CoupledTwoRadialSystem,
        *,
        radial_operator_mode: RadialOperatorMode = "element-local",
    ) -> None:
        if not isinstance(representation, CoupledTwoRadialSystem):
            raise TypeError("representation must be a CoupledTwoRadialSystem")
        if radial_operator_mode not in {"global", "element-local"}:
            raise ValueError(
                "radial_operator_mode must be 'global' or 'element-local'"
            )
        if (
            radial_operator_mode == "element-local"
            and representation.grid1.family != "femdvr"
        ):
            raise ValueError(
                "element-local radial operators require FEMDVR grids"
            )

        self.representation = representation
        self.radial_operator_mode = radial_operator_mode
        states = representation.angular_basis.states
        self.l1 = _readonly([l1 for l1, _, _, _ in states], dtype=np.int32)
        self.l2 = _readonly([l2 for _, l2, _, _ in states], dtype=np.int32)
        self.centrifugal1 = _readonly(
            -representation.coordinate1.kinetic_prefactor
            * self.l1[:, None]
            * (self.l1[:, None] + 1)
            / representation.grid1.r[None, :] ** 2
        )
        self.centrifugal2 = _readonly(
            -representation.coordinate2.kinetic_prefactor
            * self.l2[:, None]
            * (self.l2[:, None] + 1)
            / representation.grid2.r[None, :] ** 2
        )
        self.radial_potential1 = _readonly(
            _radial_potential_values(
                representation.coordinate1,
                representation.grid1,
                coordinate_index=1,
            )
        )
        self.radial_potential2 = _readonly(
            _radial_potential_values(
                representation.coordinate2,
                representation.grid2,
                coordinate_index=2,
            )
        )
        self.diagonal1 = _readonly(
            self.centrifugal1 + self.radial_potential1[None, :]
        )
        self.diagonal2 = _readonly(
            self.centrifugal2 + self.radial_potential2[None, :]
        )

        self._kinetic1 = None
        self._kinetic2_transpose = None
        if radial_operator_mode == "global":
            self._kinetic1 = _readonly(
                [
                    representation.coordinate1.kinetic_prefactor
                    * representation.grid1.second_derivative(int(l1))
                    for l1 in self.l1
                ]
            )
            kinetic2 = _readonly(
                [
                    representation.coordinate2.kinetic_prefactor
                    * representation.grid2.second_derivative(int(l2))
                    for l2 in self.l2
                ]
            )
            self._kinetic2_transpose = _readonly(
                kinetic2.transpose(0, 2, 1)
            )

    @property
    def psi_shape(self) -> tuple[int, int, int]:
        return self.representation.psi_shape

    @property
    def has_global_radial_operators(self) -> bool:
        """Whether the validation object retains dense kinetic matrices."""

        return self._kinetic1 is not None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Apply all field-free one-body terms."""

        state = self._validated_state(state)
        result = self.apply_coordinate1(state)
        result += self.apply_coordinate2(state)
        return result

    def apply_coordinate1(self, state: np.ndarray) -> np.ndarray:
        """Apply electronic-coordinate kinetic and diagonal terms."""

        state = self._validated_state(state)
        result = self.apply_kinetic_coordinate1(state)
        result += self.diagonal1[:, :, None] * state
        return result

    def apply_coordinate2(self, state: np.ndarray) -> np.ndarray:
        """Apply internuclear-coordinate kinetic and diagonal terms."""

        state = self._validated_state(state)
        result = self.apply_kinetic_coordinate2(state)
        result += self.diagonal2[:, None, :] * state
        return result

    def apply_kinetic_coordinate1(self, state: np.ndarray) -> np.ndarray:
        """Apply the first-coordinate radial kinetic derivative."""

        state = self._validated_state(state)
        if self.radial_operator_mode == "global":
            return np.matmul(self._kinetic1, state)
        return self.representation.coordinate1.kinetic_prefactor * (
            self.representation.grid1.apply_local_second_derivative(
                state,
                axis=-2,
            )
        )

    def apply_kinetic_coordinate2(self, state: np.ndarray) -> np.ndarray:
        """Apply the second-coordinate radial kinetic derivative."""

        state = self._validated_state(state)
        if self.radial_operator_mode == "global":
            return np.matmul(state, self._kinetic2_transpose)
        return self.representation.coordinate2.kinetic_prefactor * (
            self.representation.grid2.apply_local_second_derivative(
                state,
                axis=-1,
            )
        )

    def apply_centrifugal_coordinate1(self, state: np.ndarray) -> np.ndarray:
        """Apply the electronic centrifugal term."""

        state = self._validated_state(state)
        return self.centrifugal1[:, :, None] * state

    def apply_centrifugal_coordinate2(self, state: np.ndarray) -> np.ndarray:
        """Apply the internuclear centrifugal term."""

        state = self._validated_state(state)
        return self.centrifugal2[:, None, :] * state

    def apply_radial_potential_coordinate1(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """Apply angular-independent first-coordinate radial potentials."""

        state = self._validated_state(state)
        return self.radial_potential1[None, :, None] * state

    def apply_radial_potential_coordinate2(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """Apply angular-independent second-coordinate radial potentials."""

        state = self._validated_state(state)
        return self.radial_potential2[None, None, :] * state

    def _validated_state(self, state: np.ndarray) -> np.ndarray:
        return self.representation.validate_state(state)


def _radial_potential_values(
    coordinate_spec,
    grid,
    *,
    coordinate_index: int,
) -> np.ndarray:
    result = np.zeros(grid.n_points)
    for term in coordinate_spec.one_body_terms:
        if not isinstance(term, RadialPotential):
            raise TypeError(
                "coupled one-body coordinate "
                f"{coordinate_index} requires angular-independent RadialPotential "
                "terms"
            )
        result += term.diagonal(0, 0, grid)
    return result


def _readonly(values, *, dtype=None) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    result.setflags(write=False)
    return result
