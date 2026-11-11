"""Field-free Hamiltonians for coupled two-radial representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .absorbing_boundaries import TwoRadialAbsorbingBoundaries
from .coupled_coulomb import (
    CoupledCoulombCoefficientMethod,
    CoupledCoulombCouplings,
    CoupledTwoRadialCoulombHamiltonian,
)
from .system import (
    CoupledTwoRadialSystem,
    RadialOperatorMode,
    RadialPotential,
)


@dataclass(frozen=True)
class CoupledHamiltonianComponents:
    """Separately evaluated coupled field-free Hamiltonian components."""

    coordinate1: np.ndarray
    coordinate2: np.ndarray
    coulomb: np.ndarray
    cap: np.ndarray | None = None


class CoupledTwoRadialHamiltonian:
    """Compose the coupled one-body, Coulomb, and optional CAP actions.

    :meth:`physical_action` excludes a configured complex absorbing potential
    and is therefore the action used for physical field-free energies.
    :meth:`__call__` includes the CAP and is the static propagation action.
    """

    def __init__(
        self,
        representation: CoupledTwoRadialSystem,
        *,
        radial_operator_mode: RadialOperatorMode = "element-local",
        coefficient_method: CoupledCoulombCoefficientMethod = "reduced",
        couplings: CoupledCoulombCouplings | None = None,
        coupling_tolerance: float = 1.0e-12,
        max_coupling_pairs: int = 20_000_000,
        absorbing_boundaries: TwoRadialAbsorbingBoundaries | None = None,
    ) -> None:
        if not isinstance(representation, CoupledTwoRadialSystem):
            raise TypeError("representation must be a CoupledTwoRadialSystem")
        if absorbing_boundaries is not None:
            if not isinstance(
                absorbing_boundaries,
                TwoRadialAbsorbingBoundaries,
            ):
                raise TypeError(
                    "absorbing_boundaries must be a "
                    "TwoRadialAbsorbingBoundaries"
                )
            if absorbing_boundaries.representation is not representation:
                raise ValueError(
                    "absorbing boundaries belong to a different representation"
                )

        self.representation = representation
        self.one_body = CoupledTwoRadialOneBodyHamiltonian(
            representation,
            radial_operator_mode=radial_operator_mode,
        )
        self.coulomb = CoupledTwoRadialCoulombHamiltonian(
            representation,
            couplings=couplings,
            coefficient_method=coefficient_method,
            tolerance=coupling_tolerance,
            max_pairs=max_coupling_pairs,
        )
        self.absorbing_boundaries = absorbing_boundaries

    @property
    def psi_shape(self) -> tuple[int, int, int]:
        return self.representation.psi_shape

    @property
    def has_cap(self) -> bool:
        """Whether the static propagation action contains a CAP."""

        return (
            self.absorbing_boundaries is not None
            and self.absorbing_boundaries.has_cap
        )

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Apply the physical field-free Hamiltonian and an optional CAP."""

        state = self.representation.validate_state(state)
        result = self.physical_action(state)
        if self.has_cap:
            result += self.absorbing_boundaries.cap_action(state)
        return result

    def physical_action(self, state: np.ndarray) -> np.ndarray:
        """Apply the physical field-free Hamiltonian without a CAP."""

        state = self.representation.validate_state(state)
        result = self.one_body(state)
        return self.coulomb.apply(state, out=result)

    def apply_one_body_coordinate1(self, state: np.ndarray) -> np.ndarray:
        """Apply the electronic one-body contribution."""

        return self.one_body.apply_coordinate1(state)

    def apply_one_body_coordinate2(self, state: np.ndarray) -> np.ndarray:
        """Apply the internuclear one-body contribution."""

        return self.one_body.apply_coordinate2(state)

    def apply_coulomb(self, state: np.ndarray) -> np.ndarray:
        """Apply the electron--nuclear Coulomb contribution."""

        return self.coulomb(state)

    def cap_action(self, state: np.ndarray) -> np.ndarray:
        """Apply the configured CAP, or return zeros when none is present."""

        state = self.representation.validate_state(state)
        if not self.has_cap:
            return np.zeros_like(state)
        return self.absorbing_boundaries.cap_action(state)

    def components(self, state: np.ndarray) -> CoupledHamiltonianComponents:
        """Return separately evaluated time-independent components."""

        state = self.representation.validate_state(state)
        return CoupledHamiltonianComponents(
            coordinate1=self.apply_one_body_coordinate1(state),
            coordinate2=self.apply_one_body_coordinate2(state),
            coulomb=self.apply_coulomb(state),
            cap=self.cap_action(state) if self.has_cap else None,
        )

    def coefficient_diagonal(
        self,
        *,
        include_coulomb: bool = True,
        include_cap: bool = False,
    ) -> np.ndarray:
        """Return the DVR coefficient-space Hamiltonian diagonal.

        The returned array is suitable for constructing a diagonal Davidson
        or linear-system preconditioner. A CAP is excluded by default so the
        physical diagonal remains real.
        """

        if not isinstance(include_coulomb, (bool, np.bool_)):
            raise TypeError("include_coulomb must be boolean")
        if not isinstance(include_cap, (bool, np.bool_)):
            raise TypeError("include_cap must be boolean")

        diagonal = self._one_body_coefficient_diagonal()
        if include_coulomb:
            diagonal += self._coulomb_coefficient_diagonal()
        if include_cap and self.has_cap:
            cap = self.absorbing_boundaries.cap1
            if cap is not None:
                diagonal = diagonal.astype(complex, copy=False)
                diagonal += cap[None, :, None]
            cap = self.absorbing_boundaries.cap2
            if cap is not None:
                diagonal = diagonal.astype(complex, copy=False)
                diagonal += cap[None, None, :]
        return np.ascontiguousarray(diagonal)

    def davidson_diagonal(
        self,
        *,
        include_coulomb: bool = False,
    ) -> np.ndarray:
        """Return a real physical diagonal for Davidson preconditioning.

        By default this follows the existing two-radial preconditioner and
        retains only the separable one-body diagonal. The point-local Coulomb
        diagonal can be included explicitly without changing the action.
        """

        diagonal = self.coefficient_diagonal(
            include_coulomb=include_coulomb,
            include_cap=False,
        )
        return np.asarray(np.real_if_close(diagonal), dtype=float)

    def expectation(
        self,
        state: np.ndarray,
        *,
        normalize: bool = False,
        include_cap: bool = False,
    ) -> complex:
        """Return a field-free expectation, optionally normalized."""

        if not isinstance(normalize, (bool, np.bool_)):
            raise TypeError("normalize must be boolean")
        if not isinstance(include_cap, (bool, np.bool_)):
            raise TypeError("include_cap must be boolean")
        state = self.representation.validate_state(state)
        action = self(state) if include_cap else self.physical_action(state)
        value = self.representation.inner_product(state, action)
        if normalize:
            norm_squared = self.representation.inner_product(state, state).real
            if norm_squared <= 0.0:
                raise ValueError("state must have positive norm")
            value /= norm_squared
        return complex(value)

    def energy(
        self,
        state: np.ndarray,
        *,
        normalize: bool = True,
        reality_tolerance: float = 1.0e-10,
    ) -> float:
        """Return the real physical field-free energy, excluding a CAP."""

        if not np.isfinite(reality_tolerance) or reality_tolerance < 0.0:
            raise ValueError("reality_tolerance must be finite and non-negative")
        value = self.expectation(state, normalize=normalize, include_cap=False)
        scale = max(1.0, abs(value.real))
        if abs(value.imag) > reality_tolerance * scale:
            raise RuntimeError(
                "field-free energy acquired a significant imaginary part"
            )
        return float(value.real)

    def _one_body_coefficient_diagonal(self) -> np.ndarray:
        diagonal1 = np.asarray(
            [
                self.representation.coordinate1.kinetic_prefactor
                * np.diagonal(
                    self.representation.grid1.second_derivative(int(l1))
                )
                + self.one_body.diagonal1[channel]
                for channel, l1 in enumerate(self.one_body.l1)
            ]
        )
        diagonal2 = np.asarray(
            [
                self.representation.coordinate2.kinetic_prefactor
                * np.diagonal(
                    self.representation.grid2.second_derivative(int(l2))
                )
                + self.one_body.diagonal2[channel]
                for channel, l2 in enumerate(self.one_body.l2)
            ]
        )
        return (
            diagonal1[:, :, None]
            + diagonal2[:, None, :]
        )

    def _coulomb_coefficient_diagonal(self) -> np.ndarray:
        result = np.zeros(self.psi_shape)
        couplings = self.coulomb.couplings
        radial = self.representation.radial_multipoles.values
        for group, output in enumerate(couplings.output):
            for pair in range(
                couplings.offsets[group],
                couplings.offsets[group + 1],
            ):
                if couplings.input[pair] != output:
                    continue
                result[output] += (
                    couplings.coefficient[pair]
                    * radial[couplings.radial_index[pair]]
                )
        return result


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
