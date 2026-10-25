"""High-level ground states for infinite-mass two-electron atoms."""

from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from numbers import Integral
from typing import Literal

import numpy as np

from grid_tdse.ground_state import GroundStateResult, ImaginaryTimeGroundStateSolver

from .angular import AngularBasisSpec, CoulombMultipoleSpec, Parity
from .coulomb import CoulombTerm
from .grids import FEMDVRGridSpec
from .observables import (
    AngularMomentumExpectations,
    angular_momentum_expectations,
    diagonal_angular_expectation,
    validated_state,
)
from .preconditioner import TwoRadialOneBodyPreconditioner, weighted_eigenstates
from .system import (
    CoordinateSpec,
    TwoRadialSystem,
    TwoRadialSystemSpec,
    radial_potential,
)

InitialState = Literal["product"] | np.ndarray


@dataclass(frozen=True)
class TwoElectronAtomParameters:
    """Field-free two-electron atomic parameters in atomic units."""

    name: str
    nuclear_charge: float
    ionic_charge: float
    electron_count: int = 2
    electron_mass: float = 1.0
    electron_charge: float = -1.0
    infinite_nuclear_mass: bool = True


@dataclass(frozen=True)
class TwoElectronAtomGroundStateResult:
    """Physical setup and iteration result for a two-electron atom."""

    parameters: TwoElectronAtomParameters
    inversion_parity: Parity
    electron_exchange_parity: Parity
    representation: TwoRadialSystem
    initial_state: np.ndarray
    ground_state: GroundStateResult
    initial_angular_momentum: AngularMomentumExpectations
    angular_momentum: AngularMomentumExpectations
    inversion_expectation: float
    electron_exchange_expectation: float
    spin_multiplicity: int

    @property
    def state(self) -> np.ndarray:
        return self.ground_state.state

    @property
    def energy(self) -> float:
        return self.ground_state.energy

    @property
    def converged(self) -> bool:
        return self.ground_state.converged

    @property
    def iterations(self) -> int:
        return self.ground_state.iterations

    @property
    def history(self):
        return self.ground_state.history

    @property
    def state_change(self) -> float:
        return self.ground_state.state_change

    @property
    def residual_norm(self) -> float:
        return self.ground_state.residual_norm

    @property
    def termination_reason(self) -> str:
        return self.ground_state.termination_reason

    def save(
        self, path: str | PathLike[str], *, overwrite: bool = False
    ) -> Path:
        """Save the returned state and its complete representation metadata."""

        from .state_archive import save_two_radial_state

        return save_two_radial_state(
            path,
            self.state,
            self.representation,
            physical_state={
                "system_family": "two_electron_atom",
                "system": self.parameters.name,
                "state_index": 0,
                "parameters": asdict(self.parameters),
                "symmetry": {
                    "inversion_parity": self.inversion_parity,
                    "electron_exchange_parity": self.electron_exchange_parity,
                    "spin_multiplicity": self.spin_multiplicity,
                    "M_tot": self.representation.M_tot,
                },
            },
            calculation={
                "method": "imaginary_time_propagation",
                "energy": self.energy,
                "norm": self.ground_state.norm,
                "converged": self.converged,
                "iterations": self.iterations,
                "imaginary_time": self.ground_state.imaginary_time,
                "state_change": self.state_change,
                "residual_norm": self.residual_norm,
                "termination_reason": self.termination_reason,
                "total_l_squared": self.angular_momentum.total_l_squared,
                "total_l_z": self.angular_momentum.total_l_z,
                "inversion_expectation": self.inversion_expectation,
                "electron_exchange_expectation": (
                    self.electron_exchange_expectation
                ),
            },
            overwrite=overwrite,
        )


class TwoElectronAtomGroundState:
    r"""Configure and solve an infinite-nuclear-mass two-electron atom.

    The two electron coordinates use the same FEM-DVR grid and the same
    complete ``(l, m)`` angular basis. ``inversion_parity`` selects total
    spatial inversion, while ``electron_exchange_parity`` selects a symmetric
    spatial singlet (``+1``) or antisymmetric spatial triplet (``-1``).

    Compact spectroscopic aliases are deliberately not used for the native
    symmetry inputs. The complete magnetic range ``m_max=l_max`` is required
    so total orbital angular momentum remains represented. Set ``M_tot`` to
    a conserved total magnetic quantum number for compact angular-pair
    storage, or leave it as ``None`` for the full legacy representation.
    """

    _SYSTEMS = {
        "He": TwoElectronAtomParameters("He", nuclear_charge=2.0, ionic_charge=0.0),
        "Be2+": TwoElectronAtomParameters("Be2+", nuclear_charge=4.0, ionic_charge=2.0),
    }
    _ALIASES = {
        "he": "He",
        "helium": "He",
        "be2+": "Be2+",
        "be^2+": "Be2+",
        "be²⁺": "Be2+",
    }

    def __init__(
        self,
        atom: str,
        *,
        grid: FEMDVRGridSpec,
        inversion_parity: Parity = 1,
        electron_exchange_parity: Parity = 1,
        l_max: int = 2,
        m_max: int | None = None,
        lambda_max: int | None = None,
        mu_max: int | None = None,
        M_tot: int | None = None,
        initial_state: InitialState = "product",
        time_step: float = 0.25,
        max_iterations: int = 1_000,
        energy_tolerance: float = 1.0e-10,
        state_tolerance: float = 1.0e-8,
        residual_tolerance: float = 1.0e-8,
        linear_solver: str = "bicgstab",
        linear_rtol: float = 0.0,
        linear_atol: float = 1.0e-12,
        linear_max_iterations: int | None = None,
        gmres_restart: int | None = None,
        progress_interval: int | None = 10,
        use_preconditioner: bool = True,
        tile_size: int = 2048,
    ) -> None:
        self.parameters = _atomic_parameters(atom)
        self.inversion_parity = _parity(inversion_parity, "inversion_parity")
        self.electron_exchange_parity = _parity(
            electron_exchange_parity, "electron_exchange_parity"
        )
        if not isinstance(grid, FEMDVRGridSpec):
            raise TypeError("grid must be a FEMDVRGridSpec")
        self.grid_spec = grid
        if not isinstance(use_preconditioner, (bool, np.bool_)):
            raise TypeError("use_preconditioner must be boolean")
        self.use_preconditioner = bool(use_preconditioner)

        self.l_max = _nonnegative_integer(l_max, "l_max")
        self.m_max = (
            self.l_max if m_max is None else _nonnegative_integer(m_max, "m_max")
        )
        if self.m_max != self.l_max:
            raise ValueError(
                "m_max must equal l_max in the rotationally invariant "
                "two-electron atomic interface"
            )
        self.lambda_max = (
            2 * self.l_max
            if lambda_max is None
            else _nonnegative_integer(lambda_max, "lambda_max")
        )
        if self.lambda_max > 2 * self.l_max:
            raise ValueError("lambda_max must not exceed 2*l_max")
        self.mu_max = (
            self.lambda_max
            if mu_max is None
            else _nonnegative_integer(mu_max, "mu_max")
        )
        if self.mu_max > self.lambda_max:
            raise ValueError("mu_max must not exceed lambda_max")
        if self.mu_max > 2 * self.m_max:
            raise ValueError("mu_max must not exceed 2*m_max")
        self.M_tot = M_tot

        basis = AngularBasisSpec(self.l_max, self.m_max)
        coordinate = CoordinateSpec(
            grid=grid,
            basis=basis,
            kinetic_prefactor=-0.5 / self.parameters.electron_mass,
            one_body_terms=(
                radial_potential(
                    lambda radius: self.parameters.electron_charge
                    * self.parameters.nuclear_charge
                    / radius
                ),
            ),
        )
        specification = TwoRadialSystemSpec(
            coordinate1=coordinate,
            coordinate2=coordinate,
            multipoles=CoulombMultipoleSpec(self.lambda_max, self.mu_max),
            coulomb_terms=(
                CoulombTerm(
                    radial_scale=1.0,
                    charge_product=self.parameters.electron_charge**2,
                ),
            ),
            M_tot=M_tot,
        )
        self.specification = specification
        self.representation = specification.build()
        self.hamiltonian = self.representation.hamiltonian(tile_size=tile_size)
        self.solver = ImaginaryTimeGroundStateSolver(
            time_step=time_step,
            max_iterations=max_iterations,
            energy_tolerance=energy_tolerance,
            state_tolerance=state_tolerance,
            residual_tolerance=residual_tolerance,
            linear_solver=linear_solver,
            linear_rtol=linear_rtol,
            linear_atol=linear_atol,
            linear_max_iterations=linear_max_iterations,
            gmres_restart=gmres_restart,
            progress_interval=progress_interval,
        )
        self.preconditioner = (
            TwoRadialOneBodyPreconditioner(self.representation, time_step)
            if self.use_preconditioner
            else None
        )
        self.initial_state_kind = (
            "product" if isinstance(initial_state, str) else "array"
        )
        self.initial_state = self._prepare_initial_state(initial_state)

    @property
    def atom_name(self) -> str:
        return self.parameters.name

    @property
    def spin_multiplicity(self) -> int:
        """Spin multiplicity implied by the spatial exchange symmetry."""

        return 1 if self.electron_exchange_parity == 1 else 3

    def solve(self) -> TwoElectronAtomGroundStateResult:
        """Run normalized imaginary-time propagation."""

        initial = self.initial_state.copy()
        ground_state = self.solver.solve(
            initial,
            self.hamiltonian,
            self.representation.inner_product,
            preconditioner=self.preconditioner,
        )
        return TwoElectronAtomGroundStateResult(
            parameters=self.parameters,
            inversion_parity=self.inversion_parity,
            electron_exchange_parity=self.electron_exchange_parity,
            representation=self.representation,
            initial_state=initial,
            ground_state=ground_state,
            initial_angular_momentum=self.angular_momentum_expectations(initial),
            angular_momentum=self.angular_momentum_expectations(ground_state.state),
            inversion_expectation=self.inversion_expectation(ground_state.state),
            electron_exchange_expectation=self.electron_exchange_expectation(
                ground_state.state
            ),
            spin_multiplicity=self.spin_multiplicity,
        )

    __call__ = solve

    def project_symmetry(self, state: np.ndarray) -> np.ndarray:
        """Project an array onto total inversion and electron exchange parity."""

        state = validated_state(self.representation, state)
        inverted = np.empty_like(state)
        for pair, (i, j) in enumerate(self.representation.angular_pairs.pairs):
            l1 = self.representation.basis1.states[i][0]
            l2 = self.representation.basis2.states[j][0]
            block = (
                state[pair]
                if self.representation.uses_compact_angular_storage
                else state[i, j]
            )
            self.representation.set_angular_block(
                inverted, i, j, _l_parity(l1 + l2) * block
            )
        projected = 0.5 * (state + self.inversion_parity * inverted)
        exchanged = self._exchange_state(projected)
        return 0.5 * (projected + self.electron_exchange_parity * exchanged)

    def inversion_expectation(self, state: np.ndarray) -> float:
        return diagonal_angular_expectation(
            self.representation,
            state,
            lambda l1, l2, m1, m2: _l_parity(l1 + l2),
        )

    def electron_exchange_expectation(self, state: np.ndarray) -> float:
        state = validated_state(self.representation, state)
        norm_squared = self.representation.inner_product(state, state).real
        if norm_squared <= 0.0:
            raise ValueError("state has zero norm")
        exchanged = self._exchange_state(state)
        value = self.representation.inner_product(state, exchanged)
        return float(value.real / norm_squared)

    def angular_momentum_expectations(
        self, state: np.ndarray
    ) -> AngularMomentumExpectations:
        return angular_momentum_expectations(self.representation, state)

    def _prepare_initial_state(self, initial_state: InitialState) -> np.ndarray:
        if isinstance(initial_state, str):
            if initial_state != "product":
                raise ValueError("initial_state must be 'product' or a NumPy array")
            state = self._product_initial_state()
        else:
            state = self.representation.pack_state(initial_state)
            state = validated_state(self.representation, state)

        state = self.project_symmetry(state)
        norm_squared = self.representation.inner_product(state, state).real
        if norm_squared <= 0.0:
            raise ValueError(
                "initial_state has zero norm after projection onto the requested "
                "symmetry sector"
            )
        return state / np.sqrt(norm_squared)

    def _product_initial_state(self) -> np.ndarray:
        if self.M_tot not in (None, 0):
            raise ValueError(
                "the product initial state has M_tot=0; provide a custom state "
                "for another conserved sector"
            )
        s_channel = self.representation.basis1.index(0, 0)
        _, s_states = weighted_eigenstates(
            self.representation.h1[s_channel],
            self.representation.grid1.weights,
        )
        state = np.zeros(self.representation.psi_shape, dtype=s_states.dtype)

        if self.inversion_parity == 1:
            if self.electron_exchange_parity == 1:
                self.representation.set_angular_block(
                    state,
                    s_channel,
                    s_channel,
                    np.outer(s_states[:, 0], s_states[:, 0]),
                )
            else:
                if s_states.shape[1] < 2:
                    raise ValueError(
                        "the radial grid must represent two s states for an "
                        "antisymmetric even-inversion product guess"
                    )
                self.representation.set_angular_block(
                    state,
                    s_channel,
                    s_channel,
                    (
                        np.outer(s_states[:, 0], s_states[:, 1])
                        - np.outer(s_states[:, 1], s_states[:, 0])
                    )
                    / np.sqrt(2.0),
                )
        else:
            if self.l_max < 1:
                raise ValueError(
                    "l_max must be at least one for antisymmetric inversion"
                )
            p_channel = self.representation.basis1.index(1, 0)
            _, p_states = weighted_eigenstates(
                self.representation.h1[p_channel],
                self.representation.grid1.weights,
            )
            direct = np.outer(s_states[:, 0], p_states[:, 0])
            exchanged = np.outer(p_states[:, 0], s_states[:, 0])
            self.representation.set_angular_block(
                state, s_channel, p_channel, direct / np.sqrt(2.0)
            )
            self.representation.set_angular_block(
                state,
                p_channel,
                s_channel,
                self.electron_exchange_parity * exchanged / np.sqrt(2.0),
            )
        return state

    def _exchange_state(self, state: np.ndarray) -> np.ndarray:
        if not self.representation.uses_compact_angular_storage:
            return state.transpose(1, 0, 3, 2)
        exchanged = np.empty_like(state)
        for pair, (i, j) in enumerate(self.representation.angular_pairs.pairs):
            target = self.representation.angular_pairs.index(j, i)
            exchanged[target] = state[pair].T
        return exchanged


def _atomic_parameters(atom: str) -> TwoElectronAtomParameters:
    if not isinstance(atom, str):
        raise TypeError("atom must be a string")
    canonical = TwoElectronAtomGroundState._ALIASES.get(atom.strip().lower())
    if canonical is None:
        supported = ", ".join(TwoElectronAtomGroundState._SYSTEMS)
        raise ValueError(
            f"unsupported two-electron atom {atom!r}; choose from: {supported}"
        )
    return TwoElectronAtomGroundState._SYSTEMS[canonical]


def _parity(value, name: str) -> Parity:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) not in (-1, 1)
    ):
        raise ValueError(f"{name} must be +1 or -1")
    return int(value)


def _l_parity(l: int) -> int:
    return -1 if l % 2 else 1


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
