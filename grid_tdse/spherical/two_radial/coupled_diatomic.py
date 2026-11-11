"""Coupled-angular Davidson ground states for non-BO diatomic ions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np

from grid_tdse.ground_state import (
    DavidsonGroundStateSolver,
    DiagonalDavidsonPreconditioner,
    GroundStateEigenResult,
)

from .angular import AngularBasisSpec, CoulombMultipoleSpec, Parity
from .coulomb import CoulombTerm, RadialCoulombMethod
from .coupled_angular import CoupledAngularBasisSpec
from .coupled_state_archive import (
    CoupledTwoRadialState,
    load_coupled_two_radial_state,
    save_coupled_two_radial_state,
)
from .diatomic import (
    DiatomicIonParameters,
    NonBornOppenheimerDiatomicIonGroundState,
    _diatomic_parameters,
    _pair_exchange_parity,
    _parity,
)
from .grids import FEMDVRGridSpec
from .preconditioner import (
    CoupledTwoRadialOneBodyDavidsonPreconditioner,
    lowest_weighted_eigenstate,
)
from .system import (
    CoordinateSpec,
    CoupledTwoRadialSystem,
    RadialOperatorMode,
    TwoRadialSystemSpec,
    radial_potential,
)


CoupledInitialState = (
    Literal["product"]
    | str
    | PathLike[str]
    | np.ndarray
    | CoupledTwoRadialState
)
CoupledDavidsonPreconditionerName = Literal[
    "identity",
    "diagonal",
    "one-body-spectral",
]


@dataclass(frozen=True)
class CoupledDiatomicIonGroundStateObservables:
    """Representation-diagonal observables for a coupled ground state."""

    r: float
    R: float
    R_squared: float
    total_l_squared: float
    total_l_z: float
    inversion: float
    pair_exchange: float | None


@dataclass(frozen=True)
class CoupledDiatomicIonGroundStateResult:
    """Coupled representation and Davidson result for one diatomic ion."""

    parameters: DiatomicIonParameters
    inversion_parity: Parity
    pair_exchange_parity: Parity | None
    representation: CoupledTwoRadialSystem
    initial_state: np.ndarray
    ground_state: GroundStateEigenResult
    solver_options: Mapping[str, object]
    initial_state_provenance: Mapping[str, object]
    observables: CoupledDiatomicIonGroundStateObservables
    radial_density_r: np.ndarray
    radial_density_R: np.ndarray
    radial_operator_mode: RadialOperatorMode

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
    def norm(self) -> float:
        return self.representation.norm(self.state)

    @property
    def residual_norm(self) -> float:
        return self.ground_state.residual_norm

    @property
    def state_change(self) -> float:
        return self.ground_state.state_change

    @property
    def termination_reason(self) -> str:
        return self.ground_state.termination_reason

    @property
    def history(self):
        return self.ground_state.history

    def save(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Save the coupled state, fixed-space metadata, and diagnostics."""

        calculation = {
            "method": "direct_eigensolver",
            "solver": "davidson",
            "solver_options": dict(self.solver_options),
            "radial_operator_mode": self.radial_operator_mode,
            "energy": self.energy,
            "norm": self.norm,
            "converged": self.converged,
            "iterations": self.iterations,
            "state_change": self.state_change,
            "residual_norm": self.residual_norm,
            "termination_reason": self.termination_reason,
            "restarts": self.ground_state.restarts,
            "hamiltonian_actions": self.ground_state.hamiltonian_actions,
            "preconditioner_applications": (
                self.ground_state.preconditioner_applications
            ),
            "maximum_subspace_dimension": (
                self.ground_state.maximum_subspace_dimension
            ),
            "final_subspace_dimension": (
                self.ground_state.final_subspace_dimension
            ),
            "observables": asdict(self.observables),
            "initial_state": dict(self.initial_state_provenance),
            "validation_scope": "fixed-space-representation-validation",
            "physical_basis_convergence_claimed": False,
        }
        symmetry = {
            "inversion_parity": self.inversion_parity,
            "pair_exchange_parity": self.pair_exchange_parity,
            "M_tot": self.representation.M_tot,
        }
        return save_coupled_two_radial_state(
            path,
            self.state,
            self.representation,
            physical_state={
                "system_family": "non_bo_diatomic_ion",
                "system": self.parameters.name,
                "state_index": 0,
                "parameters": asdict(self.parameters),
                "symmetry": symmetry,
            },
            calculation=calculation,
            overwrite=overwrite,
        )


class CoupledNonBornOppenheimerDiatomicIonGroundState(
    NonBornOppenheimerDiatomicIonGroundState
):
    """Configure a serial coupled-angular Davidson ground-state solve."""

    def __init__(
        self,
        system: str,
        *,
        wf_representation: Literal["coupled"] = "coupled",
        electronic_grid: FEMDVRGridSpec,
        internuclear_grid: FEMDVRGridSpec,
        inversion_parity: Parity,
        pair_exchange_parity: Parity | None,
        l1_max: int = 2,
        l2_max: int = 2,
        L_max: int = 0,
        M_tot: int = 0,
        lambda_max: int | None = None,
        radial_coulomb_method: RadialCoulombMethod = "poisson",
        codata_set: str = "2022",
        initial_state: CoupledInitialState = "product",
        solver: Literal["davidson"] = "davidson",
        max_iterations: int = 80,
        residual_tolerance: float = 1.0e-8,
        maximum_subspace_dimension: int | None = 20,
        retained_ritz_vectors: int = 4,
        eigensolver_breakdown_tolerance: float = 1.0e-14,
        eigensolver_reorthogonalization: Literal[
            "if-needed", "always"
        ] = "always",
        eigensolver_reorthogonalization_threshold: float = 0.5,
        eigensolver_hermiticity_tolerance: float = 1.0e-10,
        eigensolver_reality_tolerance: float = 1.0e-12,
        track_subspace_orthogonality: bool = True,
        davidson_preconditioner: CoupledDavidsonPreconditionerName = (
            "one-body-spectral"
        ),
        davidson_include_coulomb_diagonal: bool = False,
        davidson_relative_floor: float = 1.0e-8,
        davidson_absolute_floor: float = 1.0e-12,
        davidson_level_shift: float = 0.0,
        progress_interval: int | None = 10,
        radial_operator_mode: RadialOperatorMode = "element-local",
        coupling_tolerance: float = 1.0e-12,
        max_coupling_pairs: int = 20_000_000,
    ) -> None:
        if wf_representation != "coupled":
            raise ValueError(
                "the coupled constructor requires wf_representation='coupled'"
            )
        if solver != "davidson":
            raise ValueError(
                "the coupled ground-state path requires solver='davidson'"
            )
        self.wf_representation = "coupled"
        self.parameters = _diatomic_parameters(system, codata_set)
        self.inversion_parity = _parity(inversion_parity, "inversion_parity")
        self.pair_exchange_parity = _pair_exchange_parity(
            pair_exchange_parity,
            self.parameters,
        )
        if not isinstance(electronic_grid, FEMDVRGridSpec):
            raise TypeError("electronic_grid must be a FEMDVRGridSpec")
        if not isinstance(internuclear_grid, FEMDVRGridSpec):
            raise TypeError("internuclear_grid must be a FEMDVRGridSpec")
        self.electronic_grid_spec = electronic_grid
        self.internuclear_grid_spec = internuclear_grid
        self.l1_max = _nonnegative_integer(l1_max, "l1_max")
        self.l2_max = _nonnegative_integer(l2_max, "l2_max")
        self.L_max = _nonnegative_integer(L_max, "L_max")
        self.M_tot = _integer(M_tot, "M_tot")
        complete_lambda = 2 * min(self.l1_max, self.l2_max)
        self.lambda_max = (
            complete_lambda
            if lambda_max is None
            else _nonnegative_integer(lambda_max, "lambda_max")
        )
        if self.lambda_max > complete_lambda:
            raise ValueError(
                "lambda_max must not exceed 2*min(l1_max, l2_max)"
            )
        if radial_operator_mode not in {"global", "element-local"}:
            raise ValueError(
                "radial_operator_mode must be 'global' or 'element-local'"
            )
        self.radial_operator_mode = radial_operator_mode
        self.radial_coulomb_method = radial_coulomb_method

        p = self.parameters
        angular_basis = CoupledAngularBasisSpec.for_diatomic_ion(
            self.l1_max,
            self.l2_max,
            self.L_max,
            has_identical_pair=p.has_identical_pair,
            M_tot=self.M_tot,
            inversion_parity=self.inversion_parity,
            pair_exchange_parity=self.pair_exchange_parity,
        )
        specification = TwoRadialSystemSpec(
            coordinate1=CoordinateSpec(
                grid=electronic_grid,
                basis=AngularBasisSpec(self.l1_max),
                kinetic_prefactor=-0.5 / p.electronic_reduced_mass,
            ),
            coordinate2=CoordinateSpec(
                grid=internuclear_grid,
                basis=AngularBasisSpec(self.l2_max),
                kinetic_prefactor=-0.5 / p.internuclear_reduced_mass,
                one_body_terms=(
                    radial_potential(
                        lambda radius: (
                            p.reference_charge
                            * p.partner_charge
                            / radius
                        )
                    ),
                ),
            ),
            multipoles=CoulombMultipoleSpec(
                self.lambda_max,
                self.lambda_max,
                lambda_parity=(1 if p.has_identical_pair else None),
            ),
            coulomb_terms=(
                CoulombTerm(
                    p.reference_center_scale,
                    p.electron_charge * p.reference_charge,
                ),
                CoulombTerm(
                    p.partner_center_scale,
                    p.electron_charge * p.partner_charge,
                ),
            ),
            M_tot=self.M_tot,
            radial_coulomb_method=radial_coulomb_method,
        )
        self.specification = specification
        self.representation = specification.build_coupled(angular_basis)
        self.hamiltonian = self.representation.hamiltonian(
            radial_operator_mode=radial_operator_mode,
            coupling_tolerance=coupling_tolerance,
            max_coupling_pairs=max_coupling_pairs,
        )

        solver_options = {
            "max_iterations": max_iterations,
            "maximum_subspace_dimension": maximum_subspace_dimension,
            "retained_ritz_vectors": retained_ritz_vectors,
            "residual_tolerance": residual_tolerance,
            "breakdown_tolerance": eigensolver_breakdown_tolerance,
            "reorthogonalization": eigensolver_reorthogonalization,
            "reorthogonalization_threshold": (
                eigensolver_reorthogonalization_threshold
            ),
            "hermiticity_tolerance": eigensolver_hermiticity_tolerance,
            "reality_tolerance": eigensolver_reality_tolerance,
            "track_orthogonality": track_subspace_orthogonality,
            "progress_interval": progress_interval,
        }
        self.solver = DavidsonGroundStateSolver(**solver_options)
        if davidson_preconditioner == "identity":
            self.preconditioner = None
        elif davidson_preconditioner == "diagonal":
            self.preconditioner = DiagonalDavidsonPreconditioner(
                self.hamiltonian.davidson_diagonal(
                    include_coulomb=davidson_include_coulomb_diagonal
                ),
                relative_floor=davidson_relative_floor,
                absolute_floor=davidson_absolute_floor,
                level_shift=davidson_level_shift,
            )
        elif davidson_preconditioner == "one-body-spectral":
            self.preconditioner = (
                CoupledTwoRadialOneBodyDavidsonPreconditioner(
                    self.representation,
                    relative_floor=davidson_relative_floor,
                    absolute_floor=davidson_absolute_floor,
                    level_shift=davidson_level_shift,
                )
            )
        else:
            raise ValueError(
                "davidson_preconditioner must be 'identity', 'diagonal', "
                "or 'one-body-spectral'"
            )
        self.solver_options = {
            **solver_options,
            "solver": solver,
            "preconditioner": davidson_preconditioner,
            "include_coulomb_diagonal": davidson_include_coulomb_diagonal,
            "preconditioner_relative_floor": davidson_relative_floor,
            "preconditioner_absolute_floor": davidson_absolute_floor,
            "preconditioner_level_shift": davidson_level_shift,
        }
        self.initial_state_kind = "array"
        self.initial_state_provenance: dict[str, object] = {"kind": "array"}
        self.initial_state = self._prepare_coupled_initial_state(initial_state)

    @property
    def system_name(self) -> str:
        return self.parameters.name

    def solve(self) -> CoupledDiatomicIonGroundStateResult:
        """Solve for the lowest state in the selected coupled symmetry sector."""

        initial = self.initial_state.copy()
        ground_state = self.solver.solve(
            initial,
            self.hamiltonian.physical_action,
            self.representation.inner_product,
            preconditioner=self.preconditioner,
            print_progress=True,
        )
        observables = self.observables(ground_state.state)
        density_r, density_R = self.representation.radial_marginals(
            ground_state.state
        )
        norm_squared = self.representation.inner_product(
            ground_state.state,
            ground_state.state,
        ).real
        density_r /= norm_squared
        density_R /= norm_squared
        return CoupledDiatomicIonGroundStateResult(
            parameters=self.parameters,
            inversion_parity=self.inversion_parity,
            pair_exchange_parity=self.pair_exchange_parity,
            representation=self.representation,
            initial_state=initial,
            ground_state=ground_state,
            solver_options=self.solver_options,
            initial_state_provenance=self.initial_state_provenance,
            observables=observables,
            radial_density_r=density_r,
            radial_density_R=density_R,
            radial_operator_mode=self.radial_operator_mode,
        )

    __call__ = solve

    def observables(
        self,
        state: np.ndarray,
    ) -> CoupledDiatomicIonGroundStateObservables:
        """Evaluate radial and diagonal angular observables."""

        values = self.representation.validate_state(state)
        norm = self.representation.inner_product(values, values).real
        if norm <= 0.0:
            raise ValueError("state must have positive norm")
        weights = (
            self.representation.grid1.weights[None, :, None]
            * self.representation.grid2.weights[None, None, :]
        )
        populations = np.sum(np.abs(values) ** 2 * weights, axis=(1, 2))
        populations /= norm
        states = self.representation.angular_basis.states

        def angular_expectation(function) -> float:
            return float(
                sum(
                    population * function(l1, l2, L, M)
                    for population, (l1, l2, L, M) in zip(
                        populations,
                        states,
                        strict=True,
                    )
                )
            )

        radial_density_r, radial_density_R = (
            self.representation.radial_marginals(values)
        )
        radial_density_r /= norm
        radial_density_R /= norm
        r = float(
            np.dot(
                radial_density_r * self.representation.grid1.weights,
                self.representation.grid1.r,
            )
        )
        R = float(
            np.dot(
                radial_density_R * self.representation.grid2.weights,
                self.representation.grid2.r,
            )
        )
        R_squared = float(
            np.dot(
                radial_density_R * self.representation.grid2.weights,
                self.representation.grid2.r**2,
            )
        )
        exchange = (
            angular_expectation(lambda l1, l2, L, M: (-1.0) ** l2)
            if self.parameters.has_identical_pair
            else None
        )
        return CoupledDiatomicIonGroundStateObservables(
            r=r,
            R=R,
            R_squared=R_squared,
            total_l_squared=angular_expectation(
                lambda l1, l2, L, M: L * (L + 1)
            ),
            total_l_z=angular_expectation(
                lambda l1, l2, L, M: M
            ),
            inversion=angular_expectation(
                lambda l1, l2, L, M: (-1.0) ** (l1 + l2)
            ),
            pair_exchange=exchange,
        )

    def _prepare_coupled_initial_state(
        self,
        initial_state: CoupledInitialState,
    ) -> np.ndarray:
        if isinstance(initial_state, str) and initial_state == "product":
            self.initial_state_kind = "product"
            self.initial_state_provenance = {"kind": "product"}
            state = self._coupled_product_initial_state()
        elif isinstance(initial_state, (str, PathLike, CoupledTwoRadialState)):
            archived = (
                load_coupled_two_radial_state(initial_state)
                if isinstance(initial_state, (str, PathLike))
                else initial_state
            )
            self._validate_coupled_initial_archive(archived)
            state = archived.state_for(self.representation)
            self.initial_state_kind = "archive"
            self.initial_state_provenance = {
                "kind": "archive",
                "path": (
                    str(initial_state)
                    if isinstance(initial_state, (str, PathLike))
                    else None
                ),
                "state_data_id": archived.state_data_id,
                "representation_id": archived.representation_id,
            }
        else:
            self.initial_state_kind = "array"
            self.initial_state_provenance = {"kind": "array"}
            state = self.representation.copy_state(initial_state)

        norm = self.representation.norm(state)
        if norm <= 0.0:
            raise ValueError("initial_state must have positive norm")
        return state / norm

    def _coupled_product_initial_state(self) -> np.ndarray:
        channel_state = self.representation.angular_basis.states[0]
        l1, l2, _, _ = channel_state
        operator1 = self.specification.coordinate1.build_one_body_channel(
            l1,
            0,
            self.representation.grid1,
        )
        operator2 = self.specification.coordinate2.build_one_body_channel(
            l2,
            0,
            self.representation.grid2,
        )
        _, radial1 = lowest_weighted_eigenstate(
            operator1,
            self.representation.grid1.weights,
        )
        _, radial2 = lowest_weighted_eigenstate(
            operator2,
            self.representation.grid2.weights,
        )
        state = self.representation.zeros(dtype=np.result_type(radial1, radial2))
        state[0] = radial1[:, None] * radial2[None, :]
        return state

    def _validate_coupled_initial_archive(
        self,
        archived: CoupledTwoRadialState,
    ) -> None:
        physical = archived.physical_state
        if physical.get("system_family") != "non_bo_diatomic_ion":
            raise ValueError(
                "initial-state archive is not a non-BO diatomic-ion state"
            )
        if physical.get("system") != self.system_name:
            raise ValueError(
                f"initial-state archive contains {physical.get('system')!r}; "
                f"expected {self.system_name!r}"
            )
        stored_parameters = physical.get("parameters", {})
        if stored_parameters.get("codata_set") != self.parameters.codata_set:
            raise ValueError(
                "initial-state archive CODATA set is "
                f"{stored_parameters.get('codata_set')!r}; expected "
                f"{self.parameters.codata_set!r}"
            )
        symmetry = physical.get("symmetry", {})
        expected = {
            "inversion_parity": self.inversion_parity,
            "pair_exchange_parity": self.pair_exchange_parity,
            "M_tot": self.M_tot,
        }
        for name, value in expected.items():
            if symmetry.get(name) != value:
                raise ValueError(
                    f"initial-state archive symmetry {name!r} is "
                    f"{symmetry.get(name)!r}; expected {value!r}"
                )


def _nonnegative_integer(value, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)
