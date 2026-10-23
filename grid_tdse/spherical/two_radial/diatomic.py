"""High-level ground states for non-Born--Oppenheimer diatomic ions."""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from matter_field_params import (
    get_reduced_masses_and_charges_3body,
    predefined_3body_systems,
)

from grid_tdse.ground_state import (
    GroundStateResult,
    ImaginaryTimeGroundStateSolver,
)

from .angular import AngularBasisSpec, CoulombMultipoleSpec, Parity
from .coulomb import CoulombTerm
from .grids import FEMDVRGridSpec
from .preconditioner import (
    TwoRadialOneBodyPreconditioner,
    lowest_weighted_eigenstate,
)
from .system import (
    CoordinateSpec,
    TwoRadialSystem,
    TwoRadialSystemSpec,
    radial_potential,
)

InitialState = Literal["product"] | np.ndarray


@dataclass(frozen=True)
class DiatomicIonParameters:
    """Masses, charges, and nuclear-center coordinate factors in atomic units."""

    name: str
    particles: tuple[str, str, str]
    codata_set: str
    internuclear_reduced_mass: float
    electronic_reduced_mass: float
    internuclear_reduced_charge: float
    electronic_reduced_charge: float
    reference_mass: float
    partner_mass: float
    electron_mass: float
    reference_charge: float
    partner_charge: float
    electron_charge: float
    reference_center_scale: float
    partner_center_scale: float

    @property
    def has_identical_pair(self) -> bool:
        """Whether the two particles defining the internuclear coordinate match."""

        return self.particles[0] == self.particles[1]


@dataclass(frozen=True)
class AngularMomentumExpectations:
    """Total orbital-angular-momentum expectations in atomic units."""

    total_l_squared: float
    total_l_z: float


@dataclass(frozen=True)
class DiatomicIonGroundStateResult:
    """Physical setup and iteration result for a diatomic-ion ground state."""

    parameters: DiatomicIonParameters
    representation: TwoRadialSystem
    initial_state: np.ndarray
    ground_state: GroundStateResult
    initial_angular_momentum: AngularMomentumExpectations
    angular_momentum: AngularMomentumExpectations
    inversion_expectation: float
    pair_exchange_expectation: float | None

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


class NonBornOppenheimerDiatomicIonGroundState:
    r"""Configure and solve a field-free three-body diatomic ion.

    The electronic coordinate ``r`` is measured from the center of mass of
    the two positive particles, and ``R`` joins those particles.  The native
    symmetry inputs are total inversion parity and, when the positive pair is
    identical, pair-exchange parity.  Compact molecular labels such as
    ``"sp"`` are deliberately not part of this interface.

    Only FEM-DVR is supported at this stage.  ``l_max`` and ``m_max`` apply to
    both coordinates; the complete magnetic range is required so that total
    orbital angular momentum is represented without an axial truncation.
    """

    _CANONICAL_NAMES = {
        ("proton", "proton", "electron"): "H2+",
        ("deuteron", "proton", "electron"): "HD+",
        ("deuteron", "deuteron", "electron"): "D2+",
    }

    def __init__(
        self,
        system: str,
        *,
        electronic_grid: FEMDVRGridSpec,
        internuclear_grid: FEMDVRGridSpec,
        inversion_parity: Parity,
        pair_exchange_parity: Parity | None,
        symmetry_adapted_basis: bool = True,
        l_max: int = 2,
        m_max: int | None = None,
        lambda_max: int | None = None,
        mu_max: int | None = None,
        codata_set: str = "2022",
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
        self.parameters = _diatomic_parameters(system, codata_set)
        self.inversion_parity = _parity(inversion_parity, "inversion_parity")
        self.pair_exchange_parity = _pair_exchange_parity(
            pair_exchange_parity, self.parameters
        )
        if not isinstance(symmetry_adapted_basis, (bool, np.bool_)):
            raise TypeError("symmetry_adapted_basis must be boolean")
        self.symmetry_adapted_basis = bool(symmetry_adapted_basis)
        if not isinstance(use_preconditioner, (bool, np.bool_)):
            raise TypeError("use_preconditioner must be boolean")
        self.use_preconditioner = bool(use_preconditioner)

        if not isinstance(electronic_grid, FEMDVRGridSpec):
            raise TypeError("electronic_grid must be a FEMDVRGridSpec")
        if not isinstance(internuclear_grid, FEMDVRGridSpec):
            raise TypeError("internuclear_grid must be a FEMDVRGridSpec")
        self.electronic_grid_spec = electronic_grid
        self.internuclear_grid_spec = internuclear_grid

        self.l_max = _nonnegative_integer(l_max, "l_max")
        self.m_max = (
            self.l_max if m_max is None else _nonnegative_integer(m_max, "m_max")
        )
        if self.m_max != self.l_max:
            raise ValueError(
                "m_max must equal l_max in the rotationally invariant "
                "diatomic-ion interface"
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

        electron_parity, pair_parity = self._coordinate_parities()
        electron_basis = AngularBasisSpec(
            self.l_max,
            self.m_max,
            coordinate_parity=electron_parity,
        )
        pair_basis = AngularBasisSpec(
            self.l_max,
            self.m_max,
            coordinate_parity=pair_parity,
        )
        multipoles = CoulombMultipoleSpec(
            self.lambda_max,
            self.mu_max,
            lambda_parity=(1 if self.parameters.has_identical_pair else None),
        )

        p = self.parameters
        specification = TwoRadialSystemSpec(
            coordinate1=CoordinateSpec(
                grid=electronic_grid,
                basis=electron_basis,
                kinetic_prefactor=-0.5 / p.electronic_reduced_mass,
            ),
            coordinate2=CoordinateSpec(
                grid=internuclear_grid,
                basis=pair_basis,
                kinetic_prefactor=-0.5 / p.internuclear_reduced_mass,
                one_body_terms=(
                    radial_potential(
                        lambda radius: p.reference_charge * p.partner_charge / radius
                    ),
                ),
            ),
            multipoles=multipoles,
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
    def system_name(self) -> str:
        return self.parameters.name

    @property
    def basis_is_reduced(self) -> bool:
        """Whether pair exchange made coordinate-parity reduction possible."""

        return self.symmetry_adapted_basis and self.parameters.has_identical_pair

    def solve(self) -> DiatomicIonGroundStateResult:
        """Run normalized imaginary-time propagation."""

        initial = self.initial_state.copy()
        ground_state = self.solver.solve(
            initial,
            self.hamiltonian,
            self.representation.inner_product,
            preconditioner=self.preconditioner,
        )
        return DiatomicIonGroundStateResult(
            parameters=self.parameters,
            representation=self.representation,
            initial_state=initial,
            ground_state=ground_state,
            initial_angular_momentum=self.angular_momentum_expectations(initial),
            angular_momentum=self.angular_momentum_expectations(ground_state.state),
            inversion_expectation=self.inversion_expectation(ground_state.state),
            pair_exchange_expectation=(
                self.pair_exchange_expectation(ground_state.state)
                if self.parameters.has_identical_pair
                else None
            ),
        )

    __call__ = solve

    def project_symmetry(self, state: np.ndarray) -> np.ndarray:
        """Project an array onto the requested inversion and pair symmetry."""

        state = np.asarray(state)
        if state.shape != self.representation.psi_shape:
            raise ValueError(
                f"state has shape {state.shape}; "
                f"expected {self.representation.psi_shape}"
            )
        projected = state.copy()
        for i, (l_e, _) in enumerate(self.representation.basis1.states):
            for j, (l_R, _) in enumerate(self.representation.basis2.states):
                inversion = _l_parity(l_e + l_R)
                pair_exchange = _l_parity(l_R)
                if inversion != self.inversion_parity or (
                    self.pair_exchange_parity is not None
                    and pair_exchange != self.pair_exchange_parity
                ):
                    projected[i, j] = 0
        return projected

    def inversion_expectation(self, state: np.ndarray) -> float:
        return self._diagonal_angular_expectation(
            state,
            lambda l_e, l_R, m_e, m_R: _l_parity(l_e + l_R),
        )

    def pair_exchange_expectation(self, state: np.ndarray) -> float:
        if not self.parameters.has_identical_pair:
            raise ValueError(
                f"pair exchange parity is not defined for {self.system_name}"
            )
        return self._diagonal_angular_expectation(
            state,
            lambda l_e, l_R, m_e, m_R: _l_parity(l_R),
        )

    def angular_momentum_expectations(
        self, state: np.ndarray
    ) -> AngularMomentumExpectations:
        """Return ``<L_total**2>`` and ``<L_total,z>`` for a state."""

        state = self._validated_state(state)
        norm_squared = self.representation.inner_product(state, state).real
        if norm_squared <= 0.0:
            raise ValueError("state has zero norm")

        action = np.zeros_like(state, dtype=np.result_type(state.dtype, float))
        basis1 = self.representation.basis1
        basis2 = self.representation.basis2
        for i, (l_e, m_e) in enumerate(basis1.states):
            for j, (l_R, m_R) in enumerate(basis2.states):
                action[i, j] += (
                    l_e * (l_e + 1) + l_R * (l_R + 1) + 2 * m_e * m_R
                ) * state[i, j]

                if m_e < l_e and m_R > -l_R:
                    out_i = basis1.index(l_e, m_e + 1)
                    out_j = basis2.index(l_R, m_R - 1)
                    coefficient = np.sqrt(l_e * (l_e + 1) - m_e * (m_e + 1)) * np.sqrt(
                        l_R * (l_R + 1) - m_R * (m_R - 1)
                    )
                    action[out_i, out_j] += coefficient * state[i, j]

                if m_e > -l_e and m_R < l_R:
                    out_i = basis1.index(l_e, m_e - 1)
                    out_j = basis2.index(l_R, m_R + 1)
                    coefficient = np.sqrt(l_e * (l_e + 1) - m_e * (m_e - 1)) * np.sqrt(
                        l_R * (l_R + 1) - m_R * (m_R + 1)
                    )
                    action[out_i, out_j] += coefficient * state[i, j]

        total_l_squared = (
            self.representation.inner_product(state, action).real / norm_squared
        )
        total_l_z = self._diagonal_angular_expectation(
            state,
            lambda l_e, l_R, m_e, m_R: m_e + m_R,
        )
        return AngularMomentumExpectations(
            total_l_squared=float(total_l_squared),
            total_l_z=float(total_l_z),
        )

    def _coordinate_parities(self):
        if not self.basis_is_reduced:
            return None, None
        pair_parity = self.pair_exchange_parity
        electron_parity = self.inversion_parity * pair_parity
        return electron_parity, pair_parity

    def _prepare_initial_state(self, initial_state: InitialState) -> np.ndarray:
        if isinstance(initial_state, str):
            if initial_state != "product":
                raise ValueError("initial_state must be 'product' or a NumPy array")
            state = self._product_initial_state()
        else:
            state = self._validated_state(initial_state).copy()

        state = self.project_symmetry(state)
        norm_squared = self.representation.inner_product(state, state).real
        if norm_squared <= 0.0:
            raise ValueError(
                "initial_state has zero norm after projection onto the requested "
                "symmetry sector"
            )
        return state / np.sqrt(norm_squared)

    def _product_initial_state(self) -> np.ndarray:
        if self.pair_exchange_parity is None:
            l_R = 0
            l_e = 0 if self.inversion_parity == 1 else 1
        else:
            l_R = 0 if self.pair_exchange_parity == 1 else 1
            electron_parity = self.inversion_parity * self.pair_exchange_parity
            l_e = 0 if electron_parity == 1 else 1
        if max(l_e, l_R) > self.l_max:
            raise ValueError(
                "l_max is too small for the requested inversion and pair symmetry"
            )

        electron_channel = self.representation.basis1.index(l_e, 0)
        pair_channel = self.representation.basis2.index(l_R, 0)
        _, electron_radial = lowest_weighted_eigenstate(
            self.representation.h1[electron_channel],
            self.representation.grid1.weights,
        )
        _, pair_radial = lowest_weighted_eigenstate(
            self.representation.h2[pair_channel],
            self.representation.grid2.weights,
        )
        radial_product = electron_radial[:, None] * pair_radial[None, :]
        state = np.zeros(self.representation.psi_shape, dtype=radial_product.dtype)

        if l_e == l_R == 1 and self.inversion_parity == 1:
            # Pure L_total=0, M_total=0 coupling of two l=1 coordinates.
            for m_e in range(-1, 2):
                m_R = -m_e
                coefficient = (-1.0) ** (1 - m_e) / np.sqrt(3.0)
                i = self.representation.basis1.index(1, m_e)
                j = self.representation.basis2.index(1, m_R)
                state[i, j] = coefficient * radial_product
        else:
            state[electron_channel, pair_channel] = radial_product
        return state

    def _validated_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state)
        if state.shape != self.representation.psi_shape:
            raise ValueError(
                f"state has shape {state.shape}; "
                f"expected {self.representation.psi_shape}"
            )
        if not np.issubdtype(state.dtype, np.number):
            raise TypeError("state must contain numeric values")
        if not np.isfinite(state).all():
            raise ValueError("state must contain only finite values")
        return state

    def _diagonal_angular_expectation(self, state, value):
        state = self._validated_state(state)
        weights = np.empty(
            (
                self.representation.basis1.n_channels,
                self.representation.basis2.n_channels,
            )
        )
        for i, (l_e, m_e) in enumerate(self.representation.basis1.states):
            for j, (l_R, m_R) in enumerate(self.representation.basis2.states):
                weights[i, j] = value(l_e, l_R, m_e, m_R)
        norm_squared = self.representation.inner_product(state, state).real
        if norm_squared <= 0.0:
            raise ValueError("state has zero norm")
        action = weights[:, :, None, None] * state
        expectation = self.representation.inner_product(state, action)
        return float(expectation.real / norm_squared)


def _diatomic_parameters(system: str, codata_set: str) -> DiatomicIonParameters:
    if not isinstance(system, str):
        raise TypeError("system must be a string")
    particles = predefined_3body_systems(system)
    try:
        name = NonBornOppenheimerDiatomicIonGroundState._CANONICAL_NAMES[particles]
    except KeyError as exc:
        supported = ", ".join(
            NonBornOppenheimerDiatomicIonGroundState._CANONICAL_NAMES.values()
        )
        raise ValueError(
            f"system must be a supported diatomic ion: {supported}"
        ) from exc
    values = get_reduced_masses_and_charges_3body(
        particles=particles,
        codata_set=codata_set,
        frame="nuclear_center",
    )
    (
        mu_R,
        mu_e,
        xi_R,
        xi_e,
        mass_ref,
        mass_partner,
        mass_electron,
        charge_ref,
        charge_partner,
        charge_electron,
    ) = values
    pair_mass = mass_ref + mass_partner
    return DiatomicIonParameters(
        name=name,
        particles=particles,
        codata_set=codata_set,
        internuclear_reduced_mass=mu_R,
        electronic_reduced_mass=mu_e,
        internuclear_reduced_charge=xi_R,
        electronic_reduced_charge=xi_e,
        reference_mass=mass_ref,
        partner_mass=mass_partner,
        electron_mass=mass_electron,
        reference_charge=charge_ref,
        partner_charge=charge_partner,
        electron_charge=charge_electron,
        reference_center_scale=-mass_partner / pair_mass,
        partner_center_scale=mass_ref / pair_mass,
    )


def _pair_exchange_parity(value, parameters: DiatomicIonParameters):
    if not parameters.has_identical_pair:
        if value is not None:
            raise ValueError(
                f"pair_exchange_parity is not defined for {parameters.name}: "
                "the particles defining R are distinguishable; use None"
            )
        return None
    if value is None:
        raise ValueError(f"pair_exchange_parity must be +1 or -1 for {parameters.name}")
    return _parity(value, "pair_exchange_parity")


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
