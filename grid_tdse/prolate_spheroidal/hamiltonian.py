"""Field-free one-electron Hamiltonians in prolate coordinates."""

from typing import Literal

import numpy as np
from scipy import sparse

from grid_lib.prolate_spheroidal_coordinates import (
    electron_nuclear_coulomb_matrix,
    kinetic_energy_matrix,
    restrict_matrix,
)

from grid_tdse.systems import OneElectronDiatomicSystem

from .representation import ProlateSpheroidalRepresentation


MatrixRepresentation = Literal["weak", "strong", "symmetric"]


class ProlateSpheroidalOneElectronHamiltonian:
    """Clamped two-centre Hamiltonian in fixed-``m`` sectors.

    Public states are physical nodal values. The weak matrix ``A`` and
    diagonal overlap ``S`` obey ``A c = E S c``. ``apply_fixed_m`` applies
    the strong nodal action ``S**-1 A``; ``apply_symmetric_fixed_m`` applies
    the Euclidean-Hermitian similarity transform.
    """

    _MATRIX_REPRESENTATIONS = {"weak", "strong", "symmetric"}

    def __init__(
        self,
        system: OneElectronDiatomicSystem,
        *,
        xi_boundaries,
        xi_points_per_element=8,
        eta_degree=8,
        m_max=0,
        xi_outer_dirichlet=True,
    ) -> None:
        if not isinstance(system, OneElectronDiatomicSystem):
            raise TypeError("system must be a OneElectronDiatomicSystem")
        self.system = system
        self.representation = ProlateSpheroidalRepresentation(
            internuclear_distance=system.internuclear_distance,
            xi_boundaries=xi_boundaries,
            xi_points_per_element=xi_points_per_element,
            eta_degree=eta_degree,
            m_max=m_max,
            xi_outer_dirichlet=xi_outer_dirichlet,
        )
        self.grid = self.representation.grid
        self.m_max = self.representation.m_max
        self._matrices: dict[tuple[int, str], sparse.csr_matrix] = {}

    def fixed_m_shape(self, m=0) -> tuple[int, int]:
        """Return the active nodal state shape for signed ``m``."""

        return self.representation.fixed_m_shape(m)

    def fixed_m_metric_weights(self, m=0) -> np.ndarray:
        """Return flattened diagonal overlap entries for signed ``m``."""

        return self.representation.metric_weights(m).reshape(-1)

    def _build_weak_matrix(self, m: int) -> sparse.csr_matrix:
        indices = self.representation.active_indices(m)
        kinetic = kinetic_energy_matrix(
            self.grid,
            m=m,
            active_indices=indices,
        )
        kinetic = kinetic / self.system.particle_mass
        potential = restrict_matrix(
            electron_nuclear_coulomb_matrix(
                self.grid,
                Z1=self.system.nuclear_charge1,
                Z2=self.system.nuclear_charge2,
            ),
            indices,
        )
        return (kinetic + potential).tocsr()

    def fixed_m_matrix(
        self,
        m=0,
        *,
        representation: MatrixRepresentation = "strong",
    ) -> sparse.csr_matrix:
        """Return a weak, strong, or symmetric fixed-``m`` matrix."""

        m = self.representation.validate_m(m)
        if representation not in self._MATRIX_REPRESENTATIONS:
            choices = ", ".join(sorted(self._MATRIX_REPRESENTATIONS))
            raise ValueError(f"representation must be one of: {choices}")
        key = (m, representation)
        if key not in self._matrices:
            weak_key = (m, "weak")
            if weak_key not in self._matrices:
                self._matrices[weak_key] = self._build_weak_matrix(m)
            weak = self._matrices[weak_key]
            metric = self.fixed_m_metric_weights(m)
            if representation == "weak":
                matrix = weak
            elif representation == "strong":
                matrix = sparse.diags(1.0 / metric) @ weak
            else:
                inverse_sqrt = sparse.diags(1.0 / np.sqrt(metric))
                matrix = inverse_sqrt @ weak @ inverse_sqrt
            self._matrices[key] = matrix.tocsr()
        return self._matrices[key].copy()

    def _apply_matrix(self, matrix, states, m=0) -> np.ndarray:
        values = self.representation.validate_fixed_m_state(states, m)
        shape = self.fixed_m_shape(m)
        flat = values.reshape((-1, int(np.prod(shape))))
        result = (matrix @ flat.T).T
        return np.asarray(result).reshape(values.shape)

    def apply_weak_fixed_m(self, states, m=0) -> np.ndarray:
        """Apply the weak matrix ``A`` to nodal states."""

        return self._apply_matrix(
            self.fixed_m_matrix(m, representation="weak"), states, m
        )

    def apply_fixed_m(self, states, m=0) -> np.ndarray:
        """Apply the physical strong action ``S**-1 A``."""

        return self._apply_matrix(
            self.fixed_m_matrix(m, representation="strong"), states, m
        )

    def apply_symmetric_fixed_m(self, states, m=0) -> np.ndarray:
        """Apply ``S**-1/2 A S**-1/2`` to Euclidean states."""

        return self._apply_matrix(
            self.fixed_m_matrix(m, representation="symmetric"), states, m
        )

    def fixed_m_diagonal(
        self,
        m=0,
        *,
        representation: MatrixRepresentation = "strong",
    ) -> np.ndarray:
        """Return a matrix diagonal reshaped like one fixed-``m`` state."""

        diagonal = self.fixed_m_matrix(
            m, representation=representation
        ).diagonal()
        result = np.asarray(diagonal).reshape(self.fixed_m_shape(m))
        result.setflags(write=False)
        return result

    def inner_product(self, left, right, m=0) -> complex:
        """Return the physical overlap of two nodal states."""

        return self.representation.inner_product(left, right, m)

    def invert(self, state, m=0) -> np.ndarray:
        """Apply spatial inversion to fixed-``m`` nodal states."""

        return self.representation.invert(state, m)

    def inversion_expectation(self, state, m=0):
        """Return normalized inversion expectations for nodal states."""

        return self.representation.inversion_expectation(state, m)

    def to_symmetric_fixed_m(self, states, m=0) -> np.ndarray:
        """Map nodal states to Euclidean symmetric coefficients."""

        return self.representation.to_euclidean(states, m)

    def from_symmetric_fixed_m(self, states, m=0) -> np.ndarray:
        """Map Euclidean symmetric coefficients to nodal states."""

        return self.representation.from_euclidean(states, m)

    def electronic_energy(self, state, m=0) -> float:
        """Return the field-free electronic Rayleigh quotient."""

        values = self.representation.validate_fixed_m_state(state, m)
        denominator = self.inner_product(values, values, m)
        numerator = self.inner_product(
            values, self.apply_fixed_m(values, m), m
        )
        result = numerator / denominator
        scale = max(1.0, abs(result.real))
        if abs(result.imag) > 1.0e-12 * scale:
            raise ValueError(
                "electronic energy has a significant imaginary part"
            )
        return float(result.real)

    def total_energy(self, state, m=0) -> float:
        """Return electronic energy plus clamped nuclear repulsion."""

        return self.electronic_energy(state, m) + (
            self.system.nuclear_repulsion_energy
        )

    def archive_metadata(self) -> dict:
        """Return JSON-compatible system and discretization metadata."""

        metadata = self.representation.archive_metadata()
        metadata.update(
            {
                "nuclear_charge1": self.system.nuclear_charge1,
                "nuclear_charge2": self.system.nuclear_charge2,
                "particle_mass": self.system.particle_mass,
                "system_name": self.system.name,
                "nuclear_repulsion_energy": (
                    self.system.nuclear_repulsion_energy
                ),
                "public_state_representation": "physical_nodal",
            }
        )
        return metadata
