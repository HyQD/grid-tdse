"""Hamiltonians for one-radial-coordinate spherical systems."""

from numbers import Integral
from typing import Literal

import numpy as np

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates import (
    SphericalMultipolePotential,
    SphericalReducedRadialOperators,
    radial_Coulomb,
)

from grid_tdse.systems import OneElectronDiatomicSystem

NuclearPotentialMethod = Literal["analytic", "poisson"]


def _angular_cutoff(value, name, *, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must not exceed {upper}")
    return result


def _femdvr_parameters(nodes, n_points):
    nodes = np.asarray(nodes, dtype=float)
    raw_n_points = np.asarray(n_points)
    if nodes.ndim != 1 or nodes.size < 3:
        raise ValueError(
            "nodes must contain at least three element boundaries"
        )
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must contain only finite values")
    if nodes[0] != 0.0 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("nodes must start at zero and be strictly increasing")
    if raw_n_points.shape != (nodes.size - 1,):
        raise ValueError("n_points must contain one entry per finite element")
    if not np.issubdtype(raw_n_points.dtype, np.integer):
        raise TypeError("n_points must contain integers")
    n_points = raw_n_points.astype(int, copy=False)
    if np.any(n_points < 2):
        raise ValueError("each finite element must contain at least two points")
    return nodes, n_points


class OneElectronDiatomicHamiltonian:
    r"""Clamped two-centre Hamiltonian in a spherical FEM-DVR basis.

    The reduced radial orbital convention is

    .. math::

        \psi(\mathbf r)=\frac{1}{r}\sum_{lm}u_{lm}(r)Y_{lm}(\Omega).

    Orbital arrays have radial-first trailing shape ``(n_r, n_lm)``. The
    first and last FEM-DVR nodes are removed to impose homogeneous radial
    boundary conditions. Every distinct nonzero nuclear radius must be an
    internal finite-element boundary.
    """

    _NUCLEAR_METHODS = {"analytic", "poisson"}

    def __init__(
        self,
        system: OneElectronDiatomicSystem,
        *,
        nodes,
        n_points,
        l_max,
        m_max=None,
        nuclear_L_max=None,
        nuclear_M_max=None,
        nuclear_method: NuclearPotentialMethod = "analytic",
    ) -> None:
        if not isinstance(system, OneElectronDiatomicSystem):
            raise TypeError("system must be a OneElectronDiatomicSystem")
        self.system = system
        self.l_max = _angular_cutoff(l_max, "l_max")
        if m_max is None:
            m_max = self.l_max
        self.m_max = _angular_cutoff(m_max, "m_max", upper=self.l_max)
        if nuclear_L_max is None:
            nuclear_L_max = 2 * self.l_max
        self.nuclear_L_max = _angular_cutoff(
            nuclear_L_max, "nuclear_L_max"
        )
        if nuclear_M_max is None:
            nuclear_M_max = (
                0
                if system.is_z_axis_aligned
                else min(self.nuclear_L_max, 2 * self.m_max)
            )
        self.nuclear_M_max = _angular_cutoff(
            nuclear_M_max,
            "nuclear_M_max",
            upper=self.nuclear_L_max,
        )
        if nuclear_method not in self._NUCLEAR_METHODS:
            choices = ", ".join(sorted(self._NUCLEAR_METHODS))
            raise ValueError(f"nuclear_method must be one of: {choices}")
        self.nuclear_method = nuclear_method

        nodes, n_points = _femdvr_parameters(nodes, n_points)
        self._validate_nuclear_element_boundary(nodes)
        self.nodes = tuple(float(value) for value in nodes)
        self.n_points = tuple(int(value) for value in n_points)
        self.dvr = FEMDVR(
            nodes,
            n_points,
            Linear_map,
            GaussLegendreLobatto,
        )

        self.r = np.array(self.dvr.r[1:-1], copy=True)
        self.weights = np.array(self.dvr.weights[1:-1], copy=True)
        self.second_derivative = np.array(
            self.dvr.D2[1:-1, 1:-1], copy=True
        )
        self._coordinate_operators = SphericalReducedRadialOperators(
            self.r,
            self.dvr.D1[1:-1, 1:-1],
            l_max=self.l_max,
            m_max=self.m_max,
        )
        self.first_derivative = self._coordinate_operators.first_derivative
        self.r.setflags(write=False)
        self.weights.setflags(write=False)
        self.second_derivative.setflags(write=False)

        self.potential = self._build_nuclear_potential()
        self.orbital_lm = self.potential.orbital_lm
        self.orbital_shape = self.potential.orbital_shape
        if (
            self.orbital_lm != self._coordinate_operators.orbital_lm
            or self.orbital_shape != self._coordinate_operators.orbital_shape
        ):
            raise RuntimeError(
                "grid-lib spherical operator and multipole-potential "
                "channel orderings are incompatible"
            )
        self._centrifugal = np.asarray(
            [l * (l + 1) for l, _ in self.orbital_lm], dtype=float
        )[None, :] / self.r[:, None] ** 2
        self._centrifugal.setflags(write=False)

    def _validate_nuclear_element_boundary(self, nodes) -> None:
        nuclear_radii = np.linalg.norm(
            self.system.nuclear_positions,
            axis=1,
        )
        scale = max(
            1.0,
            float(np.max(nuclear_radii)),
            float(np.max(np.abs(nodes))),
        )
        relative_tolerance = 64.0 * np.finfo(float).eps
        absolute_tolerance = relative_tolerance * scale
        missing = []
        for radius in nuclear_radii:
            if np.isclose(
                radius,
                0.0,
                rtol=0.0,
                atol=absolute_tolerance,
            ):
                continue
            if not np.any(
                np.isclose(
                    nodes[1:-1],
                    radius,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
            ):
                missing.append(float(radius))
        if missing:
            raise ValueError(
                "every nonzero nuclear radius must coincide with an "
                "internal FEM element boundary; missing radii: "
                f"{missing}"
            )

    def _build_nuclear_potential(self):
        arguments = dict(
            l_max=self.l_max,
            m_max=self.m_max,
            L_max=self.nuclear_L_max,
            M_max=self.nuclear_M_max,
        )
        if self.nuclear_method == "analytic":
            return SphericalMultipolePotential.from_point_charges(
                self.r,
                self.system.nuclear_positions,
                self.system.nuclear_charges,
                **arguments,
            )

        radial_coulomb = radial_Coulomb(
            self.dvr, self.nuclear_L_max + 1
        )
        return SphericalMultipolePotential.from_point_charges_poisson(
            self.r,
            radial_coulomb,
            self.weights,
            self.system.nuclear_positions,
            self.system.nuclear_charges,
            **arguments,
        )

    def _validated_orbitals(self, orbitals):
        values = np.asarray(orbitals)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("orbitals must be numeric")
        if values.ndim < 2 or values.shape[-2:] != self.orbital_shape:
            raise ValueError(
                "orbitals must have trailing shape " f"{self.orbital_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("orbitals must contain only finite values")
        return values

    def apply(self, orbitals):
        """Apply the electronic Hamiltonian to full angular expansions."""

        values = self._validated_orbitals(orbitals)
        second_derivative = np.einsum(
            "ab,...bj->...aj",
            self.second_derivative,
            values,
            optimize=True,
        )
        kinetic = -0.5 / self.system.particle_mass * (
            second_derivative - self._centrifugal * values
        )
        return kinetic + self.potential.apply(values)

    __call__ = apply

    def apply_z(self, orbitals):
        r"""Apply ``z = r cos(theta)`` to full angular expansions."""

        values = self._validated_orbitals(orbitals)
        return self._coordinate_operators.apply_z(values)

    def apply_x(self, orbitals):
        r"""Apply ``x = r sin(theta) cos(phi)`` to full expansions."""

        values = self._validated_orbitals(orbitals)
        return self._coordinate_operators.apply_x(values)

    def apply_y(self, orbitals):
        r"""Apply ``y = r sin(theta) sin(phi)`` to full expansions."""

        values = self._validated_orbitals(orbitals)
        return self._coordinate_operators.apply_y(values)

    def apply_pz(self, orbitals):
        r"""Apply ``p_z = -i partial_z`` to full angular expansions."""

        values = self._validated_orbitals(orbitals)
        return self._coordinate_operators.apply_pz(values)

    def apply_px(self, orbitals):
        r"""Apply ``p_x = -i partial_x`` to full expansions."""

        values = self._validated_orbitals(orbitals)
        return self._coordinate_operators.apply_px(values)

    def apply_py(self, orbitals):
        r"""Apply ``p_y = -i partial_y`` to full expansions."""

        values = self._validated_orbitals(orbitals)
        return self._coordinate_operators.apply_py(values)

    def _validated_m(self, m):
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
            raise TypeError("m must be an integer")
        m = int(m)
        if abs(m) > self.m_max:
            raise ValueError(f"abs(m) must not exceed {self.m_max}")
        return m

    def validate_m(self, m):
        """Return a validated signed magnetic quantum number."""

        return self._validated_m(m)

    def fixed_m_shape(self, m):
        """Return the radial-first state shape for signed magnetic number ``m``."""

        m = self._validated_m(m)
        return (self.r.size, self.l_max - abs(m) + 1)

    def _validated_fixed_m_orbitals(self, orbitals, m):
        shape = self.fixed_m_shape(m)
        values = np.asarray(orbitals)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("orbitals must be numeric")
        if values.ndim < 2 or values.shape[-2:] != shape:
            raise ValueError(f"orbitals must have trailing shape {shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("orbitals must contain only finite values")
        return values

    def validate_fixed_m_state(self, state, m=0):
        """Return a validated radial-first fixed-``m`` state array."""

        return self._validated_fixed_m_orbitals(state, m)

    def validate_state(self, state):
        """Return a validated full-angular radial-first state array."""

        return self._validated_orbitals(state)

    def _require_axisymmetric_geometry(self) -> None:
        if not self.system.is_z_axis_aligned:
            raise ValueError(
                "fixed-m Hamiltonian blocks require all nuclei on the "
                "Cartesian z axis"
            )

    def apply_fixed_m(self, orbitals, m):
        """Apply one conserved signed-``m`` Hamiltonian block."""

        self._require_axisymmetric_geometry()
        m = self._validated_m(m)
        values = self._validated_fixed_m_orbitals(orbitals, m)
        l_values = np.arange(abs(m), self.l_max + 1)
        centrifugal = l_values * (l_values + 1)
        centrifugal = centrifugal[None, :] / self.r[:, None] ** 2
        second_derivative = np.einsum(
            "ab,...bj->...aj",
            self.second_derivative,
            values,
            optimize=True,
        )
        kinetic = -0.5 / self.system.particle_mass * (
            second_derivative - centrifugal * values
        )
        potential = np.einsum(
            "rij,...rj->...ri",
            self.potential.fixed_m_matrix(m),
            values,
            optimize=True,
        )
        return kinetic + potential

    def apply_z_fixed_m(self, orbitals, m=0):
        r"""Apply ``z = r cos(theta)`` in one signed-``m`` block."""

        m = self._validated_m(m)
        values = self._validated_fixed_m_orbitals(orbitals, m)
        return self._coordinate_operators.apply_z_fixed_m(values, m)

    def apply_pz_fixed_m(self, orbitals, m=0):
        r"""Apply ``p_z = -i partial_z`` in one signed-``m`` block."""

        m = self._validated_m(m)
        values = self._validated_fixed_m_orbitals(orbitals, m)
        return self._coordinate_operators.apply_pz_fixed_m(values, m)

    def fixed_m_metric_weights(self, m):
        """Return diagonal metric entries in flattened radial-first order."""

        n_l = self.fixed_m_shape(m)[1]
        return np.repeat(self.weights, n_l)

    def fixed_m_matrix(self, m, *, representation="strong"):
        """Return an explicit strong or Euclidean-Hermitian block matrix."""

        self._require_axisymmetric_geometry()
        m = self._validated_m(m)
        if representation not in {"strong", "symmetric"}:
            raise ValueError("representation must be 'strong' or 'symmetric'")
        n_r, n_l = self.fixed_m_shape(m)
        l_values = np.arange(abs(m), self.l_max + 1)
        kinetic = -0.5 / self.system.particle_mass * np.kron(
            self.second_derivative, np.eye(n_l)
        )
        centrifugal = (
            l_values[None, :] * (l_values[None, :] + 1)
            / self.r[:, None] ** 2
        )
        kinetic += np.diag(
            0.5 / self.system.particle_mass * centrifugal.ravel()
        )

        matrix = kinetic.astype(
            np.result_type(kinetic.dtype, self.potential.coefficients.dtype),
            copy=False,
        )
        pointwise = self.potential.fixed_m_matrix(m)
        for radial_index in range(n_r):
            indices = radial_index * n_l + np.arange(n_l)
            matrix[np.ix_(indices, indices)] += pointwise[radial_index]

        if representation == "strong":
            return matrix
        metric_sqrt = np.sqrt(self.fixed_m_metric_weights(m))
        return metric_sqrt[:, None] * matrix / metric_sqrt[None, :]

    def to_symmetric_fixed_m(self, orbitals, m):
        """Map nodal fixed-``m`` values to the Euclidean representation."""

        values = self._validated_fixed_m_orbitals(orbitals, m)
        weight_shape = (1,) * (values.ndim - 2) + (self.r.size, 1)
        return values * np.sqrt(self.weights).reshape(weight_shape)

    def from_symmetric_fixed_m(self, orbitals, m):
        """Map Euclidean fixed-``m`` values to nodal coefficients."""

        values = self._validated_fixed_m_orbitals(orbitals, m)
        weight_shape = (1,) * (values.ndim - 2) + (self.r.size, 1)
        return values / np.sqrt(self.weights).reshape(weight_shape)

    def metric_weights(self):
        """Return diagonal metric entries in flattened full-state order."""

        return np.repeat(self.weights, self.orbital_shape[1])

    def matrix(self, *, representation="strong"):
        """Return an explicit full-angular matrix for small-grid oracles."""

        if representation not in {"strong", "symmetric"}:
            raise ValueError("representation must be 'strong' or 'symmetric'")
        n_r, n_lm = self.orbital_shape
        kinetic = -0.5 / self.system.particle_mass * np.kron(
            self.second_derivative,
            np.eye(n_lm),
        )
        kinetic += np.diag(
            (
                0.5
                / self.system.particle_mass
                * self._centrifugal
            ).ravel()
        )
        matrix = kinetic.astype(
            np.result_type(
                kinetic.dtype,
                self.potential.coefficients.dtype,
            ),
            copy=False,
        )
        pointwise = self.potential.angular_matrix()
        for radial_index in range(n_r):
            indices = radial_index * n_lm + np.arange(n_lm)
            matrix[np.ix_(indices, indices)] += pointwise[radial_index]
        if representation == "strong":
            return matrix
        metric_sqrt = np.sqrt(self.metric_weights())
        return metric_sqrt[:, None] * matrix / metric_sqrt[None, :]

    def diagonal(self) -> np.ndarray:
        """Return the strong full-angular diagonal in state shape."""

        radial_kinetic = (
            -0.5
            / self.system.particle_mass
            * np.diag(self.second_derivative)[:, None]
        )
        centrifugal = (
            0.5 / self.system.particle_mass * self._centrifugal
        )
        potential = np.diagonal(
            self.potential.angular_matrix(),
            axis1=1,
            axis2=2,
        )
        result = np.asarray(
            radial_kinetic + centrifugal + potential
        ).copy()
        result.setflags(write=False)
        return result

    def to_symmetric(self, orbitals):
        """Map full nodal states to Euclidean symmetric coefficients."""

        values = self._validated_orbitals(orbitals)
        weight_shape = (1,) * (values.ndim - 2) + (self.r.size, 1)
        return values * np.sqrt(self.weights).reshape(weight_shape)

    def from_symmetric(self, orbitals):
        """Map full Euclidean coefficients to physical nodal states."""

        values = self._validated_orbitals(orbitals)
        weight_shape = (1,) * (values.ndim - 2) + (self.r.size, 1)
        return values / np.sqrt(self.weights).reshape(weight_shape)

    def inner_product(self, left, right):
        """Return the physical inner product for radial-first arrays."""

        left = np.asarray(left)
        right = np.asarray(right)
        if left.shape != right.shape:
            raise ValueError("left and right must have the same shape")
        if left.ndim < 2 or left.shape[-2] != self.r.size:
            raise ValueError("the penultimate axis must be the radial grid")
        weight_shape = (1,) * (left.ndim - 2) + (self.r.size, 1)
        return np.sum(
            left.conj() * right * self.weights.reshape(weight_shape)
        )

    def inner_products(self, left, right) -> np.ndarray:
        """Return overlaps of several left states with one right state.

        The batched operation avoids one Python callback per Krylov basis
        vector during Arnoldi and Lanczos orthogonalization.
        """

        right_values = np.asarray(right)
        if right_values.ndim != 2:
            raise ValueError("right must contain exactly one state")
        if right_values.shape[-2] != self.r.size:
            raise ValueError("the penultimate axis must be the radial grid")
        if not np.issubdtype(right_values.dtype, np.number):
            raise TypeError("right must contain numeric values")

        left_values = np.asarray(left)
        if left_values.ndim != 3:
            raise ValueError("left must be a sequence of states")
        if left_values.shape[1:] != right_values.shape:
            raise ValueError(
                "left states and right must have the same state shape"
            )
        if not np.issubdtype(left_values.dtype, np.number):
            raise TypeError("left must contain numeric values")
        if not np.all(np.isfinite(left_values)) or not np.all(
            np.isfinite(right_values)
        ):
            raise ValueError("left and right must contain only finite values")

        weighted_right = (right_values * self.weights[:, None]).reshape(-1)
        return left_values.reshape((left_values.shape[0], -1)).conj() @ (
            weighted_right
        )

    def archive_metadata(self) -> dict:
        """Return JSON-compatible system and discretization metadata."""

        return {
            "coordinate_system": "spherical_one_radial",
            "nodes": list(self.nodes),
            "n_points": list(self.n_points),
            "active_radial_points": int(self.r.size),
            "radial_boundary_conditions": "homogeneous_dirichlet",
            "l_max": self.l_max,
            "m_max": self.m_max,
            "nuclear_L_max": self.nuclear_L_max,
            "nuclear_M_max": self.nuclear_M_max,
            "nuclear_potential_method": self.nuclear_method,
            "metric_representation": "femdvr_nodal_diagonal_overlap",
            "state_axis_order": "radial_angular",
            "nuclear_charge1": self.system.nuclear_charge1,
            "nuclear_charge2": self.system.nuclear_charge2,
            "nuclear_positions": self.system.nuclear_positions.tolist(),
            "internuclear_distance": self.system.internuclear_distance,
            "nuclear_midpoint": self.system.nuclear_midpoint.tolist(),
            "bond_axis": self.system.bond_axis.tolist(),
            "particle_mass": self.system.particle_mass,
            "system_name": self.system.name,
            "nuclear_repulsion_energy": (
                self.system.nuclear_repulsion_energy
            ),
            "public_state_representation": "physical_nodal",
        }
