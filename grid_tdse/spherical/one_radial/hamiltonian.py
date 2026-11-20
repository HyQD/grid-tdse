"""Hamiltonians for one-radial-coordinate spherical systems."""

from numbers import Integral
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates import (
    SphericalMultipolePotential,
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
    boundary conditions. During the initial implementation, the common
    nuclear radius must be an internal finite-element boundary.
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
        self.nuclear_M_max = 0
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
        self.r.setflags(write=False)
        self.weights.setflags(write=False)
        self.second_derivative.setflags(write=False)

        self.potential = self._build_nuclear_potential()
        self.orbital_lm = self.potential.orbital_lm
        self.orbital_shape = self.potential.orbital_shape
        self._z_angular = self._build_z_angular_matrix()
        self._fixed_m_z_angular = {}
        self._centrifugal = np.asarray(
            [l * (l + 1) for l, _ in self.orbital_lm], dtype=float
        )[None, :] / self.r[:, None] ** 2
        self._centrifugal.setflags(write=False)

    def _build_z_angular_matrix(self):
        r"""Return the sparse matrix of ``cos(theta)`` in channel order."""

        channel_index = {
            channel: index for index, channel in enumerate(self.orbital_lm)
        }
        rows = []
        columns = []
        coefficients = []
        for column, (l_value, m_value) in enumerate(self.orbital_lm):
            output = (l_value + 1, m_value)
            if output not in channel_index:
                continue
            row = channel_index[output]
            coefficient = np.sqrt(
                ((l_value + 1) ** 2 - m_value**2)
                / ((2 * l_value + 1) * (2 * l_value + 3))
            )
            rows.extend((row, column))
            columns.extend((column, row))
            coefficients.extend((coefficient, coefficient))
        n_channels = len(self.orbital_lm)
        return csr_matrix(
            (coefficients, (rows, columns)),
            shape=(n_channels, n_channels),
        )

    def _fixed_m_channel_indices(self, m):
        m = self._validated_m(m)
        return np.asarray(
            [
                index
                for index, (_, magnetic) in enumerate(self.orbital_lm)
                if magnetic == m
            ],
            dtype=int,
        )

    def _fixed_m_z_angular_matrix(self, m):
        m = self._validated_m(m)
        if m not in self._fixed_m_z_angular:
            indices = self._fixed_m_channel_indices(m)
            self._fixed_m_z_angular[m] = self._z_angular[indices][
                :, indices
            ].tocsr()
        return self._fixed_m_z_angular[m]

    def _apply_z_angular(self, values, angular):
        flat = values.reshape(-1, values.shape[-1])
        transformed = angular.dot(flat.T).T.reshape(values.shape)
        radial_shape = (1,) * (values.ndim - 2) + (self.r.size, 1)
        return transformed * self.r.reshape(radial_shape)

    def _validate_nuclear_element_boundary(self, nodes) -> None:
        scale = max(
            1.0,
            abs(self.system.half_bond_length),
            float(np.max(np.abs(nodes))),
        )
        relative_tolerance = 64.0 * np.finfo(float).eps
        absolute_tolerance = relative_tolerance * scale
        if not np.any(
            np.isclose(
                nodes[1:-1],
                self.system.half_bond_length,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        ):
            raise ValueError(
                "half_bond_length must coincide with an internal FEM "
                "element boundary"
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
        return self._apply_z_angular(values, self._z_angular)

    def _validated_m(self, m):
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
            raise TypeError("m must be an integer")
        m = int(m)
        if abs(m) > self.m_max:
            raise ValueError(f"abs(m) must not exceed {self.m_max}")
        return m

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

    def apply_fixed_m(self, orbitals, m):
        """Apply one conserved signed-``m`` Hamiltonian block."""

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
        return self._apply_z_angular(
            values,
            self._fixed_m_z_angular_matrix(m),
        )

    def fixed_m_metric_weights(self, m):
        """Return diagonal metric entries in flattened radial-first order."""

        n_l = self.fixed_m_shape(m)[1]
        return np.repeat(self.weights, n_l)

    def fixed_m_matrix(self, m, *, representation="strong"):
        """Return an explicit strong or Euclidean-Hermitian block matrix."""

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
