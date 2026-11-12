"""Transparent atom-centered H2+ reference implementation.

Becke weights are used only to partition numerical integration.  The spatial
basis is the ordinary superposition of unweighted atom-centered spherical
FEM-DVR functions.  Points are grouped by their active radial element on each
center, so dense matrix products contain only locally active basis functions.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)
from scipy.integrate import lebedev_rule
from scipy.special import sph_harm_y

from ..one_radial_becke import (
    BeckeMulticenterQuadrature,
    BeckePartition,
    FEMDVRQuadratureSpec,
    solve_generalized_hermitian,
)
from ..one_radial_becke.ground_state import GeneralizedEigenResult


@dataclass(frozen=True)
class AtomicH2PlusMatrices:
    """Dense matrices and diagnostics for an unweighted atomic basis."""

    overlap: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    hamiltonian: np.ndarray
    basis_size: int
    quadrature_size: int
    element_group_count: int
    overlap_condition_number: float
    assembly_seconds: float


@dataclass(frozen=True)
class AtomicH2PlusResult:
    """Ground state of clamped-nuclei H2+ in the atomic superposition."""

    matrices: AtomicH2PlusMatrices
    solution: GeneralizedEigenResult
    electronic_energy: float
    total_energy: float


class _RadialBasis:
    """Element-local evaluation of interior FEM-DVR cardinal functions."""

    def __init__(self, specification: FEMDVRQuadratureSpec):
        self.specification = specification
        self.dvr = FEMDVR(
            np.asarray(specification.nodes),
            np.asarray(specification.n_points),
            Linear_map,
            GaussLegendreLobatto,
        )
        self.size = self.dvr.r.size - 2
        self._barycentric_weights = tuple(
            1.0 / element.PN_x2 for element in self.dvr.dvr
        )

    def locate(self, radii):
        """Return the owning element, or -1 outside the radial support."""

        radii = np.asarray(radii)
        nodes = np.asarray(self.specification.nodes)
        elements = np.searchsorted(nodes, radii, side="right") - 1
        elements[radii == nodes[-1]] = self.specification.n_elements - 1
        elements[(radii < nodes[0]) | (radii > nodes[-1])] = -1
        return elements

    def evaluate_element(self, element_index, radii):
        """Evaluate only the cardinal functions belonging to one element."""

        element = self.dvr.dvr[element_index]
        left = self.specification.nodes[element_index]
        right = self.specification.nodes[element_index + 1]
        reference_points = 2.0 * (np.asarray(radii) - left) / (right - left) - 1.0
        nodes = element.x
        difference = reference_points[:, np.newaxis] - nodes[np.newaxis, :]
        close = np.abs(difference) <= 128.0 * np.finfo(float).eps
        on_node = np.any(close, axis=1)

        values = np.zeros_like(difference)
        derivatives = np.zeros_like(difference)
        regular = ~on_node
        if np.any(regular):
            regular_difference = difference[regular]
            barycentric = self._barycentric_weights[element_index]
            scaled = barycentric[np.newaxis, :] / regular_difference
            denominator = np.sum(scaled, axis=1)
            regular_values = scaled / denominator[:, np.newaxis]
            second_sum = np.sum(
                barycentric[np.newaxis, :] / regular_difference**2,
                axis=1,
            )
            values[regular] = regular_values
            derivatives[regular] = regular_values * (
                second_sum[:, np.newaxis] / denominator[:, np.newaxis]
                - 1.0 / regular_difference
            ) * (2.0 / (right - left))

        if np.any(on_node):
            rows = np.flatnonzero(on_node)
            local_nodes = np.argmax(close[on_node], axis=1)
            values[rows, local_nodes] = 1.0
            derivatives[rows] = element.D1[local_nodes]

        local_to_global = np.asarray(self.dvr.element_dof_maps[element_index])
        keep = (local_to_global != 0) & (
            local_to_global != self.dvr.r.size - 1
        )
        radial_indices = local_to_global[keep] - 1
        return values[:, keep], derivatives[:, keep], radial_indices


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _integration_grid(radial_nodes, distance, maximum_spacing):
    """Cover the union of all finite atomic basis supports."""

    nodes = [float(radial_nodes[0])]
    parents = []
    intervals = list(zip(radial_nodes[:-1], radial_nodes[1:]))
    intervals.append((radial_nodes[-1], radial_nodes[-1] + distance))
    for parent, (left, right) in enumerate(intervals):
        count = int(np.ceil((right - left) / maximum_spacing))
        subnodes = np.linspace(left, right, count + 1)
        nodes.extend(float(value) for value in subnodes[1:])
        basis_parent = parent if parent < len(radial_nodes) - 1 else -1
        parents.extend([basis_parent] * count)
    return np.asarray(nodes), np.asarray(parents, dtype=int)


def _m0_harmonics_and_gradients(unit_vectors, l_max):
    """Return real Y_l0 values and Cartesian angular gradients."""

    x, y, z = np.asarray(unit_vectors).T
    z = np.clip(z, -1.0, 1.0)
    sin_theta = np.hypot(x, y)
    theta = np.arctan2(sin_theta, z)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    regular = sin_theta > 128.0 * np.finfo(float).eps
    values = np.empty((unit_vectors.shape[0], l_max + 1))
    gradients = np.zeros((unit_vectors.shape[0], l_max + 1, 3))

    for ell in range(l_max + 1):
        values[:, ell] = sph_harm_y(ell, 0, theta, phi).real
        if ell == 0 or not np.any(regular):
            continue
        regular_phi = phi[regular]
        d_theta = (
            np.sqrt(ell * (ell + 1))
            * np.exp(-1j * regular_phi)
            * sph_harm_y(ell, 1, theta[regular], regular_phi)
        ).real
        e_theta = np.column_stack(
            (
                z[regular] * x[regular] / sin_theta[regular],
                z[regular] * y[regular] / sin_theta[regular],
                -sin_theta[regular],
            )
        )
        gradients[regular, ell, :] = d_theta[:, np.newaxis] * e_theta
    return values, gradients


def _quadrature_arrays(quadrature, parent_elements):
    points = []
    weights = []
    owners = []
    owner_parents = []
    for shell in quadrature.iter_shells():
        size = shell.points.shape[0]
        points.append(shell.points)
        weights.append(shell.weights)
        owners.append(np.full(size, shell.center_index, dtype=int))
        owner_parents.append(
            np.full(size, parent_elements[shell.element_index], dtype=int)
        )
    return (
        np.concatenate(points),
        np.concatenate(weights),
        np.concatenate(owners),
        np.concatenate(owner_parents),
    )


def _local_basis_values(
    radial_basis,
    element_index,
    radii,
    unit_vectors,
    l_max,
    center_offset,
):
    radial_values, radial_derivatives, radial_indices = (
        radial_basis.evaluate_element(element_index, radii)
    )
    harmonics, angular_gradients = _m0_harmonics_and_gradients(
        unit_vectors, l_max
    )

    inverse_radii = 1.0 / radii
    radial_factors = radial_values * inverse_radii[:, np.newaxis]
    radial_factor_derivatives = (
        radial_derivatives * inverse_radii[:, np.newaxis]
        - radial_values * inverse_radii[:, np.newaxis] ** 2
    )
    values = radial_factors[:, :, np.newaxis] * harmonics[:, np.newaxis, :]
    gradients = (
        radial_factor_derivatives[:, :, np.newaxis, np.newaxis]
        * harmonics[:, np.newaxis, :, np.newaxis]
        * unit_vectors[:, np.newaxis, np.newaxis, :]
        + radial_values[:, :, np.newaxis, np.newaxis]
        * inverse_radii[:, np.newaxis, np.newaxis, np.newaxis] ** 2
        * angular_gradients[:, np.newaxis, :, :]
    )

    angular_size = l_max + 1
    indices = center_offset + (
        radial_indices[:, np.newaxis] * angular_size
        + np.arange(angular_size)[np.newaxis, :]
    )
    return (
        values.reshape(radii.size, -1),
        gradients.reshape(radii.size, -1, 3),
        indices.reshape(-1),
    )


def solve_atomic_h2plus(
    *,
    internuclear_distance=2.0,
    radial_nodes=(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0),
    radial_n_points=8,
    l_max=2,
    integration_spacing=0.25,
    integration_radial_n_points=10,
    lebedev_order=41,
    switch_iterations=3,
    overlap_cutoff=1.0e-8,
    hermiticity_tolerance=1.0e-11,
    grid_backend="native",
    pyscf_prune="nwchem",
):
    """Solve H2+ with an unweighted atomic basis and Becke quadrature.

    ``grid_backend="pyscf"`` uses the same user-controlled radial element
    layout with interior Gauss-Legendre integration points, while PySCF
    supplies Lebedev grids, angular pruning, Becke weights, and spatial
    grouping.
    """

    distance = _positive_real(internuclear_distance, "internuclear_distance")
    integration_spacing = _positive_real(
        integration_spacing, "integration_spacing"
    )
    radial_n_points = _positive_integer(
        radial_n_points, "radial_n_points", minimum=2
    )
    integration_radial_n_points = _positive_integer(
        integration_radial_n_points,
        "integration_radial_n_points",
        minimum=2,
    )
    l_max = _positive_integer(l_max, "l_max", minimum=0)
    switch_iterations = _positive_integer(
        switch_iterations, "switch_iterations"
    )
    if grid_backend not in ("native", "pyscf"):
        raise ValueError("grid_backend must be 'native' or 'pyscf'")
    if grid_backend == "pyscf" and switch_iterations != 3:
        raise ValueError(
            "PySCF's original Becke scheme uses exactly three switch iterations"
        )

    basis_specification = FEMDVRQuadratureSpec(
        radial_nodes,
        [radial_n_points] * (len(radial_nodes) - 1),
    )
    radial_basis = _RadialBasis(basis_specification)
    centers = np.array(
        [[0.0, 0.0, -0.5 * distance], [0.0, 0.0, 0.5 * distance]]
    )
    integration_nodes, parent_elements = _integration_grid(
        basis_specification.nodes, distance, integration_spacing
    )
    integration_specification = FEMDVRQuadratureSpec(
        integration_nodes,
        [integration_radial_n_points] * parent_elements.size,
    )
    started = perf_counter()
    if grid_backend == "native":
        quadrature = BeckeMulticenterQuadrature(
            BeckePartition(centers, switch_iterations=switch_iterations),
            integration_specification,
            lebedev_order,
        )
        points, weights, owners, owner_parents = _quadrature_arrays(
            quadrature, parent_elements
        )
        quadrature_size = quadrature.point_count
    else:
        from .pyscf_grid import (
            GENERIC_LEBEDEV_ROTATION,
            build_pyscf_becke_grid,
            composite_gauss_legendre,
        )

        radial_data = composite_gauss_legendre(integration_specification)

        def radial_rule(_atom_index, _atomic_number):
            return radial_data

        angular_points = lebedev_rule(lebedev_order)[1].size
        grid = build_pyscf_becke_grid(
            centers,
            [1, 1],
            radial_rule,
            angular_points=angular_points,
            prune=pyscf_prune,
            radii_adjust=None,
            rotation=GENERIC_LEBEDEV_ROTATION,
        )
        points = grid.coordinates
        weights = grid.weights
        owners = grid.origin_atoms
        owner_parents = np.full(points.shape[0], -1, dtype=int)
        quadrature_size = grid.point_count
    displacements = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
    distances = np.linalg.norm(displacements, axis=-1)
    unit_vectors = displacements / distances[..., np.newaxis]
    elements = np.column_stack(
        [radial_basis.locate(distances[:, center]) for center in range(2)]
    )
    rows = np.arange(points.shape[0])
    use_parent = owner_parents >= 0
    elements[rows[use_parent], owners[use_parent]] = owner_parents[use_parent]

    signatures, membership = np.unique(
        elements, axis=0, return_inverse=True
    )
    angular_size = l_max + 1
    center_size = radial_basis.size * angular_size
    basis_size = 2 * center_size
    shape = (basis_size, basis_size)
    overlap = np.zeros(shape)
    kinetic = np.zeros(shape)
    potential = np.zeros(shape)

    for group_index, signature in enumerate(signatures):
        point_indices = np.flatnonzero(membership == group_index)
        if not np.any(signature >= 0):
            continue
        group_values = []
        group_gradients = []
        group_basis_indices = []
        for center_index, element_index in enumerate(signature):
            if element_index < 0:
                continue
            values, gradients, basis_indices = _local_basis_values(
                radial_basis,
                int(element_index),
                distances[point_indices, center_index],
                unit_vectors[point_indices, center_index],
                l_max,
                center_index * center_size,
            )
            group_values.append(values)
            group_gradients.append(gradients)
            group_basis_indices.append(basis_indices)

        values = np.concatenate(group_values, axis=1)
        gradients = np.concatenate(group_gradients, axis=1)
        basis_indices = np.concatenate(group_basis_indices)
        group_weights = weights[point_indices]
        nuclear_potential = -np.sum(1.0 / distances[point_indices], axis=1)

        overlap_block = values.T @ (group_weights[:, np.newaxis] * values)
        kinetic_block = sum(
            gradients[:, :, component].T
            @ (
                group_weights[:, np.newaxis]
                * gradients[:, :, component]
            )
            for component in range(3)
        )
        potential_block = values.T @ (
            (group_weights * nuclear_potential)[:, np.newaxis] * values
        )
        block = np.ix_(basis_indices, basis_indices)
        overlap[block] += overlap_block
        kinetic[block] += 0.5 * kinetic_block
        potential[block] += potential_block

    hamiltonian = kinetic + potential
    for matrix in (overlap, kinetic, potential, hamiltonian):
        matrix[:] = 0.5 * (matrix + matrix.T)
        matrix.setflags(write=False)
    eigenvalues = np.linalg.eigvalsh(overlap)
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    assembly_seconds = perf_counter() - started
    matrices = AtomicH2PlusMatrices(
        overlap=overlap,
        kinetic=kinetic,
        potential=potential,
        hamiltonian=hamiltonian,
        basis_size=basis_size,
        quadrature_size=quadrature_size,
        element_group_count=signatures.shape[0],
        overlap_condition_number=condition_number,
        assembly_seconds=assembly_seconds,
    )
    solution = solve_generalized_hermitian(
        hamiltonian,
        overlap,
        relative_cutoff=overlap_cutoff,
        number_of_states=1,
        hermiticity_tolerance=hermiticity_tolerance,
    )
    electronic_energy = float(solution.energies[0])
    return AtomicH2PlusResult(
        matrices=matrices,
        solution=solution,
        electronic_energy=electronic_energy,
        total_energy=electronic_energy + 1.0 / distance,
    )
