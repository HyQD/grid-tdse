"""Midpoint basis integrated on a multicenter Becke grid."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    diagonalize_fixed_m,
)
from ..one_radial_becke import (
    BeckeMulticenterQuadrature,
    BeckePartition,
    FEMDVRQuadratureSpec,
    solve_generalized_hermitian,
)
from .h2plus import _RadialBasis, _local_basis_values


@dataclass(frozen=True)
class MidpointBeckeResult:
    """Fixed midpoint basis with structured and Becke-integrated energies."""

    structured_energy: float
    becke_energy: float
    energy_difference: float
    basis_size: int
    quadrature_size: int
    assembly_seconds: float
    overlap_error: float
    hamiltonian_error: float
    overlap: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray


def _relative_error(candidate, reference):
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def solve_midpoint_h2plus_on_becke_grid(
    *,
    internuclear_distance=2.0,
    radial_nodes=tuple(float(value) for value in range(9)),
    radial_n_points=8,
    l_max=2,
    integration_spacing=0.5,
    integration_radial_n_points=10,
    lebedev_order=41,
    rotation=None,
):
    """Keep the midpoint basis fixed and change only its matrix quadrature."""

    distance = float(internuclear_distance)
    half_distance = 0.5 * distance
    system = OneElectronDiatomicSystem(1.0, 1.0, half_distance, name="H2+")
    basis_specification = FEMDVRQuadratureSpec(
        radial_nodes,
        [int(radial_n_points)] * (len(radial_nodes) - 1),
    )
    structured_hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=basis_specification.nodes,
        n_points=basis_specification.n_points,
        l_max=int(l_max),
        m_max=0,
        nuclear_L_max=2 * int(l_max),
    )
    structured_solution = diagonalize_fixed_m(structured_hamiltonian, m=0)

    radial_basis = _RadialBasis(basis_specification)
    angular_size = int(l_max) + 1
    basis_size = radial_basis.size * angular_size
    centers = np.array(
        [[0.0, 0.0, -half_distance], [0.0, 0.0, half_distance]]
    )
    integration_radius = basis_specification.r_max + half_distance
    n_intervals = int(np.ceil(integration_radius / integration_spacing))
    integration_nodes = np.linspace(0.0, integration_radius, n_intervals + 1)
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(centers),
        FEMDVRQuadratureSpec(
            integration_nodes,
            [int(integration_radial_n_points)] * n_intervals,
        ),
        int(lebedev_order),
        rotation=rotation,
    )

    overlap = np.zeros((basis_size, basis_size))
    kinetic = np.zeros_like(overlap)
    potential = np.zeros_like(overlap)
    started = perf_counter()

    for shell in quadrature.iter_shells():
        points = shell.points
        radii = np.linalg.norm(points, axis=1)
        elements = radial_basis.locate(radii)
        for element_index in np.unique(elements[elements >= 0]):
            selected = elements == element_index
            selected_radii = radii[selected]
            unit_vectors = points[selected] / selected_radii[:, np.newaxis]
            values, gradients, indices = _local_basis_values(
                radial_basis,
                int(element_index),
                selected_radii,
                unit_vectors,
                int(l_max),
                0,
            )
            weights = shell.weights[selected]
            distances = np.linalg.norm(
                points[selected, np.newaxis, :]
                - centers[np.newaxis, :, :],
                axis=-1,
            )
            nuclear_potential = -np.sum(1.0 / distances, axis=1)
            block = np.ix_(indices, indices)
            overlap[block] += values.T @ (weights[:, np.newaxis] * values)
            kinetic[block] += 0.5 * sum(
                gradients[:, :, component].T
                @ (weights[:, np.newaxis] * gradients[:, :, component])
                for component in range(3)
            )
            potential[block] += values.T @ (
                (weights * nuclear_potential)[:, np.newaxis] * values
            )

    hamiltonian = kinetic + potential
    overlap = 0.5 * (overlap + overlap.T)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    solution = solve_generalized_hermitian(
        hamiltonian,
        overlap,
        relative_cutoff=1.0e-12,
        number_of_states=1,
    )

    metric = structured_hamiltonian.fixed_m_metric_weights(0)
    structured_overlap = np.diag(metric)
    structured_matrix = (
        metric[:, np.newaxis]
        * structured_hamiltonian.fixed_m_matrix(0)
    )
    becke_energy = float(solution.energies[0])
    structured_energy = structured_solution.electronic_energy
    for matrix in (overlap, kinetic, potential):
        matrix.setflags(write=False)
    return MidpointBeckeResult(
        structured_energy=structured_energy,
        becke_energy=becke_energy,
        energy_difference=becke_energy - structured_energy,
        basis_size=basis_size,
        quadrature_size=quadrature.point_count,
        assembly_seconds=perf_counter() - started,
        overlap_error=_relative_error(overlap, structured_overlap),
        hamiltonian_error=_relative_error(hamiltonian, structured_matrix),
        overlap=overlap,
        kinetic=kinetic,
        potential=potential,
    )
