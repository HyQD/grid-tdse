"""Cross-representation validation helpers for the Becke prototype."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from ..one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
)
from .basis import BeckeSphericalBasis, fixed_m_channels
from .partition import BeckePartition
from .quadrature import BeckeMulticenterQuadrature
from .specification import FEMDVRQuadratureSpec


@dataclass(frozen=True)
class MidpointMatrixSet:
    """Weak-form matrices for one midpoint-centered basis."""

    overlap: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    hamiltonian: np.ndarray


@dataclass(frozen=True)
class MatrixRelativeErrors:
    """Relative Frobenius errors for four matrix components."""

    overlap: float
    kinetic: float
    potential: float
    hamiltonian: float


@dataclass(frozen=True)
class MidpointL0Validation:
    """Phase 6.2 comparison for the midpoint-centered ``l=0`` basis."""

    structured: MidpointMatrixSet
    continuum: MidpointMatrixSet
    partitioned: MidpointMatrixSet
    structured_to_continuum: MatrixRelativeErrors
    partitioned_to_continuum: MatrixRelativeErrors
    midpoint_nodes: tuple[float, ...]
    midpoint_n_points: tuple[int, ...]
    integration_radial_spacing: float
    integration_n_points: int
    lebedev_order: int
    continuum_n_points: int
    partitioned_quadrature_size: int


def _read_only_matrix_set(overlap, kinetic, potential) -> MidpointMatrixSet:
    matrices = []
    for matrix in (overlap, kinetic, potential):
        matrix = np.asarray(matrix)
        matrix = np.array(0.5 * (matrix + matrix.conj().T), copy=True)
        matrix.setflags(write=False)
        matrices.append(matrix)
    hamiltonian = np.array(matrices[1] + matrices[2], copy=True)
    hamiltonian.setflags(write=False)
    return MidpointMatrixSet(
        overlap=matrices[0],
        kinetic=matrices[1],
        potential=matrices[2],
        hamiltonian=hamiltonian,
    )


def _relative_error(candidate, reference) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(candidate - reference))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else np.inf
    return numerator / denominator


def _relative_errors(
    candidate: MidpointMatrixSet,
    reference: MidpointMatrixSet,
) -> MatrixRelativeErrors:
    return MatrixRelativeErrors(
        overlap=_relative_error(candidate.overlap, reference.overlap),
        kinetic=_relative_error(candidate.kinetic, reference.kinetic),
        potential=_relative_error(candidate.potential, reference.potential),
        hamiltonian=_relative_error(
            candidate.hamiltonian, reference.hamiltonian
        ),
    )


def _structured_l0_matrices(
    hamiltonian: OneElectronDiatomicHamiltonian,
) -> MidpointMatrixSet:
    metric = hamiltonian.fixed_m_metric_weights(0)
    overlap = np.diag(metric)
    weak_hamiltonian = metric[:, np.newaxis] * hamiltonian.fixed_m_matrix(0)
    pointwise_potential = hamiltonian.potential.fixed_m_matrix(0)[:, 0, 0]
    potential = np.diag(metric * pointwise_potential)
    kinetic = weak_hamiltonian - potential
    return _read_only_matrix_set(overlap, kinetic, potential)


def _continuum_l0_matrices(
    basis: BeckeSphericalBasis,
    system: OneElectronDiatomicSystem,
    radial_specification: FEMDVRQuadratureSpec,
    radial_order: int,
) -> MidpointMatrixSet:
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition([[0.0, 0.0, 0.0]]),
        FEMDVRQuadratureSpec(
            radial_specification.nodes,
            [radial_order] * radial_specification.n_elements,
        ),
        9,
    )
    shape = (basis.size, basis.size)
    overlap = np.zeros(shape, dtype=complex)
    kinetic = np.zeros(shape, dtype=complex)
    potential = np.zeros(shape, dtype=complex)
    total_nuclear_charge = float(np.sum(system.nuclear_charges))

    for shell in quadrature.iter_shells():
        evaluation = basis.evaluate_shell(shell)
        values = evaluation.values
        weights = shell.weights
        overlap += values.conj().T @ (weights[:, np.newaxis] * values)
        for component in range(3):
            gradients = evaluation.gradients[:, :, component]
            kinetic += 0.5 / system.particle_mass * gradients.conj().T @ (
                weights[:, np.newaxis] * gradients
            )
        angular_average = -total_nuclear_charge / max(
            shell.radius, system.half_bond_length
        )
        potential += values.conj().T @ (
            (weights * angular_average)[:, np.newaxis] * values
        )
    return _read_only_matrix_set(overlap, kinetic, potential)


def _partitioned_l0_matrices(
    basis: BeckeSphericalBasis,
    system: OneElectronDiatomicSystem,
    *,
    radial_spacing: float,
    radial_order: int,
    lebedev_order: int,
) -> tuple[MidpointMatrixSet, BeckeMulticenterQuadrature, float]:
    outer_radius = basis.radial_specs[0].r_max + system.half_bond_length
    n_elements = int(np.ceil(outer_radius / radial_spacing))
    radial_nodes = np.linspace(0.0, outer_radius, n_elements + 1)
    actual_spacing = outer_radius / n_elements
    quadrature = BeckeMulticenterQuadrature(
        BeckePartition(system.nuclear_positions),
        FEMDVRQuadratureSpec(
            radial_nodes, [radial_order] * n_elements
        ),
        lebedev_order,
    )
    shape = (basis.size, basis.size)
    overlap = np.zeros(shape, dtype=complex)
    kinetic = np.zeros(shape, dtype=complex)
    potential = np.zeros(shape, dtype=complex)

    for shell in quadrature.iter_shells():
        evaluation = basis.evaluate(shell.points)
        values = evaluation.values
        weights = shell.weights
        overlap += values.conj().T @ (weights[:, np.newaxis] * values)
        for component in range(3):
            gradients = evaluation.gradients[:, :, component]
            kinetic += 0.5 / system.particle_mass * gradients.conj().T @ (
                weights[:, np.newaxis] * gradients
            )
        distances = np.linalg.norm(
            shell.points[:, np.newaxis, :]
            - system.nuclear_positions[np.newaxis, :, :],
            axis=-1,
        )
        nuclear_potential = -np.sum(
            system.nuclear_charges / distances, axis=1
        )
        potential += values.conj().T @ (
            (weights * nuclear_potential)[:, np.newaxis] * values
        )
    return (
        _read_only_matrix_set(overlap, kinetic, potential),
        quadrature,
        actual_spacing,
    )


def validate_midpoint_l0(
    *,
    system=None,
    midpoint_nodes=(0.0, 1.0, 3.0),
    midpoint_n_points=(3, 3),
    integration_radial_spacing: float = 0.25,
    integration_n_points: int = 12,
    lebedev_order: int = 41,
    continuum_n_points: int = 20,
) -> MidpointL0Validation:
    r"""Compare midpoint ``Y_00`` matrices with multicenter Becke quadrature.

    The continuum reference integrates the same unnormalized FEM cardinal
    functions without DVR mass lumping.  For ``l=0``, the angular average of
    each nuclear Coulomb potential is analytic.  The structured reference is
    the existing midpoint solver's weak matrix, including its diagonal DVR
    overlap and pointwise-potential quadrature.  Reporting both references
    distinguishes multicenter quadrature error from finite-order DVR
    quadrature corrections.
    """

    if system is None:
        system = OneElectronDiatomicSystem(1.0, 1.0, 1.0, name="H2+")
    if not isinstance(system, OneElectronDiatomicSystem):
        raise TypeError("system must be a OneElectronDiatomicSystem")
    if isinstance(integration_radial_spacing, (bool, np.bool_)) or not isinstance(
        integration_radial_spacing, Real
    ):
        raise TypeError("integration_radial_spacing must be a real number")
    integration_radial_spacing = float(integration_radial_spacing)
    if (
        not np.isfinite(integration_radial_spacing)
        or integration_radial_spacing <= 0.0
    ):
        raise ValueError("integration_radial_spacing must be finite and positive")

    radial_specification = FEMDVRQuadratureSpec(
        midpoint_nodes, midpoint_n_points
    )
    midpoint_hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=radial_specification.nodes,
        n_points=radial_specification.n_points,
        l_max=0,
        m_max=0,
        nuclear_L_max=0,
    )
    basis = BeckeSphericalBasis(
        BeckePartition([[0.0, 0.0, 0.0]]),
        radial_specification,
        fixed_m_channels(0, 0),
    )

    structured = _structured_l0_matrices(midpoint_hamiltonian)
    continuum = _continuum_l0_matrices(
        basis, system, radial_specification, continuum_n_points
    )
    partitioned, quadrature, actual_spacing = _partitioned_l0_matrices(
        basis,
        system,
        radial_spacing=integration_radial_spacing,
        radial_order=integration_n_points,
        lebedev_order=lebedev_order,
    )
    return MidpointL0Validation(
        structured=structured,
        continuum=continuum,
        partitioned=partitioned,
        structured_to_continuum=_relative_errors(structured, continuum),
        partitioned_to_continuum=_relative_errors(partitioned, continuum),
        midpoint_nodes=radial_specification.nodes,
        midpoint_n_points=radial_specification.n_points,
        integration_radial_spacing=actual_spacing,
        integration_n_points=int(integration_n_points),
        lebedev_order=int(lebedev_order),
        continuum_n_points=int(continuum_n_points),
        partitioned_quadrature_size=quadrature.point_count,
    )
