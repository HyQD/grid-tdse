"""Cross-representation validation helpers for the Becke prototype."""

from dataclasses import dataclass
from numbers import Integral, Real

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


@dataclass(frozen=True)
class OverlapRankDecision:
    """Retained overlap rank at one relative eigenvalue cutoff."""

    relative_cutoff: float
    absolute_cutoff: float
    retained_rank: int
    discarded_rank: int


@dataclass(frozen=True)
class TwoCenterOverlapValidation:
    """Overlap spectrum and complete numerical settings for two centers."""

    overlap: np.ndarray
    eigenvalues: np.ndarray
    rank_decisions: tuple[OverlapRankDecision, ...]
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    relative_antihermitian_norm: float
    basis_size: int
    quadrature_size: int
    internuclear_distance: float
    radial_nodes: tuple[float, ...]
    radial_n_points: tuple[int, ...]
    l_max: int
    m: int
    orbital_switch_iterations: int
    integration_switch_iterations: int
    lebedev_order: int
    integration_radius: float
    integration_radial_spacing: float
    integration_n_points: int


@dataclass(frozen=True)
class TwoCenterOverlapStudy:
    """Independent parameter sweeps used by the Phase 6.3 decision gate."""

    baseline: TwoCenterOverlapValidation
    internuclear_distance: tuple[TwoCenterOverlapValidation, ...]
    radial_extent: tuple[TwoCenterOverlapValidation, ...]
    radial_order: tuple[TwoCenterOverlapValidation, ...]
    element_layout: tuple[TwoCenterOverlapValidation, ...]
    angular_cutoff: tuple[TwoCenterOverlapValidation, ...]
    lebedev_order: tuple[TwoCenterOverlapValidation, ...]
    high_l_lebedev_order: tuple[TwoCenterOverlapValidation, ...]
    becke_iterations: tuple[TwoCenterOverlapValidation, ...]


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


def _positive_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _overlap_radial_n_points(radial_n_points, n_elements: int):
    if isinstance(radial_n_points, (bool, np.bool_)):
        raise TypeError("radial_n_points must contain integers")
    if isinstance(radial_n_points, Integral):
        return (int(radial_n_points),) * n_elements
    return radial_n_points


def _rank_cutoffs(values) -> tuple[float, ...]:
    try:
        supplied = tuple(values)
    except TypeError as error:
        raise TypeError("rank_cutoffs must be a sequence") from error
    if not supplied:
        raise ValueError("rank_cutoffs must not be empty")
    cutoffs = []
    for value in supplied:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("rank_cutoffs must contain real numbers")
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(
                "rank_cutoffs must be finite and satisfy 0 <= tau < 1"
            )
        cutoffs.append(value)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("rank_cutoffs must not contain duplicates")
    return tuple(cutoffs)


def validate_two_center_overlap(
    *,
    internuclear_distance: float = 2.0,
    radial_nodes=(0.0, 2.0, 4.0),
    radial_n_points=4,
    l_max: int = 1,
    m: int = 0,
    orbital_switch_iterations: int = 3,
    integration_switch_iterations=None,
    lebedev_order: int = 17,
    integration_radial_spacing: float = 1.0,
    integration_n_points: int = 8,
    rank_cutoffs=(1.0e-8, 1.0e-10, 1.0e-12),
) -> TwoCenterOverlapValidation:
    r"""Assemble and diagnose a two-center Becke overlap matrix.

    Each integration grid extends to ``r_max + R`` from its own center.  Every
    point in the union of the two finite atom-centered basis supports is
    therefore covered by *both* integration cells.  The integration weights
    sum to unity over that complete domain, avoiding an artificial overlap
    defect caused by truncated atomic integration cells.
    """

    distance = _positive_real(
        internuclear_distance, "internuclear_distance"
    )
    radial_spacing = _positive_real(
        integration_radial_spacing, "integration_radial_spacing"
    )
    raw_nodes = np.asarray(radial_nodes)
    if raw_nodes.ndim != 1 or raw_nodes.size < 2:
        raise ValueError(
            "radial_nodes must be one-dimensional with at least two entries"
        )
    radial_n_points = _overlap_radial_n_points(
        radial_n_points, raw_nodes.size - 1
    )
    radial_specification = FEMDVRQuadratureSpec(
        radial_nodes, radial_n_points
    )
    cutoffs = _rank_cutoffs(rank_cutoffs)

    centers = np.array(
        [[0.0, 0.0, -0.5 * distance], [0.0, 0.0, 0.5 * distance]]
    )
    orbital_partition = BeckePartition(
        centers, switch_iterations=orbital_switch_iterations
    )
    if integration_switch_iterations is None:
        integration_switch_iterations = orbital_partition.switch_iterations
    integration_partition = BeckePartition(
        centers, switch_iterations=integration_switch_iterations
    )
    channels = fixed_m_channels(l_max, m)
    basis = BeckeSphericalBasis(
        orbital_partition, radial_specification, channels
    )

    integration_radius = radial_specification.r_max + distance
    n_integration_elements = int(
        np.ceil(integration_radius / radial_spacing)
    )
    integration_nodes = np.linspace(
        0.0, integration_radius, n_integration_elements + 1
    )
    actual_spacing = integration_radius / n_integration_elements
    quadrature = BeckeMulticenterQuadrature(
        integration_partition,
        FEMDVRQuadratureSpec(
            integration_nodes,
            [integration_n_points] * n_integration_elements,
        ),
        lebedev_order,
    )

    overlap = np.zeros((basis.size, basis.size), dtype=complex)
    for shell in quadrature.iter_shells():
        values = basis.evaluate(shell.points).values
        overlap += values.conj().T @ (
            shell.weights[:, np.newaxis] * values
        )

    denominator = float(np.linalg.norm(overlap))
    antihermitian_norm = float(np.linalg.norm(overlap - overlap.conj().T))
    relative_antihermitian_norm = (
        antihermitian_norm / denominator if denominator > 0.0 else 0.0
    )
    overlap = np.array(0.5 * (overlap + overlap.conj().T), copy=True)
    eigenvalues = np.linalg.eigvalsh(overlap)
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    condition_number = (
        maximum_eigenvalue / minimum_eigenvalue
        if minimum_eigenvalue > 0.0
        else np.inf
    )
    decisions = tuple(
        OverlapRankDecision(
            relative_cutoff=cutoff,
            absolute_cutoff=cutoff * maximum_eigenvalue,
            retained_rank=int(
                np.count_nonzero(eigenvalues > cutoff * maximum_eigenvalue)
            ),
            discarded_rank=int(
                np.count_nonzero(eigenvalues <= cutoff * maximum_eigenvalue)
            ),
        )
        for cutoff in cutoffs
    )

    overlap.setflags(write=False)
    eigenvalues = np.array(eigenvalues, copy=True)
    eigenvalues.setflags(write=False)
    return TwoCenterOverlapValidation(
        overlap=overlap,
        eigenvalues=eigenvalues,
        rank_decisions=decisions,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_eigenvalue=maximum_eigenvalue,
        condition_number=float(condition_number),
        relative_antihermitian_norm=relative_antihermitian_norm,
        basis_size=basis.size,
        quadrature_size=quadrature.point_count,
        internuclear_distance=distance,
        radial_nodes=radial_specification.nodes,
        radial_n_points=radial_specification.n_points,
        l_max=int(l_max),
        m=int(m),
        orbital_switch_iterations=orbital_partition.switch_iterations,
        integration_switch_iterations=integration_partition.switch_iterations,
        lebedev_order=quadrature.lebedev_order,
        integration_radius=integration_radius,
        integration_radial_spacing=actual_spacing,
        integration_n_points=int(integration_n_points),
    )


def run_two_center_overlap_study() -> TwoCenterOverlapStudy:
    """Run the reproducible Phase 6.3 one-parameter-at-a-time sweeps."""

    baseline_parameters = {
        "internuclear_distance": 2.0,
        "radial_nodes": (0.0, 2.0, 4.0),
        "radial_n_points": 4,
        "l_max": 1,
        "m": 0,
        "orbital_switch_iterations": 3,
        "integration_switch_iterations": 3,
        "lebedev_order": 29,
        "integration_radial_spacing": 1.0,
        "integration_n_points": 8,
        "rank_cutoffs": (1.0e-8, 1.0e-10, 1.0e-12),
    }
    cache = {}

    def evaluate(**updates):
        parameters = {**baseline_parameters, **updates}
        key = tuple(
            (name, repr(parameters[name])) for name in sorted(parameters)
        )
        if key not in cache:
            cache[key] = validate_two_center_overlap(**parameters)
        return cache[key]

    baseline = evaluate()
    distance_sweep = tuple(
        evaluate(internuclear_distance=value)
        for value in (1.0, 2.0, 4.0, 6.0)
    )
    extent_sweep = tuple(
        evaluate(radial_nodes=(0.0, 0.5 * value, value))
        for value in (3.0, 4.0, 6.0, 8.0)
    )
    order_sweep = tuple(
        evaluate(radial_n_points=value) for value in (3, 4, 5, 6)
    )
    layout_sweep = tuple(
        evaluate(radial_nodes=nodes)
        for nodes in (
            (0.0, 2.0, 4.0),
            (0.0, 1.0, 2.0, 4.0),
            (0.0, 0.5, 1.5, 2.5, 4.0),
        )
    )
    angular_sweep = tuple(
        evaluate(l_max=value) for value in (0, 1, 2, 3, 5, 8, 11)
    )
    lebedev_sweep = tuple(
        evaluate(lebedev_order=value) for value in (9, 17, 29, 41)
    )
    high_l_lebedev_sweep = tuple(
        evaluate(l_max=11, lebedev_order=value) for value in (29, 41)
    )
    becke_sweep = tuple(
        evaluate(
            orbital_switch_iterations=value,
            integration_switch_iterations=value,
        )
        for value in (1, 2, 3, 4)
    )
    return TwoCenterOverlapStudy(
        baseline=baseline,
        internuclear_distance=distance_sweep,
        radial_extent=extent_sweep,
        radial_order=order_sweep,
        element_layout=layout_sweep,
        angular_cutoff=angular_sweep,
        lebedev_order=lebedev_sweep,
        high_l_lebedev_order=high_l_lebedev_sweep,
        becke_iterations=becke_sweep,
    )
