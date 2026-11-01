"""H2+ ground-state validation for the two-center Becke prototype."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    diagonalize_fixed_m,
)
from .basis import BeckeSphericalBasis, fixed_m_channels
from .ground_state import GeneralizedEigenResult, solve_generalized_hermitian
from .kinetic_validation import (
    KineticIntegrationSettings,
    _integration_grid,
)
from .partition import BeckePartition
from .quadrature import BeckeMulticenterQuadrature
from .specification import FEMDVRQuadratureSpec


@dataclass(frozen=True)
class H2PlusMatrices:
    """Full-support weak matrices at one integration resolution."""

    overlap: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    hamiltonian: np.ndarray
    overlap_minimum_eigenvalue: float
    overlap_maximum_eigenvalue: float
    overlap_condition_number: float
    kinetic_minimum_eigenvalue: float
    relative_antihermitian_norms: tuple[float, float, float, float]
    basis_size: int
    quadrature_size: int
    integration_radius: float
    integration_settings: KineticIntegrationSettings


@dataclass(frozen=True)
class H2PlusBeckeResult:
    """Matrices and cutoff solutions at one quadrature resolution."""

    matrices: H2PlusMatrices
    solutions: tuple[GeneralizedEigenResult, ...]
    electronic_energies: tuple[float, ...]
    total_energies: tuple[float, ...]
    retained_ranks: tuple[int, ...]
    discarded_ranks: tuple[int, ...]


@dataclass(frozen=True)
class H2PlusMidpointReference:
    """Converged midpoint-centered comparison in a circumscribing sphere."""

    electronic_energy: float
    total_energy: float
    radial_nodes: tuple[float, ...]
    radial_n_points: int
    l_max: int
    basis_size: int
    radial_extent: float


@dataclass(frozen=True)
class H2PlusGroundStateValidation:
    """Phase 6.5 Becke and midpoint-centered ground-state comparison."""

    coarse: H2PlusBeckeResult
    refined: H2PlusBeckeResult
    reference: H2PlusBeckeResult
    midpoint: H2PlusMidpointReference
    overlap_cutoffs: tuple[float, ...]
    coarse_to_reference_energy_error: float
    refined_to_reference_energy_error: float
    becke_to_midpoint_energy_error: float
    cutoff_energy_spread: float
    internuclear_distance: float
    nuclear_repulsion_energy: float
    radial_nodes: tuple[float, ...]
    radial_n_points: tuple[int, ...]
    l_max: int
    m: int
    orbital_switch_iterations: int
    integration_switch_iterations: int


def _positive_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_integer(value, name: str, *, minimum=1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _radial_n_points(values, n_elements: int):
    if isinstance(values, (bool, np.bool_)):
        raise TypeError("radial_n_points must contain integers")
    if isinstance(values, Integral):
        return (int(values),) * n_elements
    return values


def _relative_antihermitian_norm(matrix) -> float:
    denominator = float(np.linalg.norm(matrix))
    numerator = float(np.linalg.norm(matrix - matrix.conj().T))
    return numerator / denominator if denominator > 0.0 else 0.0


def _read_only_hermitian(matrix):
    result = np.array(0.5 * (matrix + matrix.conj().T), copy=True)
    result.setflags(write=False)
    return result


def _assemble_full_support_matrices(
    basis,
    integration_partition,
    system,
    settings,
) -> H2PlusMatrices:
    integration_nodes, parent_elements = _integration_grid(
        basis.radial_specs[0].nodes,
        system.internuclear_distance,
        settings.radial_spacing,
    )
    quadrature = BeckeMulticenterQuadrature(
        integration_partition,
        FEMDVRQuadratureSpec(
            integration_nodes,
            [settings.radial_n_points] * parent_elements.size,
        ),
        settings.lebedev_order,
    )
    center_indices = np.fromiter(
        (function.center_index for function in basis.functions),
        dtype=int,
        count=basis.size,
    )
    shape = (basis.size, basis.size)
    overlap = np.zeros(shape, dtype=complex)
    kinetic = np.zeros(shape, dtype=complex)
    potential = np.zeros(shape, dtype=complex)

    for shell in quadrature.iter_shells():
        hints = np.full(
            (shell.points.shape[0], basis.partition.n_centers),
            -1,
            dtype=int,
        )
        parent = int(parent_elements[shell.element_index])
        if parent >= 0:
            hints[:, shell.center_index] = parent
        evaluation = basis.evaluate(shell.points, element_hints=hints)
        values = evaluation.values
        gradients = evaluation.gradients
        if parent < 0:
            gradients = np.array(gradients, copy=True)
            gradients[:, center_indices == shell.center_index, :] = 0.0

        overlap += values.conj().T @ (
            shell.weights[:, np.newaxis] * values
        )
        for component in range(3):
            component_gradients = gradients[:, :, component]
            kinetic += component_gradients.conj().T @ (
                shell.weights[:, np.newaxis] * component_gradients
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
            (shell.weights * nuclear_potential)[:, np.newaxis] * values
        )

    kinetic *= 0.5 / system.particle_mass
    hamiltonian = kinetic + potential
    relative_norms = tuple(
        _relative_antihermitian_norm(matrix)
        for matrix in (overlap, kinetic, potential, hamiltonian)
    )
    overlap = _read_only_hermitian(overlap)
    kinetic = _read_only_hermitian(kinetic)
    potential = _read_only_hermitian(potential)
    hamiltonian = _read_only_hermitian(hamiltonian)
    overlap_eigenvalues = np.linalg.eigvalsh(overlap)
    kinetic_eigenvalues = np.linalg.eigvalsh(kinetic)
    overlap_minimum = float(overlap_eigenvalues[0])
    overlap_maximum = float(overlap_eigenvalues[-1])
    overlap_condition = (
        overlap_maximum / overlap_minimum
        if overlap_minimum > 0.0
        else np.inf
    )
    return H2PlusMatrices(
        overlap=overlap,
        kinetic=kinetic,
        potential=potential,
        hamiltonian=hamiltonian,
        overlap_minimum_eigenvalue=overlap_minimum,
        overlap_maximum_eigenvalue=overlap_maximum,
        overlap_condition_number=float(overlap_condition),
        kinetic_minimum_eigenvalue=float(kinetic_eigenvalues[0]),
        relative_antihermitian_norms=relative_norms,
        basis_size=basis.size,
        quadrature_size=quadrature.point_count,
        integration_radius=float(integration_nodes[-1]),
        integration_settings=settings,
    )


def _solve_cutoffs(matrices, cutoffs, nuclear_repulsion):
    solutions = tuple(
        solve_generalized_hermitian(
            matrices.hamiltonian,
            matrices.overlap,
            relative_cutoff=cutoff,
            number_of_states=1,
        )
        for cutoff in cutoffs
    )
    electronic_energies = tuple(
        float(solution.energies[0]) for solution in solutions
    )
    return H2PlusBeckeResult(
        matrices=matrices,
        solutions=solutions,
        electronic_energies=electronic_energies,
        total_energies=tuple(
            energy + nuclear_repulsion for energy in electronic_energies
        ),
        retained_ranks=tuple(
            solution.metric.retained_rank for solution in solutions
        ),
        discarded_ranks=tuple(
            solution.metric.discarded_rank for solution in solutions
        ),
    )


def _validated_cutoffs(values):
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError("overlap_cutoffs must be a sequence") from error
    if not values:
        raise ValueError("overlap_cutoffs must not be empty")
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("overlap_cutoffs must contain real numbers")
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(
                "overlap_cutoffs must be finite and satisfy 0 <= tau < 1"
            )
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError("overlap_cutoffs must not contain duplicates")
    return tuple(result)


def _midpoint_nodes(half_bond_length, radial_extent, maximum_spacing):
    nodes = [0.0]
    for left, right in (
        (0.0, half_bond_length),
        (half_bond_length, radial_extent),
    ):
        n_intervals = int(np.ceil((right - left) / maximum_spacing))
        segment = np.linspace(left, right, n_intervals + 1)
        nodes.extend(float(value) for value in segment[1:])
    return tuple(nodes)


def validate_h2plus_ground_state(
    *,
    internuclear_distance: float = 2.0,
    radial_nodes=(0.0, 2.0, 4.0),
    radial_n_points=4,
    l_max: int = 1,
    m: int = 0,
    particle_mass: float = 1.0,
    orbital_switch_iterations: int = 3,
    integration_switch_iterations=None,
    overlap_cutoffs=(1.0e-8, 1.0e-10, 1.0e-12),
    coarse_settings=KineticIntegrationSettings(1.0, 4, 9),
    refined_settings=KineticIntegrationSettings(0.5, 10, 29),
    reference_settings=KineticIntegrationSettings(0.25, 14, 41),
    midpoint_radial_spacing: float = 1.0,
    midpoint_radial_n_points: int = 12,
    midpoint_l_max: int = 10,
) -> H2PlusGroundStateValidation:
    """Assemble and solve the two-center H2+ Hamiltonian."""

    distance = _positive_real(
        internuclear_distance, "internuclear_distance"
    )
    particle_mass = _positive_real(particle_mass, "particle_mass")
    midpoint_radial_spacing = _positive_real(
        midpoint_radial_spacing, "midpoint_radial_spacing"
    )
    midpoint_radial_n_points = _positive_integer(
        midpoint_radial_n_points,
        "midpoint_radial_n_points",
        minimum=2,
    )
    midpoint_l_max = _positive_integer(
        midpoint_l_max, "midpoint_l_max", minimum=0
    )
    cutoffs = _validated_cutoffs(overlap_cutoffs)
    raw_nodes = np.asarray(radial_nodes)
    if raw_nodes.ndim != 1 or raw_nodes.size < 2:
        raise ValueError(
            "radial_nodes must be one-dimensional with at least two entries"
        )
    radial_n_points = _radial_n_points(
        radial_n_points, raw_nodes.size - 1
    )
    radial_specification = FEMDVRQuadratureSpec(
        radial_nodes, radial_n_points
    )
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise TypeError("m must be an integer")
    m = int(m)
    if m != 0:
        raise ValueError("the H2+ ground-state validation requires m = 0")
    for name, settings in (
        ("coarse_settings", coarse_settings),
        ("refined_settings", refined_settings),
        ("reference_settings", reference_settings),
    ):
        if not isinstance(settings, KineticIntegrationSettings):
            raise TypeError(f"{name} must be KineticIntegrationSettings")

    system = OneElectronDiatomicSystem(
        1.0,
        1.0,
        0.5 * distance,
        particle_mass=particle_mass,
        name="H2+",
    )
    orbital_partition = BeckePartition(
        system.nuclear_positions,
        switch_iterations=orbital_switch_iterations,
    )
    if integration_switch_iterations is None:
        integration_switch_iterations = orbital_partition.switch_iterations
    integration_partition = BeckePartition(
        system.nuclear_positions,
        switch_iterations=integration_switch_iterations,
    )
    basis = BeckeSphericalBasis(
        orbital_partition,
        radial_specification,
        fixed_m_channels(l_max, m),
    )
    if basis.size == 0:
        raise ValueError("radial specification produces an empty basis")

    matrix_cache = {}
    becke_results = []
    for settings in (
        coarse_settings,
        refined_settings,
        reference_settings,
    ):
        if settings not in matrix_cache:
            matrix_cache[settings] = _assemble_full_support_matrices(
                basis,
                integration_partition,
                system,
                settings,
            )
        becke_results.append(
            _solve_cutoffs(
                matrix_cache[settings],
                cutoffs,
                system.nuclear_repulsion_energy,
            )
        )
    coarse, refined, reference = becke_results

    midpoint_extent = (
        radial_specification.r_max + system.half_bond_length
    )
    midpoint_nodes = _midpoint_nodes(
        system.half_bond_length,
        midpoint_extent,
        midpoint_radial_spacing,
    )
    midpoint_hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=midpoint_nodes,
        n_points=[midpoint_radial_n_points] * (len(midpoint_nodes) - 1),
        l_max=midpoint_l_max,
        m_max=0,
        nuclear_L_max=2 * midpoint_l_max,
        nuclear_method="analytic",
    )
    midpoint_solution = diagonalize_fixed_m(
        midpoint_hamiltonian, m=0, n_states=1
    )
    midpoint = H2PlusMidpointReference(
        electronic_energy=midpoint_solution.electronic_energy,
        total_energy=midpoint_solution.total_energy,
        radial_nodes=midpoint_nodes,
        radial_n_points=midpoint_radial_n_points,
        l_max=midpoint_l_max,
        basis_size=int(np.prod(midpoint_hamiltonian.fixed_m_shape(0))),
        radial_extent=midpoint_extent,
    )

    reference_energies = np.asarray(reference.electronic_energies)
    return H2PlusGroundStateValidation(
        coarse=coarse,
        refined=refined,
        reference=reference,
        midpoint=midpoint,
        overlap_cutoffs=cutoffs,
        coarse_to_reference_energy_error=abs(
            coarse.electronic_energies[0] - reference_energies[0]
        ),
        refined_to_reference_energy_error=abs(
            refined.electronic_energies[0] - reference_energies[0]
        ),
        becke_to_midpoint_energy_error=abs(
            reference_energies[0] - midpoint.electronic_energy
        ),
        cutoff_energy_spread=float(np.ptp(reference_energies)),
        internuclear_distance=distance,
        nuclear_repulsion_energy=system.nuclear_repulsion_energy,
        radial_nodes=radial_specification.nodes,
        radial_n_points=radial_specification.n_points,
        l_max=int(l_max),
        m=int(m),
        orbital_switch_iterations=orbital_partition.switch_iterations,
        integration_switch_iterations=integration_partition.switch_iterations,
    )
