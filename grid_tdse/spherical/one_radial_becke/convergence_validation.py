"""Independent H2+ convergence studies for the Becke prototype."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from ..one_radial import OneElectronDiatomicSystem
from .basis import BeckeSphericalBasis, fixed_m_channels
from .h2plus_validation import (
    H2PlusMatrices,
    H2PlusMidpointReference,
    _assemble_full_support_matrices,
    _midpoint_reference,
    _solve_cutoffs,
    _validated_cutoffs,
)
from .kinetic_validation import KineticIntegrationSettings
from .partition import BeckePartition
from .specification import FEMDVRQuadratureSpec


@dataclass(frozen=True)
class H2PlusConvergencePoint:
    """One independently varied H2+ discretization."""

    label: str
    electronic_energy: float
    total_energy: float
    midpoint_energy_error: float
    retained_rank: int
    discarded_rank: int
    overlap_condition_number: float
    relative_residual_norm: float
    projected_residual_norm: float
    basis_size: int
    quadrature_size: int
    radial_nodes: tuple[float, ...]
    radial_n_points: tuple[int, ...]
    l_max: int
    lebedev_order: int
    orbital_switch_iterations: int
    integration_switch_iterations: int
    overlap_cutoff: float


@dataclass(frozen=True)
class H2PlusConvergenceStudy:
    """Phase 6.6 one-parameter-at-a-time convergence results."""

    baseline: H2PlusConvergencePoint
    radial_order: tuple[H2PlusConvergencePoint, ...]
    element_layout: tuple[H2PlusConvergencePoint, ...]
    radial_extent: tuple[H2PlusConvergencePoint, ...]
    lebedev_order: tuple[H2PlusConvergencePoint, ...]
    angular_cutoff: tuple[H2PlusConvergencePoint, ...]
    becke_iterations: tuple[H2PlusConvergencePoint, ...]
    overlap_cutoff: tuple[H2PlusConvergencePoint, ...]
    near_dependent_overlap_cutoff: tuple[H2PlusConvergencePoint, ...]
    midpoint: H2PlusMidpointReference
    integration_settings: KineticIntegrationSettings
    internuclear_distance: float
    nuclear_repulsion_energy: float


def _positive_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _sequence(values, name: str):
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _common_radial_orders(value, n_elements):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("radial orders must be integers")
    return (int(value),) * n_elements


def run_h2plus_convergence_study(
    *,
    internuclear_distance: float = 2.0,
    baseline_radial_nodes=(0.0, 2.0, 4.0),
    baseline_radial_order: int = 4,
    baseline_l_max: int = 1,
    baseline_becke_iterations: int = 3,
    integration_switch_iterations: int = 3,
    integration_settings=KineticIntegrationSettings(1.0, 8, 29),
    overlap_cutoffs=(1.0e-8, 1.0e-10, 1.0e-12),
    radial_orders=(3, 4, 5, 6),
    element_layouts=(
        (0.0, 2.0, 4.0),
        (0.0, 1.0, 2.0, 4.0),
        (0.0, 0.5, 1.5, 2.5, 4.0),
    ),
    radial_extents=(2.0, 4.0, 6.0, 8.0),
    lebedev_orders=(9, 17, 29, 41),
    angular_cutoffs=(0, 1, 2, 3, 4),
    becke_iterations=(1, 2, 3, 4),
    include_near_dependent_cutoff_study: bool = True,
    near_dependent_l_max: int = 11,
    midpoint_reference_atomic_extent: float = 12.0,
    midpoint_radial_spacing: float = 1.0,
    midpoint_radial_n_points: int = 15,
    midpoint_l_max: int = 12,
) -> H2PlusConvergenceStudy:
    """Vary each representation or quadrature parameter independently."""

    distance = _positive_real(
        internuclear_distance, "internuclear_distance"
    )
    midpoint_reference_atomic_extent = _positive_real(
        midpoint_reference_atomic_extent,
        "midpoint_reference_atomic_extent",
    )
    midpoint_radial_spacing = _positive_real(
        midpoint_radial_spacing, "midpoint_radial_spacing"
    )
    if not isinstance(integration_settings, KineticIntegrationSettings):
        raise TypeError(
            "integration_settings must be KineticIntegrationSettings"
        )
    if not isinstance(
        include_near_dependent_cutoff_study, (bool, np.bool_)
    ):
        raise TypeError(
            "include_near_dependent_cutoff_study must be boolean"
        )
    cutoffs = _validated_cutoffs(overlap_cutoffs)
    radial_orders = _sequence(radial_orders, "radial_orders")
    element_layouts = _sequence(element_layouts, "element_layouts")
    radial_extents = _sequence(radial_extents, "radial_extents")
    lebedev_orders = _sequence(lebedev_orders, "lebedev_orders")
    angular_cutoffs = _sequence(angular_cutoffs, "angular_cutoffs")
    becke_iterations = _sequence(becke_iterations, "becke_iterations")

    baseline_specification = FEMDVRQuadratureSpec(
        baseline_radial_nodes,
        _common_radial_orders(
            baseline_radial_order, len(baseline_radial_nodes) - 1
        ),
    )
    system = OneElectronDiatomicSystem(
        1.0,
        1.0,
        0.5 * distance,
        name="H2+",
    )
    midpoint = _midpoint_reference(
        system,
        midpoint_reference_atomic_extent,
        midpoint_radial_spacing,
        midpoint_radial_n_points,
        midpoint_l_max,
    )
    matrix_cache: dict[tuple, H2PlusMatrices] = {}
    point_cache = {}

    def point(
        label,
        *,
        radial_nodes=baseline_specification.nodes,
        radial_order=baseline_radial_order,
        l_max=baseline_l_max,
        orbital_iterations=baseline_becke_iterations,
        lebedev_order=integration_settings.lebedev_order,
        cutoff=1.0e-10,
    ):
        specification = FEMDVRQuadratureSpec(
            radial_nodes,
            _common_radial_orders(radial_order, len(radial_nodes) - 1),
        )
        settings = KineticIntegrationSettings(
            integration_settings.radial_spacing,
            integration_settings.radial_n_points,
            lebedev_order,
        )
        matrix_key = (
            specification.nodes,
            specification.n_points,
            int(l_max),
            int(orbital_iterations),
            int(integration_switch_iterations),
            settings,
        )
        if matrix_key not in matrix_cache:
            orbital_partition = BeckePartition(
                system.nuclear_positions,
                switch_iterations=orbital_iterations,
            )
            integration_partition = BeckePartition(
                system.nuclear_positions,
                switch_iterations=integration_switch_iterations,
            )
            basis = BeckeSphericalBasis(
                orbital_partition,
                specification,
                fixed_m_channels(l_max, 0),
            )
            matrix_cache[matrix_key] = _assemble_full_support_matrices(
                basis,
                integration_partition,
                system,
                settings,
            )
        point_key = (matrix_key, float(cutoff), str(label))
        if point_key not in point_cache:
            matrices = matrix_cache[matrix_key]
            result = _solve_cutoffs(
                matrices,
                (cutoff,),
                system.nuclear_repulsion_energy,
            )
            solution = result.solutions[0]
            electronic_energy = result.electronic_energies[0]
            point_cache[point_key] = H2PlusConvergencePoint(
                label=str(label),
                electronic_energy=electronic_energy,
                total_energy=result.total_energies[0],
                midpoint_energy_error=abs(
                    electronic_energy - midpoint.electronic_energy
                ),
                retained_rank=result.retained_ranks[0],
                discarded_rank=result.discarded_ranks[0],
                overlap_condition_number=(
                    matrices.overlap_condition_number
                ),
                relative_residual_norm=float(
                    solution.diagnostics.relative_residual_norms[0]
                ),
                projected_residual_norm=float(
                    solution.diagnostics.projected_residual_norms[0]
                ),
                basis_size=matrices.basis_size,
                quadrature_size=matrices.quadrature_size,
                radial_nodes=specification.nodes,
                radial_n_points=specification.n_points,
                l_max=int(l_max),
                lebedev_order=settings.lebedev_order,
                orbital_switch_iterations=int(orbital_iterations),
                integration_switch_iterations=int(
                    integration_switch_iterations
                ),
                overlap_cutoff=float(cutoff),
            )
        return point_cache[point_key]

    baseline = point("baseline")
    radial_order_sweep = tuple(
        point(str(value), radial_order=value) for value in radial_orders
    )
    element_layout_sweep = tuple(
        point(str(tuple(nodes)), radial_nodes=nodes)
        for nodes in element_layouts
    )
    baseline_element_width = max(
        np.diff(baseline_specification.nodes)
    )

    def extent_nodes(value):
        value = _positive_real(value, "radial extent")
        n_elements = int(np.ceil(value / baseline_element_width))
        return tuple(np.linspace(0.0, value, n_elements + 1))

    radial_extent_sweep = tuple(
        point(
            f"{float(value):g}",
            radial_nodes=extent_nodes(value),
        )
        for value in radial_extents
    )
    lebedev_sweep = tuple(
        point(str(value), lebedev_order=value) for value in lebedev_orders
    )
    angular_sweep = tuple(
        point(str(value), l_max=value) for value in angular_cutoffs
    )
    becke_sweep = tuple(
        point(str(value), orbital_iterations=value)
        for value in becke_iterations
    )
    cutoff_sweep = tuple(
        point(f"{value:.0e}", cutoff=value) for value in cutoffs
    )
    if include_near_dependent_cutoff_study:
        near_dependent_sweep = tuple(
            point(
                f"{value:.0e}",
                l_max=near_dependent_l_max,
                cutoff=value,
            )
            for value in cutoffs
        )
    else:
        near_dependent_sweep = ()

    return H2PlusConvergenceStudy(
        baseline=baseline,
        radial_order=radial_order_sweep,
        element_layout=element_layout_sweep,
        radial_extent=radial_extent_sweep,
        lebedev_order=lebedev_sweep,
        angular_cutoff=angular_sweep,
        becke_iterations=becke_sweep,
        overlap_cutoff=cutoff_sweep,
        near_dependent_overlap_cutoff=near_dependent_sweep,
        midpoint=midpoint,
        integration_settings=integration_settings,
        internuclear_distance=distance,
        nuclear_repulsion_energy=system.nuclear_repulsion_energy,
    )
