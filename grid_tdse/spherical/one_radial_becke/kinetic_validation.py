"""Two-center weak kinetic-energy validation for the Becke prototype."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .basis import BeckeSphericalBasis, fixed_m_channels
from .partition import BeckePartition
from .quadrature import BeckeMulticenterQuadrature
from .specification import FEMDVRQuadratureSpec


@dataclass(frozen=True)
class KineticIntegrationSettings:
    """Independent radial and angular controls for one kinetic integral."""

    radial_spacing: float
    radial_n_points: int
    lebedev_order: int

    def __post_init__(self) -> None:
        if isinstance(self.radial_spacing, (bool, np.bool_)) or not isinstance(
            self.radial_spacing, Real
        ):
            raise TypeError("radial_spacing must be a real number")
        radial_spacing = float(self.radial_spacing)
        if not np.isfinite(radial_spacing) or radial_spacing <= 0.0:
            raise ValueError("radial_spacing must be finite and positive")
        if isinstance(self.radial_n_points, (bool, np.bool_)) or not isinstance(
            self.radial_n_points, Integral
        ):
            raise TypeError("radial_n_points must be an integer")
        radial_n_points = int(self.radial_n_points)
        if radial_n_points < 2:
            raise ValueError("radial_n_points must be at least two")
        if isinstance(self.lebedev_order, (bool, np.bool_)) or not isinstance(
            self.lebedev_order, Integral
        ):
            raise TypeError("lebedev_order must be an integer")

        object.__setattr__(self, "radial_spacing", radial_spacing)
        object.__setattr__(self, "radial_n_points", radial_n_points)
        object.__setattr__(self, "lebedev_order", int(self.lebedev_order))


@dataclass(frozen=True)
class TwoCenterKineticMatrix:
    """One weak-form kinetic matrix and its quadrature diagnostics."""

    kinetic: np.ndarray
    eigenvalues: np.ndarray
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    relative_antihermitian_norm: float
    basis_size: int
    quadrature_size: int
    integration_radius: float
    maximum_radial_spacing: float
    integration_settings: KineticIntegrationSettings


@dataclass(frozen=True)
class KineticElementComparison:
    """One refined kinetic element compared with high-order integration."""

    row: int
    column: int
    refined_value: complex
    reference_value: complex
    absolute_error: float
    relative_error: float


@dataclass(frozen=True)
class TwoCenterKineticValidation:
    """Phase 6.4 convergence and positivity result."""

    coarse: TwoCenterKineticMatrix
    refined: TwoCenterKineticMatrix
    reference: TwoCenterKineticMatrix
    coarse_relative_matrix_error: float
    refined_relative_matrix_error: float
    selected_elements: tuple[KineticElementComparison, ...]
    internuclear_distance: float
    radial_nodes: tuple[float, ...]
    radial_n_points: tuple[int, ...]
    l_max: int
    m: int
    particle_mass: float
    orbital_switch_iterations: int
    integration_switch_iterations: int


def _positive_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _radial_n_points(values, n_elements: int):
    if isinstance(values, (bool, np.bool_)):
        raise TypeError("radial_n_points must contain integers")
    if isinstance(values, Integral):
        return (int(values),) * n_elements
    return values


def _integration_grid(radial_nodes, distance, maximum_spacing):
    nodes = [float(radial_nodes[0])]
    parent_elements = []
    intervals = list(zip(radial_nodes[:-1], radial_nodes[1:]))
    intervals.append((radial_nodes[-1], radial_nodes[-1] + distance))
    for parent, (left, right) in enumerate(intervals):
        n_subintervals = int(np.ceil((right - left) / maximum_spacing))
        subnodes = np.linspace(left, right, n_subintervals + 1)
        nodes.extend(float(value) for value in subnodes[1:])
        basis_parent = parent if parent < len(radial_nodes) - 1 else -1
        parent_elements.extend([basis_parent] * n_subintervals)
    return np.asarray(nodes), np.asarray(parent_elements, dtype=int)


def _kinetic_matrix(
    basis,
    integration_partition,
    distance,
    particle_mass,
    settings,
) -> TwoCenterKineticMatrix:
    integration_nodes, parent_elements = _integration_grid(
        basis.radial_specs[0].nodes,
        distance,
        settings.radial_spacing,
    )
    integration_specification = FEMDVRQuadratureSpec(
        integration_nodes,
        [settings.radial_n_points] * parent_elements.size,
    )
    quadrature = BeckeMulticenterQuadrature(
        integration_partition,
        integration_specification,
        settings.lebedev_order,
    )
    center_indices = np.fromiter(
        (function.center_index for function in basis.functions),
        dtype=int,
        count=basis.size,
    )
    kinetic = np.zeros((basis.size, basis.size), dtype=complex)
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
        gradients = evaluation.gradients

        # The outer integration interval starts at the compact radial support
        # boundary.  Its owning-center derivative is the zero exterior trace,
        # whereas the other center's basis remains active there.
        if parent < 0:
            gradients = np.array(gradients, copy=True)
            gradients[:, center_indices == shell.center_index, :] = 0.0

        for component in range(3):
            component_gradients = gradients[:, :, component]
            kinetic += component_gradients.conj().T @ (
                shell.weights[:, np.newaxis] * component_gradients
            )
    kinetic *= 0.5 / particle_mass

    denominator = float(np.linalg.norm(kinetic))
    antihermitian_norm = float(np.linalg.norm(kinetic - kinetic.conj().T))
    relative_antihermitian_norm = (
        antihermitian_norm / denominator if denominator > 0.0 else 0.0
    )
    kinetic = np.array(0.5 * (kinetic + kinetic.conj().T), copy=True)
    eigenvalues = np.linalg.eigvalsh(kinetic)
    kinetic.setflags(write=False)
    eigenvalues = np.array(eigenvalues, copy=True)
    eigenvalues.setflags(write=False)
    return TwoCenterKineticMatrix(
        kinetic=kinetic,
        eigenvalues=eigenvalues,
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        relative_antihermitian_norm=relative_antihermitian_norm,
        basis_size=basis.size,
        quadrature_size=quadrature.point_count,
        integration_radius=float(integration_nodes[-1]),
        maximum_radial_spacing=float(np.max(np.diff(integration_nodes))),
        integration_settings=settings,
    )


def _relative_matrix_error(candidate, reference) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(candidate - reference))
    return numerator / denominator


def _selected_indices(selected_elements, basis_size, center_size):
    if selected_elements is None:
        selected_elements = (
            (0, 0),
            (0, min(2, center_size - 1)),
            (0, center_size),
            (min(1, center_size - 1), center_size + min(1, center_size - 1)),
            (center_size, center_size),
        )
    try:
        selected_elements = tuple(tuple(pair) for pair in selected_elements)
    except TypeError as error:
        raise TypeError(
            "selected_elements must be a sequence of index pairs"
        ) from error
    for pair in selected_elements:
        if len(pair) != 2 or any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, Integral)
            for index in pair
        ):
            raise TypeError("selected_elements must contain integer index pairs")
        if any(not 0 <= int(index) < basis_size for index in pair):
            raise ValueError("selected element index is out of range")
    return tuple((int(row), int(column)) for row, column in selected_elements)


def validate_two_center_kinetic(
    *,
    internuclear_distance: float = 2.0,
    radial_nodes=(0.0, 2.0, 4.0),
    radial_n_points=4,
    l_max: int = 1,
    m: int = 0,
    particle_mass: float = 1.0,
    orbital_switch_iterations: int = 3,
    integration_switch_iterations=None,
    coarse_settings=KineticIntegrationSettings(1.0, 6, 17),
    refined_settings=KineticIntegrationSettings(0.5, 10, 29),
    reference_settings=KineticIntegrationSettings(0.25, 14, 41),
    selected_elements=None,
) -> TwoCenterKineticValidation:
    r"""Validate the positive weak form ``T = grad(chi)^dagger grad(chi)/2m``.

    The three integrations share the same weighted basis.  Only integration
    controls change, so their differences measure quadrature error rather
    than a change of representation.
    """

    distance = _positive_real(
        internuclear_distance, "internuclear_distance"
    )
    particle_mass = _positive_real(particle_mass, "particle_mass")
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
    for name, settings in (
        ("coarse_settings", coarse_settings),
        ("refined_settings", refined_settings),
        ("reference_settings", reference_settings),
    ):
        if not isinstance(settings, KineticIntegrationSettings):
            raise TypeError(f"{name} must be KineticIntegrationSettings")

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
    basis = BeckeSphericalBasis(
        orbital_partition,
        radial_specification,
        fixed_m_channels(l_max, m),
    )
    if basis.size == 0:
        raise ValueError("radial specification produces an empty basis")

    cache = {}
    matrices = []
    for settings in (
        coarse_settings,
        refined_settings,
        reference_settings,
    ):
        if settings not in cache:
            cache[settings] = _kinetic_matrix(
                basis,
                integration_partition,
                distance,
                particle_mass,
                settings,
            )
        matrices.append(cache[settings])
    coarse, refined, reference = matrices
    center_size = basis.size // 2
    comparisons = []
    for row, column in _selected_indices(
        selected_elements, basis.size, center_size
    ):
        refined_value = complex(refined.kinetic[row, column])
        reference_value = complex(reference.kinetic[row, column])
        absolute_error = abs(refined_value - reference_value)
        scale = abs(reference_value)
        relative_error = absolute_error / scale if scale > 0.0 else np.inf
        comparisons.append(
            KineticElementComparison(
                row=row,
                column=column,
                refined_value=refined_value,
                reference_value=reference_value,
                absolute_error=float(absolute_error),
                relative_error=float(relative_error),
            )
        )

    return TwoCenterKineticValidation(
        coarse=coarse,
        refined=refined,
        reference=reference,
        coarse_relative_matrix_error=_relative_matrix_error(
            coarse.kinetic, reference.kinetic
        ),
        refined_relative_matrix_error=_relative_matrix_error(
            refined.kinetic, reference.kinetic
        ),
        selected_elements=tuple(comparisons),
        internuclear_distance=distance,
        radial_nodes=radial_specification.nodes,
        radial_n_points=radial_specification.n_points,
        l_max=int(l_max),
        m=int(m),
        particle_mass=particle_mass,
        orbital_switch_iterations=orbital_partition.switch_iterations,
        integration_switch_iterations=integration_partition.switch_iterations,
    )
