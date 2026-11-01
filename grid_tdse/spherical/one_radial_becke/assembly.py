"""Dense weak-form matrix assembly for the Becke spherical basis."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .basis import BeckeSphericalBasis, fixed_m_channels, full_lm_channels
from .geometry import CenterGeometry, classify_centers
from .quadrature import BeckeMulticenterQuadrature


@dataclass(frozen=True)
class MatrixAssemblyDiagnostics:
    """Hermiticity and center-block norms for one assembled matrix."""

    relative_antihermitian_norm: float
    same_center_norm: float
    cross_center_norm: float


@dataclass(frozen=True)
class BeckeAssemblyParameters:
    """Complete numerical settings recorded with an assembly result."""

    nuclear_positions: tuple[tuple[float, float, float], ...]
    nuclear_charges: tuple[float, ...]
    particle_mass: float
    basis_radial_nodes: tuple[tuple[float, ...], ...]
    basis_radial_n_points: tuple[tuple[int, ...], ...]
    quadrature_radial_nodes: tuple[tuple[float, ...], ...]
    quadrature_radial_n_points: tuple[tuple[int, ...], ...]
    angular_channels: tuple[tuple[tuple[int, int], ...], ...]
    orbital_switch_iterations: int
    integration_switch_iterations: int
    lebedev_order: int
    lebedev_rotation: tuple[tuple[float, float, float], ...]
    basis_frame: tuple[tuple[float, float, float], ...]
    hermiticity_tolerance: float
    overlap_relative_cutoff: float | None = None


@dataclass(frozen=True)
class BeckeAssemblyDiagnostics:
    """Diagnostics attached to a dense multicenter assembly."""

    overlap: MatrixAssemblyDiagnostics
    kinetic: MatrixAssemblyDiagnostics
    potential: MatrixAssemblyDiagnostics
    hamiltonian: MatrixAssemblyDiagnostics
    overlap_min_eigenvalue: float
    overlap_max_eigenvalue: float
    overlap_condition_number: float
    geometry: CenterGeometry
    angular_mode: str
    fixed_m: int | None
    basis_size: int
    quadrature_size: int
    parameters: BeckeAssemblyParameters


@dataclass(frozen=True)
class BeckeDenseMatrices:
    """Hermitian overlap and one-electron Hamiltonian matrices."""

    overlap: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    hamiltonian: np.ndarray
    diagnostics: BeckeAssemblyDiagnostics


def _relative_antihermitian_norm(matrix: np.ndarray) -> float:
    denominator = float(np.linalg.norm(matrix))
    numerator = float(np.linalg.norm(matrix - matrix.conj().T))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else np.inf
    return numerator / denominator


def _center_block_norms(matrix, center_indices):
    same_center = center_indices[:, np.newaxis] == center_indices[np.newaxis, :]
    return (
        float(np.linalg.norm(matrix[same_center])),
        float(np.linalg.norm(matrix[~same_center])),
    )


def _is_full_lm_table(table) -> bool:
    l_max = max(ell for ell, _ in table)
    return set(table) == set(full_lm_channels(l_max))


def _fixed_m_for_table(table) -> int | None:
    magnetic_numbers = {m for _, m in table}
    if len(magnetic_numbers) != 1:
        return None
    magnetic_number = magnetic_numbers.pop()
    l_max = max(ell for ell, _ in table)
    if set(table) != set(fixed_m_channels(l_max, magnetic_number)):
        return None
    return magnetic_number


def _angular_mode(basis, geometry):
    tables = basis.angular_channels
    full_lm = all(_is_full_lm_table(table) for table in tables)
    fixed_numbers = tuple(_fixed_m_for_table(table) for table in tables)
    common_fixed_m = (
        fixed_numbers[0]
        if all(
            value is not None and value == fixed_numbers[0]
            for value in fixed_numbers
        )
        else None
    )

    if geometry.classification == "noncollinear":
        if not full_lm:
            raise ValueError(
                "noncollinear assembly requires a complete full_lm table "
                "at every center"
            )
        return "full_lm", None

    if common_fixed_m is not None:
        if geometry.axis is not None:
            molecular_axis = np.asarray(geometry.axis)
            local_z_axis = basis.frame[:, 2]
            if not np.isclose(
                abs(np.dot(molecular_axis, local_z_axis)),
                1.0,
                atol=1.0e-12,
                rtol=0.0,
            ):
                if full_lm:
                    return "full_lm", None
                raise ValueError(
                    "a fixed_m basis frame must have its z axis along the "
                    "collinear molecular axis"
                )
        return "fixed_m", common_fixed_m

    if full_lm:
        return "full_lm", None
    raise ValueError(
        "angular channels must define either one complete fixed_m block or "
        "a complete full_lm table"
    )


def _validate_assembly_inputs(
    basis,
    quadrature,
    nuclear_charges,
    particle_mass,
    hermiticity_tolerance,
):
    if not isinstance(basis, BeckeSphericalBasis):
        raise TypeError("basis must be a BeckeSphericalBasis")
    if not isinstance(quadrature, BeckeMulticenterQuadrature):
        raise TypeError("quadrature must be a BeckeMulticenterQuadrature")

    basis_centers = basis.partition.centers
    quadrature_centers = quadrature.partition.centers
    scale = max(1.0, float(np.max(np.abs(basis_centers))))
    if basis_centers.shape != quadrature_centers.shape or not np.allclose(
        basis_centers,
        quadrature_centers,
        atol=128.0 * np.finfo(float).eps * scale,
        rtol=0.0,
    ):
        raise ValueError(
            "basis and quadrature partitions must use the same ordered centers"
        )

    for basis_specification, quadrature_specification in zip(
        basis.radial_specs, quadrature.radial_specs
    ):
        if basis_specification.nodes != quadrature_specification.nodes:
            raise ValueError(
                "basis and quadrature must use matching finite-element "
                "boundaries; radial orders may differ"
            )

    charges = np.asarray(nuclear_charges, dtype=float)
    if charges.shape != (basis.partition.n_centers,):
        raise ValueError("nuclear_charges must contain one value per center")
    if not np.all(np.isfinite(charges)) or np.any(charges <= 0.0):
        raise ValueError("nuclear_charges must be finite and positive")

    if isinstance(particle_mass, (bool, np.bool_)) or not isinstance(
        particle_mass, Real
    ):
        raise TypeError("particle_mass must be a real number")
    particle_mass = float(particle_mass)
    if not np.isfinite(particle_mass) or particle_mass <= 0.0:
        raise ValueError("particle_mass must be finite and positive")

    if isinstance(hermiticity_tolerance, (bool, np.bool_)) or not isinstance(
        hermiticity_tolerance, Real
    ):
        raise TypeError("hermiticity_tolerance must be a real number")
    hermiticity_tolerance = float(hermiticity_tolerance)
    if not np.isfinite(hermiticity_tolerance) or hermiticity_tolerance <= 0.0:
        raise ValueError("hermiticity_tolerance must be finite and positive")
    return charges, particle_mass, hermiticity_tolerance


def _as_tuple_matrix(matrix):
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _parameters(
    basis,
    quadrature,
    charges,
    particle_mass,
    hermiticity_tolerance,
):
    return BeckeAssemblyParameters(
        nuclear_positions=_as_tuple_matrix(basis.partition.centers),
        nuclear_charges=tuple(float(value) for value in charges),
        particle_mass=particle_mass,
        basis_radial_nodes=tuple(spec.nodes for spec in basis.radial_specs),
        basis_radial_n_points=tuple(
            spec.n_points for spec in basis.radial_specs
        ),
        quadrature_radial_nodes=tuple(
            spec.nodes for spec in quadrature.radial_specs
        ),
        quadrature_radial_n_points=tuple(
            spec.n_points for spec in quadrature.radial_specs
        ),
        angular_channels=basis.angular_channels,
        orbital_switch_iterations=basis.partition.switch_iterations,
        integration_switch_iterations=quadrature.partition.switch_iterations,
        lebedev_order=quadrature.lebedev_order,
        lebedev_rotation=_as_tuple_matrix(quadrature.rotation),
        basis_frame=_as_tuple_matrix(basis.frame),
        hermiticity_tolerance=hermiticity_tolerance,
    )


def assemble_becke_matrices(
    basis: BeckeSphericalBasis,
    quadrature: BeckeMulticenterQuadrature,
    nuclear_charges,
    *,
    particle_mass: float = 1.0,
    hermiticity_tolerance: float = 1.0e-11,
) -> BeckeDenseMatrices:
    """Assemble dense overlap, weak kinetic, and nuclear-potential matrices.

    The orbital partition contained in ``basis`` and the integration partition
    contained in ``quadrature`` remain independent.  The complete molecular
    Coulomb potential is evaluated on every integration shell.
    """

    charges, particle_mass, hermiticity_tolerance = _validate_assembly_inputs(
        basis,
        quadrature,
        nuclear_charges,
        particle_mass,
        hermiticity_tolerance,
    )
    geometry = classify_centers(basis.partition.centers)
    angular_mode, fixed_m = _angular_mode(basis, geometry)

    shape = (basis.size, basis.size)
    overlap = np.zeros(shape, dtype=complex)
    kinetic = np.zeros(shape, dtype=complex)
    potential = np.zeros(shape, dtype=complex)
    centers = basis.partition.centers

    for shell in quadrature.iter_shells():
        evaluation = basis.evaluate_shell(shell)
        values = evaluation.values
        gradients = evaluation.gradients
        weights = shell.weights

        overlap += values.conj().T @ (weights[:, np.newaxis] * values)
        for component in range(3):
            component_gradients = gradients[:, :, component]
            kinetic += component_gradients.conj().T @ (
                weights[:, np.newaxis] * component_gradients
            )

        distances = np.linalg.norm(
            shell.points[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=-1,
        )
        if np.any(distances == 0.0):
            raise FloatingPointError("a quadrature point coincides with a nucleus")
        nuclear_potential = -np.sum(charges / distances, axis=1)
        potential += values.conj().T @ (
            (weights * nuclear_potential)[:, np.newaxis] * values
        )

    kinetic *= 0.5 / particle_mass
    hamiltonian = kinetic + potential
    raw_matrices = {
        "overlap": overlap,
        "kinetic": kinetic,
        "potential": potential,
        "hamiltonian": hamiltonian,
    }
    relative_norms = {
        name: _relative_antihermitian_norm(matrix)
        for name, matrix in raw_matrices.items()
    }
    failures = {
        name: value
        for name, value in relative_norms.items()
        if not np.isfinite(value) or value > hermiticity_tolerance
    }
    if failures:
        details = ", ".join(
            f"{name}={value:.3e}" for name, value in failures.items()
        )
        raise FloatingPointError(
            "assembled matrices exceed the relative anti-Hermitian tolerance: "
            + details
        )

    overlap = 0.5 * (overlap + overlap.conj().T)
    kinetic = 0.5 * (kinetic + kinetic.conj().T)
    potential = 0.5 * (potential + potential.conj().T)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)

    overlap_eigenvalues = np.linalg.eigvalsh(overlap)
    minimum_eigenvalue = float(overlap_eigenvalues[0])
    maximum_eigenvalue = float(overlap_eigenvalues[-1])
    condition_number = (
        maximum_eigenvalue / minimum_eigenvalue
        if minimum_eigenvalue > 0.0
        else np.inf
    )

    center_indices = np.fromiter(
        (function.center_index for function in basis.functions),
        dtype=int,
        count=basis.size,
    )

    def matrix_diagnostics(name, matrix):
        same_center, cross_center = _center_block_norms(matrix, center_indices)
        return MatrixAssemblyDiagnostics(
            relative_antihermitian_norm=relative_norms[name],
            same_center_norm=same_center,
            cross_center_norm=cross_center,
        )

    diagnostics = BeckeAssemblyDiagnostics(
        overlap=matrix_diagnostics("overlap", overlap),
        kinetic=matrix_diagnostics("kinetic", kinetic),
        potential=matrix_diagnostics("potential", potential),
        hamiltonian=matrix_diagnostics("hamiltonian", hamiltonian),
        overlap_min_eigenvalue=minimum_eigenvalue,
        overlap_max_eigenvalue=maximum_eigenvalue,
        overlap_condition_number=float(condition_number),
        geometry=geometry,
        angular_mode=angular_mode,
        fixed_m=fixed_m,
        basis_size=basis.size,
        quadrature_size=quadrature.point_count,
        parameters=_parameters(
            basis,
            quadrature,
            charges,
            particle_mass,
            hermiticity_tolerance,
        ),
    )

    for matrix in (overlap, kinetic, potential, hamiltonian):
        matrix.setflags(write=False)
    return BeckeDenseMatrices(
        overlap=overlap,
        kinetic=kinetic,
        potential=potential,
        hamiltonian=hamiltonian,
        diagnostics=diagnostics,
    )
