"""PySCF-backed atom-centered grids with user-supplied radial rules."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..one_radial_becke import FEMDVRQuadratureSpec


RadialRule = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def _axis_angle_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    cosine = np.cos(angle)
    return (
        cosine * np.eye(3)
        + (1.0 - cosine) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


GENERIC_LEBEDEV_ROTATION = _axis_angle_rotation([1.0, 2.0, 3.0], 0.437)


@dataclass(frozen=True)
class PySCFBeckeGrid:
    """A flattened, spatially grouped atom-centered integration grid."""

    coordinates: np.ndarray
    weights: np.ndarray
    origin_atoms: np.ndarray
    quadrature_weights: np.ndarray

    @property
    def point_count(self) -> int:
        """Number of grid points."""

        return self.weights.size


def composite_gauss_legendre(
    specification: FEMDVRQuadratureSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interior Gauss-Legendre points and weights in every FE interval.

    The finite-element boundaries and the order in each interval are supplied
    by ``specification``.  Unlike a Lobatto rule, no point lies exactly on an
    element boundary, which makes arbitrary-point FEM basis gradients
    unambiguous.
    """

    radii = []
    weights = []
    for left, right, order in zip(
        specification.nodes[:-1],
        specification.nodes[1:],
        specification.n_points,
    ):
        reference_radii, reference_weights = np.polynomial.legendre.leggauss(order)
        scale = 0.5 * (right - left)
        radii.append(0.5 * (right + left) + scale * reference_radii)
        weights.append(scale * reference_weights)
    return np.concatenate(radii), np.concatenate(weights)


def fem_lobatto(
    specification: FEMDVRQuadratureSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Håkon's element-local FEM-Lobatto radial rule.

    Internal boundaries occur twice with the weights from their two parent
    elements.  The origin is removed because its three-dimensional volume
    weight is zero and evaluating a Coulomb potential there is singular.
    """

    from grid_lib.pseudospectral_grids import GaussLegendreLobatto, Linear_map

    radii = []
    weights = []
    for left, right, order in zip(
        specification.nodes[:-1],
        specification.nodes[1:],
        specification.n_points,
    ):
        rule = GaussLegendreLobatto(
            order - 1,
            Linear_map(left, right),
            symmetrize=False,
        )
        keep = rule.r != 0.0
        radii.append(np.asarray(rule.r[keep]))
        weights.append(np.asarray(rule.weights[keep]))
    return np.concatenate(radii), np.concatenate(weights)


def _validated_atomic_numbers(atomic_numbers, n_centers):
    raw = np.asarray(atomic_numbers)
    if raw.shape != (n_centers,):
        raise ValueError("atomic_numbers must contain one entry per center")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("atomic_numbers must contain integers")
    result = raw.astype(int, copy=False)
    if np.any((result < 1) | (result > 118)):
        raise ValueError("atomic_numbers must be between 1 and 118")
    return result


def _validated_angular_points(angular_points):
    if isinstance(angular_points, (bool, np.bool_)) or not isinstance(
        angular_points, Integral
    ):
        raise TypeError("angular_points must be an integer")
    if int(angular_points) < 1:
        raise ValueError("angular_points must be positive")
    return int(angular_points)


def _validated_radial_output(radial_rule, atom_index, atomic_number):
    radii, weights = radial_rule(atom_index, atomic_number)
    radii = np.asarray(radii, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("the radial rule must return a nonempty 1D radius array")
    if weights.shape != radii.shape:
        raise ValueError("radial radii and weights must have the same shape")
    if not np.all(np.isfinite(radii)) or not np.all(np.isfinite(weights)):
        raise ValueError("radial radii and weights must be finite")
    if np.any(radii <= 0.0):
        raise ValueError("radial radii must be positive")
    if np.any(weights <= 0.0):
        raise ValueError("radial weights must be positive")
    return radii, weights


def build_pyscf_becke_grid(
    centers,
    atomic_numbers,
    radial_rule: RadialRule,
    *,
    angular_points=590,
    prune="nwchem",
    radii_adjust=None,
    sort_grid=True,
    rotation=None,
) -> PySCFBeckeGrid:
    """Build a molecular Becke grid around arbitrary atomic centers.

    ``radial_rule(atom_index, atomic_number)`` returns physical radii and
    one-dimensional integration weights.  Its array length is unrestricted;
    the nominal radial count required by PySCF is deliberately ignored.

    Parameters
    ----------
    centers
        Cartesian center positions in bohr, shape ``(n_centers, 3)``.
    atomic_numbers
        Nuclear charges used by PySCF's pruning and size corrections.
    radial_rule
        User-owned radial quadrature callback.
    angular_points
        Maximum number of Lebedev points per radial shell.
    prune
        ``"nwchem"`` for PySCF's default angular pruning, ``None`` for no
        pruning, or a PySCF-compatible pruning callable.
    radii_adjust
        Optional PySCF atomic-size adjustment.  ``None`` gives the same
        geometry-only Becke partition as the current prototype.
    sort_grid
        Spatially group the final molecular grid into PySCF boxes.
    rotation
        Optional proper rotation applied to every atom-centered angular grid
        before Becke partitioning.  This is useful for avoiding alignment of
        Lebedev axes with nuclei.
    """

    try:
        from pyscf import dft, gto
        from pyscf.data import elements
    except ImportError as error:
        raise ImportError(
            "PySCF grid support requires the optional 'pyscf' dependency"
        ) from error

    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1:] != (3,) or centers.shape[0] == 0:
        raise ValueError("centers must have shape (n_centers, 3)")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must be finite")
    atomic_numbers = _validated_atomic_numbers(atomic_numbers, centers.shape[0])
    angular_points = _validated_angular_points(angular_points)
    if not callable(radial_rule):
        raise TypeError("radial_rule must be callable")
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-14, rtol=0):
            raise ValueError("rotation must be orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-14, rtol=0):
            raise ValueError("rotation must be proper")

    labels = [
        f"{elements.ELEMENTS[atomic_number]}{atom_index + 1}"
        for atom_index, atomic_number in enumerate(atomic_numbers)
    ]
    dummy_basis = {
        elements.ELEMENTS[atomic_number]: [[0, [1.0, 1.0]]]
        for atomic_number in np.unique(atomic_numbers)
    }
    molecule = gto.M(
        atom=list(zip(labels, centers)),
        unit="Bohr",
        basis=dummy_basis,
        charge=int(np.sum(atomic_numbers)),
        spin=0,
        verbose=0,
    )

    radial_data = tuple(
        _validated_radial_output(radial_rule, atom_index, int(atomic_number))
        for atom_index, atomic_number in enumerate(atomic_numbers)
    )

    def pyscf_radial_method(_nominal_count, _charge, atom_index, **_kwargs):
        return radial_data[atom_index]

    if prune == "nwchem":
        pruning_function = dft.gen_grid.nwchem_prune
    elif prune is None:
        def pruning_function(_charge, radii, maximum_angular_points):
            return np.full(radii.size, maximum_angular_points, dtype=int)
    elif callable(prune):
        pruning_function = prune
    else:
        raise TypeError("prune must be 'nwchem', None, or callable")

    grid = dft.gen_grid.Grids(molecule)
    grid.atom_grid = {label: (1, angular_points) for label in labels}
    grid.radi_method = pyscf_radial_method
    grid.prune = pruning_function
    grid.becke_scheme = dft.gen_grid.original_becke
    grid.radii_adjust = radii_adjust
    grid.alignment = 1
    atom_grids = grid.gen_atomic_grids(
        molecule,
        grid.atom_grid,
        grid.radi_method,
        grid.level,
        grid.prune,
    )
    if rotation is not None:
        atom_grids = {
            label: (coordinates @ rotation.T, volumes)
            for label, (coordinates, volumes) in atom_grids.items()
        }
    coordinates, weights = grid.get_partition(
        molecule,
        atom_grids,
        grid.radii_adjust,
        grid.atomic_radii,
        grid.becke_scheme,
    )
    origin_atoms = np.concatenate(
        [
            np.full(atom_grids[label][1].size, atom_index, dtype=np.int32)
            for atom_index, label in enumerate(labels)
        ]
    )
    quadrature_weights = np.concatenate(
        [atom_grids[label][1] for label in labels]
    )
    if sort_grid:
        order = dft.gen_grid.arg_group_grids(molecule, coordinates)
        coordinates = coordinates[order]
        weights = weights[order]
        origin_atoms = origin_atoms[order]
        quadrature_weights = quadrature_weights[order]

    arrays = tuple(
        np.asarray(array)
        for array in (coordinates, weights, origin_atoms, quadrature_weights)
    )
    for array in arrays:
        array.setflags(write=False)
    return PySCFBeckeGrid(*arrays)
