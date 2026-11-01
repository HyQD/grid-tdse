"""Radial-Lebedev quadrature for multicenter Becke integration."""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Iterator

import numpy as np
from grid_lib.pseudospectral_grids import GaussLegendreLobatto, Linear_map
from scipy.integrate import lebedev_rule

from .partition import BeckePartition
from .specification import FEMDVRQuadratureSpec


def _axis_angle_rotation(axis, angle: float) -> np.ndarray:
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


_DEFAULT_LEBEDEV_ROTATION = _axis_angle_rotation([1.0, 2.0, 3.0], 0.437)


@dataclass(frozen=True)
class BeckeQuadratureShell:
    """One radial shell of an atom-centered Becke quadrature."""

    center_index: int
    element_index: int
    local_radial_index: int
    radius: float
    radial_weight: float
    points: np.ndarray
    directions: np.ndarray
    angular_weights: np.ndarray
    partition_weights: np.ndarray
    base_volume_weights: np.ndarray
    weights: np.ndarray


class BeckeMulticenterQuadrature:
    r"""Element-local radial-Lebedev quadrature with Becke weights.

    For center ``C``, radius ``r_j``, and rotated Lebedev direction
    ``n_q``, a shell contains

    .. math::

        \mathbf x_{Cjq}=\mathbf R_C+r_j\mathbf n_q,

    with final integration weights

    .. math::

        W_{Cjq}=w_j^r r_j^2 w_q^\Omega q_C(\mathbf x_{Cjq}).

    Shells are generated lazily. Internal radial element boundaries occur in
    both adjacent elements, carrying the appropriate element-local weights.

    Parameters
    ----------
    partition : BeckePartition
        Integration partition. A distinct object may be used later for the
        orbital-space partition.
    radial_specs : FEMDVRQuadratureSpec or sequence thereof
        One radial specification shared by all centers, or one specification
        per center.
    lebedev_order : int
        Order accepted by :func:`scipy.integrate.lebedev_rule`.
    rotation : array_like, shape (3, 3), optional
        Proper Cartesian rotation applied to the Lebedev directions. A fixed
        generic rotation is used by default.
    """

    _ALIGNMENT_TOLERANCE = 1.0e-12

    def __init__(
        self,
        partition: BeckePartition,
        radial_specs,
        lebedev_order: int,
        *,
        rotation=None,
    ) -> None:
        if not isinstance(partition, BeckePartition):
            raise TypeError("partition must be a BeckePartition")
        self._partition = partition
        self._radial_specs = self._validated_radial_specs(radial_specs)
        self._lebedev_order = self._validated_lebedev_order(lebedev_order)
        self._rotation = self._validated_rotation(rotation)

        xyz, angular_weights = lebedev_rule(self._lebedev_order)
        directions = xyz.T @ self._rotation.T
        self._validate_directions(directions)
        self._directions = np.asarray(directions)
        self._angular_weights = np.asarray(angular_weights)
        self._directions.setflags(write=False)
        self._angular_weights.setflags(write=False)

        scale = max(
            1.0,
            float(np.max(self._partition.pair_distances)),
            max(spec.r_max for spec in self._radial_specs),
        )
        self._nuclear_tolerance = 128.0 * np.finfo(float).eps * scale

    def _validated_radial_specs(self, radial_specs):
        if isinstance(radial_specs, FEMDVRQuadratureSpec):
            return (radial_specs,) * self._partition.n_centers
        try:
            specifications = tuple(radial_specs)
        except TypeError as error:
            raise TypeError(
                "radial_specs must be a FEMDVRQuadratureSpec or a sequence"
            ) from error
        if len(specifications) != self._partition.n_centers:
            raise ValueError("radial_specs must contain one entry per center")
        if not all(
            isinstance(specification, FEMDVRQuadratureSpec)
            for specification in specifications
        ):
            raise TypeError("every radial specification must be FEM-DVR")
        return specifications

    @staticmethod
    def _validated_lebedev_order(order: int) -> int:
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral):
            raise TypeError("lebedev_order must be an integer")
        order = int(order)
        try:
            lebedev_rule(order)
        except NotImplementedError as error:
            raise ValueError(f"unsupported Lebedev order: {order}") from error
        return order

    @staticmethod
    def _validated_rotation(rotation) -> np.ndarray:
        if rotation is None:
            result = np.array(_DEFAULT_LEBEDEV_ROTATION, copy=True)
        else:
            result = np.asarray(rotation, dtype=float)
            if result.shape != (3, 3):
                raise ValueError("rotation must have shape (3, 3)")
            result = np.array(result, copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError("rotation must contain only finite values")
        if not np.allclose(result.T @ result, np.eye(3), atol=2e-14, rtol=0.0):
            raise ValueError("rotation must be orthogonal")
        if not np.isclose(np.linalg.det(result), 1.0, atol=2e-14, rtol=0.0):
            raise ValueError("rotation must be a proper rotation")
        result.setflags(write=False)
        return result

    def _validate_directions(self, directions: np.ndarray) -> None:
        tolerance = self._ALIGNMENT_TOLERANCE
        if np.any(np.abs(directions[:, 2]) >= 1.0 - tolerance):
            raise ValueError("rotated Lebedev directions contain a spherical pole")

        centers = self._partition.centers
        for center_a in range(self._partition.n_centers):
            for center_b in range(center_a + 1, self._partition.n_centers):
                vector = centers[center_b] - centers[center_a]
                unit_vector = vector / np.linalg.norm(vector)
                if np.any(np.abs(directions @ unit_vector) >= 1.0 - tolerance):
                    raise ValueError(
                        "rotated Lebedev direction aligns with an intercenter axis"
                    )

    @property
    def partition(self) -> BeckePartition:
        """Integration partition used by the quadrature."""

        return self._partition

    @property
    def radial_specs(self) -> tuple[FEMDVRQuadratureSpec, ...]:
        """Radial specifications in center order."""

        return self._radial_specs

    @property
    def lebedev_order(self) -> int:
        """Lebedev quadrature order."""

        return self._lebedev_order

    @property
    def rotation(self) -> np.ndarray:
        """Read-only rotation applied to the Lebedev rule."""

        return self._rotation

    @property
    def directions(self) -> np.ndarray:
        """Read-only rotated Lebedev directions with shape ``(n_omega, 3)``."""

        return self._directions

    @property
    def angular_weights(self) -> np.ndarray:
        """Read-only solid-angle weights summing to ``4*pi``."""

        return self._angular_weights

    @property
    def point_count(self) -> int:
        """Total number of points across all generated radial shells."""

        shell_count = sum(spec.nonzero_shell_count for spec in self._radial_specs)
        return shell_count * self._angular_weights.size

    def _element_rules(self, specification: FEMDVRQuadratureSpec):
        for element_index, (left, right, n_points) in enumerate(
            zip(
                specification.nodes[:-1],
                specification.nodes[1:],
                specification.n_points,
            )
        ):
            rule = GaussLegendreLobatto(
                n_points - 1,
                Linear_map(left, right),
                symmetrize=False,
            )
            yield element_index, rule.r, rule.weights

    def iter_shells(self, center=None) -> Iterator[BeckeQuadratureShell]:
        """Yield element-local radial shells, optionally for one center."""

        if center is None:
            center_indices = range(self._partition.n_centers)
        else:
            if isinstance(center, (bool, np.bool_)) or not isinstance(
                center, Integral
            ):
                raise TypeError("center must be an integer or None")
            center = int(center)
            if not 0 <= center < self._partition.n_centers:
                raise ValueError("center index is out of range")
            center_indices = (center,)

        for center_index in center_indices:
            center_position = self._partition.centers[center_index]
            specification = self._radial_specs[center_index]
            for element_index, radii, radial_weights in self._element_rules(
                specification
            ):
                for local_index, (radius, radial_weight) in enumerate(
                    zip(radii, radial_weights)
                ):
                    if radius == 0.0:
                        continue
                    points = center_position + radius * self._directions
                    distances = np.linalg.norm(
                        points[:, np.newaxis, :]
                        - self._partition.centers[np.newaxis, :, :],
                        axis=-1,
                    )
                    if np.any(distances <= self._nuclear_tolerance):
                        raise FloatingPointError(
                            "a quadrature point coincides with a center"
                        )
                    partition_weights = self._partition.weights(points)[
                        ..., center_index
                    ]
                    base_weights = (
                        float(radial_weight)
                        * float(radius) ** 2
                        * self._angular_weights
                    )
                    yield BeckeQuadratureShell(
                        center_index=center_index,
                        element_index=element_index,
                        local_radial_index=local_index,
                        radius=float(radius),
                        radial_weight=float(radial_weight),
                        points=points,
                        directions=self._directions,
                        angular_weights=self._angular_weights,
                        partition_weights=partition_weights,
                        base_volume_weights=base_weights,
                        weights=base_weights * partition_weights,
                    )

    @staticmethod
    def _shell_integral(shell: BeckeQuadratureShell, function: Callable):
        values = np.asarray(function(shell.points))
        if values.ndim == 0:
            values = np.broadcast_to(values, (shell.weights.size,))
        if values.shape[0] != shell.weights.size:
            raise ValueError(
                "integrand values must have the angular point axis first"
            )
        return np.tensordot(shell.weights, values, axes=(0, 0))

    def integrate_by_center(self, function: Callable) -> np.ndarray:
        """Integrate a callable and retain one contribution per center."""

        if not callable(function):
            raise TypeError("function must be callable")
        results = []
        for center_index in range(self._partition.n_centers):
            result = None
            for shell in self.iter_shells(center_index):
                contribution = self._shell_integral(shell, function)
                result = contribution if result is None else result + contribution
            results.append(result)
        return np.stack(results, axis=0)

    def integrate(self, function: Callable):
        """Integrate a scalar- or array-valued Cartesian callable."""

        return np.sum(self.integrate_by_center(function), axis=0)
