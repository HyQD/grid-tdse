"""Atom-centered spherical FE-DVR basis values and Cartesian gradients."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from grid_lib.pseudospectral_grids import (
    FEMDVR,
    GaussLegendreLobatto,
    Linear_map,
)
from scipy.special import sph_harm_y

from .partition import BeckePartition
from .specification import FEMDVRQuadratureSpec


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def full_lm_channels(l_max: int) -> tuple[tuple[int, int], ...]:
    """Return all complex spherical-harmonic channels through ``l_max``."""

    l_max = _nonnegative_integer(l_max, "l_max")
    return tuple(
        (ell, m) for ell in range(l_max + 1) for m in range(-ell, ell + 1)
    )


def fixed_m_channels(l_max: int, m: int) -> tuple[tuple[int, int], ...]:
    """Return channels in one conserved signed-``m`` block."""

    l_max = _nonnegative_integer(l_max, "l_max")
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise TypeError("m must be an integer")
    m = int(m)
    if abs(m) > l_max:
        raise ValueError("abs(m) must not exceed l_max")
    return tuple((ell, m) for ell in range(abs(m), l_max + 1))


@dataclass(frozen=True)
class BeckeBasisFunction:
    """Index metadata for one weighted atom-centered basis function."""

    center_index: int
    radial_index: int
    ell: int
    m: int


@dataclass(frozen=True)
class BeckeBasisEvaluation:
    """Weighted basis values and Cartesian gradients at requested points."""

    values: np.ndarray
    gradients: np.ndarray


class _FEMDVRCardinalBasis:
    """Prototype element-aware evaluator for interior FEM-DVR cardinals."""

    def __init__(self, specification: FEMDVRQuadratureSpec) -> None:
        self.specification = specification
        self.dvr = FEMDVR(
            np.asarray(specification.nodes),
            np.asarray(specification.n_points),
            Linear_map,
            GaussLegendreLobatto,
        )
        self.nodes = np.array(self.dvr.r[1:-1], copy=True)
        self.nodes.setflags(write=False)
        self._barycentric_weights = tuple(
            1.0 / element.PN_x2
            for element in self.dvr.dvr
        )

    @property
    def size(self) -> int:
        return self.nodes.size

    def _element_indices(self, radii: np.ndarray, hints) -> np.ndarray:
        nodes = np.asarray(self.specification.nodes)
        elements = np.searchsorted(nodes, radii, side="right") - 1
        elements[radii == nodes[-1]] = self.specification.n_elements - 1
        outside = (radii < nodes[0]) | (radii > nodes[-1])
        elements[outside] = -1

        if hints is None:
            return elements
        raw_hints = np.asarray(hints)
        if raw_hints.shape != radii.shape:
            raise ValueError("element_hints must match the radii shape")
        if not np.issubdtype(raw_hints.dtype, np.integer):
            raise TypeError("element_hints must contain integers")
        hints = raw_hints.astype(int, copy=False)
        if np.any((hints < -1) | (hints >= self.specification.n_elements)):
            raise ValueError("element hint is out of range")
        selected = hints >= 0
        tolerance = 128.0 * np.finfo(float).eps * max(
            1.0, self.specification.r_max
        )
        for element in range(self.specification.n_elements):
            mask = selected & (hints == element)
            left = nodes[element]
            right = nodes[element + 1]
            outside_element = (radii < left - tolerance) | (
                radii > right + tolerance
            )
            if np.any(mask & outside_element):
                raise ValueError("element hint does not contain its radial point")
        elements[selected] = hints[selected]
        return elements

    def evaluate(self, radii, *, element_hints=None):
        """Return cardinal values and physical radial derivatives."""

        radii = np.asarray(radii, dtype=float)
        if not np.all(np.isfinite(radii)):
            raise ValueError("radii must contain only finite values")
        original_shape = radii.shape
        flat_radii = radii.reshape(-1)
        flat_hints = None
        if element_hints is not None:
            flat_hints = np.asarray(element_hints).reshape(-1)
        elements = self._element_indices(flat_radii, flat_hints)
        values = np.zeros((flat_radii.size, self.size))
        derivatives = np.zeros_like(values)

        for element_index in range(self.specification.n_elements):
            point_indices = np.flatnonzero(elements == element_index)
            if point_indices.size == 0:
                continue
            left = self.specification.nodes[element_index]
            right = self.specification.nodes[element_index + 1]
            reference_points = (
                2.0 * (flat_radii[point_indices] - left) / (right - left) - 1.0
            )
            element = self.dvr.dvr[element_index]
            local_values, local_derivatives = self._local_values_and_derivatives(
                element_index, reference_points
            )
            local_to_global = self.dvr.element_dof_maps[element_index]
            for local_index, global_index in enumerate(local_to_global):
                if global_index == 0 or global_index == self.dvr.r.size - 1:
                    continue
                interior_index = global_index - 1
                values[point_indices, interior_index] = local_values[:, local_index]
                derivatives[point_indices, interior_index] = local_derivatives[
                    :, local_index
                ]

        output_shape = original_shape + (self.size,)
        return values.reshape(output_shape), derivatives.reshape(output_shape)

    def _local_values_and_derivatives(self, element_index, reference_points):
        element = self.dvr.dvr[element_index]
        nodes = element.x
        barycentric_weights = self._barycentric_weights[element_index]
        values = np.empty((reference_points.size, nodes.size))
        derivatives = np.empty_like(values)
        reference_tolerance = 128.0 * np.finfo(float).eps
        physical_scale = 2.0 / (
            self.specification.nodes[element_index + 1]
            - self.specification.nodes[element_index]
        )

        for point_index, point in enumerate(reference_points):
            difference = point - nodes
            close = np.flatnonzero(np.abs(difference) <= reference_tolerance)
            if close.size:
                node_index = int(close[0])
                values[point_index] = 0.0
                values[point_index, node_index] = 1.0
                derivatives[point_index] = element.D1[node_index]
                continue

            scaled = barycentric_weights / difference
            denominator = np.sum(scaled)
            cardinal_values = scaled / denominator
            second_sum = np.sum(barycentric_weights / difference**2)
            reference_derivatives = cardinal_values * (
                second_sum / denominator - 1.0 / difference
            )
            values[point_index] = cardinal_values
            derivatives[point_index] = reference_derivatives * physical_scale

        return values, derivatives


def _spherical_harmonics_and_gradients(unit_vectors, channels):
    """Return ``Y_lm`` and its dimensionless Cartesian angular gradient."""

    unit_vectors = np.asarray(unit_vectors)
    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    z = np.clip(unit_vectors[:, 2], -1.0, 1.0)
    sin_theta = np.hypot(x, y)
    theta = np.arctan2(sin_theta, z)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    pole = sin_theta <= 128.0 * np.finfo(float).eps
    regular = ~pole

    values = np.empty((unit_vectors.shape[0], len(channels)), dtype=complex)
    gradients = np.zeros(
        (unit_vectors.shape[0], len(channels), 3), dtype=complex
    )
    for channel_index, (ell, m) in enumerate(channels):
        harmonic = np.asarray(sph_harm_y(ell, m, theta, phi), dtype=complex)
        if np.any(pole):
            harmonic = np.array(harmonic, copy=True)
            harmonic[pole] = 0.0
            if m == 0:
                normalization = np.sqrt((2 * ell + 1) / (4.0 * np.pi))
                harmonic[pole] = normalization * np.where(
                    z[pole] >= 0.0, 1.0, (-1.0) ** ell
                )
        values[:, channel_index] = harmonic

        if np.any(regular):
            regular_theta = theta[regular]
            regular_phi = phi[regular]
            regular_harmonic = harmonic[regular]
            d_theta = m * z[regular] / sin_theta[regular] * regular_harmonic
            if m < ell:
                coefficient = np.sqrt((ell - m) * (ell + m + 1))
                d_theta += (
                    coefficient
                    * np.exp(-1j * regular_phi)
                    * sph_harm_y(
                        ell, m + 1, regular_theta, regular_phi
                    )
                )
            e_theta = np.column_stack(
                (
                    z[regular] * x[regular] / sin_theta[regular],
                    z[regular] * y[regular] / sin_theta[regular],
                    -sin_theta[regular],
                )
            )
            e_phi = np.column_stack(
                (
                    -y[regular] / sin_theta[regular],
                    x[regular] / sin_theta[regular],
                    np.zeros(np.count_nonzero(regular)),
                )
            )
            gradients[regular, channel_index, :] = (
                d_theta[:, np.newaxis] * e_theta
                + (1j * m * regular_harmonic / sin_theta[regular])[
                    :, np.newaxis
                ]
                * e_phi
            )

        if np.any(pole) and abs(m) == 1:
            amplitude = np.sqrt(
                (2 * ell + 1) * ell * (ell + 1) / (16.0 * np.pi)
            )
            north = pole & (z >= 0.0)
            south = pole & (z < 0.0)
            if m == 1:
                gradients[north, channel_index, 0] = -amplitude
                gradients[north, channel_index, 1] = -1j * amplitude
                gradients[south, channel_index, 0] = (-1) ** ell * amplitude
                gradients[south, channel_index, 1] = (
                    (-1) ** ell * 1j * amplitude
                )
            else:
                gradients[north, channel_index, 0] = amplitude
                gradients[north, channel_index, 1] = -1j * amplitude
                gradients[south, channel_index, 0] = (
                    (-1) ** (ell + 1) * amplitude
                )
                gradients[south, channel_index, 1] = (
                    (-1) ** ell * 1j * amplitude
                )

    return values, gradients


class BeckeSphericalBasis:
    """Weighted multicenter spherical-harmonic FEM-DVR basis.

    Basis functions are ordered by center, interior radial cardinal index,
    and then the explicit angular-channel order supplied for that center.
    All centers share one parallel local Cartesian frame.
    """

    def __init__(
        self,
        partition: BeckePartition,
        radial_specs,
        angular_channels,
        *,
        frame=None,
    ) -> None:
        if not isinstance(partition, BeckePartition):
            raise TypeError("partition must be a BeckePartition")
        self._partition = partition
        self._radial_specs = self._validated_radial_specs(radial_specs)
        self._channels = self._validated_channels(angular_channels)
        self._frame = self._validated_frame(frame)
        self._radial_bases = tuple(
            _FEMDVRCardinalBasis(specification)
            for specification in self._radial_specs
        )

        functions = []
        offsets = [0]
        for center_index, (radial_basis, channels) in enumerate(
            zip(self._radial_bases, self._channels)
        ):
            for radial_index in range(radial_basis.size):
                for ell, m in channels:
                    functions.append(
                        BeckeBasisFunction(center_index, radial_index, ell, m)
                    )
            offsets.append(len(functions))
        self._functions = tuple(functions)
        self._center_offsets = tuple(offsets)

        scale = max(
            1.0,
            float(np.max(self._partition.pair_distances)),
            max(specification.r_max for specification in self._radial_specs),
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
    def _is_channel(value) -> bool:
        try:
            if len(value) != 2:
                return False
            ell, m = value
        except (TypeError, ValueError):
            return False
        return all(
            not isinstance(number, (bool, np.bool_))
            and isinstance(number, Integral)
            for number in (ell, m)
        )

    def _validated_channels(self, angular_channels):
        try:
            supplied = tuple(angular_channels)
        except TypeError as error:
            raise TypeError("angular_channels must be a sequence") from error
        if supplied and all(self._is_channel(channel) for channel in supplied):
            center_channels = (supplied,) * self._partition.n_centers
        else:
            if len(supplied) != self._partition.n_centers:
                raise ValueError(
                    "angular_channels must be common or contain one table per center"
                )
            center_channels = tuple(tuple(table) for table in supplied)

        validated = []
        for table in center_channels:
            if not table:
                raise ValueError("each center must contain at least one channel")
            channels = []
            for channel in table:
                if not self._is_channel(channel):
                    raise TypeError("each angular channel must be an integer pair")
                ell, m = (int(channel[0]), int(channel[1]))
                if ell < 0 or abs(m) > ell:
                    raise ValueError(
                        "angular channels must satisfy ell >= 0 and |m| <= ell"
                    )
                channels.append((ell, m))
            if len(set(channels)) != len(channels):
                raise ValueError("angular channels must not contain duplicates")
            validated.append(tuple(channels))
        return tuple(validated)

    @staticmethod
    def _validated_frame(frame) -> np.ndarray:
        result = np.eye(3) if frame is None else np.asarray(frame, dtype=float)
        if result.shape != (3, 3):
            raise ValueError("frame must have shape (3, 3)")
        result = np.array(result, copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError("frame must contain only finite values")
        if not np.allclose(result.T @ result, np.eye(3), atol=2e-14, rtol=0.0):
            raise ValueError("frame must be orthogonal")
        if not np.isclose(np.linalg.det(result), 1.0, atol=2e-14, rtol=0.0):
            raise ValueError("frame must be right-handed")
        result.setflags(write=False)
        return result

    @property
    def partition(self) -> BeckePartition:
        return self._partition

    @property
    def radial_specs(self) -> tuple[FEMDVRQuadratureSpec, ...]:
        return self._radial_specs

    @property
    def angular_channels(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return self._channels

    @property
    def frame(self) -> np.ndarray:
        return self._frame

    @property
    def functions(self) -> tuple[BeckeBasisFunction, ...]:
        return self._functions

    @property
    def size(self) -> int:
        return len(self._functions)

    def _validated_hints(self, element_hints, leading_shape):
        if element_hints is None:
            return None
        raw_hints = np.asarray(element_hints)
        expected_shape = leading_shape + (self._partition.n_centers,)
        if raw_hints.shape != expected_shape:
            raise ValueError(f"element_hints must have shape {expected_shape}")
        if not np.issubdtype(raw_hints.dtype, np.integer):
            raise TypeError("element_hints must contain integers")
        return raw_hints.reshape(-1, self._partition.n_centers)

    def evaluate(self, points, *, element_hints=None) -> BeckeBasisEvaluation:
        """Evaluate weighted basis functions and Cartesian gradients."""

        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")
        leading_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        flat_hints = self._validated_hints(element_hints, leading_shape)
        partition_evaluation = self._partition.evaluate(flat_points)
        if np.any(
            partition_evaluation.center_distances <= self._nuclear_tolerance
        ):
            raise ValueError("basis evaluation points must not coincide with a center")

        values = np.zeros((flat_points.shape[0], self.size), dtype=complex)
        gradients = np.zeros(
            (flat_points.shape[0], self.size, 3), dtype=complex
        )
        for center_index, (radial_basis, channels) in enumerate(
            zip(self._radial_bases, self._channels)
        ):
            radii = partition_evaluation.center_distances[..., center_index]
            hints = None if flat_hints is None else flat_hints[:, center_index]
            radial_values, radial_derivatives = radial_basis.evaluate(
                radii, element_hints=hints
            )
            global_unit_vectors = partition_evaluation.center_unit_vectors[
                ..., center_index, :
            ]
            local_unit_vectors = global_unit_vectors @ self._frame
            harmonics, local_angular_gradients = (
                _spherical_harmonics_and_gradients(local_unit_vectors, channels)
            )
            angular_gradients = local_angular_gradients @ self._frame.T

            inverse_radii = 1.0 / radii
            radial_factors = radial_values * inverse_radii[:, np.newaxis]
            radial_factor_derivatives = (
                radial_derivatives * inverse_radii[:, np.newaxis]
                - radial_values * inverse_radii[:, np.newaxis] ** 2
            )
            local_values = (
                radial_factors[:, :, np.newaxis]
                * harmonics[:, np.newaxis, :]
            )
            local_gradients = (
                radial_factor_derivatives[:, :, np.newaxis, np.newaxis]
                * harmonics[:, np.newaxis, :, np.newaxis]
                * global_unit_vectors[:, np.newaxis, np.newaxis, :]
                + radial_values[:, :, np.newaxis, np.newaxis]
                * inverse_radii[:, np.newaxis, np.newaxis, np.newaxis] ** 2
                * angular_gradients[:, np.newaxis, :, :]
            )

            partition_weights = partition_evaluation.weights[..., center_index]
            partition_gradients = partition_evaluation.weight_gradients[
                ..., center_index, :
            ]
            weighted_values = partition_weights[:, np.newaxis, np.newaxis] * (
                local_values
            )
            weighted_gradients = (
                partition_weights[:, np.newaxis, np.newaxis, np.newaxis]
                * local_gradients
                + local_values[..., np.newaxis]
                * partition_gradients[:, np.newaxis, np.newaxis, :]
            )
            start = self._center_offsets[center_index]
            stop = self._center_offsets[center_index + 1]
            values[:, start:stop] = weighted_values.reshape(
                flat_points.shape[0], -1
            )
            gradients[:, start:stop, :] = weighted_gradients.reshape(
                flat_points.shape[0], -1, 3
            )

        return BeckeBasisEvaluation(
            values=values.reshape(leading_shape + (self.size,)),
            gradients=gradients.reshape(leading_shape + (self.size, 3)),
        )

    def evaluate_shell(self, shell) -> BeckeBasisEvaluation:
        """Evaluate a quadrature shell with its owning element trace."""

        hints = np.full(
            (shell.points.shape[0], self._partition.n_centers), -1, dtype=int
        )
        hints[:, shell.center_index] = shell.element_index
        return self.evaluate(shell.points, element_hints=hints)
