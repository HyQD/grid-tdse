"""Fixed-m prolate-spheroidal wavefunction representation."""

from numbers import Integral

import numpy as np

from grid_lib.prolate_spheroidal_coordinates import (
    active_indices as prolate_active_indices,
    from_euclidean_coefficients,
    overlap_diagonal,
    setup_prolate_spheroidal_grid_from_boundaries,
    to_euclidean_coefficients,
)


class ProlateSpheroidalRepresentation:
    """Axis-free tensor-product representation for conserved signed ``m``.

    Public nodal states use trailing shape ``fixed_m_shape(m)``. With the
    default outer Dirichlet condition this is ``(n_xi - 1, n_eta)``; the
    omitted row is the finite ``xi_max`` boundary.
    """

    def __init__(
        self,
        *,
        internuclear_distance,
        xi_boundaries,
        xi_points_per_element=8,
        eta_degree=8,
        m_max=0,
        xi_outer_dirichlet=True,
    ) -> None:
        self.m_max = _nonnegative_integer(m_max, "m_max")
        if not isinstance(xi_outer_dirichlet, (bool, np.bool_)):
            raise TypeError("xi_outer_dirichlet must be boolean")
        self.xi_outer_dirichlet = bool(xi_outer_dirichlet)
        self.grid = setup_prolate_spheroidal_grid_from_boundaries(
            internuclear_distance=internuclear_distance,
            xi_boundaries=xi_boundaries,
            xi_points_per_element=xi_points_per_element,
            eta_degree=eta_degree,
        )
        self.xi_boundaries = tuple(
            float(value) for value in self.grid.xi_grid.nodes
        )
        self.xi_points_per_element = _point_count_tuple(
            xi_points_per_element,
            len(self.xi_boundaries) - 1,
        )
        self.eta_degree = int(self.grid.eta.size)

        self._indices = {}
        self._metric = {}
        self._shape = {}
        for m in range(-self.m_max, self.m_max + 1):
            indices = np.asarray(
                prolate_active_indices(
                    self.grid,
                    m=m,
                    xi_outer_dirichlet=self.xi_outer_dirichlet,
                ),
                dtype=int,
            )
            shape = self._rectangular_active_shape(indices)
            metric = overlap_diagonal(self.grid)[indices].reshape(shape)
            indices.setflags(write=False)
            metric.setflags(write=False)
            self._indices[m] = indices
            self._metric[m] = metric
            self._shape[m] = shape

    def _rectangular_active_shape(self, indices) -> tuple[int, int]:
        n_eta = self.grid.eta.size
        if indices.size % n_eta:
            raise ValueError("active prolate grid is not rectangular")
        shape = (indices.size // n_eta, n_eta)
        expected = np.arange(indices.size)
        if not np.array_equal(indices, expected):
            raise ValueError(
                "the current nodal representation requires a contiguous "
                "rectangular active grid"
            )
        return shape

    def validate_m(self, m) -> int:
        """Return a validated signed magnetic quantum number."""

        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
            raise TypeError("m must be an integer")
        result = int(m)
        if abs(result) > self.m_max:
            raise ValueError(f"abs(m) must not exceed {self.m_max}")
        return result

    def fixed_m_shape(self, m=0) -> tuple[int, int]:
        """Return the public nodal-state shape for one signed ``m`` sector."""

        return self._shape[self.validate_m(m)]

    def active_indices(self, m=0) -> np.ndarray:
        """Return read-only flattened full-grid indices for one sector."""

        return self._indices[self.validate_m(m)]

    def metric_weights(self, m=0) -> np.ndarray:
        """Return the read-only diagonal physical metric in nodal shape."""

        return self._metric[self.validate_m(m)]

    def validate_fixed_m_state(self, state, m=0) -> np.ndarray:
        """Validate a nodal state or batch and return an array view."""

        shape = self.fixed_m_shape(m)
        values = np.asarray(state)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("state must contain numeric values")
        if values.ndim < 2 or values.shape[-2:] != shape:
            raise ValueError(f"state must have trailing shape {shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("state must contain only finite values")
        return values

    def to_euclidean(self, state, m=0) -> np.ndarray:
        """Map nodal values to the Euclidean ``S**(1/2)`` representation."""

        m = self.validate_m(m)
        values = self.validate_fixed_m_state(state, m)
        shape = self.fixed_m_shape(m)
        flat = values.reshape((*values.shape[:-2], -1))
        transformed = to_euclidean_coefficients(
            self.grid,
            flat,
            active_indices=self.active_indices(m),
        )
        return transformed.reshape(values.shape[:-2] + shape)

    def from_euclidean(self, state, m=0) -> np.ndarray:
        """Map Euclidean values back to physical nodal coefficients."""

        m = self.validate_m(m)
        values = self.validate_fixed_m_state(state, m)
        shape = self.fixed_m_shape(m)
        flat = values.reshape((*values.shape[:-2], -1))
        transformed = from_euclidean_coefficients(
            self.grid,
            flat,
            active_indices=self.active_indices(m),
        )
        return transformed.reshape(values.shape[:-2] + shape)

    def inner_product(self, left, right, m=0) -> complex:
        """Return the physical inner product for nodal states."""

        left_values = self.validate_fixed_m_state(left, m)
        right_values = self.validate_fixed_m_state(right, m)
        if left_values.shape != right_values.shape:
            raise ValueError("left and right must have the same shape")
        weight_shape = (
            (1,) * (left_values.ndim - 2)
            + self.metric_weights(m).shape
        )
        return np.sum(
            left_values.conj()
            * right_values
            * self.metric_weights(m).reshape(weight_shape)
        )

    def invert(self, state, m=0) -> np.ndarray:
        """Apply spatial inversion to one or more fixed-``m`` states.

        In prolate coordinates, inversion maps ``eta`` to ``-eta`` and
        ``phi`` to ``phi + pi``. The azimuthal factor therefore contributes
        ``(-1)**m`` in addition to reflection of the ``eta`` nodes.
        """

        m = self.validate_m(m)
        values = self.validate_fixed_m_state(state, m)
        phase = -1 if abs(m) % 2 else 1
        return phase * values[..., :, ::-1]

    def inversion_expectation(self, state, m=0):
        """Return the normalized inversion expectation value.

        A single state returns a float. A batch of states returns an array
        over the leading dimensions.
        """

        m = self.validate_m(m)
        values = self.validate_fixed_m_state(state, m)
        inverted = self.invert(values, m)
        weights = self.metric_weights(m)
        numerator = np.sum(
            values.conj() * inverted * weights,
            axis=(-2, -1),
        )
        denominator = np.sum(
            values.conj() * values * weights,
            axis=(-2, -1),
        )
        if np.any(np.abs(denominator) == 0.0):
            raise ValueError("inversion expectation requires nonzero states")
        result = numerator / denominator
        scale = np.maximum(1.0, np.abs(result.real))
        if np.any(np.abs(result.imag) > 1.0e-12 * scale):
            raise ValueError(
                "inversion expectation has a significant imaginary part"
            )
        real = np.asarray(result.real)
        if real.ndim == 0:
            return float(real)
        return real

    def restrict_full_grid(self, values, m=0) -> np.ndarray:
        """Restrict full product-grid values to the public active shape."""

        m = self.validate_m(m)
        array = np.asarray(values)
        if array.ndim < 2 or array.shape[-2:] != self.grid.shape:
            raise ValueError(
                f"values must have trailing full-grid shape {self.grid.shape}"
            )
        flat = array.reshape((*array.shape[:-2], self.grid.size))
        restricted = flat[..., self.active_indices(m)]
        return restricted.reshape(
            array.shape[:-2] + self.fixed_m_shape(m)
        )

    def embed_active_state(self, state, m=0, *, fill_value=0.0) -> np.ndarray:
        """Embed active nodal values in the full tensor-product grid."""

        m = self.validate_m(m)
        values = self.validate_fixed_m_state(state, m)
        result = np.full(
            values.shape[:-2] + self.grid.shape,
            fill_value,
            dtype=np.result_type(values.dtype, fill_value),
        )
        flat = result.reshape((*values.shape[:-2], self.grid.size))
        flat[..., self.active_indices(m)] = values.reshape(
            (*values.shape[:-2], -1)
        )
        return result

    def archive_metadata(self) -> dict:
        """Return JSON-compatible grid and representation metadata."""

        return {
            "coordinate_system": "prolate_spheroidal",
            "internuclear_distance": self.grid.internuclear_distance,
            "xi_boundaries": list(self.xi_boundaries),
            "xi_points_per_element": list(self.xi_points_per_element),
            "eta_degree": self.eta_degree,
            "m_max": self.m_max,
            "xi_outer_dirichlet": self.xi_outer_dirichlet,
            "full_grid_shape": list(self.grid.shape),
            "metric_representation": "nodal_diagonal_overlap",
            "flattening_order": "C_eta_fastest",
        }


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _point_count_tuple(values, n_elements: int) -> tuple[int, ...]:
    raw = np.asarray(values)
    if raw.ndim == 0:
        return (int(raw),) * n_elements
    return tuple(int(value) for value in raw)
