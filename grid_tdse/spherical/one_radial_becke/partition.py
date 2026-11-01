"""Geometry-only multicenter Becke partitions and analytic gradients."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class BeckePartitionEvaluation:
    """Values produced by evaluating a :class:`BeckePartition`.

    A scalar Cartesian point produces center-indexed arrays with shapes
    ``(n_centers, ...)``. An input with shape ``(..., 3)`` preserves its
    leading dimensions. Pair arrays use the final two center axes ``(A, B)``;
    their diagonal entries are sentinels with ``mu_AA = 0`` and ``s_AA = 1``.
    """

    center_displacements: np.ndarray
    center_distances: np.ndarray
    center_unit_vectors: np.ndarray
    pair_variables: np.ndarray
    pair_variable_gradients: np.ndarray
    pair_switches: np.ndarray
    pair_switch_gradients: np.ndarray
    unnormalized_weights: np.ndarray
    unnormalized_weight_gradients: np.ndarray
    weights: np.ndarray
    weight_gradients: np.ndarray


class BeckePartition:
    r"""Standard geometry-only Becke partition for distinct centers.

    For each ordered pair of centers, the elliptic coordinate and pair
    switch favoring center ``A`` are

    .. math::

        \mu_{AB} = \frac{r_A-r_B}{R_{AB}}, \qquad
        s_{AB} = \frac{1-f(\mu_{AB})}{2}.

    The smoothing map ``f(x) = (3*x - x**3)/2`` is applied repeatedly.
    Atomic cell functions are products of the pair switches and are finally
    normalized to a partition of unity. Atomic-size corrections are not
    included in this initial implementation.

    Parameters
    ----------
    centers : array_like, shape (n_centers, 3)
        Finite Cartesian center positions. At least one center is required,
        and coincident centers are rejected.
    switch_iterations : int, optional
        Positive number of Becke smoothing iterations. The standard value is
        three.
    """

    def __init__(self, centers, switch_iterations: int = 3) -> None:
        center_array = np.asarray(centers, dtype=float)
        if center_array.ndim != 2 or center_array.shape[1] != 3:
            raise ValueError("centers must have shape (n_centers, 3)")
        if center_array.shape[0] < 1:
            raise ValueError("at least one center is required")
        if not np.all(np.isfinite(center_array)):
            raise ValueError("centers must contain only finite values")
        if (
            isinstance(switch_iterations, (bool, np.bool_))
            or not isinstance(switch_iterations, Integral)
        ):
            raise TypeError("switch_iterations must be an integer")
        switch_iterations = int(switch_iterations)
        if switch_iterations < 1:
            raise ValueError("switch_iterations must be positive")

        pair_vectors = center_array[:, np.newaxis, :] - center_array[np.newaxis, :, :]
        pair_distances = np.linalg.norm(pair_vectors, axis=-1)
        off_diagonal = ~np.eye(center_array.shape[0], dtype=bool)
        if np.any(pair_distances[off_diagonal] == 0.0):
            raise ValueError("centers must be distinct")

        self._centers = np.array(center_array, copy=True)
        self._pair_distances = pair_distances
        self._switch_iterations = switch_iterations
        self._centers.setflags(write=False)
        self._pair_distances.setflags(write=False)

    @property
    def centers(self) -> np.ndarray:
        """Read-only Cartesian center positions."""

        return self._centers

    @property
    def n_centers(self) -> int:
        """Number of centers in the partition."""

        return self._centers.shape[0]

    @property
    def pair_distances(self) -> np.ndarray:
        """Read-only matrix of intercenter distances."""

        return self._pair_distances

    @property
    def switch_iterations(self) -> int:
        """Number of repeated Becke smoothing operations."""

        return self._switch_iterations

    def _validated_points(self, points) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.ndim < 1 or values.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("points must contain only finite values")
        return values

    def _smooth(self, pair_variables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.array(pair_variables, copy=True)
        derivatives = np.ones_like(values)
        for _ in range(self._switch_iterations):
            derivatives *= 1.5 * (1.0 - values**2)
            values = 1.5 * values - 0.5 * values**3
        return values, derivatives

    def evaluate(self, points) -> BeckePartitionEvaluation:
        """Evaluate partition data and analytic Cartesian gradients.

        Parameters
        ----------
        points : array_like, shape (..., 3)
            One point or an arbitrary batch of Cartesian points.

        Returns
        -------
        BeckePartitionEvaluation
            Center distances, pair data, normalized weights, and their
            gradients. Gradient arrays have a final Cartesian axis.
        """

        points = self._validated_points(points)
        displacements = points[..., np.newaxis, :] - self._centers
        distances = np.linalg.norm(displacements, axis=-1)
        unit_vectors = np.zeros_like(displacements)
        np.divide(
            displacements,
            distances[..., np.newaxis],
            out=unit_vectors,
            where=distances[..., np.newaxis] > 0.0,
        )

        safe_pair_distances = np.array(self._pair_distances, copy=True)
        diagonal = np.diag_indices(self.n_centers)
        safe_pair_distances[diagonal] = 1.0

        pair_variables = (
            distances[..., :, np.newaxis] - distances[..., np.newaxis, :]
        ) / safe_pair_distances
        pair_variables = np.clip(pair_variables, -1.0, 1.0)
        pair_variables[..., diagonal[0], diagonal[1]] = 0.0

        pair_variable_gradients = (
            unit_vectors[..., :, np.newaxis, :]
            - unit_vectors[..., np.newaxis, :, :]
        ) / safe_pair_distances[..., np.newaxis]
        pair_variable_gradients[..., diagonal[0], diagonal[1], :] = 0.0

        smoothed, smooth_derivatives = self._smooth(pair_variables)
        pair_switches = 0.5 * (1.0 - smoothed)
        pair_switch_gradients = (
            -0.5
            * smooth_derivatives[..., np.newaxis]
            * pair_variable_gradients
        )
        pair_switches[..., diagonal[0], diagonal[1]] = 1.0
        pair_switch_gradients[..., diagonal[0], diagonal[1], :] = 0.0

        unnormalized_weights = np.prod(pair_switches, axis=-1)
        unnormalized_weight_gradients = np.zeros(
            unnormalized_weights.shape + (3,), dtype=float
        )
        center_indices = np.arange(self.n_centers)
        for center_a in range(self.n_centers):
            for center_b in range(self.n_centers):
                if center_a == center_b:
                    continue
                other_pairs = (center_indices != center_a) & (
                    center_indices != center_b
                )
                other_product = np.prod(
                    pair_switches[..., center_a, other_pairs], axis=-1
                )
                unnormalized_weight_gradients[..., center_a, :] += (
                    pair_switch_gradients[..., center_a, center_b, :]
                    * other_product[..., np.newaxis]
                )

        normalization = np.sum(unnormalized_weights, axis=-1)
        if np.any(normalization <= 0.0):
            raise FloatingPointError("Becke weight normalization vanished")
        normalization_gradient = np.sum(
            unnormalized_weight_gradients, axis=-2
        )
        weights = unnormalized_weights / normalization[..., np.newaxis]
        weight_gradients = (
            unnormalized_weight_gradients * normalization[..., np.newaxis, np.newaxis]
            - unnormalized_weights[..., np.newaxis]
            * normalization_gradient[..., np.newaxis, :]
        ) / normalization[..., np.newaxis, np.newaxis] ** 2

        return BeckePartitionEvaluation(
            center_displacements=displacements,
            center_distances=distances,
            center_unit_vectors=unit_vectors,
            pair_variables=pair_variables,
            pair_variable_gradients=pair_variable_gradients,
            pair_switches=pair_switches,
            pair_switch_gradients=pair_switch_gradients,
            unnormalized_weights=unnormalized_weights,
            unnormalized_weight_gradients=unnormalized_weight_gradients,
            weights=weights,
            weight_gradients=weight_gradients,
        )

    def weights(self, points) -> np.ndarray:
        """Return normalized center weights at Cartesian points."""

        return self.evaluate(points).weights

    def gradients(self, points) -> np.ndarray:
        """Return Cartesian gradients of normalized center weights."""

        return self.evaluate(points).weight_gradients
