"""Geometry classification for multicenter spherical representations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CenterGeometry:
    """Affine geometry metadata for a set of distinct Cartesian centers."""

    classification: str
    affine_rank: int
    axis: tuple[float, float, float] | None
    frame: tuple[tuple[float, float, float], ...]
    tolerance: float


def _right_handed_frame(axis: np.ndarray) -> np.ndarray:
    """Return a deterministic frame whose third column is ``axis``."""

    reference = np.eye(3)[np.argmin(np.abs(axis))]
    x_axis = np.cross(reference, axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    return np.column_stack((x_axis, y_axis, axis))


def classify_centers(centers, *, tolerance=None) -> CenterGeometry:
    """Classify distinct centers as single-center, collinear, or noncollinear.

    The default collinearity tolerance scales with the largest singular value
    of the centered coordinate matrix.  Supplying ``tolerance`` selects an
    explicit absolute tolerance instead.
    """

    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (n_centers, 3)")
    if centers.shape[0] < 1:
        raise ValueError("at least one center is required")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite values")

    if centers.shape[0] > 1:
        separations = np.linalg.norm(
            centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1
        )
        off_diagonal = ~np.eye(centers.shape[0], dtype=bool)
        if np.any(separations[off_diagonal] == 0.0):
            raise ValueError("centers must be distinct")

    centered = centers - np.mean(centers, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    scale = float(singular_values[0]) if singular_values.size else 0.0
    if tolerance is None:
        absolute_tolerance = float(
            256.0
            * np.finfo(float).eps
            * max(centers.shape)
            * scale
        )
    else:
        absolute_tolerance = float(tolerance)
        if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
    affine_rank = int(np.count_nonzero(singular_values > absolute_tolerance))

    if centers.shape[0] == 1:
        return CenterGeometry(
            classification="single_center",
            affine_rank=0,
            axis=None,
            frame=tuple(tuple(float(value) for value in row) for row in np.eye(3)),
            tolerance=absolute_tolerance,
        )

    if affine_rank <= 1:
        axis = np.array(right_vectors[0], copy=True)
        largest_component = int(np.argmax(np.abs(axis)))
        if axis[largest_component] < 0.0:
            axis *= -1.0
        frame = _right_handed_frame(axis)
        return CenterGeometry(
            classification="collinear",
            affine_rank=affine_rank,
            axis=tuple(float(value) for value in axis),
            frame=tuple(
                tuple(float(value) for value in row) for row in frame
            ),
            tolerance=absolute_tolerance,
        )

    return CenterGeometry(
        classification="noncollinear",
        affine_rank=affine_rank,
        axis=None,
        frame=tuple(tuple(float(value) for value in row) for row in np.eye(3)),
        tolerance=absolute_tolerance,
    )
