"""Application of radial masks and CAPs to two-radial states."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .grids import RadialGrid
from .system import TwoRadialSystem


class TwoRadialAbsorbingBoundaries:
    """Prepared mask and complex-potential values for two radial coordinates."""

    def __init__(
        self,
        representation: TwoRadialSystem,
        *,
        mask1=None,
        mask2=None,
        cap1=None,
        cap2=None,
    ) -> None:
        if not isinstance(representation, TwoRadialSystem):
            raise TypeError("representation must be a TwoRadialSystem")
        _validate_exclusive(mask1, cap1, "coordinate 1")
        _validate_exclusive(mask2, cap2, "coordinate 2")
        self.representation = representation
        self.mask1 = _mask_values(mask1, representation.grid1, "mask1")
        self.mask2 = _mask_values(mask2, representation.grid2, "mask2")
        self.cap1 = _cap_values(cap1, representation.grid1, "cap1")
        self.cap2 = _cap_values(cap2, representation.grid2, "cap2")
        self.metadata = {
            "mask1": _profile_metadata(mask1, self.mask1),
            "mask2": _profile_metadata(mask2, self.mask2),
            "cap1": _profile_metadata(cap1, self.cap1),
            "cap2": _profile_metadata(cap2, self.cap2),
        }

    @property
    def has_mask(self) -> bool:
        return self.mask1 is not None or self.mask2 is not None

    @property
    def has_cap(self) -> bool:
        return self.cap1 is not None or self.cap2 is not None

    def apply_mask(self, state: np.ndarray, time: float) -> np.ndarray:
        """Apply the configured masks once after an accepted time step."""

        del time
        values = _state(state, self.representation.psi_shape)
        if not self.has_mask:
            return values.copy()
        radial = np.ones(
            (self.representation.grid1.n_points, self.representation.grid2.n_points)
        )
        if self.mask1 is not None:
            radial *= self.mask1[:, None]
        if self.mask2 is not None:
            radial *= self.mask2[None, :]
        return values * _radial_broadcast(radial, values.ndim)

    def cap_action(self, state: np.ndarray) -> np.ndarray:
        """Return the diagonal complex-absorbing-potential action."""

        values = _state(state, self.representation.psi_shape)
        if not self.has_cap:
            return np.zeros_like(values)
        radial = np.zeros(
            (self.representation.grid1.n_points, self.representation.grid2.n_points),
            dtype=complex,
        )
        if self.cap1 is not None:
            radial += self.cap1[:, None]
        if self.cap2 is not None:
            radial += self.cap2[None, :]
        return values * _radial_broadcast(radial, values.ndim)


def _mask_values(profile, grid: RadialGrid, name: str):
    if profile is None:
        return None
    values = _evaluate(profile, grid, name)
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must return real values")
    values = np.asarray(values, dtype=float)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return values


def _cap_values(profile, grid: RadialGrid, name: str):
    if profile is None:
        return None
    values = np.asarray(_evaluate(profile, grid, name), dtype=complex)
    if np.any(values.imag > 0.0):
        raise ValueError(f"the imaginary part of {name} must be non-positive")
    return values


def _evaluate(profile, grid: RadialGrid, name: str) -> np.ndarray:
    if callable(profile):
        values = profile(grid.r.copy(), _outer_radius(grid))
    else:
        values = profile
    values = np.asarray(values)
    if values.shape != grid.r.shape:
        raise ValueError(
            f"{name} returned shape {values.shape}; expected {grid.r.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _outer_radius(grid: RadialGrid) -> float:
    specification = grid.specification
    if hasattr(specification, "nodes"):
        return float(specification.nodes[-1])
    return float(specification.r_max)


def _validate_exclusive(mask, cap, coordinate: str) -> None:
    if mask is not None and cap is not None:
        raise ValueError(f"mask and cap cannot both be set for {coordinate}")


def _state(state, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(state)
    if values.shape != shape:
        raise ValueError(f"state has shape {values.shape}; expected {shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("state must contain numeric values")
    return values


def _radial_broadcast(values: np.ndarray, state_ndim: int) -> np.ndarray:
    return values.reshape((1,) * (state_ndim - 2) + values.shape)


def _profile_metadata(profile, values) -> dict[str, Any] | None:
    if profile is None:
        return None
    parameters = {}
    if hasattr(profile, "__dict__"):
        for name, value in vars(profile).items():
            if name.startswith("_"):
                continue
            scalar = np.asarray(value)
            parameters[name] = scalar.item() if scalar.ndim == 0 else repr(value)
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return {
        "module": type(profile).__module__,
        "type": type(profile).__name__,
        "parameters": parameters,
        "values_sha256": digest.hexdigest(),
    }
