"""Self-describing archives for coupled-angular two-radial states."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .coupled_angular import CoupledAngularBasisSpec
from .system import CoupledTwoRadialSystem
from .state_archive import (
    _array_id,
    _canonical_json,
    _grid_metadata,
    _grid_specification,
    _package_version,
    _representation_id,
    _validate_radial_grid,
)


_FORMAT_NAME = "grid-tdse-coupled-two-radial-state"
_FORMAT_VERSION = 1
_REQUIRED_ARRAYS = {
    "state",
    "radial_points1",
    "radial_weights1",
    "radial_points2",
    "radial_weights2",
    "angular_states",
    "metadata_json",
}


@dataclass(frozen=True)
class CoupledTwoRadialState:
    """A coupled state and the metadata required to interpret it safely."""

    coefficients: np.ndarray
    radial_points1: np.ndarray
    radial_weights1: np.ndarray
    radial_points2: np.ndarray
    radial_weights2: np.ndarray
    angular_states: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_archive_contents(self)

    @property
    def physical_state(self) -> dict[str, Any]:
        return self.metadata["physical_state"]

    @property
    def calculation(self) -> dict[str, Any]:
        return self.metadata["calculation"]

    @property
    def representation_id(self) -> str:
        return self.metadata["representation_id"]

    @property
    def state_data_id(self) -> str:
        return self.metadata["state_data_id"]

    @property
    def angular_basis(self) -> CoupledAngularBasisSpec:
        return CoupledAngularBasisSpec.from_metadata(
            self.metadata["representation"]["angular_basis"]
        )

    @property
    def solver_converged(self) -> bool:
        value = self.calculation.get("converged")
        if not isinstance(value, bool):
            raise ValueError(
                "coupled ground-state archive does not record a boolean "
                "solver-convergence status"
            )
        return value

    def require_solver_converged(
        self,
        *,
        allow_unconverged: bool = False,
    ) -> None:
        """Reject an unconverged eigensolver result unless explicitly allowed."""

        if not isinstance(allow_unconverged, (bool, np.bool_)):
            raise TypeError("allow_unconverged must be boolean")
        if not self.solver_converged and not allow_unconverged:
            raise ValueError(
                "coupled initial-state archive did not satisfy its Davidson "
                "convergence criterion; pass allow_unconverged=True only as "
                "an explicit unsafe override"
            )

    def validate_representation(
        self,
        representation: CoupledTwoRadialSystem,
    ) -> None:
        """Require exact radial and coupled-angular representation identity."""

        if not isinstance(representation, CoupledTwoRadialSystem):
            raise TypeError("representation must be a CoupledTwoRadialSystem")
        stored = self.metadata["representation"]
        _validate_radial_grid(
            "coordinate 1",
            self.radial_points1,
            self.radial_weights1,
            stored["coordinate1"]["grid"],
            representation.grid1,
        )
        _validate_radial_grid(
            "coordinate 2",
            self.radial_points2,
            self.radial_weights2,
            stored["coordinate2"]["grid"],
            representation.grid2,
        )
        if self.angular_basis != representation.angular_basis:
            raise ValueError("coupled angular basis does not match")
        expected = np.asarray(representation.angular_basis.states, dtype=np.int64)
        if not np.array_equal(self.angular_states, expected):
            raise ValueError("coupled angular state table does not match")
        expected_multipoles = {
            "lambda_max": representation.multipoles.lambda_max,
            "mu_max": representation.multipoles.mu_max,
            "lambda_parity": representation.multipoles.lambda_parity,
        }
        if stored["coulomb_multipoles"] != expected_multipoles:
            raise ValueError("Coulomb multipole specification does not match")

    def state_for(self, representation: CoupledTwoRadialSystem) -> np.ndarray:
        """Return an independent state after exact representation validation."""

        self.validate_representation(representation)
        return self.coefficients.copy()


def save_coupled_two_radial_state(
    path: str | os.PathLike[str],
    coefficients: np.ndarray,
    representation: CoupledTwoRadialSystem,
    *,
    physical_state: Mapping[str, Any],
    calculation: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically save a coupled state without using pickle data."""

    if not isinstance(representation, CoupledTwoRadialSystem):
        raise TypeError("representation must be a CoupledTwoRadialSystem")
    if not isinstance(physical_state, Mapping):
        raise TypeError("physical_state must be a mapping")
    if calculation is not None and not isinstance(calculation, Mapping):
        raise TypeError("calculation must be a mapping or None")
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError("overwrite must be boolean")

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"state archive already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"state archive directory does not exist: {destination.parent}"
        )

    state = representation.validate_state(coefficients)
    if not np.isfinite(state).all():
        raise ValueError("state must contain only finite values")
    state = np.ascontiguousarray(state)
    arrays = _representation_arrays(representation)
    representation_metadata = _representation_metadata(representation)
    metadata = {
        "format": _FORMAT_NAME,
        "format_version": _FORMAT_VERSION,
        "units": "atomic",
        "created_with": {
            "grid_tdse_version": _package_version(),
            "numpy_version": np.__version__,
        },
        "physical_state": dict(physical_state),
        "representation": representation_metadata,
        "calculation": {} if calculation is None else dict(calculation),
    }
    metadata["representation_id"] = _representation_id(
        representation_metadata,
        arrays,
    )
    metadata["state_data_id"] = _array_id(state)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez(
                stream,
                state=state,
                metadata_json=np.asarray(_canonical_json(metadata)),
                **arrays,
            )
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            os.link(temporary_path, destination)
            temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_coupled_two_radial_state(
    path: str | os.PathLike[str],
) -> CoupledTwoRadialState:
    """Load and checksum-validate a coupled state archive."""

    with np.load(Path(path), allow_pickle=False) as archive:
        missing = _REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(
                "coupled state archive is missing required arrays: "
                + ", ".join(sorted(missing))
            )
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                "coupled state archive contains invalid JSON metadata"
            ) from exc
        state = CoupledTwoRadialState(
            coefficients=archive["state"].copy(),
            radial_points1=archive["radial_points1"].copy(),
            radial_weights1=archive["radial_weights1"].copy(),
            radial_points2=archive["radial_points2"].copy(),
            radial_weights2=archive["radial_weights2"].copy(),
            angular_states=archive["angular_states"].copy(),
            metadata=metadata,
        )

    arrays = {
        "radial_points1": state.radial_points1,
        "radial_weights1": state.radial_weights1,
        "radial_points2": state.radial_points2,
        "radial_weights2": state.radial_weights2,
        "angular_states": state.angular_states,
    }
    expected_representation_id = _representation_id(
        state.metadata["representation"],
        arrays,
    )
    if state.representation_id != expected_representation_id:
        raise ValueError("coupled archive representation checksum does not match")
    if state.state_data_id != _array_id(state.coefficients):
        raise ValueError("coupled archive coefficient checksum does not match")
    return state


def _representation_arrays(
    representation: CoupledTwoRadialSystem,
) -> dict[str, np.ndarray]:
    return {
        "radial_points1": np.asarray(representation.grid1.r).copy(),
        "radial_weights1": np.asarray(representation.grid1.weights).copy(),
        "radial_points2": np.asarray(representation.grid2.r).copy(),
        "radial_weights2": np.asarray(representation.grid2.weights).copy(),
        "angular_states": np.asarray(
            representation.angular_basis.states,
            dtype=np.int64,
        ),
    }


def _representation_metadata(
    representation: CoupledTwoRadialSystem,
) -> dict[str, Any]:
    return {
        "wf_representation": "coupled",
        "angular_basis_name": "bipolar-spherical-harmonics",
        "angular_basis": representation.angular_basis.to_metadata(),
        "coordinate1": {"grid": _grid_metadata(representation.grid1)},
        "coordinate2": {"grid": _grid_metadata(representation.grid2)},
        "coulomb_multipoles": {
            "lambda_max": representation.multipoles.lambda_max,
            "mu_max": representation.multipoles.mu_max,
            "lambda_parity": representation.multipoles.lambda_parity,
        },
        "radial_coulomb_representation": (
            representation.radial_multipoles.representation
        ),
        "conventions": {
            "spherical_harmonics": "Condon-Shortley",
            "angular_coupling": "Clebsch-Gordan",
            "channel_order": ["l_1", "l_2", "L", "M"],
        },
    }


def _validate_archive_contents(state: CoupledTwoRadialState) -> None:
    metadata = state.metadata
    if not isinstance(metadata, dict):
        raise TypeError("coupled archive metadata must be a dictionary")
    if metadata.get("format") != _FORMAT_NAME:
        raise ValueError("file is not a coupled two-radial state archive")
    if metadata.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported coupled state archive format version")
    for name in (
        "physical_state",
        "representation",
        "calculation",
        "representation_id",
        "state_data_id",
    ):
        if name not in metadata:
            raise ValueError(f"coupled archive metadata is missing {name!r}")

    representation = metadata["representation"]
    if representation.get("wf_representation") != "coupled":
        raise ValueError("archive representation is not coupled")
    basis = CoupledAngularBasisSpec.from_metadata(
        representation["angular_basis"]
    )
    coefficients = np.asarray(state.coefficients)
    expected_shape = (
        basis.n_channels,
        state.radial_points1.size,
        state.radial_points2.size,
    )
    if coefficients.shape != expected_shape:
        raise ValueError(
            f"stored coupled state has shape {coefficients.shape}; expected "
            f"{expected_shape}"
        )
    if not np.issubdtype(coefficients.dtype, np.number):
        raise TypeError("stored coupled state must contain numeric values")
    if not np.isfinite(coefficients).all():
        raise ValueError("stored coupled state contains non-finite values")
    angular_states = np.asarray(state.angular_states)
    expected_states = np.asarray(basis.states, dtype=np.int64)
    if angular_states.shape != expected_states.shape or not np.array_equal(
        angular_states,
        expected_states,
    ):
        raise ValueError("stored coupled angular states are inconsistent")
    for coordinate in (1, 2):
        points = np.asarray(getattr(state, f"radial_points{coordinate}"))
        weights = np.asarray(getattr(state, f"radial_weights{coordinate}"))
        if points.ndim != 1 or weights.shape != points.shape:
            raise ValueError(
                f"stored coordinate {coordinate} radial arrays are inconsistent"
            )
        if not np.isfinite(points).all() or not np.isfinite(weights).all():
            raise ValueError(
                f"stored coordinate {coordinate} radial arrays are non-finite"
            )
        _grid_specification(representation[f"coordinate{coordinate}"]["grid"])
