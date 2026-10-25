"""Self-describing state archives for two-radial calculations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .angular import AngularBasisSpec, AngularPairBasis, CoulombMultipoleSpec
from .grids import FEMDVRGridSpec, RadialGrid, RadialGridSpec, RadialSincDVRGridSpec
from .system import TwoRadialSystem


_FORMAT_NAME = "grid-tdse-two-radial-state"
_FORMAT_VERSION = 2
_SUPPORTED_FORMAT_VERSIONS = {1, 2}
_REQUIRED_ARRAYS = {
    "state",
    "radial_points1",
    "radial_weights1",
    "angular_states1",
    "radial_points2",
    "radial_weights2",
    "angular_states2",
    "metadata_json",
}


@dataclass(frozen=True)
class TwoRadialState:
    """A state together with the representation data needed to interpret it."""

    coefficients: np.ndarray
    radial_points1: np.ndarray
    radial_weights1: np.ndarray
    angular_states1: np.ndarray
    radial_points2: np.ndarray
    radial_weights2: np.ndarray
    angular_states2: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_archive_contents(self)

    @property
    def physical_state(self) -> dict[str, Any]:
        """Return the stored physical-system and symmetry description."""

        return self.metadata["physical_state"]

    @property
    def calculation(self) -> dict[str, Any]:
        """Return the stored calculation diagnostics."""

        return self.metadata["calculation"]

    @property
    def grid1_specification(self) -> RadialGridSpec:
        return _grid_specification(
            self.metadata["representation"]["coordinate1"]["grid"]
        )

    @property
    def grid2_specification(self) -> RadialGridSpec:
        return _grid_specification(
            self.metadata["representation"]["coordinate2"]["grid"]
        )

    @property
    def basis1(self) -> AngularBasisSpec:
        return _angular_basis(
            self.metadata["representation"]["coordinate1"]["angular_basis"]
        )

    @property
    def basis2(self) -> AngularBasisSpec:
        return _angular_basis(
            self.metadata["representation"]["coordinate2"]["angular_basis"]
        )

    @property
    def multipoles(self) -> CoulombMultipoleSpec:
        values = self.metadata["representation"]["coulomb_multipoles"]
        return CoulombMultipoleSpec(
            lambda_max=values["lambda_max"],
            mu_max=values["mu_max"],
            lambda_parity=values["lambda_parity"],
        )

    @property
    def M_tot(self) -> int | None:
        """Stored total magnetic sector, or ``None`` for full storage."""

        return self.metadata["representation"].get("M_tot")

    @property
    def representation_id(self) -> str:
        return self.metadata["representation_id"]

    @property
    def state_data_id(self) -> str:
        return self.metadata["state_data_id"]

    def validate_representation(
        self,
        representation: TwoRadialSystem,
        *,
        allow_angular_superset: bool = False,
    ) -> None:
        """Validate radial identity and angular compatibility.

        Radial grids must agree exactly. When ``allow_angular_superset`` is
        true, every stored angular channel must occur in the target, but the
        target may contain additional channels.
        """

        _validate_radial_grid(
            "coordinate 1",
            self.radial_points1,
            self.radial_weights1,
            self.metadata["representation"]["coordinate1"]["grid"],
            representation.grid1,
        )
        _validate_radial_grid(
            "coordinate 2",
            self.radial_points2,
            self.radial_weights2,
            self.metadata["representation"]["coordinate2"]["grid"],
            representation.grid2,
        )
        _validate_angular_basis(
            "coordinate 1",
            self.angular_states1,
            representation.basis1.states,
            allow_superset=allow_angular_superset,
        )
        if not allow_angular_superset and self.M_tot != representation.M_tot:
            raise ValueError("M_tot storage sector does not match")
        _validate_angular_basis(
            "coordinate 2",
            self.angular_states2,
            representation.basis2.states,
            allow_superset=allow_angular_superset,
        )

    def state_for(self, representation: TwoRadialSystem) -> np.ndarray:
        """Return a copy after exact representation validation."""

        self.validate_representation(representation)
        return self.coefficients.copy()

    def embed_into(self, representation: TwoRadialSystem) -> np.ndarray:
        """Embed the state into an angular superset on the same radial grids."""

        self.validate_representation(representation, allow_angular_superset=True)
        states1 = [tuple(int(value) for value in row) for row in self.angular_states1]
        states2 = [tuple(int(value) for value in row) for row in self.angular_states2]
        map1 = [representation.basis1.index(*state) for state in states1]
        map2 = [representation.basis2.index(*state) for state in states2]
        result = np.zeros(representation.psi_shape, dtype=self.coefficients.dtype)
        source_pairs = AngularPairBasis(self.basis1, self.basis2, self.M_tot)
        for source_pair, (source1, source2) in enumerate(source_pairs.pairs):
            target1 = map1[source1]
            target2 = map2[source2]
            target_pair = representation.angular_pairs.get(target1, target2)
            if target_pair is None:
                raise ValueError(
                    "target representation does not retain all stored angular pairs"
                )
            block = (
                self.coefficients[source_pair]
                if source_pairs.is_compact
                else self.coefficients[source1, source2]
            )
            representation.set_angular_block(result, target1, target2, block)
        return result


def save_two_radial_state(
    path: str | os.PathLike[str],
    coefficients: np.ndarray,
    representation: TwoRadialSystem,
    *,
    physical_state: Mapping[str, Any],
    calculation: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Save a self-describing two-radial state at the exact user path."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"state archive already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"state archive directory does not exist: {destination.parent}"
        )

    state = np.asarray(coefficients)
    if state.shape != representation.psi_shape:
        raise ValueError(
            f"state has shape {state.shape}; expected {representation.psi_shape}"
        )
    if not np.issubdtype(state.dtype, np.number):
        raise TypeError("state must contain numeric values")
    if not np.isfinite(state).all():
        raise ValueError("state must contain only finite values")
    state = np.ascontiguousarray(state)

    angular_states1 = np.asarray(representation.basis1.states, dtype=np.int64)
    angular_states2 = np.asarray(representation.basis2.states, dtype=np.int64)
    arrays = {
        "radial_points1": np.asarray(representation.grid1.r).copy(),
        "radial_weights1": np.asarray(representation.grid1.weights).copy(),
        "angular_states1": angular_states1,
        "radial_points2": np.asarray(representation.grid2.r).copy(),
        "radial_weights2": np.asarray(representation.grid2.weights).copy(),
        "angular_states2": angular_states2,
    }
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
        representation_metadata, arrays
    )
    metadata["state_data_id"] = _array_id(state)
    metadata_json = _canonical_json(metadata)

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
                metadata_json=np.asarray(metadata_json),
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


def load_two_radial_state(path: str | os.PathLike[str]) -> TwoRadialState:
    """Load and validate a state archive without enabling pickle support."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        missing = _REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"state archive is missing required arrays: {names}")
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("state archive contains invalid JSON metadata") from exc
        state = TwoRadialState(
            coefficients=archive["state"].copy(),
            radial_points1=archive["radial_points1"].copy(),
            radial_weights1=archive["radial_weights1"].copy(),
            angular_states1=archive["angular_states1"].copy(),
            radial_points2=archive["radial_points2"].copy(),
            radial_weights2=archive["radial_weights2"].copy(),
            angular_states2=archive["angular_states2"].copy(),
            metadata=metadata,
        )

    arrays = {
        "radial_points1": state.radial_points1,
        "radial_weights1": state.radial_weights1,
        "angular_states1": state.angular_states1,
        "radial_points2": state.radial_points2,
        "radial_weights2": state.radial_weights2,
        "angular_states2": state.angular_states2,
    }
    expected_representation_id = _representation_id(
        state.metadata["representation"], arrays
    )
    if state.representation_id != expected_representation_id:
        raise ValueError("state archive representation checksum does not match")
    if state.state_data_id != _array_id(state.coefficients):
        raise ValueError("state archive coefficient checksum does not match")
    return state


def two_radial_compatibility_metadata(
    representation: TwoRadialSystem,
    *,
    physical_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return restart metadata for a two-radial real-time checkpoint.

    The representation identifier includes the realized radial points,
    quadrature weights, and ordered angular channel tables. The full grid and
    angular specifications are retained so the representation can be
    reconstructed without storing Python objects.
    """

    if not isinstance(representation, TwoRadialSystem):
        raise TypeError("representation must be a TwoRadialSystem")
    if not isinstance(physical_state, Mapping):
        raise TypeError("physical_state must be a mapping")
    arrays = {
        "radial_points1": np.asarray(representation.grid1.r),
        "radial_weights1": np.asarray(representation.grid1.weights),
        "angular_states1": np.asarray(
            representation.basis1.states,
            dtype=np.int64,
        ),
        "radial_points2": np.asarray(representation.grid2.r),
        "radial_weights2": np.asarray(representation.grid2.weights),
        "angular_states2": np.asarray(
            representation.basis2.states,
            dtype=np.int64,
        ),
    }
    representation_metadata = _representation_metadata(representation)
    metadata = {
        "physical_state": dict(physical_state),
        "representation": representation_metadata,
        "representation_id": _representation_id(
            representation_metadata,
            arrays,
        ),
        "state_shape": list(representation.psi_shape),
    }
    return json.loads(_canonical_json(metadata))


def _representation_metadata(representation: TwoRadialSystem) -> dict[str, Any]:
    metadata = {
        "coordinate1": {
            "grid": _grid_metadata(representation.grid1),
            "angular_basis": _basis_metadata(representation.basis1),
        },
        "coordinate2": {
            "grid": _grid_metadata(representation.grid2),
            "angular_basis": _basis_metadata(representation.basis2),
        },
        "coulomb_multipoles": {
            "lambda_max": representation.multipoles.lambda_max,
            "mu_max": representation.multipoles.mu_max,
            "lambda_parity": representation.multipoles.lambda_parity,
        },
    }
    if representation.M_tot is not None:
        metadata["M_tot"] = representation.M_tot
    return metadata


def _grid_metadata(grid: RadialGrid) -> dict[str, Any]:
    specification = grid.specification
    if isinstance(specification, FEMDVRGridSpec):
        return {
            "family": "femdvr",
            "nodes": list(specification.nodes),
            "n_points": list(specification.n_points),
        }
    if isinstance(specification, RadialSincDVRGridSpec):
        return {
            "family": "sinc",
            "r_max": specification.r_max,
            "n_points": specification.n_points,
        }
    raise TypeError(f"unsupported radial grid specification: {type(specification)!r}")


def _basis_metadata(basis: AngularBasisSpec) -> dict[str, Any]:
    return {
        "l_max": basis.l_max,
        "m_max": basis.m_max,
        "coordinate_parity": basis.coordinate_parity,
        "states": [list(state) for state in basis.states],
    }


def _grid_specification(metadata: Mapping[str, Any]) -> RadialGridSpec:
    family = metadata.get("family")
    if family == "femdvr":
        return FEMDVRGridSpec(metadata["nodes"], metadata["n_points"])
    if family == "sinc":
        return RadialSincDVRGridSpec(metadata["r_max"], metadata["n_points"])
    raise ValueError(f"unsupported radial grid family in state archive: {family!r}")


def _angular_basis(metadata: Mapping[str, Any]) -> AngularBasisSpec:
    basis = AngularBasisSpec(
        l_max=metadata["l_max"],
        m_max=metadata["m_max"],
        coordinate_parity=metadata["coordinate_parity"],
    )
    stored_states = tuple(tuple(state) for state in metadata["states"])
    if basis.states != stored_states:
        raise ValueError("stored angular state table is inconsistent with its basis")
    return basis


def _validate_archive_contents(state: TwoRadialState) -> None:
    metadata = state.metadata
    if not isinstance(metadata, dict):
        raise TypeError("state archive metadata must be a dictionary")
    if metadata.get("format") != _FORMAT_NAME:
        raise ValueError("file is not a grid-tdse two-radial state archive")
    if metadata.get("format_version") not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            "unsupported two-radial state archive format version: "
            f"{metadata.get('format_version')!r}"
        )
    for name in (
        "physical_state",
        "representation",
        "calculation",
        "representation_id",
        "state_data_id",
    ):
        if name not in metadata:
            raise ValueError(f"state archive metadata is missing {name!r}")

    coefficients = np.asarray(state.coefficients)
    representation = metadata["representation"]
    M_tot = representation.get("M_tot")
    expected_ndim = 3 if M_tot is not None else 4
    if coefficients.ndim != expected_ndim:
        raise ValueError(
            f"stored state must be a {expected_ndim}-dimensional array for "
            f"M_tot={M_tot!r}"
        )
    if not np.issubdtype(coefficients.dtype, np.number):
        raise TypeError("stored state must contain numeric values")
    if not np.isfinite(coefficients).all():
        raise ValueError("stored state contains non-finite values")

    for coordinate in (1, 2):
        points = np.asarray(getattr(state, f"radial_points{coordinate}"))
        weights = np.asarray(getattr(state, f"radial_weights{coordinate}"))
        angular = np.asarray(getattr(state, f"angular_states{coordinate}"))
        if points.ndim != 1 or weights.shape != points.shape:
            raise ValueError(
                f"stored coordinate {coordinate} radial arrays are inconsistent"
            )
        if not np.isfinite(points).all() or not np.isfinite(weights).all():
            raise ValueError(
                f"stored coordinate {coordinate} radial arrays are non-finite"
            )
        if angular.ndim != 2 or angular.shape[1] != 2:
            raise ValueError(
                f"stored coordinate {coordinate} angular states must have shape (n, 2)"
            )
        if len({tuple(row) for row in angular.tolist()}) != angular.shape[0]:
            raise ValueError(
                f"stored coordinate {coordinate} angular states contain duplicates"
            )
        expected_radial = (
            coefficients.shape[coordinate]
            if M_tot is not None
            else coefficients.shape[coordinate + 1]
        )
        if points.size != expected_radial:
            raise ValueError(
                f"stored coordinate {coordinate} arrays do not match state shape"
            )

    for coordinate, angular in (
        ("coordinate1", state.angular_states1),
        ("coordinate2", state.angular_states2),
    ):
        basis = _angular_basis(representation[coordinate]["angular_basis"])
        if not np.array_equal(np.asarray(basis.states), angular):
            raise ValueError(
                f"stored {coordinate} angular array disagrees with metadata"
            )
        _grid_specification(representation[coordinate]["grid"])
    pair_basis = AngularPairBasis(
        _angular_basis(representation["coordinate1"]["angular_basis"]),
        _angular_basis(representation["coordinate2"]["angular_basis"]),
        M_tot,
    )
    expected_shape = (
        (
            pair_basis.n_pairs,
            state.radial_points1.size,
            state.radial_points2.size,
        )
        if pair_basis.is_compact
        else (
            state.angular_states1.shape[0],
            state.angular_states2.shape[0],
            state.radial_points1.size,
            state.radial_points2.size,
        )
    )
    if coefficients.shape != expected_shape:
        raise ValueError(
            f"stored state has shape {coefficients.shape}; expected {expected_shape}"
        )
    _ = state.multipoles


def _validate_radial_grid(
    name: str,
    stored_points: np.ndarray,
    stored_weights: np.ndarray,
    stored_metadata: Mapping[str, Any],
    target: RadialGrid,
) -> None:
    if _grid_metadata(target) != dict(stored_metadata):
        raise ValueError(f"{name} radial grid specification does not match")
    if not np.array_equal(stored_points, target.r):
        raise ValueError(f"{name} radial points do not match")
    if not np.array_equal(stored_weights, target.weights):
        raise ValueError(f"{name} radial weights do not match")


def _validate_angular_basis(
    name: str,
    stored: np.ndarray,
    target: tuple[tuple[int, int], ...],
    *,
    allow_superset: bool,
) -> None:
    source_states = tuple(tuple(int(value) for value in row) for row in stored)
    if allow_superset:
        missing = [state for state in source_states if state not in target]
        if missing:
            raise ValueError(
                f"{name} target angular basis is missing stored channels: {missing}"
            )
    elif source_states != target:
        raise ValueError(f"{name} angular state table does not match")


def _representation_id(
    representation_metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(_canonical_json(representation_metadata).encode())
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(_array_digest_bytes(np.asarray(arrays[name])))
    return digest.hexdigest()


def _array_id(values: np.ndarray) -> str:
    return hashlib.sha256(_array_digest_bytes(values)).hexdigest()


def _array_digest_bytes(values: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(values)
    header = _canonical_json(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    ).encode()
    return header + contiguous.tobytes(order="C")


def _canonical_json(values: Mapping[str, Any]) -> str:
    return json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_value,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _package_version() -> str:
    try:
        return version("HyQD-grid-tdse")
    except PackageNotFoundError:
        return "unknown"
