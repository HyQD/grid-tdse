"""Persistent, restartable checkpoints for real-time propagation."""

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


_FORMAT_NAME = "grid-tdse-real-time-checkpoint"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RealTimeCheckpointSegment:
    """One uninterrupted propagation stage stored in a checkpoint."""

    initial_time: float
    target_final_time: float
    requested_time_step: float
    time_step: float
    checkpoint_interval: int | None
    initial_norm: float
    times: np.ndarray
    norms: np.ndarray
    linear_iterations: np.ndarray
    linear_residual_norms: np.ndarray
    samples: dict[str, np.ndarray]
    integrator: dict[str, Any]
    dynamics: dict[str, Any]

    def __post_init__(self) -> None:
        initial_time = _finite_real(self.initial_time, "segment initial time")
        target_final_time = _finite_real(
            self.target_final_time,
            "segment target final time",
        )
        requested_time_step = _positive_real(
            self.requested_time_step,
            "segment requested time step",
        )
        time_step = _positive_real(self.time_step, "segment time step")
        checkpoint_interval = _optional_positive_integer(
            self.checkpoint_interval,
            "segment checkpoint interval",
        )
        initial_norm = _nonnegative_real(
            self.initial_norm,
            "segment initial norm",
        )
        if target_final_time <= initial_time:
            raise ValueError(
                "checkpoint segment target final time must exceed its initial time"
            )

        times = _finite_real_array(self.times, "segment times", ndim=1)
        norms = _finite_real_array(self.norms, "segment norms", ndim=1)
        iterations = np.asarray(self.linear_iterations)
        residuals = _finite_real_array(
            self.linear_residual_norms,
            "segment linear residual norms",
            ndim=1,
        )
        if times.size == 0:
            raise ValueError("checkpoint segment must contain an initial time")
        if norms.shape != times.shape:
            raise ValueError("checkpoint segment norms must match its times")
        if iterations.ndim != 1 or iterations.size != times.size - 1:
            raise ValueError(
                "checkpoint segment linear iterations must contain one value "
                "per completed step"
            )
        if not np.issubdtype(iterations.dtype, np.integer):
            raise TypeError("checkpoint segment linear iterations must be integers")
        if np.any(iterations < 0):
            raise ValueError(
                "checkpoint segment linear iterations must be non-negative"
            )
        if residuals.size != times.size - 1:
            raise ValueError(
                "checkpoint segment linear residuals must contain one value per "
                "completed step"
            )
        if np.any(norms < 0.0):
            raise ValueError("checkpoint segment norms must be non-negative")
        if np.any(residuals < 0.0):
            raise ValueError(
                "checkpoint segment linear residual norms must be non-negative"
            )
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("checkpoint segment times must be strictly increasing")
        if times.size > 1 and not np.allclose(
            np.diff(times),
            time_step,
            rtol=1.0e-13,
            atol=1.0e-14,
        ):
            raise ValueError(
                "checkpoint segment times are inconsistent with its time step"
            )
        if not np.isclose(times[0], initial_time, rtol=0.0, atol=1.0e-13):
            raise ValueError("checkpoint segment initial time is inconsistent")
        if times[-1] > target_final_time + 1.0e-13:
            raise ValueError("checkpoint segment extends beyond its target final time")

        checked_samples = {}
        for name, values in self.samples.items():
            if not isinstance(name, str) or not name:
                raise ValueError("checkpoint sample names must be nonempty strings")
            sample = _finite_array(values, f"checkpoint sample {name!r}")
            if sample.ndim == 0 or sample.shape[0] != times.size:
                raise ValueError(
                    f"checkpoint sample {name!r} must have one value per time"
                )
            checked_samples[name] = _read_only_copy(sample)

        object.__setattr__(self, "times", _read_only_copy(times))
        object.__setattr__(self, "norms", _read_only_copy(norms))
        object.__setattr__(self, "linear_iterations", _read_only_copy(iterations))
        object.__setattr__(
            self,
            "linear_residual_norms",
            _read_only_copy(residuals),
        )
        object.__setattr__(self, "samples", checked_samples)
        object.__setattr__(self, "initial_time", initial_time)
        object.__setattr__(self, "target_final_time", target_final_time)
        object.__setattr__(self, "requested_time_step", requested_time_step)
        object.__setattr__(self, "time_step", time_step)
        object.__setattr__(self, "checkpoint_interval", checkpoint_interval)
        object.__setattr__(self, "initial_norm", initial_norm)
        object.__setattr__(
            self,
            "integrator",
            _json_mapping(self.integrator, "integrator metadata"),
        )
        object.__setattr__(
            self,
            "dynamics",
            _json_mapping(self.dynamics, "dynamics metadata"),
        )

    @property
    def n_steps(self) -> int:
        """Number of completed steps in this segment."""

        return self.times.size - 1

    @property
    def current_time(self) -> float:
        """Time of the stored state at the end of this segment."""

        return float(self.times[-1])

    @property
    def complete(self) -> bool:
        """Whether this segment reached its requested target final time."""

        return bool(
            np.isclose(
                self.current_time,
                self.target_final_time,
                rtol=0.0,
                atol=1.0e-13,
            )
        )


@dataclass(frozen=True)
class RealTimeCheckpoint:
    """Stored state, propagation segments, and compatibility metadata."""

    state: np.ndarray
    segments: tuple[RealTimeCheckpointSegment, ...]
    compatibility: dict[str, Any]
    metadata: dict[str, Any]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        state = _finite_array(self.state, "checkpoint state")
        if state.ndim == 0 or state.size == 0:
            raise ValueError("checkpoint state must be a nonempty array")
        if not self.segments:
            raise ValueError("checkpoint must contain at least one propagation segment")
        if any(
            not isinstance(segment, RealTimeCheckpointSegment)
            for segment in self.segments
        ):
            raise TypeError(
                "checkpoint segments must be RealTimeCheckpointSegment objects"
            )
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if not np.isclose(
                previous.current_time,
                current.initial_time,
                rtol=0.0,
                atol=1.0e-13,
            ):
                raise ValueError("checkpoint propagation segments are not contiguous")

        object.__setattr__(self, "state", _read_only_copy(state))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(
            self,
            "compatibility",
            _json_mapping(self.compatibility, "compatibility metadata"),
        )
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, "checkpoint metadata"),
        )
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def current_time(self) -> float:
        """Time associated with the stored state."""

        return self.segments[-1].current_time

    @property
    def current_step(self) -> int:
        """Completed step count in the most recent propagation segment."""

        return self.segments[-1].n_steps

    @property
    def total_steps(self) -> int:
        """Total completed step count across all propagation segments."""

        return sum(segment.n_steps for segment in self.segments)

    @property
    def latest_segment(self) -> RealTimeCheckpointSegment:
        """Most recent propagation segment."""

        return self.segments[-1]

    @property
    def state_data_id(self) -> str:
        """SHA-256 identifier of the stored state data."""

        return _array_id(self.state)

    def state_copy(self) -> np.ndarray:
        """Return a writable copy of the checkpointed state."""

        return self.state.copy()

    def validate_compatibility(self, metadata: Mapping[str, Any]) -> None:
        """Require supplied representation and physical metadata to match."""

        candidate = _json_mapping(metadata, "compatibility metadata")
        if candidate != self.compatibility:
            raise ValueError("checkpoint compatibility metadata does not match")


def save_real_time_checkpoint(
    path: str | os.PathLike[str],
    checkpoint: RealTimeCheckpoint,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save a checkpoint at the exact user-provided path."""

    if not isinstance(checkpoint, RealTimeCheckpoint):
        raise TypeError("checkpoint must be a RealTimeCheckpoint")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"checkpoint already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {destination.parent}"
        )

    arrays: dict[str, np.ndarray] = {"state": np.asarray(checkpoint.state)}
    segment_metadata = []
    for segment_index, segment in enumerate(checkpoint.segments):
        prefix = f"segment_{segment_index:04d}"
        arrays[f"{prefix}_times"] = segment.times
        arrays[f"{prefix}_norms"] = segment.norms
        arrays[f"{prefix}_linear_iterations"] = segment.linear_iterations
        arrays[f"{prefix}_linear_residual_norms"] = (
            segment.linear_residual_norms
        )
        sample_names = list(segment.samples)
        for sample_index, name in enumerate(sample_names):
            arrays[f"{prefix}_sample_{sample_index:04d}"] = segment.samples[name]
        segment_metadata.append(
            {
                "initial_time": segment.initial_time,
                "target_final_time": segment.target_final_time,
                "requested_time_step": segment.requested_time_step,
                "time_step": segment.time_step,
                "checkpoint_interval": segment.checkpoint_interval,
                "initial_norm": segment.initial_norm,
                "integrator": segment.integrator,
                "dynamics": segment.dynamics,
                "sample_names": sample_names,
            }
        )

    trajectory_arrays = {
        name: values for name, values in arrays.items() if name != "state"
    }
    archive_metadata = {
        "format": _FORMAT_NAME,
        "format_version": _FORMAT_VERSION,
        "units": "atomic",
        "created_with": {
            "grid_tdse_version": _package_version(),
            "numpy_version": np.__version__,
        },
        "compatibility": checkpoint.compatibility,
        "metadata": checkpoint.metadata,
        "segments": segment_metadata,
        "state_data_id": _array_id(checkpoint.state),
        "trajectory_data_id": _arrays_id(trajectory_arrays),
    }
    metadata_json = _canonical_json(archive_metadata)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez(stream, metadata_json=np.asarray(metadata_json), **arrays)
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


def load_real_time_checkpoint(
    path: str | os.PathLike[str],
) -> RealTimeCheckpoint:
    """Load and validate a checkpoint without enabling pickle support."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "state" not in archive.files or "metadata_json" not in archive.files:
            raise ValueError("real-time checkpoint is missing required arrays")
        try:
            archive_metadata = json.loads(str(archive["metadata_json"].item()))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                "real-time checkpoint contains invalid JSON metadata"
            ) from exc
        if archive_metadata.get("format") != _FORMAT_NAME:
            raise ValueError("file is not a grid-tdse real-time checkpoint")
        if archive_metadata.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                "unsupported real-time checkpoint format version: "
                f"{archive_metadata.get('format_version')!r}"
            )

        segments = []
        trajectory_arrays = {}
        for segment_index, values in enumerate(archive_metadata.get("segments", [])):
            prefix = f"segment_{segment_index:04d}"
            required = {
                f"{prefix}_times",
                f"{prefix}_norms",
                f"{prefix}_linear_iterations",
                f"{prefix}_linear_residual_norms",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    "real-time checkpoint is missing segment arrays: "
                    + ", ".join(sorted(missing))
                )
            segment_arrays = {
                name: archive[name].copy() for name in sorted(required)
            }
            samples = {}
            for sample_index, name in enumerate(values.get("sample_names", [])):
                key = f"{prefix}_sample_{sample_index:04d}"
                if key not in archive.files:
                    raise ValueError(
                        f"real-time checkpoint is missing sample array {key!r}"
                    )
                samples[name] = archive[key].copy()
                segment_arrays[key] = samples[name]
            trajectory_arrays.update(segment_arrays)
            segments.append(
                RealTimeCheckpointSegment(
                    initial_time=values["initial_time"],
                    target_final_time=values["target_final_time"],
                    requested_time_step=values["requested_time_step"],
                    time_step=values["time_step"],
                    checkpoint_interval=values.get("checkpoint_interval"),
                    initial_norm=values["initial_norm"],
                    times=archive[f"{prefix}_times"].copy(),
                    norms=archive[f"{prefix}_norms"].copy(),
                    linear_iterations=archive[
                        f"{prefix}_linear_iterations"
                    ].copy(),
                    linear_residual_norms=archive[
                        f"{prefix}_linear_residual_norms"
                    ].copy(),
                    samples=samples,
                    integrator=values["integrator"],
                    dynamics=values["dynamics"],
                )
            )
        checkpoint = RealTimeCheckpoint(
            state=archive["state"].copy(),
            segments=tuple(segments),
            compatibility=archive_metadata.get("compatibility", {}),
            metadata=archive_metadata.get("metadata", {}),
            source_path=source,
        )

    if checkpoint.state_data_id != archive_metadata.get("state_data_id"):
        raise ValueError("real-time checkpoint state checksum does not match")
    if _arrays_id(trajectory_arrays) != archive_metadata.get("trajectory_data_id"):
        raise ValueError("real-time checkpoint trajectory checksum does not match")
    return checkpoint


def _finite_array(values, name: str, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _finite_real_array(
    values,
    name: str,
    *,
    ndim: int | None = None,
) -> np.ndarray:
    result = _finite_array(values, name, ndim=ndim)
    if np.iscomplexobj(result):
        raise TypeError(f"{name} must contain real values")
    return result


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(value, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a positive integer or None")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _read_only_copy(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values).copy()
    result.flags.writeable = False
    return result


def _json_mapping(values: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        return json.loads(_canonical_json(dict(values)))
    except TypeError as exc:
        raise TypeError(f"{name} must contain only JSON-serializable values") from exc


def _array_id(values: np.ndarray) -> str:
    return hashlib.sha256(_array_digest_bytes(values)).hexdigest()


def _arrays_id(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(_array_digest_bytes(arrays[name]))
    return digest.hexdigest()


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
