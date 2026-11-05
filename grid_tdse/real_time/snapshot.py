"""Retained full-state snapshots for serial real-time propagation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Mapping
import warnings

import numpy as np

from .checkpoint import load_real_time_checkpoint

_FORMAT_NAME = "grid-tdse-real-time-snapshot"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RealTimeStateSnapshot:
    """One self-describing wavefunction at an accepted physical time."""

    state: np.ndarray
    accepted_step: int
    time: float
    norm: float
    compatibility: dict[str, Any]
    dynamics: dict[str, Any]
    integrator: dict[str, Any]
    metadata: dict[str, Any]
    parent_state_data_id: str
    source_path: Path | None = None

    def __post_init__(self) -> None:
        state = _finite_array(self.state, "snapshot state")
        if state.ndim == 0 or state.size == 0:
            raise ValueError("snapshot state must be a nonempty array")
        step = _nonnegative_integer(self.accepted_step, "accepted_step")
        time = _finite_real(self.time, "snapshot time")
        norm = _nonnegative_real(self.norm, "snapshot norm")
        parent = _state_identifier(self.parent_state_data_id)
        object.__setattr__(self, "state", _read_only_copy(state))
        object.__setattr__(self, "accepted_step", step)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "norm", norm)
        object.__setattr__(
            self,
            "compatibility",
            _json_mapping(self.compatibility, "compatibility metadata"),
        )
        object.__setattr__(
            self,
            "dynamics",
            _json_mapping(self.dynamics, "dynamics metadata"),
        )
        object.__setattr__(
            self,
            "integrator",
            _json_mapping(self.integrator, "integrator metadata"),
        )
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, "snapshot metadata"),
        )
        object.__setattr__(self, "parent_state_data_id", parent)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def state_data_id(self) -> str:
        """SHA-256 identifier of the stored coefficient array."""

        return real_time_state_data_id(self.state)

    def state_copy(self) -> np.ndarray:
        """Return a writable copy of the stored state."""

        return self.state.copy()

    def validate_compatibility(self, metadata: Mapping[str, Any]) -> None:
        """Require supplied physical and representation metadata to match."""

        candidate = _json_mapping(metadata, "compatibility metadata")
        if candidate != self.compatibility:
            raise ValueError("snapshot compatibility metadata does not match")


@dataclass(frozen=True)
class RealTimeSnapshotRecord:
    """Location and identity of one retained state."""

    accepted_step: int
    time: float
    path: Path
    storage: Literal["snapshot", "checkpoint"]
    final: bool
    state_data_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_step",
            _nonnegative_integer(self.accepted_step, "accepted_step"),
        )
        object.__setattr__(self, "time", _finite_real(self.time, "snapshot time"))
        object.__setattr__(self, "path", Path(self.path))
        if self.storage not in {"snapshot", "checkpoint"}:
            raise ValueError("snapshot storage must be 'snapshot' or 'checkpoint'")
        if not isinstance(self.final, (bool, np.bool_)):
            raise TypeError("snapshot final flag must be boolean")
        object.__setattr__(self, "final", bool(self.final))
        object.__setattr__(
            self,
            "state_data_id",
            _state_identifier(self.state_data_id),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest record."""

        return {
            "accepted_step": self.accepted_step,
            "time": self.time,
            "path": str(self.path),
            "storage": self.storage,
            "final": self.final,
            "state_data_id": self.state_data_id,
        }


class RealTimeSnapshotSelector:
    """Select accepted states by interval, global step, or absolute time."""

    def __init__(
        self,
        *,
        interval: int | None = None,
        steps: Iterable[int] | None = None,
        times: Iterable[float] | None = None,
        include_initial: bool = True,
        include_final: bool = True,
    ) -> None:
        self.interval = _optional_positive_integer(interval, "snapshot interval")
        self.steps = _snapshot_steps(steps)
        self.times = _snapshot_times(times)
        self.include_initial = _boolean(include_initial, "include_initial")
        self.include_final = _boolean(include_final, "include_final")
        self._matched_steps: set[int] = set()
        self._matched_time_indices: set[int] = set()
        self._warning_emitted = False

    @property
    def unmatched_steps(self) -> tuple[int, ...]:
        """Requested global steps not observed by this selector."""

        return tuple(step for step in self.steps if step not in self._matched_steps)

    @property
    def unmatched_times(self) -> tuple[float, ...]:
        """Requested absolute times not observed by this selector."""

        return tuple(
            time
            for index, time in enumerate(self.times)
            if index not in self._matched_time_indices
        )

    def selected(self, step, local_step, time, *, final=False) -> bool:
        """Return whether an accepted state belongs to the requested union."""

        step = _nonnegative_integer(step, "accepted_step")
        local_step = _nonnegative_integer(local_step, "local accepted_step")
        time = _finite_real(time, "snapshot time")
        final = _boolean(final, "final")

        step_selected = step in self.steps
        if step_selected:
            self._matched_steps.add(step)
        time_selected = False
        for index, requested in enumerate(self.times):
            if _snapshot_times_match(time, requested):
                self._matched_time_indices.add(index)
                time_selected = True

        return bool(
            (local_step == 0 and self.include_initial)
            or (step > 0 and self.interval is not None and step % self.interval == 0)
            or step_selected
            or time_selected
            or (final and self.include_final)
        )

    def warn_unmatched(self) -> None:
        """Warn once about requested steps or times that were not observed."""

        if self._warning_emitted:
            return
        self._warning_emitted = True
        parts = []
        if self.unmatched_steps:
            parts.append(f"steps={list(self.unmatched_steps)}")
        if self.unmatched_times:
            parts.append(f"times={list(self.unmatched_times)}")
        if not parts:
            return
        warnings.warn(
            "wavefunction snapshot requests did not match accepted propagation "
            f"states: {', '.join(parts)}; requested times must coincide with an "
            "accepted propagation time",
            RuntimeWarning,
            stacklevel=2,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the selection policy and unmatched requests."""

        return {
            "interval": self.interval,
            "steps": list(self.steps),
            "times": list(self.times),
            "include_initial": self.include_initial,
            "include_final": self.include_final,
            "unmatched_steps": list(self.unmatched_steps),
            "unmatched_times": list(self.unmatched_times),
        }


class RealTimeSnapshotWriter:
    """Retain selected accepted states and write a machine-readable manifest."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        interval: int | None = None,
        steps: Iterable[int] | None = None,
        times: Iterable[float] | None = None,
        include_initial: bool = True,
        include_final: bool = True,
        prefix: str = "wavefunction",
        overwrite: bool = False,
        compatibility_metadata: Mapping[str, Any] | None = None,
        dynamics_metadata: Mapping[str, Any] | None = None,
        integrator_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        parent_state_data_id: str,
        accepted_step_offset: int = 0,
        inner_product=None,
        checkpoint_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"snapshot directory does not exist: {self.directory}"
            )
        self.selector = RealTimeSnapshotSelector(
            interval=interval,
            steps=steps,
            times=times,
            include_initial=include_initial,
            include_final=include_final,
        )
        self.interval = self.selector.interval
        self.steps = self.selector.steps
        self.times = self.selector.times
        self.include_initial = self.selector.include_initial
        self.include_final = self.selector.include_final
        if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
            raise ValueError("snapshot prefix must be a nonempty file-name component")
        self.prefix = prefix
        self.overwrite = _boolean(overwrite, "snapshot overwrite")
        self.compatibility = _json_mapping(
            {} if compatibility_metadata is None else compatibility_metadata,
            "compatibility metadata",
        )
        self.dynamics = _json_mapping(
            {} if dynamics_metadata is None else dynamics_metadata,
            "dynamics metadata",
        )
        self.integrator = _json_mapping(
            {} if integrator_metadata is None else integrator_metadata,
            "integrator metadata",
        )
        self.metadata = _json_mapping(
            {} if metadata is None else metadata,
            "snapshot metadata",
        )
        self.parent_state_data_id = _state_identifier(parent_state_data_id)
        self.accepted_step_offset = _nonnegative_integer(
            accepted_step_offset,
            "accepted_step_offset",
        )
        if inner_product is not None and not callable(inner_product):
            raise TypeError("inner_product must be callable or None")
        self.inner_product = inner_product
        self.checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path)
        )
        self._records: list[RealTimeSnapshotRecord] = []

    @property
    def records(self) -> tuple[RealTimeSnapshotRecord, ...]:
        """Retained records in observation order."""

        return tuple(self._records)

    @property
    def manifest_path(self) -> Path:
        return self.directory / f"{self.prefix}-manifest.json"

    def __call__(
        self,
        state,
        accepted_step,
        time,
        diagnostics,
        final,
        checkpoint,
    ) -> None:
        local_step = _nonnegative_integer(accepted_step, "accepted_step")
        step = self.accepted_step_offset + local_step
        final = _boolean(final, "final")
        checkpoint = _boolean(checkpoint, "checkpoint")
        time = _finite_real(time, "snapshot time")
        if not self.selector.selected(step, local_step, time, final=final):
            return
        values = _finite_array(state, "snapshot state")
        state_data_id = real_time_state_data_id(values)
        norm = self._norm(values, diagnostics)
        stem = f"{self.prefix}-step-{step:08d}-t{_time_token(time)}"
        if checkpoint:
            record = self._retain_checkpoint(
                stem,
                step,
                time,
                final,
                state_data_id,
            )
        else:
            destination = self.directory / f"{stem}.npz"
            snapshot = RealTimeStateSnapshot(
                state=values,
                accepted_step=step,
                time=time,
                norm=norm,
                compatibility=self.compatibility,
                dynamics=self.dynamics,
                integrator=self.integrator,
                metadata=self.metadata,
                parent_state_data_id=self.parent_state_data_id,
            )
            _save_or_validate_snapshot(destination, snapshot, self.overwrite)
            record = RealTimeSnapshotRecord(
                accepted_step=step,
                time=time,
                path=destination,
                storage="snapshot",
                final=final,
                state_data_id=state_data_id,
            )
        self._append_record(record)

    def finish(self) -> Path:
        """Atomically write the retained-state manifest."""

        self.selector.warn_unmatched()
        manifest = {
            "format": "grid-tdse-real-time-snapshot-manifest",
            "format_version": 1,
            "parent_state_data_id": self.parent_state_data_id,
            "compatibility": self.compatibility,
            "dynamics": self.dynamics,
            "integrator": self.integrator,
            "metadata": self.metadata,
            "selection": self.selector.as_dict(),
            "records": [record.as_dict() for record in self._records],
        }
        _atomic_json(self.manifest_path, manifest, overwrite=self.overwrite)
        return self.manifest_path

    def _norm(self, state, diagnostics) -> float:
        if diagnostics is not None:
            return _nonnegative_real(diagnostics.norm, "snapshot norm")
        if self.inner_product is None:
            raise ValueError("inner_product is required for an initial snapshot")
        value = complex(np.asarray(self.inner_product(state, state)))
        tolerance = 1.0e-10 * max(1.0, abs(value.real))
        if abs(value.imag) > tolerance or value.real < 0.0:
            raise ValueError("inner_product produced an invalid snapshot norm")
        return float(value.real)

    def _retain_checkpoint(
        self,
        stem,
        step,
        time,
        final,
        state_data_id,
    ) -> RealTimeSnapshotRecord:
        if self.checkpoint_path is None:
            raise RuntimeError("checkpoint snapshot has no configured checkpoint path")
        if not self.checkpoint_path.is_file():
            raise RuntimeError(
                "checkpoint must be written before its snapshot is retained"
            )
        destination = self.directory / f"{stem}.chkpt"
        if destination.exists():
            if self.overwrite:
                destination.unlink()
            else:
                stored = load_real_time_checkpoint(destination)
                if (
                    stored.state_data_id != state_data_id
                    or stored.total_steps != step
                    or not np.isclose(stored.current_time, time, atol=1.0e-13)
                ):
                    raise FileExistsError(
                        f"snapshot checkpoint already exists: {destination}"
                    )
                return RealTimeSnapshotRecord(
                    step,
                    time,
                    destination,
                    "checkpoint",
                    final,
                    state_data_id,
                )
        os.link(self.checkpoint_path, destination)
        stored = load_real_time_checkpoint(destination)
        if (
            stored.state_data_id != state_data_id
            or stored.total_steps != step
            or not np.isclose(stored.current_time, time, atol=1.0e-13)
        ):
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "retained checkpoint state does not match observed state"
            )
        return RealTimeSnapshotRecord(
            step,
            time,
            destination,
            "checkpoint",
            final,
            state_data_id,
        )

    def _append_record(self, record: RealTimeSnapshotRecord) -> None:
        for index, previous in enumerate(self._records):
            if previous.path == record.path:
                if previous != record:
                    raise ValueError("snapshot path refers to inconsistent records")
                self._records[index] = record
                return
        self._records.append(record)


def save_real_time_snapshot(
    path: str | os.PathLike[str],
    snapshot: RealTimeStateSnapshot,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save a lightweight full-state snapshot."""

    if not isinstance(snapshot, RealTimeStateSnapshot):
        raise TypeError("snapshot must be a RealTimeStateSnapshot")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"snapshot already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"snapshot directory does not exist: {destination.parent}"
        )
    archive_metadata = {
        "format": _FORMAT_NAME,
        "format_version": _FORMAT_VERSION,
        "units": "atomic",
        "created_with": {
            "grid_tdse_version": _package_version(),
            "numpy_version": np.__version__,
        },
        "accepted_step": snapshot.accepted_step,
        "time": snapshot.time,
        "norm": snapshot.norm,
        "compatibility": snapshot.compatibility,
        "dynamics": snapshot.dynamics,
        "integrator": snapshot.integrator,
        "metadata": snapshot.metadata,
        "parent_state_data_id": snapshot.parent_state_data_id,
        "state_data_id": snapshot.state_data_id,
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
            np.savez(
                stream,
                state=np.asarray(snapshot.state),
                metadata_json=np.asarray(metadata_json),
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


def load_real_time_snapshot(
    path: str | os.PathLike[str],
) -> RealTimeStateSnapshot:
    """Load and validate a lightweight full-state snapshot."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "state" not in archive.files or "metadata_json" not in archive.files:
            raise ValueError("real-time snapshot is missing required arrays")
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("real-time snapshot contains invalid metadata") from exc
        if metadata.get("format") != _FORMAT_NAME:
            raise ValueError("file is not a grid-tdse real-time snapshot")
        if metadata.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                "unsupported real-time snapshot format version: "
                f"{metadata.get('format_version')!r}"
            )
        snapshot = RealTimeStateSnapshot(
            state=archive["state"].copy(),
            accepted_step=metadata["accepted_step"],
            time=metadata["time"],
            norm=metadata["norm"],
            compatibility=metadata["compatibility"],
            dynamics=metadata["dynamics"],
            integrator=metadata["integrator"],
            metadata=metadata["metadata"],
            parent_state_data_id=metadata["parent_state_data_id"],
            source_path=source,
        )
    if snapshot.state_data_id != metadata.get("state_data_id"):
        raise ValueError("real-time snapshot state checksum does not match")
    return snapshot


def real_time_state_data_id(state: np.ndarray) -> str:
    """Return a stable SHA-256 identifier for a numeric state array."""

    values = np.ascontiguousarray(np.asarray(state))
    header = _canonical_json(
        {"dtype": values.dtype.str, "shape": list(values.shape)}
    ).encode()
    return hashlib.sha256(header + values.tobytes(order="C")).hexdigest()


def _save_or_validate_snapshot(path, snapshot, overwrite) -> None:
    if not path.exists() or overwrite:
        save_real_time_snapshot(path, snapshot, overwrite=overwrite)
        return
    stored = load_real_time_snapshot(path)
    if (
        stored.state_data_id != snapshot.state_data_id
        or stored.accepted_step != snapshot.accepted_step
        or not np.isclose(stored.time, snapshot.time, atol=1.0e-13)
    ):
        raise FileExistsError(f"snapshot already exists: {path}")


def _atomic_json(path: Path, values: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == values:
            return
        raise FileExistsError(f"snapshot manifest already exists: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(json.dumps(values, indent=2, allow_nan=False) + "\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _finite_array(values, name: str) -> np.ndarray:
    result = np.asarray(values)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _read_only_copy(values) -> np.ndarray:
    result = np.asarray(values).copy()
    result.flags.writeable = False
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


def _nonnegative_real(value, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_positive_integer(value, name: str) -> int | None:
    if value is None:
        return None
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _snapshot_steps(values) -> tuple[int, ...]:
    if values is None:
        return ()
    try:
        candidates = tuple(values)
    except TypeError as exc:
        raise TypeError("snapshot steps must be an iterable or None") from exc
    return tuple(
        sorted(
            {
                _nonnegative_integer(value, "snapshot step")
                for value in candidates
            }
        )
    )


def _snapshot_times(values) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        candidates = tuple(values)
    except TypeError as exc:
        raise TypeError("snapshot times must be an iterable or None") from exc
    return tuple(sorted({_finite_real(value, "snapshot time") for value in candidates}))


def _snapshot_times_match(actual: float, requested: float) -> bool:
    scale = max(1.0, abs(actual), abs(requested))
    tolerance = 64.0 * np.finfo(float).eps * scale
    return bool(np.isclose(actual, requested, rtol=0.0, atol=tolerance))


def _boolean(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _json_mapping(values, name: str) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = _canonical_json(dict(values))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON-serializable") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must be a mapping")
    return decoded


def _canonical_json(values) -> str:
    return json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _state_identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("state identifier must be a lowercase SHA-256 digest")
    return value


def _time_token(time: float) -> str:
    return format(time, ".16g").replace("-", "m").replace(".", "p").replace("+", "")


def _package_version() -> str:
    try:
        return version("HyQD-grid-tdse")
    except PackageNotFoundError:
        return "0+unknown"
