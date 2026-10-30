"""Iteration sampling and logging for non-BO diatomic ground states."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from os import PathLike
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, TextIO

from grid_tdse.ground_state import GroundStateIteration

from .observables import DiatomicIonObservables


@dataclass(frozen=True)
class DiatomicIonGroundStateSample:
    """One sampled imaginary-time state and its wall-clock timings.

    ``diagnostics.step_wall_time`` covers propagation plus the solver's norm,
    energy, state-change, and residual diagnostics.  The observable and
    logging timings are recorded separately, while ``total_iteration_wall_time``
    is their sum.
    """

    diagnostics: GroundStateIteration
    observables: DiatomicIonObservables
    observable_wall_time: float
    logging_wall_time: float = 0.0
    total_iteration_wall_time: float = 0.0

    @property
    def iteration(self) -> int:
        return self.diagnostics.iteration

    @property
    def imaginary_time(self) -> float:
        return self.diagnostics.imaginary_time


class DiatomicIonGroundStateReporter:
    """Collect observable samples and mirror them to terminal and a logfile."""

    def __init__(
        self,
        evaluate: Callable[[object], DiatomicIonObservables],
        *,
        log_path: str | PathLike[str] | None = None,
        log_overwrite: bool = False,
        is_root: bool = True,
        reduce_max: Callable[[float], float] = float,
        synchronize_error: Callable[[str | None], str | None] | None = None,
        terminal: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        if not isinstance(log_overwrite, bool):
            raise TypeError("log_overwrite must be boolean")
        if not isinstance(is_root, bool):
            raise TypeError("is_root must be boolean")
        if not callable(reduce_max):
            raise TypeError("reduce_max must be callable")
        if synchronize_error is not None and not callable(synchronize_error):
            raise TypeError("synchronize_error must be callable or None")
        if not isinstance(terminal, bool):
            raise TypeError("terminal must be boolean")

        self.evaluate = evaluate
        self.is_root = is_root
        self.reduce_max = reduce_max
        self.terminal = terminal
        self.stream = stream
        self.samples: list[DiatomicIonGroundStateSample] = []
        self.log_path = None if log_path is None else Path(log_path)
        self._log_stream = None

        error = None
        exception = None
        if is_root and self.log_path is not None:
            try:
                mode = "w" if log_overwrite else "x"
                self._log_stream = self.log_path.open(mode, encoding="utf-8")
            except OSError as exc:
                exception = exc
                error = f"{type(exc).__name__}: {exc}"
        if synchronize_error is not None:
            error = synchronize_error(error)
        if error is not None:
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
            if exception is not None:
                raise exception
            raise RuntimeError(f"root could not open ground-state logfile: {error}")

    def observe(
        self,
        state,
        diagnostics: GroundStateIteration,
    ) -> None:
        """Evaluate, print, write, and retain one sampled state."""

        observable_started = perf_counter()
        observables = self.evaluate(state)
        local_observable_time = perf_counter() - observable_started
        observable_time = self.reduce_max(local_observable_time)
        step_time = self.reduce_max(diagnostics.step_wall_time)
        diagnostics = replace(diagnostics, step_wall_time=step_time)
        sample = DiatomicIonGroundStateSample(
            diagnostics=diagnostics,
            observables=observables,
            observable_wall_time=observable_time,
        )

        logging_started = perf_counter()
        if self.is_root:
            block = format_diatomic_ground_state_sample(sample)
            if self.terminal:
                print(block, file=self.stream or sys.stdout, flush=True)
            if self._log_stream is not None:
                self._log_stream.write(block + "\n")
                self._log_stream.flush()
        local_logging_time = perf_counter() - logging_started
        logging_time = self.reduce_max(local_logging_time)
        sample = replace(
            sample,
            logging_wall_time=logging_time,
            total_iteration_wall_time=(
                step_time + observable_time + logging_time
            ),
        )
        self.samples.append(sample)

        if self.is_root:
            timing = (
                "Sampling timing: "
                f"observables={observable_time:.6f} s  "
                f"logging={logging_time:.6f} s  "
                f"total={sample.total_iteration_wall_time:.6f} s"
            )
            if self.terminal:
                print(timing, file=self.stream or sys.stdout, flush=True)
            if self._log_stream is not None:
                self._log_stream.write(timing + "\n")
                self._log_stream.flush()

    def close(self) -> None:
        """Flush and close the optional logfile."""

        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def format_diatomic_ground_state_sample(
    sample: DiatomicIonGroundStateSample,
) -> str:
    """Return the stable human-readable representation of one sample."""

    record = sample.diagnostics
    values = sample.observables
    exchange = (
        "N/A"
        if values.pair_exchange is None
        else f"{values.pair_exchange:.12f}"
    )
    return "\n".join(
        (
            f"ITP sample {record.iteration} (tau={record.imaginary_time:.8f}):",
            f"E = {record.energy:.12f} Eh  |dE| = {record.energy_change:.3e} Eh",
            (
                f"dPsi = {record.state_change:.3e}  "
                f"residual = {record.residual_norm:.3e}  "
                f"norm = {record.norm:.12f}"
            ),
            f"IT step time = {record.step_wall_time:.6f} s",
            f"<I> = {values.inversion:.12f}  <P> = {exchange}",
            f"<r> = {values.r:.12f}  <R> = {values.R:.12f}",
            (
                f"<L^2> = {values.total_l_squared:.12f}  "
                f"<L_z> = {values.total_l_z:.12f}"
            ),
            (
                f"<l_r^2> = {values.l_r_squared:.12f}  "
                f"<l_r,z> = {values.l_r_z:.12f}"
            ),
            (
                f"<l_R^2> = {values.l_R_squared:.12f}  "
                f"<l_R,z> = {values.l_R_z:.12f}"
            ),
            (
                f"<cos(theta_R)> = {values.cos_theta_R:.12f}  "
                f"<cos^2(theta_R)> = {values.cos2_theta_R:.12f}"
            ),
            f"d_r = {_format_vector(values.dipole_r)} ea0",
            f"d_R = {_format_vector(values.dipole_R)} ea0",
            f"d_total = {_format_vector(values.dipole_total)} ea0",
        )
    )


def ground_state_sample_metadata(samples) -> dict:
    """Return a compact JSON-compatible columnar sample history."""

    samples = tuple(samples)
    columns = {
        "iteration": [],
        "imaginary_time": [],
        "energy": [],
        "energy_change": [],
        "state_change": [],
        "residual_norm": [],
        "norm": [],
        "step_wall_time": [],
        "observable_wall_time": [],
        "logging_wall_time": [],
        "total_iteration_wall_time": [],
        "observables": [],
    }
    for sample in samples:
        record = sample.diagnostics
        columns["iteration"].append(record.iteration)
        columns["imaginary_time"].append(record.imaginary_time)
        columns["energy"].append(record.energy)
        columns["energy_change"].append(record.energy_change)
        columns["state_change"].append(record.state_change)
        columns["residual_norm"].append(record.residual_norm)
        columns["norm"].append(record.norm)
        columns["step_wall_time"].append(record.step_wall_time)
        columns["observable_wall_time"].append(sample.observable_wall_time)
        columns["logging_wall_time"].append(sample.logging_wall_time)
        columns["total_iteration_wall_time"].append(
            sample.total_iteration_wall_time
        )
        columns["observables"].append(asdict(sample.observables))
    return {
        "units": {
            "energy": "hartree",
            "time": "atomic_time",
            "wall_time": "second",
            "length": "bohr",
            "angular_momentum": "atomic",
            "dipole": "electron_charge_bohr",
        },
        "columns": columns,
    }


def _format_vector(values) -> str:
    return "(" + ", ".join(f"{value:.12f}" for value in values) + ")"
