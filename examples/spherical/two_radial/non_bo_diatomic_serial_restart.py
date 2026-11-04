"""Validate serial non-BO snapshots, checkpoints, and continuation.

The example runs the same finite length-gauge pulse in two ways: one
uninterrupted propagation and two stages joined through a rolling checkpoint.
It retains full-wavefunction snapshots independently of the observable
sampling interval and verifies that the final states and sampled time series
agree.  The default grid is deliberately small enough for a local smoke test.

For example::

    python examples/spherical/two_radial/non_bo_diatomic_serial_restart.py \
        --run-directory phase9e-smoke \
        --overwrite
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isfinite, pi
from pathlib import Path
from time import perf_counter

import numpy as np

from grid_tdse.real_time import load_real_time_checkpoint
from grid_tdse.spherical.two_radial import (
    DiatomicIonDynamicsSamplingPolicy,
    DiatomicIonDynamicsTimeSeries,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    phase_aligned_two_radial_state_difference,
)


@dataclass(frozen=True)
class SineSquaredElectricField:
    """Finite sine-square electric field with a sinusoidal carrier."""

    amplitude: float
    duration: float
    carrier_cycles: float = 1.0
    gauge: str = "length"

    def __call__(self, time: float) -> float:
        if time < 0.0 or time > self.duration:
            return 0.0
        envelope = np.sin(pi * time / self.duration) ** 2
        carrier = np.sin(2.0 * pi * self.carrier_cycles * time / self.duration)
        return float(self.amplitude * envelope * carrier)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), default="H2+")
    parser.add_argument("--l-max", type=int, default=1)
    parser.add_argument(
        "--electronic-nodes",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 2.0),
    )
    parser.add_argument("--electronic-points-per-element", type=int, default=3)
    parser.add_argument(
        "--internuclear-nodes",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 2.0),
    )
    parser.add_argument("--internuclear-points-per-element", type=int, default=3)
    parser.add_argument("--codata-set", choices=("2018", "2022"), default="2022")
    parser.add_argument("--ground-state-path", type=Path)
    parser.add_argument(
        "--initial-state-policy",
        choices=("require", "compute-if-missing", "recompute"),
        default="compute-if-missing",
    )
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--split-time", type=float, default=0.02)
    parser.add_argument("--final-time", type=float, default=0.04)
    parser.add_argument("--field-strength", type=float, default=0.02)
    parser.add_argument("--carrier-cycles", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--snapshot-interval", type=int, default=2)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("phase9e_serial_restart"),
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    if arguments.l_max < 1:
        parser.error("--l-max must be at least 1 for a length-gauge pulse")
    for name in ("time_step", "final_time", "field_strength", "carrier_cycles"):
        value = getattr(arguments, name)
        if not isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < arguments.split_time < arguments.final_time:
        parser.error("--split-time must lie strictly between zero and --final-time")
    for name in ("split_time", "final_time"):
        steps = getattr(arguments, name) / arguments.time_step
        if not np.isclose(steps, round(steps), atol=1.0e-12):
            parser.error(
                f"--{name.replace('_', '-')} must be an integer multiple "
                "of --time-step"
            )
    if arguments.sample_interval < 1 or arguments.snapshot_interval < 1:
        parser.error("sampling and snapshot intervals must be positive")
    for option, nodes in (
        ("electronic", arguments.electronic_nodes),
        ("internuclear", arguments.internuclear_nodes),
    ):
        if len(nodes) < 2 or nodes[0] != 0.0:
            parser.error(f"--{option}-nodes must start at zero")
        if any(not isfinite(value) for value in nodes):
            parser.error(f"--{option}-nodes must contain finite values")
        if any(right <= left for left, right in zip(nodes, nodes[1:])):
            parser.error(f"--{option}-nodes must be strictly increasing")
    return arguments


def _grid(nodes, points_per_element) -> FEMDVRGridSpec:
    return FEMDVRGridSpec(nodes, [points_per_element] * (len(nodes) - 1))


def _default_ground_state_path(arguments, directory: Path) -> Path:
    system = arguments.system.replace("+", "p")
    return directory / f"ground-state-{system}-l{arguments.l_max}.npz"


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"run directory is not empty: {path}; pass --overwrite to reuse it"
        )
    path.mkdir(parents=True, exist_ok=True)


arguments = _arguments()
run_directory = arguments.run_directory.resolve()
_prepare_directory(run_directory, overwrite=arguments.overwrite)
first_snapshot_directory = run_directory / "snapshots-stage-1"
second_snapshot_directory = run_directory / "snapshots-stage-2"
first_snapshot_directory.mkdir(exist_ok=True)
second_snapshot_directory.mkdir(exist_ok=True)
checkpoint_path = run_directory / "restart.chkpt"
result_path = run_directory / "result.json"

electronic_grid = _grid(
    arguments.electronic_nodes,
    arguments.electronic_points_per_element,
)
internuclear_grid = _grid(
    arguments.internuclear_nodes,
    arguments.internuclear_points_per_element,
)
ground_state_path = (
    _default_ground_state_path(arguments, run_directory)
    if arguments.ground_state_path is None
    else arguments.ground_state_path
)
identical_pair = arguments.system != "HD+"
pair_exchange_parity = 1 if identical_pair else None
field = SineSquaredElectricField(
    amplitude=arguments.field_strength,
    duration=arguments.final_time,
    carrier_cycles=arguments.carrier_cycles,
)


def _calculation(
    *,
    initial_state,
    initial_time,
    final_time,
    checkpoint=None,
    checkpoint_interval=None,
    snapshot_directory=None,
    snapshot_prefix="wavefunction",
):
    automatic = initial_state is None
    return NonBornOppenheimerDiatomicIonTDSE(
        arguments.system,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        initial_state=initial_state,
        pair_exchange_parity=pair_exchange_parity,
        initial_state_policy=(
            arguments.initial_state_policy if automatic else "require"
        ),
        ground_state_path=ground_state_path if automatic else None,
        ground_state_symmetry_adapted_basis=identical_pair,
        ground_state_options=(
            {
                "max_iterations": 500,
                "residual_tolerance": 1.0e-8,
                "maximum_subspace_dimension": 30,
                "retained_ritz_vectors": 4,
                "davidson_preconditioner": "one-body-spectral",
                "progress_interval": 10,
                "sample_interval": 10,
            }
            if automatic
            else None
        ),
        fields=(field,),
        field_gauge="length",
        l_max=arguments.l_max,
        m_max=arguments.l_max,
        lambda_max=2 * arguments.l_max,
        mu_max=2 * arguments.l_max,
        M_tot=0,
        codata_set=arguments.codata_set,
        radial_coulomb_method="poisson",
        radial_operator_mode="element-local",
        integrator="sil",
        integrator_options={
            "max_krylov_dimension": 20,
            "rtol": 1.0e-11,
            "atol": 1.0e-13,
            "max_refinements": 8,
            "reorthogonalize": True,
            "breakdown_tolerance": 1.0e-14,
        },
        initial_time=initial_time,
        final_time=final_time,
        time_step=arguments.time_step,
        observables=None,
        dynamics_sampling_policy=DiatomicIonDynamicsSamplingPolicy(
            interval=arguments.sample_interval
        ),
        checkpoint_path=checkpoint,
        checkpoint_interval=checkpoint_interval,
        checkpoint_overwrite=arguments.overwrite or initial_time > 0.0,
        checkpoint_metadata={"phase": "9e", "example": Path(__file__).name},
        wavefunction_snapshot_directory=snapshot_directory,
        wavefunction_snapshot_interval=(
            arguments.snapshot_interval if snapshot_directory is not None else None
        ),
        wavefunction_snapshot_prefix=snapshot_prefix,
        wavefunction_snapshot_overwrite=arguments.overwrite,
        wavefunction_snapshot_metadata={
            "code_provenance": {"example": Path(__file__).name}
        },
        progress_interval=10,
    )


print("Preparing the shared Davidson initial state", flush=True)
preparation = _calculation(
    initial_state=None,
    initial_time=0.0,
    final_time=arguments.time_step,
)
initial_archive = preparation.initialization.archive

print("Running uninterrupted reference", flush=True)
started = perf_counter()
uninterrupted = _calculation(
    initial_state=initial_archive,
    initial_time=0.0,
    final_time=arguments.final_time,
)
uninterrupted_result = uninterrupted.run(progress=True)
uninterrupted_wall_time = perf_counter() - started

print("Running checkpointed stage 1", flush=True)
started = perf_counter()
stage1 = _calculation(
    initial_state=initial_archive,
    initial_time=0.0,
    final_time=arguments.split_time,
    checkpoint=checkpoint_path,
    checkpoint_interval=arguments.snapshot_interval,
    snapshot_directory=first_snapshot_directory,
    snapshot_prefix="stage1",
)
stage1_result = stage1.run(progress=True)
stage1_wall_time = perf_counter() - started
checkpoint = load_real_time_checkpoint(checkpoint_path)

print("Continuing checkpointed stage 2", flush=True)
started = perf_counter()
stage2 = _calculation(
    initial_state=checkpoint.state_copy(),
    initial_time=arguments.split_time,
    final_time=arguments.final_time,
    checkpoint=checkpoint_path,
    checkpoint_interval=arguments.snapshot_interval,
    snapshot_directory=second_snapshot_directory,
    snapshot_prefix="stage2",
)
stage2_result = stage2.continue_from(checkpoint, progress=True)
stage2_wall_time = perf_counter() - started
final_checkpoint = load_real_time_checkpoint(checkpoint_path)

state_difference = phase_aligned_two_radial_state_difference(
    uninterrupted_result.state,
    stage2_result.state,
    uninterrupted.representation.inner_product,
)
combined_series = DiatomicIonDynamicsTimeSeries(
    stage1.dynamics_time_series.samples + stage2.dynamics_time_series.samples[1:]
)
reference_records = uninterrupted.dynamics_time_series.records()
combined_records = combined_series.records()
series_difference = 0.0
categorical_series_match = True
for candidate, reference in zip(combined_records, reference_records, strict=True):
    for name, candidate_value in candidate.items():
        if name.startswith("propagation."):
            continue
        reference_value = reference[name]
        if isinstance(candidate_value, (bool, np.bool_)):
            continue
        if isinstance(candidate_value, str):
            categorical_series_match &= candidate_value == reference_value
            continue
        series_difference = max(
            series_difference,
            abs(float(candidate_value) - float(reference_value)),
        )

status = (
    "PASS"
    if state_difference < 1.0e-11
    and series_difference < 1.0e-10
    and categorical_series_match
    and len(final_checkpoint.segments) == 2
    else "FAIL"
)
summary = {
    "phase": "9e",
    "status": status,
    "system": arguments.system,
    "l_max": arguments.l_max,
    "time_step": arguments.time_step,
    "split_time": arguments.split_time,
    "final_time": arguments.final_time,
    "state_difference": state_difference,
    "time_series_max_difference": series_difference,
    "categorical_time_series_match": categorical_series_match,
    "checkpoint": str(checkpoint_path),
    "checkpoint_segments": len(final_checkpoint.segments),
    "checkpoint_total_steps": final_checkpoint.total_steps,
    "stage1_snapshot_manifest": str(stage1.wavefunction_snapshot_manifest),
    "stage2_snapshot_manifest": str(stage2.wavefunction_snapshot_manifest),
    "stage1_snapshots": [record.as_dict() for record in stage1.wavefunction_snapshots],
    "stage2_snapshots": [record.as_dict() for record in stage2.wavefunction_snapshots],
    "wall_time": {
        "uninterrupted": uninterrupted_wall_time,
        "stage1": stage1_wall_time,
        "stage2": stage2_wall_time,
    },
}
result_path.write_text(
    json.dumps(summary, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(f"Phase 9e gate:             {status}")
print(f"Final-state difference:   {state_difference:.3e}")
print(f"Time-series difference:   {series_difference:.3e}")
print(f"Checkpoint segments:      {len(final_checkpoint.segments)}")
print(f"Stage 1 final norm:       {stage1_result.norm:.12f}")
print(f"Stage 2 final norm:       {stage2_result.norm:.12f}")
print(f"Result JSON:              {result_path}")
if status != "PASS":
    raise RuntimeError("Phase 9e serial restart validation failed")
