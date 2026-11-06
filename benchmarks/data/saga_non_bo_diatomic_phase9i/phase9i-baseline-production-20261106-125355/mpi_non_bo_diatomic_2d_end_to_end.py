"""Run the production-style Phase 9h dynamics example with Cartesian MPI.

This is the distributed counterpart of ``non_bo_diatomic_end_to_end.py``.
It intentionally does not run a serial comparison inside the MPI job; run the
serial example separately when a cross-check is desired.

Phase 9h snapshots and restart states are gathered to rank zero and written in
the portable two-radial state format.  This is suitable for acceptance tests
and modest examples.  Scalable distributed state I/O is deferred to Phase 10.

For example::

    mpiexec -n 4 python \
        examples/spherical/two_radial/mpi_non_bo_diatomic_2d_end_to_end.py \
        --process-grid 2x2 --final-time 2.0 \
        --run-directory phase9h-mpi-smoke
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from mpi4py import MPI

from grid_tdse.real_time import RealTimeSnapshotSelector
from grid_tdse.spherical.two_radial import (
    DiatomicIonDynamicsSamplingPolicy,
    TwoRadialDistributed2DLayout,
    TwoRadialDistributed2DRealTimeSolver,
    load_distributed_two_radial_2d_state,
    load_two_radial_state,
    prepare_distributed_non_bo_diatomic_dynamics_initial_state,
    save_distributed_two_radial_2d_state,
)

from non_bo_diatomic_end_to_end_common import (
    absorbing_boundaries,
    add_common_arguments,
    configuration_record,
    dynamics_representation,
    ground_state_options,
    physical_setup,
    physical_state_metadata,
    plot_dynamics,
    prepare_run_directory,
    print_initial_observables,
    propagation_diagnostics_record,
    resolved_paths,
    validate_arguments,
    write_json,
)


def process_grid(value: str) -> tuple[int, int]:
    try:
        first, second = value.lower().split("x", maxsplit=1)
        result = int(first), int(second)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "process grid must have the form P_r x P_R, for example 2x2"
        ) from exc
    if result[0] <= 0 or result[1] <= 0:
        raise argparse.ArgumentTypeError("process-grid dimensions must be positive")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--process-grid", type=process_grid)
    parser.add_argument(
        "--restart-state",
        type=Path,
        help="resume from a gathered Phase 9h MPI checkpoint state",
    )
    values = parser.parse_args()
    validate_arguments(parser, values)
    return values


values = arguments()
communicator = MPI.COMM_WORLD
if values.process_grid is not None:
    if np.prod(values.process_grid) != communicator.size:
        raise ValueError("process-grid dimensions must multiply to MPI size")

run_directory = None
directory_error = None
if communicator.rank == 0:
    try:
        run_directory = prepare_run_directory(
            values.run_directory,
            overwrite=values.overwrite,
        )
    except Exception as exc:
        directory_error = (type(exc).__name__, str(exc))
directory_error = communicator.bcast(directory_error, root=0)
if directory_error is not None:
    raise RuntimeError(
        "rank zero could not prepare the run directory: "
        f"{directory_error[1]}"
    )
run_directory = Path(communicator.bcast(str(run_directory), root=0))
paths = resolved_paths(values, run_directory)
checkpoint_directory = run_directory / "checkpoints"
if communicator.rank == 0:
    if paths["wavefunctions"] is not None:
        paths["wavefunctions"].mkdir(parents=True, exist_ok=True)
    if not values.no_checkpoints:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
communicator.Barrier()

setup = physical_setup(values)
representation = dynamics_representation(values, setup)
layout = TwoRadialDistributed2DLayout(
    representation,
    communicator.size,
    process_grid=values.process_grid,
)
boundaries = absorbing_boundaries(values, representation)

step_offset = 0
initial_time = 0.0
if values.restart_state is None:
    initialization = prepare_distributed_non_bo_diatomic_dynamics_initial_state(
        layout,
        communicator,
        system=values.system,
        electronic_grid=setup["electronic_grid"],
        internuclear_grid=setup["internuclear_grid"],
        pair_exchange_parity=setup["pair_exchange_parity"],
        archive_path=paths["ground_state"],
        policy=values.initial_state_policy,
        inversion_parity=1,
        symmetry_adapted_basis=True,
        codata_set=values.codata_set,
        radial_coulomb_method="poisson",
        ground_state_options=ground_state_options(),
    )
    initial_state = initialization.state
    initialization_metadata = initialization.metadata
else:
    restart_metadata = None
    restart_error = None
    if communicator.rank == 0:
        try:
            archive = load_two_radial_state(values.restart_state)
            restart_metadata = archive.calculation
        except Exception as exc:
            restart_error = (type(exc).__name__, str(exc))
    restart_error = communicator.bcast(restart_error, root=0)
    if restart_error is not None:
        raise RuntimeError(
            "rank zero could not inspect the restart state: "
            f"{restart_error[1]}"
        )
    restart_metadata = communicator.bcast(restart_metadata, root=0)
    if "time" not in restart_metadata or "accepted_step" not in restart_metadata:
        raise ValueError("restart state lacks Phase 9h time/step metadata")
    initial_time = float(restart_metadata["time"])
    step_offset = int(restart_metadata["accepted_step"])
    initial_state = load_distributed_two_radial_2d_state(
        values.restart_state,
        layout,
        communicator,
    )
    initialization_metadata = {
        "source": "checkpoint-state",
        "archive_path": str(values.restart_state.resolve()),
        "time": initial_time,
        "accepted_step": step_offset,
    }

if setup["final_time"] <= initial_time:
    raise ValueError("--final-time must be later than the restart-state time")

solver = TwoRadialDistributed2DRealTimeSolver(
    layout,
    setup["parameters"],
    communicator,
    fields=(setup["pulse"],),
    field_gauge=values.field_gauge,
    initial_time=initial_time,
    final_time=setup["final_time"],
    time_step=values.time_step,
    integrator="sia",
    max_krylov_dimension=20,
    krylov_rtol=values.krylov_rtol,
    krylov_atol=values.krylov_atol,
    max_krylov_refinements=8,
    krylov_reorthogonalize=True,
    krylov_breakdown_tolerance=1.0e-14,
    sampling_policy=DiatomicIonDynamicsSamplingPolicy(
        interval=values.sample_interval
    ),
    absorbing_boundaries=boundaries,
    progress_interval=values.progress_interval,
    profile=True,
)
initial_snapshot = solver.observable_evaluator.distributed_2d(
    initial_state,
    solver.reductions,
    initial_time,
)
if communicator.rank == 0:
    print_initial_observables(
        initial_snapshot,
        codata_set=values.codata_set,
        electronic_grid=setup["electronic_grid"],
        internuclear_grid=setup["internuclear_grid"],
    )

snapshot_records = []
checkpoint_records = []
physical_state = physical_state_metadata(setup)
snapshot_selector = (
    None
    if paths["wavefunctions"] is None
    else RealTimeSnapshotSelector(
        interval=values.wavefunction_snapshot_interval,
        steps=values.wavefunction_snapshot_steps,
        times=values.wavefunction_snapshot_times,
        include_initial=True,
        include_final=True,
    )
)


def retain_state(state, step, time, diagnostics, final):
    del diagnostics
    global_step = step_offset + step
    snapshot_selected = (
        snapshot_selector is not None
        and snapshot_selector.selected(
            global_step,
            step,
            time,
            final=final,
        )
    )
    checkpoint_selected = (
        not values.no_checkpoints
        and (final or global_step % values.checkpoint_interval == 0)
        and step != 0
    )
    record = {
        "accepted_step": global_step,
        "time": float(time),
        "final": bool(final),
        "execution": "cartesian-mpi-root-gather",
    }
    if snapshot_selected:
        destination = paths["wavefunctions"] / (
            f"wavefunction-step-{global_step:08d}.npz"
        )
        saved = save_distributed_two_radial_2d_state(
            destination,
            state,
            communicator,
            physical_state=physical_state,
            calculation={**record, "kind": "wavefunction-snapshot"},
            overwrite=values.overwrite,
        )
        if communicator.rank == 0:
            snapshot_records.append({**record, "path": str(saved)})
    if checkpoint_selected:
        destination = checkpoint_directory / (
            f"checkpoint-step-{global_step:08d}.npz"
        )
        saved = save_distributed_two_radial_2d_state(
            destination,
            state,
            communicator,
            physical_state=physical_state,
            calculation={**record, "kind": "restart-state"},
            overwrite=values.overwrite,
        )
        if communicator.rank == 0:
            checkpoint_records.append({**record, "path": str(saved)})


communicator.Barrier()
started = MPI.Wtime()
result = solver.run(initial_state, state_observer=retain_state)
local_wall_time = MPI.Wtime() - started
wall_time = communicator.allreduce(local_wall_time, op=MPI.MAX)

if communicator.rank == 0:
    if snapshot_selector is not None:
        snapshot_selector.warn_unmatched()
    observables_path = write_json(
        run_directory / "observables.json",
        result.time_series.records(),
    )
    radial_density_path = result.time_series.save_radial_densities(
        run_directory / "radial-densities.npz",
        overwrite=values.overwrite,
    )
    figure_path = plot_dynamics(
        result.time_series,
        setup["pulse"],
        paths["figure"],
        show=values.show_figure,
        overwrite=values.overwrite,
    )
    summary = {
        "phase": "9h",
        "execution": "cartesian-mpi",
        "mpi_ranks": communicator.size,
        "process_grid": list(layout.process_grid),
        "configuration": configuration_record(values, setup),
        "initialization": initialization_metadata,
        "accepted_steps_this_segment": result.n_steps,
        "accepted_step_offset": step_offset,
        "initial_norm": result.propagation.initial_norm,
        "final_norm": result.norm,
        "wall_time_seconds": wall_time,
        "propagation_diagnostics": propagation_diagnostics_record(
            result.propagation
        ),
        "ground_state_path": str(paths["ground_state"]),
        "wavefunction_directory": (
            None
            if paths["wavefunctions"] is None
            else str(paths["wavefunctions"])
        ),
        "wavefunction_snapshots": snapshot_records,
        "wavefunction_snapshot_selection": (
            None if snapshot_selector is None else snapshot_selector.as_dict()
        ),
        "checkpoint_directory": (
            None if values.no_checkpoints else str(checkpoint_directory)
        ),
        "checkpoint_states": checkpoint_records,
        "observables_path": str(observables_path),
        "radial_density_path": str(radial_density_path),
        "figure_path": str(figure_path),
        "hamiltonian_profile": asdict(result.hamiltonian_profile),
        "reduction_profile": asdict(result.reduction_profile),
        "io_note": (
            "Phase 9h snapshots/checkpoints gather the state to rank zero; "
            "scalable distributed I/O is Phase 10."
        ),
    }
    summary_path = write_json(run_directory / "summary.json", summary)
    print("\nPhase 9h Cartesian-MPI end-to-end dynamics")
    print(f"System:                  {values.system}")
    print(f"l_max, M_tot:            {values.l_max}, {values.M_tot}")
    print(f"MPI ranks/process grid:  {communicator.size}/{layout.process_grid}")
    print(f"Field gauge:             {values.field_gauge}")
    print(f"Pulse duration:          {setup['pulse_duration']:.12f} a.u.")
    print(f"Requested final time:    {setup['requested_final_time']:.12f} a.u.")
    print(f"Actual final time:       {setup['final_time']:.12f} a.u.")
    print(f"Requested/effective dt:  {values.time_step:.12f} a.u.")
    print(f"Accepted steps:          {result.n_steps}")
    print(f"Final norm:              {result.norm:.12f}")
    print(f"Wall time:               {wall_time:.6f} s")
    print(f"Wavefunction directory:  {paths['wavefunctions']}")
    print(f"Diagnostic figure:       {figure_path}")
    print(f"Summary JSON:            {summary_path}")
