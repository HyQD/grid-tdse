"""Run a production-style serial non-BO diatomic dynamics calculation.

The default is H2+ in the legacy ``sp`` ground-state sector, embedded into a
full electronic-parity ``M_tot=0`` dynamics basis.  A three-cycle, 800-nm
``SineSquareLaser`` pulse at 3e13 W/cm^2 is propagated with SIA and cubic
absorbers.  The complete observable contract is sampled every accepted step.

The pulse and requested propagation endpoint are independent.  For a quick
local smoke test, retain the pulse definition but stop early, for example::

    python examples/spherical/two_radial/non_bo_diatomic_end_to_end.py \
        --final-time 2.0 --run-directory phase9h-serial-smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from grid_lib.spherical_coordinates import CubicCAP

from grid_tdse.real_time import load_real_time_checkpoint
from grid_tdse.spherical.two_radial import (
    DiatomicIonDynamicsSamplingPolicy,
    NonBornOppenheimerDiatomicIonTDSE,
)

from non_bo_diatomic_end_to_end_common import (
    add_common_arguments,
    configuration_record,
    ground_state_options,
    physical_setup,
    plot_dynamics,
    prepare_run_directory,
    print_initial_observables,
    propagation_diagnostics_record,
    resolved_paths,
    validate_arguments,
    write_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--restart-checkpoint",
        type=Path,
        help="exactly resume a checkpoint made by this example",
    )
    values = parser.parse_args()
    validate_arguments(parser, values)
    return values


def calculation(
    values,
    setup,
    *,
    initial_state,
    initial_time,
    ground_state_path=None,
    initial_state_policy="require",
    checkpoint_path=None,
    wavefunction_directory=None,
):
    cap_r = None
    cap_R = None
    if not values.no_cap:
        cap_r = CubicCAP(alpha=values.cap_alpha, eta=values.cap_eta_r)
        cap_R = CubicCAP(alpha=values.cap_alpha, eta=values.cap_eta_R)
    return NonBornOppenheimerDiatomicIonTDSE(
        values.system,
        electronic_grid=setup["electronic_grid"],
        internuclear_grid=setup["internuclear_grid"],
        initial_state=initial_state,
        pair_exchange_parity=setup["pair_exchange_parity"],
        initial_state_policy=initial_state_policy,
        ground_state_path=ground_state_path,
        ground_state_inversion_parity=1,
        ground_state_symmetry_adapted_basis=True,
        ground_state_options=(
            ground_state_options() if initial_state is None else None
        ),
        fields=(setup["pulse"],),
        field_gauge=values.field_gauge,
        l_max=values.l_max,
        m_max=values.l_max,
        lambda_max=2 * values.l_max,
        mu_max=2 * values.l_max,
        M_tot=values.M_tot,
        codata_set=values.codata_set,
        radial_coulomb_method="poisson",
        radial_operator_mode="element-local",
        coulomb_kernel=values.coulomb_kernel,
        radial_kernel=values.radial_kernel,
        dipole_kernel=values.dipole_kernel,
        integrator="sia",
        integrator_options={
            "max_krylov_dimension": 20,
            "rtol": values.krylov_rtol,
            "atol": values.krylov_atol,
            "max_refinements": 8,
            "reorthogonalize": True,
            "breakdown_tolerance": 1.0e-14,
        },
        initial_time=initial_time,
        final_time=setup["final_time"],
        time_step=values.time_step,
        cap_r=cap_r,
        cap_R=cap_R,
        observables=None,
        dynamics_sampling_policy=DiatomicIonDynamicsSamplingPolicy(
            interval=values.sample_interval
        ),
        checkpoint_path=checkpoint_path,
        checkpoint_interval=(
            None if checkpoint_path is None else values.checkpoint_interval
        ),
        checkpoint_overwrite=values.overwrite or values.restart_checkpoint is not None,
        checkpoint_metadata={
            "phase": "9h",
            "example": Path(__file__).name,
        },
        wavefunction_snapshot_directory=wavefunction_directory,
        wavefunction_snapshot_interval=(
            values.wavefunction_snapshot_interval
            if wavefunction_directory is not None
            else None
        ),
        wavefunction_snapshot_steps=(
            values.wavefunction_snapshot_steps
            if wavefunction_directory is not None
            else None
        ),
        wavefunction_snapshot_times=(
            values.wavefunction_snapshot_times
            if wavefunction_directory is not None
            else None
        ),
        wavefunction_snapshot_initial=True,
        wavefunction_snapshot_final=True,
        wavefunction_snapshot_prefix="wavefunction",
        wavefunction_snapshot_overwrite=values.overwrite,
        wavefunction_snapshot_metadata={
            "code_provenance": {"example": Path(__file__).name}
        },
        progress_interval=values.progress_interval,
    )


values = arguments()
run_directory = prepare_run_directory(
    values.run_directory,
    overwrite=values.overwrite,
)
paths = resolved_paths(values, run_directory)
setup = physical_setup(values)
if paths["wavefunctions"] is not None:
    paths["wavefunctions"].mkdir(parents=True, exist_ok=True)

checkpoint = None
initial_time = 0.0
initial_state = None
initialization = None
if values.restart_checkpoint is None:
    print("Preparing the Davidson initial state", flush=True)
    preparation = calculation(
        values,
        setup,
        initial_state=None,
        initial_time=0.0,
        ground_state_path=paths["ground_state"],
        initial_state_policy=values.initial_state_policy,
    )
    initial_state = preparation.initialization.archive
    initialization = preparation.initialization.metadata
else:
    checkpoint = load_real_time_checkpoint(values.restart_checkpoint)
    initial_time = checkpoint.current_time
    initial_state = checkpoint.state_copy()
    initialization = {
        "source": "checkpoint",
        "checkpoint_path": str(values.restart_checkpoint.resolve()),
        "accepted_step": checkpoint.current_step,
        "time": checkpoint.current_time,
        "state_data_id": checkpoint.state_data_id,
    }

if setup["final_time"] <= initial_time:
    raise ValueError("--final-time must be later than the restart time")

run = calculation(
    values,
    setup,
    initial_state=initial_state,
    initial_time=initial_time,
    checkpoint_path=paths["checkpoint"],
    wavefunction_directory=paths["wavefunctions"],
)
initial_snapshot = run.dynamics_observable_evaluator(
    run.initial_state,
    initial_time,
)
print_initial_observables(
    initial_snapshot,
    codata_set=values.codata_set,
    electronic_grid=setup["electronic_grid"],
    internuclear_grid=setup["internuclear_grid"],
)
print("Running serial SIA dynamics", flush=True)
started = perf_counter()
if checkpoint is None:
    result = run.run(progress=True)
else:
    result = run.resume(checkpoint, progress=True)
wall_time = perf_counter() - started

observables_path = write_json(
    run_directory / "observables.json",
    run.dynamics_time_series.records(),
)
radial_density_path = run.dynamics_time_series.save_radial_densities(
    run_directory / "radial-densities.npz",
    overwrite=values.overwrite,
)
figure_path = plot_dynamics(
    run.dynamics_time_series,
    setup["pulse"],
    paths["figure"],
    show=values.show_figure,
    overwrite=values.overwrite,
)
summary = {
    "phase": "9h",
    "execution": "serial",
    "configuration": configuration_record(values, setup),
    "initialization": initialization,
    "accepted_steps": result.n_steps,
    "initial_norm": result.initial_norm,
    "final_norm": result.norm,
    "wall_time_seconds": wall_time,
    "propagation_diagnostics": propagation_diagnostics_record(result),
    "ground_state_path": str(paths["ground_state"]),
    "checkpoint_path": (
        None if paths["checkpoint"] is None else str(paths["checkpoint"])
    ),
    "wavefunction_directory": (
        None
        if paths["wavefunctions"] is None
        else str(paths["wavefunctions"])
    ),
    "wavefunction_snapshots": [
        {
            "accepted_step": record.accepted_step,
            "time": record.time,
            "path": str(record.path),
            "storage": record.storage,
            "final": record.final,
        }
        for record in run.wavefunction_snapshots
    ],
    "observables_path": str(observables_path),
    "radial_density_path": str(radial_density_path),
    "figure_path": str(figure_path),
}
summary_path = write_json(run_directory / "summary.json", summary)

print("\nPhase 9h serial end-to-end dynamics")
print(f"System:                  {values.system}")
print(f"l_max, M_tot:            {values.l_max}, {values.M_tot}")
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
