"""Qualify coupled-angular Cartesian MPI propagation on Olivia.

The driver uses a deliberately small HD+ finite space while exercising odd
and even Coulomb multipoles, a z-polarized length-gauge interaction, batched
Krylov reductions, distributed observables, and representation-safe restart.
One-rank output provides the numerical reference for the one- and two-node
MPI cases launched by ``qualify_grid_tdse_coupled_step16.sh``.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import socket
from time import perf_counter

from mpi4py import MPI
import numpy as np

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    CoupledAngularBasisSpec,
    CoupledTwoRadialAxialDipoleInteraction,
    DiatomicIonDynamicsSamplingPolicy,
    FEMDVRGridSpec,
    TwoRadialDistributed2DHamiltonian,
    TwoRadialDistributed2DLayout,
    TwoRadialDistributed2DLengthGaugeInteraction,
    TwoRadialDistributed2DRealTimeSolver,
    TwoRadialAbsorbingBoundaries,
    TwoRadialSystemSpec,
    gather_two_radial_2d_state,
    load_coupled_two_radial_state,
    load_distributed_two_radial_2d_state,
    radial_potential,
    save_distributed_two_radial_2d_state,
    scatter_two_radial_2d_state,
)
from grid_tdse.spherical.two_radial.diatomic import _diatomic_parameters

FINAL_TIME = 0.02
MIDPOINT = 0.01
TIME_STEP = 0.01
TOLERANCE = 1.0e-8


class _Pulse:
    gauge = "length"

    def __call__(self, time):
        return 0.03 * np.sin(np.pi * time / (2.0 * FINAL_TIME)) ** 2


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-grid",
        type=int,
        nargs=2,
        required=True,
        metavar=("P_r", "P_R"),
    )
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _problem():
    parameters = _diatomic_parameters("HD+", "2018")
    electronic_grid = FEMDVRGridSpec(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [3, 3, 3, 3],
    )
    internuclear_grid = FEMDVRGridSpec(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [3, 3, 3, 3],
    )
    specification = TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=electronic_grid,
            basis=AngularBasisSpec(1, 1),
            kinetic_prefactor=-0.5 / parameters.electronic_reduced_mass,
        ),
        coordinate2=CoordinateSpec(
            grid=internuclear_grid,
            basis=AngularBasisSpec(1, 1),
            kinetic_prefactor=-0.5 / parameters.internuclear_reduced_mass,
            one_body_terms=(
                radial_potential(
                    lambda radius: parameters.reference_charge
                    * parameters.partner_charge
                    / radius
                ),
            ),
        ),
        multipoles=CoulombMultipoleSpec(2, 2),
        coulomb_terms=(
            CoulombTerm(
                parameters.reference_center_scale,
                parameters.electron_charge * parameters.reference_charge,
            ),
            CoulombTerm(
                parameters.partner_center_scale,
                parameters.electron_charge * parameters.partner_charge,
            ),
        ),
        M_tot=0,
        radial_coulomb_method="analytic",
    )
    representation = specification.build_coupled(
        CoupledAngularBasisSpec(1, 1, 2, M_tot=0)
    )
    absorbing_boundaries = TwoRadialAbsorbingBoundaries(
        representation,
        cap1=lambda radius, outer: -0.05j * (radius / outer) ** 3,
        cap2=lambda radius, outer: -0.05j * (radius / outer) ** 3,
    )
    return representation, parameters, absorbing_boundaries


def _solver(
    layout,
    parameters,
    communicator,
    pulse,
    absorbing_boundaries,
    initial_time,
    final_time,
):
    return TwoRadialDistributed2DRealTimeSolver(
        layout,
        parameters,
        communicator,
        fields=(pulse,),
        field_gauge="length",
        initial_time=initial_time,
        final_time=final_time,
        time_step=TIME_STEP,
        integrator="sia",
        max_krylov_dimension=20,
        krylov_rtol=1.0e-11,
        krylov_atol=1.0e-13,
        max_krylov_refinements=6,
        sampling_policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
        absorbing_boundaries=absorbing_boundaries,
        progress_interval=None,
        krylov_reduction_mode="batched",
        profile=True,
    )


def _series(result):
    return {
        "times": [float(sample.time) for sample in result.time_series.samples],
        "norms": [float(sample.snapshot.norm) for sample in result.time_series.samples],
        "cap_norm_losses": [
            float(sample.cap_norm_loss) for sample in result.time_series.samples
        ],
        "energy_numerators": [
            float(sample.snapshot.energy_numerator)
            for sample in result.time_series.samples
        ],
        "total_dipoles_z": [
            float(sample.snapshot.conditional.dipole_total[2])
            for sample in result.time_series.samples
        ],
        "total_l_squared": [
            float(sample.snapshot.conditional.total_l_squared)
            for sample in result.time_series.samples
        ],
    }


def _maximum_difference(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(f"incompatible comparison shapes: {left.shape}, {right.shape}")
    return float(np.max(np.abs(left - right), initial=0.0))


def _write_json(path, report, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"result already exists: {path}")
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    arguments = _arguments()
    communicator = MPI.COMM_WORLD
    rank = communicator.rank
    size = communicator.size
    process_grid = tuple(arguments.process_grid)
    if process_grid[0] * process_grid[1] != size:
        raise ValueError(
            f"process grid {process_grid} requires {np.prod(process_grid)} ranks, "
            f"received {size}"
        )

    representation, parameters, absorbing_boundaries = _problem()
    layout = TwoRadialDistributed2DLayout(
        representation,
        size,
        process_grid=process_grid,
    )
    state = None
    if rank == 0:
        state = representation.random_state(
            np.random.default_rng(20261112),
            dtype=np.complex128,
            normalize=True,
        )
    local = scatter_two_radial_2d_state(layout, communicator, state)

    distributed_hamiltonian = TwoRadialDistributed2DHamiltonian(
        layout,
        communicator,
        profile=True,
    )
    components = distributed_hamiltonian.components(local)
    coordinate1 = gather_two_radial_2d_state(
        components.coordinate1,
        communicator,
    )
    coordinate2 = gather_two_radial_2d_state(
        components.coordinate2,
        communicator,
    )
    coulomb = gather_two_radial_2d_state(components.coulomb, communicator)
    total = gather_two_radial_2d_state(components.total, communicator)

    pulse = _Pulse()
    distributed_dipole = TwoRadialDistributed2DLengthGaugeInteraction(
        layout,
        parameters,
        fields=(pulse,),
    )
    field = gather_two_radial_2d_state(
        distributed_dipole(local, 0.013),
        communicator,
    )

    component_errors = None
    if rank == 0:
        serial_hamiltonian = representation.hamiltonian(
            radial_operator_mode="element-local"
        )
        expected_coordinate1 = serial_hamiltonian.apply_one_body_coordinate1(state)
        expected_coordinate2 = serial_hamiltonian.apply_one_body_coordinate2(state)
        expected_coulomb = serial_hamiltonian.apply_coulomb(state)
        expected_total = expected_coordinate1 + expected_coordinate2 + expected_coulomb
        serial_dipole = CoupledTwoRadialAxialDipoleInteraction.for_diatomic_ion(
            representation,
            parameters,
            gauge="length",
            fields=(pulse,),
        )
        component_errors = {
            "coordinate1": _maximum_difference(
                coordinate1,
                expected_coordinate1,
            ),
            "coordinate2": _maximum_difference(
                coordinate2,
                expected_coordinate2,
            ),
            "coulomb": _maximum_difference(coulomb, expected_coulomb),
            "field": _maximum_difference(field, serial_dipole(state, 0.013)),
            "total": _maximum_difference(total, expected_total),
        }
    component_errors = communicator.bcast(component_errors, root=0)

    full_solver = _solver(
        layout,
        parameters,
        communicator,
        pulse,
        absorbing_boundaries,
        0.0,
        FINAL_TIME,
    )
    start = perf_counter()
    full_result = full_solver.run(local)
    elapsed = perf_counter() - start
    maximum_elapsed = communicator.reduce(elapsed, op=MPI.MAX, root=0)
    full_state = gather_two_radial_2d_state(full_result.state, communicator)

    physical_state = {
        "system_family": "non_bo_diatomic_ion",
        "system": "HD+",
        "symmetry": {"M_tot": 0},
    }
    if arguments.write_reference:
        if size != 1:
            raise ValueError("the reference must be written with one MPI rank")
        half_solver = _solver(
            layout,
            parameters,
            communicator,
            pulse,
            absorbing_boundaries,
            0.0,
            MIDPOINT,
        )
        half_result = half_solver.run(local)
        save_distributed_two_radial_2d_state(
            arguments.checkpoint_archive,
            half_result.state,
            communicator,
            physical_state=physical_state,
            calculation={"time": MIDPOINT, "purpose": "step16-restart"},
            overwrite=arguments.overwrite,
        )
        save_distributed_two_radial_2d_state(
            arguments.reference_archive,
            full_result.state,
            communicator,
            physical_state=physical_state,
            calculation={"time": FINAL_TIME, "purpose": "step16-reference"},
            overwrite=arguments.overwrite,
        )

    communicator.Barrier()
    restarted_initial = load_distributed_two_radial_2d_state(
        arguments.checkpoint_archive,
        layout,
        communicator,
    )
    restart_solver = _solver(
        layout,
        parameters,
        communicator,
        pulse,
        absorbing_boundaries,
        MIDPOINT,
        FINAL_TIME,
    )
    restarted_result = restart_solver.run(restarted_initial)
    restarted_state = gather_two_radial_2d_state(
        restarted_result.state,
        communicator,
    )

    hosts = communicator.gather(socket.gethostname(), root=0)
    local_coefficients = communicator.gather(local.coefficients.size, root=0)
    peak_rss = communicator.gather(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        root=0,
    )

    if rank != 0:
        return

    reference = load_coupled_two_radial_state(arguments.reference_archive)
    reference_state = reference.coefficients
    reference_report = None
    if not arguments.write_reference:
        reference_report = json.loads(
            arguments.reference_json.read_text(encoding="utf-8")
        )
    current_series = _series(full_result)
    reference_series = (
        current_series if reference_report is None else reference_report["time_series"]
    )
    state_error = _maximum_difference(full_state, reference_state)
    restart_error = _maximum_difference(restarted_state, full_state)
    reference_restart_error = _maximum_difference(restarted_state, reference_state)
    time_series_error = max(
        _maximum_difference(current_series[name], reference_series[name])
        for name in (
            "times",
            "norms",
            "cap_norm_losses",
            "energy_numerators",
            "total_dipoles_z",
            "total_l_squared",
        )
    )
    final_cap_norm_loss = current_series["cap_norm_losses"][-1]
    maximum_component_error = max(component_errors.values())
    status = "PASS"
    if (
        max(
            maximum_component_error,
            state_error,
            restart_error,
            reference_restart_error,
            time_series_error,
        )
        >= TOLERANCE
        or final_cap_norm_loss <= 0.0
    ):
        status = "FAIL"

    report = {
        "status": status,
        "mpi_ranks": size,
        "process_grid": list(process_grid),
        "hosts": sorted(set(hosts)),
        "system": "HD+",
        "wf_representation": "coupled",
        "angular_cutoffs": {
            "l1_max": 1,
            "l2_max": 1,
            "L_max": 2,
            "M_tot": 0,
        },
        "global_shape": list(representation.psi_shape),
        "local_coefficients": {
            "minimum": int(min(local_coefficients)),
            "maximum": int(max(local_coefficients)),
        },
        "component_errors": component_errors,
        "maximum_component_error": maximum_component_error,
        "state_difference_from_one_rank": state_error,
        "restart_difference_from_uninterrupted": restart_error,
        "restart_difference_from_one_rank": reference_restart_error,
        "time_series_maximum_difference": time_series_error,
        "final_cap_norm_loss": final_cap_norm_loss,
        "time_series": current_series,
        "replicated_angular_inventory": full_solver.replicated_angular_inventory,
        "hamiltonian_profile": asdict(full_result.hamiltonian_profile),
        "reduction_profile": asdict(full_result.reduction_profile),
        "wall_seconds": float(maximum_elapsed),
        "peak_rss_kib": {
            "minimum": int(min(peak_rss)),
            "maximum": int(max(peak_rss)),
        },
        "checkpoint_archive": str(arguments.checkpoint_archive),
        "reference_archive": str(arguments.reference_archive),
    }
    _write_json(arguments.result_json, report, arguments.overwrite)

    print(f"Coupled Step 16 status:       {status}")
    print(f"MPI ranks:                    {size}")
    print(f"Process grid:                 {process_grid}")
    print(f"Hosts:                        {', '.join(report['hosts'])}")
    print(
        f"Coupled channels/rank:        {full_solver.replicated_angular_inventory['channels']}"
    )
    print(f"Maximum component error:      {maximum_component_error:.3e}")
    print(f"State difference from 1 rank: {state_error:.3e}")
    print(f"Restart difference:           {restart_error:.3e}")
    print(f"Time-series difference:       {time_series_error:.3e}")
    print(f"Final CAP norm loss:          {final_cap_norm_loss:.3e}")
    print(f"Wall time:                    {maximum_elapsed:.6f} s")
    print(f"Result JSON:                  {arguments.result_json}")
    if status != "PASS":
        raise SystemExit("coupled Step 16 qualification exceeded tolerance")


if __name__ == "__main__":
    main()
