"""Validate Cartesian-MPI length-gauge dynamics against a serial reference.

The initial Davidson state is loaded or computed on rank zero and scattered
to the selected Cartesian process grid.  Propagation itself retains only the
permanently owned radial tile on each rank.

For example::

    mpiexec -n 4 python examples/mpi_non_bo_diatomic_2d_dynamics.py \
        --process-grid 2x2 \
        --ground-state-path phase9d_h2p_ground.npz \
        --initial-state-policy compute-if-missing \
        --output-json phase9d_2x2.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isfinite, pi
from pathlib import Path

import numpy as np
from mpi4py import MPI

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    DiatomicIonDynamicsSamplingPolicy,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    TwoRadialDistributed2DLayout,
    TwoRadialDistributed2DRealTimeSolver,
    TwoRadialSystemSpec,
    gather_two_radial_2d_state,
    load_two_radial_state,
    phase_aligned_two_radial_state_difference,
    prepare_distributed_non_bo_diatomic_dynamics_initial_state,
    radial_potential,
)
from grid_tdse.spherical.two_radial.diatomic import (
    _diatomic_parameters,
    _pair_exchange_parity,
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


def _process_grid(value: str) -> tuple[int, int]:
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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), default="H2+")
    parser.add_argument("--process-grid", type=_process_grid)
    parser.add_argument("--l-max", type=int, default=1)
    parser.add_argument(
        "--electronic-nodes", type=float, nargs="+", default=(0.0, 1.0, 2.0)
    )
    parser.add_argument("--electronic-points-per-element", type=int, default=3)
    parser.add_argument(
        "--internuclear-nodes", type=float, nargs="+", default=(0.0, 1.0, 2.0)
    )
    parser.add_argument("--internuclear-points-per-element", type=int, default=3)
    parser.add_argument("--codata-set", choices=("2018", "2022"), default="2022")
    parser.add_argument("--ground-state-path", type=Path, required=True)
    parser.add_argument(
        "--initial-state-policy",
        choices=("require", "compute-if-missing", "recompute"),
        default="require",
    )
    parser.add_argument("--integrator", choices=("sil", "sia"), default="sil")
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=0.04)
    parser.add_argument("--field-strength", type=float, default=0.02)
    parser.add_argument("--carrier-cycles", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-11)
    parser.add_argument("--krylov-atol", type=float, default=1.0e-13)
    parser.add_argument("--output-json", type=Path, default=Path("phase9d.json"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-serial-comparison",
        action="store_false",
        dest="serial_comparison",
    )
    arguments = parser.parse_args()
    if arguments.l_max < 1:
        parser.error("--l-max must be at least 1 for length-gauge dynamics")
    if arguments.time_step <= 0.0 or not isfinite(arguments.time_step):
        parser.error("--time-step must be positive")
    if arguments.final_time <= 0.0 or not isfinite(arguments.final_time):
        parser.error("--final-time must be positive")
    if arguments.sample_interval <= 0:
        parser.error("--sample-interval must be positive")
    if arguments.krylov_rtol <= 0.0 or arguments.krylov_atol < 0.0:
        parser.error("invalid Krylov tolerance")
    for option, nodes in (
        ("electronic", arguments.electronic_nodes),
        ("internuclear", arguments.internuclear_nodes),
    ):
        if len(nodes) < 2 or nodes[0] != 0.0:
            parser.error(f"--{option}-nodes must start at zero")
        if any(right <= left for left, right in zip(nodes, nodes[1:])):
            parser.error(f"--{option}-nodes must be strictly increasing")
    return arguments


def _grid(nodes, points_per_element):
    return FEMDVRGridSpec(
        nodes,
        [points_per_element] * (len(nodes) - 1),
    )


def _representation(arguments, electronic_grid, internuclear_grid):
    parameters = _diatomic_parameters(arguments.system, arguments.codata_set)
    pair_parity = _pair_exchange_parity(
        1 if parameters.has_identical_pair else None,
        parameters,
    )
    electron_basis = AngularBasisSpec(arguments.l_max, arguments.l_max)
    pair_basis = AngularBasisSpec(
        arguments.l_max,
        arguments.l_max,
        coordinate_parity=pair_parity if parameters.has_identical_pair else None,
    )
    specification = TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=electronic_grid,
            basis=electron_basis,
            kinetic_prefactor=-0.5 / parameters.electronic_reduced_mass,
        ),
        coordinate2=CoordinateSpec(
            grid=internuclear_grid,
            basis=pair_basis,
            kinetic_prefactor=-0.5 / parameters.internuclear_reduced_mass,
            one_body_terms=(
                radial_potential(
                    lambda radius: parameters.reference_charge
                    * parameters.partner_charge
                    / radius
                ),
            ),
        ),
        multipoles=CoulombMultipoleSpec(
            2 * arguments.l_max,
            2 * arguments.l_max,
            lambda_parity=1 if parameters.has_identical_pair else None,
        ),
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
        radial_coulomb_method="poisson",
    )
    return parameters, pair_parity, specification.build()


def _maximum_time_series_difference(left, right):
    maximum = 0.0
    for left_record, right_record in zip(left.records(), right.records(), strict=True):
        if left_record.keys() != right_record.keys():
            raise ValueError("serial and MPI time-series schemas differ")
        for name, left_value in left_record.items():
            if name.startswith("propagation."):
                continue
            right_value = right_record[name]
            if isinstance(left_value, (int, float)) and not isinstance(
                left_value, bool
            ):
                maximum = max(maximum, abs(float(left_value) - float(right_value)))
            elif left_value != right_value:
                raise ValueError(f"time-series field {name!r} differs")
    return maximum


arguments = _arguments()
communicator = MPI.COMM_WORLD
if arguments.process_grid is not None:
    if np.prod(arguments.process_grid) != communicator.size:
        raise ValueError("process-grid dimensions must multiply to MPI size")
if arguments.output_json.exists() and not arguments.overwrite:
    raise FileExistsError(
        f"output JSON exists: {arguments.output_json}; pass --overwrite"
    )

electronic_grid = _grid(
    arguments.electronic_nodes,
    arguments.electronic_points_per_element,
)
internuclear_grid = _grid(
    arguments.internuclear_nodes,
    arguments.internuclear_points_per_element,
)
parameters, pair_parity, representation = _representation(
    arguments,
    electronic_grid,
    internuclear_grid,
)
layout = TwoRadialDistributed2DLayout(
    representation,
    communicator.size,
    process_grid=arguments.process_grid,
)
initialization = prepare_distributed_non_bo_diatomic_dynamics_initial_state(
    layout,
    communicator,
    system=arguments.system,
    electronic_grid=electronic_grid,
    internuclear_grid=internuclear_grid,
    pair_exchange_parity=pair_parity,
    archive_path=arguments.ground_state_path,
    policy=arguments.initial_state_policy,
    symmetry_adapted_basis=parameters.has_identical_pair,
    codata_set=arguments.codata_set,
    radial_coulomb_method="poisson",
    ground_state_options={
        "max_iterations": 500,
        "residual_tolerance": 1.0e-8,
        "maximum_subspace_dimension": 30,
        "retained_ritz_vectors": 4,
        "progress_interval": 10,
        "sample_interval": 10,
    },
)
pulse = SineSquaredElectricField(
    arguments.field_strength,
    arguments.final_time,
    arguments.carrier_cycles,
)
sampling_policy = DiatomicIonDynamicsSamplingPolicy(interval=arguments.sample_interval)
solver = TwoRadialDistributed2DRealTimeSolver(
    layout,
    parameters,
    communicator,
    fields=(pulse,),
    final_time=arguments.final_time,
    time_step=arguments.time_step,
    integrator=arguments.integrator,
    max_krylov_dimension=20,
    krylov_rtol=arguments.krylov_rtol,
    krylov_atol=arguments.krylov_atol,
    max_krylov_refinements=8,
    sampling_policy=sampling_policy,
    progress_interval=10,
    profile=True,
)

communicator.Barrier()
started = MPI.Wtime()
distributed = solver.run(initialization.state)
local_wall_time = MPI.Wtime() - started
wall_time = communicator.allreduce(local_wall_time, op=MPI.MAX)
global_state = (
    gather_two_radial_2d_state(distributed.state, communicator)
    if arguments.serial_comparison
    else None
)

state_difference = None
time_series_difference = None
serial_wall_time = None
passed = max(abs(distributed.norm_history - distributed.norm_history[0])) < 1.0e-9
if communicator.rank == 0 and arguments.serial_comparison:
    archive = load_two_radial_state(arguments.ground_state_path)
    serial = NonBornOppenheimerDiatomicIonTDSE(
        arguments.system,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        initial_state=archive,
        pair_exchange_parity=pair_parity,
        fields=(pulse,),
        field_gauge="length",
        l_max=arguments.l_max,
        m_max=arguments.l_max,
        lambda_max=2 * arguments.l_max,
        mu_max=2 * arguments.l_max,
        M_tot=0,
        codata_set=arguments.codata_set,
        radial_coulomb_method="poisson",
        radial_operator_mode="element-local",
        integrator=arguments.integrator,
        integrator_options={
            "max_krylov_dimension": 20,
            "rtol": arguments.krylov_rtol,
            "atol": arguments.krylov_atol,
            "max_refinements": 8,
        },
        final_time=arguments.final_time,
        time_step=arguments.time_step,
        observables=None,
        dynamics_sampling_policy=sampling_policy,
        progress_interval=None,
    )
    serial_started = MPI.Wtime()
    serial_result = serial.run(progress=False)
    serial_wall_time = MPI.Wtime() - serial_started
    state_difference = phase_aligned_two_radial_state_difference(
        serial_result.state,
        global_state,
        representation.inner_product,
    )
    time_series_difference = _maximum_time_series_difference(
        serial.dynamics_time_series,
        distributed.time_series,
    )
    passed = passed and state_difference < 1.0e-8
    passed = passed and time_series_difference < 1.0e-8

passed = communicator.bcast(passed if communicator.rank == 0 else None, root=0)
if communicator.rank == 0:
    report = {
        "phase": "9d",
        "status": "PASS" if passed else "FAIL",
        "system": arguments.system,
        "mpi_ranks": communicator.size,
        "process_grid": layout.process_grid,
        "integrator": arguments.integrator,
        "n_steps": distributed.n_steps,
        "norm": distributed.norm,
        "maximum_norm_drift": float(
            np.max(abs(distributed.norm_history - distributed.norm_history[0]))
        ),
        "state_difference": state_difference,
        "time_series_maximum_difference": time_series_difference,
        "distributed_wall_time": wall_time,
        "serial_wall_time": serial_wall_time,
        "initialization": initialization.metadata,
        "time_series": distributed.time_series.records(),
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nPhase 9d Cartesian-MPI length-gauge validation")
    print(f"Status:                         {report['status']}")
    print(f"MPI ranks:                      {communicator.size}")
    print(f"Process grid:                   {layout.process_grid}")
    print(f"Integrator:                     {arguments.integrator}")
    print(f"Accepted steps:                 {distributed.n_steps}")
    print(f"Norm:                           {distributed.norm:.12f}")
    print(f"Maximum norm drift:             {report['maximum_norm_drift']:.3e}")
    if state_difference is not None:
        print(f"Serial state difference:        {state_difference:.3e}")
        print(f"Time-series maximum difference: {time_series_difference:.3e}")
    print(f"Distributed wall time:          {wall_time:.6f} s")
    print(f"Result JSON:                    {arguments.output_json}")

if not passed:
    raise SystemExit(1)
