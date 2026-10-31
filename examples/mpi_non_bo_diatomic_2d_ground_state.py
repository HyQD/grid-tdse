"""Cartesian MPI SIL/SIA ground state for a non-BO diatomic ion.

For example, run a four-rank ``2 x 2`` SIL calculation with::

    mpiexec -n 4 python examples/mpi_non_bo_diatomic_2d_ground_state.py \
        --process-grid 2x2 --max-iterations 20

The process-grid dimensions must multiply to the MPI size. Omitting
``--process-grid`` lets the layout select a valid factorization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from grid_tdse.ground_state import ImaginaryTimeGroundStateSolver
from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributed2DGroundStateSolver,
    TwoRadialDistributed2DLayout,
    gather_two_radial_2d_state,
    load_distributed_two_radial_2d_state,
    load_two_radial_state,
    scatter_two_radial_2d_state,
)


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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), default="H2+")
parser.add_argument("--integrator", choices=("sil", "sia"), default="sil")
parser.add_argument(
    "--process-grid",
    type=_process_grid,
    help="Cartesian MPI process grid P_r x P_R; default: automatic",
)
parser.add_argument("--l-max", type=int, default=2)
parser.add_argument("--time-step", type=float, default=0.5)
parser.add_argument("--max-iterations", type=int, default=4_000)
parser.add_argument("--sample-interval", type=int, default=10)
parser.add_argument(
    "--initial-state",
    type=Path,
    help="optional compatible serial two-radial state archive",
)
parser.add_argument(
    "--log-file",
    help="optional root-rank progress logfile; must not already exist",
)
parser.add_argument(
    "--no-serial-comparison",
    action="store_false",
    dest="compare_with_serial",
    help="do not gather the final state or run the serial reference",
)
arguments = parser.parse_args()

if arguments.l_max < 0:
    parser.error("--l-max must be nonnegative")
if arguments.time_step <= 0.0:
    parser.error("--time-step must be positive")
if arguments.max_iterations <= 0:
    parser.error("--max-iterations must be positive")
if arguments.sample_interval <= 0:
    parser.error("--sample-interval must be positive")

communicator = MPI.COMM_WORLD
if (
    arguments.process_grid is not None
    and np.prod(arguments.process_grid) != communicator.size
):
    parser.error(
        "--process-grid dimensions must multiply to the number of MPI ranks"
    )

system_name = arguments.system
identical_pair = system_name != "HD+"
l_max = arguments.l_max
m_max = l_max
lambda_max = 2 * l_max
mu_max = lambda_max

energy_tolerance = 1.0e-10
state_tolerance = 1.0e-8
residual_tolerance = 1.0e-8
max_krylov_dimension = 20
krylov_rtol = 1.0e-10
krylov_atol = 0.0
max_krylov_refinements = 8

electronic_grid = FEMDVRGridSpec(
    nodes=[0.0, 4.0, 8.0, 12.0, 16.0],
    n_points=[11, 11, 11, 11],
)
internuclear_grid = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 4.0, 6.0, 8.0],
    n_points=[21, 21, 21, 21],
)

setup_started = MPI.Wtime()
calculation = NonBornOppenheimerDiatomicIonGroundState(
    system_name,
    electronic_grid=electronic_grid,
    internuclear_grid=internuclear_grid,
    inversion_parity=+1,
    pair_exchange_parity=+1 if identical_pair else None,
    symmetry_adapted_basis=identical_pair,
    l_max=l_max,
    m_max=m_max,
    lambda_max=lambda_max,
    mu_max=mu_max,
    M_tot=0,
    radial_coulomb_method="poisson",
    radial_operator_mode="element-local",
    codata_set="2022",
    initial_state="product",
    integrator=arguments.integrator,
    use_preconditioner=False,
    time_step=arguments.time_step,
    max_iterations=arguments.max_iterations,
    energy_tolerance=energy_tolerance,
    state_tolerance=state_tolerance,
    residual_tolerance=residual_tolerance,
    max_krylov_dimension=max_krylov_dimension,
    krylov_rtol=krylov_rtol,
    krylov_atol=krylov_atol,
    max_krylov_refinements=max_krylov_refinements,
    krylov_reorthogonalize=True,
    progress_interval=arguments.sample_interval,
)
layout = TwoRadialDistributed2DLayout(
    calculation.representation,
    communicator.size,
    process_grid=arguments.process_grid,
)
local_setup_time = MPI.Wtime() - setup_started
setup_time = communicator.allreduce(local_setup_time, op=MPI.MAX)

if arguments.initial_state is None:
    serial_initial_state = calculation.initial_state
    local_initial_state = scatter_two_radial_2d_state(
        layout,
        communicator,
        calculation.initial_state if communicator.rank == 0 else None,
    )
else:
    serial_initial_state = None
    if communicator.rank == 0:
        archived = load_two_radial_state(arguments.initial_state)
        archived.validate_representation(calculation.representation)
        serial_initial_state = archived.coefficients
    local_initial_state = load_distributed_two_radial_2d_state(
        arguments.initial_state,
        layout,
        communicator,
    )

solver = TwoRadialDistributed2DGroundStateSolver(
    layout,
    communicator,
    time_step=arguments.time_step,
    max_iterations=arguments.max_iterations,
    energy_tolerance=energy_tolerance,
    state_tolerance=state_tolerance,
    residual_tolerance=residual_tolerance,
    integrator=arguments.integrator,
    max_krylov_dimension=max_krylov_dimension,
    krylov_rtol=krylov_rtol,
    krylov_atol=krylov_atol,
    max_krylov_refinements=max_krylov_refinements,
    krylov_reorthogonalize=True,
    progress_interval=arguments.sample_interval,
    observable_evaluator=calculation.observable_evaluator,
    sample_interval=arguments.sample_interval,
    log_path=arguments.log_file,
    profile=True,
)

communicator.Barrier()
warmup_started = MPI.Wtime()
solver.hamiltonian(local_initial_state)
local_warmup_time = MPI.Wtime() - warmup_started
warmup_time = communicator.allreduce(local_warmup_time, op=MPI.MAX)
solver.hamiltonian.reset_profile()
solver.reductions.reset_profile()

communicator.Barrier()
solve_started = MPI.Wtime()
result = solver.solve(local_initial_state)
local_solve_time = MPI.Wtime() - solve_started
solve_time = communicator.allreduce(local_solve_time, op=MPI.MAX)

hamiltonian_profile = result.hamiltonian_profile
reduction_profile = result.reduction_profile
profile_names = (
    "hamiltonian_calls",
    "coordinate1_seconds",
    "coordinate2_seconds",
    "coulomb_seconds",
    "coordinate1_halo_seconds",
    "coordinate2_halo_seconds",
    "coordinate1_weak_seconds",
    "coordinate2_weak_seconds",
    "local_inner_product_calls",
    "local_inner_product_seconds",
    "collective_calls",
    "collective_values",
    "collective_seconds",
)
local_profile = np.asarray(
    [
        hamiltonian_profile.hamiltonian_calls,
        hamiltonian_profile.coordinate1_seconds,
        hamiltonian_profile.coordinate2_seconds,
        hamiltonian_profile.coulomb_seconds,
        hamiltonian_profile.coordinate1_halo_seconds,
        hamiltonian_profile.coordinate2_halo_seconds,
        hamiltonian_profile.coordinate1_weak_seconds,
        hamiltonian_profile.coordinate2_weak_seconds,
        reduction_profile.local_inner_product_calls,
        reduction_profile.local_inner_product_seconds,
        reduction_profile.collective_calls,
        reduction_profile.collective_values,
        reduction_profile.collective_seconds,
    ],
    dtype=float,
)
maximum_profile_values = np.empty_like(local_profile)
communicator.Allreduce(local_profile, maximum_profile_values, op=MPI.MAX)
maximum_profile = dict(zip(profile_names, maximum_profile_values, strict=True))

state = None
if arguments.compare_with_serial:
    state = gather_two_radial_2d_state(result.state, communicator)

if communicator.rank == 0:
    last = result.history[-1]
    local_sizes = np.asarray(
        [np.prod(layout.owned_shape(rank)) for rank in range(layout.n_ranks)],
        dtype=np.int64,
    )
    print(f"MPI ranks:                {communicator.size}")
    print(f"Process grid:             {layout.process_grid}")
    print(f"System:                   {system_name}")
    print(f"Integrator:               {arguments.integrator.upper()}")
    print(f"Initial state:            {arguments.initial_state or 'generated product'}")
    print(f"l_max:                    {l_max}")
    print(f"m_max:                    {m_max}")
    print(f"lambda_max:               {lambda_max}")
    print(f"mu_max:                   {mu_max}")
    print(f"Global coefficients:      {np.prod(layout.global_shape)}")
    print(
        "Local coefficients:       "
        f"{local_sizes.min()}--{local_sizes.max()}"
    )
    print(f"Converged:                {result.converged}")
    print(f"Iterations:               {result.iterations}")
    print(f"Energy:                   {result.energy:.12f} Eh")
    print(f"Energy change:            {last.energy_change:.3e} Eh")
    print(f"State change:             {result.state_change:.3e}")
    print(f"Eigenstate residual:      {result.residual_norm:.3e}")
    print(f"Norm:                     {result.norm:.12f}")
    print(f"Representation setup:     {setup_time:.6f} s")
    print(f"Hamiltonian warmup:       {warmup_time:.6f} s")
    print(f"Distributed solve time:   {solve_time:.6f} s")
    print(
        "Solve time per iteration: "
        f"{solve_time / result.iterations:.6f} s"
    )
    print(
        "Hamiltonian calls:        "
        f"{int(maximum_profile['hamiltonian_calls'])}"
    )
    print(
        "Krylov operator calls:    "
        f"{int(np.sum(result.krylov_operator_calls))}"
    )
    coordinate1_exchange_seconds = (
        maximum_profile["coordinate1_halo_seconds"]
        + maximum_profile["coordinate1_weak_seconds"]
    )
    coordinate2_exchange_seconds = (
        maximum_profile["coordinate2_halo_seconds"]
        + maximum_profile["coordinate2_weak_seconds"]
    )
    print(
        "Coordinate-1 work (max):  "
        f"{maximum_profile['coordinate1_seconds']:.6f} s"
    )
    print(
        "Coordinate-2 work (max):  "
        f"{maximum_profile['coordinate2_seconds']:.6f} s"
    )
    print(
        "Coulomb work (max):       "
        f"{maximum_profile['coulomb_seconds']:.6f} s"
    )
    print(
        "Coordinate-1 exchange:    "
        f"{coordinate1_exchange_seconds:.6f} s"
    )
    print(
        "Coordinate-2 exchange:    "
        f"{coordinate2_exchange_seconds:.6f} s"
    )
    print(
        "Local reductions (max):   "
        f"{maximum_profile['local_inner_product_seconds']:.6f} s "
        f"({int(maximum_profile['local_inner_product_calls'])} values)"
    )
    print(
        "Collective reductions (max): "
        f"{maximum_profile['collective_seconds']:.6f} s "
        f"({int(maximum_profile['collective_calls'])} calls, "
        f"{int(maximum_profile['collective_values'])} values)"
    )

    if arguments.compare_with_serial:
        serial_solver = ImaginaryTimeGroundStateSolver(
            integrator=arguments.integrator,
            time_step=arguments.time_step,
            max_iterations=arguments.max_iterations,
            energy_tolerance=energy_tolerance,
            state_tolerance=state_tolerance,
            residual_tolerance=residual_tolerance,
            max_krylov_dimension=max_krylov_dimension,
            krylov_rtol=krylov_rtol,
            krylov_atol=krylov_atol,
            max_krylov_refinements=max_krylov_refinements,
            krylov_reorthogonalize=True,
            progress_interval=None,
        )
        serial_started = MPI.Wtime()
        serial = serial_solver.solve(
            serial_initial_state,
            calculation.hamiltonian,
            calculation.representation.inner_product,
        )
        serial_time = MPI.Wtime() - serial_started
        overlap = calculation.representation.inner_product(serial.state, state)
        aligned = (
            state
            if abs(overlap) == 0.0
            else state * overlap.conjugate() / abs(overlap)
        )
        difference = aligned - serial.state
        difference_norm = np.sqrt(
            calculation.representation.inner_product(
                difference,
                difference,
            ).real
        )
        print(
            "Serial energy difference: "
            f"{abs(result.energy - serial.energy):.3e} Eh"
        )
        print(f"Serial state difference:  {difference_norm:.3e}")
        print(f"Serial solve time:        {serial_time:.6f} s")
