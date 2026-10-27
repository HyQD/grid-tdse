"""Distributed Lanczos imaginary-time calculation for a non-BO diatomic ion.

Install the optional MPI support and launch, for example, with::

    mpiexec -n 2 python examples/mpi_non_bo_diatomic_ground_state.py

The internuclear grid has eight FEM-DVR elements and therefore supports at
most eight ranks with the current element-aligned decomposition.
"""

import argparse

import numpy as np
from mpi4py import MPI

from grid_tdse.ground_state import ImaginaryTimeGroundStateSolver
from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributedGroundStateSolver,
    TwoRadialDistributedLayout,
    gather_two_radial_state,
    scatter_two_radial_state,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--system",
    choices=("H2+", "HD+", "D2+"),
    default="H2+",
)
parser.add_argument(
    "--no-serial-comparison",
    action="store_false",
    dest="compare_with_serial",
    help="skip the serial reference after the distributed solve",
)
parser.add_argument(
    "--max-iterations",
    type=int,
    default=4_000,
    help="maximum number of imaginary-time steps (default: 4000)",
)
arguments = parser.parse_args()

SYSTEM = arguments.system
CODATA_SET = "2022"
M_tot = 0
l_max = 2
COMPARE_WITH_SERIAL = arguments.compare_with_serial

IDENTICAL_PAIR = SYSTEM != "HD+"
PAIR_EXCHANGE_PARITY = +1 if IDENTICAL_PAIR else None
SYMMETRY_ADAPTED_BASIS = IDENTICAL_PAIR

TIME_STEP = 0.5
MAX_ITERATIONS = arguments.max_iterations
ENERGY_TOLERANCE = 1.0e-10
STATE_TOLERANCE = 1.0e-8
RESIDUAL_TOLERANCE = 1.0e-8
MAX_KRYLOV_DIMENSION = 20
KRYLOV_RTOL = 1.0e-10
KRYLOV_ATOL = 0.0
MAX_KRYLOV_REFINEMENTS = 8
PROGRESS_INTERVAL = 10

electronic_grid = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 5.0],
    n_points=[5, 5],
)
internuclear_grid = FEMDVRGridSpec(
    nodes=np.linspace(0.0, 8.0, 9),
    n_points=[5] * 8,
)

# This high-level object prepares the physical representation and deterministic
# product state. Its serial solver is not used for the distributed calculation.
calculation = NonBornOppenheimerDiatomicIonGroundState(
    SYSTEM,
    electronic_grid=electronic_grid,
    internuclear_grid=internuclear_grid,
    inversion_parity=+1,
    pair_exchange_parity=PAIR_EXCHANGE_PARITY,
    symmetry_adapted_basis=SYMMETRY_ADAPTED_BASIS,
    l_max=l_max,
    m_max=l_max,
    lambda_max=2 * l_max,
    mu_max=2 * l_max,
    M_tot=M_tot,
    radial_coulomb_method="poisson",
    radial_operator_mode="element-local",
    codata_set=CODATA_SET,
    initial_state="product",
    integrator="sil",
    use_preconditioner=False,
    time_step=TIME_STEP,
    max_iterations=MAX_ITERATIONS,
    energy_tolerance=ENERGY_TOLERANCE,
    state_tolerance=STATE_TOLERANCE,
    residual_tolerance=RESIDUAL_TOLERANCE,
    max_krylov_dimension=MAX_KRYLOV_DIMENSION,
    krylov_rtol=KRYLOV_RTOL,
    krylov_atol=KRYLOV_ATOL,
    max_krylov_refinements=MAX_KRYLOV_REFINEMENTS,
    krylov_reorthogonalize=True,
    progress_interval=PROGRESS_INTERVAL,
)

communicator = MPI.COMM_WORLD
layout = TwoRadialDistributedLayout(
    calculation.representation,
    communicator.size,
)
local_initial_state = scatter_two_radial_state(
    layout,
    communicator,
    calculation.initial_state if communicator.rank == 0 else None,
)
solver = TwoRadialDistributedGroundStateSolver(
    layout,
    communicator,
    time_step=TIME_STEP,
    max_iterations=MAX_ITERATIONS,
    energy_tolerance=ENERGY_TOLERANCE,
    state_tolerance=STATE_TOLERANCE,
    residual_tolerance=RESIDUAL_TOLERANCE,
    max_krylov_dimension=MAX_KRYLOV_DIMENSION,
    krylov_rtol=KRYLOV_RTOL,
    krylov_atol=KRYLOV_ATOL,
    max_krylov_refinements=MAX_KRYLOV_REFINEMENTS,
    krylov_reorthogonalize=True,
    progress_interval=PROGRESS_INTERVAL,
)

# Warm the rank-local and communication paths before timing so first-call
# compilation and setup do not contaminate the solve measurement.
solver.hamiltonian(local_initial_state)

# Synchronize before timing so the reported value is the slowest rank's solve
# time rather than rank zero's local time. Warmup, state gathering, and the
# optional serial reference are deliberately excluded from this measurement.
communicator.Barrier()
distributed_start = MPI.Wtime()
result = solver.solve(local_initial_state)
local_distributed_time = MPI.Wtime() - distributed_start
distributed_time = communicator.allreduce(
    local_distributed_time,
    op=MPI.MAX,
)
state = gather_two_radial_state(result.state, communicator)

if communicator.rank == 0:
    last = result.history[-1]
    print(f"MPI ranks:                {communicator.size}")
    print(f"System:                   {SYSTEM}")
    print(f"Converged:                {result.converged}")
    print(f"Iterations:               {result.iterations}")
    print(f"Energy:                   {result.energy:.12f} Eh")
    print(f"Energy change:            {last.energy_change:.3e} Eh")
    print(f"State change:             {result.state_change:.3e}")
    print(f"Eigenstate residual:      {result.residual_norm:.3e}")
    print(f"Norm:                     {result.norm:.12f}")
    print(f"Distributed solve time:   {distributed_time:.6f} s")

    if COMPARE_WITH_SERIAL:
        serial_solver = ImaginaryTimeGroundStateSolver(
            integrator="sil",
            time_step=TIME_STEP,
            max_iterations=MAX_ITERATIONS,
            energy_tolerance=ENERGY_TOLERANCE,
            state_tolerance=STATE_TOLERANCE,
            residual_tolerance=RESIDUAL_TOLERANCE,
            max_krylov_dimension=MAX_KRYLOV_DIMENSION,
            krylov_rtol=KRYLOV_RTOL,
            krylov_atol=KRYLOV_ATOL,
            max_krylov_refinements=MAX_KRYLOV_REFINEMENTS,
            krylov_reorthogonalize=True,
            progress_interval=None,
        )
        serial_start = MPI.Wtime()
        serial = serial_solver.solve(
            calculation.initial_state,
            calculation.hamiltonian,
            calculation.representation.inner_product,
        )
        serial_time = MPI.Wtime() - serial_start
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
        print(f"Serial energy difference: {abs(result.energy - serial.energy):.3e} Eh")
        print(f"Serial state difference:  {difference_norm:.3e}")
        print(f"Serial solve time:        {serial_time:.6f} s")
        print(f"Serial/MPI time ratio:    {serial_time / distributed_time:.6f}")
