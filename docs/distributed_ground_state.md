# Distributed non-BO diatomic ground states

`TwoRadialDistributedGroundStateSolver` connects the distributed two-radial
state, field-free Hamiltonian, and physical reductions to short-iterative
Lanczos imaginary-time propagation.

The implementation targets the first complete MPI calculation for
non-Born--Oppenheimer diatomic ions. It supports FEM-DVR, the element-aligned
distribution of the internuclear coordinate `R`, and both full and compact
fixed-`M_tot` angular storage.

## Algorithm

Every rank holds a `DistributedTwoRadialState` containing all angular
channels, all electronic radial points, and only its permanently owned
internuclear radial points. During each Lanczos imaginary-time step:

1. The generic representation-independent ground-state solver advances each
   rank's local coefficient array.
2. Hamiltonian calls are forwarded to `TwoRadialDistributedHamiltonian`.
3. Neighboring ranks exchange the FEM-DVR bridge data needed by the
   internuclear kinetic-energy action.
4. Every Lanczos inner product uses `TwoRadialDistributedReductions` and a
   collective sum.
5. The state is normalized with the global physical norm.
6. Energy change, phase-invariant state change, and eigenstate residual are
   evaluated collectively.

All ranks receive the same reduced scalars and therefore make the same
Lanczos-dimension, refinement, and convergence decisions. The root rank alone
prints iteration progress.

The implementation reuses `ImaginaryTimeGroundStateSolver` and
`integrator-lib`; it does not contain a separate MPI implementation of the
Lanczos algorithm.

## Installation and execution

MPI support is optional because MPI is not required for serial calculations:

```bash
python -m pip install -e ".[mpi]"
```

The `mpi4py` installation must be compatible with the MPI implementation used
to launch the program. On a cluster, load the intended compiler and MPI
modules before installing `mpi4py` in the environment.

Run the example with, for example:

```bash
mpiexec -n 2 python examples/mpi_non_bo_diatomic_ground_state.py
```

All ranks must call `solve` collectively and in the same control flow. The
current `R` decomposition requires at least one FEM-DVR element per rank.

The example:

- builds an H2+ representation with `M_tot=0`;
- scatters the deterministic product state from rank zero;
- performs distributed SIL imaginary-time propagation;
- gathers the final state on rank zero;
- compares its energy and phase-aligned state with a serial SIL calculation.

Set `COMPARE_WITH_SERIAL=False` when the additional serial calculation is not
desired.

## Interface

```python
from mpi4py import MPI
from grid_tdse.spherical.two_radial import (
    TwoRadialDistributedGroundStateSolver,
    TwoRadialDistributedLayout,
    scatter_two_radial_state,
)

comm = MPI.COMM_WORLD
layout = TwoRadialDistributedLayout(representation, comm.size)
local_state = scatter_two_radial_state(
    layout,
    comm,
    initial_state if comm.rank == 0 else None,
)
solver = TwoRadialDistributedGroundStateSolver(
    layout,
    comm,
    time_step=0.25,
    max_iterations=2000,
    max_krylov_dimension=20,
)
result = solver.solve(local_state)
```

`result.state` is the permanently owned local state. Its energies, norms,
state changes, residuals, and Krylov diagnostics are global quantities and are
the same on every rank. Use `gather_two_radial_state` when a serial global
array is needed on one rank.

## Local validation

The test suite uses a thread-based communicator to execute the actual
neighbor exchange and collective-reduction paths without requiring a local
MPI installation. One-, two-, and three-rank results are compared with serial
Lanczos imaginary-time propagation. The checks cover:

- assembled state coefficients;
- energy histories;
- physical normalization;
- phase-invariant state changes;
- eigenstate residuals;
- Krylov dimensions;
- convergence from a represented eigenstate.

Actual MPI deployment and correctness validation on Saga are a separate next
step. Performance conclusions must not be drawn from the thread-based local
tests.

## Current scope

The distributed ground-state driver supports SIL for the Hermitian field-free
Hamiltonian. It does not support Crank--Nicolson, SIA, CAPs, masks,
time-dependent fields, or distributed checkpoint histories. Distributed
real-time SIL and SIA are subsequent implementation steps.
