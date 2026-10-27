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

### Local Conda MPI installation

Use one MPI implementation consistently. The following procedure installs
Open MPI and its matching compiler inside an existing Conda environment. It
avoids accidentally compiling `mpi4py` with a system MPI implementation and
then launching it with a different `mpiexec`.

Activate the intended environment and install Open MPI and the Conda C
compiler from the same channel:

```bash
conda activate <environment-name>
conda install -c conda-forge openmpi c-compiler
hash -r
```

Both MPI commands must now come from the active environment:

```bash
command -v mpicc
command -v mpiexec
```

Their paths should begin with `${CONDA_PREFIX}/bin/`. If `mpi4py` was
previously installed against another MPI implementation, remove it before
rebuilding:

```bash
python -m pip uninstall -y mpi4py
```

Build `mpi4py` against the Open MPI wrapper from the active environment:

```bash
MPICC="${CONDA_PREFIX}/bin/mpicc" \
python -m pip install --no-cache-dir --no-binary=mpi4py "mpi4py>=4,<5"
```

The `c-compiler` package is required because Conda's Open MPI wrapper calls a
Conda compiler such as `x86_64-conda-linux-gnu-cc`. An error saying that the
wrapper cannot find this compiler indicates that `c-compiler` is missing.

Confirm which MPI library `mpi4py` loaded:

```bash
python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'
```

Then verify both process startup and collective communication on two local
ranks:

```bash
mpiexec --mca btl self,sm -n 2 python -c \
'from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank, c.allreduce(c.rank + 1))'
```

Both ranks should print the reduced value `3`. The Open MPI option
`--mca btl self,sm` deliberately restricts this local, single-node check to
self and shared-memory transports. Do not use that restriction for a
multi-node calculation or a Saga job.

Finally, check the Python environment independently of MPI execution:

```bash
python -m pip check
```

Do not mix a Conda `mpicc` with `/usr/bin/mpiexec`, or a system `mpicc` with a
Conda `mpiexec`. Successful import of `mpi4py` alone is insufficient: the
two-rank collective test must also succeed.

### Run the distributed ground-state example

From the `grid-tdse` repository root, run the H2+ example locally with two MPI
ranks and one numerical-library thread per rank:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
mpiexec --mca btl self,sm -n 2 \
python examples/mpi_non_bo_diatomic_ground_state.py
```

The same local validation can be repeated with `-n 1` and `-n 4`. The example
prints the distributed energy, norm, state change, and eigenstate residual. On
rank zero it also performs a serial calculation and reports the energy and
phase-aligned state differences. The distributed calculation has been
validated locally with one, two, and four MPI ranks.

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
step. Performance conclusions must not be drawn from the thread-based tests or
the small local MPI validation.

## Current scope

The distributed ground-state driver supports SIL for the Hermitian field-free
Hamiltonian. It does not support Crank--Nicolson, SIA, CAPs, masks,
time-dependent fields, or distributed checkpoint histories. Distributed
real-time SIL and SIA are subsequent implementation steps.
