# Distributed two-radial Hamiltonian and reductions

`TwoRadialDistributedHamiltonian` applies the field-free two-radial
Hamiltonian to `DistributedTwoRadialState` objects. Either radial FEM-DVR
coordinate can be distributed. Coordinate 2 (`R` for a non-BO diatomic ion)
remains the default, while coordinate 1 (`r`) can be selected to compare the
two one-dimensional layouts on unequal grids.

## Local and communicated actions

The Hamiltonian separates naturally into three parts:

- The one-body term for the unpartitioned coordinate is rank-local.
- The Coulomb multipoles are point-local in both radial DVR coordinates. Each
  rank applies the optimized angular coupling kernel only at its owned radial
  points.
- The selected coordinate's diagonal terms are local, while its FEM-DVR
  second derivative requires neighboring bridge data and assembly.

The distributed-coordinate derivative follows the weak-form finite-element
assembly used by `grid-lib`:

1. A rank receives its left bridge state from the rank that permanently owns
   that bridge.
2. Every local element applies its negative stiffness matrix.
3. Contributions from elements on the same rank are scatter-added locally.
4. The right rank sends its weak-form contribution at the shared bridge back
   to the left owner.
5. The owner adds both element contributions and applies the inverse assembled
   mass at each owned point.
6. The kinetic prefactor and local diagonal potential are applied.

Only neighboring ranks exchange data. Permanent state storage continues to
contain every bridge exactly once.

The optimized one-dimensional action starts the bridge-state receive before
applying the unpartitioned-coordinate and Coulomb terms. It then evaluates
the interior elements along the selected coordinate while the receive is in
flight and waits only before the boundary element needs the halo. The returned weak bridge contribution is
also transferred nonblockingly while the owned contribution and diagonal
terms are finalized. Communication timings therefore measure request setup
and the non-overlapped wait, rather than all computation performed while a
request is active.

Rank-local Coulomb multipole slices and the Hamiltonian's halo, weak-action,
and exchange work arrays are allocated once and reused. The complete action is
accumulated directly into the unpartitioned-coordinate output instead of
allocating another total-output array. The public result remains an
independent distributed state; the persistent arrays are an internal
implementation detail.

## MPI interface

The distributed Hamiltonian accepts a communicator that supplies `rank`,
`size`, `Irecv`, and `Isend`. The reduction helper additionally uses
buffer-based `Allreduce` when it is available. An `mpi4py.MPI.Comm` provides
this interface. Communication uses NumPy buffers and nonblocking
point-to-point operations.

```python
from mpi4py import MPI
from grid_tdse.spherical.two_radial import (
    TwoRadialDistributedHamiltonian,
    TwoRadialDistributedLayout,
)

comm = MPI.COMM_WORLD
layout = TwoRadialDistributedLayout(
    representation,
    comm.size,
    partitioned_coordinate=1,  # omit to retain the coordinate-2 default
)
hamiltonian = TwoRadialDistributedHamiltonian(layout, comm)

local_Hpsi = hamiltonian(local_psi)
```

`hamiltonian.components(local_psi)` returns the two one-coordinate actions and
the Coulomb action separately. This is useful for profiling and numerical
diagnostics.

## Local multi-rank validation

`hamiltonian.apply_all(local_states)` emulates all rank-local actions and
bridge assembly in one process. It does not assemble the input state and apply
the serial Hamiltonian internally. This makes it suitable for deterministic
equivalence testing without MPI:

```python
local_states = layout.distribute(psi)
local_actions = hamiltonian.apply_all(local_states)
Hpsi = layout.assemble(local_actions)
```

The benchmark
`benchmarks/non_bo_diatomic_distributed_hamiltonian.py` compares this path
with the serial element-local Hamiltonian. It can use a reproducible complex
state or a compatible archived ground state. The emulated timings are not MPI
scaling measurements.

The benchmark accepts `--partition-coordinate r` and
`--partition-coordinate R`. Running both choices with the same state provides
a lightweight unequal-grid comparison before submitting MPI measurements:

```bash
python benchmarks/non_bo_diatomic_distributed_hamiltonian.py \
    --ranks 2 --partition-coordinate r
python benchmarks/non_bo_diatomic_distributed_hamiltonian.py \
    --ranks 2 --partition-coordinate R
```

## Physical reductions

`TwoRadialDistributedReductions` evaluates each physical inner product by
integrating only permanently owned points along the selected coordinate and
then summing the local contributions with `allreduce`. Bridge points therefore
occur exactly once.

The available operations include:

- `local_inner_product(left, right)` for one rank's contribution;
- `local_inner_products(left_states, right)` for several local contributions;
- `inner_product(left, right)` for the collective physical inner product;
- `inner_products(left_states, right)` for one batched collective reduction;
- `norm_squared(state)` and `norm(state)`;
- `expectation(state, operator_state)`;
- `rayleigh_quotient(state, operator_state)`;
- `residual_norm(state, operator_state, eigenvalue)`.

`expectation` returns the unnormalized value. It does not divide by the state
norm, which is important after a mask or absorbing boundary has reduced the
norm. `rayleigh_quotient` is the explicit normalized alternative. Recording
the norm alongside observables permits any desired normalization during
post-processing.

`inner_product_all` supplies the corresponding local multi-rank validation
path without a communicator.

The batched operation is used by distributed SIL and SIA orthogonalization.
It reduces several scalar values with one collective, which lowers latency
without changing the physical inner product. Profiling reports both the
number of collectives and the number of scalar values reduced. A one-rank
layout bypasses MPI reduction calls entirely.

## Coulomb slices

The distributed Hamiltonian extracts and retains the corresponding rank-local
multipole slice along the selected coordinate during setup, so repeated
actions do not rebuild it. The slice keeps the complete angular and
unpartitioned radial axes and never constructs a full distributed-coordinate
state or output array. The existing serial full-state Coulomb path is
unchanged.

## Current scope

The implementation currently provides:

- the field-free Hamiltonian;
- neighboring bridge-state exchange;
- distributed weak-form derivative assembly;
- local Coulomb and angular actions;
- collective inner products, norms, expectations, Rayleigh quotients, and
  eigenstate residuals;
- serial and simulated multi-rank equivalence checks.

The complete prepared `TwoRadialSystem`, including setup tables, is still
replicated on every rank. The state and one selected radial Hamiltonian action
are distributed, but operator setup storage has not yet been partitioned.

The [distributed non-BO diatomic ground-state
solver](distributed_ground_state.md) is the first propagation consumer of this
foundation. The implementation does not yet provide time-dependent dipole
actions, CAP actions, mask application, distributed real-time propagation, or
distributed checkpoint histories.
