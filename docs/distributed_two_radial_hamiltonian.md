# Distributed two-radial Hamiltonian and reductions

`TwoRadialDistributedHamiltonian` applies the field-free two-radial
Hamiltonian to `DistributedTwoRadialState` objects. The first implementation
targets the non-Born--Oppenheimer diatomic layout in which `R` is distributed
and every rank retains all angular channels and the complete electronic
coordinate `r`.

## Local and communicated actions

The Hamiltonian separates naturally into three parts:

- The coordinate-1 kinetic and diagonal terms are rank-local because every
  rank stores the complete `r` coordinate.
- The Coulomb multipoles are point-local in both radial DVR coordinates. Each
  rank applies the optimized angular coupling kernel only at its owned `R`
  points.
- The coordinate-2 diagonal terms are local, while the FEM-DVR second
  derivative requires neighboring bridge data and assembly.

The distributed coordinate-2 derivative follows the weak-form finite-element
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

## MPI interface

The distributed Hamiltonian accepts a communicator that supplies `rank`,
`size`, `Irecv`, and `Isend`. An `mpi4py.MPI.Comm` provides this interface.
Communication uses NumPy buffers and nonblocking point-to-point operations.

```python
from mpi4py import MPI
from grid_tdse.spherical.two_radial import (
    TwoRadialDistributedHamiltonian,
    TwoRadialDistributedLayout,
)

comm = MPI.COMM_WORLD
layout = TwoRadialDistributedLayout(representation, comm.size)
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

## Physical reductions

`TwoRadialDistributedReductions` evaluates each physical inner product by
integrating only permanently owned `R` points and then summing the local
contributions with `allreduce`. Bridge points therefore occur exactly once.

The available operations include:

- `local_inner_product(left, right)` for one rank's contribution;
- `inner_product(left, right)` for the collective physical inner product;
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

## Coulomb slices

`TwoRadialHamiltonian.apply_coulomb_R_slice` exposes the optimized Coulomb
kernel on selected coordinate-2 points. It keeps the complete angular and
coordinate-1 axes but never constructs a full coordinate-2 state or output
array. The existing full-state Coulomb path is unchanged.

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
replicated on every rank. The state and coordinate-2 Hamiltonian action are
distributed, but operator setup storage has not yet been partitioned.

The implementation does not yet provide a distributed propagation driver,
time-dependent dipole action, CAP action, mask application, or distributed
checkpoint history. Krylov ground-state propagation is the next consumer of
this foundation.
