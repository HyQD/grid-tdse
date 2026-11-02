# Improved parallelization plan for two-radial TDSE calculations

## Purpose and development order

This plan extends the existing MPI foundation into a production-capable
parallel implementation for two-radial wavefunctions. Development proceeds in
the following strict order:

1. complete the non-Born--Oppenheimer diatomic-ion path;
2. use serial Davidson for production ground states and prioritize distributed
   SIL and SIA for real-time dynamics;
3. add distributed Crank--Nicolson only after the SIL/SIA path is complete;
4. reuse the resulting generic infrastructure for two-electron atoms;
5. add atomic electron-exchange compression only after a full-square atomic
   implementation has been validated.

The initial target systems are H2+, HD+, and D2+. The later atomic targets are
He and Be2+.

## Repository responsibilities

The separation between packages should remain:

- `grid-lib` owns serial FEM-DVR element data, element-to-global maps, local
  derivative actions, and bridge-function metadata. It must not depend on MPI.
- `integrator-lib` owns matrix-free Lanczos and Arnoldi exponential actions,
  adaptive refinement, reorthogonalization, and diagnostics. It must not
  depend on MPI.
- `grid-tdse` owns distributed state layouts, MPI communication, physical
  reductions, distributed Hamiltonian actions, checkpoint orchestration, and
  high-level atomic and molecular interfaces.

The MPI code should depend only on communicator protocols where practical so
that ownership and Hamiltonian equivalence can be tested without launching MPI.

## Scientific representations

### Non-BO diatomic ions

The wavefunction uses an electronic Jacobi radius `r`, an internuclear radius
`R`, and an angular expansion. These radial coordinates are physically
different and commonly use different FEM-DVR grids. Nuclear pair exchange does
not interchange `r` and `R`.

The production foundation now includes a two-dimensional element-aligned
layout over a rectangular `P_r x P_R` process grid. Each rank retains all
angular channels but owns only a connected Cartesian tile of the electronic
and internuclear radial coordinates. The earlier one-dimensional layouts are
retained as the `P_r=1` and `P_R=1` limits for compatibility, validation, and
small-rank execution.

### Two-electron atoms

The atomic wavefunction uses equal electronic grids `(r1,r2)`. Electron
exchange maps an angular pair and radial block according to

```text
(i, j, r1, r2) -> (j, i, r2, r1).
```

The existing serial atomic representation projects onto the requested exchange
parity but stores the complete radial square. A first parallel atomic
implementation should preserve that full-square storage. An optional
exchange-compressed representation can later assign an off-diagonal radial
tile and its transpose to the same rank.

## Lessons from two-electron FEDVR calculations

The radial tiling described by Feist provides the model for the longer-term
layout:

- retain the complete contributing angular expansion on each radial tile;
- split at finite-element boundaries;
- communicate only bridge-function edges for derivative operators;
- use scalar or batched reductions for norms and Krylov inner products;
- allow multiple tiles per rank for flexible rank counts and cache behavior;
- minimize per-call overhead when a rank owns only one or a few elements;
- precompute angular couplings rather than rebuilding them in every
  Hamiltonian action;
- parallelize checkpoint output before serial I/O becomes a large-scale
  bottleneck.

The identical-electron transpose-tile assignment is atomic-specific. A non-BO
`(r,R)` process grid uses ordinary rectangular tiles and does not need a
transpose partner.

A simple L-shaped radial domain is not a general non-BO optimization. Removing
the region where both `r` and `R` are large can remove dissociative ionization,
three-body breakup, or parts of neutral dissociation where an electron remains
localized on a separating nucleus. Any reduced physical domain requires a
separate channel-specific justification.

## Current implementation status

The following foundations are complete:

- element-local FEM-DVR derivative primitives;
- element-local serial two-radial Hamiltonian actions;
- compact conserved-`M_tot` angular storage;
- serial SIL and SIA for real and imaginary time;
- serial Krylov ground-state iteration;
- serial thick-restarted Lanczos and preconditioned Davidson ground-state
  solvers;
- converged Davidson continuation chains for H2+, HD+, and D2+, including H2+
  angular stress tests through `l_max=16`;
- one-dimensional element-aligned distributed storage in `R`;
- neighboring `R` bridge exchange and weak-form contribution assembly;
- distributed Coulomb actions on locally owned `R` points;
- distributed physical inner products, norms, energies, and residuals;
- one-dimensional distributed SIL imaginary-time propagation;
- local and Saga agreement between one-dimensional distributed and serial
  ground-state calculations;
- coordinate-selectable one-dimensional partitioning in either radial
  direction;
- a rectangular two-dimensional process-grid layout with scatter, gather, and
  redistribution;
- a two-dimensional distributed Hamiltonian with independent radial halo and
  weak-form exchanges;
- local and Saga agreement between two-dimensional and serial Hamiltonian
  actions for H2+, HD+, and D2+;
- two-dimensional distributed SIL and SIA imaginary-time propagation with
  batched physical reductions and distributed observable sampling;
- local thread-emulated and real-MPI agreement between Cartesian and serial
  SIL/SIA iterations;
- a fixed-core Saga comparison of serial, shared-memory, pure-MPI, and hybrid
  Hamiltonian execution;
- serial and one-dimensional distributed ground-state sampling, including
  expectation values, dipoles, symmetry diagnostics, iteration timing, and
  interval-based terminal and logfile output;
- distributed-compatible CN preconditioner prototypes.

The following limitations remain:

- the two-dimensional process-grid dimensions remain bounded by the available
  FEM-DVR elements in their respective radial directions;
- rank-local Coulomb data and the principal one-dimensional work arrays are
  reused, and the two-dimensional Hamiltonian reuses persistent work and
  communication buffers, while further kernel-level allocation profiling
  remains useful;
- Krylov orthogonalization batches basis projections, but norms and iteration
  diagnostics still require latency-sensitive scalar reductions;
- distributed real-time field interactions and absorbers are not complete;
- distributed checkpoint output gathers the complete state to rank zero;
- distributed CN lacks an MPI-aware linear solver;
- the atomic parallel path has not been validated.

The non-BO roadmap is complete through Phase 8. Serial Davidson has removed
ground-state preparation from the critical path, so distributed Davidson and
further ITP optimization are deferred. Phase 9, the complete distributed
SIL/SIA real-time dynamics path, is next.

## Saga findings to date

### Environment and launch configuration

The validated Saga environment used:

- `OpenMPI/5.0.10-GCC-15.2.0`;
- `mpi4py` 4.1.2 built from source with the module's `mpicc`;
- `srun --mpi=pmix`;
- one MPI rank per CPU core;
- `NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`;
- rank-local Numba cache directories;
- core binding through Slurm.

The Phase 6 fixed-core measurements found pure MPI fastest at every tested
core count for both `l_max=2` and `l_max=4`. Those runs kept
`NUMBA_NUM_THREADS=1`; their larger `--cpus-per-task` values therefore tested
library-thread availability, not threading of the Numba Coulomb kernel. The
current production default remains one physical CPU core and one numerical
thread per MPI rank, but an explicit fixed-core Numba-thread study is required
before ruling out hybrid execution for the high-angular-cutoff Coulomb action.

### Correctness validation

Local MPI calculations with one, two, and four ranks converged to the same H2+
state as the serial implementation. Saga calculations for H2+ and HD+ with
multiple rank counts retained unit norm, matching energies and residuals, and
phase-aligned serial-state differences around `1e-11` or smaller.

For an earlier validation grid, the distributed H2+ calculation converged in
1903 iterations to `-0.579827137696` Eh for one, two, and four local ranks. The
Saga HD+ sanity checks converged to `-0.566541180708` Eh with energy differences
at machine precision and phase-aligned state differences below approximately
`1.2e-11`.

The Cartesian Saga ground-state validation converged H2+, HD+, and D2+ with
`(1,2)`, `(2,1)`, and `(2,2)` process grids. Every calculation retained unit
norm, matched the serial energy within `2.220e-16` Eh, and matched the
phase-aligned serial state within `1.264e-11`. A converged H2+ SIA reference
reproduced the SIL result. A two-node `(2,2)` H2+ calculation on `c1-30` and
`c1-31` also reproduced the serial state, completing the multi-node
correctness gate. The archived measurements are documented in
[`benchmarks/non_bo_diatomic_mpi_scaling_saga.md`](../benchmarks/non_bo_diatomic_mpi_scaling_saga.md).

### Angular strong scaling

The representative FEM-DVR grids used for the angular study were:

```text
r: element boundaries [0, 4, 8, 12, 16], 11 points per element
R: element boundaries [0, 2, 4, 6, 8], 21 points per element
```

The four `R` elements limit the current decomposition to four ranks. The
angular cutoffs were related by

```text
m_max = l_max
lambda_max = 2*l_max
mu_max = lambda_max.
```

Representative solve times per imaginary-time iteration were:

| `l_max` | workload | 1 rank | 2 ranks | 4 ranks | 4-rank speedup | 4-rank efficiency |
|---:|:---|---:|---:|---:|---:|---:|
| 2 | converged run | 0.122013 s | 0.102085 s | 0.083818 s | 1.46 | 36% |
| 4 | converged run | 0.950942 s | 0.553180 s | 0.336337 s | 2.83 | 71% |
| 6 | fixed 100-step pilot | 15.815445 s | 8.224047 s | 4.444762 s | 3.56 | 89% |

The `l_max=6` two-rank representative is the median of the normal-node results
`7.942116`, `8.224047`, and `8.809443` seconds per iteration. Its mean is about
`8.325` seconds per iteration, corresponding to approximately 95% mean
two-rank efficiency. The result demonstrates that MPI overhead is not
amortized by the smallest angular calculation, while the angularly expensive
Hamiltonian scales well.

The `l_max=6` pilot was intentionally stopped after 100 iterations and was not
a converged physical ground-state result. Fixed-iteration pilots are useful for
controlled performance measurements, while fully converged runs remain
necessary for physical validation.

### Node-performance outlier

Two nominally identical two-rank `l_max=6` pilots on node `c1-2` required about
`22.95` seconds per iteration. Runs on `c1-8` and `c5-30` required about
`7.94`--`8.81` seconds per iteration and produced identical numerical results.
All jobs showed high CPU utilization, so utilization alone did not diagnose
the throughput loss.

The evidence associates the slow measurements with node conditions rather
than the MPI decomposition. Future performance studies should:

- record node placement and Slurm job information;
- repeat each configuration at least three times;
- report the median and retain outliers in the raw data;
- use identical grids, angular cutoffs, iteration counts, and tolerances;
- exclude a repeatedly anomalous node only with documented evidence;
- use exclusive-node allocations only when their additional cost is justified.

### Interpretation

The earlier one-dimensional layout is effective for angularly expensive
problems but cannot expose more ranks than there are elements in its selected
partitioning direction. The two-dimensional layout removes the replication of
the other radial coordinate and can use up to `P_r * P_R` ranks, subject to the
element counts and useful local tile size in both directions. Increasing
points inside existing elements raises work per rank without increasing this
topological rank limit; increasing the number of elements can raise it.

The observed behavior is consistent with a sparse, memory-bandwidth-sensitive
Hamiltonian. Cache effects, memory bandwidth, per-routine overhead, and local
tile size must therefore be recorded alongside MPI communication costs.

## Benchmark record

Reproducible scripts belong under `examples/Saga`. Summarized results and
machine-readable measurements should be stored separately:

```text
benchmarks/
    non_bo_diatomic_mpi_scaling_saga.md
    data/
        saga_non_bo_diatomic_mpi_scaling.csv
        saga_non_bo_diatomic_mpi_phase2.csv
        saga_non_bo_diatomic_mpi_phase3.csv
        saga_non_bo_diatomic_mpi_phase4.csv
        saga_non_bo_diatomic_mpi_phase5.csv
        saga_non_bo_diatomic_hamiltonian_phase6.csv
        saga_non_bo_diatomic_hamiltonian_phase6_provenance.json
```

The CSV should contain at least:

```text
date,commit,system,l_max,r_grid,R_grid,ranks,node,iterations,
solve_time,time_per_iteration,energy,residual,norm,status,notes
```

Raw Slurm output does not need to be committed. The benchmark report should
record software versions, MPI module, grid construction, angular cutoffs,
symmetry sector, integrator options, thread settings, node placement, peak
memory, wall time, speedup, parallel efficiency, and core-time consumption.

## Target distributed architecture

The implemented shared abstraction is a two-dimensional element layout:

```text
TwoRadialDistributed2DLayout
    process grid: P1 x P2
    all retained angular channels per tile
    contiguous coordinate-1 element range
    contiguous coordinate-2 element range
    owned, bridge, and halo degrees of freedom in both directions
    optional system-specific exchange policy
```

The process-grid aspect ratio is chosen from element counts and estimated tile
cost. Future refinements can additionally account for radial orders, angular
workload, memory limits, and measured halo volume. Equal-size square blocks
are not appropriate for general non-BO grids; the implementation permits
unequal connected element ranges.

The existing one-dimensional layouts remain available as the `(1,P2)` and
`(P1,1)` process-grid limits, preserving validation and simple low-rank
execution paths.

## Non-BO diatomic implementation path

The non-BO path must be completed before atomic parallel development begins.

| phase | scope | status |
|---:|:---|:---|
| 1 | archive and instrument the one-dimensional implementation | complete |
| 2 | optimize reductions, communication, and allocation | complete |
| 3 | make one-dimensional partitioning coordinate-selectable | complete |
| 4 | implement the two-dimensional state layout | complete |
| 5 | implement the two-dimensional Hamiltonian | complete |
| 6 | compare serial, threaded, pure-MPI, and hybrid execution | complete |
| 7 | complete distributed SIL/SIA ground-state propagation | complete |
| 8 | stress-test and archive production ground states | complete |
| 9 | complete distributed SIL/SIA dynamics | next |
| 10 | implement scalable distributed state I/O | pending |
| 11 | benchmark the complete SIL/SIA path on Saga | pending |
| 12 | deploy and benchmark on Olivia and Betzy | pending |
| 13 | add distributed Crank--Nicolson | pending |

### Phase 1: archive and instrument the current implementation

**Status:** complete. The historical Saga report and CSV are archived, the
existing one-dimensional H2+/HD+ correctness validation is complete, and the
component, communication, collective, and Krylov diagnostics are implemented.
Three repeated one-, two-, and four-rank Saga measurements distinguish useful
Hamiltonian work, neighbor exchange, local reductions, and collective latency.
They identify the approximately 405000 scalar reductions per solve as the
dominant scaling limit for the representative `l_max=2` calculation.

1. Add the Saga benchmark report and CSV described above.
2. Mark the existing one-dimensional Saga correctness validation complete.
3. Record separate times for electronic-coordinate work, internuclear-coordinate
   work, Coulomb work, halo exchange, weak-contribution exchange, and collective
   reductions.
4. Record Krylov operator calls, accepted and rejected substeps, refinement
   depth, Krylov dimension, and error estimates.
5. Report maximum rank time for every timed collective phase.

**Completion gate:** repeated one-, two-, and four-rank measurements can
distinguish computational work, neighbor communication, and collective latency.

### Phase 2: optimize the existing one-dimensional path

**Status:** complete. The one-dimensional Hamiltonian retains rank-local
Coulomb slices, reuses its communication and operator work arrays, separates
interior and boundary `R` elements, and overlaps nonblocking bridge
communication with independent work. Distributed SIL/SIA orthogonalization
uses batched inner products and buffer-based collectives. Thread-communicator
equivalence tests, actual MPI correctness checks, and repeated one-, two-, and
four-rank Saga measurements pass. For the representative `l_max=2` workload,
the two- and four-rank median solve times improved by approximately 2.21x over
Phase 1, and collective time decreased by factors of approximately 7.3 and
9.5, respectively.

1. Precompute and retain rank-local radial multipole slices.
2. Reuse halo, weak-action, Hamiltonian-output, and Krylov work buffers.
3. Split local `R` elements into interior and boundary groups.
4. Start nonblocking halo exchange, compute independent electronic and Coulomb
   work plus interior `R` elements, and wait only before boundary work.
5. Overlap returned weak bridge contributions with other independent work where
   practical.
6. Batch Krylov inner products into fewer collective reductions.
7. Retain numerical equivalence tests after every optimization.

**Completion gate:** the optimized path reproduces the existing distributed
state and energy while reducing allocation overhead or communication wait time
on representative Saga workloads.

### Phase 3: make one-dimensional partitioning coordinate-selectable

**Status:** complete. Ownership, state storage, Coulomb slices,
physical reductions, element-local bridge exchange, and SIL imaginary-time
propagation now support either radial coordinate. Coordinate 2 remains the
default. Full and compact angular-storage tests agree with the serial
Hamiltonian, both layouts agree with the same serial ground-state iterations,
the coordinate-1 path passes a two-rank MPI smoke test, and gathered archives
record the selected coordinate in distributed-layout metadata. The MPI
example exposes `--partition-coordinate r|R`. On Saga, fixed-iteration H2+
smoke runs for both `r` and `R` partitioning passed with one, two, and four MPI
ranks. Energies, norms, residuals, and phase-aligned states agreed with the
serial comparison. The initial two-rank `r` run encountered a transient missing
file in the shared Python environment; its clean retry passed and is not
treated as a solver failure. Unequal-grid performance comparison remains part
of the later benchmark phases. The results are archived in the
[Saga MPI benchmark](../benchmarks/non_bo_diatomic_mpi_scaling_saga.md).

1. Generalize ownership metadata so either coordinate can be partitioned.
2. Implement coordinate-1 bridge exchange using the same element-local rules.
3. Retain coordinate-2 partitioning as the default for compatibility.
4. Compare partitioning `r` and partitioning `R` for unequal non-BO grids.

**Completion gate:** both one-dimensional layouts agree with the serial
Hamiltonian and ground state, and the selected coordinate can be archived in
layout metadata.

### Phase 4: implement the two-dimensional `(r,R)` layout

**Status:** complete and validated on Saga. The additive
`TwoRadialDistributed2DLayout`
supports explicit or cost-selected rectangular process grids, row-major rank
topology, connected element-tile ownership, per-axis bridges and halos, global
Cartesian halo owners, edge-neighbor and row/column rank groups, full and
compact angular storage, exact scatter/gather, process-grid redistribution,
serial archive round trips, and phase-aligned comparison helpers. Tests cover
unequal element counts and polynomial orders, a prime rank count, several
process grids, ownership validation, and invalid decompositions. A real
four-rank local MPI run exactly scattered, redistributed from `2 x 2` to
`1 x 4`, and gathered a complex H2+ state. On Saga, a three-rank automatic
layout selected `(3, 1)`, while four-rank jobs exactly redistributed from
`(2, 2)` to both `(1, 4)` and `(4, 1)`. All three jobs recovered the original
3081 coefficients with zero phase-aligned difference at printed precision.
The cluster results are archived in the
[Saga MPI benchmark](../benchmarks/non_bo_diatomic_mpi_scaling_saga.md).

1. Define a rectangular `P_r x P_R` process topology.
2. Define permanent ownership, bridge ownership, halos, and neighbors in both
   directions.
3. Support connected groups of element tiles per rank.
4. Balance unequal grids using estimated coefficient and Hamiltonian cost.
5. Implement scatter, gather, redistribution, and phase-aligned equivalence
   helpers.
6. Preserve the existing serial archive format for small gathered states.

**Completion gate:** layout tests cover unequal element counts, unequal
polynomial orders, uneven rank counts, bridge ownership, and exact
scatter/gather round trips.

### Phase 5: implement the two-dimensional Hamiltonian

**Status:** complete and validated on Saga. The
additive `TwoRadialDistributed2DHamiltonian` applies both element-local radial
operators with independent edge-only halo and weak-form exchanges, while
diagonal one-body terms and precomputed tile-local Coulomb multipoles remain
communication-free. Persistent work and communication buffers are reused, and
interior elements are evaluated while each halo exchange is in flight. Local
mask and CAP actions use the same owned Cartesian tiles. Emulated process-grid
tests cover full and fixed-`M_tot` angular storage and `(1,1)`, `(1,2)`,
`(2,1)`, and `(2,2)` layouts. Random complex H2+, HD+, and D2+ actions agree
with the serial element-local Hamiltonian. A real four-rank local MPI run on a
`(2,2)` grid agreed with serial to `3.6e-15` in the complete action. The
reusable `examples/mpi_two_radial_2d_hamiltonian.py` driver was then used for
Saga deployment validation. On Saga, serial-limit `(1,1)`, one-coordinate `(1,2)`
and `(2,1)`, and Cartesian `(2,2)` process grids all agreed with the serial
Hamiltonian. The validation covered H2+, HD+, and D2+, including an H2+
`l_max=2` case; component and complete-action maximum errors were at most
`7.944e-15`, and every Coulomb comparison was exact at printed precision. The
cluster results are archived in the
[Saga MPI benchmark](../benchmarks/non_bo_diatomic_mpi_scaling_saga.md).

1. Apply both radial derivatives from element-local tile data.
2. Exchange left/right and up/down bridge data only for derivative terms.
3. Use two-stage edge synchronization or a generalized unique-owner weak-form
   assembly so corner contributions are correct without diagonal messages.
4. Apply radial potentials, Coulomb angular couplings, masks, and CAP values
   locally on each tile.
5. Precompute tile-local Coulomb multipoles and immutable coupling data.
6. Reuse persistent buffers and overlap communication with interior work.

**Completion gate:** random complex-state component actions and complete
Hamiltonian actions agree with the serial implementation for H2+, HD+, and D2+
on several process-grid shapes.

### Phase 6: establish shared-memory and hybrid execution baselines

**Status:** complete at the Hamiltonian level. Controlled Saga measurements
were performed for `l_max=2` and `l_max=4` with 1, 2, 4, and 8 allocated cores,
three repetitions per rank/thread factorization, and 5000 timed Hamiltonian
actions per job. All configurations agreed numerically with the serial action.
Pure MPI was fastest at every core count for both workloads; shared-memory
threading provided little benefit and hybrid performance primarily followed
its MPI rank count. The measurements, provenance, and interpretation are
archived in
[`benchmarks/non_bo_diatomic_mpi_scaling_saga.md`](../benchmarks/non_bo_diatomic_mpi_scaling_saga.md).

The default for subsequent SIL/SIA work is therefore one physical CPU core per
MPI rank, one OpenMP/BLAS/Numba thread per rank, and core binding. The complete
ground-state and real-time propagation paths must still repeat the fixed-core
execution-model comparison because Krylov reductions and time-dependent work
can change the preferred balance.

Benchmark the operational two-dimensional Hamiltonian before committing the
subsequent SIL/SIA solvers to one execution model. Compare configurations at a
fixed total CPU-core count so that a threaded run is not mistaken for a free
speedup over MPI.

1. Record the NumPy/SciPy build configuration, BLAS implementation, thread-pool
   state, CPU model, node placement, and binding for every run.
2. Use representative unequal non-BO grids and angular cutoffs large enough to
   amortize startup and communication overhead. Use fixed Hamiltonian-action or
   fixed-iteration pilots; do not infer physical convergence from them.
3. At each total core count `P`, compare:
   - serial execution: one MPI rank with one thread;
   - shared-memory execution: one MPI rank with `P` BLAS/OpenMP threads;
   - pure MPI execution: `P` ranks with one thread per rank; and
   - selected hybrid configurations with `q` ranks and `t` threads per rank,
     where `q * t = P`.
4. Match Slurm `--cpus-per-task` to the requested thread count, bind ranks and
   threads to physical cores, and prevent oversubscription. Control at least
   `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
   `NUMBA_NUM_THREADS`; record `OMP_PLACES` and `OMP_PROC_BIND` when OpenMP
   threads are enabled.
5. Treat BLAS threading and Numba parallelism as separate mechanisms. Do not
   assume that increasing NumPy's thread count accelerates element-local Numba
   kernels unless those kernels have independently been made parallel.
6. Repeat each timing point at least three times on the same node type and, as
   far as practical, comparable node placement. Report medians and retain
   documented outliers.
7. Report wall time, allocated core time, CPU utilization, memory, component
   profiles, and numerical equivalence at identical physical inputs and
   tolerances.
8. Use the results to choose the default execution model and to decide whether
   subsequent job templates should remain pure MPI or expose a documented
   hybrid option.

This phase establishes a Hamiltonian-level baseline. The comparison must be
repeated for the complete ground-state and real-time propagation paths after
those paths are operational, because Krylov reductions and time-dependent
work can change the preferred rank/thread balance.

**Completion gate:** satisfied. Representative Hamiltonian workloads identify
the useful serial, threaded, pure-MPI, and hybrid configurations at fixed core
counts; all configurations agree numerically; no configuration is
oversubscribed; and the recommended execution and binding settings are
documented for subsequent SIL/SIA development.

### Phase 7: complete distributed SIL/SIA ground-state propagation

**Status:** complete. The local implementation and Saga deployment are
validated.
`TwoRadialDistributed2DGroundStateSolver` connects the Cartesian state and
Hamiltonian to SIL or SIA using quadrature-weighted batched reductions.
Generated states and compatible serial archives are supported. Distributed
sampling evaluates expectation values, dipoles, symmetry diagnostics, and
timings without gathering the state; only the root rank prints and writes the
optional logfile. Thread-emulated tests cover `(1,1)`, `(1,2)`, `(2,1)`, and
`(2,2)` layouts, SIL/SIA agreement with serial iterations, and H2+, HD+, and
D2+ one-step equivalence. Real four-rank local MPI smoke tests passed for SIL
and SIA on `(2,2)`; the SIL gathered state agreed with serial to
`3.520e-16` and its energy to `5.551e-17` Eh after one step. These are local
correctness measurements, not converged physical results.

Saga then converged H2+, HD+, and D2+ with SIL on `(1,2)`, `(2,1)`, and
`(2,2)` process grids. All layouts reproduced the serial energies within
`2.220e-16` Eh and phase-aligned states within `1.264e-11`. A converged H2+
SIA calculation reproduced the SIL result, and distributed sampling retained
the requested inversion and pair-exchange sectors. A four-rank `(2,2)` H2+
calculation spanning nodes `c1-30` and `c1-31` converged to the same state.
The one-dimensional four-rank baseline took `112.761` seconds and the
Cartesian `(2,2)` SIL calculation took `117.252` seconds, showing modest 2D
overhead at this low rank count while removing the one-coordinate four-rank
limit. Full records and provenance are archived in the Saga benchmark report.

1. Connect the two-dimensional distributed state, Hamiltonian, physical inner
   products, and batched Krylov projections to SIL and SIA callbacks.
2. Generalize distributed SIL imaginary-time propagation to the 2D layout.
3. Use SIL as the production method for the Hermitian ground-state problem.
4. Retain SIA as an independent general-exponential reference.
5. Preserve normalized iteration, energy change, phase-invariant state change,
   and eigenstate-residual convergence checks.
6. Extend the existing ground-state sampling and interval logfile path to the
   two-dimensional layout without gathering the state for each sample.
7. Verify conserved `M_tot`, inversion, and applicable pair-exchange sectors.
8. Support initialization from generated states and compatible archives.
9. Compare time to solution, Hamiltonian calls, reductions, and memory with the
   one-dimensional MPI and serial solvers.
10. Validate locally on several small process grids before performing
    single-node and then multi-node Saga correctness runs.

Ground-state calculations must not use a CAP.

**Completion gate:** satisfied. H2+, HD+, and D2+ converge to serial-reference
energies and states on multiple 2D process grids, and a two-node correctness
run reproduces the single-node and serial H2+ result.

### Phase 8: stress-test and archive production ground states

This phase established whether ground-state preparation would limit dynamics.
The conclusion is that it does not at the currently relevant problem sizes.

1. Serial preconditioned Davidson converged H2+, HD+, and D2+ for
   `l_max = 0, 2, 4, 6, 8` on the `8e21-4e21_16-8` reference grid, with all
   residuals below `1e-8` and all available reference energies reproduced to
   eight decimal places.
2. Continuation from the preceding angular cutoff reduced the higher-cutoff
   calculations to approximately 40--55 Hamiltonian actions.
3. H2+ continuation remained robust through `l_max=14` on the reference grid
   and through `l_max=16` on the `4e11-4e21_16-8` grid.
4. The lower-resolution chain provides an angular stress test but not a
   radially converged energy. Angular-limit extrapolation must therefore be
   performed independently at each radial resolution.
5. ITP and Lanczos remain independent validation paths, but further ITP
   optimization and distributed Davidson are deferred until state size or
   serial setup time makes them necessary.
6. Converged production states remain self-describing and record their parent
   state, solver settings, physical parameters, representation, observables,
   and provenance.
7. Large production states belong in a versioned external archive with
   checksums and a repository manifest. Only small states with a concrete test
   or example purpose should be committed.

#### Phase 8a: characterize the Coulomb-kernel optimization target

The `l_max=6` and `l_max=8` continuation runs show that the Coulomb action
becomes the dominant Hamiltonian component as the number of angular couplings
grows. Establish a reproducible kernel baseline before changing its storage or
loop structure:

1. Time the production compact Coulomb kernel in isolation for real
   ground-state-like and complex dynamics-like states at `l_max=4,6,8`.
   Exclude construction, JIT compilation, communication, and correctness
   checks from the timed interval, but retain an untimed serial comparison.
2. Sweep the internal tile size on the complex `l_max=8` case. Use repeated
   medians and, where Saga permits `perf` access, record cycles, instructions,
   cache references, and cache misses to distinguish arithmetic work from
   memory/cache limitations.
3. At a fixed allocation of 16 physical cores, compare 16 ranks x 1 Numba
   thread, 8 x 2, and 4 x 4 for both real and complex states. Keep BLAS at one
   thread and bind each rank to its allocated cores so this measures Numba
   threading rather than oversubscription.
4. Archive the raw JSON records, job manifest, provenance, Slurm accounting,
   summaries, and optional counter files before implementing an optimized
   kernel.

The measurements selected tile size 512 as a stable default. Pure MPI with one
physical core and one numerical thread per rank was fastest at the tested
fixed allocation, while all real and complex configurations agreed with the
reference Coulomb action.

**Phase 8a gate:** satisfied. The archived benchmark defines a performance
regression baseline for subsequent kernel work.

**Completion gate:** satisfied for beginning dynamics. At least one practical
starting state can be loaded from a reproducible archive or generated quickly
with serial Davidson. Remaining archive catalog and checksum work is
nonblocking housekeeping.

### Phase 9: complete distributed SIL/SIA dynamics

Complete Phase 9 in the following subphases. Length-gauge Hermitian dynamics
comes first; velocity gauge and non-Hermitian absorbers follow only after the
shared state, sampling, and restart contracts are stable.

#### Phase 9a: dynamics representation and initialization

**Implementation status:** complete. The serial archive and automatic
Davidson paths, Cartesian root-only setup and scatter, and local MPI
equivalence tests are in place. Saga deployment validation completed on
2026-11-02 for H2+, HD+, and D2+ with the `require`,
`compute-if-missing`, and `recompute` policies. The accepted validation jobs,
states, outputs, and machine-readable results are archived under
`benchmarks/data/saga_non_bo_diatomic_phase9a/phase9a-20261102-004030`.

1. Construct a dynamics representation that retains both even and odd
   electronic angular channels while preserving `M_tot=0` for a linearly
   z-polarized field.
2. Preserve pair-exchange symmetry for H2+ and D2+; HD+ has no corresponding
   identical-pair symmetry.
3. Embed a symmetry-adapted Davidson ground state into the larger dynamics
   representation and validate the system, CODATA set, radial grids, angular
   limits, physical parameters, and state shape.
4. Support explicit initialization policies:

   ```text
   require
   compute-if-missing
   recompute
   ```

5. Under `compute-if-missing`, search for a metadata-compatible archive and,
   when none exists, compute a serial Davidson state, verify convergence and
   symmetries, save it, and embed it into the dynamics basis.
6. In an MPI job, only rank zero may perform the serial Davidson setup; report
   setup time separately and distribute the resulting state after a barrier.
7. Initially require `A(t0)=0` in velocity gauge. A nonzero initial vector
   potential requires an explicit gauge transformation of the field-free
   ground state.

**Phase 9a gate:** archived and automatically generated ground states produce
the same serial and Cartesian-MPI initial dynamics state. Satisfied on Saga:
all accepted four-rank `2 x 2` jobs reconstructed the serial state with zero
phase-aligned and coefficient differences at printed precision, while only
rank zero entered Davidson when computation was requested.

#### Phase 9b: shared dynamics sampling

Implement one observable evaluator and time-series schema shared by serial,
one-dimensional MPI, and Cartesian MPI. Sample at a user-configurable accepted
step interval and always support initial, final, and checkpoint samples.

Record:

- time and accepted-step index;
- norm and norm divergence;
- gauge-covariant energy and energy gain;
- electric field `E(t)` or vector potential `A(t)`;
- inversion and, when defined, pair-exchange expectations;
- `<z_e>`, `<z_R>`, `<r>`, and `<R>`;
- `<L^2>`, `<L_z>`, `<l_r^2>`, `<l_r,z>`, `<l_R^2>`, and `<l_R,z>`;
- `<cos(theta_R)>` and `<cos^2(theta_R)>`;
- electron--center distances;
- electronic, internuclear, and total dipole vectors;
- propagation wall time, Krylov dimensions, refinements, error estimates, and
  accepted or rejected internal substeps.

Do not identify the propagation-Hamiltonian expectation `<H(t)>` with the
physical instantaneous energy. Following Aurbakken et al., sample the
gauge-covariant energy operator. In dipole length gauge,

```text
K_L = H_0.
```

With Jacobi-coordinate reduced charges `xi_r`, `xi_R`, reduced masses `mu_r`,
`mu_R`, velocity couplings `nu_i = xi_i / mu_i`, and total vector potential
`A(t)`, the velocity-gauge operator is

```text
K_V(t) = H_0 - A(t) [nu_r p_r,z + nu_R p_R,z]
         + A(t)^2/2 [xi_r^2/mu_r + xi_R^2/mu_R].
```

The propagation code may continue to omit the spatially uniform `A(t)^2`
term because it changes only the global phase, but energy sampling must restore
it. For multiple pulses, sum their vector potentials before squaring.

After a mask or CAP, retain unnormalized expectation numerators. Optional
norm-divided conditional values must use distinct names; never renormalize
silently. Exclude numerical absorbers from the physical energy operator.

The legacy `pulse.py` distance expression uses squared mean radii and is not a
true mean or RMS distance. Store a clearly named legacy-compatible proxy for
comparison and a physically defined RMS distance based on the corresponding
second moments.

**Phase 9b gate:** serial, one-dimensional MPI, and Cartesian-MPI evaluators
agree for identical test states, including states with norm below one.
Satisfied locally: the common numerator reduction and dynamics schema agree
for serial, both one-coordinate decompositions, and a Cartesian `2 x 2`
decomposition on deliberately subnormalized states. The remaining deployment
check is to repeat the gate with Saga MPI collectives before phase 9c.

#### Phase 9c: serial field-free and length-gauge validation

1. Start from an archived or automatically generated Davidson state.
2. Propagate field-free with SIL and verify norm conservation and stationarity
   up to a global phase.
3. Add a z-polarized length-gauge pulse and evaluate the Hamiltonian at each
   temporal midpoint.
4. Verify convergence with time step and Krylov tolerance.
5. Compare SIL and SIA for the Hermitian calculation.

**Phase 9c gate:** a serial H2+ pulse calculation produces reproducible state
and observable time series from a Davidson initial state.

#### Phase 9d: Cartesian-MPI length-gauge dynamics

1. Add the distributed z-polarized length-gauge action.
2. Keep its radial multiplication tile-local and reuse the Cartesian
   Hamiltonian and batched physical reductions.
3. Compare one field action, one accepted propagation step, a short time
   series, and the final phase-aligned state with serial results.
4. Validate `(1,2)`, `(2,1)`, and `(2,2)` process grids before multi-node use.

**Phase 9d gate:** serial and distributed length-gauge pulse calculations
agree within documented tolerances.

#### Phase 9e: serial snapshots, checkpoints, and restart

Provide independent options for lightweight observable samples, retained
wavefunction snapshots, and restart-oriented checkpoints. Full-wavefunction
snapshots support a user-specified accepted-step interval plus optional initial
and final states. They are written after the full accepted step and after any
configured mask, never after an internal Krylov refinement.

Each snapshot stores the complete complex state, physical time, accepted-step
index, representation, physical and CODATA parameters, field and gauge
metadata, integrator settings, absorber metadata, norm, checksum, parent-state
identity, and code provenance. A snapshot coincident with a checkpoint should
be reusable for restart rather than duplicated.

Preserve checkpoint history, field metadata, integrator diagnostics, and
restart time. Permit a restarted state to encounter a different subsequent
field while strictly validating all representation and physical metadata.

**Phase 9e gate:** uninterrupted and checkpoint/restart serial calculations
produce equivalent states and time series.

#### Phase 9f: masks, CAPs, and SIA

1. Add separate electronic and internuclear masks first.
2. Add CAP propagation with SIA after mask behavior is validated.
3. Reject simultaneous mask and CAP use on the same coordinate.
4. Verify expected norm loss, raw and conditional observable semantics, and
   restart equivalence.

**Phase 9f gate:** absorbed serial dynamics have internally consistent norm,
energy, and observable histories.

#### Phase 9g: velocity-gauge dynamics

1. Validate the serial momentum action and extend Cartesian bridge exchange to
   the radial derivatives required by the distributed velocity-gauge action.
2. Compare serial and distributed velocity actions, accepted steps, time
   series, and final states.
3. Compare corresponding length- and velocity-gauge calculations, including
   their gauge-covariant energies.

**Phase 9g gate:** paired gauge calculations converge to the same physical
observables within spatial, angular, and temporal discretization errors.

#### Phase 9h: end-to-end dynamics example

Provide a production-style H2+ example that loads or computes a Davidson
ground state, embeds it into the dynamics representation, applies a finite
z-polarized pulse, samples the complete observable contract, writes optional
wavefunction snapshots and checkpoints, restarts, and reproduces its serial
result with Cartesian MPI.

**Phase 9 completion gate:** the end-to-end example passes field-free,
length-gauge, absorber, restart, and gauge-comparison acceptance tests with
documented tolerances.

### Phase 10: implement scalable distributed state I/O

1. Remove the requirement to gather production-scale snapshots or checkpoints
   to rank zero. Never perform such a gather implicitly.
2. Evaluate MPI-enabled HDF5 hyperslab storage against rank-local shards with a
   manifest.
3. Preserve the complete state plus all representation, physical, field,
   gauge, integrator, absorber, time, checksum, and provenance metadata.
4. Support retained full-wavefunction snapshots at a user-specified accepted
   step interval, independently of observable and checkpoint intervals.
5. Support restart with a different compatible MPI process grid.
6. Make incomplete or partially committed writes detectable and keep existing
   valid checkpoints recoverable.
7. Estimate the uncompressed wavefunction size, expected snapshot count, and
   total storage requirement before propagation begins.

**Completion gate:** multi-rank snapshots and checkpoints can be written and
reloaded without constructing the full state on one rank, including
redistribution to a different rank count.

### Phase 11: benchmark the complete SIL/SIA path on Saga

Perform controlled studies in this order:

1. fixed radial grids with increasing angular cutoffs;
2. fixed `R` grid with increasing electronic `r` elements and extent;
3. fixed `r` grid with increasing `R` points per existing element;
4. fixed `r` grid with increasing numbers of `R` elements;
5. fixed physical problem with increasing 2D rank count for strong scaling;
6. proportional grid and rank growth for weak scaling;
7. repeat the fixed-core serial, threaded, pure-MPI, and hybrid comparison from
   Phase 6 using the complete real-time propagation path;
8. one-node scaling before multi-node scaling;
9. Davidson setup and accepted real-time-step costs, reported separately;
10. SIL and SIA comparison without a CAP, followed by SIA with a CAP;
11. observable-sampling overhead, including gauge-covariant energy;
12. snapshot and checkpoint bandwidth, storage, and restart cost;
13. length- and velocity-gauge agreement and relative cost.

Report wall time, core time, speedup, efficiency, peak memory per rank,
Hamiltonian calls, collective count, communicated bytes, Krylov dimensions,
accepted/rejected refinements, sampling overhead, I/O bandwidth, storage, and
observable agreement. Do not infer physical convergence from a fixed-iteration
performance pilot.

**Completion gate:** the report identifies the useful rank range for each
problem size, the onset of communication saturation, and a recommended Saga
job configuration for production non-BO dynamics.

### Phase 12: deploy and benchmark the SIL/SIA path on Olivia and Betzy

Treat cluster deployment as a separate portability phase. The Python solver,
distributed layouts, Hamiltonian, and integrators must remain shared; only the
software environment, MPI launch configuration, resource requests, cache
placement, and site-specific performance settings should vary by cluster.

1. Add cluster-specific setup and launch documentation under
   `examples/Olivia` and `examples/Betzy`, while continuing to use the shared
   MPI Python examples and benchmark drivers.
2. Build an independent environment on each cluster. Do not copy an installed
   Conda environment, `mpi4py` build, compiled extension, or Numba cache from
   Saga or between clusters.
3. On Olivia, use the site-supported container workflow and CPU software stack,
   and configure OpenMPI for the HPE Slingshot interconnect according to the
   current Sigma2 guidance. Keep distinct single-node and multi-node launch
   templates where their MPI transport settings differ.
4. On Betzy, build `mpi4py` against the MPI module loaded for the job and use a
   separate cluster-local environment and cache directories. Confirm current
   machine availability, modules, partitions, and launch recommendations
   before fixing the templates.
5. Make every job report the Python executable, package commits, MPI module,
   `MPI.Get_library_version()`, node placement, rank mapping, CPU binding,
   thread settings, and cache locations.
6. Start with one CPU per MPI rank and one thread in Numba, OpenMP, OpenBLAS,
   and MKL. Do not request GPU nodes or additional CPUs per rank until an
   explicitly threaded or GPU local kernel exists.
7. Validate a two-rank collective test on a compute node, followed by one-,
   two-, and four-rank H2+ field-free and length-gauge correctness runs.
   Require agreement with serial and Saga norms, gauge-covariant energies,
   observable time series, and phase-aligned states.
8. Repeat the HD+ sanity check to cover unequal nuclear masses and the absence
   of homonuclear pair-exchange adaptation.
9. Run a representative subset of the complete Saga SIL/SIA benchmark matrix
   on each cluster: angular growth, radial growth, strong scaling, Davidson
   initialization, real-time dynamics without a CAP, SIA dynamics with a CAP,
   and distributed snapshot and checkpoint I/O.
10. Use identical physical inputs, commits, tolerances, and fixed-iteration
    performance pilots for cross-cluster comparisons. Record both time to
    solution and allocation cost; do not compare raw timings from mismatched
    problem definitions.
11. Repeat each performance point at least three times, record node placement,
    report medians, and retain documented outliers. Account for the different
    CPU, memory-per-core, NUMA, and network topologies when interpreting the
    results.
12. Begin with partial-node or single-node jobs. Attempt full-node and
    multi-node scaling only when the number of radial tiles and the measured
    per-rank workload can use the allocation efficiently.

Store machine-readable results and summaries alongside the Saga record, for
example:

```text
benchmarks/
    non_bo_diatomic_mpi_scaling_olivia.md
    non_bo_diatomic_mpi_scaling_betzy.md
    data/
        olivia_non_bo_diatomic_mpi_scaling.csv
        betzy_non_bo_diatomic_mpi_scaling.csv
```

**Completion gate:** the complete distributed SIL/SIA path passes numerical
correctness checks on Olivia and Betzy, reproducible cluster-specific setup and
job templates are documented, and the benchmark reports identify useful rank
ranges and recommended production configurations on all three clusters. Any
machine that is no longer generally available should retain a documented
validation record without blocking subsequent development.

### Phase 13: add distributed Crank--Nicolson

Distributed CN follows, rather than blocks, the production SIL/SIA path.

1. Implement an MPI-aware GMRES or BiCGSTAB solver using the distributed
   physical inner product and reductions.
2. Connect diagonal Jacobi, element-block Jacobi, and overlapping additive
   Schwarz preconditioners.
3. Retain the global spectral preconditioner as a serial reference only.
4. Compare setup time, iteration count, communication, memory, and total step
   time with SIL/SIA.
5. Validate Hermitian dynamics, checkpointing, observables, and absorber
   behavior where mathematically appropriate.

**Completion gate:** distributed CN is numerically validated and its practical
performance range relative to SIL/SIA is documented. SIL/SIA remain the
primary route unless CN demonstrates a clear advantage for a defined workload.

## Two-electron atomic implementation path

Atomic development begins only after the non-BO completion gates above have
been met.

### Atomic phase 1: validate the existing one-dimensional engine

1. Construct the distributed layout from a `TwoElectronAtomGroundState`
   representation.
2. Partition the second electron coordinate while retaining the first in full.
3. Project the initial state onto exchange parity before scattering.
4. Confirm that the atomic Hamiltonian preserves singlet and triplet spatial
   exchange sectors without repeated projection.
5. Compare distributed Hamiltonian actions, energies, norms, residuals, and
   exchange expectations with serial He and Be2+ calculations.

This asymmetric decomposition is mathematically valid even though it does not
exploit equal grids or reduce both radial dimensions.

### Atomic phase 2: reuse the generic full-square 2D layout

1. Use the shared rectangular 2D engine with equal `r1` and `r2` grids.
2. Store the complete radial square and all retained angular pairs.
3. Use square or near-square process grids where they minimize halo surfaces.
4. Validate SIL/SIA ground-state and real-time propagation.
5. Require identical one-electron field and absorber actions so electron
   exchange parity remains a symmetry.

**Completion gate:** full-square atomic MPI results agree with the serial
atomic implementation and preserve exchange parity.

### Atomic phase 3: add optional exchange-compressed storage

1. Define canonical exchange-related angular pairs.
2. Store one representative of exchange-related radial/angular data.
3. Assign each off-diagonal radial tile and its transpose to the same rank.
4. Handle diagonal tiles explicitly for symmetric and antisymmetric sectors.
5. Generalize archives, observables, dipole actions, masks, CAPs, and
   checkpoints to the compressed representation.
6. Compare full-square and compressed memory, communication, cache behavior,
   and runtime before selecting a default.

This phase is atomic-specific, but it should reuse the generic tile, neighbor,
reduction, Krylov, and I/O infrastructure.

### Atomic phase 4: atomic Saga benchmarks

Benchmark radial-grid growth, angular growth, rank scaling, exchange sectors,
SIL/SIA dynamics, absorber behavior, and checkpoint I/O. Compare against the
two-electron radial tiling principles described by Feist while retaining the
uncoupled angular basis used by `grid-tdse`.

## Deferred alternatives

The following should be deferred until radial tiling has been exhausted:

- distributing angular channels, which introduces communication following the
  sparse angular-coupling graph;
- hybrid MPI/threading, which requires explicit local threaded kernels and
  careful control of numerical thread pools;
- general nonrectangular radial domains;
- GPU execution;
- topology-specific optimization that compromises the communicator-based
  testing model.

## Overall completion criterion

The parallel two-radial program is complete when:

1. serial Davidson ground states initialize laser-driven dynamics that run
   with distributed SIL and SIA on rectangular 2D grids;
2. masks, CAPs, observables, wavefunction snapshots, checkpoints, and restarts
   operate without a rank-zero state or I/O bottleneck;
3. Saga correctness, strong scaling, weak scaling, memory, and core-time costs
   are documented;
4. the SIL/SIA implementation is deployed and numerically validated on Olivia
   and Betzy, with cluster-specific setup and performance guidance documented;
5. distributed CN is available as a validated secondary method;
6. the same generic infrastructure supports full-square two-electron atomic
   calculations;
7. optional atomic exchange compression is validated independently rather than
   being required by the shared MPI foundation.

## Reference

- Johannes Feist, *Two-photon double ionization of helium*, sections 5.5 and
  5.6.
- Einar Aurbakken et al., "Transient spectroscopy from time-dependent
  electronic-structure theory without multipole expansions," *Physical Review
  A* **109**, 013109 (2024), doi:10.1103/PhysRevA.109.013109.
