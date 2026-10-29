# Improved parallelization plan for two-radial TDSE calculations

## Purpose and development order

This plan extends the existing MPI foundation into a production-capable
parallel implementation for two-radial wavefunctions. Development proceeds in
the following strict order:

1. complete the non-Born--Oppenheimer diatomic-ion path;
2. prioritize distributed SIL and SIA for ground-state and real-time dynamics;
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

The present distributed layout retains all angular channels and the complete
electronic coordinate `r` on every rank, while distributing element-aligned
intervals of `R`. This is a one-dimensional process grid and is equivalent to
the `P_r=1` limit of the desired two-dimensional layout.

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
- one-dimensional element-aligned distributed storage in `R`;
- neighboring `R` bridge exchange and weak-form contribution assembly;
- distributed Coulomb actions on locally owned `R` points;
- distributed physical inner products, norms, energies, and residuals;
- distributed SIL imaginary-time propagation;
- local and Saga agreement with serial ground-state calculations;
- distributed-compatible CN preconditioner prototypes.

The following limitations remain:

- the rank count cannot exceed the number of `R` elements;
- every rank retains the complete `r` coordinate;
- the optimized one-dimensional path overlaps neighboring communication but
  the future two-dimensional path must do so independently in both radial
  directions;
- rank-local Coulomb data and the principal one-dimensional work arrays are
  reused, while further kernel-level allocation profiling remains useful;
- Krylov orthogonalization batches basis projections, but norms and iteration
  diagnostics still require latency-sensitive scalar reductions;
- distributed real-time field interactions and absorbers are not complete;
- distributed checkpoint output gathers the complete state to rank zero;
- distributed CN lacks an MPI-aware linear solver;
- the atomic parallel path has not been validated.

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

Increasing `--cpus-per-task` does not help the current pure-MPI path because
each rank performs serial local work. Additional CPUs per task would be
reserved but unused. Hybrid MPI/threading should be considered only after an
explicit threaded local kernel has been implemented and benchmarked.

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

The current one-dimensional layout is effective for angularly expensive
problems but cannot expose more ranks than there are `R` elements. Increasing
the electronic grid increases work and memory on every rank without raising
the rank limit. Increasing the number of `R` elements raises the limit, while
adding points inside existing `R` elements only increases work per rank.

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

The desired shared abstraction is a two-dimensional element layout:

```text
TwoRadialDistributedLayout2D
    process grid: P1 x P2
    all retained angular channels per tile
    contiguous coordinate-1 element range
    contiguous coordinate-2 element range
    owned, bridge, and halo degrees of freedom in both directions
    optional system-specific exchange policy
```

The process-grid aspect ratio should be chosen from element counts, radial
orders, angular workload, memory limits, and estimated halo volume. Equal-size
square blocks are not appropriate for general non-BO grids. Load balancing
should use estimated tile cost and permit multiple connected tiles per rank.

The existing layout remains available as `P1=1`, preserving validation and a
simple low-rank execution path.

## Non-BO diatomic implementation path

The non-BO path must be completed before atomic parallel development begins.

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

**Status:** implemented locally; representative Saga validation remains. The
one-dimensional Hamiltonian now retains rank-local Coulomb slices, reuses its
communication and operator work arrays, separates interior and boundary `R`
elements, and overlaps nonblocking bridge communication with independent
work. Distributed SIL/SIA orthogonalization uses batched inner products and
buffer-based collectives. Thread-communicator equivalence tests and an actual
two-rank MPI smoke test pass. The completion gate still requires repeated
instrumented Saga measurements on representative workloads.

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

1. Generalize ownership metadata so either coordinate can be partitioned.
2. Implement coordinate-1 bridge exchange using the same element-local rules.
3. Retain coordinate-2 partitioning as the default for compatibility.
4. Compare partitioning `r` and partitioning `R` for unequal non-BO grids.

**Completion gate:** both one-dimensional layouts agree with the serial
Hamiltonian and ground state, and the selected coordinate can be archived in
layout metadata.

### Phase 4: implement the two-dimensional `(r,R)` layout

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

### Phase 6: complete distributed SIL/SIA ground-state propagation

1. Generalize distributed SIL imaginary-time propagation to the 2D layout.
2. Use SIL as the production method for the Hermitian ground-state problem.
3. Retain SIA as an independent general-exponential reference.
4. Preserve normalized iteration, energy change, phase-invariant state change,
   and eigenstate-residual convergence checks.
5. Verify conserved `M_tot`, inversion, and applicable pair-exchange sectors.
6. Support initialization from generated states and compatible archives.
7. Compare time to solution, Hamiltonian calls, reductions, and memory with the
   one-dimensional MPI and serial solvers.

Ground-state calculations must not use a CAP.

**Completion gate:** H2+, HD+, and D2+ converge to serial-reference energies
and states on multiple 2D process grids, including at least one multi-node
correctness run after single-node scaling is established.

### Phase 7: complete distributed SIL/SIA dynamics

1. Add the z-polarized length- and velocity-gauge field actions to the
   distributed Hamiltonian.
2. Keep the length-gauge radial factors tile-local, and extend bridge exchange
   to the radial derivatives required by the velocity-gauge action.
3. Evaluate time-dependent Hamiltonians at the temporal midpoint, consistent
   with the serial implementation.
4. Use SIL for Hermitian propagation without a CAP. A separately applied mask
   may be supported with its nonunitary effect recorded through the norm.
5. Use SIA when a CAP makes the Hamiltonian non-Hermitian.
6. Support separate electronic and internuclear masks or CAPs, rejecting a mask
   and CAP on the same coordinate.
7. Record the norm and unnormalized observables. Do not silently convert them
   into conditional expectation values after absorption.
8. Initially support the existing `norm_t` and `expec_z` observables, then add
   further two-radial observables through distributed local contributions and
   reductions.
9. Preserve checkpoint history, field metadata, integrator diagnostics, and
   restart time.
10. Permit a restarted state to be exposed to a different subsequent field,
   while strictly validating representation, grid, angular, symmetry, CODATA,
   and state-shape metadata.

Validation should include:

- field-free stationarity of an archived ground state up to a global phase;
- norm conservation for Hermitian SIL and SIA propagation;
- SIL/SIA agreement for a Hermitian pulse calculation;
- serial/distributed agreement for length and velocity gauge separately;
- convergence with time step and Krylov tolerance;
- CAP-induced norm loss with SIA;
- mask-induced norm loss with SIL or SIA;
- checkpoint/restart equivalence to uninterrupted propagation;
- preservation of conserved `M_tot` for a z-polarized field.

**Completion gate:** a non-BO H2+ pulse calculation can be started from an
archived ground state, propagated and checkpointed with SIL or SIA, restarted,
and reproduced by the serial implementation within specified tolerances.

### Phase 8: implement scalable distributed state I/O

1. Remove the requirement to gather production-scale states to rank zero.
2. Evaluate MPI-enabled HDF5 hyperslab storage against rank-local shards with a
   manifest.
3. Preserve all metadata required by `TwoRadialState` validation.
4. Support restart with a different compatible MPI process grid.
5. Make incomplete checkpoint writes detectable.

**Completion gate:** a multi-rank checkpoint can be written and reloaded
without constructing the full state on one rank, including redistribution to a
different rank count.

### Phase 9: benchmark the complete SIL/SIA path on Saga

Perform controlled studies in this order:

1. fixed radial grids with increasing angular cutoffs;
2. fixed `R` grid with increasing electronic `r` elements and extent;
3. fixed `r` grid with increasing `R` points per existing element;
4. fixed `r` grid with increasing numbers of `R` elements;
5. fixed physical problem with increasing 2D rank count for strong scaling;
6. proportional grid and rank growth for weak scaling;
7. one-node scaling before multi-node scaling;
8. ground-state iteration and accepted real-time-step costs;
9. SIL and SIA comparison without a CAP, followed by SIA with a CAP;
10. checkpoint bandwidth and restart cost.

Report wall time, core time, speedup, efficiency, peak memory per rank,
Hamiltonian calls, collective count, communicated bytes, Krylov dimensions,
accepted/rejected refinements, and observable agreement. Do not infer physical
convergence from a fixed-iteration performance pilot.

**Completion gate:** the report identifies the useful rank range for each
problem size, the onset of communication saturation, and a recommended Saga
job configuration for production non-BO dynamics.

### Phase 10: deploy and benchmark the SIL/SIA path on Olivia and Betzy

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
   two-, and four-rank H2+ correctness runs. Require agreement with serial and
   Saga reference energies, norms, residuals, and phase-aligned states.
8. Repeat the HD+ sanity check to cover unequal nuclear masses and the absence
   of homonuclear pair-exchange adaptation.
9. Run a representative subset of the complete Saga SIL/SIA benchmark matrix
   on each cluster: angular growth, radial growth, strong scaling, ground-state
   propagation, real-time dynamics without a CAP, SIA dynamics with a CAP, and
   distributed checkpoint I/O.
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

### Phase 11: add distributed Crank--Nicolson

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

1. non-BO ground states and laser-driven dynamics run with distributed SIL and
   SIA on rectangular 2D grids;
2. masks, CAPs, observables, checkpoints, and restarts operate without a
   rank-zero state bottleneck;
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
