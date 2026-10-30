# Non-BO diatomic MPI validation and scaling on Saga

## Scope

This report archives the one-dimensional, internuclear-coordinate MPI results
obtained before the two-dimensional radial decomposition is introduced. It
contains the Phase 1 reference and the Phase 2 optimized-path validation for
the non-Born--Oppenheimer parallelization path in
[`dev/improved_parallelization_plan.md`](../dev/improved_parallelization_plan.md).

The current layout retains every angular channel and the complete electronic
coordinate `r` on each rank. Contiguous FEM-DVR elements of the internuclear
coordinate `R` are distributed, so the maximum useful rank count equals the
number of `R` elements.

## Runtime configuration

The validated Saga jobs used:

- `OpenMPI/5.0.10-GCC-15.2.0`;
- `mpi4py` 4.1.2 built against that module;
- `srun --mpi=pmix`;
- one CPU and one numerical-library thread per MPI rank;
- rank-local Numba cache directories; and
- core binding through Slurm.

The angular-scaling calculations used the following grids:

```text
r boundaries: [0, 4, 8, 12, 16], 11 points per element
R boundaries: [0, 2, 4, 6, 8],    21 points per element
```

They used `M_tot=0`, the symmetry-adapted gerade/even H2+ sector, and

```text
m_max = l_max
lambda_max = 2*l_max
mu_max = lambda_max.
```

The original job outputs did not record the Git commit or every node name.
Those missing provenance fields are left empty in the accompanying CSV rather
than reconstructed after the fact.

The instrumented Phase 1 series was run on 2026-10-29 at commit
`1b3e869e85e0ef0a4278e04f54e31d96a7447a5f`. The MPI runtime identified
itself as Open MPI 5.0.10, revision `v5.0.10`, built on 2026-02-23.

## Correctness validation

An earlier validation grid produced the same H2+ energy,
`-0.579827137696` Eh, after 1903 iterations with one, two, and four ranks.
Energy differences from the serial solver were at most `4.441e-16` Eh and
phase-aligned state differences were at most `1.897e-11`.

The complementary HD+ calculation converged to `-0.566541180708` Eh with one,
two, four, and eight ranks. The serial energy differences were at most
`3.331e-16` Eh, phase-aligned state differences were below `1.2e-11`, and all
runs retained unit norm. This validates the nonidentical-nuclei path as well as
the homonuclear symmetry-adapted path.

The one-dimensional distributed correctness gate is therefore complete for
H2+ and HD+.

The instrumented two-rank H2+ correctness job `20708277` independently
reproduced the serial energy to `2.220e-16` Eh and the phase-aligned state to
`9.223e-12`. Its distributed and serial solve times were `293.089293` and
`240.434689` seconds, respectively. This small `l_max=2` calculation therefore
does not amortize the distributed-reduction overhead.

## Angular strong scaling

Representative maximum-rank solve times per imaginary-time iteration were:

| `l_max` | measurement | 1 rank | 2 ranks | 4 ranks | 4-rank speedup | 4-rank efficiency |
|---:|:---|---:|---:|---:|---:|---:|
| 2 | converged | 0.122013 s | 0.102085 s | 0.083818 s | 1.46 | 36% |
| 4 | converged | 0.950942 s | 0.553180 s | 0.336337 s | 2.83 | 71% |
| 6 | fixed 100 steps | 15.815445 s | 8.224047 s | 4.444762 s | 3.56 | 89% |

The `l_max=2` and `l_max=4` calculations converged in 2938 and 3137
iterations, respectively. The `l_max=6` calculation was deliberately stopped
after 100 iterations. Its energy and residual are performance-pilot values,
not a converged physical result.

The representative two-rank `l_max=6` value is the median of normal-node
measurements `7.942116`, `8.224047`, and `8.809443` seconds per iteration. Two
additional jobs on node `c1-2` required approximately `22.95` seconds per
iteration while producing identical numerical results and high CPU
utilization. Those measurements are retained as node-performance outliers and
are not used for the representative scaling value.

The results show that MPI overhead is not amortized by the smallest angular
calculation, whereas the expensive angular Hamiltonian at `l_max=6` scales
well through the four available `R` elements.

## Phase-level instrumentation

The MPI ground-state example now supports opt-in profiling inside the
distributed solver and reports maximum accumulated rank time for:

- electronic-coordinate work;
- local internuclear-coordinate work;
- Coulomb work;
- state-halo exchange;
- weak-contribution exchange;
- local physical-inner-product work; and
- collective reductions.

It also reports Hamiltonian calls and the SIL operator-call, accepted-substep,
rejected-substep, refinement-depth, Krylov-dimension, and error-estimate
diagnostics. The warm-up action is reset out of these counters.

The instrumented series used the converged `l_max=2` H2+ calculation and three
repetitions at each rank count. All nine calculations converged in 2938
iterations to `-0.577283887513` Eh with unit norm and residuals below
`9.954e-09`.

Median solve times and strong-scaling metrics relative to the one-rank
distributed calculation were:

| ranks | median solve time | median time/iteration | speedup | efficiency |
|---:|---:|---:|---:|---:|
| 1 | 373.335462 s | 0.127071 s | 1.000 | 100% |
| 2 | 290.130479 s | 0.098751 s | 1.287 | 64% |
| 4 | 253.186881 s | 0.086177 s | 1.475 | 37% |

The median maximum-rank component times were:

| component | 1 rank | 2 ranks | 4 ranks |
|:---|---:|---:|---:|
| electronic work | 51.091097 s | 34.076490 s | 28.643018 s |
| internuclear work | 42.400392 s | 23.238635 s | 15.253308 s |
| Coulomb work | 23.295633 s | 14.584868 s | 9.535839 s |
| halo exchange | 0.000000 s | 3.218578 s | 4.087024 s |
| weak exchange | 0.000000 s | 1.602589 s | 2.354792 s |
| local reductions | 134.549954 s | 101.693416 s | 86.521676 s |
| collective reductions | 38.445198 s | 69.443636 s | 74.409232 s |

The phase maxima are diagnostic and are not generally additive because a
different rank can be slowest for each phase. They nevertheless identify the
reduction path as the dominant scaling limit. Approximately 405000--408000
physical-inner-product reductions were performed per solve, or about 10.2
reductions per Hamiltonian application. Useful Hamiltonian work decreased with
rank, and halo plus weak-form exchange remained small, while collective time
increased from one to two ranks and did not improve at four ranks.

The Phase 1 scalar reduction used the Python-object
`communicator.allreduce(value)` interface, including for a one-rank MPI
communicator. These measurements motivated the Phase 2 work to bypass the
one-rank collective, use buffer-based collectives, and batch Krylov inner
products into fewer global reductions.

The one-rank repetitions all ran on `c5-60`, and the two-rank repetitions all
ran on `c1-30`. The four-rank repetitions ran on `c1-30`, `c1-36`, and `c1-29`.
The `c1-29` repetition took `270.433658` seconds compared with `253.186881`
and `248.631411` seconds on the other nodes. The numerical results were
identical, so the spread is retained as node-to-node variability rather than
discarded. These repeated measurements satisfy the Phase 1 instrumentation
completion gate.

## Phase 2 optimized one-dimensional path

Phase 2 retained rank-local Coulomb slices, reused the principal Hamiltonian
and communication buffers, overlapped bridge communication with independent
work, bypassed one-rank MPI reductions, and batched Krylov projections into
buffer-based collectives. The optimized implementation was measured with the
same `l_max=2` H2+ problem and three repetitions at each rank count.

The Phase 2 series was prepared at `2026-10-29T21:02:37+01:00` with:

- grid-tdse commit `dc2c9871c537765b964494d81297ef1988046a07`;
- integrator-lib commit `04a161fbaa42955c68e374b391bcd76d77aa8edc`;
- integrator-lib version `0.3.0`;
- `OpenMPI/5.0.10-GCC-15.2.0`; and
- Open MPI 5.0.10 revision `v5.0.10`, built on 2026-02-23.

All nine repeated calculations converged in 2938 iterations to
`-0.577283887513` Eh with residuals at or below `9.953e-09`. The separate
two-rank correctness job `20723663` retained unit norm and agreed with the
serial calculation to `1.110e-16` Eh in energy and `6.321e-12` in the
phase-aligned state.

The Phase 2 median solve times and strong-scaling metrics were:

| ranks | median solve time | median time/iteration | speedup | efficiency |
|---:|---:|---:|---:|---:|
| 1 | 208.698799 s | 0.071034 s | 1.000 | 100% |
| 2 | 131.408640 s | 0.044727 s | 1.588 | 79% |
| 4 | 114.735577 s | 0.039052 s | 1.819 | 45% |

Relative to the Phase 1 medians, the complete distributed solve improved by:

| ranks | Phase 1 | Phase 2 | improvement | wall-time reduction |
|---:|---:|---:|---:|---:|
| 1 | 373.335462 s | 208.698799 s | 1.79x | 44% |
| 2 | 290.130479 s | 131.408640 s | 2.21x | 55% |
| 4 | 253.186881 s | 114.735577 s | 2.21x | 55% |

The Phase 2 median maximum-rank component times were:

| component | 1 rank | 2 ranks | 4 ranks |
|:---|---:|---:|---:|
| electronic work | 56.054248 s | 37.649037 s | 35.890121 s |
| internuclear work | 45.582452 s | 21.956427 s | 17.540019 s |
| Coulomb work | 19.135526 s | 11.573115 s | 7.204206 s |
| halo exchange | 0.000000 s | 3.001203 s | 4.595528 s |
| weak exchange | 0.000000 s | 2.633486 s | 3.737757 s |
| local reductions | 17.290535 s | 10.020739 s | 7.090590 s |
| collective reductions | 0.000000 s | 9.496208 s | 7.843090 s |

One-rank execution now bypasses MPI collectives entirely. At two ranks,
`405971` scalar values were reduced with `171465` collective calls; at four
ranks, `405438` values were reduced with `171157` calls. The amount of physical
projection work therefore remained comparable to Phase 1 while the collective
count fell by approximately 58%.

Median collective time decreased from `69.443636` to `9.496208` seconds at two
ranks and from `74.409232` to `7.843090` seconds at four ranks. Median local
reduction time improved by factors of approximately 7.8, 10.1, and 12.2 at
one, two, and four ranks. Precomputed local Coulomb slices reduced median
Coulomb time by approximately 18%--24%.

The phase maxima remain non-additive, and node-to-node variation is visible in
the repeated two- and four-rank measurements. Medians are consequently used
for all comparisons. For this small workload, two ranks provide the better
allocation-cost compromise: four ranks reduce median wall time by another 13%
but use substantially more total core time.

These correctness and repeated performance measurements satisfy the Phase 2
completion gate.

## Phase 3 coordinate-selectable partitioning

Phase 3 generalized the one-dimensional decomposition so either the electronic
coordinate `r` or the internuclear coordinate `R` can be distributed. The
implementation was introduced by commit
`3f2cb7d03325982019becc295aff3676fe621f55` and merged to `main` by commit
`46d7be6`. The exact `HEAD` checked
out for the Saga jobs was not captured, so the accompanying CSV deliberately
leaves the runtime-commit field empty.

The Saga validation used the same `l_max=2` H2+ representation and radial grids
listed above. Fixed 20-iteration smoke tests were run with one, two, and four
ranks for both partitioned coordinates. Every completed run produced

```text
Energy:              -0.357645049472 Eh
State change:         1.562e-02
Eigenstate residual:  2.926e-02
Norm:                 1.000000000000
```

The phase-aligned differences from the serial calculation were between
`3.088e-12` and `5.085e-12`; energy differences were between `2.109e-15` and
`1.094e-14` Eh. These are fixed-iteration correctness results, not converged
ground-state energies.

The smoke-test solve times were:

| partitioned coordinate | 1 rank | 2 ranks | 4 ranks |
|:---:|---:|---:|---:|
| `r` | 4.808367 s | 3.147618 s | 2.451119 s |
| `R` | 5.049671 s | 3.227014 s | 2.519233 s |

The original two-rank `r` job, `20728584`, encountered a transient shared-file
failure while importing `coverage.inorout` from the Conda environment. The file
was present and importable when the environment was inspected subsequently.
The clean retry, job `20728607`, completed normally and agreed with the serial
calculation. The incident is retained in the data record as an environment
failure and is not treated as a distributed-solver failure.

Fully converged two-rank calculations then validated both layouts against the
serial solver:

| partitioned coordinate | job | iterations | energy | residual | solve time | time/iteration | serial energy difference | serial state difference |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r` | 20728610 | 2938 | -0.577283887513 Eh | 9.953e-09 | 135.308057 s | 0.046054 s | 1.110e-16 Eh | 6.126e-12 |
| `R` | 20728611 | 2938 | -0.577283887513 Eh | 9.952e-09 | 138.816755 s | 0.047249 s | 1.110e-16 Eh | 6.321e-12 |

Both calculations retained unit norm and reached a state change of
`4.990e-09`. The `r`-partitioned run was approximately 2.5% faster than the
`R`-partitioned run, but a single timing per layout is insufficient to infer a
general performance preference. The important result is full-convergence and
serial equivalence for both ownership directions. Phase 3's correctness gate
is therefore complete; repeated unequal-grid performance comparisons remain
part of the later benchmark phases.

## Data

The Phase 1 machine-readable records are in
[`data/saga_non_bo_diatomic_mpi_scaling.csv`](data/saga_non_bo_diatomic_mpi_scaling.csv).
The Phase 2 records are in
[`data/saga_non_bo_diatomic_mpi_phase2.csv`](data/saga_non_bo_diatomic_mpi_phase2.csv).
The Phase 3 coordinate-partitioning records are in
[`data/saga_non_bo_diatomic_mpi_phase3.csv`](data/saga_non_bo_diatomic_mpi_phase3.csv).
Raw Slurm output is intentionally not tracked.
