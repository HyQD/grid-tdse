# Non-BO diatomic MPI validation and scaling on Saga

## Scope

This report archives the one-dimensional, internuclear-coordinate MPI results
obtained before the two-dimensional radial decomposition is introduced. It is
the Phase 1 reference for the non-Born--Oppenheimer parallelization path in
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

The current scalar reduction uses the Python-object
`communicator.allreduce(value)` interface, including for a one-rank MPI
communicator. The Phase 2 priorities are therefore to bypass the one-rank
collective, use buffer-based collectives, and batch Krylov inner products into
fewer global reductions.

The one-rank repetitions all ran on `c5-60`, and the two-rank repetitions all
ran on `c1-30`. The four-rank repetitions ran on `c1-30`, `c1-36`, and `c1-29`.
The `c1-29` repetition took `270.433658` seconds compared with `253.186881`
and `248.631411` seconds on the other nodes. The numerical results were
identical, so the spread is retained as node-to-node variability rather than
discarded. These repeated measurements satisfy the Phase 1 instrumentation
completion gate.

## Data

The machine-readable records are in
[`data/saga_non_bo_diatomic_mpi_scaling.csv`](data/saga_non_bo_diatomic_mpi_scaling.csv).
Raw Slurm output is intentionally not tracked.
