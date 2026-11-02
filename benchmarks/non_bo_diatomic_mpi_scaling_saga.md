# Non-BO diatomic MPI validation and scaling on Saga

## Scope

This report archives Saga validation and scaling results for Phases 1--7 and
the Phase 9a dynamics-initialization deployment gate of
the non-Born--Oppenheimer parallelization path in
[`dev/improved_parallelization_plan.md`](../dev/improved_parallelization_plan.md).
Phases 1--3 cover the one-dimensional decomposition and its coordinate
selection. Phase 4 validates two-dimensional process-grid topology and state
collectives, Phase 5 validates the distributed two-dimensional Hamiltonian,
Phase 6 compares serial, threaded, pure-MPI, and hybrid Hamiltonian execution
at fixed allocated core counts, Phase 7 validates complete Cartesian SIL and
SIA ground-state propagation, and Phase 9a validates serial Davidson
initialization and Cartesian distribution of the enlarged dynamics state.

The Phase 1 and Phase 2 layout retains every angular channel and the complete
electronic coordinate `r` on each rank. Contiguous FEM-DVR elements of the
internuclear coordinate `R` are distributed, so the maximum useful rank count
equals the number of `R` elements. Phase 3 permits either radial coordinate to
be distributed, Phase 4 partitions both over a Cartesian process grid, and
Phase 5 applies the Hamiltonian on that layout.

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

## Phase 4 Cartesian two-coordinate layout

Phase 4 introduced `TwoRadialDistributed2DLayout`, which partitions both
radial coordinates over a rectangular `P_r x P_R` process grid. These Saga
jobs validate topology selection and state collectives only; they do not apply
the Hamiltonian and are not performance benchmarks.

The validation was prepared at `2026-10-30T12:49:54+01:00` with:

- grid-tdse commit `e8f78946d4ede751a6d6a5141c92603c42994fe7`;
- grid-lib commit `8950c3ae01425488a6d78ba054a3b394dd851727`;
- integrator-lib commit `04a161fbaa42955c68e374b391bcd76d77aa8edc`;
- `OpenMPI/5.0.10-GCC-15.2.0`;
- Open MPI 5.0.10 revision `v5.0.10`, built on 2026-02-23; and
- the project-local `grid_tdse` Python environment.

The jobs used the same radial grids listed above, H2+, `M_tot=0`, and
`l_max=m_max=lambda_max=mu_max=0`. The resulting state contained 3081 complex
coefficients. Three single-node jobs ran on Saga node `c1-19`:

| job | ranks | initial process grid | final process grid | local coefficients | phase-aligned difference | exact match | elapsed |
|---:|---:|:---:|:---:|---:|---:|:---:|---:|
| 20729132 | 3 | automatic `(3, 1)` | `(3, 1)` | 790--1501 | 0.000e+00 | true | 31 s |
| 20729144 | 4 | `(2, 2)` | `(1, 4)` | 741--780 | 0.000e+00 | true | 24 s |
| 20729149 | 4 | `(2, 2)` | `(4, 1)` | 711--790 | 0.000e+00 | true | 24 s |

All jobs completed normally. The automatic three-rank case selected `(3, 1)`.
Its larger ownership range is expected when four indivisible FEM-DVR elements
are assigned to three contiguous partitions: one rank must own two elements.
Both four-rank jobs exactly recovered the original state after redistribution
from the balanced `(2, 2)` layout to either one-dimensional extreme. This
checks both radial ownership directions and the collective redistribution
path on the cluster MPI stack.

The reported elapsed times include environment and MPI startup. They must not
be interpreted as layout scaling results because the example performs no
repeated Hamiltonian work. The CSV records `MaxRSS` from each application step,
because the top-level Slurm job rows did not report it. These results complete
the Phase 4 Saga deployment gate and establish the state-layout baseline for
the Phase 5 two-dimensional Hamiltonian.

## Phase 5 Cartesian two-coordinate Hamiltonian

Phase 5 introduced `TwoRadialDistributed2DHamiltonian`, which applies both
element-local radial operators and the Coulomb interaction on Cartesian
two-coordinate state tiles. These short Saga jobs are correctness and
deployment tests rather than scaling measurements. Each job scattered the
same deterministic complex state, evaluated the distributed components and
complete Hamiltonian, gathered the results, and compared them with the serial
element-local implementation.

The validation was prepared at `2026-10-30T13:23:55+01:00` with:

- grid-tdse commit `fafa53dc8602a9c0f2ba37ea57f78e3531ccdaed`;
- grid-lib commit `8950c3ae01425488a6d78ba054a3b394dd851727`;
- integrator-lib commit `04a161fbaa42955c68e374b391bcd76d77aa8edc`;
- `OpenMPI/5.0.10-GCC-15.2.0`;
- Open MPI 5.0.10 revision `v5.0.10`, built on 2026-02-23; and
- the project-local `grid_tdse` Python environment.

The electronic grid used bounds `(0, 2, 5)` with five FEM-DVR points per
element, and the internuclear grid used bounds `(0, 1.5, 4)` with five points
per element. The calculations used `M_tot=0`, analytic radial Coulomb
multipoles, and deterministic seed 1729. Identical nuclei used the
symmetry-adapted basis and positive pair-exchange parity; HD+ retained the
unreduced angular space.

| job | system | `l_max` | ranks | process grid | global coefficients | local coefficients | coordinate 1 error | coordinate 2 error | Coulomb error | complete error | elapsed |
|---:|:---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 20730392 | H2+ | 1 | 1 | `(1, 1)` | 49 | 49--49 | 3.662e-15 | 0.000e+00 | 0.000e+00 | 3.662e-15 | 50 s |
| 20730394 | H2+ | 1 | 2 | `(1, 2)` | 49 | 21--28 | 3.662e-15 | 1.144e-16 | 0.000e+00 | 3.662e-15 | 35 s |
| 20730397 | H2+ | 1 | 2 | `(2, 1)` | 49 | 21--28 | 3.553e-15 | 0.000e+00 | 0.000e+00 | 3.553e-15 | 29 s |
| 20730400 | H2+ | 1 | 4 | `(2, 2)` | 49 | 9--16 | 3.553e-15 | 1.144e-16 | 0.000e+00 | 3.553e-15 | 29 s |
| 20730446 | H2+ | 2 | 4 | `(2, 2)` | 392 | 72--128 | 7.944e-15 | 2.289e-16 | 0.000e+00 | 7.944e-15 | 34 s |
| 20730452 | HD+ | 1 | 4 | `(2, 2)` | 294 | 54--96 | 7.324e-15 | 2.220e-16 | 0.000e+00 | 7.324e-15 | 32 s |
| 20730453 | D2+ | 1 | 4 | `(2, 2)` | 49 | 9--16 | 3.553e-15 | 1.241e-16 | 0.000e+00 | 3.553e-15 | 28 s |

All seven jobs completed normally. Both one-coordinate decompositions and the
genuinely two-dimensional `(2, 2)` decomposition reproduce the serial action.
The largest complete-action discrepancy was `7.944e-15`, and every Coulomb
comparison was zero at printed precision. The larger HD+ state is expected:
unlike H2+ and D2+, the heteronuclear system cannot use identical-particle
pair-exchange reduction.

The elapsed times include module, environment, MPI, and Numba startup and must
not be interpreted as Hamiltonian scaling measurements. The CSV records
`MaxRSS` from each application step because the top-level Slurm job rows did
not report it. These results complete the Phase 5 Saga deployment gate.

## Phase 6 shared-memory and hybrid execution baseline

Phase 6 compared serial, shared-memory, pure-MPI, and hybrid execution of the
two-dimensional Hamiltonian at equal allocated core counts. Each configuration
performed five warm-up actions followed by 5000 timed actions and was repeated
three times. All 60 jobs completed normally on one Saga node per job.

The calculations used H2+, `M_tot=0`, the radial grids listed above, Poisson
Coulomb multipoles, and grid-tdse commit
`fda07b85e48c381c269fb299f8c90313dea03c93`. The runtime used Python 3.14.9,
NumPy 2.5.2, SciPy 1.18.0, OpenBLAS 0.3.34, and
`OpenMPI/5.0.10-GCC-15.2.0`. Slurm launched ranks with `srun --mpi=pmix` and
core binding. OpenMP, OpenBLAS, and MKL thread counts equaled the assigned
threads per rank; Numba remained at one thread because the current Numba
kernels are serial.

Every configuration agreed with the serial element-local Hamiltonian. The
maximum absolute action error was `1.172e-13` for `l_max=2` and `2.274e-13`
for `l_max=4`. The workloads contained 24648 and 89349 complex coefficients,
respectively.

The median wall times per action were:

| `l_max` | cores | shared memory | best hybrid | pure MPI | pure-MPI speedup |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | -- | -- | 4.466 ms | 1.000 |
| 2 | 2 | 4.717 ms | -- | 3.030 ms | 1.474 |
| 2 | 4 | 4.529 ms | 2.980 ms | 2.260 ms | 1.976 |
| 2 | 8 | 3.972 ms | 2.182 ms | 1.659 ms | 2.692 |
| 4 | 1 | -- | -- | 28.124 ms | 1.000 |
| 4 | 2 | 24.062 ms | -- | 11.332 ms | 2.482 |
| 4 | 4 | 29.135 ms | 11.202 ms | 6.237 ms | 4.509 |
| 4 | 8 | 28.000 ms | 7.312 ms | 3.915 ms | 7.183 |

Here the one-core row is the serial reference, represented as one MPI rank
with one thread. Pure MPI was the fastest execution model at every core count
for both workloads. For `l_max=4`, it was approximately 2.12 times faster than
shared memory at two cores, 1.80 times faster than the best hybrid
configuration at four cores, and 1.87 times faster than the best hybrid
configuration at eight cores.

The superlinear reported `l_max=4` speedups at two and four cores should not be
read as algorithmic parallel efficiencies above 100%. Timing spreads for that
series ranged from approximately 15% to 31% for many threaded and distributed
configurations, reflecting cache, node, and runtime variation. The archive
retains all repetitions and uses medians without removing outliers. The gaps
between pure MPI and the alternatives are nevertheless large enough that the
execution-model ranking is unambiguous.

Increasing numerical-library thread counts did not materially accelerate the
dominant element-local work. The shared-memory timings were nearly flat, and
the eight-thread `l_max=2` run used only about 50% of its allocated-core
capacity. Hybrid performance mainly followed its MPI rank count. This is
consistent with the current Numba kernels being serial and BLAS threading not
covering the dominant Hamiltonian work.

The Phase 6 recommendation for subsequent SIL/SIA development on Saga is
therefore pure MPI: one physical CPU core per rank, one thread for OpenMP,
OpenBLAS, MKL, and Numba, core binding, and a process grid chosen for the two
radial dimensions. A hybrid option is not justified by these Hamiltonian-only
measurements. The comparison must be repeated for complete ground-state and
real-time propagation, where Krylov reductions and time-dependent work may
change the balance. Pure MPI also increases aggregate memory with rank count,
even though memory per rank decreases, so memory remains a production-sizing
constraint.

These results satisfy the Phase 6 Hamiltonian-level completion gate.

## Phase 7 Cartesian SIL/SIA ground-state propagation

Phase 7 connected the Cartesian state layout and Hamiltonian to normalized
SIL and SIA imaginary-time propagation, batched physical reductions, and
distributed observable sampling. The Saga validation used grid-tdse commit
`ab8bc5c49db3bfe096b10137d89f4ce17ec498c2`, grid-lib commit
`8950c3ae01425488a6d78ba054a3b394dd851727`, integrator-lib commit
`04a161fbaa42955c68e374b391bcd76d77aa8edc`, Python 3.14.9, mpi4py 4.1.2,
and `OpenMPI/5.0.10-GCC-15.2.0`. Each rank used one CPU core and one numerical
thread. The full calculations used the radial grids and angular-cutoff
relations documented above with `l_max=2`.

Four-rank `(2,2)` smoke tests first advanced the `l_max=0` H2+ state for 20
steps. SIL and SIA both reached `-0.267367161696` Eh with unit norm, state
change `1.802e-02`, and residual `3.441e-02`. Their distributed energies
agreed with their serial references within `2.220e-16` Eh, and their
phase-aligned states agreed within `2.423e-12`. These fixed-step smoke values
are correctness checks, not converged ground-state results.

The converged SIL results were:

| system | process grid | ranks | iterations | energy (Eh) | solve time | serial state difference |
|:---|:---:|---:|---:|---:|---:|---:|
| H2+ | `1x2` | 2 | 2938 | -0.577283887513 | 182.588 s | 6.884e-12 |
| H2+ | `2x1` | 2 | 2938 | -0.577283887513 | 175.976 s | 5.571e-12 |
| H2+ | `2x2` | 4 | 2938 | -0.577283887513 | 117.252 s | 1.037e-11 |
| HD+ | `1x2` | 2 | 2715 | -0.571643021288 | 251.454 s | 1.264e-11 |
| HD+ | `2x1` | 2 | 2715 | -0.571643021288 | 236.261 s | 9.201e-12 |
| HD+ | `2x2` | 4 | 2715 | -0.571643021288 | 142.630 s | 6.632e-12 |
| D2+ | `1x2` | 2 | 3443 | -0.579201162618 | 190.402 s | 6.655e-12 |
| D2+ | `2x1` | 2 | 3443 | -0.579201162618 | 188.258 s | 8.856e-12 |
| D2+ | `2x2` | 4 | 3443 | -0.579201162618 | 122.364 s | 7.817e-12 |

Every run converged with unit norm and an eigenstate residual below
`9.987e-09`. Process-grid orientation did not change the energy or iteration
count. Serial energy differences were at most `2.220e-16` Eh. The `(2,2)`
layout was fastest for all three ions, while `(2,1)` was only modestly faster
than `(1,2)`.

The converged four-rank H2+ SIA reference reproduced the SIL energy and
iteration count. Its distributed solve took `168.942` seconds, compared with
`117.252` seconds for SIL on the same process grid. SIA reduced 639900 scalar
values in 171312 collectives, while SIL reduced 407120 values in 171752
collectives. The larger Arnoldi projection batches account for part of the
additional SIA cost and support retaining SIL as the production Hermitian
ground-state method.

A four-rank one-dimensional `R`-partitioned SIL baseline took `112.761`
seconds and agreed with serial to `6.373e-12` in the state. The Cartesian
`(2,2)` result took approximately 4% longer in this single comparison and
spent `13.907` rather than `8.889` seconds in collective reductions. This is a
modest low-rank overhead; the two-dimensional layout removes the four-rank
limit of the one-coordinate decomposition. These are single timing samples,
not a repeated performance ranking.

Distributed interval sampling preserved the requested symmetry sector. H2+
and D2+ ended with `<I>=1` and `<P>=1`; HD+ ended with `<I>=1` and correctly
reported `<P>` as unavailable. All three field-free states had zero total
angular momentum projection, zero vector dipoles, `<cos(theta_R)>=0`, and
`<cos^2(theta_R)>=1/3`. The final radial and angular expectations were:

| system | `<r>` | `<R>` | `<l_r^2>` | `<l_R^2>` |
|:---|---:|---:|---:|---:|
| H2+ | 1.382695358159 | 1.912485692182 | 0.184928443907 | 0.184928443907 |
| HD+ | 1.406939370008 | 1.949478261595 | 0.189809210382 | 0.189809210382 |
| D2+ | 1.373808073659 | 1.889882474318 | 0.179817078672 | 0.179817078672 |

Finally, a four-rank H2+ `(2,2)` calculation placed two ranks on each of Saga
nodes `c1-30` and `c1-31`. It converged in 2938 iterations to the same energy,
unit norm, and `9.953e-09` residual as the single-node calculations. Its
serial energy and state differences were `2.220e-16` Eh and `1.037e-11`.
This establishes MPI correctness across a network boundary. Its `104.146`
second solve time is retained as a correctness measurement and is not treated
as evidence of multi-node acceleration without repetitions.

These results satisfy the Phase 7 completion gate: H2+, HD+, and D2+ converge
to their serial references on multiple Cartesian process grids, SIL and SIA
agree for H2+, distributed observables and logging are operational, and a
multi-node correctness calculation passes.

## Phase 9a dynamics initialization

Phase 9a introduced a dynamics representation with both even and odd
electronic angular channels, metadata-validated Davidson state archives, and
the `require`, `compute-if-missing`, and `recompute` initialization policies.
In MPI execution, rank zero alone loads or computes the serial state before a
barrier and scatter to the Cartesian layout.

The Saga deployment validation ran on 2026-11-02 with Python 3.14.9, NumPy
2.5.2, SciPy 1.18.0, mpi4py, and `OpenMPI/5.0.10-GCC-15.2.0`. It used a small
correctness problem with two five-point elements on each radial coordinate,
`l_max=m_max=1`, `lambda_max=mu_max=2`, `M_tot=0`, and CODATA 2022. The
four-rank jobs used a `2 x 2` process grid. The implementation was merged to
`main` by commit `8e000da`, but the run directory did not record the exact
runtime checkout SHA; it is deliberately not reconstructed here.

The accepted retry set comprised eight jobs:

| job | system | mode | policy | source | Davidson starts | setup time |
|---:|:---:|:---:|:---|:---:|---:|---:|
| 20744797 | H2+ | serial seed | `recompute` | computed | 1 | 4.797 s |
| 20744798 | HD+ | serial seed | `recompute` | computed | 1 | 5.500 s |
| 20744799 | D2+ | serial seed | `recompute` | computed | 1 | 3.508 s |
| 20744800 | H2+ | Cartesian MPI | `compute-if-missing` | computed | 1 | 3.334 s |
| 20744801 | H2+ | Cartesian MPI | `require` | archive | 0 | 0.0050 s |
| 20744802 | H2+ | Cartesian MPI | `recompute` | computed | 1 | 3.253 s |
| 20744803 | HD+ | Cartesian MPI | `require` | archive | 0 | 0.0055 s |
| 20744804 | D2+ | Cartesian MPI | `require` | archive | 0 | 0.0052 s |

All eight jobs completed with exit code `0:0` and generated their expected
JSON records. Every state had norm `1.0000000000000002`. The four-rank jobs
had zero phase-aligned serial-state difference and zero maximum coefficient
error at printed precision. H2+ and D2+ retained internuclear-coordinate
parity `+1`; HD+ correctly used no pair-exchange restriction. All dynamics
representations retained `M_tot=0` and unrestricted electronic-coordinate
parity so both field-coupled parity sectors are present.

The three serial seed jobs also verified that velocity-gauge initialization
rejects nonzero `A(t0)`. Counting the Davidson progress marker in each output
confirmed that computation occurred exactly once in each seed,
`compute-if-missing`, and `recompute` job and never in a `require` job. This
validates the root-only setup contract rather than merely numerical state
equivalence.

The first three submitted seed jobs completed Davidson but failed while the
temporary validation worker serialized a NumPy scalar to JSON. The worker was
patched to convert the norm to a Python `float` and to honor the requested
seed policy. The original diagnostic outputs, patched worker, hash, and clean
retry records are retained in the archive. The incident did not expose a
grid-tdse library failure.

These results satisfy the Phase 9a completion gate: archived and automatically
generated Davidson ground states produce identical serial and Cartesian-MPI
initial dynamics states for all three supported diatomic ions.

## Data

The Phase 1 machine-readable records are in
[`data/saga_non_bo_diatomic_mpi_scaling.csv`](data/saga_non_bo_diatomic_mpi_scaling.csv).
The Phase 2 records are in
[`data/saga_non_bo_diatomic_mpi_phase2.csv`](data/saga_non_bo_diatomic_mpi_phase2.csv).
The Phase 3 coordinate-partitioning records are in
[`data/saga_non_bo_diatomic_mpi_phase3.csv`](data/saga_non_bo_diatomic_mpi_phase3.csv).
The Phase 4 Cartesian-layout records are in
[`data/saga_non_bo_diatomic_mpi_phase4.csv`](data/saga_non_bo_diatomic_mpi_phase4.csv).
The Phase 5 Cartesian-Hamiltonian records are in
[`data/saga_non_bo_diatomic_mpi_phase5.csv`](data/saga_non_bo_diatomic_mpi_phase5.csv).
The Phase 6 execution-model summary is in
[`data/saga_non_bo_diatomic_hamiltonian_phase6.csv`](data/saga_non_bo_diatomic_hamiltonian_phase6.csv),
with its compact runtime and problem provenance in
[`data/saga_non_bo_diatomic_hamiltonian_phase6_provenance.json`](data/saga_non_bo_diatomic_hamiltonian_phase6_provenance.json).
The Phase 7 Cartesian ground-state records are in
[`data/saga_non_bo_diatomic_mpi_phase7.csv`](data/saga_non_bo_diatomic_mpi_phase7.csv),
with runtime, problem, and raw-archive provenance in
[`data/saga_non_bo_diatomic_mpi_phase7_provenance.json`](data/saga_non_bo_diatomic_mpi_phase7_provenance.json).
The compact Phase 9a validation archive, including its Slurm output, result
JSON, worker, manifests, and small reproducibility states, is in
[`data/saga_non_bo_diatomic_phase9a/phase9a-20261102-004030`](data/saga_non_bo_diatomic_phase9a/phase9a-20261102-004030).
Raw Slurm output from the larger Phase 1--7 scaling series is intentionally
not tracked.
