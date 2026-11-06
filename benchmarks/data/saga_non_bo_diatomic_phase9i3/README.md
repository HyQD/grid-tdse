# Saga Phase 9i.3 batched Krylov validation

This directory archives the controlled Saga validation of batched physical
inner-product reductions in the Cartesian-MPI real-time SIL/SIA path. The
runs were made on 2026-11-06 at grid-tdse commit
`b60655d4bd64eb8f545ddca9fa7df638c792eda1`.

All 20 submitted jobs completed normally: each reduction mode used one
Davidson ground-state seed job and three repetitions at one, two, and four
MPI ranks. No traceback, assertion failure, import failure, MPI abort,
cancellation, timeout, or out-of-memory marker was found. The runs used
Python 3.14.9, NumPy 2.5.2, Numba 0.67.0, Open MPI 5.0.10, and the `py3nj`
Gaunt backend.

## Workload

Both matrices used the same production-style H2+ calculation:

- `l_max=2` and the end-to-end example's default radial grids;
- length-gauge laser interaction and complex absorbing potential;
- SIA propagation with `dt=0.25` to a final time of 20 a.u.;
- 80 requested timesteps;
- reference Coulomb, radial, and dipole kernels; and
- wavefunction snapshots, checkpoints, and interactive plotting disabled.

The only intentional difference was the Krylov projection strategy. The
scalar matrix reduced every projection separately, whereas the batched matrix
used one collective for each group of projections.

## Performance

| ranks | grid | scalar wall time | batched wall time | speedup | wall-time reduction | scalar collectives | batched collectives | collective reduction |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1 x 1` | 61.6563 s | 50.8875 s | 1.212 | 17.47% | 0 | 0 | n/a |
| 2 | `1 x 2` | 40.0698 s | 31.0439 s | 1.291 | 22.53% | 95,256 | 18,176 | 80.92% |
| 4 | `2 x 2` | 27.3014 s | 23.0325 s | 1.185 | 15.64% | 95,216 | 18,172 | 80.91% |

At two and four ranks the batched path used approximately 5.24 times fewer
MPI collectives. The number of reduced scalar values was unchanged at 103,356
and 103,316, respectively, confirming that batching changes collective
granularity rather than the mathematical projections. Median time spent in
the profiled collectives decreased by 66.63% at two ranks and 32.48% at four
ranks.

The one-rank case performs no MPI collectives. Its 17.47% improvement therefore
measures reduced local projection and Python-call overhead rather than
communication savings. All configurations used three repetitions, with wall
time spreads below 2.2%.

Exact aggregate values are available in [`comparison.csv`](comparison.csv)
and in each run's `dynamics_summary.csv`.

## Numerical validation

The scalar and batched calculations retained identical numerical work:

- 4,402--4,403 Hamiltonian calls, depending only on process grid;
- mean Krylov dimensions of 17.7625--17.775;
- maximum Krylov dimension 19;
- 80 rejected/refined substeps; and
- identical collective-value counts at equal rank count.

All nine repetition-by-repetition comparisons passed. The maximum difference
over norms, propagation diagnostics, sampled observables, and the complete
electronic and internuclear radial-density time series was exactly
`0.000e+00`. Maximum norm loss remained below `1.25e-14`.

## Decision

The batched Krylov path passes the Phase 9i.3 numerical and performance gates.
It should become the default for Cartesian-MPI SIL/SIA propagation while the
scalar path remains available as an explicit reference mode.

## Contents

- `phase9i3-krylov-scalar-production-20261106-184659/` contains the scalar
  seed and nine production-SIA records;
- `phase9i3-krylov-batched-production-20261106-190244/` contains the batched
  seed and nine production-SIA records;
- each run retains its summary CSV, jobs, Slurm accounting, output logs,
  copied workers, environment metadata, observables, radial-density archives,
  figures, state archive, and input checksums;
- `comparison.csv` contains the derived scalar/batched comparison; and
- `SHA256SUMS` authenticates every retained sanitized artifact.

The raw Saga transfer was verified against checksums generated on Saga and is
retained outside the repository. Generated Python bytecode was excluded.
Absolute Saga paths in textual artifacts use `SAGA_PROJECT` and `SAGA_USER`
placeholders, and trailing whitespace was normalized. Binary NumPy and PNG
artifacts contained neither private identifier and were retained byte-for-byte.
The published checksums were generated after sanitization.
