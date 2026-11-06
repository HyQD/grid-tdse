# Saga Phase 9i.2 two-radial kernel validation

This directory archives the controlled Saga validation of the Phase 9i.2
two-radial Hamiltonian candidates. The runs were made on 2026-11-06 at
grid-tdse commit `bfcf1a0c228d860f9bbd38c67f31d963aaae8052` and are compared
with the pre-optimization production baseline in
[`../saga_non_bo_diatomic_phase9i/`](../saga_non_bo_diatomic_phase9i/).

All submitted jobs completed normally. No traceback, assertion failure,
import failure, MPI abort, cancellation, timeout, or out-of-memory marker was
found. The runs used Python 3.14.9, NumPy 2.5.2, SciPy 1.18.0, Numba 0.67.0,
Open MPI 5.0.10, and the `py3nj` Gaunt backend.

## Fixed-action Hamiltonian validation

The Hamiltonian workload used H2+ with `l_max=8`, the `4e11` electronic grid
on `[0, 16]`, the `4e21` internuclear grid on `[0, 8]`, a Coulomb tile size of
2048, two warm-up actions, and 50 timed actions. Each process-grid
configuration was repeated three times.

### Multipole-grouped Coulomb candidate

This series enabled the `multipole-grouped` Coulomb kernel while retaining
the reference radial kernels.

| ranks | grid | median action | spread | speedup over reference | electronic radial | internuclear radial | Coulomb | peak RSS/rank |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1 x 1` | 0.354723 s | 5.32% | 1.67 | 0.020569 s | 0.029013 s | 0.293908 s | 489.7 MiB |
| 2 | `1 x 2` | 0.187445 s | 6.48% | 1.54 | 0.012952 s | 0.015432 s | 0.154764 s | 452.8 MiB |
| 4 | `2 x 2` | 0.103862 s | 15.23% | 1.45 | 0.006867 s | 0.008432 s | 0.084541 s | 359.9 MiB |
| 16 | `4 x 4` | 0.028259 s | 15.08% | 1.38 | 0.002585 s | 0.002484 s | 0.021964 s | 352.8 MiB |

The maximum absolute difference from the serial reference was
`9.095e-13`, matching the established reference tolerance.

### Combined Coulomb, radial, and dipole candidates

This series combined `multipole-grouped` Coulomb with the opt-in `row-wise`
radial and length-gauge dipole kernels.

| ranks | grid | median action | spread | speedup over reference | electronic radial | internuclear radial | Coulomb | peak RSS/rank |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1 x 1` | 0.362013 s | 7.15% | 1.63 | 0.024349 s | 0.031060 s | 0.293858 s | 504.4 MiB |
| 2 | `1 x 2` | 0.195233 s | 1.02% | 1.48 | 0.013361 s | 0.016997 s | 0.158896 s | 462.1 MiB |
| 4 | `2 x 2` | 0.095502 s | 4.78% | 1.58 | 0.006465 s | 0.008059 s | 0.076919 s | 375.8 MiB |
| 16 | `4 x 4` | 0.027965 s | 3.90% | 1.40 | 0.002410 s | 0.002647 s | 0.021300 s | 360.9 MiB |

The maximum distributed/serial action difference was `2.291e-13`. The
row-wise radial path was slower at one and two ranks. Its small apparent gains
at four and sixteen ranks are not sufficient to select it as a default,
especially because the Coulomb-only series had spreads of approximately 15%
at those rank counts.

## Production-style SIA validation

The dynamics workload used H2+ with `l_max=2`, the end-to-end example's
default radial grids, length-gauge laser and CAP, `dt=0.25`, and a final time
of 20 a.u. Each process-grid configuration was repeated three times from the
same Davidson ground state.

| kernels | ranks | grid | median wall time | spread | speedup over reference | time/step | H calls | time/H call | mean Krylov dimension | rejected substeps |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Coulomb only | 1 | `1 x 1` | 60.07 s | 1.00% | 0.84 | 0.7509 s | 4402 | 0.01365 s | 17.77 | 80 |
| Coulomb only | 2 | `1 x 2` | 32.72 s | 2.08% | 0.97 | 0.4090 s | 4403 | 0.00743 s | 17.75 | 80 |
| Coulomb only | 4 | `2 x 2` | 25.50 s | 2.14% | 0.96 | 0.3187 s | 4402 | 0.00579 s | 17.77 | 80 |
| Combined | 1 | `1 x 1` | 43.29 s | 0.81% | 1.17 | 0.5411 s | 4402 | 0.00983 s | 17.77 | 80 |
| Combined | 2 | `1 x 2` | 33.01 s | 2.71% | 0.96 | 0.4126 s | 4403 | 0.00750 s | 17.79 | 80 |
| Combined | 4 | `2 x 2` | 26.12 s | 10.13% | 0.94 | 0.3264 s | 4403 | 0.00593 s | 17.79 | 80 |

The one-rank runs were executed on different Saga nodes from the archived
reference and are strongly node-dependent. At two and four ranks, neither
candidate configuration improved the production-style runtime. Numerical
work remained rank independent at 4402--4403 Hamiltonian calls, and the
maximum norm loss across the combined runs was `9.437e-15`.

## Decision

- Keep the reference Coulomb, radial, and dipole kernels as library defaults.
- Recommend the opt-in `multipole-grouped` Coulomb kernel for isolated
  high-angular-momentum Hamiltonian work, where it delivered a 1.38--1.67
  times speedup in this validation.
- Retain the row-wise radial and dipole kernels as experimental opt-in paths.
  Reconsider them only after a dedicated dipole benchmark or a larger
  representative dynamics workload demonstrates an end-to-end benefit.

## Contents

- `phase9i-coulomb-only-20261106-151338/` contains the 12 fixed-action
  Coulomb-only Hamiltonian records;
- `phase9i-coulomb-only-dynamics-20261106-153507/` contains the shared state,
  seed result, and nine Coulomb-only production-SIA records;
- `phase9i-hamiltonian-candidate-production-20261106-144729/` contains the 12
  combined fixed-action records, shared state, seed result, and nine combined
  production-SIA records;
- each run includes summary CSV files, jobs, Slurm accounting, output logs,
  copied workers, environment metadata, and input checksums; and
- `SHA256SUMS` authenticates every retained sanitized artifact.

The raw Saga transfer is retained outside the repository. Binary NumPy
archives were checked for the private identifiers and retained byte-for-byte.
Absolute Saga paths in textual artifacts use `SAGA_PROJECT` and `SAGA_USER`
placeholders, fixed-width trailing whitespace was normalized, and the
published checksums were generated after these transformations.
