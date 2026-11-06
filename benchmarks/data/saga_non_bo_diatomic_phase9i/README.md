# Saga Phase 9i two-radial production baseline

This directory archives the controlled Saga performance baseline collected
before Phase 9i optimization of the two-radial Hamiltonian kernels and
real-time Krylov propagation. The run was made on 2026-11-06 at grid-tdse
commit `df482e5f5ab260a1fbd513dd50ba3c13841aaab6`.

All 22 submitted jobs completed normally: 12 fixed-action Hamiltonian jobs,
one Davidson ground-state seed job, and nine production-style SIA jobs. No
traceback, assertion failure, import failure, MPI abort, cancellation,
timeout, or out-of-memory marker was found. Python 3.14.9, NumPy 2.5.2,
Numba 0.67.0, Open MPI 5.0.10, and the `py3nj` Gaunt backend were used.

## Fixed-action Hamiltonian baseline

The Hamiltonian series used H2+ with `l_max=8`, the `4e11` electronic grid on
`[0, 16]`, the `4e21` internuclear grid on `[0, 8]`, a Coulomb tile size of
2048, two warm-up actions, and 50 timed actions. Each process-grid
configuration was repeated three times.

| ranks | grid | median action | spread | speedup | efficiency | electronic radial | internuclear radial | Coulomb | peak RSS/rank |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1 x 1` | 0.591683 s | 4.12% | 1.00 | 100% | 0.022145 s | 0.031090 s | 0.526089 s | 487.2 MiB |
| 2 | `1 x 2` | 0.289019 s | 1.15% | 2.05 | 102% | 0.011377 s | 0.013938 s | 0.259718 s | 448.4 MiB |
| 4 | `2 x 2` | 0.150479 s | 1.39% | 3.93 | 98% | 0.005738 s | 0.006916 s | 0.134513 s | 359.9 MiB |
| 16 | `4 x 4` | 0.039131 s | 0.21% | 15.12 | 94% | 0.002310 s | 0.002286 s | 0.033012 s | 344.8 MiB |

The maximum absolute error against the serial reference was
`9.095e-13`. The Coulomb action accounted for approximately 89% of the
single-rank action and achieved a 15.9-fold speedup at 16 ranks.

## Production-style SIA baseline

The dynamics series used H2+ with `l_max=2`, the end-to-end example's default
radial grids, length-gauge laser and CAP, `dt=0.25`, and a final time of
20 a.u. This gives 80 requested timesteps. Wavefunction snapshots,
checkpoints, and interactive plotting were disabled. Each process-grid
configuration was repeated three times from the same Davidson ground state.

| ranks | grid | median wall time | spread | speedup | efficiency | median time/step | median H calls | time/H call | mean Krylov dimension | rejected substeps |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1 x 1` | 50.44 s | 1.40% | 1.00 | 100% | 0.6305 s | 4402 | 0.01146 s | 17.76 | 80 |
| 2 | `1 x 2` | 31.83 s | 0.49% | 1.58 | 79% | 0.3979 s | 4403 | 0.00723 s | 17.76 | 80 |
| 4 | `2 x 2` | 24.58 s | 1.21% | 2.05 | 51% | 0.3073 s | 4402 | 0.00558 s | 17.77 | 80 |

The rank-independent Krylov statistics confirm equivalent numerical work.
The maximum norm loss was `1.25e-14`. Every requested timestep triggered one
rejection/refinement, making adaptive-step overhead a candidate for later
investigation. The small `l_max=2` production problem saturates earlier than
the isolated `l_max=8` Hamiltonian action.

## Contents

- `phase9i-baseline-production-20261106-125355/hamiltonian/results/` contains
  the 12 fixed-action JSON records;
- `phase9i-baseline-production-20261106-125355/dynamics/` contains the shared
  state, seed result, and nine dynamics results;
- `hamiltonian_summary.csv` and `dynamics_summary.csv` contain the exact
  aggregate measurements;
- `jobs.csv`, `sacct.txt`, output logs, copied workers, environment metadata,
  and input checksums provide run provenance; and
- `SHA256SUMS` authenticates every retained sanitized artifact.

The raw Saga transfer was verified successfully against checksums generated
on Saga before sanitization and is retained outside the repository. Generated
Python bytecode was excluded. Absolute Saga paths use `SAGA_PROJECT` and
`SAGA_USER` placeholders, fixed-width trailing whitespace was normalized, and
the published checksums were generated after these transformations.
