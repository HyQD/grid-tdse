# Saga two-radial construction-memory audit

This directory archives the Saga construction-memory audit performed on
2026-11-05 at grid-tdse commit
`6919c52ee298f66b2d15608029c71a7eb391cf6a`. The audit measured the memory
retained while constructing the global two-radial representation and the
element-local Cartesian-MPI Hamiltonian. It performed no Hamiltonian action,
ground-state solve, or real-time propagation.

Each job used one real MPI process while emulating rank zero of the stated
logical process grid. This measures the per-rank cost of arrays that are
constructed independently on every rank without replicating a potentially
multi-gigabyte allocation across the full logical layout. The complex state
tile size was calculated but the state itself was not allocated.

All six jobs completed normally. `py3nj` 0.2.1 was available on the compute
node, and grid-lib selected the `py3nj` Gaunt backend for every case. No
traceback, memory error, out-of-memory marker, cancellation, timeout, or MPI
abort was present in the retained output.

## Results

The production series used a `4e11` electronic grid, a `4e21` internuclear
grid, and a logical `2 x 2` process grid:

| `l_max` | radial points | final RSS/rank | process peak/rank | inventoried arrays/rank | dense/pair arrays unused by the local action | logical state/rank | unused/state |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 8 | `39 x 79` | `381.9 MiB` | `394.5 MiB` | `25.4 MiB` | `20.9 MiB` | `3.2 MiB` | `6.5` |
| 12 | `39 x 79` | `614.5 MiB` | `673.0 MiB` | `101.5 MiB` | `56.6 MiB` | `9.5 MiB` | `6.0` |
| 16 | `39 x 79` | `1139.4 MiB` | `2166.6 MiB` | `389.1 MiB` | `119.4 MiB` | `20.9 MiB` | `5.7` |

The assessment series increased both radial grids with angular cutoff:

| `l_max` | logical grid | radial points | final RSS/rank | process peak/rank | Slurm `MaxRSS`/rank | inventoried arrays/rank | dense/pair arrays unused by the local action | logical state/rank | unused/state |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `16 x 8` | `319 x 159` | `711.4 MiB` | `761.8 MiB` | `806.3 MiB` | `346.5 MiB` | `337.1 MiB` | `1.6 MiB` | `208.4` |
| 12 | `24 x 12` | `479 x 239` | `2663.0 MiB` | `3001.1 MiB` | `3050.5 MiB` | `2137.7 MiB` | `2073.9 MiB` | `4.7 MiB` | `437.3` |
| 16 | `32 x 16` | `639 x 319` | `8932.2 MiB` | `10292.8 MiB` | `10356.6 MiB` | `8123.6 MiB` | `7804.2 MiB` | `10.5 MiB` | `746.4` |

`summary.csv` retains these rounded comparison values in machine-readable
form. The result JSON files contain the exact measurements, construction-stage
timings, complete deduplicated NumPy allocation inventory, runtime metadata,
and problem definitions.

The process peak-memory measurements and Slurm `MaxRSS` agree within about 2%
for the assessment `l_max=12` and `16` cases. At assessment `l_max=16`,
approximately 96% of the inventoried array memory is held by dense one-body
and expanded angular-pair operators that the element-local action does not
use. They occupy about 746 times the logical state tile per rank. The audit
therefore reproduces the external performance assessment on Saga and
establishes that the retention is an implementation property rather than a
workstation-specific effect.

Slurm accounting was collected for all assessment jobs. A production-series
`sacct.txt` was not generated before synchronization, so no production
`MaxRSS` values are reported or reconstructed.

## Contents

- `construction-memory-production-20261105-151709/`: production-grid
  manifests, copied scripts, backend record, Slurm output, and three result
  JSON files;
- `construction-memory-assessment-l8/`: assessment `l_max=8` manifest, copied
  scripts, backend record, output, accounting, and result JSON;
- `construction-memory-assessment-20261105-153240/`: assessment `l_max=12,16`
  manifest, copied scripts, backend record, output, accounting, and result
  JSON files;
- `summary.csv`: combined compact result table; and
- `SHA256SUMS`: checksums for every retained sanitized artifact.

Absolute Saga paths use `SAGA_USER` and `SAGA_PROJECT` placeholders, and
trailing whitespace from fixed-width Slurm tables was normalized. The
checksums were generated after these transformations and authenticate the
published files rather than the original cluster copies.
