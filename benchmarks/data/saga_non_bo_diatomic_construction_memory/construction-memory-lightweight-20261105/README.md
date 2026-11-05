# Saga lightweight two-radial construction-memory validation

This directory archives the controlled Saga validation of the lightweight
element-local two-radial Hamiltonian construction path. The runs were made on
2026-11-05 at grid-tdse commit
`aeb33abcdac842ff55f8ca219e52e52a8ede22c3` and are directly comparable with
the pre-optimization audit in `../construction-memory-20261105/`.

Each job used one physical MPI process while emulating rank zero of the stated
logical process grid. It constructed the rank-local Hamiltonian but performed
no Hamiltonian action, eigensolve, or propagation. The logical complex state
tile was sized but not allocated.

All six jobs completed normally. No traceback, memory error, out-of-memory
marker, cancellation, timeout, or MPI abort was found. `py3nj` 0.2.1 was
available and selected on every compute node, and every run reported
`element-local` radial operator storage.

## Lightweight results

The production series used a `4e11` electronic grid, a `4e21` internuclear
grid, and a logical `2 x 2` process grid:

| `l_max` | radial points | final RSS/rank | process peak/rank | Slurm `MaxRSS`/rank | inventoried arrays/rank | unused dense/pair arrays | logical state/rank |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 8 | `39 x 79` | `359.0 MiB` | `368.2 MiB` | `412.2 MiB` | `4.5 MiB` | `0.0 MiB` | `3.2 MiB` |
| 12 | `39 x 79` | `568.9 MiB` | `674.0 MiB` | `718.8 MiB` | `44.9 MiB` | `0.0 MiB` | `9.5 MiB` |
| 16 | `39 x 79` | `1064.5 MiB` | `2167.9 MiB` | `2214.6 MiB` | `269.7 MiB` | `0.0 MiB` | `20.9 MiB` |

The assessment series increased both radial grids with angular cutoff:

| `l_max` | logical grid | radial points | final RSS/rank | process peak/rank | Slurm `MaxRSS`/rank | inventoried arrays/rank | unused dense/pair arrays | logical state/rank |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 8 | `16 x 8` | `319 x 159` | `374.6 MiB` | `380.9 MiB` | `424.7 MiB` | `9.4 MiB` | `0.0 MiB` | `1.6 MiB` |
| 12 | `24 x 12` | `479 x 239` | `595.3 MiB` | `707.9 MiB` | `752.2 MiB` | `63.8 MiB` | `0.0 MiB` | `4.7 MiB` |
| 16 | `32 x 16` | `639 x 319` | `1134.9 MiB` | `2242.7 MiB` | `2290.1 MiB` | `319.4 MiB` | `0.0 MiB` | `10.5 MiB` |

## Comparison with the pre-optimization assessment

| `l_max` | old process peak | new process peak | reduction | old Slurm `MaxRSS` | new Slurm `MaxRSS` | reduction | old arrays | new arrays | reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `761.8 MiB` | `380.9 MiB` | `50.0%` | `806.3 MiB` | `424.7 MiB` | `47.3%` | `346.5 MiB` | `9.4 MiB` | `97.3%` |
| 12 | `3001.1 MiB` | `707.9 MiB` | `76.4%` | `3050.5 MiB` | `752.2 MiB` | `75.3%` | `2137.7 MiB` | `63.8 MiB` | `97.0%` |
| 16 | `10292.8 MiB` | `2242.7 MiB` | `78.2%` | `10356.6 MiB` | `2290.1 MiB` | `77.9%` | `8123.6 MiB` | `319.4 MiB` | `96.1%` |

The unused dense one-body and expanded angular-pair categories are absent in
all lightweight inventories. At fixed `l_max`, enlarging the radial grids now
adds only 12.7, 33.9, and 74.8 MiB to the measured process peak for
`l_max=8`, `12`, and `16`, respectively. The remaining peak is therefore
primarily angular-cutoff dependent rather than caused by replicated large
radial arrays.

The process peak and Slurm `MaxRSS` differ by a nearly constant 44--47 MiB,
consistent with runtime and launcher overhead outside the benchmark's process
measurement. A 4-GiB request is sufficient for all six one-process diagnostic
configurations with a conservative margin.

## Contents

- `construction-memory-production-lightweight-20261105-163557/`: production
  manifests, copied workers, output, results, and accounting for
  `l_max=8,12,16`;
- `construction-memory-assessment-lightweight-l8-20261105/`: assessment
  `l_max=8` artifacts;
- `construction-memory-assessment-lightweight-l12-l16-20261105/`: assessment
  `l_max=12,16` artifacts;
- `summary.csv`: rounded combined measurements; and
- `SHA256SUMS`: checksums for every retained sanitized artifact.

Absolute Saga paths use `SAGA_USER` and `SAGA_PROJECT` placeholders. Trailing
whitespace from fixed-width Slurm tables was normalized. Checksums were
generated after these transformations and authenticate the published files.
