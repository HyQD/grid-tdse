# Olivia coupled Step 16 qualification

This directory records the sanitized aggregate results of the first complete
Olivia qualification of the coupled two-radial implementation. It deliberately
contains no username, project number, or private cluster path.

- Date: 2026-11-12
- Deployment: `20261112-104035-cpu-ompi508-coupled-step16`
- Qualification-support commit: `04c0b35`
- MPI: Open MPI 5.0.8 using Olivia `cm`/`ofi` with the `cxi` provider
- Promoted to `current`: `2026-11-12T15:23:35+01:00`

## Gates

| Gate | Slurm job | Evidence | Result |
| --- | ---: | --- | --- |
| full test suite | `2848231` | 807 tests passed; two MPI ranks launched | PASS |
| serial/MPI equivalence | `2848232` | state `1.294e-18`; time series `3.331e-16` | PASS |
| single-node stress | `2848233` | 1, 2, 4, 8, and 16 ranks; best speedup `4.805` | PASS |
| Step 16 preparation | `2849982` | coupled and general reference calculations | PASS |
| combined two-node run | `2850231` | placement, transport, coupled, and general checks | PASS |

Every successful job completed with Slurm state `COMPLETED` and exit code
`0:0`.

## Coupled calculation

| Ranks | Process grid | Nodes | Wall time (s) | State difference | Restart difference | Time-series difference | Maximum component error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1-by-1 | 1 | 0.292 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `4.441e-16` |
| 4 | 2-by-2 | 1 | 0.349 | `2.797e-17` | `5.722e-17` | `1.776e-15` | `4.965e-16` |
| 16 | 4-by-4 | 2 | 0.311021 | `6.206e-17` | `6.206e-17` | `1.776e-15` | `4.965e-16` |

The 16-rank CAP loss was `9.413e-04`, equal to the reference at the reported
precision. This small calculation is a correctness and transport test, not a
production scaling benchmark.

## General Cartesian-MPI calculation

| Ranks | Process grid | Nodes | Wall time (s) | Speedup | State difference | Time-series difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1-by-1 | 1 | 60.091487 | 1.000 | `4.250e-15` | `8.882e-15` |
| 2 | 1-by-2 | 2 | 31.744866 | 1.893 | `3.672e-15` | `1.110e-14` |
| 16 | 4-by-4 | 2 | 11.490447 | 5.230 | `5.227e-15` | `7.550e-15` |

The combined two-node job consumed 32.6 billed CPU-hours because Olivia
accounts for whole allocated nodes.

## Raw evidence

Raw Slurm output, placement records, result JSON files, and the deployment
manifest are retained in the private `grid-tdse-data` workspace. The archived
copy contains 43 files, and all files passed verification against its recorded
SHA-256 checksums. Before copying any raw artifact into this public repository,
remove usernames, project identifiers, absolute cluster paths, and other
site-private values, then record checksums of the sanitized files.
