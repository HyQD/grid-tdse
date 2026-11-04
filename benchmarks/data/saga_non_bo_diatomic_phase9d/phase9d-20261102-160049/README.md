# Saga Phase 9d Cartesian-MPI dynamics validation

This directory archives the Saga deployment validation of Cartesian-MPI
length-gauge real-time dynamics performed on 2026-11-02. Four H2+ jobs tested
SIL on `1 x 2`, `2 x 1`, and `2 x 2` process grids and SIA on a `2 x 2`
process grid. Every job completed with exit code `0:0`.

The first job computed a compact `l_max=1` Davidson ground state and saved it
to the shared archive retained under `states/`. The three dependent jobs loaded
that archive. Every calculation propagated four accepted steps with a finite
z-polarized sine-squared electric-field pulse and compared the gathered final
state and complete sampled time series with the corresponding serial
calculation.

All four calculations passed:

| integrator | process grid | ranks | norm | maximum norm drift | serial state difference | time-series maximum difference |
|:---|:---:|---:|---:|---:|---:|---:|
| SIL | `1 x 2` | 2 | `1.000000000000` | `2.220e-16` | `5.810e-16` | `1.776e-15` |
| SIL | `2 x 1` | 2 | `1.000000000000` | `2.220e-16` | `5.810e-16` | `1.998e-15` |
| SIL | `2 x 2` | 4 | `1.000000000000` | `2.220e-16` | `5.810e-16` | `1.998e-15` |
| SIA | `2 x 2` | 4 | `1.000000000000` | `2.220e-16` | `5.810e-16` | `1.998e-15` |

The acceptance thresholds were `1e-9` for norm drift and `1e-8` for both the
phase-aligned final-state difference and maximum observable-series difference.
No traceback, assertion failure, MPI abort, cancellation, timeout, or
out-of-memory condition was found in the retained output.

This compact run is a correctness and deployment gate. Its timings include
small-problem overhead and are not a performance benchmark.

Absolute Saga paths in this published archive use `SAGA_USER` and
`SAGA_PROJECT` placeholders. `SHA256SUMS` was regenerated after this
sanitization and authenticates the retained sanitized files rather than the
original cluster copies.

## Contents

- `job_phase9d.sh`: exact Saga MPI job wrapper;
- `mpi_non_bo_diatomic_2d_dynamics.py`: exact validation example executed by
  the jobs;
- `phase9d_jobs.csv`: submission manifest and dependency record;
- `provenance.txt`: source commits, Python version, and MPI runtime;
- `modules.txt` and `environment.txt`: loaded modules and installed Python
  environment;
- `sacct.txt`: Slurm accounting for all four jobs;
- `output/`: retained Slurm output;
- `results/`: machine-readable reports containing the complete sampled time
  series;
- `states/`: the shared Davidson ground-state archive; and
- `SHA256SUMS`: checksums for every retained sanitized artifact.
