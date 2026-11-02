# Saga Phase 9b dynamics-sampling validation

This directory archives the Saga deployment validation of the shared non-BO
diatomic dynamics-sampling path performed on 2026-11-02. Job `20745753` ran
the validation worker with four MPI ranks and completed with exit code `0:0`.

The gate compares every sampled observable and diagnostic produced by:

- the serial evaluator;
- one-dimensional MPI partitioning over the electronic coordinate `r`;
- one-dimensional MPI partitioning over the internuclear coordinate `R`; and
- Cartesian MPI partitioning on a `2 x 2` process grid.

The deliberately subnormalized test state has amplitude `0.37`, hence norm
`0.37^2 = 0.1369`. This checks that raw expectation numerators are retained
and that conditional, norm-divided values remain distinct. The velocity-gauge
test uses two vector-potential contributions that sum to `A = 0.2`. Samples
are taken at accepted steps 0, 1, and 3.

All comparisons passed the `5e-12` tolerance. The largest absolute
differences from the serial evaluator were:

| layout | maximum difference |
|:---|---:|
| one-dimensional `r` | `1.776e-15` |
| one-dimensional `R` | `1.776e-15` |
| Cartesian `2 x 2` | `3.553e-15` |

This is a correctness and reduction-semantics gate. The injected linear
actions and reported conditional energy are synthetic test fixtures, not a
physical molecular propagation or a performance benchmark.

## Contents

- `job_phase9b.sh`: Saga job wrapper and compute-node import preflight;
- `phase9b_gate.py`: exact validation worker executed by the job;
- `phase9b_jobs.csv`: submission manifest;
- `provenance.txt`: source commits, Python version, and MPI runtime;
- `sacct.txt`: Slurm accounting for job `20745753`;
- `output/`: retained Slurm output;
- `results/`: machine-readable gate result; and
- `SHA256SUMS`: checksums for every retained run artifact.

The generated `__pycache__` directory is intentionally excluded.
