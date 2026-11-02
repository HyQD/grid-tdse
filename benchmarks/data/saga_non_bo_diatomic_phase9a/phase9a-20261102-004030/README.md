# Saga Phase 9a dynamics-initialization validation

This directory archives the Saga deployment validation of the non-BO
diatomic dynamics-initialization path performed on 2026-11-02. The accepted
validation set is listed in `jobs_retry.csv` and consists of jobs
`20744797`--`20744804`.

The validation covers:

- serial Davidson generation for H2+, HD+, and D2+;
- the `require`, `compute-if-missing`, and `recompute` initialization
  policies;
- root-only Davidson setup followed by scattering to a four-rank `2 x 2`
  Cartesian layout;
- exact reconstruction of the serial dynamics state after gathering;
- homonuclear pair-exchange symmetry for H2+ and D2+ and its absence for
  HD+; and
- rejection of a velocity-gauge initial condition with nonzero `A(t0)`.

All eight accepted jobs completed with exit code `0:0`. They generated the
eight JSON files in `results/`. Every distributed state had unit norm to
floating-point precision, zero phase-aligned difference from the serial
state, and zero maximum coefficient error at printed precision. Davidson was
started exactly once by each seed, `compute-if-missing`, and `recompute` job,
and was not started by any `require` job.

The Phase 9a implementation was merged to `main` by commit `8e000da`, but the
run directory did not record the exact grid-tdse checkout SHA. That field is
therefore not reconstructed after the fact. The job output does record the
loaded modules, package versions, Python path, and MPI library.

## Initial failed attempt

The three outputs without the `_r1` suffix are retained as diagnostic
provenance. Their Davidson calculations completed successfully, but the
temporary deployment worker attempted to serialize a NumPy scalar directly
to JSON. The worker was patched to convert the norm to a Python `float` and
to pass the requested seed policy. The patched worker hash and patch date are
recorded in `provenance.txt`. This was a validation-worker failure, not a
failure in the grid-tdse initialization implementation.

## Contents

- `job_phase9a.sh`: Saga job wrapper and compute-node import preflight;
- `phase9a_worker.py`: patched validation worker;
- `jobs.csv`: original submission manifest;
- `jobs_retry.csv`: accepted retry manifest;
- `sacct_retry.txt`: Slurm accounting for the accepted jobs;
- `provenance.txt`: worker patch record and SHA-256 hash;
- `SHA256SUMS`: checksums for the reusable worker, result, and state files;
- `output/`: original and accepted Slurm output;
- `results/`: accepted machine-readable results; and
- `states/`: the small serial and MPI initialization archives used by the
  validation.

The generated `__pycache__` directory is intentionally excluded.
