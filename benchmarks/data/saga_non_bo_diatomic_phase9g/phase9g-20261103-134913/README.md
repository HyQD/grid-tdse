# Saga Phase 9g velocity-gauge convergence validation

This directory archives the Saga deployment validation of Cartesian-MPI
length- and velocity-gauge real-time dynamics performed on 2026-11-03. The
calculation used H2+, CODATA 2018, a `4e11` electronic FEM-DVR grid on
`[0, 16]`, a `4e21` internuclear grid on `[0, 8]`, and a `4 x 4` process grid
for the production dynamics.

Paired length- and velocity-gauge calculations were run for
`l_max = 2, 4, 6, 8` with time steps `0.02` and `0.01`. The `l_max=8` pair was
also refined to `0.005` and `0.0025`. All 25 canonical result reports passed,
including four ground-state initialization reports and a compact
velocity-gauge MPI smoke test. Norm drift remained of order `1e-14` or less.

The fine-grid gauge differences decreased with the angular cutoff:

| `l_max` | maximum dipole-z difference | maximum energy difference |
|---:|---:|---:|
| 2 | `4.331e-05` | `2.706e-08` |
| 4 | `6.645e-06` | `1.496e-08` |
| 6 | `1.621e-06` | `6.019e-09` |
| 8 | `5.627e-07` | `3.020e-09` |

At `l_max=8` and `dt=0.0025`, the maximum length/velocity differences were
`1.487e-09` for gauge-covariant energy, `2.576e-10` for `<r>`, `5.547e-13`
for `<R>`, `4.773e-09` for `<l_R^2>`, `1.297e-11` for
`<cos^2(theta_R)>`, and `5.734e-07` for the total z dipole. These differences
are separated from the measured `dt=0.005` to `dt=0.0025` time-step errors by
factors of approximately 5, 40, 10, 661, 103, and 56, respectively.

Canonical `<L^2>` and `<l_r^2>` are intentionally excluded from the
gauge-invariant acceptance gate because canonical angular momentum changes
under a gauge transformation. The retained physical-observable discrepancies
decrease with `l_max` and are consistent with finite angular and radial
representation errors. The time-step refinements also show the expected
convergence of midpoint propagation.

Two original jobs are retained as diagnostic provenance. Job `20821882`
(`l_max=4`, length gauge, `dt=0.02`) failed during a transient NumPy import,
and job `20821894` (`l_max=8`, velocity gauge, `dt=0.02`) encountered the
corresponding transient Numba import problem and was cancelled after its MPI
ranks stalled. A serial compute-node import preflight was added to the
run-local job wrapper. Replacement jobs `20826255` and `20826256` completed
and produced the canonical passing reports.

This validation satisfies the Phase 9g deployment gate. The archive is a
correctness and convergence record, not a controlled performance benchmark.
In particular, the `l_max=8`, `dt=0.02` calculations triggered costly Krylov
refinement and should not be used to compare steady-state throughput.

Absolute Saga paths in this published archive use `SAGA_USER` and
`SAGA_PROJECT` placeholders. `SHA256SUMS` was regenerated after this
sanitization and authenticates the retained sanitized files rather than the
original cluster copies.

## Contents

- `jobs.csv`: submission manifest, including retries and time-step
  refinements;
- `provenance.txt`: source commits, Python executable, MPI runtime, and field
  package location;
- `all_job_ids.txt` and `sacct.txt`: complete Slurm job inventory and
  accounting;
- `job_mpi.sh`: final run-local Saga wrapper with the import preflight;
- `mpi_non_bo_diatomic_2d_dynamics.py`: exact dynamics example from the
  recorded grid-tdse commit;
- `output/`: retained Slurm output, including the two superseded failures;
- `results/`: 25 machine-readable reports containing the sampled time series;
- `states/`: the smoke state and reusable Davidson ground states for
  `l_max=2,4,6,8`;
- `numerical_summary.txt`: concise numerical conclusions; and
- `SHA256SUMS`: checksums for every retained sanitized artifact.
