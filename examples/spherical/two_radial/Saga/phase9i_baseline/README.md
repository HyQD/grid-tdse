# Phase 9i production baseline on Saga

This workflow freezes the performance baseline required before changing the
two-radial Hamiltonian kernels or real-time Krylov implementation. It runs two
complementary workloads:

- a deterministic H2+ `l_max=8` field-free Hamiltonian benchmark on the
  `4e11` electronic and `4e21` internuclear grids, with separate radial and
  Coulomb timings;
- the production-style H2+ `l_max=2` SIA example with its default grids,
  laser, CAP, observables, and `dt=0.25`, stopped after 20 a.u.

The default matrix uses three repetitions, Hamiltonian rank counts
`1 2 4 16`, and dynamics rank counts `1 2 4`. Jobs are serialized by Slurm
dependencies to avoid interference between benchmark cases. Pass
`--concurrent` only for workflow testing, not for the archived baseline.
The `--coulomb-kernel` option selects either the immutable `reference` path
or the Phase 9i.2 `multipole-grouped` candidate. It defaults to `reference`.
The independent `--radial-kernel` and `--dipole-kernel` selectors choose the
compiled row-wise element-local radial action and compact length-gauge dipole
action, respectively. Both also default to `reference`; velocity-gauge dipole
actions retain their reference contraction because the measured row-wise
candidate was slower.

## 1. Update and verify the Saga checkout

After the benchmark branch has been merged, source the existing site
configuration and update the checkout:

```bash
source /cluster/projects/<project-id>/<username>/grid_tdse_paths.sh

cd "${SRC_ROOT}/grid-tdse"
git switch main
git pull --ff-only origin main

python -m pip install \
    --no-deps \
    --no-build-isolation \
    --editable .

python -m pip check
python - <<'PY'
import importlib.util
import grid_tdse
import integrator_lib
from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend
from mpi4py import MPI

specification = importlib.util.find_spec("py3nj")
print("grid-tdse:", grid_tdse.__file__)
print("integrator-lib:", integrator_lib.__file__)
print("py3nj available:", specification is not None)
print("py3nj path:", None if specification is None else specification.origin)
print("Gaunt backend:", get_gaunt_backend(resolve=True))
print(MPI.Get_library_version().rstrip("\x00"))
PY
```

Confirm the workflow scripts before submission:

```bash
bash -n \
    examples/spherical/two_radial/Saga/phase9i_baseline/job_phase9i_baseline.sh

bash -n \
    examples/spherical/two_radial/Saga/phase9i_baseline/submit_phase9i_baseline.sh

examples/spherical/two_radial/Saga/phase9i_baseline/submit_phase9i_baseline.sh \
    --help
```

## 2. Optional reduced smoke submission

This four-job-plus-seed matrix validates the workflow without creating the
performance record:

```bash
examples/spherical/two_radial/Saga/phase9i_baseline/submit_phase9i_baseline.sh \
    --stage all \
    --repetitions 1 \
    --hamiltonian-ranks 1 4 \
    --dynamics-ranks 1 4 \
    --actions 5 \
    --warmup-actions 1 \
    --dynamics-final-time 2.0 \
    --run-name phase9i-baseline-smoke
```

## 3. Submit the controlled baseline

Use the defaults for the archive-quality run:

```bash
examples/spherical/two_radial/Saga/phase9i_baseline/submit_phase9i_baseline.sh \
    --stage all
```

No QoS is selected by default. Site settings can be supplied without editing
the templates, for example:

```bash
export GRID_TDSE_PHASE9I_WALLTIME=00:30:00
export GRID_TDSE_PHASE9I_MEM_PER_CPU=4G

# Set only when required by the current Saga allocation policy:
# export GRID_TDSE_PHASE9I_QOS=<qos>
# export GRID_TDSE_PHASE9I_PARTITION=<partition>

examples/spherical/two_radial/Saga/phase9i_baseline/submit_phase9i_baseline.sh \
    --stage all
```

The submission creates a self-contained run directory below
`${PROJECT_ROOT}/jobs/grid-tdse/phase9i-baseline`, copies every worker used by
the jobs, records the Git commit and input checksums, and writes
`latest_run.txt`.

## 4. Monitor the jobs

```bash
run_root=$(cat \
    "${PROJECT_ROOT}/jobs/grid-tdse/phase9i-baseline/latest_run.txt")

job_ids=$(
    awk -F, 'NR > 1 && $1 ~ /^[0-9]+$/ {print $1}' \
        "${run_root}/jobs.csv" |
    paste -sd, -
)

printf 'Run root: %s\n' "${run_root}"
printf 'Job IDs:  %s\n' "${job_ids}"

squeue \
    -j "${job_ids}" \
    --format='%.18i %.34j %.10T %.10M %.10l %.24R'
```

Follow a particular job by its numeric ID:

```bash
job_id=<job-id>
output_file=$(
    find "${run_root}/output" \
        -maxdepth 1 \
        -type f \
        -name "*-${job_id}.out" \
        -print \
        -quit
)

tail -F "${output_file}"
```

## 5. Inspect and summarize completed jobs

```bash
grep --text -H -E \
    'Traceback|AssertionError|FileNotFoundError|ImportError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY' \
    "${run_root}/output/"*.out || true

find "${run_root}/hamiltonian/results" \
    -maxdepth 1 -type f -name '*.json' | wc -l

find "${run_root}/dynamics/runs" \
    -mindepth 2 -maxdepth 2 -type f -name summary.json | wc -l

python \
    "${run_root}/summarize_non_bo_diatomic_phase9i_baseline.py" \
    "${run_root}"
```

The full default run should contain 12 Hamiltonian JSON records and nine
dynamics summaries, in addition to the ground-state seed result. The
summarizer writes `hamiltonian_summary.csv` and `dynamics_summary.csv` in the
run root.

Save the Slurm accounting record:

```bash
sacct -X \
    -j "${job_ids}" \
    --format=JobID,JobName%34,State,ExitCode,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS,NodeList \
    > "${run_root}/sacct.txt"

cat "${run_root}/sacct.txt"
```

Do not begin Phase 9i kernel changes until this run has been checked for
failures, summarized, copied locally, and archived with its checksums.

## 6. Compare the Phase 9i.2 Hamiltonian candidate

After the candidate branch has passed local parity and MPI tests, run the
same controlled matrix with the candidate selected explicitly:

```bash
examples/spherical/two_radial/Saga/phase9i_baseline/submit_phase9i_baseline.sh \
    --stage all \
    --coulomb-kernel multipole-grouped \
    --radial-kernel row-wise \
    --dipole-kernel row-wise \
    --run-name phase9i-hamiltonian-candidate-production
```

For a reduced workflow check before the production matrix:

```bash
examples/spherical/two_radial/Saga/phase9i_baseline/submit_phase9i_baseline.sh \
    --stage all \
    --coulomb-kernel multipole-grouped \
    --radial-kernel row-wise \
    --dipole-kernel row-wise \
    --repetitions 1 \
    --hamiltonian-ranks 1 4 \
    --dynamics-ranks 1 4 \
    --actions 5 \
    --warmup-actions 1 \
    --dynamics-final-time 2.0 \
    --run-name phase9i-hamiltonian-candidate-smoke
```

Keep candidate and reference records in separate run directories. The JSON,
CSV, terminal summary, and job manifest all record the selected kernels so
that results cannot be combined accidentally.
