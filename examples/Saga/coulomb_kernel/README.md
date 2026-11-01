# Coulomb-kernel characterization on Saga

This workflow prepares the first three steps of the non-BO Coulomb-kernel
optimization study without changing the production kernel:

1. isolate the real and complex Coulomb action at `l_max = 4, 6, 8`;
2. measure tile-size sensitivity and, optionally, hardware counters;
3. compare pure MPI with Numba threading at a fixed allocation of 16 cores.

Compilation, Hamiltonian construction, state distribution, correctness
checks, and JSON writing are outside the benchmark's timed interval. The
timed action is the same compact Coulomb kernel used by the production
Hamiltonian. Every submitted configuration is repeated three times by
default and retains an untimed comparison with the public distributed and
serial Coulomb actions. Jobs are dependency-serialized by default so only one
configuration performs its Python imports and benchmark work at a time. Each
job also performs a single-process compute-node import preflight and builds
its job-local Matplotlib font cache before launching MPI ranks. These measures
avoid shared-filesystem import storms on Saga.

## 1. Update and validate the Saga checkout

Start on a Saga login node and activate the normal project environment:

```bash
module --quiet reset
module load StdEnv
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
project_root="/cluster/projects/<project-id>/<username>"
source "${project_root}/grid_tdse_paths.sh"
module load "${GRID_TDSE_MPI_MODULE}"

conda activate base
conda activate "${ENV_PREFIX}"
hash -r

cd "${SRC_ROOT}/grid-tdse"
git switch main
git pull --ff-only origin main
python -m pip install --no-deps --no-build-isolation --editable ".[dev]"
python -m pip check
```

Confirm the active Python, MPI library, and scripts:

```bash
command -v python
python --version
python -c 'from mpi4py import MPI; print(MPI.Get_library_version().rstrip("\x00"))'
python benchmarks/non_bo_diatomic_coulomb_execution.py --help
bash -n examples/Saga/coulomb_kernel/job_coulomb_kernel.sh
bash -n examples/Saga/coulomb_kernel/submit_coulomb_kernel_benchmarks.sh
```

The submission helper creates a unique directory below
`${PROJECT_ROOT}/jobs/grid-tdse/coulomb-kernel`, copies the exact job,
benchmark, and summary scripts used there, and writes provenance and a CSV
job manifest. The manifest records the Slurm dependency for every job.

## 2. Select the kernel tile size

Submit the complex `l_max = 8` tile sweep. It uses 16 one-thread MPI ranks on
one node and tests tile sizes from 32 through 2048:

```bash
cd "${SRC_ROOT}/grid-tdse"

examples/Saga/coulomb_kernel/submit_coulomb_kernel_benchmarks.sh \
    --stage tile \
    --system H2+ \
    --run-name h2p-coulomb-tile-l8-sequential

run_root=$(cat \
    "${PROJECT_ROOT}/jobs/grid-tdse/coulomb-kernel/latest_tile_run.txt")
echo "${run_root}"
column -s, -t "${run_root}/coulomb_jobs.csv"
```

Monitor only this run:

```bash
job_ids=$(tail -n +2 "${run_root}/coulomb_jobs.csv" | cut -d, -f1 | paste -sd, -)
squeue -j "${job_ids}" -o "%.18i %.42j %.2t %.10M %.30R"
```

Only one job should be running. Later jobs should be pending with a dependency
reason. The chain uses `afterany`, so one failed configuration does not prevent
the remaining independent measurements from running. Every output file must
contain `Preflight imports complete` before its MPI benchmark output.

When all jobs have left the queue, check their states and scan for failures:

```bash
sacct -j "${job_ids}" \
    --format=JobID,JobName%38,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS,NodeList \
    | tee "${run_root}/sacct.txt"

grep --text -H -E \
    'Traceback|FileNotFoundError|ImportError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY' \
    "${run_root}/output/"*.out || true

grep --text -l 'Result JSON:' "${run_root}/output/"*.out | wc -l
```

The tile sweep must produce 21 result files: seven tile sizes times three
repetitions. Do not select a tile from an incomplete sweep.

Create the summary:

```bash
python "${run_root}/summarize_non_bo_diatomic_coulomb_execution.py" \
    "${run_root}/results" \
    --csv "${run_root}/summary.csv"
```

Choose the tile with the lowest stable median, not merely the fastest single
run. If adjacent tile sizes differ by only a few percent, prefer the one with
the smaller repetition spread. Use that value as `<best-tile>` below.

### Optional hardware counters

`perf` access depends on the compute-node security configuration. Test it in
an allocation first. A permission error is a site restriction, not a
`grid-tdse` failure.

```bash
srun --account="${PROJECT_ID}" --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --time=00:05:00 --pty bash
perf stat true
exit
```

If available, run one repetition of the same sweep with counters:

```bash
examples/Saga/coulomb_kernel/submit_coulomb_kernel_benchmarks.sh \
    --stage tile \
    --system H2+ \
    --repetitions 1 \
    --perf \
    --run-name h2p-coulomb-tile-l8-perf
```

The workflow writes one comma-separated `perf stat` file per MPI rank under
the run's `perf/` directory. The default events are `task-clock`, `cycles`,
`instructions`, `cache-references`, and `cache-misses`. These counters cover
the complete benchmark process; the large action count makes the repeated
Coulomb actions dominant, but wall-time comparisons must still use the JSON
timings, which isolate the kernel exactly.

## 3. Measure angular growth and fixed-core threading

First benchmark both `float64` ground-state-like input and `complex128`
dynamics-like input at `l_max = 4, 6, 8`:

```bash
best_tile=<best-tile>

examples/Saga/coulomb_kernel/submit_coulomb_kernel_benchmarks.sh \
    --stage angular \
    --system H2+ \
    --tile-size "${best_tile}" \
    --run-name "h2p-coulomb-angular-z${best_tile}"

angular_root=$(cat \
    "${PROJECT_ROOT}/jobs/grid-tdse/coulomb-kernel/latest_angular_run.txt")
```

Then compare 16 MPI ranks x 1 Numba thread, 8 x 2, and 4 x 4. All cases use
exactly 16 physical cores on one node:

```bash
examples/Saga/coulomb_kernel/submit_coulomb_kernel_benchmarks.sh \
    --stage threads \
    --system H2+ \
    --tile-size "${best_tile}" \
    --run-name "h2p-coulomb-threads-z${best_tile}"

threads_root=$(cat \
    "${PROJECT_ROOT}/jobs/grid-tdse/coulomb-kernel/latest_threads_run.txt")
```

For each run, repeat the state/error scan, `sacct`, and summary commands from
step 2. For example:

```bash
for current_root in "${angular_root}" "${threads_root}"; do
    job_ids=$(tail -n +2 "${current_root}/coulomb_jobs.csv" \
        | cut -d, -f1 | paste -sd, -)
    sacct -j "${job_ids}" \
        --format=JobID,JobName%38,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS,NodeList \
        > "${current_root}/sacct.txt"
    grep --text -H -E \
        'Traceback|FileNotFoundError|ImportError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY' \
        "${current_root}/output/"*.out || true
    python "${current_root}/summarize_non_bo_diatomic_coulomb_execution.py" \
        "${current_root}/results" \
        --csv "${current_root}/summary.csv"
done
```

The thread study deliberately fixes total cores. Interpret
`speedup_from_fixed_core_pure_mpi` relative to the 16-rank x 1-thread case.
A hybrid configuration is useful only if its repeated median improves while
correctness and CPU utilization remain sound. Do not infer a benefit from
requesting extra CPUs per rank unless `NUMBA_NUM_THREADS` is increased too;
this workflow sets both consistently and keeps BLAS at one thread.

### Concurrent submission override

The `--concurrent` option disables dependency serialization. Use it only after
the same environment and benchmark have completed reliably on the target
cluster and only when the additional simultaneous imports and filesystem load
are intentional. The recommended Saga workflow omits this option.

## Retrieve completed records

Run this command on the local workstation, replacing `<run-name>` with the
actual directory name:

```bash
mkdir -p "$HOME/Documents/HyQD/saga-results/grid-tdse/coulomb-kernel"
rsync -av \
    "${USER}@saga.sigma2.no:/cluster/projects/${PROJECT_ID}/${USER}/jobs/grid-tdse/coulomb-kernel/<run-name>/" \
    "$HOME/Documents/HyQD/saga-results/grid-tdse/coulomb-kernel/<run-name>/"
```

Keep the raw JSON, manifest, provenance, Slurm accounting, summaries, output,
and any `perf` files together. Archive compact summary/provenance data in the
repository only after reviewing the complete runs.
