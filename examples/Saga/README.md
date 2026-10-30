# Saga job-script examples

These scripts provide starting points for serial and MPI `grid-tdse` jobs on
Saga. They use the project-local Conda layout documented in
[`docs/saga_setup.md`](../../docs/saga_setup.md).

Before submitting them:

1. Replace `<project-id>` and `<username>` in each job script.
2. Find an available cluster Open MPI module with `module spider OpenMPI`.
3. Add its exact name to `grid_tdse_paths.sh`, for example:

   ```bash
   export GRID_TDSE_MPI_MODULE="OpenMPI/<exact-version>"
   ```

4. Build `mpi4py` with that same module as described below.

The MPI module used to build `mpi4py` and the module loaded by every MPI job
must be identical.

## Install MPI support in the Saga environment

Load the cluster software and activate the existing project environment:

```bash
module --quiet reset
module load StdEnv
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
source /cluster/projects/<project-id>/<username>/grid_tdse_paths.sh
module load "${GRID_TDSE_MPI_MODULE}"

conda activate base
conda activate "${ENV_PREFIX}"
hash -r
```

Verify that Python comes from the project environment and `mpicc` comes from
the selected Saga module:

```bash
command -v python
command -v mpicc
module list
```

Do not install Conda Open MPI in the Saga environment. Build `mpi4py` against
the cluster module instead:

```bash
python -m pip uninstall -y mpi4py
MPICC="$(command -v mpicc)" \
python -m pip install --no-cache-dir --no-binary=mpi4py "mpi4py>=4,<5"
```

Confirm the linked MPI library:

```bash
python -c 'from mpi4py import MPI; print(MPI.Get_library_version().rstrip("\x00"))'
python -m pip check
```

Use `srun`, not `mpiexec` or `mpirun`, to launch MPI applications on Saga.
Sigma2 recommends `srun --mpi=pmix` for Open MPI jobs.

## Update the HyQD libraries

Update `integrator-lib` before `grid-tdse`:

```bash
cd "${SRC_ROOT}/integrator-lib"
git switch main
git pull --ff-only origin main

cd "${SRC_ROOT}/grid-tdse"
git switch main
git pull --ff-only origin main
```

Refresh editable metadata without replacing the MPI installation:

```bash
python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/integrator-lib"
python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/grid-tdse[dev]"
```

Then repeat the MPI library and dependency checks above.

## Copy the templates to the jobs directory

Copy the scripts before inserting project-specific values so the Git checkout
remains clean:

```bash
mkdir -p "${PROJECT_ROOT}/jobs/grid-tdse"
cp "${SRC_ROOT}/grid-tdse/examples/Saga/job_mpi.sh" \
   "${PROJECT_ROOT}/jobs/grid-tdse/"
cp "${SRC_ROOT}/grid-tdse/examples/Saga/job_serial.sh" \
   "${PROJECT_ROOT}/jobs/grid-tdse/"
cp "${SRC_ROOT}/grid-tdse/examples/Saga/submit_mpi_ground_state_rank_series.sh" \
   "${PROJECT_ROOT}/jobs/grid-tdse/"
cp "${SRC_ROOT}/grid-tdse/examples/Saga/submit_mpi_angular_scaling.sh" \
   "${PROJECT_ROOT}/jobs/grid-tdse/"
cp "${SRC_ROOT}/grid-tdse/examples/Saga/job_hamiltonian_baseline.sh" \
   "${PROJECT_ROOT}/jobs/grid-tdse/"
cp "${SRC_ROOT}/grid-tdse/examples/Saga/submit_hamiltonian_execution_baseline.sh" \
   "${PROJECT_ROOT}/jobs/grid-tdse/"

cd "${PROJECT_ROOT}/jobs/grid-tdse"
chmod +x job_mpi.sh job_serial.sh \
    submit_mpi_ground_state_rank_series.sh \
    submit_mpi_angular_scaling.sh \
    job_hamiltonian_baseline.sh \
    submit_hamiltonian_execution_baseline.sh
```

Replace `<project-id>` and `<username>` in the copied job scripts. Jobs
already submitted to Slurm retain the script content present at submission
time, so resubmit a job after copying an updated template.

When `PROJECT_ID` is exported by `grid_tdse_paths.sh`, the MPI template can be
configured from the jobs directory with:

```bash
sed -i \
    -e "s|<project-id>|${PROJECT_ID}|g" \
    -e "s|<username>|${USER}|g" \
    job_mpi.sh

grep -n -E 'account=|project_root=' job_mpi.sh
bash -n job_mpi.sh
```

Apply these substitutions only to the copied job script, not to the template
inside the Git checkout.

Configure the phase-6 template in the same way:

```bash
sed -i \
    -e "s|<project-id>|${PROJECT_ID}|g" \
    -e "s|<username>|${USER}|g" \
    job_hamiltonian_baseline.sh

grep -n -E 'account=|project_root=' job_hamiltonian_baseline.sh
bash -n job_hamiltonian_baseline.sh
bash -n submit_hamiltonian_execution_baseline.sh
```

## Submit a non-MPI job

The serial prototype accepts a Python script followed by its arguments:

```bash
sbatch job_serial.sh examples/non_bo_diatomic_ground_state.py
```

It requests one task and one CPU and launches the program with `srun` on a
compute node.

## Submit an MPI correctness run

Submit H2+ with one, two, and four ranks while retaining the default
serial-reference comparison:

```bash
for ranks in 1 2 4; do
    sbatch \
        --job-name="correct_h2p_${ranks}r" \
        --ntasks="${ranks}" \
        job_mpi.sh \
        examples/mpi_non_bo_diatomic_ground_state.py \
        --system H2+
done
```

HD+ provides a complementary sanity check of the nonidentical-particle path:

```bash
for ranks in 1 4; do
    sbatch \
        --job-name="correct_hdp_${ranks}r" \
        --ntasks="${ranks}" \
        job_mpi.sh \
        examples/mpi_non_bo_diatomic_ground_state.py \
        --system HD+
done
```

The default calculation also runs the matching serial reference on rank zero.
Its output reports the distributed solve time separately, so gathering and the
serial reference are excluded from that measurement. Confirm convergence,
unit norm, matching energies and iteration counts, and agreement with the
phase-aligned serial state before running timing-only jobs.

To compare the two one-dimensional layouts on the same unequal `r` and `R`
grids, repeat a validated rank count with each partition choice:

```bash
for coordinate in r R; do
    sbatch \
        --job-name="correct_h2p_${coordinate}_2r" \
        --ntasks=2 \
        job_mpi.sh \
        examples/mpi_non_bo_diatomic_ground_state.py \
        --system H2+ \
        --partition-coordinate "${coordinate}"
done
```

The default is `R`; specifying it explicitly here makes the comparison record
self-describing.

## Submit the initial rank series

Submit separate one-node jobs with 1, 2, and 4 ranks:

```bash
bash submit_mpi_ground_state_rank_series.sh H2+
```

These timing jobs pass `--no-serial-comparison`, because repeating the serial
reference while the remaining MPI ranks wait would waste the allocation and
distort the total job time. Run the correctness jobs with the comparison
enabled before interpreting the timing series.

Every rank is assigned one CPU core, and NumPy, BLAS, OpenMP, and Numba thread
counts are fixed to one to avoid nested-thread oversubscription. All jobs are
restricted to one node so this first series measures only the current
shared-node MPI decomposition. Each rank receives its own job-local Numba
cache directory to avoid concurrent writes during JIT compilation.

Do not increase `--cpus-per-task` for these MPI calculations: the current
distributed kernels use one serial CPU core per rank, so extra CPUs would be
reserved but unused. The distinction between ranks, CPUs per task, `R`-element
ownership, and multi-node jobs is explained in
[`docs/saga_setup.md`](../../docs/saga_setup.md#choose-mpi-ranks-cpus-and-nodes).

After completion, inspect resource usage with the job IDs printed by `sbatch`:

```bash
sacct -j <job-id> \
    --format=JobID,JobName,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS,NodeList
```

Compare the `Distributed solve time` lines from the four output files. Report
speedup relative to the one-rank time as `T1 / Tp` and parallel efficiency as
`T1 / (p * Tp)`. Treat this as an initial single-node performance indication,
not the full strong- and weak-scaling benchmark planned for a later step.

## Scale the angular cutoff

The MPI example accepts `--l-max`. For this controlled study it derives the
remaining angular cutoffs as:

```text
m_max = l_max
lambda_max = 2*l_max
mu_max = lambda_max
```

The radial grids, symmetry sector, initial state, timestep, and convergence
tolerances remain fixed. Before timing a new cutoff, run a correctness case
with the serial comparison enabled, for example:

```bash
sbatch \
    --job-name=correct_h2p_l4_1r \
    --ntasks=1 \
    job_mpi.sh \
    examples/mpi_non_bo_diatomic_ground_state.py \
    --system H2+ \
    --l-max 4
```

The angular-scaling helper submits 1-, 2-, and 4-rank timing jobs for each
requested cutoff. With no cutoff arguments, it starts conservatively with
`l_max=2` and `4`:

```bash
bash submit_mpi_angular_scaling.sh H2+
```

After inspecting their runtime and memory use, submit `l_max=6` and `8`:

```bash
bash submit_mpi_angular_scaling.sh H2+ 6 8
```

If the smaller runs indicate that a larger cutoff can exceed the template's
30-minute `devel` limit, remove the `devel` quality-of-service directive from
the copied `job_mpi.sh` and request an appropriate wall time before submitting
the larger series.

Extract the size and timing diagnostics with:

```bash
grep --text -H -E \
    'MPI ranks|l_max:|Angular pairs|Coulomb couplings|Global coefficients|Local coefficients|Maximum local state|Iterations|Representation setup|Hamiltonian warmup|Distributed solve time|Solve time per iteration|Hamiltonian calls|Krylov|Electronic work|Internuclear work|Coulomb work|Halo exchange|Weak exchange|Local reductions|Collective reductions' \
    ang_H2p_l*.out
```

Repeat each series at least three times and use the median solve time. The
example reports the angular-pair and Coulomb-coupling counts, global and local
coefficient counts, approximate maximum local state size, setup and warmup
times, total distributed solve time, and solve time per iteration. Setup,
warmup, state gathering, and the optional serial reference are excluded from
`Distributed solve time`.

The component times are accumulated independently on every rank and reduced
with `MPI.MAX` after the solve. They therefore report the slowest rank for each
phase rather than rank zero's local time. Electronic, local internuclear, and
Coulomb timings measure computation. Halo exchange, returned weak-form bridge
exchange, and collective-reduction timings include the time spent waiting for
other ranks. Local physical-inner-product work is reported separately from its
collective reduction. The collective line reports both MPI call count and the
number of scalar values reduced; batched Krylov projections lower the former
without hiding the latter.

The phase maxima are not generally additive: the slowest electronic rank need
not be the slowest communication rank. Use them to locate imbalance and wait
time, while continuing to use `Distributed solve time` for speedup and
efficiency. The warm-up Hamiltonian action is reset out of all component
counters.

Archive summarized results using the schema and conventions in
[`benchmarks/non_bo_diatomic_mpi_scaling_saga.md`](../../benchmarks/non_bo_diatomic_mpi_scaling_saga.md).
Raw Slurm output should remain outside the repository.

For every fixed `l_max`, calculate strong-scaling speedup and efficiency from
the one-rank MPI result:

```text
speedup(p) = T(1) / T(p)
efficiency(p) = T(1) / (p*T(p))
```

Do not compare total solve times across angular cutoffs without also comparing
their iteration counts. Use `Solve time per iteration` to assess how the local
Hamiltonian workload grows, while retaining total solve time as the practical
time-to-solution measure.

## Compare serial, threaded, MPI, and hybrid Hamiltonian execution

Phase 6 benchmarks a fixed number of two-dimensional Hamiltonian actions. It
does not perform imaginary-time propagation and must not be interpreted as a
ground-state convergence calculation. The benchmark uses the same physical
input and deterministic complex state for every execution configuration and
performs an untimed comparison with the serial element-local Hamiltonian.

The submission helper compares every factorization

```text
total cores = MPI ranks * threads per rank
```

for each requested total core count. Its defaults are 1, 2, 4, and 8 total
cores with three repetitions. Thus, the four-core group contains `1 x 4`,
`2 x 2`, and `4 x 1`, allowing shared-memory, hybrid, and pure-MPI execution
to be compared at the same allocation size.

Start with a short deployment smoke test:

```bash
export GRID_TDSE_PHASE6_REPETITIONS=1
export GRID_TDSE_PHASE6_ACTIONS=2
export GRID_TDSE_PHASE6_WARMUP_ACTIONS=1
export GRID_TDSE_PHASE6_RUN_ROOT="${PROJECT_ROOT}/jobs/grid-tdse/phase6-smoke"

bash submit_hamiltonian_execution_baseline.sh 1 2
```

After checking correctness and runtime, unset the smoke overrides and submit
the controlled series:

```bash
unset GRID_TDSE_PHASE6_REPETITIONS
unset GRID_TDSE_PHASE6_ACTIONS
unset GRID_TDSE_PHASE6_WARMUP_ACTIONS
export GRID_TDSE_PHASE6_RUN_ROOT="${PROJECT_ROOT}/jobs/grid-tdse/phase6-hamiltonian"

bash submit_hamiltonian_execution_baseline.sh 1 2 4 8
```

The helper refuses to replace an existing manifest. Select a new run
directory for every series. Each run produces:

- a Slurm output file below `output/`;
- a complete JSON record below `results/`; and
- `phase6_jobs.csv`, mapping job IDs to core, rank, thread, and repetition
  settings.

The JSON file records NumPy and SciPy build information, detected BLAS thread
pools, thread environment variables, CPU model and affinity, Slurm placement,
MPI version, Git revision, memory use, component profiles, and numerical
agreement. `threadpoolctl` is included in the development dependencies and
must be present in the Saga environment.

The job template assigns `SLURM_CPUS_PER_TASK` threads to BLAS/OpenMP and
binds each task to that many physical cores. Numba remains independently fixed
to one thread because the current element-local Numba kernels are not parallel
kernels. Override `GRID_TDSE_NUMBA_THREADS` only in a separate experiment;
mixing that change into the baseline would make configurations incomparable.

Useful optional placement controls are:

```bash
export GRID_TDSE_PHASE6_CONSTRAINT=<Slurm-feature>
export GRID_TDSE_PHASE6_NODELIST=<single-node>
export GRID_TDSE_PHASE6_EXCLUDE=<comma-separated-nodes>
export GRID_TDSE_PHASE6_EXCLUSIVE=1
```

Use a node constraint when one is documented for the desired Saga node type.
A fixed node is useful for diagnosing an outlier but should not silently become
the only production benchmark. Exclusive-node runs reduce interference but
can consume substantially more allocation; use them deliberately.

Inspect the numerical and timing summaries with:

```bash
grep --text -H -E \
    'Execution kind|MPI ranks|Threads per rank|Total allocated cores|Process grid|Global coefficients|Timed actions|Wall time per action|Process CPU time|Allocated-core utilization|Peak memory|Coordinate-[12] work|Coulomb work|halo|weak|Hamiltonian max error|Result JSON' \
    "${GRID_TDSE_PHASE6_RUN_ROOT}/output/"*.out
```

Inspect Slurm resource accounting using the job IDs from the manifest:

```bash
job_ids=$(tail -n +2 "${GRID_TDSE_PHASE6_RUN_ROOT}/phase6_jobs.csv" \
    | cut -d, -f1 \
    | paste -sd, -)

sacct -j "${job_ids}" \
    --format=JobID,JobName%34,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS,NodeList
```

Generate medians, timing spreads, speedups, efficiencies, and an archive-ready
CSV after all jobs have finished:

```bash
cd "${SRC_ROOT}/grid-tdse"
python benchmarks/summarize_non_bo_diatomic_2d_execution.py \
    "${GRID_TDSE_PHASE6_RUN_ROOT}/results" \
    --csv "${GRID_TDSE_PHASE6_RUN_ROOT}/summary.csv"
```

The summarizer keeps every source filename and job ID in the CSV and reports
the full minimum-to-maximum timing spread. Inspect configurations with large
spreads rather than deleting their slow repetitions.

Compare medians across the three repetitions, retain and document outliers,
and require the same Hamiltonian result before interpreting speedup. The
benchmark's allocated-core utilization is based on summed process CPU time
divided by wall time times allocated cores. Use it together with Slurm's CPU
statistics and the recorded thread-pool state; it is not by itself proof that
every library used all assigned threads efficiently.

The completed Saga baseline used `l_max=2` and `l_max=4` workloads with 5000
timed actions. Pure MPI was the fastest configuration at every tested core
count from two through eight cores. Use one physical CPU core per MPI rank and
set OpenMP, OpenBLAS, MKL, and Numba to one thread per rank for subsequent
SIL/SIA jobs unless a propagation-level benchmark demonstrates otherwise. In
particular, do not request extra CPUs per task for the current element-local
Numba kernels: they are serial kernels, so unused per-rank CPUs increase the
allocation without accelerating that work.

For the recommended pure-MPI configuration, the essential Slurm settings are:

```text
--ntasks=<number-of-MPI-ranks>
--cpus-per-task=1
--cpu-bind=cores
```

and the numerical thread variables should all be one. Choose a process grid
whose dimensions multiply to the rank count and are supported by the
electronic and internuclear element counts. The archived measurements and
their timing-variability caveats are in
[`benchmarks/non_bo_diatomic_mpi_scaling_saga.md`](../../benchmarks/non_bo_diatomic_mpi_scaling_saga.md).

## Notes

- The current internuclear grid has four FEM-DVR elements, so it supports at
  most four ranks with the element-aligned decomposition.
- Keep all validation runs on identical grids, angular truncations, symmetry
  sectors, CODATA data, initial states, timesteps, and tolerances.
- Use the `devel` quality of service only for short validation jobs. Remove
  `#SBATCH --qos=devel` when a longer production allocation is needed.
- Do not copy the local laptop option `--mca btl self,sm` to Saga. It disables
  network transports and is unsuitable for multi-node jobs.

The Saga launch recommendations follow the official Sigma2 documentation for
[Saga job scripts](https://documentation.sigma2.no/jobs/job_scripts/saga_job_scripts.html)
and [running MPI applications](https://documentation.sigma2.no/jobs/guides/running_mpi_jobs.html).
