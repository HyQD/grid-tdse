# Converged non-BO ground-state chains on Saga

This workflow stress-tests the Cartesian MPI SIL ground-state solver and
creates continuation states for later real-time dynamics. It submits one
dependent Slurm chain per molecule and electronic grid:

- the production 11-point electronic FEM-DVR chain uses
  `l_max = 0, 2, 4, 6, 8, 10`;
- the exploratory 21-point chain initially uses
  `l_max = 0, 2, 4, 6`;
- `--extend-21` also submits `l_max = 8, 10` after the lower cutoffs have
  proved affordable.

Every converged state is embedded into the next angular basis. A failed or
timed-out job prevents its descendants from starting because Slurm uses
`afterok` dependencies. Unconverged calculations return a nonzero status and
do not write a state archive.

## Symmetry sectors

The legacy `XH_nBO_rDVR/XHplus/3D/ground.py` label `sp` corresponds to the
modern homonuclear H2+/D2+ sector used here: total inversion parity `+1`,
positive-particle pair-exchange parity `+1`, pair-exchange-adapted angular
bases, and `M_tot = 0`.

For HD+ the positive particles are distinguishable, so pair exchange is not a
symmetry. Its legacy label is `s`, and the modern calculation uses inversion
parity `+1`, no pair-exchange parity, no pair-exchange basis reduction, and
`M_tot = 0`.

Both grids use the element boundaries

```text
r: 0, 4, 8, 12, 16
R: 0, 2, 4, 6, 8
```

with 21 points per internuclear element. The two chains differ only in using
11 or 21 points per electronic element.

## 1. Update and verify the Saga environment

On a Saga login node:

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

cd "${SRC_ROOT}/grid-tdse"
git switch main
git pull --ff-only origin main
python -m pip install --no-deps --no-build-isolation --editable ".[dev]"
```

`hash -r` clears Bash's remembered executable locations after environment
activation, ensuring `python` and related commands are resolved again from
the active Conda environment.

Verify the checkout, dependencies, and MPI linkage:

```bash
python -m pip check
command -v python
git rev-parse HEAD
python -c 'from mpi4py import MPI; print(MPI.Get_library_version().rstrip("\x00"))'
python examples/mpi_non_bo_diatomic_ground_state_chain.py --help
bash -n examples/Saga/ground_state_chains/job_ground_state_chain.sh
bash -n examples/Saga/ground_state_chains/submit_ground_state_chains.sh
```

Do not update the checkout or environment while a chain is queued or running.
The submission helper copies the worker and job wrapper into the run directory
and records their checksums, but the imported editable package still comes
from `${SRC_ROOT}/grid-tdse`.

## 2. Submit one system first

The safest first production submission is the full 11-point H2+ chain:

```bash
cd "${SRC_ROOT}/grid-tdse"

examples/Saga/ground_state_chains/submit_ground_state_chains.sh \
    --system H2+ \
    --electronic-grid 11 \
    --run-name h2p-e11-ground-state-chain
```

Inspect the generated manifest and dependencies:

```bash
run_root=$(cat "${PROJECT_ROOT}/jobs/grid-tdse/ground-state-chains/latest_run.txt")
column -s, -t "${run_root}/jobs.csv"
squeue --me -o '%.18i %.30j %.2t %.10M %.6D %R'
```

The default one-node mapping is:

| `l_max` | ranks | process grid |
|---:|---:|:---:|
| 0, 2 | 4 | `2x2` |
| 4 | 8 | `2x4` |
| 6, 8, 10 | 16 | `4x4` |

Each MPI rank receives one CPU, and all Numba, OpenMP, OpenBLAS, and MKL thread
pools are limited to one thread. Requesting extra CPUs per rank would reserve
cores that the present local kernels are not designed to use.

## 3. Submit all systems and both grids

After the H2+ 11-point chain behaves as expected, submit H2+, HD+, and D2+.
This creates six independent dependency chains:

```bash
examples/Saga/ground_state_chains/submit_ground_state_chains.sh \
    --all \
    --electronic-grid both
```

The 21-point chains stop at `l_max=6` by default. Extend them only after
reviewing elapsed time, memory, and convergence through that cutoff:

```bash
examples/Saga/ground_state_chains/submit_ground_state_chains.sh \
    --all \
    --electronic-grid 21 \
    --extend-21 \
    --run-name extended-e21-ground-state-chains
```

That command starts new chains at `l_max=0`; it does not append to an older
run directory. This keeps every dependency chain and provenance record
self-contained.

## 4. Optional resource overrides

The conservative wall times can be overridden before submission. Variable
names encode the electronic points per element and cutoff:

```bash
export GRID_TDSE_CHAIN_TIME_E11_L8=12:00:00
export GRID_TDSE_CHAIN_TIME_E11_L10=1-12:00:00
export GRID_TDSE_CHAIN_TIME_E21_L6=1-00:00:00
```

Other supported overrides are:

```bash
export GRID_TDSE_CHAIN_MAX_ITERATIONS=20000
export GRID_TDSE_CHAIN_SAMPLE_INTERVAL=10
export GRID_TDSE_CHAIN_MEM_PER_CPU=3G
# Normally retain the validated default of 0.5:
export GRID_TDSE_CHAIN_TIME_STEP=0.5
# Set only when required by current Saga policy:
export GRID_TDSE_CHAIN_PARTITION=normal
export GRID_TDSE_CHAIN_QOS="<qos-name>"
```

Unset an override to return to the defaults. Do not use the short development
QoS for these production chains.

## 5. Monitor convergence and failures

Locate the current run and list its output:

```bash
run_root=$(cat "${PROJECT_ROOT}/jobs/grid-tdse/ground-state-chains/latest_run.txt")
find "${run_root}/output" -maxdepth 1 -type f -name '*.out' -print | sort
```

Follow a running job with `tail -f`, then summarize completed calculations:

```bash
grep --text -H -E \
    'System:|Electronic points/element:|Initial state:|l_max:|Process grid:|Converged:|Iterations:|Energy:|Eigenstate residual:|Norm:|Distributed solve time:|State archive:' \
    "${run_root}/output/"*.out
```

Check for failures:

```bash
grep --text -H -E \
    'Traceback|FileNotFoundError|ImportError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY|not written' \
    "${run_root}/output/"*.out
```

An empty result is good. Use the manifest to collect Slurm accounting:

```bash
job_ids=$(tail -n +2 "${run_root}/jobs.csv" | cut -d, -f1 | paste -sd, -)
sacct -j "${job_ids}" \
    --format=JobID,JobName%38,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS,NodeList \
    > "${run_root}/sacct.txt"
```

For every completed cutoff require `Converged: True`, a residual near or below
`1e-8`, unit norm, and a reported state archive. Also confirm that each
`l_max>0` output names the preceding archive as its initial state.

## 6. Retrieve and retain the results

From the local workstation, replace the placeholders and run name:

```bash
run_name=h2p-e11-ground-state-chain
mkdir -p "$HOME/Documents/HyQD/saga-results/grid-tdse/ground-state-chains"

rsync -av \
    "<username>@saga.sigma2.no:/cluster/projects/<project-id>/<username>/jobs/grid-tdse/ground-state-chains/${run_name}/" \
    "$HOME/Documents/HyQD/saga-results/grid-tdse/ground-state-chains/${run_name}/"
```

The `.npz` files are restart states, not disposable benchmark output. Keep the
complete run directory, including `jobs.csv`, `provenance.txt`, logs, Slurm
output, and accounting. Do not commit large production states directly to Git.
After measuring their sizes, prefer an external versioned data archive with
checksums and a small repository manifest. Only genuinely small reference
states needed by tests should be considered for the repository itself.
