# Two-radial construction-memory diagnostic on Saga

This workflow validates the lightweight element-local construction path by
answering two questions:

1. how much memory each MPI rank retains without dense global one-body and
   expanded pair operators; and
2. whether `grid-lib` resolves Gaunt coefficients through `py3nj` or SymPy on
   the Saga login and compute nodes.

It performs no Hamiltonian action, ground-state solve, or propagation.

The `production` profile uses the current `4e11` electronic and `4e21`
internuclear grids. The `assessment` profile reproduces the radial dimensions
from the performance assessment at `l_max=8` and `l_max=16`, with an
intermediate `48e11/24e11` case at `l_max=12`. The pre-optimization baseline is
archived under
`benchmarks/data/saga_non_bo_diatomic_construction_memory/`. Assessment jobs
retain the conservative 14-GiB request until the `l_max=16` post-change peak
has been measured on Saga.

## Prepare Saga

Update and activate the established environment as described in
[`../README.md`](../README.md). Confirm that `mpi4py` uses the configured Saga
Open MPI module. Then check the login-node angular backend:

```bash
python - <<'PY'
import importlib.util
from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend

specification = importlib.util.find_spec("py3nj")
print("py3nj available:", specification is not None)
print("py3nj path:", None if specification is None else specification.origin)
print("Gaunt backend:", get_gaunt_backend(resolve=True))
PY
```

The desired final line is `Gaunt backend: py3nj`. If it reports `sympy`, install
the compatible `py3nj` package before interpreting angular setup timings:

```bash
conda install --prefix "${ENV_PREFIX}" --channel conda-forge py3nj
```

## Submit the production-grid series

Run the submitter from the repository checkout:

```bash
cd "${SRC_ROOT}/grid-tdse"

examples/spherical/two_radial/Saga/construction_memory/submit_construction_memory.sh \
    --profile production
```

The three independent configurations are serialized with `afterany` by
default. This avoids simultaneous stress on the project environment. The
manifest and copied scripts are stored under the printed run root.

Monitor them with:

```bash
run_root=$(cat \
    "${PROJECT_ROOT}/jobs/grid-tdse/construction-memory/latest_production_run.txt")

column -s, -t "${run_root}/jobs.csv"

job_ids=$(awk -F, 'NR > 1 {print $1}' "${run_root}/jobs.csv" | paste -sd, -)
squeue -j "${job_ids}" --format='%.18i %.32j %.10T %.10M %.10l %.20R'
```

## Submit the assessment-sized series

Run `l_max=8` first and verify its memory record before requesting the larger
cases:

```bash
examples/spherical/two_radial/Saga/construction_memory/submit_construction_memory.sh \
    --profile assessment \
    --l-max 8 \
    --run-name construction-memory-assessment-l8
```

If it completes normally, submit `l_max=12,16`:

```bash
examples/spherical/two_radial/Saga/construction_memory/submit_construction_memory.sh \
    --profile assessment \
    --l-max 12 16
```

These initial measurements use one MPI process. That is intentional: one
process measures the per-rank retained and transient cost without multiplying
an unknown post-change peak. The benchmark emulates rank zero of the target
logical process grid (`2x2` for production and `2*l_max` by `l_max` for
assessment). It computes the corresponding complex state-tile size but does
not allocate that state. Afterward, replication can be checked with four real
ranks on the production profile or the assessment `l_max=8` case by passing
`--ranks 4`.

## Inspect completed jobs

```bash
run_root=$(cat \
    "${PROJECT_ROOT}/jobs/grid-tdse/construction-memory/latest_production_run.txt")

grep --text -H -E \
    'Profile:|l_max:|MPI processes:|Logical layout ranks:|Radial points:|Radial operator storage:|Gaunt backend:|py3nj available:|Final RSS/rank:|Peak RSS/rank:|Result JSON:' \
    "${run_root}/output/"*.out

grep --text -H -E \
    'Traceback|MemoryError|OUT_OF_MEMORY|CANCELLED|TIMEOUT|MPI_ABORT' \
    "${run_root}/output/"*.out || true

python "${run_root}/summarize_non_bo_diatomic_construction_memory.py" \
    "${run_root}/results"
```

Record Slurm's independent high-water measurement:

```bash
job_ids=$(awk -F, 'NR > 1 {print $1}' "${run_root}/jobs.csv" | paste -sd, -)

sacct -j "${job_ids}" \
    --format=JobID,JobName%34,State,ExitCode,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS,NodeList \
    > "${run_root}/sacct.txt"

cat "${run_root}/sacct.txt"
```

Interpret the JSON `current_rss_mib` as retained process memory and
`peak_rss_mib`/Slurm `MaxRSS` as high-water measurements. The array inventory
deduplicates NumPy views by their root allocation. A successful lightweight run
has no `expanded_pair_operators` or `dense_one_body` category. Compare RSS,
process peak, Slurm `MaxRSS`, and total inventoried arrays with the matching
archived baseline; do not infer transient-memory removal from retained arrays
alone.
