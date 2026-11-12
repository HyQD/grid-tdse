#!/bin/bash

#SBATCH --job-name=grid_tdse_single_node
#SBATCH --partition=small
#SBATCH --qos=devel
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=grid_tdse_single_node-%j.out

set -euo pipefail

: "${GRID_TDSE_PATHS_FILE:?Submit with GRID_TDSE_PATHS_FILE=<absolute-path>}"
: "${GRID_TDSE_DEPLOYMENT:?Submit with GRID_TDSE_DEPLOYMENT=<deployment-id>}"
source "${GRID_TDSE_PATHS_FILE}"

module --force purge
module load "${GRID_TDSE_MPI_STACK}"
module load "${GRID_TDSE_MPI_MODULE}"

deployment_prefix="${GRID_TDSE_ENV_ROOT}/${GRID_TDSE_DEPLOYMENT}"
manifest_root="${GRID_TDSE_ROOT}/manifests/${GRID_TDSE_DEPLOYMENT}"
source_manifest="${manifest_root}/source-commits.txt"

test -d "${deployment_prefix}"
test -s "${source_manifest}"

while read -r repository expected_commit extra
do
    if [[ -n "${extra:-}" || ! "${expected_commit}" =~ ^[0-9a-f]{40}$ ]]
    then
        echo "Invalid source manifest entry for ${repository}" >&2
        exit 1
    fi

    repository_root="${GRID_TDSE_SRC_ROOT}/${repository}"
    actual_commit=$(git -C "${repository_root}" rev-parse HEAD)
    source_status=$(git -C "${repository_root}" status --short)

    if [[ "${actual_commit}" != "${expected_commit}" ]]
    then
        printf 'Source commit mismatch for %s: %s != %s\n' \
            "${repository}" "${actual_commit}" "${expected_commit}" >&2
        exit 1
    fi

    if [[ -n "${source_status}" ]]
    then
        printf 'Source worktree is dirty: %s\n%s\n' \
            "${repository}" "${source_status}" >&2
        exit 1
    fi
done < "${source_manifest}"

runtime_environment="${manifest_root}/runtime-environment.sh"
test -s "${runtime_environment}"
source "${runtime_environment}"

export PATH="${deployment_prefix}/bin:${PATH}"
hash -r

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=sm,self

run_root="${GRID_TDSE_WORK_ROOT}/qualification/${GRID_TDSE_DEPLOYMENT}/single-node-${SLURM_JOB_ID}"
result_root="${run_root}/results"
state_path="${run_root}/states/h2p_production_l2.npz"
cache_root="${LOCALSCRATCH}/grid-tdse-single-node-${SLURM_JOB_ID}"

mkdir -p \
    "${result_root}" \
    "${run_root}/states" \
    "${cache_root}"

cd "${GRID_TDSE_SRC_ROOT}/grid-tdse"

echo "Running l_max=8 production-grid construction-memory check"
srun \
    --ntasks=1 \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    bash -c '
        rank_cache="$1/memory"
        mkdir -p \
            "${rank_cache}/numba" \
            "${rank_cache}/matplotlib" \
            "${rank_cache}/pycache"
        export NUMBA_CACHE_DIR="${rank_cache}/numba"
        export MPLCONFIGDIR="${rank_cache}/matplotlib"
        export PYTHONPYCACHEPREFIX="${rank_cache}/pycache"
        shift
        exec "$@"
    ' bash "${cache_root}" \
    python benchmarks/non_bo_diatomic_construction_memory.py \
        --profile-name olivia-single-node-production \
        --l-max 8 \
        --layout-ranks 16 \
        --process-grid 4 4 \
        --result-json "${result_root}/construction-memory-l8.json" \
        --overwrite

ranks=(1 2 4 8 16)
process_grids=(1x1 1x2 2x2 2x4 4x4)

for index in "${!ranks[@]}"
do
    rank_count=${ranks[index]}
    process_grid=${process_grids[index]}
    result_json="${result_root}/dynamics-${rank_count}r.json"

    printf '\nRunning production-grid dynamics with %s ranks (%s)\n' \
        "${rank_count}" "${process_grid}"

    srun \
        --ntasks="${rank_count}" \
        --cpus-per-task=1 \
        --cpu-bind=cores \
        bash -c '
            cache_root=$1
            rank_count=$2
            shift 2
            rank_cache="${cache_root}/dynamics-${rank_count}-${SLURM_PROCID}"
            mkdir -p \
                "${rank_cache}/numba" \
                "${rank_cache}/matplotlib" \
                "${rank_cache}/pycache"
            export NUMBA_CACHE_DIR="${rank_cache}/numba"
            export MPLCONFIGDIR="${rank_cache}/matplotlib"
            export PYTHONPYCACHEPREFIX="${rank_cache}/pycache"
            exec "$@"
        ' bash "${cache_root}" "${rank_count}" \
        python examples/spherical/two_radial/mpi_non_bo_diatomic_2d_dynamics.py \
            --system H2+ \
            --process-grid "${process_grid}" \
            --l-max 2 \
            --electronic-nodes 0 4 8 12 16 \
            --electronic-points-per-element 11 \
            --internuclear-nodes 0 2 4 6 8 \
            --internuclear-points-per-element 21 \
            --codata-set 2018 \
            --ground-state-path "${state_path}" \
            --initial-state-policy compute-if-missing \
            --integrator sia \
            --krylov-reduction-mode batched \
            --krylov-rtol 1.0e-13 \
            --krylov-atol 1.0e-15 \
            --time-step 0.25 \
            --final-time 25.0 \
            --field-strength 0.02 \
            --field-gauge length \
            --carrier-cycles 1 \
            --sample-interval 10 \
            --output-json "${result_json}" \
            --overwrite
done

python - "${result_root}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
memory = json.loads(
    (root / "construction-memory-l8.json").read_text(encoding="utf-8")
)

if memory["problem"]["radial_operator_mode"] != "element-local":
    raise SystemExit("construction did not use element-local radial storage")
if memory["backend"]["resolved"] != "py3nj":
    raise SystemExit("construction did not use the py3nj backend")

wall_times = {}
print("\nSingle-node production-grid scaling")
print("ranks  wall(s)  speedup  state-difference  time-series-difference")

for ranks in (1, 2, 4, 8, 16):
    report = json.loads(
        (root / f"dynamics-{ranks}r.json").read_text(encoding="utf-8")
    )
    if report["status"] != "PASS":
        raise SystemExit(f"{ranks}-rank dynamics status is {report['status']}")
    if report["mpi_ranks"] != ranks:
        raise SystemExit(f"expected {ranks} ranks, got {report['mpi_ranks']}")
    if report["state_difference"] >= 1.0e-8:
        raise SystemExit(f"{ranks}-rank state difference exceeds tolerance")
    if report["time_series_maximum_difference"] >= 1.0e-8:
        raise SystemExit(f"{ranks}-rank time-series difference exceeds tolerance")

    wall_times[ranks] = report["distributed_wall_time"]

serial_wall = wall_times[1]
for ranks in (1, 2, 4, 8, 16):
    report = json.loads(
        (root / f"dynamics-{ranks}r.json").read_text(encoding="utf-8")
    )
    print(
        f"{ranks:5d}  {wall_times[ranks]:7.3f}  "
        f"{serial_wall / wall_times[ranks]:7.3f}  "
        f"{report['state_difference']:.3e}  "
        f"{report['time_series_maximum_difference']:.3e}"
    )

best_parallel = min(wall_times[ranks] for ranks in (2, 4, 8, 16))
if best_parallel >= serial_wall:
    raise SystemExit("no tested single-node MPI layout improved on one rank")

print("Single-node stress correctness: PASS")
print(f"Best observed speedup: {serial_wall / best_parallel:.3f}")
PY
