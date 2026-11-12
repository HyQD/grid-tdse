#!/bin/bash

#SBATCH --job-name=grid_tdse_coupled16
#SBATCH --partition=large
#SBATCH --qos=devel
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=grid_tdse_coupled_step16-%j.out

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
runtime_environment="${manifest_root}/runtime-environment.sh"
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
driver="${script_root}/qualify_coupled_step16.py"

test -d "${deployment_prefix}"
test -s "${source_manifest}"
test -s "${runtime_environment}"
test -s "${driver}"

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

source "${runtime_environment}"
export PATH="${deployment_prefix}/bin:${PATH}"
hash -r

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Force the documented Olivia multi-node Slingshot transport.  Failure is
# preferable to an unnoticed fallback to TCP.
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_av=table
export PRTE_MCA_ras_base_launch_orted_on_hn=1
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid

run_root="${GRID_TDSE_WORK_ROOT}/qualification/${GRID_TDSE_DEPLOYMENT}/coupled-step16-${SLURM_JOB_ID}"
result_root="${run_root}/results"
state_root="${run_root}/states"
cache_root="${LOCALSCRATCH}/grid-tdse-coupled-step16-${SLURM_JOB_ID}"
reference_archive="${state_root}/reference-final.npz"
checkpoint_archive="${state_root}/reference-midpoint.npz"
reference_json="${result_root}/reference-1r.json"

mkdir -p "${result_root}" "${state_root}" "${cache_root}"

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#allocated_nodes[@]}" -ne 2 ]]
then
    echo "Expected exactly two allocated nodes" >&2
    printf 'Allocated node: %s\n' "${allocated_nodes[@]}" >&2
    exit 1
fi

printf 'Allocated node: %s\n' "${allocated_nodes[@]}"
echo "Python: $(command -v python)"
echo "Deployment: ${deployment_prefix}"
echo "Runtime environment: ${runtime_environment}"
echo "Open MPI transport: cm/ofi"
module list
python -m pip check

run_case()
{
    local label=$1
    local nodes=$2
    local ranks=$3
    local ranks_per_node=$4
    local process_rows=$5
    local process_columns=$6
    shift 6

    local result_json="${result_root}/${label}.json"
    printf '\nRunning %s with %s ranks on %s node(s), grid %sx%s\n' \
        "${label}" "${ranks}" "${nodes}" \
        "${process_rows}" "${process_columns}"

    srun \
        --exact \
        --nodes="${nodes}" \
        --ntasks="${ranks}" \
        --ntasks-per-node="${ranks_per_node}" \
        --cpus-per-task=1 \
        --distribution=block:block \
        --cpu-bind=cores \
        --kill-on-bad-exit=1 \
        bash -c '
            cache_root=$1
            label=$2
            shift 2
            rank_cache="${cache_root}/${label}-${SLURM_PROCID}"
            mkdir -p \
                "${rank_cache}/numba" \
                "${rank_cache}/matplotlib" \
                "${rank_cache}/pycache"
            export NUMBA_CACHE_DIR="${rank_cache}/numba"
            export MPLCONFIGDIR="${rank_cache}/matplotlib"
            export PYTHONPYCACHEPREFIX="${rank_cache}/pycache"
            exec "$@"
        ' bash "${cache_root}" "${label}" \
        python "${driver}" \
            --process-grid "${process_rows}" "${process_columns}" \
            --reference-archive "${reference_archive}" \
            --checkpoint-archive "${checkpoint_archive}" \
            --reference-json "${reference_json}" \
            --result-json "${result_json}" \
            --overwrite \
            "$@"
}

run_case reference-1r 1 1 1 1 1 --write-reference
run_case one-node-4r 1 4 4 2 2
run_case two-node-16r 2 16 8 4 4

python - "${result_root}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cases = (
    ("reference-1r", 1, [1, 1], 1),
    ("one-node-4r", 4, [2, 2], 1),
    ("two-node-16r", 16, [4, 4], 2),
)

print("\nCoupled Step 16 Olivia qualification")
print(
    "case             ranks  grid  nodes  wall(s)  max-H-error  "
    "state-error  restart-error  CAP-loss  collectives"
)

for label, ranks, process_grid, nodes in cases:
    path = root / f"{label}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "PASS":
        raise SystemExit(f"{label} status is {report['status']}")
    if report["mpi_ranks"] != ranks:
        raise SystemExit(f"{label} used {report['mpi_ranks']} ranks")
    if report["process_grid"] != process_grid:
        raise SystemExit(f"{label} used process grid {report['process_grid']}")
    if len(report["hosts"]) != nodes:
        raise SystemExit(f"{label} used hosts {report['hosts']}")

    print(
        f"{label:16s} {ranks:5d}  "
        f"{process_grid[0]}x{process_grid[1]}  {nodes:5d}  "
        f"{report['wall_seconds']:7.3f}  "
        f"{report['maximum_component_error']:.3e}  "
        f"{report['state_difference_from_one_rank']:.3e}  "
        f"{report['restart_difference_from_uninterrupted']:.3e}  "
        f"{report['final_cap_norm_loss']:.3e}  "
        f"{report['reduction_profile']['collective_calls']:11d}"
    )

print("Coupled Step 16 Olivia qualification: PASS")
print(f"Results: {root}")
PY
