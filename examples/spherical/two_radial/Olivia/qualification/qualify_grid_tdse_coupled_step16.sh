#!/bin/bash

#SBATCH --job-name=grid_tdse_prepare
#SBATCH --partition=small
#SBATCH --qos=devel
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
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
script_root="${PWD}"
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

run_root="${GRID_TDSE_WORK_ROOT}/qualification/${GRID_TDSE_DEPLOYMENT}/coupled-step16-${SLURM_JOB_ID}"
result_root="${run_root}/coupled/results"
state_root="${run_root}/coupled/states"
general_result_root="${run_root}/general/results"
general_state_root="${run_root}/general/states"
cache_root="${LOCALSCRATCH}/grid-tdse-coupled-step16-${SLURM_JOB_ID}"
reference_archive="${state_root}/reference-final.npz"
checkpoint_archive="${state_root}/reference-midpoint.npz"
reference_json="${result_root}/reference-1r.json"

mkdir -p \
    "${result_root}" \
    "${state_root}" \
    "${general_result_root}" \
    "${general_state_root}" \
    "${cache_root}"

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#allocated_nodes[@]}" -ne 1 ]]
then
    echo "Expected exactly one allocated node" >&2
    printf 'Allocated node: %s\n' "${allocated_nodes[@]}" >&2
    exit 1
fi

printf 'Allocated node: %s\n' "${allocated_nodes[@]}"
echo "Python: $(command -v python)"
echo "Deployment: ${deployment_prefix}"
echo "Runtime environment: ${runtime_environment}"
echo "Open MPI transport: ob1/sm,self"
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
            export OMPI_MCA_pml=ob1
            export OMPI_MCA_btl=sm,self
            unset OMPI_MCA_mtl
            unset OMPI_MCA_mtl_ofi_av
            unset PRTE_MCA_ras_base_launch_orted_on_hn
            unset FI_PROVIDER
            unset FI_CXI_RX_MATCH_MODE
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

general_state="${general_state_root}/h2p_production_l2.npz"
general_reference="${general_result_root}/dynamics-1r.json"

printf '\nPreparing the general H2+ reference and ground state\n'
cd "${GRID_TDSE_SRC_ROOT}/grid-tdse"
srun \
    --exact \
    --nodes=1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --distribution=block:block \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    bash -c '
        cache_root=$1
        shift
        export OMPI_MCA_pml=ob1
        export OMPI_MCA_btl=sm,self
        unset OMPI_MCA_mtl
        unset OMPI_MCA_mtl_ofi_av
        unset PRTE_MCA_ras_base_launch_orted_on_hn
        unset FI_PROVIDER
        unset FI_CXI_RX_MATCH_MODE
        rank_cache="${cache_root}/general-reference-${SLURM_PROCID}"
        mkdir -p \
            "${rank_cache}/numba" \
            "${rank_cache}/matplotlib" \
            "${rank_cache}/pycache"
        export NUMBA_CACHE_DIR="${rank_cache}/numba"
        export MPLCONFIGDIR="${rank_cache}/matplotlib"
        export PYTHONPYCACHEPREFIX="${rank_cache}/pycache"
        exec "$@"
    ' bash "${cache_root}" \
    python examples/spherical/two_radial/mpi_non_bo_diatomic_2d_dynamics.py \
        --system H2+ \
        --process-grid 1x1 \
        --l-max 2 \
        --electronic-nodes 0 4 8 12 16 \
        --electronic-points-per-element 11 \
        --internuclear-nodes 0 2 4 6 8 \
        --internuclear-points-per-element 21 \
        --codata-set 2018 \
        --ground-state-path "${general_state}" \
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
        --output-json "${general_reference}" \
        --overwrite

python - "${result_root}" "${general_reference}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
general_reference = pathlib.Path(sys.argv[2])
cases = (
    ("reference-1r", 1, [1, 1], 1),
    ("one-node-4r", 4, [2, 2], 1),
)

print("\nCoupled Step 16 preparation")
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

general = json.loads(general_reference.read_text(encoding="utf-8"))
if general["status"] != "PASS":
    raise SystemExit(f"general reference status is {general['status']}")
if general["mpi_ranks"] != 1 or general["process_grid"] != [1, 1]:
    raise SystemExit("general reference did not use the 1x1 layout")

print("General H2+ one-rank reference: PASS")
print("Coupled Step 16 preparation: PASS")
print(f"Preparation root: {root.parents[1]}")
PY
