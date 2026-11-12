#!/bin/bash

#SBATCH --job-name=grid_tdse_multi_node
#SBATCH --partition=large
#SBATCH --qos=devel
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --output=grid_tdse_multi_node-%j.out

set -euo pipefail

: "${GRID_TDSE_PATHS_FILE:?Submit with GRID_TDSE_PATHS_FILE=<absolute-path>}"
: "${GRID_TDSE_DEPLOYMENT:?Submit with GRID_TDSE_DEPLOYMENT=<deployment-id>}"
: "${GRID_TDSE_PREPARATION_JOB_ID:?Submit with GRID_TDSE_PREPARATION_JOB_ID=<job-id>}"
source "${GRID_TDSE_PATHS_FILE}"

module --force purge
module load "${GRID_TDSE_MPI_STACK}"
module load "${GRID_TDSE_MPI_MODULE}"

deployment_prefix="${GRID_TDSE_ENV_ROOT}/${GRID_TDSE_DEPLOYMENT}"
manifest_root="${GRID_TDSE_ROOT}/manifests/${GRID_TDSE_DEPLOYMENT}"
source_manifest="${manifest_root}/source-commits.txt"
runtime_environment="${manifest_root}/runtime-environment.sh"
script_root="${PWD}"
coupled_driver="${script_root}/qualify_coupled_step16.py"

test -d "${deployment_prefix}"
test -s "${source_manifest}"
test -s "${runtime_environment}"
test -s "${coupled_driver}"

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

# Olivia multi-node Open MPI transport documented by Sigma2 for custom builds
# and containers.  The forced cm/ofi path fails instead of silently falling
# back to TCP when Slingshot/libfabric initialization is unavailable.
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_av=table
export PRTE_MCA_ras_base_launch_orted_on_hn=1
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid

run_root="${GRID_TDSE_WORK_ROOT}/qualification/${GRID_TDSE_DEPLOYMENT}/multi-node-${SLURM_JOB_ID}"
preparation_root="${GRID_TDSE_WORK_ROOT}/qualification/${GRID_TDSE_DEPLOYMENT}/coupled-step16-${GRID_TDSE_PREPARATION_JOB_ID}"
coupled_result_root="${run_root}/coupled/results"
general_result_root="${run_root}/general/results"
placement_root="${run_root}/placement"
cache_root="${LOCALSCRATCH}/grid-tdse-multi-node-${SLURM_JOB_ID}"

coupled_reference_archive="${preparation_root}/coupled/states/reference-final.npz"
coupled_checkpoint_archive="${preparation_root}/coupled/states/reference-midpoint.npz"
coupled_reference_json="${preparation_root}/coupled/results/reference-1r.json"
general_state_path="${preparation_root}/general/states/h2p_production_l2.npz"
general_reference_json="${preparation_root}/general/results/dynamics-1r.json"

mkdir -p \
    "${coupled_result_root}" \
    "${general_result_root}" \
    "${placement_root}" \
    "${cache_root}"

for preparation_file in \
    "${coupled_reference_archive}" \
    "${coupled_checkpoint_archive}" \
    "${coupled_reference_json}" \
    "${general_state_path}" \
    "${general_reference_json}"
do
    if [[ ! -s "${preparation_file}" ]]
    then
        echo "Missing preparation artifact: ${preparation_file}" >&2
        exit 1
    fi
done

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
echo "Preparation: ${preparation_root}"
echo "Open MPI transport: cm/ofi"
echo "Libfabric provider: ${FI_PROVIDER}"
module list

python -m pip check

cd "${GRID_TDSE_SRC_ROOT}/grid-tdse"

printf '\nRunning coupled Step 16 with 16 ranks across two nodes\n'
srun \
    --exact \
    --nodes=2 \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --cpus-per-task=1 \
    --distribution=block:block \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    bash -c '
        cache_root=$1
        rank_cache="${cache_root}/coupled-16r-${SLURM_PROCID}"
        mkdir -p \
            "${rank_cache}/numba" \
            "${rank_cache}/matplotlib" \
            "${rank_cache}/pycache"
        export NUMBA_CACHE_DIR="${rank_cache}/numba"
        export MPLCONFIGDIR="${rank_cache}/matplotlib"
        export PYTHONPYCACHEPREFIX="${rank_cache}/pycache"
        exec "$@"
    ' bash "${cache_root}" \
    python "${coupled_driver}" \
        --process-grid 4 4 \
        --reference-archive "${coupled_reference_archive}" \
        --checkpoint-archive "${coupled_checkpoint_archive}" \
        --reference-json "${coupled_reference_json}" \
        --result-json "${coupled_result_root}/two-node-16r.json" \
        --overwrite

# The single-node gate already covers 4- and 8-rank layouts.  Retain only the
# smallest cross-node launch and the Step 16 layout in this whole-node job.
ranks=(2 16)
tasks_per_node=(1 8)
process_grids=(1x2 4x4)

for index in "${!ranks[@]}"
do
    rank_count=${ranks[index]}
    per_node=${tasks_per_node[index]}
    process_grid=${process_grids[index]}
    placement_file="${placement_root}/placement-${rank_count}r.txt"
    result_json="${general_result_root}/dynamics-${rank_count}r.json"

    printf '\nChecking placement with %s ranks (%s per node)\n' \
        "${rank_count}" "${per_node}"

    srun \
        --exact \
        --nodes=2 \
        --ntasks="${rank_count}" \
        --ntasks-per-node="${per_node}" \
        --cpus-per-task=1 \
        --distribution=block:block \
        --cpu-bind=cores \
        --kill-on-bad-exit=1 \
        bash -c '
            cache_root=$1
            rank_count=$2
            shift 2
            rank_cache="${cache_root}/placement-${rank_count}-${SLURM_PROCID}"
            mkdir -p \
                "${rank_cache}/numba" \
                "${rank_cache}/matplotlib" \
                "${rank_cache}/pycache"
            export NUMBA_CACHE_DIR="${rank_cache}/numba"
            export MPLCONFIGDIR="${rank_cache}/matplotlib"
            export PYTHONPYCACHEPREFIX="${rank_cache}/pycache"
            exec "$@"
        ' bash "${cache_root}" "${rank_count}" \
        python -c '
from mpi4py import MPI
import socket
import sys

communicator = MPI.COMM_WORLD
shared_communicator = communicator.Split_type(MPI.COMM_TYPE_SHARED)
local_rank = shared_communicator.rank
shared_communicator.Free()
total = communicator.allreduce(communicator.rank, op=MPI.SUM)
if communicator.rank == 0 and communicator.size == 2:
    print(
        "MPI library: "
        + MPI.Get_library_version().rstrip("\x00"),
        file=sys.stderr,
        flush=True,
    )
print(
    f"rank={communicator.rank} local={local_rank} "
    f"host={socket.gethostname()} rank_sum={total}",
    flush=True,
)
' | sort -V > "${placement_file}"

    unique_nodes=$(
        awk '
            {
                for (field = 1; field <= NF; field++) {
                    if ($field ~ /^host=/) {
                        sub(/^host=/, "", $field)
                        print $field
                    }
                }
            }
        ' "${placement_file}" |
        sort -u |
        wc -l
    )
    if [[ "${unique_nodes}" -ne 2 ]]
    then
        echo "Rank placement did not span two nodes" >&2
        cat "${placement_file}" >&2
        exit 1
    fi

    cat "${placement_file}"
    printf 'Running production-grid dynamics with process grid %s\n' \
        "${process_grid}"

    srun \
        --exact \
        --nodes=2 \
        --ntasks="${rank_count}" \
        --ntasks-per-node="${per_node}" \
        --cpus-per-task=1 \
        --distribution=block:block \
        --cpu-bind=cores \
        --kill-on-bad-exit=1 \
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
            --ground-state-path "${general_state_path}" \
            --initial-state-policy require \
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

python - \
    "${coupled_result_root}/two-node-16r.json" \
    "${general_result_root}" \
    "${general_reference_json}" \
    "${placement_root}" <<'PY'
import json
import pathlib
import sys

coupled_path = pathlib.Path(sys.argv[1])
result_root = pathlib.Path(sys.argv[2])
reference_path = pathlib.Path(sys.argv[3])
placement_root = pathlib.Path(sys.argv[4])
ranks_to_grid = {
    2: "1x2",
    16: "4x4",
}

coupled = json.loads(coupled_path.read_text(encoding="utf-8"))
if coupled["status"] != "PASS":
    raise SystemExit(f"coupled Step 16 status is {coupled['status']}")
if coupled["mpi_ranks"] != 16 or coupled["process_grid"] != [4, 4]:
    raise SystemExit("coupled Step 16 did not use the 16-rank 4x4 layout")
if len(coupled["hosts"]) != 2:
    raise SystemExit(f"coupled Step 16 used hosts {coupled['hosts']}")

print("\nCoupled Step 16 two-node result")
print(
    "ranks  grid  nodes  wall(s)  max-H-error  state-error  "
    "restart-error  CAP-loss"
)
print(
    f"{coupled['mpi_ranks']:5d}  4x4  "
    f"{len(coupled['hosts']):5d}  {coupled['wall_seconds']:7.3f}  "
    f"{coupled['maximum_component_error']:.3e}  "
    f"{coupled['state_difference_from_one_rank']:.3e}  "
    f"{coupled['restart_difference_from_uninterrupted']:.3e}  "
    f"{coupled['final_cap_norm_loss']:.3e}"
)
print("Coupled Step 16 Olivia qualification: PASS")

reports = {}
print("\nTwo-node production-grid scaling")
print(
    "ranks  grid  wall(s)  speedup-vs-1r  state-difference  "
    "time-series-difference"
)

for ranks, process_grid in ranks_to_grid.items():
    placement_lines = (
        placement_root / f"placement-{ranks}r.txt"
    ).read_text(encoding="utf-8").splitlines()
    hosts = {
        field.removeprefix("host=")
        for line in placement_lines
        for field in line.split()
        if field.startswith("host=")
    }
    if len(placement_lines) != ranks or len(hosts) != 2:
        raise SystemExit(
            f"{ranks}-rank placement did not contain {ranks} ranks on two nodes"
        )

    report = json.loads(
        (result_root / f"dynamics-{ranks}r.json").read_text(encoding="utf-8")
    )
    reports[ranks] = report
    if report["status"] != "PASS":
        raise SystemExit(f"{ranks}-rank dynamics status is {report['status']}")
    if report["mpi_ranks"] != ranks:
        raise SystemExit(f"expected {ranks} ranks, got {report['mpi_ranks']}")
    if "x".join(str(value) for value in report["process_grid"]) != process_grid:
        raise SystemExit(f"unexpected process grid for {ranks} ranks")
    if report["state_difference"] >= 1.0e-8:
        raise SystemExit(f"{ranks}-rank state difference exceeds tolerance")
    if report["time_series_maximum_difference"] >= 1.0e-8:
        raise SystemExit(f"{ranks}-rank time-series difference exceeds tolerance")

reference = json.loads(reference_path.read_text(encoding="utf-8"))
if reference["status"] != "PASS":
    raise SystemExit(f"one-rank reference status is {reference['status']}")
if reference["mpi_ranks"] != 1 or reference["process_grid"] != [1, 1]:
    raise SystemExit("one-rank reference did not use the 1x1 layout")
reference_wall = reference["distributed_wall_time"]
for ranks, process_grid in ranks_to_grid.items():
    report = reports[ranks]
    wall_time = report["distributed_wall_time"]
    print(
        f"{ranks:5d}  {process_grid:>4s}  {wall_time:7.3f}  "
        f"{reference_wall / wall_time:14.3f}  "
        f"{report['state_difference']:.3e}  "
        f"{report['time_series_maximum_difference']:.3e}"
    )

print("Multi-node stress correctness: PASS")
PY
