#!/bin/bash

#SBATCH --job-name=grid_tdse_multi_node
#SBATCH --partition=large
#SBATCH --qos=devel
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --output=grid_tdse_multi_node-%j.out

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
result_root="${run_root}/results"
placement_root="${run_root}/placement"
state_path="${run_root}/states/h2p_production_l2.npz"
cache_root="${LOCALSCRATCH}/grid-tdse-multi-node-${SLURM_JOB_ID}"

mkdir -p \
    "${result_root}" \
    "${placement_root}" \
    "${run_root}/states" \
    "${cache_root}"

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
echo "Open MPI transport: cm/ofi"
echo "Libfabric provider: ${FI_PROVIDER}"
module list

python -m pip check

cd "${GRID_TDSE_SRC_ROOT}/grid-tdse"

ranks=(2 4 8 16)
tasks_per_node=(1 2 4 8)
process_grids=(1x2 2x2 2x4 4x4)

for index in "${!ranks[@]}"
do
    rank_count=${ranks[index]}
    per_node=${tasks_per_node[index]}
    process_grid=${process_grids[index]}
    placement_file="${placement_root}/placement-${rank_count}r.txt"
    result_json="${result_root}/dynamics-${rank_count}r.json"

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

python - "${result_root}" "${placement_root}" <<'PY'
import json
import pathlib
import sys

result_root = pathlib.Path(sys.argv[1])
placement_root = pathlib.Path(sys.argv[2])
ranks_to_grid = {
    2: "1x2",
    4: "2x2",
    8: "2x4",
    16: "4x4",
}

reports = {}
print("\nTwo-node production-grid scaling")
print(
    "ranks  grid  wall(s)  relative-to-2r  state-difference  "
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

reference_wall = reports[2]["distributed_wall_time"]
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
