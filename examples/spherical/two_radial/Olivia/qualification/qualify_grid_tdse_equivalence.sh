#!/bin/bash

#SBATCH --job-name=grid_tdse_equivalence
#SBATCH --partition=small
#SBATCH --qos=devel
#SBATCH --time=00:15:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=grid_tdse_equivalence-%j.out

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

    printf 'Source verified: %s %s\n' "${repository}" "${actual_commit}"
done < "${source_manifest}"

export PATH="${deployment_prefix}/bin:${PATH}"
hash -r

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=sm,self

run_root="${GRID_TDSE_WORK_ROOT}/qualification/${GRID_TDSE_DEPLOYMENT}/equivalence-${SLURM_JOB_ID}"
state_path="${run_root}/states/h2p_l1.npz"
result_json="${run_root}/results/equivalence.json"
cache_root="${LOCALSCRATCH}/grid-tdse-equivalence-${SLURM_JOB_ID}"

mkdir -p \
    "${run_root}/states" \
    "${run_root}/results" \
    "${cache_root}/numba"

export NUMBA_CACHE_ROOT="${cache_root}/numba"

cd "${GRID_TDSE_SRC_ROOT}/grid-tdse"

srun \
    --ntasks="${SLURM_NTASKS}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    bash -c '
        rank_cache="${NUMBA_CACHE_ROOT}/rank-${SLURM_PROCID}"
        mkdir -p \
            "${rank_cache}/numba" \
            "${rank_cache}/matplotlib" \
            "${rank_cache}/pycache"
        export NUMBA_CACHE_DIR="${rank_cache}/numba"
        export MPLCONFIGDIR="${rank_cache}/matplotlib"
        export PYTHONPYCACHEPREFIX="${rank_cache}/pycache"
        exec "$@"
    ' bash \
    python examples/spherical/two_radial/mpi_non_bo_diatomic_2d_dynamics.py \
        --system H2+ \
        --process-grid 1x2 \
        --l-max 1 \
        --electronic-nodes 0 1 2 \
        --electronic-points-per-element 3 \
        --internuclear-nodes 0 1 2 \
        --internuclear-points-per-element 3 \
        --codata-set 2018 \
        --ground-state-path "${state_path}" \
        --initial-state-policy compute-if-missing \
        --integrator sia \
        --krylov-reduction-mode batched \
        --time-step 0.01 \
        --final-time 0.02 \
        --field-strength 0.02 \
        --field-gauge length \
        --carrier-cycles 1 \
        --sample-interval 1 \
        --output-json "${result_json}" \
        --overwrite

python - "${result_json}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))

if report["status"] != "PASS":
    raise SystemExit(f"equivalence status is {report['status']}")
if report["mpi_ranks"] != 2:
    raise SystemExit(f"expected two ranks, got {report['mpi_ranks']}")
if report["state_difference"] >= 1.0e-8:
    raise SystemExit("serial/MPI state difference exceeds tolerance")
if report["time_series_maximum_difference"] >= 1.0e-8:
    raise SystemExit("serial/MPI time-series difference exceeds tolerance")

print("Grid-tdse serial/MPI equivalence: PASS")
print(f"State difference: {report['state_difference']:.3e}")
print(
    "Time-series maximum difference: "
    f"{report['time_series_maximum_difference']:.3e}"
)
print(f"Result JSON: {path}")
PY
