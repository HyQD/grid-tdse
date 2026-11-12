#!/bin/bash

#SBATCH --job-name=grid_tdse_qualify
#SBATCH --partition=small
#SBATCH --qos=devel
#SBATCH --time=00:30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=grid_tdse_qualify-%j.out

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

cache_root="${LOCALSCRATCH}/grid-tdse-${SLURM_JOB_ID}"
mkdir -p \
    "${cache_root}/matplotlib" \
    "${cache_root}/numba/serial" \
    "${cache_root}/pycache"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_ROOT="${cache_root}/numba"
export NUMBA_CACHE_DIR="${cache_root}/numba/serial"
export PYTHONPYCACHEPREFIX="${cache_root}/pycache"

cd "${LOCALSCRATCH}"

echo "Python: $(command -v python)"
echo "Deployment: ${deployment_prefix}"
module list

python -m pip check
python -c 'import grid_tdse; print("grid-tdse:", grid_tdse.__file__)'
python -c 'from mpi4py import MPI; print(MPI.Get_library_version().rstrip("\x00"))'

cd "${GRID_TDSE_SRC_ROOT}/grid-tdse"
python -m pytest -q

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
        exec python -c "from mpi4py import MPI; communicator=MPI.COMM_WORLD; print(f\"rank={communicator.rank} size={communicator.size}\")"
    '
