#!/bin/bash

#SBATCH --job-name=grid_tdse_coulomb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${GRID_TDSE_PROJECT_ROOT:?Submit with GRID_TDSE_PROJECT_ROOT exported}"

module --quiet reset
module load StdEnv
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
source "${GRID_TDSE_PROJECT_ROOT}/grid_tdse_paths.sh"

: "${GRID_TDSE_MPI_MODULE:?Set GRID_TDSE_MPI_MODULE in grid_tdse_paths.sh}"
module load "${GRID_TDSE_MPI_MODULE}"

conda activate base
conda activate "${ENV_PREFIX}"
hash -r

if [[ "$(command -v python)" != "${ENV_PREFIX}/bin/python" ]]; then
    echo "Failed to activate ${ENV_PREFIX}" >&2
    exit 1
fi

threads_per_rank=${SLURM_CPUS_PER_TASK:-1}
numba_threads=${GRID_TDSE_NUMBA_THREADS:-${threads_per_rank}}
if (( numba_threads < 1 || numba_threads > threads_per_rank )); then
    echo "GRID_TDSE_NUMBA_THREADS must be in [1, ${threads_per_rank}]" >&2
    exit 2
fi

export NUMBA_NUM_THREADS="${numba_threads}"
export OMP_NUM_THREADS="${numba_threads}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_PLACES="${GRID_TDSE_OMP_PLACES:-cores}"
export OMP_PROC_BIND="${GRID_TDSE_OMP_PROC_BIND:-close}"
export GRID_TDSE_VALIDATE_THREADS=1
export OMPI_MCA_coll="^hcoll"

cache_root="${TMPDIR:-/tmp}/grid-tdse-${SLURM_JOB_ID}"
mkdir -p "${cache_root}/matplotlib" "${cache_root}/numba"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_ROOT="${cache_root}/numba"

cd "${SRC_ROOT}/grid-tdse"

script=${1:-benchmarks/non_bo_diatomic_coulomb_execution.py}
if (( $# > 0 )); then
    shift
fi

echo "Python:             $(command -v python)"
echo "MPI module:         ${GRID_TDSE_MPI_MODULE}"
echo "MPI ranks:          ${SLURM_NTASKS}"
echo "Threads per rank:   ${threads_per_rank}"
echo "Numba threads/rank: ${NUMBA_NUM_THREADS}"
echo "Total cores:        $((SLURM_NTASKS * threads_per_rank))"
echo "OMP places:         ${OMP_PLACES}"
echo "OMP binding:        ${OMP_PROC_BIND}"
echo "Perf events:        ${GRID_TDSE_PERF_EVENTS:-disabled}"
echo "Script:             ${script}"
module list
python -c 'from mpi4py import MPI; print(MPI.Get_library_version().rstrip("\x00"))'

time srun \
    --mpi=pmix \
    --ntasks="${SLURM_NTASKS}" \
    --cpus-per-task="${threads_per_rank}" \
    --hint=nomultithread \
    --distribution=block:block \
    --cpu-bind=cores \
    bash -c '
        rank_cache="${NUMBA_CACHE_ROOT}/rank-${SLURM_PROCID}"
        mkdir -p "${rank_cache}"
        export NUMBA_CACHE_DIR="${rank_cache}"
        if [[ -n ${GRID_TDSE_PERF_EVENTS:-} ]]; then
            : "${GRID_TDSE_PERF_DIR:?Set GRID_TDSE_PERF_DIR with perf events}"
            command -v perf >/dev/null || {
                echo "perf is unavailable on $(hostname)" >&2
                exit 127
            }
            perf_file="${GRID_TDSE_PERF_DIR}/perf-${SLURM_JOB_ID}-rank-${SLURM_PROCID}.csv"
            exec perf stat \
                --no-big-num \
                --field-separator=, \
                --output="${perf_file}" \
                --event="${GRID_TDSE_PERF_EVENTS}" \
                python -u "$@"
        fi
        exec python -u "$@"
    ' bash "${script}" "$@"
