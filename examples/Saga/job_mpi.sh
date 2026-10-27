#!/bin/bash

#SBATCH --account=<project-id>
#SBATCH --job-name=grid_tdse_mpi
#SBATCH --qos=devel
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.out

set -euo pipefail

project_root="/cluster/projects/<project-id>/<username>"

module --quiet reset
module load StdEnv
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
source "${project_root}/grid_tdse_paths.sh"

: "${GRID_TDSE_MPI_MODULE:?Set GRID_TDSE_MPI_MODULE in grid_tdse_paths.sh}"
module load "${GRID_TDSE_MPI_MODULE}"

conda activate base
conda activate "${ENV_PREFIX}"
hash -r

if [[ "$(command -v python)" != "${ENV_PREFIX}/bin/python" ]]; then
    echo "Failed to activate ${ENV_PREFIX}" >&2
    exit 1
fi

cache_root="${TMPDIR:-/tmp}/grid-tdse-${SLURM_JOB_ID}"
mkdir -p "${cache_root}/matplotlib" "${cache_root}/numba"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_coll="^hcoll"

cd "${SRC_ROOT}/grid-tdse"

script=${1:-examples/mpi_non_bo_diatomic_ground_state.py}
if (( $# > 0 )); then
    shift
fi

echo "Python:     $(command -v python)"
echo "MPI module: ${GRID_TDSE_MPI_MODULE}"
echo "MPI ranks:  ${SLURM_NTASKS}"
echo "Script:     ${script}"
module list
python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'

time srun \
    --mpi=pmix \
    --ntasks="${SLURM_NTASKS}" \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    python "${script}" "$@"
