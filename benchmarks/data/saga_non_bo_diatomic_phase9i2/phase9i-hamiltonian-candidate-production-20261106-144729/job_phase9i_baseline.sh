#!/bin/bash

#SBATCH --job-name=grid_tdse_phase9i
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=4G
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

if [[ $(command -v python) != "${ENV_PREFIX}/bin/python" ]]; then
    echo "Failed to activate ${ENV_PREFIX}" >&2
    exit 1
fi

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_coll="^hcoll"
export GRID_TDSE_VALIDATE_THREADS=1
export GRID_TDSE_TOTAL_CORES="${SLURM_NTASKS}"

cache_root="${TMPDIR:-/tmp}/grid-tdse-${SLURM_JOB_ID}"
mkdir -p "${cache_root}/matplotlib" "${cache_root}/numba"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_ROOT="${cache_root}/numba"

cd "${SRC_ROOT}/grid-tdse"

: "${GRID_TDSE_EXPECTED_COMMIT:?Submission did not record the expected commit}"
current_commit=$(git rev-parse HEAD)
if [[ ${current_commit} != "${GRID_TDSE_EXPECTED_COMMIT}" ]]; then
    echo "Checkout changed after submission" >&2
    echo "Expected: ${GRID_TDSE_EXPECTED_COMMIT}" >&2
    echo "Current:  ${current_commit}" >&2
    exit 1
fi

script=${1:?Pass a benchmark worker as the first argument}
shift

echo "Python:            $(command -v python)"
echo "Repository commit: ${current_commit}"
echo "MPI module:        ${GRID_TDSE_MPI_MODULE}"
echo "MPI ranks:         ${SLURM_NTASKS}"
echo "CPUs per rank:     ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory per rank:   ${SLURM_MEM_PER_CPU:-unknown} MiB"
echo "Node list:         ${SLURM_JOB_NODELIST:-unknown}"
echo "Script:            ${script}"
module list

echo "Running serial compute-node import and Gaunt-backend preflight"
python - <<'PY'
import importlib.util
import sys

import grid_tdse
import integrator_lib
import numba
import numpy
from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend
from mpi4py import MPI

specification = importlib.util.find_spec("py3nj")
print("Python:", sys.version)
print("NumPy:", numpy.__version__, numpy.__file__)
print("Numba:", numba.__version__, numba.__file__)
print("grid-tdse:", grid_tdse.__file__)
print("integrator-lib:", integrator_lib.__file__)
print("py3nj available:", specification is not None)
print("py3nj path:", None if specification is None else specification.origin)
print("Gaunt backend:", get_gaunt_backend(resolve=True))
print(MPI.Get_library_version().rstrip("\x00"))
PY

time srun \
    --mpi=pmix \
    --ntasks="${SLURM_NTASKS}" \
    --cpus-per-task=1 \
    --hint=nomultithread \
    --distribution=block:block \
    --cpu-bind=cores \
    bash -c '
        rank_cache="${NUMBA_CACHE_ROOT}/rank-${SLURM_PROCID}"
        mkdir -p "${rank_cache}"
        export NUMBA_CACHE_DIR="${rank_cache}"
        exec python -u "$@"
    ' bash "${script}" "$@"
