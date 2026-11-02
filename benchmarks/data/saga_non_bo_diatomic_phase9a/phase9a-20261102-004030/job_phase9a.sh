#!/bin/bash

set -euo pipefail

project_root=${GRID_TDSE_PROJECT_ROOT:?Submit with GRID_TDSE_PROJECT_ROOT exported}

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
export NUMBA_CACHE_ROOT="${cache_root}/numba"
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_coll="^hcoll"

script=${1:?Pass the archived ground-state-chain Python worker}
shift

cd "${SRC_ROOT}/grid-tdse"

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-unknown} at $(date)"
echo "Python:       $(command -v python)"
echo "MPI module:   ${GRID_TDSE_MPI_MODULE}"
echo "MPI ranks:    ${SLURM_NTASKS}"
echo "Worker:       ${script}"
module list

echo "Running serial compute-node import and font-cache preflight"
python - <<'PY'
import numpy as np
import scipy
import PIL
from PIL import ImageColor
import matplotlib
from matplotlib import font_manager
import numba
import grid_tdse
from mpi4py import MPI

font_path = font_manager.findfont("DejaVu Sans")
print("NumPy:", np.__version__, np.__file__)
print("SciPy:", scipy.__version__, scipy.__file__)
print("Pillow:", PIL.__version__, ImageColor.__file__)
print("Matplotlib:", matplotlib.__version__, font_path)
print("Numba:", numba.__version__)
print("grid-tdse:", grid_tdse.__file__)
print(MPI.Get_library_version().rstrip("\x00"))
print("Preflight imports complete")
PY

time srun \
    --mpi=pmix \
    --ntasks="${SLURM_NTASKS}" \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    bash -c '
        rank_cache="${NUMBA_CACHE_ROOT}/rank-${SLURM_PROCID}"
        mkdir -p "${rank_cache}"
        export NUMBA_CACHE_DIR="${rank_cache}"
        exec python -u "$@"
    ' bash "${script}" "$@"
