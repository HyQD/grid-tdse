#!/bin/bash

#SBATCH --account=<project-id>
#SBATCH --job-name=grid_tdse_serial
#SBATCH --qos=devel
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
conda activate base
conda activate "${ENV_PREFIX}"
hash -r

if [[ "$(command -v python)" != "${ENV_PREFIX}/bin/python" ]]; then
    echo "Failed to activate ${ENV_PREFIX}" >&2
    exit 1
fi

if (( $# == 0 )); then
    echo "Usage: sbatch $0 SCRIPT [SCRIPT_ARGUMENT ...]" >&2
    exit 2
fi

cache_root="${TMPDIR:-/tmp}/grid-tdse-${SLURM_JOB_ID}"
mkdir -p "${cache_root}/matplotlib" "${cache_root}/numba"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SRC_ROOT}/grid-tdse"

script=$1
shift

echo "Python: $(command -v python)"
echo "Script: ${script}"

time srun \
    --ntasks=1 \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    bash -c 'exec python -u "$@"' bash "${script}" "$@"
