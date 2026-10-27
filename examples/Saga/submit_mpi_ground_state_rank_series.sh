#!/bin/bash

set -euo pipefail

system=${1:-H2+}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
job_script="${script_dir}/job_mpi.sh"
system_label=${system//+/p}

for ranks in 1 2 4 8; do
    sbatch \
        --job-name="gs_${system_label}_${ranks}r" \
        --ntasks="${ranks}" \
        "${job_script}" \
        examples/mpi_non_bo_diatomic_ground_state.py \
        --system "${system}" \
        --no-serial-comparison
done
