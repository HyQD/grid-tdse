#!/bin/bash

set -euo pipefail

system=${1:-H2+}
shift || true

if (( $# == 0 )); then
    l_max_values=(2 4)
else
    l_max_values=("$@")
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
job_script="${script_dir}/job_mpi.sh"
system_label=${system//+/p}

for l_max in "${l_max_values[@]}"; do
    if [[ ! "${l_max}" =~ ^[0-9]+$ ]]; then
        echo "l_max values must be nonnegative integers: ${l_max}" >&2
        exit 2
    fi

    for ranks in 1 2 4 8; do
        sbatch \
            --job-name="ang_${system_label}_l${l_max}_${ranks}r" \
            --ntasks="${ranks}" \
            "${job_script}" \
            examples/mpi_non_bo_diatomic_ground_state.py \
            --system "${system}" \
            --l-max "${l_max}" \
            --no-serial-comparison
    done
done
