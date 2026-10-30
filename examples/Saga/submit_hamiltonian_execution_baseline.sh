#!/bin/bash

set -euo pipefail

if (( $# == 0 )); then
    total_core_counts=(1 2 4 8)
else
    total_core_counts=("$@")
fi

repetitions=${GRID_TDSE_PHASE6_REPETITIONS:-3}
system=${GRID_TDSE_PHASE6_SYSTEM:-H2+}
l_max=${GRID_TDSE_PHASE6_L_MAX:-2}
actions=${GRID_TDSE_PHASE6_ACTIONS:-20}
warmup_actions=${GRID_TDSE_PHASE6_WARMUP_ACTIONS:-2}
walltime=${GRID_TDSE_PHASE6_WALLTIME:-00:30:00}
run_root=${GRID_TDSE_PHASE6_RUN_ROOT:-"$(pwd)/phase6-hamiltonian"}

for value in "${repetitions}" "${actions}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "repetitions and actions must be positive integers" >&2
        exit 2
    fi
done
if [[ ! "${l_max}" =~ ^[0-9]+$ ]]; then
    echo "l_max must be a nonnegative integer" >&2
    exit 2
fi
if [[ ! "${warmup_actions}" =~ ^[0-9]+$ ]]; then
    echo "warmup action count must be a nonnegative integer" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
job_script="${script_dir}/job_hamiltonian_baseline.sh"
output_dir="${run_root}/output"
result_dir="${run_root}/results"
mkdir -p "${output_dir}" "${result_dir}"

manifest="${run_root}/phase6_jobs.csv"
if [[ -e "${manifest}" ]]; then
    echo "Refusing to overwrite existing manifest: ${manifest}" >&2
    exit 1
fi
echo "job_id,repetition,total_cores,mpi_ranks,threads_per_rank,kind,system,l_max,output_pattern" > "${manifest}"

extra_sbatch=()
if [[ ${GRID_TDSE_PHASE6_EXCLUSIVE:-0} == 1 ]]; then
    extra_sbatch+=(--exclusive)
fi
if [[ -n ${GRID_TDSE_PHASE6_CONSTRAINT:-} ]]; then
    extra_sbatch+=(--constraint="${GRID_TDSE_PHASE6_CONSTRAINT}")
fi
if [[ -n ${GRID_TDSE_PHASE6_NODELIST:-} ]]; then
    extra_sbatch+=(--nodelist="${GRID_TDSE_PHASE6_NODELIST}")
fi
if [[ -n ${GRID_TDSE_PHASE6_EXCLUDE:-} ]]; then
    extra_sbatch+=(--exclude="${GRID_TDSE_PHASE6_EXCLUDE}")
fi

system_label=${system//+/p}
for total_cores in "${total_core_counts[@]}"; do
    if [[ ! "${total_cores}" =~ ^[1-9][0-9]*$ ]]; then
        echo "total core counts must be positive integers: ${total_cores}" >&2
        exit 2
    fi
    for (( ranks = 1; ranks <= total_cores; ranks++ )); do
        if (( total_cores % ranks != 0 )); then
            continue
        fi
        threads=$((total_cores / ranks))
        if (( ranks == 1 && threads == 1 )); then
            kind=serial
        elif (( ranks == 1 )); then
            kind=shared
        elif (( threads == 1 )); then
            kind=mpi
        else
            kind=hybrid
        fi
        for (( repetition = 1; repetition <= repetitions; repetition++ )); do
            job_name="p6_${system_label}_c${total_cores}_q${ranks}_t${threads}_r${repetition}"
            output_pattern="${output_dir}/${job_name}-%j.out"
            job_id=$(
                sbatch \
                    --parsable \
                    --job-name="${job_name}" \
                    --nodes=1 \
                    --ntasks="${ranks}" \
                    --cpus-per-task="${threads}" \
                    --time="${walltime}" \
                    --output="${output_pattern}" \
                    --export="ALL,GRID_TDSE_TOTAL_CORES=${total_cores}" \
                    "${extra_sbatch[@]}" \
                    "${job_script}" \
                    benchmarks/non_bo_diatomic_2d_execution.py \
                    --system "${system}" \
                    --l-max "${l_max}" \
                    --actions "${actions}" \
                    --warmup-actions "${warmup_actions}" \
                    --result-directory "${result_dir}"
            )
            job_id=${job_id%%;*}
            echo "${job_id},${repetition},${total_cores},${ranks},${threads},${kind},${system},${l_max},${output_pattern}" >> "${manifest}"
            echo "Submitted ${job_name} as ${job_id}"
        done
    done
done

echo "Manifest: ${manifest}"
echo "Results:  ${result_dir}"
