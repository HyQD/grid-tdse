#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_coulomb_kernel_benchmarks.sh --stage STAGE [OPTIONS]

Stages:
  tile       Complex l_max=8 tile-size sweep at 16 MPI ranks.
  angular    Real/complex l_max=4,6,8 sweep at the selected tile size.
  threads    Fixed-16-core MPI/Numba-thread sweep at l_max=8.

Options:
  --system SYSTEM       H2+, HD+, or D2+ (default: H2+)
  --tile-size N         selected tile size for angular/threads (default: 256)
  --repetitions N       repetitions per configuration (default: 3)
  --run-name NAME       run-directory name (default: stage plus timestamp)
  --perf                collect per-rank perf stat counters
  --concurrent          submit jobs without dependency serialization
  -h, --help            show this message

Site overrides are available through GRID_TDSE_COULOMB_{PARTITION,QOS,
NODELIST,EXCLUDE,CONSTRAINT,EXCLUSIVE,MEM_PER_CPU,WALLTIME}.
EOF
}

stage=
system=H2+
tile_size=256
repetitions=3
run_name=
collect_perf=0
concurrent=0

while (( $# > 0 )); do
    case $1 in
        --stage)
            stage=${2:?--stage requires a value}
            shift 2
            ;;
        --system)
            system=${2:?--system requires a value}
            shift 2
            ;;
        --tile-size)
            tile_size=${2:?--tile-size requires a value}
            shift 2
            ;;
        --repetitions)
            repetitions=${2:?--repetitions requires a value}
            shift 2
            ;;
        --run-name)
            run_name=${2:?--run-name requires a value}
            shift 2
            ;;
        --perf)
            collect_perf=1
            shift
            ;;
        --concurrent)
            concurrent=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case ${stage} in
    tile|angular|threads) ;;
    *)
        echo "--stage must be tile, angular, or threads" >&2
        exit 2
        ;;
esac
case ${system} in
    H2+|HD+|D2+) ;;
    *)
        echo "--system must be H2+, HD+, or D2+" >&2
        exit 2
        ;;
esac
for value in "${tile_size}" "${repetitions}"; do
    if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
        echo "tile size and repetitions must be positive integers" >&2
        exit 2
    fi
done

: "${PROJECT_ID:?Source grid_tdse_paths.sh before submitting}"
: "${PROJECT_ROOT:?Source grid_tdse_paths.sh before submitting}"
: "${SRC_ROOT:?Source grid_tdse_paths.sh before submitting}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/../../.." && pwd)
if [[ ${repository_root} != "${SRC_ROOT}/grid-tdse" ]]; then
    echo "Warning: submitting scripts from ${repository_root}" >&2
fi

timestamp=$(date +%Y%m%d-%H%M%S)
run_name=${run_name:-"coulomb-${stage}-${timestamp}"}
if [[ ! ${run_name} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "run name may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
fi
run_root="${PROJECT_ROOT}/jobs/grid-tdse/coulomb-kernel/${run_name}"
if [[ -e ${run_root} ]]; then
    echo "Refusing to reuse existing run directory: ${run_root}" >&2
    exit 1
fi

output_dir="${run_root}/output"
result_dir="${run_root}/results"
perf_dir="${run_root}/perf"
mkdir -p "${output_dir}" "${result_dir}" "${perf_dir}"

job_script="${run_root}/job_coulomb_kernel.sh"
benchmark_script="${run_root}/non_bo_diatomic_coulomb_execution.py"
summary_script="${run_root}/summarize_non_bo_diatomic_coulomb_execution.py"
cp "${script_dir}/job_coulomb_kernel.sh" "${job_script}"
cp "${repository_root}/benchmarks/non_bo_diatomic_coulomb_execution.py" \
    "${benchmark_script}"
cp "${repository_root}/benchmarks/summarize_non_bo_diatomic_coulomb_execution.py" \
    "${summary_script}"
chmod +x "${job_script}"

manifest="${run_root}/coulomb_jobs.csv"
echo "job_id,stage,repetition,system,l_max,state_dtype,tile_size,total_cores,mpi_ranks,numba_threads,process_grid,actions,dependency,output_pattern" > "${manifest}"

walltime=${GRID_TDSE_COULOMB_WALLTIME:-02:00:00}
mem_per_cpu=${GRID_TDSE_COULOMB_MEM_PER_CPU:-2G}
extra_sbatch=()
if [[ -n ${GRID_TDSE_COULOMB_PARTITION:-} ]]; then
    extra_sbatch+=(--partition="${GRID_TDSE_COULOMB_PARTITION}")
fi
if [[ -n ${GRID_TDSE_COULOMB_QOS:-} ]]; then
    extra_sbatch+=(--qos="${GRID_TDSE_COULOMB_QOS}")
fi
if [[ -n ${GRID_TDSE_COULOMB_NODELIST:-} ]]; then
    extra_sbatch+=(--nodelist="${GRID_TDSE_COULOMB_NODELIST}")
fi
if [[ -n ${GRID_TDSE_COULOMB_EXCLUDE:-} ]]; then
    extra_sbatch+=(--exclude="${GRID_TDSE_COULOMB_EXCLUDE}")
fi
if [[ -n ${GRID_TDSE_COULOMB_CONSTRAINT:-} ]]; then
    extra_sbatch+=(--constraint="${GRID_TDSE_COULOMB_CONSTRAINT}")
fi
if [[ ${GRID_TDSE_COULOMB_EXCLUSIVE:-0} == 1 ]]; then
    extra_sbatch+=(--exclusive)
fi

export GRID_TDSE_PROJECT_ROOT="${PROJECT_ROOT}"
export GRID_TDSE_PERF_DIR="${perf_dir}"
if (( collect_perf )); then
    export GRID_TDSE_PERF_EVENTS=${GRID_TDSE_PERF_EVENTS:-task-clock,cycles,instructions,cache-references,cache-misses}
else
    unset GRID_TDSE_PERF_EVENTS || true
fi

previous_job_id=
submission_mode=sequential
if (( concurrent )); then
    submission_mode=concurrent
fi

submit_case() {
    local l_max=$1
    local state_dtype=$2
    local case_tile_size=$3
    local total_cores=$4
    local ranks=$5
    local numba_threads=$6
    local process_grid=$7
    local actions=$8
    local label=$9
    local repetition job_name output_pattern job_id dependency_record
    local -a dependency_options

    if (( total_cores != ranks * numba_threads )); then
        echo "Invalid fixed-core configuration: ${ranks} x ${numba_threads}" >&2
        exit 2
    fi
    for (( repetition = 1; repetition <= repetitions; repetition++ )); do
        job_name="ck_${stage}_${label}_r${repetition}"
        output_pattern="${output_dir}/${job_name}-%j.out"
        dependency_options=()
        dependency_record=none
        if (( ! concurrent )) && [[ -n ${previous_job_id} ]]; then
            dependency_options=(
                --dependency="afterany:${previous_job_id}"
            )
            dependency_record="afterany:${previous_job_id}"
        fi
        job_id=$(
            sbatch \
                --parsable \
                --account="${PROJECT_ID}" \
                --job-name="${job_name}" \
                --nodes=1 \
                --ntasks="${ranks}" \
                --cpus-per-task="${numba_threads}" \
                --hint=nomultithread \
                --mem-per-cpu="${mem_per_cpu}" \
                --time="${walltime}" \
                --output="${output_pattern}" \
                --export="ALL,GRID_TDSE_TOTAL_CORES=${total_cores},GRID_TDSE_NUMBA_THREADS=${numba_threads}" \
                "${dependency_options[@]}" \
                "${extra_sbatch[@]}" \
                "${job_script}" \
                "${benchmark_script}" \
                --system "${system}" \
                --l-max "${l_max}" \
                --state-dtype "${state_dtype}" \
                --tile-size "${case_tile_size}" \
                --process-grid "${process_grid%x*}" "${process_grid#*x}" \
                --warmup-actions 2 \
                --actions "${actions}" \
                --result-directory "${result_dir}"
        )
        job_id=${job_id%%;*}
        echo "${job_id},${stage},${repetition},${system},${l_max},${state_dtype},${case_tile_size},${total_cores},${ranks},${numba_threads},${process_grid},${actions},${dependency_record},${output_pattern}" >> "${manifest}"
        echo "Submitted ${job_name}: ${job_id} (${dependency_record})"
        previous_job_id=${job_id}
    done
}

system_label=${system//+/p}
case ${stage} in
    tile)
        for candidate in 32 64 128 256 512 1024 2048; do
            submit_case 8 complex "${candidate}" 16 16 1 4x4 2000 \
                "${system_label}_l8_c16_q16_t1_z${candidate}"
        done
        ;;
    angular)
        for state_dtype in real complex; do
            dtype_label=${state_dtype:0:1}
            submit_case 4 "${state_dtype}" "${tile_size}" 8 8 1 2x4 10000 \
                "${system_label}_l4_${dtype_label}_c8_q8_t1_z${tile_size}"
            submit_case 6 "${state_dtype}" "${tile_size}" 16 16 1 4x4 5000 \
                "${system_label}_l6_${dtype_label}_c16_q16_t1_z${tile_size}"
            submit_case 8 "${state_dtype}" "${tile_size}" 16 16 1 4x4 2000 \
                "${system_label}_l8_${dtype_label}_c16_q16_t1_z${tile_size}"
        done
        ;;
    threads)
        for state_dtype in real complex; do
            dtype_label=${state_dtype:0:1}
            submit_case 8 "${state_dtype}" "${tile_size}" 16 16 1 4x4 2000 \
                "${system_label}_l8_${dtype_label}_c16_q16_t1_z${tile_size}"
            submit_case 8 "${state_dtype}" "${tile_size}" 16 8 2 2x4 2000 \
                "${system_label}_l8_${dtype_label}_c16_q8_t2_z${tile_size}"
            submit_case 8 "${state_dtype}" "${tile_size}" 16 4 4 2x2 2000 \
                "${system_label}_l8_${dtype_label}_c16_q4_t4_z${tile_size}"
        done
        ;;
esac

{
    echo "date=$(date --iso-8601=seconds)"
    echo "stage=${stage}"
    echo "system=${system}"
    echo "submission_mode=${submission_mode}"
    echo "grid_tdse_commit=$(git -C "${repository_root}" rev-parse HEAD)"
    echo "mpi_module=${GRID_TDSE_MPI_MODULE:-not-loaded-in-submission-shell}"
    echo "python=$(command -v python)"
    python -c 'from mpi4py import MPI; print("mpi_library=" + MPI.Get_library_version().rstrip("\x00"))'
    sha256sum "${job_script}" "${benchmark_script}" "${summary_script}"
} > "${run_root}/provenance.txt"

latest_file="${PROJECT_ROOT}/jobs/grid-tdse/coulomb-kernel/latest_${stage}_run.txt"
printf '%s\n' "${run_root}" > "${latest_file}"

echo "Run root:  ${run_root}"
echo "Manifest:  ${manifest}"
echo "Results:   ${result_dir}"
echo "Latest:    ${latest_file}"
echo "Mode:      ${submission_mode}"
if [[ -n ${previous_job_id} ]]; then
    echo "Final job: ${previous_job_id}"
fi
