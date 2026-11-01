#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  submit_ground_state_chains.sh --system H2+|HD+|D2+ [options]
  submit_ground_state_chains.sh --all [options]

Options:
  --electronic-grid 11|21|both  Submit one or both r-grid chains (default: both)
  --integrator NAME            sil, sia, or exponential-euler (default: sil)
  --time-step VALUE            Imaginary-time step (default: 0.5)
  --max-iterations N           Maximum ITP steps per cutoff (default: 15000)
  --max-krylov-dimension N     Maximum Krylov dimension (default: 20)
  --max-l-max N                Stop after this even cutoff (default: grid policy)
  --extend-21                  Extend the 21-point chain from l_max=6 to 8 and 10
  --run-name NAME              Run-directory name (default: ground-state-<timestamp>)
  -h, --help                   Show this help

Every l_max>0 job has an afterok dependency on the preceding cutoff. The
11-point chain is 0 -> 2 -> 4 -> 6 -> 8 -> 10. By default the exploratory
21-point chain is 0 -> 2 -> 4 -> 6. --max-l-max overrides those endpoints.

Per-cutoff overrides use GRID_TDSE_CHAIN_TIME_STEP_L<N> and
GRID_TDSE_CHAIN_MAX_ITERATIONS_L<N>, for example
GRID_TDSE_CHAIN_TIME_STEP_L4=0.05.
EOF
}

systems=()
electronic_grid=both
integrator=sil
time_step=0.5
max_iterations=15000
max_krylov_dimension=20
max_l_max=""
extend_21=false
run_name="ground-state-$(date +%Y%m%d-%H%M%S)"

while (( $# > 0 )); do
    case $1 in
        --system)
            [[ $# -ge 2 ]] || { echo "--system requires a value" >&2; exit 2; }
            systems+=("$2")
            shift 2
            ;;
        --all)
            systems=(H2+ HD+ D2+)
            shift
            ;;
        --electronic-grid)
            [[ $# -ge 2 ]] || { echo "--electronic-grid requires a value" >&2; exit 2; }
            electronic_grid=$2
            shift 2
            ;;
        --integrator)
            [[ $# -ge 2 ]] || { echo "--integrator requires a value" >&2; exit 2; }
            integrator=$2
            shift 2
            ;;
        --time-step)
            [[ $# -ge 2 ]] || { echo "--time-step requires a value" >&2; exit 2; }
            time_step=$2
            shift 2
            ;;
        --max-iterations)
            [[ $# -ge 2 ]] || { echo "--max-iterations requires a value" >&2; exit 2; }
            max_iterations=$2
            shift 2
            ;;
        --max-krylov-dimension)
            [[ $# -ge 2 ]] || { echo "--max-krylov-dimension requires a value" >&2; exit 2; }
            max_krylov_dimension=$2
            shift 2
            ;;
        --max-l-max)
            [[ $# -ge 2 ]] || { echo "--max-l-max requires a value" >&2; exit 2; }
            max_l_max=$2
            shift 2
            ;;
        --extend-21)
            extend_21=true
            shift
            ;;
        --run-name)
            [[ $# -ge 2 ]] || { echo "--run-name requires a value" >&2; exit 2; }
            run_name=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

(( ${#systems[@]} > 0 )) || { echo "Choose --system or --all" >&2; exit 2; }
case ${electronic_grid} in
    11) electronic_points=(11) ;;
    21) electronic_points=(21) ;;
    both) electronic_points=(11 21) ;;
    *) echo "--electronic-grid must be 11, 21, or both" >&2; exit 2 ;;
esac
case ${integrator} in
    sil|sia|exponential-euler) ;;
    *) echo "--integrator must be sil, sia, or exponential-euler" >&2; exit 2 ;;
esac
[[ ${time_step} =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || {
    echo "--time-step must be a positive number" >&2
    exit 2
}
awk -v value="${time_step}" 'BEGIN { exit !(value > 0) }' || {
    echo "--time-step must be positive" >&2
    exit 2
}
[[ ${max_krylov_dimension} =~ ^[1-9][0-9]*$ ]] || {
    echo "--max-krylov-dimension must be a positive integer" >&2
    exit 2
}
[[ ${max_iterations} =~ ^[1-9][0-9]*$ ]] || {
    echo "--max-iterations must be a positive integer" >&2
    exit 2
}
if [[ -n ${max_l_max} ]]; then
    case ${max_l_max} in
        0|2|4|6|8|10) ;;
        *) echo "--max-l-max must be one of 0, 2, 4, 6, 8, or 10" >&2; exit 2 ;;
    esac
fi
time_step=${GRID_TDSE_CHAIN_TIME_STEP:-${time_step}}
max_iterations=${GRID_TDSE_CHAIN_MAX_ITERATIONS:-${max_iterations}}
max_krylov_dimension=${GRID_TDSE_CHAIN_MAX_KRYLOV_DIMENSION:-${max_krylov_dimension}}
[[ ${time_step} =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] \
    && awk -v value="${time_step}" 'BEGIN { exit !(value > 0) }' || {
        echo "GRID_TDSE_CHAIN_TIME_STEP must be positive" >&2
        exit 2
    }
[[ ${max_iterations} =~ ^[1-9][0-9]*$ ]] || {
    echo "GRID_TDSE_CHAIN_MAX_ITERATIONS must be a positive integer" >&2
    exit 2
}
[[ ${max_krylov_dimension} =~ ^[1-9][0-9]*$ ]] || {
    echo "GRID_TDSE_CHAIN_MAX_KRYLOV_DIMENSION must be a positive integer" >&2
    exit 2
}
for cutoff in 0 2 4 6 8 10; do
    variable="GRID_TDSE_CHAIN_TIME_STEP_L${cutoff}"
    value=${!variable:-}
    if [[ -n ${value} ]]; then
        [[ ${value} =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] \
            && awk -v candidate="${value}" 'BEGIN { exit !(candidate > 0) }' || {
                echo "${variable} must be positive" >&2
                exit 2
            }
    fi
    variable="GRID_TDSE_CHAIN_MAX_ITERATIONS_L${cutoff}"
    value=${!variable:-}
    if [[ -n ${value} && ! ${value} =~ ^[1-9][0-9]*$ ]]; then
        echo "${variable} must be a positive integer" >&2
        exit 2
    fi
done
for system in "${systems[@]}"; do
    case ${system} in
        H2+|HD+|D2+) ;;
        *) echo "Unsupported system: ${system}" >&2; exit 2 ;;
    esac
done

: "${PROJECT_ID:?Source grid_tdse_paths.sh before running this script}"
: "${PROJECT_ROOT:?Source grid_tdse_paths.sh before running this script}"
: "${SRC_ROOT:?Source grid_tdse_paths.sh before running this script}"
: "${ENV_PREFIX:?Source grid_tdse_paths.sh before running this script}"
: "${GRID_TDSE_MPI_MODULE:?Set GRID_TDSE_MPI_MODULE in grid_tdse_paths.sh}"

repository="${SRC_ROOT}/grid-tdse"
template_dir="${repository}/examples/Saga/ground_state_chains"
worker_source="${repository}/examples/mpi_non_bo_diatomic_ground_state_chain.py"
run_root="${PROJECT_ROOT}/jobs/grid-tdse/ground-state-chains/${run_name}"
output_root="${run_root}/output"
log_root="${run_root}/logs"
state_root="${run_root}/states"

if [[ -e ${run_root} ]]; then
    echo "Run directory already exists: ${run_root}" >&2
    exit 1
fi
mkdir -p "${output_root}" "${log_root}" "${state_root}"
cp "${template_dir}/job_ground_state_chain.sh" "${run_root}/"
cp "${template_dir}/submit_ground_state_chains.sh" "${run_root}/"
cp "${worker_source}" "${run_root}/"
chmod +x "${run_root}/job_ground_state_chain.sh"

job_script="${run_root}/job_ground_state_chain.sh"
worker="${run_root}/mpi_non_bo_diatomic_ground_state_chain.py"
manifest="${run_root}/jobs.csv"
printf '%s\n' \
    'job_id,system,electronic_points_per_element,l_max,ranks,process_grid,integrator,time_step,max_iterations,max_krylov_dimension,walltime,dependency,input_state,output_state,output_pattern,log_file' \
    > "${manifest}"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "grid_tdse_commit=$(git -C "${repository}" rev-parse HEAD)"
    echo "grid_lib_commit=$(git -C "${SRC_ROOT}/grid-lib" rev-parse HEAD)"
    echo "integrator_lib_commit=$(git -C "${SRC_ROOT}/integrator-lib" rev-parse HEAD)"
    echo "mpi_module=${GRID_TDSE_MPI_MODULE}"
    echo "python=$(command -v python)"
    echo "integrator=${integrator}"
    echo "default_time_step=${time_step}"
    echo "default_max_iterations=${max_iterations}"
    echo "max_krylov_dimension=${max_krylov_dimension}"
    echo "requested_max_l_max=${max_l_max:-grid-policy}"
    python -c 'from mpi4py import MPI; print("mpi_library=" + MPI.Get_library_version().rstrip("\x00"))'
    sha256sum "${worker}" "${job_script}" "${run_root}/submit_ground_state_chains.sh"
} > "${run_root}/provenance.txt"

ranks_for() {
    case $1 in
        0|2) echo 4 ;;
        4) echo 8 ;;
        6|8|10) echo 16 ;;
    esac
}

grid_for() {
    case $1 in
        0|2) echo 2x2 ;;
        4) echo 2x4 ;;
        6|8|10) echo 4x4 ;;
    esac
}

default_walltime() {
    local points=$1 lmax=$2
    if [[ ${points} == 11 ]]; then
        case ${lmax} in
            0) echo 00:30:00 ;;
            2) echo 01:00:00 ;;
            4) echo 02:00:00 ;;
            6) echo 08:00:00 ;;
            8) echo 1-00:00:00 ;;
            10) echo 2-00:00:00 ;;
        esac
    else
        case ${lmax} in
            0) echo 01:00:00 ;;
            2) echo 02:00:00 ;;
            4) echo 04:00:00 ;;
            6) echo 16:00:00 ;;
            8) echo 2-00:00:00 ;;
            10) echo 3-00:00:00 ;;
        esac
    fi
}

walltime_for() {
    local points=$1 lmax=$2 variable value
    variable="GRID_TDSE_CHAIN_TIME_E${points}_L${lmax}"
    value=${!variable:-}
    if [[ -n ${value} ]]; then
        echo "${value}"
    else
        default_walltime "${points}" "${lmax}"
    fi
}

time_step_for() {
    local lmax=$1 variable value
    variable="GRID_TDSE_CHAIN_TIME_STEP_L${lmax}"
    value=${!variable:-${time_step}}
    [[ ${value} =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] \
        && awk -v candidate="${value}" 'BEGIN { exit !(candidate > 0) }' || {
            echo "${variable} must be positive" >&2
            return 2
        }
    echo "${value}"
}

max_iterations_for() {
    local lmax=$1 variable value
    variable="GRID_TDSE_CHAIN_MAX_ITERATIONS_L${lmax}"
    value=${!variable:-${max_iterations}}
    [[ ${value} =~ ^[1-9][0-9]*$ ]] || {
        echo "${variable} must be a positive integer" >&2
        return 2
    }
    echo "${value}"
}

submit_chain() {
    local system=$1 points=$2 label symmetry cutoff_list endpoint candidate
    local previous_job="" previous_state="" lmax ranks grid walltime stem
    local stage_time_step stage_max_iterations
    local state_dir log_file output_state output_pattern dependency job_id
    local -a dependency_args site_args initial_args

    case ${system} in
        H2+) label=H2p; symmetry=sp ;;
        HD+) label=HDp; symmetry=s ;;
        D2+) label=D2p; symmetry=sp ;;
    esac
    if [[ -n ${max_l_max} ]]; then
        endpoint=${max_l_max}
    elif [[ ${points} == 21 && ${extend_21} == false ]]; then
        endpoint=6
    else
        endpoint=10
    fi
    cutoff_list=""
    for candidate in 0 2 4 6 8 10; do
        if (( candidate <= endpoint )); then
            cutoff_list+=" ${candidate}"
        fi
    done
    state_dir="${state_root}/${label}/e${points}"
    mkdir -p "${state_dir}" "${log_root}/${label}/e${points}"

    for lmax in ${cutoff_list}; do
        ranks=$(ranks_for "${lmax}")
        grid=$(grid_for "${lmax}")
        walltime=$(walltime_for "${points}" "${lmax}")
        stage_time_step=$(time_step_for "${lmax}")
        stage_max_iterations=$(max_iterations_for "${lmax}")
        stem="${label}_${symmetry}_e${points}_l${lmax}"
        output_state="${state_dir}/${stem}.npz"
        log_file="${log_root}/${label}/e${points}/${stem}.log"
        output_pattern="${output_root}/${stem}-%j.out"
        dependency=""
        dependency_args=()
        initial_args=()
        if [[ -n ${previous_job} ]]; then
            dependency="afterok:${previous_job}"
            dependency_args=(--dependency="${dependency}")
            initial_args=(--initial-state "${previous_state}")
        fi
        site_args=()
        [[ -z ${GRID_TDSE_CHAIN_PARTITION:-} ]] || site_args+=(--partition="${GRID_TDSE_CHAIN_PARTITION}")
        [[ -z ${GRID_TDSE_CHAIN_QOS:-} ]] || site_args+=(--qos="${GRID_TDSE_CHAIN_QOS}")

        job_id=$(sbatch --parsable \
            --account="${PROJECT_ID}" \
            --job-name="gs_${stem}" \
            --nodes=1 \
            --ntasks="${ranks}" \
            --cpus-per-task=1 \
            --mem-per-cpu="${GRID_TDSE_CHAIN_MEM_PER_CPU:-2G}" \
            --time="${walltime}" \
            --output="${output_pattern}" \
            --export="ALL,GRID_TDSE_PROJECT_ROOT=${PROJECT_ROOT}" \
            "${dependency_args[@]}" \
            "${site_args[@]}" \
            "${job_script}" \
            "${worker}" \
            --system "${system}" \
            --electronic-points-per-element "${points}" \
            --l-max "${lmax}" \
            --process-grid "${grid}" \
            --output-state "${output_state}" \
            --log-file "${log_file}" \
            --integrator "${integrator}" \
            --time-step "${stage_time_step}" \
            --max-iterations "${stage_max_iterations}" \
            --max-krylov-dimension "${max_krylov_dimension}" \
            --sample-interval "${GRID_TDSE_CHAIN_SAMPLE_INTERVAL:-10}" \
            "${initial_args[@]}")
        job_id=${job_id%%;*}
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${job_id}" "${system}" "${points}" "${lmax}" "${ranks}" \
            "${grid}" "${integrator}" "${stage_time_step}" "${stage_max_iterations}" \
            "${max_krylov_dimension}" "${walltime}" "${dependency}" "${previous_state}" \
            "${output_state}" "${output_pattern}" "${log_file}" >> "${manifest}"
        echo "Submitted ${stem}: ${job_id}${dependency:+ (${dependency})}; " \
            "dt=${stage_time_step}, max_iterations=${stage_max_iterations}"
        previous_job=${job_id}
        previous_state=${output_state}
    done
}

for system in "${systems[@]}"; do
    for points in "${electronic_points[@]}"; do
        submit_chain "${system}" "${points}"
    done
done

printf '%s\n' "${run_root}" > "${PROJECT_ROOT}/jobs/grid-tdse/ground-state-chains/latest_run.txt"
echo "Run root: ${run_root}"
echo "Manifest: ${manifest}"
