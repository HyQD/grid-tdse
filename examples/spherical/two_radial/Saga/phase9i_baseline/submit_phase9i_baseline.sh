#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_phase9i_baseline.sh [OPTIONS]

Options:
  --stage STAGE                all, hamiltonian, or dynamics (default: all)
  --repetitions N              repetitions per configuration (default: 3)
  --hamiltonian-ranks VALUES   rank counts (default: 1 2 4 16)
  --dynamics-ranks VALUES      rank counts (default: 1 2 4)
  --actions N                  timed Hamiltonian actions (default: 50)
  --warmup-actions N           warmup Hamiltonian actions (default: 2)
  --dynamics-final-time T      short SIA endpoint in a.u. (default: 20.0)
  --run-name NAME              run-directory name (default: timestamped)
  --concurrent                 allow independent benchmark jobs to overlap
  -h, --help                   show this message

The default serialized execution avoids benchmark interference. Dynamics
jobs always depend on successful creation of the shared Davidson ground state.

Site overrides are available through GRID_TDSE_PHASE9I_{PARTITION,QOS,
NODELIST,EXCLUDE,CONSTRAINT,WALLTIME,MEM_PER_CPU}.
EOF
}

stage=all
repetitions=3
hamiltonian_ranks=(1 2 4 16)
dynamics_ranks=(1 2 4)
actions=50
warmup_actions=2
dynamics_final_time=20.0
run_name=
concurrent=0

while (( $# > 0 )); do
    case $1 in
        --stage)
            stage=${2:?--stage requires a value}
            shift 2
            ;;
        --repetitions)
            repetitions=${2:?--repetitions requires a value}
            shift 2
            ;;
        --hamiltonian-ranks)
            shift
            hamiltonian_ranks=()
            while (( $# > 0 )) && [[ $1 != --* ]]; do
                hamiltonian_ranks+=("$1")
                shift
            done
            ;;
        --dynamics-ranks)
            shift
            dynamics_ranks=()
            while (( $# > 0 )) && [[ $1 != --* ]]; do
                dynamics_ranks+=("$1")
                shift
            done
            ;;
        --actions)
            actions=${2:?--actions requires a value}
            shift 2
            ;;
        --warmup-actions)
            warmup_actions=${2:?--warmup-actions requires a value}
            shift 2
            ;;
        --dynamics-final-time)
            dynamics_final_time=${2:?--dynamics-final-time requires a value}
            shift 2
            ;;
        --run-name)
            run_name=${2:?--run-name requires a value}
            shift 2
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
    all|hamiltonian|dynamics) ;;
    *)
        echo "--stage must be all, hamiltonian, or dynamics" >&2
        exit 2
        ;;
esac
for value in "${repetitions}" "${actions}"; do
    if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
        echo "repetitions and actions must be positive integers" >&2
        exit 2
    fi
done
if [[ ! ${warmup_actions} =~ ^[0-9]+$ ]]; then
    echo "warmup-actions must be a nonnegative integer" >&2
    exit 2
fi
if [[ ! ${dynamics_final_time} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v value="${dynamics_final_time}" 'BEGIN { exit !(value > 0.0) }'; then
    echo "dynamics-final-time must be positive" >&2
    exit 2
fi
if [[ ${stage} != dynamics && ${#hamiltonian_ranks[@]} -eq 0 ]]; then
    echo "--hamiltonian-ranks requires at least one value" >&2
    exit 2
fi
if [[ ${stage} != hamiltonian && ${#dynamics_ranks[@]} -eq 0 ]]; then
    echo "--dynamics-ranks requires at least one value" >&2
    exit 2
fi

hamiltonian_grid() {
    case $1 in
        1) echo 1x1 ;;
        2) echo 1x2 ;;
        4) echo 2x2 ;;
        8) echo 2x4 ;;
        16) echo 4x4 ;;
        *) return 1 ;;
    esac
}

dynamics_grid() {
    case $1 in
        1) echo 1x1 ;;
        2) echo 1x2 ;;
        4) echo 2x2 ;;
        10) echo 5x2 ;;
        *) return 1 ;;
    esac
}

if [[ ${stage} != dynamics ]]; then
    for ranks in "${hamiltonian_ranks[@]}"; do
        if ! hamiltonian_grid "${ranks}" >/dev/null; then
            echo "unsupported Hamiltonian rank count: ${ranks}" >&2
            exit 2
        fi
    done
fi
if [[ ${stage} != hamiltonian ]]; then
    for ranks in "${dynamics_ranks[@]}"; do
        if ! dynamics_grid "${ranks}" >/dev/null; then
            echo "unsupported dynamics rank count: ${ranks}" >&2
            exit 2
        fi
    done
fi

: "${PROJECT_ID:?Source grid_tdse_paths.sh before submitting}"
: "${PROJECT_ROOT:?Source grid_tdse_paths.sh before submitting}"
: "${SRC_ROOT:?Source grid_tdse_paths.sh before submitting}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/../../../../.." && pwd)
timestamp=$(date +%Y%m%d-%H%M%S)
run_name=${run_name:-"phase9i-baseline-${timestamp}"}
if [[ ! ${run_name} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "run name may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
fi
tracked_status=$(git -C "${repository_root}" status --porcelain --untracked-files=no)
if [[ -n ${tracked_status} ]]; then
    echo "Refusing to benchmark a checkout with tracked modifications:" >&2
    printf '%s\n' "${tracked_status}" >&2
    exit 1
fi

base="${PROJECT_ROOT}/jobs/grid-tdse/phase9i-baseline"
run_root="${base}/${run_name}"
if [[ -e ${run_root} ]]; then
    echo "Refusing to reuse existing run directory: ${run_root}" >&2
    exit 1
fi
mkdir -p \
    "${run_root}/output" \
    "${run_root}/hamiltonian/results" \
    "${run_root}/dynamics/runs" \
    "${run_root}/dynamics/states"

job_script="${run_root}/job_phase9i_baseline.sh"
hamiltonian_worker="${run_root}/non_bo_diatomic_2d_execution.py"
dynamics_worker="${run_root}/mpi_non_bo_diatomic_2d_end_to_end.py"
dynamics_common="${run_root}/non_bo_diatomic_end_to_end_common.py"
summary_script="${run_root}/summarize_non_bo_diatomic_phase9i_baseline.py"
cp "${script_dir}/job_phase9i_baseline.sh" "${job_script}"
cp "${repository_root}/benchmarks/non_bo_diatomic_2d_execution.py" \
    "${hamiltonian_worker}"
cp "${repository_root}/examples/spherical/two_radial/mpi_non_bo_diatomic_2d_end_to_end.py" \
    "${dynamics_worker}"
cp "${repository_root}/examples/spherical/two_radial/non_bo_diatomic_end_to_end_common.py" \
    "${dynamics_common}"
cp "${repository_root}/benchmarks/summarize_non_bo_diatomic_phase9i_baseline.py" \
    "${summary_script}"
chmod +x "${job_script}"

expected_commit=$(git -C "${repository_root}" rev-parse HEAD)
printf '%s\n' "${expected_commit}" > "${run_root}/git_commit.txt"
printf '%s' "${tracked_status}" > "${run_root}/git_status.txt"
sha256sum \
    "${job_script}" \
    "${hamiltonian_worker}" \
    "${dynamics_worker}" \
    "${dynamics_common}" \
    "${summary_script}" \
    > "${run_root}/input_checksums.sha256"

python - <<'PY' > "${run_root}/login_environment.txt"
import importlib.util
import sys

import grid_tdse
import integrator_lib
import numba
import numpy
from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend
from mpi4py import MPI

specification = importlib.util.find_spec("py3nj")
print("Python executable:", sys.executable)
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

manifest="${run_root}/jobs.csv"
echo "job_id,workload,repetition,l_max,mpi_ranks,process_grid,actions,final_time,dependency,result,output_pattern" > "${manifest}"

walltime=${GRID_TDSE_PHASE9I_WALLTIME:-00:30:00}
mem_per_cpu=${GRID_TDSE_PHASE9I_MEM_PER_CPU:-4G}
extra_sbatch=()
for option in PARTITION QOS NODELIST EXCLUDE CONSTRAINT; do
    variable="GRID_TDSE_PHASE9I_${option}"
    value=${!variable:-}
    if [[ -n ${value} ]]; then
        extra_sbatch+=("--${option,,}=${value}")
    fi
done

export GRID_TDSE_PROJECT_ROOT="${PROJECT_ROOT}"
export GRID_TDSE_EXPECTED_COMMIT="${expected_commit}"
previous_job_id=

if [[ ${stage} != dynamics ]]; then
    for ranks in "${hamiltonian_ranks[@]}"; do
        grid=$(hamiltonian_grid "${ranks}")
        for (( repetition = 1; repetition <= repetitions; repetition++ )); do
            dependency_options=()
            dependency_record=none
            if (( ! concurrent )) && [[ -n ${previous_job_id} ]]; then
                dependency_options=(--dependency="afterany:${previous_job_id}")
                dependency_record="afterany:${previous_job_id}"
            fi
            job_name="p9i_ham_l8_q${ranks}_r${repetition}"
            result_json="${run_root}/hamiltonian/results/${job_name}.json"
            output_pattern="${run_root}/output/%x-%j.out"
            job_id=$(
                sbatch \
                    --parsable \
                    --account="${PROJECT_ID}" \
                    --job-name="${job_name}" \
                    --nodes=1 \
                    --ntasks="${ranks}" \
                    --cpus-per-task=1 \
                    --hint=nomultithread \
                    --mem-per-cpu="${mem_per_cpu}" \
                    --time="${walltime}" \
                    --output="${output_pattern}" \
                    "${dependency_options[@]}" \
                    "${extra_sbatch[@]}" \
                    "${job_script}" \
                    "${hamiltonian_worker}" \
                    --system H2+ \
                    --l-max 8 \
                    --process-grid "${grid%x*}" "${grid#*x}" \
                    --electronic-nodes 0 4 8 12 16 \
                    --electronic-points 11 \
                    --internuclear-nodes 0 2 4 6 8 \
                    --internuclear-points 21 \
                    --tile-size 2048 \
                    --warmup-actions "${warmup_actions}" \
                    --actions "${actions}" \
                    --result-json "${result_json}"
            )
            job_id=${job_id%%;*}
            echo "${job_id},hamiltonian,${repetition},8,${ranks},${grid},${actions},none,${dependency_record},${result_json},${output_pattern}" >> "${manifest}"
            echo "Submitted ${job_name}: ${job_id} (${dependency_record})"
            previous_job_id=${job_id}
        done
    done
fi

if [[ ${stage} != hamiltonian ]]; then
    state_path="${run_root}/dynamics/states/H2p_phase9i_l2_codata2018.npz"
    seed_directory="${run_root}/dynamics/seed"
    seed_dependency_options=()
    seed_dependency_record=none
    if (( ! concurrent )) && [[ -n ${previous_job_id} ]]; then
        seed_dependency_options=(--dependency="afterany:${previous_job_id}")
        seed_dependency_record="afterany:${previous_job_id}"
    fi
    seed_name=p9i_seed_l2
    seed_output_pattern="${run_root}/output/%x-%j.out"
    seed_id=$(
        sbatch \
            --parsable \
            --account="${PROJECT_ID}" \
            --job-name="${seed_name}" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --hint=nomultithread \
            --mem-per-cpu="${mem_per_cpu}" \
            --time="${walltime}" \
            --output="${seed_output_pattern}" \
            "${seed_dependency_options[@]}" \
            "${extra_sbatch[@]}" \
            "${job_script}" \
            "${dynamics_worker}" \
            --process-grid 1x1 \
            --final-time 0.25 \
            --ground-state-path "${state_path}" \
            --initial-state-policy compute-if-missing \
            --run-directory "${seed_directory}" \
            --no-wavefunction-snapshots \
            --no-checkpoints \
            --no-show-figure \
            --progress-interval 1
    )
    seed_id=${seed_id%%;*}
    echo "${seed_id},seed,0,2,1,1x1,none,0.25,${seed_dependency_record},${seed_directory}/summary.json,${seed_output_pattern}" >> "${manifest}"
    echo "Submitted ${seed_name}: ${seed_id} (${seed_dependency_record})"
    previous_job_id=${seed_id}
    first_dynamics=1

    for ranks in "${dynamics_ranks[@]}"; do
        grid=$(dynamics_grid "${ranks}")
        for (( repetition = 1; repetition <= repetitions; repetition++ )); do
            if (( concurrent )); then
                dependency_record="afterok:${seed_id}"
            elif (( first_dynamics )); then
                dependency_record="afterok:${seed_id}"
            else
                dependency_record="afterany:${previous_job_id}"
            fi
            dependency_options=(--dependency="${dependency_record}")
            job_name="p9i_sia_l2_q${ranks}_r${repetition}"
            result_directory="${run_root}/dynamics/runs/${job_name}"
            result_json="${result_directory}/summary.json"
            output_pattern="${run_root}/output/%x-%j.out"
            job_id=$(
                sbatch \
                    --parsable \
                    --account="${PROJECT_ID}" \
                    --job-name="${job_name}" \
                    --nodes=1 \
                    --ntasks="${ranks}" \
                    --cpus-per-task=1 \
                    --hint=nomultithread \
                    --mem-per-cpu="${mem_per_cpu}" \
                    --time="${walltime}" \
                    --output="${output_pattern}" \
                    "${dependency_options[@]}" \
                    "${extra_sbatch[@]}" \
                    "${job_script}" \
                    "${dynamics_worker}" \
                    --process-grid "${grid}" \
                    --final-time "${dynamics_final_time}" \
                    --ground-state-path "${state_path}" \
                    --initial-state-policy require \
                    --run-directory "${result_directory}" \
                    --no-wavefunction-snapshots \
                    --no-checkpoints \
                    --no-show-figure \
                    --progress-interval 20
            )
            job_id=${job_id%%;*}
            echo "${job_id},dynamics,${repetition},2,${ranks},${grid},none,${dynamics_final_time},${dependency_record},${result_json},${output_pattern}" >> "${manifest}"
            echo "Submitted ${job_name}: ${job_id} (${dependency_record})"
            previous_job_id=${job_id}
            first_dynamics=0
        done
    done
fi

mkdir -p "${base}"
printf '%s\n' "${run_root}" > "${base}/latest_run.txt"
echo "Run root: ${run_root}"
echo "Manifest: ${manifest}"
echo "Summarize after completion with:"
echo "  python ${summary_script} ${run_root}"
