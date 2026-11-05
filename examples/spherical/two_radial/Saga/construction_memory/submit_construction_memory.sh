#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_construction_memory.sh --profile PROFILE [OPTIONS]

Profiles:
  production   Current 4e11 electronic and 4e21 internuclear grids.
  assessment   Assessment-sized grids: 32e11/16e11 at lmax=8,
               48e11/24e11 at lmax=12, and 64e11/32e11 at lmax=16.

Options:
  --l-max VALUES...    values to submit (default: 8 12 16)
  --ranks N            MPI ranks per construction (default: 1)
  --run-name NAME      run-directory name (default: profile plus timestamp)
  --concurrent         submit without afterany dependency serialization
  -h, --help           show this message

Site overrides are available through GRID_TDSE_MEMORY_{PARTITION,QOS,
NODELIST,EXCLUDE,CONSTRAINT,WALLTIME,MEM_PER_CPU}.
EOF
}

profile=
l_max_values=(8 12 16)
ranks=1
run_name=
concurrent=0

while (( $# > 0 )); do
    case $1 in
        --profile)
            profile=${2:?--profile requires a value}
            shift 2
            ;;
        --l-max)
            shift
            l_max_values=()
            while (( $# > 0 )) && [[ $1 != --* ]]; do
                l_max_values+=("$1")
                shift
            done
            ;;
        --ranks)
            ranks=${2:?--ranks requires a value}
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

case ${profile} in
    production|assessment) ;;
    *)
        echo "--profile must be production or assessment" >&2
        exit 2
        ;;
esac
if [[ ! ${ranks} =~ ^[1-9][0-9]*$ ]]; then
    echo "--ranks must be a positive integer" >&2
    exit 2
fi
if (( ${#l_max_values[@]} == 0 )); then
    echo "--l-max requires at least one value" >&2
    exit 2
fi
for l_max in "${l_max_values[@]}"; do
    if [[ ! ${l_max} =~ ^(8|12|16)$ ]]; then
        echo "This controlled series accepts l_max 8, 12, or 16: ${l_max}" >&2
        exit 2
    fi
done

: "${PROJECT_ID:?Source grid_tdse_paths.sh before submitting}"
: "${PROJECT_ROOT:?Source grid_tdse_paths.sh before submitting}"
: "${SRC_ROOT:?Source grid_tdse_paths.sh before submitting}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/../../../../.." && pwd)
timestamp=$(date +%Y%m%d-%H%M%S)
run_name=${run_name:-"construction-memory-${profile}-${timestamp}"}
if [[ ! ${run_name} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "run name may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
fi

base="${PROJECT_ROOT}/jobs/grid-tdse/construction-memory"
run_root="${base}/${run_name}"
if [[ -e ${run_root} ]]; then
    echo "Refusing to reuse existing run directory: ${run_root}" >&2
    exit 1
fi
mkdir -p "${run_root}/output" "${run_root}/results"

job_script="${run_root}/job_construction_memory.sh"
benchmark_script="${run_root}/non_bo_diatomic_construction_memory.py"
summary_script="${run_root}/summarize_non_bo_diatomic_construction_memory.py"
cp "${script_dir}/job_construction_memory.sh" "${job_script}"
cp "${repository_root}/benchmarks/non_bo_diatomic_construction_memory.py" \
    "${benchmark_script}"
cp "${repository_root}/benchmarks/summarize_non_bo_diatomic_construction_memory.py" \
    "${summary_script}"
chmod +x "${job_script}"

git -C "${repository_root}" rev-parse HEAD > "${run_root}/git_commit.txt"
python - <<'PY' > "${run_root}/login_gaunt_backend.txt"
import importlib.util
import sys
from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend

specification = importlib.util.find_spec("py3nj")
print("Python executable:", sys.executable)
print("py3nj available:", specification is not None)
print("py3nj path:", None if specification is None else specification.origin)
print("Gaunt backend:", get_gaunt_backend(resolve=True))
PY

manifest="${run_root}/jobs.csv"
echo "job_id,profile,l_max,mpi_processes,layout_ranks,process_grid,electronic_elements,electronic_points,internuclear_elements,internuclear_points,memory_per_rank,dependency,result_json" > "${manifest}"

walltime=${GRID_TDSE_MEMORY_WALLTIME:-00:30:00}
if [[ ${profile} == assessment ]]; then
    mem_per_cpu=${GRID_TDSE_MEMORY_MEM_PER_CPU:-14G}
else
    mem_per_cpu=${GRID_TDSE_MEMORY_MEM_PER_CPU:-4G}
fi
extra_sbatch=()
for option in PARTITION QOS NODELIST EXCLUDE CONSTRAINT; do
    variable="GRID_TDSE_MEMORY_${option}"
    value=${!variable:-}
    if [[ -n ${value} ]]; then
        extra_sbatch+=("--${option,,}=${value}")
    fi
done

export GRID_TDSE_PROJECT_ROOT="${PROJECT_ROOT}"
previous_job_id=
for l_max in "${l_max_values[@]}"; do
    if [[ ${profile} == production ]]; then
        electronic_elements=4
        electronic_points=11
        internuclear_elements=4
        internuclear_points=21
        if (( ranks == 1 )); then
            layout_ranks=4
            process_grid=2x2
        else
            layout_ranks=${ranks}
            process_grid=
        fi
    else
        electronic_elements=$((4 * l_max))
        electronic_points=11
        internuclear_elements=$((2 * l_max))
        internuclear_points=11
        if (( ranks == 1 )); then
            layout_ranks=$((2 * l_max * l_max))
            process_grid="$((2 * l_max))x${l_max}"
        else
            layout_ranks=${ranks}
            process_grid=
        fi
    fi

    layout_options=(--layout-ranks "${layout_ranks}")
    if [[ -n ${process_grid} ]]; then
        layout_options+=(--process-grid "${process_grid%x*}" "${process_grid#*x}")
    fi

    dependency_options=()
    dependency_record=none
    if (( ! concurrent )) && [[ -n ${previous_job_id} ]]; then
        dependency_options=(--dependency="afterany:${previous_job_id}")
        dependency_record="afterany:${previous_job_id}"
    fi
    job_name="mem_${profile}_l${l_max}_${ranks}r"
    result_json="${run_root}/results/${job_name}.json"
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
            --output="${run_root}/output/%x-%j.out" \
            "${dependency_options[@]}" \
            "${extra_sbatch[@]}" \
            "${job_script}" \
            "${benchmark_script}" \
            --profile-name "${profile}" \
            --l-max "${l_max}" \
            "${layout_options[@]}" \
            --electronic-elements "${electronic_elements}" \
            --electronic-rmax 16 \
            --electronic-points "${electronic_points}" \
            --internuclear-elements "${internuclear_elements}" \
            --internuclear-rmax 8 \
            --internuclear-points "${internuclear_points}" \
            --result-json "${result_json}"
    )
    job_id=${job_id%%;*}
    echo "${job_id},${profile},${l_max},${ranks},${layout_ranks},${process_grid:-auto},${electronic_elements},${electronic_points},${internuclear_elements},${internuclear_points},${mem_per_cpu},${dependency_record},${result_json}" >> "${manifest}"
    echo "Submitted ${job_name}: ${job_id} (${dependency_record})"
    previous_job_id=${job_id}
done

mkdir -p "${base}"
printf '%s\n' "${run_root}" > "${base}/latest_${profile}_run.txt"
echo "Run root: ${run_root}"
echo "Login-node backend:"
cat "${run_root}/login_gaunt_backend.txt"
