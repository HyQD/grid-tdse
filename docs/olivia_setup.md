# Setting up grid-tdse on Olivia

This guide describes a reproducible CPU installation of `grid-tdse`, its HyQD
dependencies, the optional `py3nj` backend, and MPI support on Olivia. It also
describes how to update and roll back the installation.

The guide reflects the repository dependency bounds and Sigma2 documentation
available on 2026-11-09. Steps 1--14 were executed successfully on Olivia on
that date with the pinned CPU modules below: the containerized deployment
reported no broken requirements, all 644 tests passed, `py3nj` was selected as
the Gaunt backend, `mpi4py` linked to Open MPI 5.0.8, and two MPI ranks launched
successfully. The grid-tdse serial/MPI numerical-equivalence gate also passed
with differences far below its `1e-8` threshold. Step 15 passed on 2026-11-10
for 1, 2, 4, 8, and 16 ranks, with a best single-node speedup of `4.805` and
serial differences no larger than `1.11e-14`. Multi-node qualification remains
required before scientific production calculations.

Olivia differs from Saga in two important ways:

- Sigma2 does not permit native user Conda or pip environments on Olivia.
  Python environments must use the site-supported HPC-container-wrapper or an
  Apptainer container.
- Olivia uses Open MPI over the HPE Slingshot interconnect. Custom and
  containerized MPI environments must use the site MPI consistently and must
  be tested on compute nodes.

Consult the current official documentation before installation:

- [Olivia overview](https://documentation.sigma2.no/hpc_machines/olivia.html)
- [Installing and using software on Olivia](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html)
- [Open MPI on Olivia](https://documentation.sigma2.no/hpc_machines/olivia/openmpi.html)
- [Olivia job types](https://documentation.sigma2.no/jobs/job_types/olivia_job_types.html)
- [Olivia job scripts](https://documentation.sigma2.no/jobs/job_scripts/olivia_job_scripts.html)
- [Storage areas](https://documentation.sigma2.no/files_storage/clusters.html)

## 1. Scope and placeholders

This setup targets Olivia's x86-64 AMD CPU nodes. Do not use the ARM64
Grace-Hopper nodes: `grid-tdse` currently has no GPU kernels, and the
accelerator nodes would require a separate ARM64 environment.

Replace these placeholders before running commands:

- `<project-id>`: the Sigma2 project ID, for example `nn1234k`;
- `<username>`: the Olivia username;
- `<deployment-id>`: a unique immutable label. For the initial deployment in
  this guide, use `<date>-cpu-ompi508-initial`, where `<date>` has the
  `YYYYMMDD` format.

Text in angle brackets is a placeholder. Do not type the angle brackets.

The module catalogue inspected on 2026-11-09 provided these CPU modules:

```text
OpenMPI/5.0.8-GCC-14.3.0
hpc-container-wrapper/0.5.1-CPU
```

The Open MPI module was marked as the default, and the selected container
wrapper is the newest CPU-specific version. They are pinned below so that the
same toolchain is used for environment construction, the `mpi4py` build, and
subsequent jobs. If Olivia removes either module, stop and qualify a new
deployment with the replacement modules rather than silently changing an
existing deployment.

## 2. Installation layout

Use `/cluster/projects` for software and immutable deployment records,
`/cluster/work/projects` for jobs and results, and Olivia's `$LOCALSCRATCH`
for temporary per-job caches.

This guide uses:

```text
/cluster/projects/<project-id>/<username>/grid-tdse/
├── current -> environments/<deployment-id>
├── environments/
│   └── <deployment-id>/
├── manifests/
│   └── <deployment-id>/
│       ├── environment.yml
│       ├── hyqd-requirements.txt
│       ├── install-mpi.sh
│       └── source-commits.txt
├── olivia_grid_tdse_paths.sh
└── src/
    └── HyQD/
        ├── electromagnetic-fields/
        ├── grid-lib/
        ├── grid-tdse/
        ├── integrator-lib/
        └── matter_field_params/

/cluster/work/projects/<project-id>/<username>/grid-tdse/
├── jobs/
├── qualification/
├── results/
└── runs/
```

Do not install under `$HOME`. Olivia home directories have limited capacity
and file counts, while Python environments contain many small files.

Create the directories:

```bash
project_root="/cluster/projects/<project-id>/<username>/grid-tdse"
work_root="/cluster/work/projects/<project-id>/<username>/grid-tdse"

mkdir -p "${project_root}/environments"
mkdir -p "${project_root}/manifests"
mkdir -p "${project_root}/src/HyQD"
mkdir -p "${work_root}/jobs"
mkdir -p "${work_root}/qualification"
mkdir -p "${work_root}/results"
mkdir -p "${work_root}/runs"
```

## 3. Create the reusable paths file

Open `${project_root}/olivia_grid_tdse_paths.sh` in a text editor, for example:

```bash
nano "${project_root}/olivia_grid_tdse_paths.sh"
```

Alternatively, use `vi` or `vim`:

```bash
vim "${project_root}/olivia_grid_tdse_paths.sh"
```

Paste the following content, replace the project and username placeholders,
and save the file:

```bash
#!/bin/bash

export GRID_TDSE_PROJECT_ID="<project-id>"
export GRID_TDSE_ROOT="/cluster/projects/<project-id>/<username>/grid-tdse"
export GRID_TDSE_WORK_ROOT="/cluster/work/projects/<project-id>/<username>/grid-tdse"
export GRID_TDSE_PATHS_FILE="${GRID_TDSE_ROOT}/olivia_grid_tdse_paths.sh"
export GRID_TDSE_SRC_ROOT="${GRID_TDSE_ROOT}/src/HyQD"
export GRID_TDSE_ENV_ROOT="${GRID_TDSE_ROOT}/environments"
export GRID_TDSE_CURRENT="${GRID_TDSE_ROOT}/current"
export GRID_TDSE_MPI_STACK="NRIS/CPU"
export GRID_TDSE_MPI_MODULE="OpenMPI/5.0.8-GCC-14.3.0"
export GRID_TDSE_CONTAINER_WRAPPER_MODULE="hpc-container-wrapper/0.5.1-CPU"
```

Load it with:

```bash
source "${project_root}/olivia_grid_tdse_paths.sh"
```

Validate the file immediately. This catches an omitted substitution before a
literal angle bracket is interpreted by the shell as redirection:

```bash
bash -n "${project_root}/olivia_grid_tdse_paths.sh"
source "${project_root}/olivia_grid_tdse_paths.sh"

case "${GRID_TDSE_PROJECT_ID}:${GRID_TDSE_ROOT}:${GRID_TDSE_WORK_ROOT}" in
    *'<'*|*'>'*)
        echo "FAIL: the paths file still contains a placeholder" >&2
        false
        ;;
    *)
        echo "Paths file contains concrete values"
        ;;
esac

printf 'Project:    %s\n' "${GRID_TDSE_PROJECT_ID}"
printf 'Software:   %s\n' "${GRID_TDSE_ROOT}"
printf 'Work:       %s\n' "${GRID_TDSE_WORK_ROOT}"
printf 'Paths file: %s\n' "${GRID_TDSE_PATHS_FILE}"

test -d "${GRID_TDSE_SRC_ROOT}"
test -d "${GRID_TDSE_ENV_ROOT}"
test -d "${GRID_TDSE_WORK_ROOT}/jobs"
```

The file defines paths and module names only. It does not load modules or
select a deployment. It does not need to be executable. Use `source`, not
`bash olivia_grid_tdse_paths.sh`, so that the exported variables remain in the
current shell.

The module names are deliberately versioned. Do not replace them with the
unversioned names `OpenMPI` or `hpc-container-wrapper`.

## 4. Perform a login-node module-catalogue preflight

Do not load `NRIS/CPU` on a login node. Olivia reports a warning when a compute
stack is loaded on a login host such as `uan01` or `uan02`.

The purpose of this step is only to:

1. remove modules inherited from an earlier shell or workflow;
2. select the site-supported stack for commands run on a login node; and
3. confirm, without loading them, that Olivia's module catalogue still
   advertises the two CPU modules pinned by this guide.

`module spider` searches the catalogue and shows module prerequisites; it does
not load the queried module. Run this preflight on the login node:

```bash
module --force purge
module load NRIS/Login
module --terse list
PAGER=cat LMOD_PAGER=cat module spider hpc-container-wrapper/0.5.1-CPU
PAGER=cat LMOD_PAGER=cat module spider OpenMPI/5.0.8-GCC-14.3.0
```

The temporary pager settings request direct terminal output. If the site
configuration nevertheless opens `less`, press `q` to leave the pager before
entering the next shell command. Inside `less`, pressing `s` starts the
save-output command; text typed afterward is interpreted as a filename rather
than as a shell command.

The two `module spider` commands should identify the requested modules and
their prerequisite stacks. If either exact version is absent, stop and inspect
the general catalogue with `module spider OpenMPI` and
`module spider hpc-container-wrapper`; the guide's pinned toolchain then needs
to be revised and qualified as a new deployment. The `GPU` wrapper variants
target a different architecture and must not be used for this deployment.

At this point, check only the recorded strings. Do not attempt to load
`NRIS/CPU`, Open MPI, or the container wrapper on the login node:

```bash
source "${GRID_TDSE_ROOT}/olivia_grid_tdse_paths.sh"
printf 'Open MPI module: %s\n' "${GRID_TDSE_MPI_MODULE}"
printf 'Container wrapper module: %s\n' \
    "${GRID_TDSE_CONTAINER_WRAPPER_MODULE}"

case "${GRID_TDSE_MPI_MODULE}:${GRID_TDSE_CONTAINER_WRAPPER_MODULE}" in
    *'<'*|*'>'*)
        echo "ERROR: a module value still contains a placeholder" >&2
        ;;
    *)
        echo "Recorded module values are concrete"
        ;;
esac
```

The actual module load and compiler checks are performed inside a CPU
compute-node allocation in Step 9.

Choose the deployment ID and create its manifest directory while still on the
login node:

```bash
export deployment_id="$(date +%Y%m%d)-cpu-ompi508-initial"
manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"

test ! -e "${manifest_root}"
mkdir -p "${manifest_root}"
printf '%s\n' "${deployment_id}" \
    > "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt"

cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt"
```

The naming convention is `<date>-<target>-<MPI>-<purpose>`, with the date in
`YYYYMMDD` format. For example, a later dependency refresh using the same MPI
toolchain could be named `20261115-cpu-ompi508-update01`. Never reuse an
existing deployment ID or use mutable names such as `latest` or `current` as
the deployment ID. The command above inserts the current date automatically
and writes the exact value to a persistent pointer. Later shells reconstruct
the variable from that file instead of relying on an exported shell variable
surviving a logout or allocation transition. Do not evaluate the date command
again for the same deployment.

Repository management and manifest editing may be performed on the login
node. Compilation and container construction move to a CPU compute node in
Step 9.

## 5. Configure network access for the container build

The current
[Sigma2 instructions for creating a Python or Conda environment on Olivia](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html#creating-a-new-python-or-conda-environment)
use these proxies for internet access from Olivia compute nodes:

```bash
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
```

Verify the current values in the official software documentation before use.
Do not put credentials or access tokens in the environment manifest. These
exports are needed in the compute shell used for the container build; they do
not require loading `NRIS/CPU` on the login node.

## 6. Clone the HyQD repositories

Clone all five source repositories over HTTPS. The loop is idempotent: an
existing checkout is fast-forwarded, while an absent checkout is cloned.
This works for public repositories without configuring or copying a GitHub
SSH key onto Olivia:

```bash
cd "${GRID_TDSE_SRC_ROOT}"

repositories=(
    grid-lib
    integrator-lib
    electromagnetic-fields
    matter_field_params
    grid-tdse
)

for repository in "${repositories[@]}"
do
    repository_root="${GRID_TDSE_SRC_ROOT}/${repository}"

    if [[ -d "${repository_root}/.git" ]]
    then
        git -C "${repository_root}" switch main
        git -C "${repository_root}" pull --ff-only origin main
    elif [[ -e "${repository_root}" ]]
    then
        printf 'FAIL: non-Git path already exists: %s\n' \
            "${repository_root}" >&2
        false
    else
        git clone \
            "https://github.com/HyQD/${repository}.git" \
            "${repository_root}"
    fi
done
```

An SSH clone that fails with `Permission denied (publickey)` means the Olivia
account has no GitHub-authorized SSH key or forwarded SSH agent. It does not
indicate a problem with a public repository. Use the HTTPS URLs above unless
SSH access is intentionally configured. Never copy a private key to Olivia
merely to clone these public repositories.

Do not build from dirty worktrees. Check all five repositories explicitly:

```bash
source_status=0

for repository in "${repositories[@]}"
do
    status=$(git -C "${GRID_TDSE_SRC_ROOT}/${repository}" status --short)

    if [[ -n "${status}" ]]
    then
        printf '\nDIRTY: %s\n%s\n' "${repository}" "${status}"
        source_status=1
    else
        printf 'CLEAN: %s\n' "${repository}"
    fi
done

if (( source_status != 0 ))
then
    echo "Stop: at least one source worktree is dirty" >&2
    false
fi
```

Do not discard local work merely to make this check pass. Commit or otherwise
preserve it in the corresponding repository first.

## 7. Capture exact source revisions

Record the commits used by this deployment:

```bash
deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"

grid_lib_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/grid-lib" rev-parse HEAD)
integrator_lib_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/integrator-lib" rev-parse HEAD)
fields_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/electromagnetic-fields" rev-parse HEAD)
matter_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/matter_field_params" rev-parse HEAD)
grid_tdse_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/grid-tdse" rev-parse HEAD)

printf '%s\n' \
    "grid-lib ${grid_lib_commit}" \
    "integrator-lib ${integrator_lib_commit}" \
    "electromagnetic-fields ${fields_commit}" \
    "matter_field_params ${matter_commit}" \
    "grid-tdse ${grid_tdse_commit}" \
    > "${manifest_root}/source-commits.txt"

cat "${manifest_root}/source-commits.txt"

while read -r repository commit extra
do
    test -z "${extra:-}"
    [[ "${commit}" =~ ^[0-9a-f]{40}$ ]]
    test "$(git -C "${GRID_TDSE_SRC_ROOT}/${repository}" branch --show-current)" = main
done < "${manifest_root}/source-commits.txt"
```

The commit manifest is necessary because several HyQD packages currently use
development versions that do not change for every source update.

## 8. Create the numerical environment manifest

Copy the version-controlled Olivia environment template into the immutable
manifest:

```bash
environment_template="${GRID_TDSE_SRC_ROOT}/grid-tdse/examples/spherical/two_radial/Olivia/qualification/environment.yml"

test -s "${environment_template}"
cp "${environment_template}" "${manifest_root}/environment.yml"
test -s "${manifest_root}/environment.yml"

if grep -n -E '<[^>]+>' "${manifest_root}/environment.yml"
then
    echo "FAIL: environment manifest contains a placeholder" >&2
    false
fi

sed -n '1,240p' "${manifest_root}/environment.yml"
```

The tracked template currently contains:

```yaml
channels:
  - conda-forge
dependencies:
  - python=3.14
  - pip
  - setuptools>=69
  - wheel
  - numpy=2.5.2
  - scipy=1.18.0
  - numba=0.67.0
  - opt_einsum=3.4.*
  - sympy=1.14.*
  - packaging
  - tqdm
  - matplotlib
  - periodictable=2.1.*
  - pytest=8.*
  - pytest-cov>=5
  - threadpoolctl>=3.5
  - meson
  - meson-python
  - ninja
  - gfortran_linux-64
```

These pins match the current grid-tdse dependency bounds. Re-read the five
`pyproject.toml` files before a later deployment and update the manifest when
their bounds change. Do not install a Conda Open MPI package.

Create `${manifest_root}/hyqd-requirements.txt` from the recorded commits:

```bash
grid_lib_commit=$(awk '$1 == "grid-lib" {print $2}' "${manifest_root}/source-commits.txt")
integrator_lib_commit=$(awk '$1 == "integrator-lib" {print $2}' "${manifest_root}/source-commits.txt")
fields_commit=$(awk '$1 == "electromagnetic-fields" {print $2}' "${manifest_root}/source-commits.txt")
matter_commit=$(awk '$1 == "matter_field_params" {print $2}' "${manifest_root}/source-commits.txt")
grid_tdse_commit=$(awk '$1 == "grid-tdse" {print $2}' "${manifest_root}/source-commits.txt")

printf '%s\n' \
    "HyQD-grid-lib @ git+https://github.com/HyQD/grid-lib.git@${grid_lib_commit}" \
    "HyQD-integrator-lib @ git+https://github.com/HyQD/integrator-lib.git@${integrator_lib_commit}" \
    "HyQD-electromagnetic-fields @ git+https://github.com/HyQD/electromagnetic-fields.git@${fields_commit}" \
    "HyQD-matter-field-params @ git+https://github.com/HyQD/matter_field_params.git@${matter_commit}" \
    "HyQD-grid-tdse @ git+https://github.com/HyQD/grid-tdse.git@${grid_tdse_commit}" \
    > "${manifest_root}/hyqd-requirements.txt"

test "$(wc -l < "${manifest_root}/hyqd-requirements.txt")" -eq 5
cat "${manifest_root}/hyqd-requirements.txt"
```

The HTTPS URLs avoid depending on an SSH agent inside the container build.
The repositories must be publicly readable, or the packages must instead be
built as wheels and supplied from a project-local wheelhouse. Never put a
GitHub token directly in this file.

## 9. Build a new immutable deployment

Perform container builds and compilation in a compute-node allocation, not on
a login node. From the login node, request a suitable initial allocation:

```bash
salloc \
    --account="${GRID_TDSE_PROJECT_ID}" \
    --partition=small \
    --qos=devel \
    --time=02:00:00 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem-per-cpu=4G
```

On Olivia, `salloc` starts an interactive shell on the allocated compute node;
no additional `srun --pty` command is needed. This follows Sigma2's current
[interactive-job instructions](https://documentation.sigma2.no/jobs/interactive_jobs.html#requesting-an-interactive-job).
The new shell may initially warn that it inherited `NRIS/Login` on a CPU node.
This is harmless; the `module --force purge` command below removes the
inherited login stack before `NRIS/CPU` is loaded.

The requested two hours are an upper bound for completing the initial
container construction and compilation through Steps 9--12, not the expected
duration of the hostname check. Exiting the interactive allocation releases it
before that limit. Two hours is the current maximum for Olivia's
[`devel` QoS](https://documentation.sigma2.no/jobs/job_types/olivia_job_types.html#devel).

If you only want to test compute-node access and verify the pinned modules,
request a separate ten-minute preflight allocation instead:

```bash
salloc \
    --account="${GRID_TDSE_PROJECT_ID}" \
    --partition=small \
    --qos=devel \
    --time=00:10:00 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem-per-cpu=2G
```

Leave that preflight shell and allocation after the checks, then request the
longer four-core allocation before constructing the deployment.

Confirm that the new shell is running on a compute host. Its short hostname
must not start with `uan`; if it does, `salloc` did not place the shell on the
allocated compute node and you must not continue with the build. Run this
check immediately after `salloc` reports that the allocated node is ready:

```bash
compute_host="$(hostname -s)"

printf 'Hostname:       %s\n' "${compute_host}"
printf 'Slurm job ID:   %s\n' "${SLURM_JOB_ID:-not-set}"
printf 'Allocated nodes: %s\n' "${SLURM_JOB_NODELIST:-not-set}"

if [[ "${compute_host}" == uan* ]]
then
    echo "FAIL: this is an Olivia login node; do not continue"
elif [[ -z "${SLURM_JOB_ID:-}" ]]
then
    echo "FAIL: this shell is not associated with a Slurm allocation"
else
    echo "PASS: running in a Slurm allocation on a compute host"
fi
```

Continue only after the command prints `PASS`. Then load the pinned CPU
software stack:

```bash
: "${GRID_TDSE_PATHS_FILE:?The paths-file variable was not inherited}"
source "${GRID_TDSE_PATHS_FILE}"

module --force purge
module load "${GRID_TDSE_MPI_STACK}"
module load "${GRID_TDSE_CONTAINER_WRAPPER_MODULE}"
module load "${GRID_TDSE_MPI_MODULE}"

module --terse list
command -v conda-containerize
command -v mpicc
mpicc --showme:version
```

The loaded-module list must contain
`hpc-container-wrapper/0.5.1-CPU` and
`OpenMPI/5.0.8-GCC-14.3.0`. If either pinned module is unavailable, stop the
build. Reinspect the current Olivia modules, select replacements, and use a
new deployment ID; do not mutate the toolchain of an existing deployment.

Recreate the shell variables used in the earlier steps and configure compute-
node network access:

```bash
deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"
deployment_prefix="${GRID_TDSE_ENV_ROOT}/${deployment_id}"

test -s "${manifest_root}/environment.yml"
test -s "${manifest_root}/hyqd-requirements.txt"
printf 'Building deployment: %s\n' "${deployment_id}"

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

: "${LOCALSCRATCH:?Olivia did not define LOCALSCRATCH for this allocation}"
export APPTAINER_CACHEDIR="${LOCALSCRATCH}/apptainer-cache"
export APPTAINER_TMPDIR="${LOCALSCRATCH}/apptainer-tmp"
export PIP_CACHE_DIR="${LOCALSCRATCH}/pip-cache"
export XDG_CACHE_HOME="${LOCALSCRATCH}/xdg-cache"
mkdir -p \
    "${APPTAINER_CACHEDIR}" \
    "${APPTAINER_TMPDIR}" \
    "${PIP_CACHE_DIR}" \
    "${XDG_CACHE_HOME}"
```

Record the exact compute-node module and MPI compiler configuration:

```bash
module list 2> "${manifest_root}/modules.txt"
mpicc --showme:version > "${manifest_root}/mpicc-version.txt"
mpicc --showme:command > "${manifest_root}/mpicc-command.txt"
```

The Open MPI module loaded while building `mpi4py` must also be loaded in
every MPI job using that deployment.

Build a new prefix rather than updating the production deployment in place:

```bash
test ! -e "${deployment_prefix}"

set -o pipefail
conda-containerize new \
    --mamba \
    -r "${manifest_root}/hyqd-requirements.txt" \
    --prefix "${deployment_prefix}" \
    "${manifest_root}/environment.yml" 2>&1 |
tee "${manifest_root}/container-build.log"
build_status=${PIPESTATUS[0]}
set +o pipefail
test "${build_status}" -eq 0
```

Do not add `--slim` initially. The non-slim environment makes the host
software stack available and is the safer starting point for integrating the
site Open MPI. After the deployment is qualified, a slim variant may be tested
as a separate deployment.

Select the new environment in the current shell:

```bash
export PATH="${deployment_prefix}/bin:${PATH}"
hash -r

command -v python
python --version
python -m pip --version
```

At this stage the serial installation should work, but `mpi4py` and the
optional `py3nj` acceleration have not yet been qualified.

## 10. Validate the serial installation

Check dependency metadata and imports:

```bash
test "$(command -v python)" = "${deployment_prefix}/bin/python"
python -m pip check

python - <<'PY'
import electromagnetic_fields
import grid_lib
import grid_tdse
import integrator_lib
import matter_field_params
import numba
import numpy
import scipy

print("grid-tdse:", grid_tdse.__file__)
print("grid-lib:", grid_lib.__file__)
print("integrator-lib:", integrator_lib.__file__)
print("electromagnetic-fields:", electromagnetic_fields.__file__)
print("matter-field-params:", matter_field_params.__file__)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Numba:", numba.__version__)

assert numpy.__version__ == "2.5.2"
assert scipy.__version__ == "1.18.0"
assert numba.__version__ == "0.67.0"
PY
```

Imported packages may report paths beginning with a generated name such as
`/OLIVIA_TYKKY_...`. This is the expected mount point inside the running
HPC-container-wrapper container. The persistent deployment remains at
`${deployment_prefix}` on the host filesystem.

This check passes only when `pip check` reports no broken requirements, all
five HyQD imports succeed, and the three numerical versions match the pinned
manifest. A different generated `/OLIVIA_TYKKY_...` prefix is expected and is
not a failure.

Record the resolved packages:

```bash
python -m pip freeze --all > "${manifest_root}/pip-freeze-base.txt"
python -m pip check > "${manifest_root}/pip-check-base.txt"
```

Run the test suite in a short compute-node job rather than on a login node.

## 11. Install and verify py3nj

`py3nj` is optional: `grid-lib` falls back to SymPy when it is absent. The
accelerated backend is nevertheless recommended for production angular setup.

Some host text editors can encounter an `ncurses` ABI conflict after the CPU
compiler stack is loaded. Create the short post-install script with Bash's
built-in `printf` instead:

```bash
printf '%s\n' \
    '#!/bin/bash' \
    'set -euo pipefail' \
    '' \
    'python -m pip install --no-cache-dir "py3nj==0.2.1"' \
    > "${manifest_root}/install-py3nj.sh"

bash -n "${manifest_root}/install-py3nj.sh"
sed -n '1,20p' "${manifest_root}/install-py3nj.sh"
```

Make it executable and update only the candidate deployment:

```bash
chmod +x "${manifest_root}/install-py3nj.sh"

# Do not expose the deployment being updated through the host PATH.
export PATH="${PATH#${deployment_prefix}/bin:}"
hash -r
command -v conda-containerize

set -o pipefail
conda-containerize update \
    "${deployment_prefix}" \
    --post-install "${manifest_root}/install-py3nj.sh" 2>&1 |
tee "${manifest_root}/py3nj-install.log"
update_status=${PIPESTATUS[0]}
set +o pipefail
test "${update_status}" -eq 0
```

Then verify:

```bash
export PATH="${deployment_prefix}/bin:${PATH}"
hash -r

python - <<'PY'
import importlib.util
import py3nj
from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend

specification = importlib.util.find_spec("py3nj")
print("py3nj:", py3nj.__version__, specification.origin)
print("Gaunt backend:", get_gaunt_backend(resolve=True))
PY
```

The desired final line is `Gaunt backend: py3nj`. If the build fails, retain
the SymPy backend for correctness and save the complete build log before
changing compilers or build dependencies.

## 12. Install mpi4py against Olivia Open MPI

This is the deployment's critical qualification step. Load the same module
stack used earlier and verify `mpicc` before updating the container:

```bash
module --force purge
module load "${GRID_TDSE_MPI_STACK}"
module load "${GRID_TDSE_CONTAINER_WRAPPER_MODULE}"
module load "${GRID_TDSE_MPI_MODULE}"

command -v mpicc
mpicc --showme:version
export MPICC="$(command -v mpicc)"
```

Create `${manifest_root}/install-mpi.sh` without invoking a host text editor:

```bash
printf '%s\n' \
    '#!/bin/bash' \
    'set -euo pipefail' \
    '' \
    ': "${MPICC:?MPICC must name Olivia Open MPI compiler wrapper}"' \
    'python -m pip uninstall -y mpi4py || true' \
    'python -m pip install --no-cache-dir --no-binary=mpi4py "mpi4py>=4,<5"' \
    > "${manifest_root}/install-mpi.sh"

bash -n "${manifest_root}/install-mpi.sh"
sed -n '1,30p' "${manifest_root}/install-mpi.sh"
```

Make it executable and update the candidate deployment:

```bash
chmod +x "${manifest_root}/install-mpi.sh"

# Do not expose the deployment being updated through the host PATH.
export PATH="${PATH#${deployment_prefix}/bin:}"
hash -r
command -v conda-containerize

set -o pipefail
conda-containerize update \
    "${deployment_prefix}" \
    --post-install "${manifest_root}/install-mpi.sh" 2>&1 |
tee "${manifest_root}/mpi4py-install.log"
update_status=${PIPESTATUS[0]}
set +o pipefail
test "${update_status}" -eq 0
```

If `mpicc` is not visible during the post-install step, stop. Do not install a
Conda MPI implementation as a workaround. Preserve the build log and consult
Sigma2 support about exposing the current `NRIS/CPU` Open MPI toolchain to the
HPC-container-wrapper build.

Verify the library reported by `mpi4py`:

```bash
export PATH="${deployment_prefix}/bin:${PATH}"
hash -r

python - <<'PY'
import mpi4py
from mpi4py import MPI

print("mpi4py:", mpi4py.__version__, mpi4py.__file__)
print(MPI.Get_library_version().rstrip("\x00"))
PY

python -m pip check

python -m pip freeze --all > "${manifest_root}/pip-freeze.txt"
python -m pip check > "${manifest_root}/pip-check.txt"
```

The output must identify the Olivia Open MPI implementation selected above.
Finding MPICH, a Conda Open MPI, or a different site Open MPI is a failed
qualification.

## 13. Run the serial and single-node MPI qualification job

After leaving the interactive build allocation, load `NRIS/Login` on the login
node. Install the version-controlled qualification script into the work-area
job directory instead of pasting a second copy into an editor:

```bash
(
    set -euo pipefail

    module --force purge
    module load NRIS/Login
    export GRID_TDSE_PATHS_FILE="/cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh"
    source "${GRID_TDSE_PATHS_FILE}"

    qualification_source="${GRID_TDSE_SRC_ROOT}/grid-tdse/examples/spherical/two_radial/Olivia/qualification"
    qualification_script="${GRID_TDSE_WORK_ROOT}/jobs/qualify-grid-tdse.sh"
    source_script="${qualification_source}/qualify_grid_tdse.sh"

    mkdir -p "${GRID_TDSE_WORK_ROOT}/jobs"
    if [[ ! -s "${source_script}" ]]
    then
        echo "STOP: qualification source is missing or empty: ${source_script}" >&2
        exit 1
    fi

    install -m 0755 "${source_script}" "${qualification_script}"
    test -s "${qualification_script}"
    bash -n "${qualification_script}"
)
```

The tracked script verifies that all five source checkouts are clean and still
match the immutable commit manifest before it runs tests. It also redirects
Numba, Matplotlib, and Python bytecode caches to `$LOCALSCRATCH`.

Submit the candidate deployment explicitly and persist the returned job ID:

```bash
(
    set -euo pipefail

    export GRID_TDSE_PATHS_FILE="/cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh"
    source "${GRID_TDSE_PATHS_FILE}"

    deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
    qualification_script="${GRID_TDSE_WORK_ROOT}/jobs/qualify-grid-tdse.sh"
    test -s "${qualification_script}"
    bash -n "${qualification_script}"

    cd "${GRID_TDSE_WORK_ROOT}/jobs"
    job_id=$(
        sbatch \
            --parsable \
            --account="${GRID_TDSE_PROJECT_ID}" \
            --export="ALL,GRID_TDSE_PATHS_FILE=${GRID_TDSE_PATHS_FILE},GRID_TDSE_DEPLOYMENT=${deployment_id}" \
            "${qualification_script}"
    )

    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]
    then
        echo "STOP: sbatch did not return a numeric job ID: ${job_id}" >&2
        exit 1
    fi

    printf 'Qualification job: %s\n' "${job_id}"
    printf '%s\n' "${job_id}" \
        > "${GRID_TDSE_WORK_ROOT}/jobs/latest-qualification-job-id.txt"
)
```

This leaves the existing `current` deployment unchanged while the candidate
is being tested.

Monitor the queued or running job with:

```bash
# These commands also restore the variables after a logout or reconnect.
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh
deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
job_id=$(cat "${GRID_TDSE_WORK_ROOT}/jobs/latest-qualification-job-id.txt")

squeue \
    -j "${job_id}" \
    --format='%.18i %.30j %.10T %.10M %.10l %.24R'
```

`PD` means pending and `R` means running. Once the job disappears from
`squeue`, inspect accounting and the output file rather than assuming success:

```bash
sacct -X \
    -j "${job_id}" \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,NodeList

output_file=$(find "${GRID_TDSE_WORK_ROOT}/jobs" \
    -maxdepth 1 -type f -name "*-${job_id}.out" -print -quit)

if [[ -z "${output_file}" ]]
then
    echo "FAIL: no output file found for job ${job_id}" >&2
    false
fi

printf 'Output: %s\n' "${output_file}"

grep --text -E \
    'Traceback|FAILED|ERROR|ImportError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY' \
    "${output_file}" || true

tail -n 100 "${output_file}"
```

The official Olivia MPI example uses `srun`. Do not add Saga-specific
`--mpi=pmix` options unless `srun --mpi=list` on Olivia confirms that they are
appropriate.

The qualification passes only when:

1. `pip check` reports no broken requirements;
2. the complete local test suite passes;
3. both MPI ranks report the expected size;
4. `MPI.Get_library_version()` reports the selected Olivia Open MPI;
5. there are no libfabric, PMIx, import, or Numba-cache errors.

## 14. Run a grid-tdse serial/MPI numerical-equivalence gate

The generic `mpi4py` launch in Step 13 proves that two ranks start, but it does
not exercise grid-tdse's distributed Hamiltonian or propagation. Before
promoting the deployment, run a deliberately small H2+ dynamics calculation
that compares the two-rank result with the serial implementation.

On the login node, install the version-controlled equivalence script next to
the Step 13 job script:

```bash
(
    set -euo pipefail

    module --force purge
    module load NRIS/Login
    export GRID_TDSE_PATHS_FILE="/cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh"
    source "${GRID_TDSE_PATHS_FILE}"

    qualification_source="${GRID_TDSE_SRC_ROOT}/grid-tdse/examples/spherical/two_radial/Olivia/qualification"
    source_script="${qualification_source}/qualify_grid_tdse_equivalence.sh"
    equivalence_script="${GRID_TDSE_WORK_ROOT}/jobs/qualify-grid-tdse-equivalence.sh"

    if [[ ! -s "${source_script}" ]]
    then
        echo "STOP: equivalence source is missing or empty: ${source_script}" >&2
        exit 1
    fi

    install -m 0755 "${source_script}" "${equivalence_script}"
    test -s "${equivalence_script}"
    bash -n "${equivalence_script}"
)
```

Validate, submit, and record the job ID:

```bash
(
    set -euo pipefail

    export GRID_TDSE_PATHS_FILE="/cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh"
    source "${GRID_TDSE_PATHS_FILE}"

    equivalence_script="${GRID_TDSE_WORK_ROOT}/jobs/qualify-grid-tdse-equivalence.sh"
    deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")

    test -s "${equivalence_script}"
    bash -n "${equivalence_script}"

    cd "${GRID_TDSE_WORK_ROOT}/jobs"
    equivalence_job_id=$(
        sbatch \
            --parsable \
            --account="${GRID_TDSE_PROJECT_ID}" \
            --export="ALL,GRID_TDSE_PATHS_FILE=${GRID_TDSE_PATHS_FILE},GRID_TDSE_DEPLOYMENT=${deployment_id}" \
            "${equivalence_script}"
    )

    if [[ ! "${equivalence_job_id}" =~ ^[0-9]+$ ]]
    then
        echo "STOP: sbatch did not return a numeric job ID: ${equivalence_job_id}" >&2
        exit 1
    fi

    printf 'Equivalence job: %s\n' "${equivalence_job_id}"
    printf '%s\n' "${equivalence_job_id}" \
        > "${GRID_TDSE_WORK_ROOT}/jobs/latest-equivalence-job-id.txt"
)
```

Once it finishes, inspect Slurm accounting, the output, and the JSON report:

```bash
# These commands also restore the variables after a logout or reconnect.
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh
deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
equivalence_job_id=$(cat \
    "${GRID_TDSE_WORK_ROOT}/jobs/latest-equivalence-job-id.txt")

sacct -X \
    -j "${equivalence_job_id}" \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,NodeList

equivalence_output=$(find "${GRID_TDSE_WORK_ROOT}/jobs" \
    -maxdepth 1 -type f -name "*-${equivalence_job_id}.out" -print -quit)
equivalence_root="${GRID_TDSE_WORK_ROOT}/qualification/${deployment_id}/equivalence-${equivalence_job_id}"

if [[ -z "${equivalence_output}" ]]
then
    echo "FAIL: no output file found for job ${equivalence_job_id}" >&2
    false
fi

printf 'Output: %s\n' "${equivalence_output}"

grep --text -E \
    'Traceback|FAILED|ERROR|ImportError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY' \
    "${equivalence_output}" || true

tail -n 100 "${equivalence_output}"
sed -n '1,240p' "${equivalence_root}/results/equivalence.json"
```

The gate passes only when Slurm reports `COMPLETED` with exit code `0:0`, the
output contains `Grid-tdse serial/MPI equivalence: PASS`, and the JSON status
is `PASS` with both numerical differences below `1e-8`.

Python 3.14 may also print invalid-escape `SyntaxWarning` messages from
mathematical docstrings in `grid-lib`. They should be corrected separately,
but they are non-fatal and do not invalidate a passing numerical comparison.

## 15. Run a representative single-node stress and scaling gate

The tiny Step 14 calculation proves numerical equivalence but does not stress
construction, sustained SIA propagation, or shared-memory-node MPI scaling.
Before requesting multiple nodes, run the tracked single-node gate. It performs
an `l_max=8` production-grid construction-memory check and propagates the
production H2+ radial grid for 100 steps with 1, 2, 4, 8, and 16 ranks.
The gate uses `krylov_rtol=1e-13` and `krylov_atol=1e-15`. The looser example
defaults, and an intermediate `1e-12`/`1e-14` test, did not keep every
100-step serial comparison below the gate's `1e-8` threshold on Olivia.

Install, validate, and submit the tracked script from a login node:

```bash
(
    set -euo pipefail

    module --force purge
    module load NRIS/Login
    export GRID_TDSE_PATHS_FILE="/cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh"
    source "${GRID_TDSE_PATHS_FILE}"

    qualification_source="${GRID_TDSE_SRC_ROOT}/grid-tdse/examples/spherical/two_radial/Olivia/qualification"
    source_script="${qualification_source}/qualify_grid_tdse_single_node.sh"
    job_root="${GRID_TDSE_WORK_ROOT}/jobs"
    single_node_script="${job_root}/qualify-grid-tdse-single-node.sh"
    deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")

    if [[ ! -s "${source_script}" ]]
    then
        echo "STOP: single-node source is missing or empty: ${source_script}" >&2
        exit 1
    fi

    install -m 0755 "${source_script}" "${single_node_script}"
    test -s "${single_node_script}"
    bash -n "${single_node_script}"

    single_node_job_id=$(
        sbatch \
            --parsable \
            --chdir="${job_root}" \
            --account="${GRID_TDSE_PROJECT_ID}" \
            --export="ALL,GRID_TDSE_PATHS_FILE=${GRID_TDSE_PATHS_FILE},GRID_TDSE_DEPLOYMENT=${deployment_id}" \
            "${single_node_script}"
    )

    if [[ ! "${single_node_job_id}" =~ ^[0-9]+$ ]]
    then
        echo "STOP: sbatch did not return a numeric job ID: ${single_node_job_id}" >&2
        exit 1
    fi

    printf 'Single-node job: %s\n' "${single_node_job_id}"
    printf '%s\n' "${single_node_job_id}" \
        > "${GRID_TDSE_WORK_ROOT}/jobs/latest-single-node-job-id.txt"
)
```

Copy each parenthesized block as a unit. The subshell enables Bash's strict
error handling, checks that the source and installed scripts are non-empty, and
records a job ID only after `sbatch` returns a numeric value. This prevents a
failed extraction or copy from leaving an empty script that is later submitted.
The explicit `--chdir` also ensures that Slurm writes the relative output path
from the tracked script into `${GRID_TDSE_WORK_ROOT}/jobs`, regardless of the
login shell's current directory.

The job requests 16 cores and 64 GiB on one `small`-partition CPU node, within
the documented [Olivia small-job limits](https://documentation.sigma2.no/jobs/job_scripts/olivia_job_scripts.html#small).
Its 30-minute limit is an upper bound; inspect queue policy before increasing
the request. After a logout or when the job finishes, recover its ID and
inspect the accounting and output:

```bash
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh
deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
single_node_job_id=$(cat \
    "${GRID_TDSE_WORK_ROOT}/jobs/latest-single-node-job-id.txt")

sacct \
    -j "${single_node_job_id}" \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList

single_node_output=$(find "${GRID_TDSE_WORK_ROOT}/jobs" \
    -maxdepth 1 -type f -name "*-${single_node_job_id}.out" -print -quit)
single_node_root="${GRID_TDSE_WORK_ROOT}/qualification/${deployment_id}/single-node-${single_node_job_id}"

if [[ -z "${single_node_output}" ]]
then
    echo "FAIL: no output file found for job ${single_node_job_id}" >&2
    false
fi

grep --text -E \
    'Traceback|FAILED|ERROR|ImportError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY' \
    "${single_node_output}" || true
tail -n 120 "${single_node_output}"

find "${single_node_root}/results" \
    -maxdepth 1 -type f -name '*.json' -print |
sort
```

The gate passes only when Slurm reports `COMPLETED` with exit code `0:0`, the
output contains `Single-node stress correctness: PASS`, every rank count
agrees with its serial reference below `1e-8`, and at least one MPI layout is
faster than the one-rank distributed calculation. Retain the full timing table
for later Olivia-versus-Saga and multi-node comparisons. Do not assume that
the largest rank count is optimal.

The initial validated run produced the following distributed propagation
times. These are qualification measurements, not general performance claims:

| MPI ranks | Process grid | Wall time (s) | Speedup | Serial state difference |
| ---: | :---: | ---: | ---: | ---: |
| 1 | `1x1` | 58.525 | 1.000 | `4.250e-15` |
| 2 | `1x2` | 34.251 | 1.709 | `3.672e-15` |
| 4 | `2x2` | 20.389 | 2.870 | `6.761e-15` |
| 8 | `2x4` | 15.078 | 3.882 | `6.092e-15` |
| 16 | `4x4` | 12.179 | 4.805 | `6.073e-15` |

All five layouts completed 100 steps with norm drift below `6.2e-15`. The
16-rank layout minimized wall time, while declining parallel efficiency shows
that lower rank counts may provide a better allocation-cost tradeoff.

## 16. Multi-node MPI configuration

The single-node `ob1` settings above are not valid for a multi-node
qualification. Sigma2's current
[Open MPI guidance for Olivia](https://documentation.sigma2.no/hpc_machines/olivia/openmpi.html#multi-node-jobs)
requires custom builds and containers to use the `cm`/`ofi` path through
`libfabric`. The tracked gate sets:

```bash
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=ofi
export OMPI_MCA_mtl_ofi_av=table
export PRTE_MCA_ras_base_launch_orted_on_hn=1
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid
```

Forcing `cm`/`ofi` is intentional: if Slingshot or `libfabric` cannot be
initialized, the job must fail rather than silently fall back to TCP. The
initial qualification uses the default `cxi` provider. Treat the optional
`lnx` provider as a separate performance experiment after the correctness gate
passes.

After setting the multi-node transport variables, do not import
`mpi4py.MPI` directly in the batch shell. Such a process is a singleton rather
than a task in the multi-node Slurm step, so `cxi` attempts a single-node
initialization and fails at `fi_domain` with error 38. The tracked gate obtains
and prints `MPI.Get_library_version()` from rank zero inside its first
two-node `srun` placement check.

The multi-node gate requests two whole CPU nodes and runs 2, 4, 8, and 16
ranks, evenly divided between the nodes. Each layout performs an MPI collective
placement check followed by the same 100-step H2+ propagation used in Step 15.
Every result is compared with the serial implementation using the validated
Krylov tolerances.

Olivia `large` jobs are allocated and billed as whole nodes regardless of how
few ranks they launch. This two-node run is therefore a controlled transport,
correctness, and scaling qualification, not an allocation-efficiency claim.
See the official
[large-job documentation](https://documentation.sigma2.no/jobs/job_scripts/olivia_job_scripts.html#large)
before changing its resources.

Install, validate, and submit the tracked script from a login node:

```bash
(
    set -euo pipefail

    module --force purge
    module load NRIS/Login
    export GRID_TDSE_PATHS_FILE="/cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh"
    source "${GRID_TDSE_PATHS_FILE}"

    repository_root="${GRID_TDSE_SRC_ROOT}/grid-tdse"
    script_path="examples/spherical/two_radial/Olivia/qualification/qualify_grid_tdse_multi_node.sh"
    job_root="${GRID_TDSE_WORK_ROOT}/jobs"
    multi_node_script="${job_root}/qualify-grid-tdse-multi-node.sh"
    deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")

    mkdir -p "${job_root}"
    git -C "${repository_root}" fetch origin main

    if ! git -C "${repository_root}" cat-file -e \
        "origin/main:${script_path}"
    then
        echo "STOP: the multi-node script is not present on origin/main" >&2
        exit 1
    fi

    temporary_script=$(mktemp \
        "${job_root}/.qualify-grid-tdse-multi-node.XXXXXX")

    cleanup()
    {
        if [[ -e "${temporary_script}" ]]
        then
            rm -- "${temporary_script}"
        fi
    }
    trap cleanup EXIT

    git -C "${repository_root}" show \
        "origin/main:${script_path}" \
        > "${temporary_script}"

    if [[ ! -s "${temporary_script}" ]]
    then
        echo "STOP: extracted multi-node script is empty" >&2
        exit 1
    fi

    bash -n "${temporary_script}"
    install -m 0755 "${temporary_script}" "${multi_node_script}"
    test -s "${multi_node_script}"
    bash -n "${multi_node_script}"

    multi_node_job_id=$(
        sbatch \
            --parsable \
            --chdir="${job_root}" \
            --account="${GRID_TDSE_PROJECT_ID}" \
            --export="ALL,GRID_TDSE_PATHS_FILE=${GRID_TDSE_PATHS_FILE},GRID_TDSE_DEPLOYMENT=${deployment_id}" \
            "${multi_node_script}"
    )

    if [[ ! "${multi_node_job_id}" =~ ^[0-9]+$ ]]
    then
        echo "STOP: sbatch did not return a numeric job ID: ${multi_node_job_id}" >&2
        exit 1
    fi

    printf 'Multi-node job: %s\n' "${multi_node_job_id}"
    printf '%s\n' "${multi_node_job_id}" \
        > "${job_root}/latest-multi-node-job-id.txt"

    squeue \
        -j "${multi_node_job_id}" \
        --format='%.18i %.30j %.10T %.10M %.10l %.24R'
)
```

The source-manifest checks intentionally require the deployment's five source
checkouts to remain pinned and clean. The commands above fetch `origin/main`
but do not switch, merge, or pull the checked-out commit. They extract only the
reviewed script into the work area, so an immutable deployment may run a newer
qualification driver without changing its installed code.

After the job leaves the queue, inspect its accounting, numerical results, and
rank-placement records:

```bash
(
    set -euo pipefail

    export GRID_TDSE_PATHS_FILE="/cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh"
    source "${GRID_TDSE_PATHS_FILE}"

    deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
    job_root="${GRID_TDSE_WORK_ROOT}/jobs"
    multi_node_job_id=$(cat "${job_root}/latest-multi-node-job-id.txt")
    output_file="${job_root}/grid_tdse_multi_node-${multi_node_job_id}.out"
    run_root="${GRID_TDSE_WORK_ROOT}/qualification/${deployment_id}/multi-node-${multi_node_job_id}"

    if [[ ! "${multi_node_job_id}" =~ ^[0-9]+$ ]]
    then
        echo "STOP: invalid recorded job ID: ${multi_node_job_id}" >&2
        exit 1
    fi

    sacct \
        -j "${multi_node_job_id}" \
        --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,NodeList

    test -f "${output_file}"
    grep --text -n -E \
        'Traceback|FAILED|ERROR|Error|AssertionError|MemoryError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY|exceeds tolerance' \
        "${output_file}" || true
    grep --text -E \
        'Allocated node:|MPI library:|Status:|MPI ranks:|Serial state difference:|Time-series maximum difference:|Distributed wall time:|Two-node production-grid scaling|Multi-node stress correctness:' \
        "${output_file}" || true
    tail -n 200 "${output_file}"

    find "${run_root}" \
        -maxdepth 2 \
        -type f \
        -print |
    sort
)
```

The gate passes only when:

1. Slurm reports `COMPLETED` with exit code `0:0`;
2. every placement record contains the requested number of ranks on exactly
   two nodes;
3. the forced `cm`/`ofi` and `cxi` transport initializes without fallback or
   provider errors;
4. every rank layout agrees with the serial state and observable time series
   below `1e-8`;
5. the output contains `Multi-node stress correctness: PASS`.

Retain the timings even if additional ranks do not improve this small problem.
The purpose of the first multi-node run is to qualify correctness and transport;
larger production-shaped scaling studies should follow only after it passes.

## 17. Promote or roll back a deployment

Do not overwrite a validated environment. Promote a candidate by changing the
`current` symlink only after the qualification gates in Steps 13--16 pass:

```bash
readlink -f "${GRID_TDSE_CURRENT}"
ln -sfn "environments/${deployment_id}" "${GRID_TDSE_CURRENT}"
readlink -f "${GRID_TDSE_CURRENT}"
```

To roll back, point `current` to the preceding deployment ID:

```bash
ln -sfn "environments/<previous-deployment-id>" "${GRID_TDSE_CURRENT}"
```

Existing Slurm jobs retain the script and environment path visible when they
start. Record the resolved deployment path in every result manifest:

```bash
readlink -f "${GRID_TDSE_CURRENT}"
cat "${GRID_TDSE_ROOT}/manifests/<deployment-id>/source-commits.txt"
```

## 18. Update grid-tdse or a dependency

The safest update is a new immutable deployment. Perform the following ten
substeps instead of modifying the deployment selected by `current`.

### 18.1 Pull each changed repository with `--ff-only`

On a login node, load the login environment and the reusable paths file. In
the first command, replace the two placeholders with the real project ID and
username:

```bash
module --force purge
module load NRIS/Login
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh

repositories=(
    grid-lib
    integrator-lib
    electromagnetic-fields
    matter_field_params
    grid-tdse
)

for repository in "${repositories[@]}"
do
    git -C "${GRID_TDSE_SRC_ROOT}/${repository}" switch main
    git -C "${GRID_TDSE_SRC_ROOT}/${repository}" pull --ff-only origin main
done
```

This updates all five repositories, including dependencies that may not have
changed. A non-fast-forward failure should be investigated rather than
overridden with a merge or force operation.

### 18.2 Confirm that every source worktree is clean

```bash
source_status=0

for repository in "${repositories[@]}"
do
    status=$(
        git -C "${GRID_TDSE_SRC_ROOT}/${repository}" status --short
    )

    if [[ -n "${status}" ]]
    then
        printf '\nDIRTY: %s\n%s\n' "${repository}" "${status}"
        source_status=1
    else
        printf 'CLEAN: %s\n' "${repository}"
    fi
done

if (( source_status != 0 ))
then
    echo "Stop: at least one source worktree is dirty" >&2
    false
fi
```

Do not discard local work merely to make this check pass. Commit, preserve,
or otherwise resolve it in the corresponding repository first.

### 18.3 Choose a new deployment ID

Use a timestamp so that two deployments created on the same day cannot
collide. Change the descriptive suffix when the compiler or MPI stack changes:

```bash
export deployment_id="$(date +%Y%m%d-%H%M%S)-cpu-ompi508-update"
manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"
deployment_prefix="${GRID_TDSE_ENV_ROOT}/${deployment_id}"

test ! -e "${manifest_root}"
test ! -e "${deployment_prefix}"
mkdir -p "${manifest_root}"

readlink -f "${GRID_TDSE_CURRENT}" \
    > "${manifest_root}/previous-deployment.txt"
printf '%s\n' "${deployment_id}" \
    > "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt"

printf 'Candidate: %s\n' "${deployment_id}"
printf 'Previous:  %s\n' \
    "$(cat "${manifest_root}/previous-deployment.txt")"
```

The `latest-candidate-deployment.txt` file carries the selected ID into later
login or compute shells. It is only a convenience pointer; the immutable
manifest remains authoritative.

### 18.4 Capture all five source commits

```bash
grid_lib_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/grid-lib" rev-parse HEAD)
integrator_lib_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/integrator-lib" rev-parse HEAD)
fields_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/electromagnetic-fields" rev-parse HEAD)
matter_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/matter_field_params" rev-parse HEAD)
grid_tdse_commit=$(git -C "${GRID_TDSE_SRC_ROOT}/grid-tdse" rev-parse HEAD)

printf '%s\n' \
    "grid-lib ${grid_lib_commit}" \
    "integrator-lib ${integrator_lib_commit}" \
    "electromagnetic-fields ${fields_commit}" \
    "matter_field_params ${matter_commit}" \
    "grid-tdse ${grid_tdse_commit}" \
    > "${manifest_root}/source-commits.txt"

cat "${manifest_root}/source-commits.txt"
```

Record unchanged dependencies as well as the repository that motivated the
update.

### 18.5 Regenerate the environment and HyQD requirement manifests

Start from the template tracked by the new grid-tdse commit, compare it with
the previous qualified numerical environment, and inspect the current package
metadata before accepting it:

```bash
previous_deployment=$(
    basename "$(cat "${manifest_root}/previous-deployment.txt")"
)
previous_manifest="${GRID_TDSE_ROOT}/manifests/${previous_deployment}"
environment_template="${GRID_TDSE_SRC_ROOT}/grid-tdse/examples/spherical/two_radial/Olivia/qualification/environment.yml"

cp \
    "${environment_template}" \
    "${manifest_root}/environment.yml"

diff -u \
    "${previous_manifest}/environment.yml" \
    "${manifest_root}/environment.yml" || true

for repository in "${repositories[@]}"
do
    printf '\n===== %s/pyproject.toml =====\n' "${repository}"
    sed -n '/dependencies[[:space:]]*=/,/]/p' \
        "${GRID_TDSE_SRC_ROOT}/${repository}/pyproject.toml"
done

sed -n '1,240p' "${manifest_root}/environment.yml"
```

If a dependency bound changed, edit the copied manifest with `nano` or
`vi` before continuing. Then generate source requirements pinned to the new
commits:

```bash
printf '%s\n' \
    "HyQD-grid-lib @ git+https://github.com/HyQD/grid-lib.git@${grid_lib_commit}" \
    "HyQD-integrator-lib @ git+https://github.com/HyQD/integrator-lib.git@${integrator_lib_commit}" \
    "HyQD-electromagnetic-fields @ git+https://github.com/HyQD/electromagnetic-fields.git@${fields_commit}" \
    "HyQD-matter-field-params @ git+https://github.com/HyQD/matter_field_params.git@${matter_commit}" \
    "HyQD-grid-tdse @ git+https://github.com/HyQD/grid-tdse.git@${grid_tdse_commit}" \
    > "${manifest_root}/hyqd-requirements.txt"

cat "${manifest_root}/hyqd-requirements.txt"
```

### 18.6 Build a new containerized prefix

From the login node, obtain a compute allocation:

```bash
salloc \
    --account="${GRID_TDSE_PROJECT_ID}" \
    --partition=small \
    --qos=devel \
    --time=02:00:00 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem-per-cpu=4G
```

In the resulting compute shell, reconstruct the deployment variables, load
the pinned toolchain, and build the new prefix:

```bash
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh

deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"
deployment_prefix="${GRID_TDSE_ENV_ROOT}/${deployment_id}"

module --force purge
module load "${GRID_TDSE_MPI_STACK}"
module load "${GRID_TDSE_CONTAINER_WRAPPER_MODULE}"
module load "${GRID_TDSE_MPI_MODULE}"

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export APPTAINER_CACHEDIR="${LOCALSCRATCH}/apptainer-cache"
export APPTAINER_TMPDIR="${LOCALSCRATCH}/apptainer-tmp"
export PIP_CACHE_DIR="${LOCALSCRATCH}/pip-cache"
export XDG_CACHE_HOME="${LOCALSCRATCH}/xdg-cache"
mkdir -p \
    "${APPTAINER_CACHEDIR}" \
    "${APPTAINER_TMPDIR}" \
    "${PIP_CACHE_DIR}" \
    "${XDG_CACHE_HOME}"

module list 2> "${manifest_root}/modules.txt"
mpicc --showme:version > "${manifest_root}/mpicc-version.txt"
mpicc --showme:command > "${manifest_root}/mpicc-command.txt"

test ! -e "${deployment_prefix}"
conda-containerize new \
    --mamba \
    -r "${manifest_root}/hyqd-requirements.txt" \
    --prefix "${deployment_prefix}" \
    "${manifest_root}/environment.yml"
```

### 18.7 Install `py3nj` and Olivia-linked `mpi4py`

Still in the compute allocation, create both post-install scripts:

```bash
printf '%s\n' \
    '#!/bin/bash' \
    'set -euo pipefail' \
    '' \
    'python -m pip install --no-cache-dir "py3nj==0.2.1"' \
    > "${manifest_root}/install-py3nj.sh"

printf '%s\n' \
    '#!/bin/bash' \
    'set -euo pipefail' \
    '' \
    ': "${MPICC:?MPICC must name Olivia Open MPI compiler wrapper}"' \
    'python -m pip uninstall -y mpi4py || true' \
    'python -m pip install --no-cache-dir --no-binary=mpi4py "mpi4py>=4,<5"' \
    > "${manifest_root}/install-mpi.sh"

chmod +x \
    "${manifest_root}/install-py3nj.sh" \
    "${manifest_root}/install-mpi.sh"
bash -n "${manifest_root}/install-py3nj.sh"
bash -n "${manifest_root}/install-mpi.sh"

export MPICC="$(command -v mpicc)"
conda-containerize update \
    "${deployment_prefix}" \
    --post-install "${manifest_root}/install-py3nj.sh"
conda-containerize update \
    "${deployment_prefix}" \
    --post-install "${manifest_root}/install-mpi.sh"
```

Select and verify the completed candidate:

```bash
export PATH="${deployment_prefix}/bin:${PATH}"
hash -r

python -m pip check
python - <<'PY'
import mpi4py
import py3nj
from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend
from mpi4py import MPI

print("mpi4py:", mpi4py.__version__)
print(MPI.Get_library_version().rstrip("\x00"))
print("py3nj:", py3nj.__version__)
print("Gaunt backend:", get_gaunt_backend(resolve=True))
PY

python -m pip freeze --all > "${manifest_root}/pip-freeze.txt"
python -m pip check > "${manifest_root}/pip-check.txt"
```

The output must report the pinned Olivia Open MPI and `Gaunt backend: py3nj`.
Exit the compute allocation after these checks complete.

### 18.8 Run the qualification gates

Back on the login node, refresh the tracked scripts in the work area. Submit
the test-suite/MPI-launch job, numerical-equivalence job, representative
single-node gate, and two-node gate as an `afterok` chain:

```bash
(
    set -euo pipefail

    module --force purge
    module load NRIS/Login
    export GRID_TDSE_PATHS_FILE="/cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh"
    source "${GRID_TDSE_PATHS_FILE}"

    deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
    manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"
    qualification_source="${GRID_TDSE_SRC_ROOT}/grid-tdse/examples/spherical/two_radial/Olivia/qualification"
    job_root="${GRID_TDSE_WORK_ROOT}/jobs"

    source_scripts=(
        qualify_grid_tdse.sh
        qualify_grid_tdse_equivalence.sh
        qualify_grid_tdse_single_node.sh
        qualify_grid_tdse_multi_node.sh
    )
    installed_scripts=(
        qualify-grid-tdse.sh
        qualify-grid-tdse-equivalence.sh
        qualify-grid-tdse-single-node.sh
        qualify-grid-tdse-multi-node.sh
    )

    mkdir -p "${job_root}"
    for index in "${!source_scripts[@]}"
    do
        source_script="${qualification_source}/${source_scripts[index]}"
        installed_script="${job_root}/${installed_scripts[index]}"

        if [[ ! -s "${source_script}" ]]
        then
            echo "STOP: qualification source is missing or empty: ${source_script}" >&2
            exit 1
        fi

        install -m 0755 "${source_script}" "${installed_script}"
        test -s "${installed_script}"
        bash -n "${installed_script}"
    done

    cd "${job_root}"

    qualification_job_id=$(
        sbatch \
            --parsable \
            --account="${GRID_TDSE_PROJECT_ID}" \
            --export="ALL,GRID_TDSE_PATHS_FILE=${GRID_TDSE_PATHS_FILE},GRID_TDSE_DEPLOYMENT=${deployment_id}" \
            qualify-grid-tdse.sh
    )
    [[ "${qualification_job_id}" =~ ^[0-9]+$ ]]

    equivalence_job_id=$(
        sbatch \
            --parsable \
            --dependency="afterok:${qualification_job_id}" \
            --account="${GRID_TDSE_PROJECT_ID}" \
            --export="ALL,GRID_TDSE_PATHS_FILE=${GRID_TDSE_PATHS_FILE},GRID_TDSE_DEPLOYMENT=${deployment_id}" \
            qualify-grid-tdse-equivalence.sh
    )
    [[ "${equivalence_job_id}" =~ ^[0-9]+$ ]]

    single_node_job_id=$(
        sbatch \
            --parsable \
            --dependency="afterok:${equivalence_job_id}" \
            --account="${GRID_TDSE_PROJECT_ID}" \
            --export="ALL,GRID_TDSE_PATHS_FILE=${GRID_TDSE_PATHS_FILE},GRID_TDSE_DEPLOYMENT=${deployment_id}" \
            qualify-grid-tdse-single-node.sh
    )
    [[ "${single_node_job_id}" =~ ^[0-9]+$ ]]

    multi_node_job_id=$(
        sbatch \
            --parsable \
            --dependency="afterok:${single_node_job_id}" \
            --account="${GRID_TDSE_PROJECT_ID}" \
            --export="ALL,GRID_TDSE_PATHS_FILE=${GRID_TDSE_PATHS_FILE},GRID_TDSE_DEPLOYMENT=${deployment_id}" \
            qualify-grid-tdse-multi-node.sh
    )
    [[ "${multi_node_job_id}" =~ ^[0-9]+$ ]]

    printf 'Qualification job: %s\n' "${qualification_job_id}"
    printf 'Equivalence job:   %s\n' "${equivalence_job_id}"
    printf 'Single-node job:   %s\n' "${single_node_job_id}"
    printf 'Multi-node job:    %s\n' "${multi_node_job_id}"
    printf '%s\n' "${qualification_job_id}" \
        > "${manifest_root}/qualification-job-id.txt"
    printf '%s\n' "${equivalence_job_id}" \
        > "${manifest_root}/equivalence-job-id.txt"
    printf '%s\n' "${single_node_job_id}" \
        > "${manifest_root}/single-node-job-id.txt"
    printf '%s\n' "${multi_node_job_id}" \
        > "${manifest_root}/multi-node-job-id.txt"
)
```

When all four jobs leave the queue, verify their states and outputs:

```bash
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh
deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"
qualification_job_id=$(cat "${manifest_root}/qualification-job-id.txt")
equivalence_job_id=$(cat "${manifest_root}/equivalence-job-id.txt")
single_node_job_id=$(cat "${manifest_root}/single-node-job-id.txt")
multi_node_job_id=$(cat "${manifest_root}/multi-node-job-id.txt")

job_ids="${qualification_job_id},${equivalence_job_id},${single_node_job_id},${multi_node_job_id}"
sacct -X \
    -j "${job_ids}" \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,NodeList

for job_id in \
    "${qualification_job_id}" \
    "${equivalence_job_id}" \
    "${single_node_job_id}" \
    "${multi_node_job_id}"
do
    output_file=$(find "${GRID_TDSE_WORK_ROOT}/jobs" \
        -maxdepth 1 -type f -name "*-${job_id}.out" -print -quit)
    printf '\n===== %s =====\n' "${output_file}"
    grep --text -E \
        'Traceback|FAILED|ERROR|ImportError|MPI_ABORT|CANCELLED|TIMEOUT|OUT_OF_MEMORY' \
        "${output_file}" || true
    tail -n 40 "${output_file}"
done
```

Do not promote unless all four jobs are `COMPLETED` with exit code `0:0`, the
test suite passes, the equivalence output contains
`Grid-tdse serial/MPI equivalence: PASS`, and the scaling output contains
both `Single-node stress correctness: PASS` and
`Multi-node stress correctness: PASS`.

### 18.9 Promote the new prefix by changing `current`

```bash
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh

deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"
deployment_prefix="${GRID_TDSE_ENV_ROOT}/${deployment_id}"
previous_prefix=$(cat "${manifest_root}/previous-deployment.txt")

test -d "${deployment_prefix}"
test -d "${previous_prefix}"

printf 'Before: %s\n' "$(readlink -f "${GRID_TDSE_CURRENT}")"
ln -sfn "environments/${deployment_id}" "${GRID_TDSE_CURRENT}"
printf 'After:  %s\n' "$(readlink -f "${GRID_TDSE_CURRENT}")"

test "$(readlink -f "${GRID_TDSE_CURRENT}")" = "${deployment_prefix}"
date --iso-8601=seconds > "${manifest_root}/promoted-at.txt"
```

If the final test fails, restore the previous deployment immediately:

```bash
previous_deployment=$(basename "${previous_prefix}")
ln -sfn \
    "environments/${previous_deployment}" \
    "${GRID_TDSE_CURRENT}"
```

### 18.10 Retain the previous prefix through production validation

Confirm that both deployments remain available and record their relationship:

```bash
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh

deployment_id=$(cat "${GRID_TDSE_ROOT}/latest-candidate-deployment.txt")
manifest_root="${GRID_TDSE_ROOT}/manifests/${deployment_id}"
deployment_prefix="${GRID_TDSE_ENV_ROOT}/${deployment_id}"
previous_prefix=$(cat "${manifest_root}/previous-deployment.txt")

test -d "${previous_prefix}"
test -d "${deployment_prefix}"

printf 'previous=%s\ncurrent=%s\n' \
    "${previous_prefix}" \
    "$(readlink -f "${GRID_TDSE_CURRENT}")" \
    > "${manifest_root}/deployment-transition.txt"

cat "${manifest_root}/deployment-transition.txt"
find "${GRID_TDSE_ENV_ROOT}" \
    -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
sort
```

Do not delete the previous prefix during this procedure. Retain it until the
new deployment has passed the representative single-node and, when relevant,
multi-node production validation. Rollback then remains a symlink change.

Rebuild the complete environment when Python, NumPy, SciPy, Numba, `py3nj`,
or the Open MPI module changes. A source-only change could be applied with
`conda-containerize update`, but a new deployment is preferable for benchmark
provenance and rollback. This is especially important while some HyQD packages
retain the same development version across multiple source commits.

## 19. Per-job provenance

Every production or benchmark job should record the immutable deployment,
software stack, Slurm placement, thread and cache settings, physical command
line, completion status, and output checksums. The following pattern provides
concrete commands that can be adapted to each job.

### 19.1 Pin the deployment when submitting

Do not let a queued job resolve the mutable `current` link after submission.
On the login node, resolve it first and export the resulting deployment ID to
the batch job:

```bash
module --force purge
module load NRIS/Login
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh

deployment_prefix=$(readlink -f "${GRID_TDSE_CURRENT}")
deployment_id=$(basename "${deployment_prefix}")
run_label="h2p-production"

test -d "${deployment_prefix}"
test -s "${GRID_TDSE_ROOT}/manifests/${deployment_id}/source-commits.txt"

job_id=$(
    sbatch \
        --parsable \
        --account="${GRID_TDSE_PROJECT_ID}" \
        --export="ALL,GRID_TDSE_DEPLOYMENT=${deployment_id},GRID_TDSE_RUN_LABEL=${run_label}" \
        path/to/job-script.sh
)

printf 'Deployment: %s\n' "${deployment_id}"
printf 'Job ID:     %s\n' "${job_id}"
printf '%s\n' "${job_id}" \
    > "${GRID_TDSE_WORK_ROOT}/jobs/latest-${run_label}-job-id.txt"
```

Replace `path/to/job-script.sh` and `run_label` with the real values. Use only
letters, digits, underscores, and hyphens in the label because it becomes part
of directory and pointer-file names. The deployment ID, rather than the
resolved path, is exported because the batch script reconstructs the path
from the shared paths file.

### 19.2 Capture provenance near the start of the batch script

Place the following commands after the `#SBATCH` directives and after loading
the required `NRIS/CPU` and Open MPI modules, but before running grid-tdse:

```bash
set -euo pipefail

: "${GRID_TDSE_DEPLOYMENT:?Submit with GRID_TDSE_DEPLOYMENT=<deployment-id>}"
: "${GRID_TDSE_RUN_LABEL:?Submit with GRID_TDSE_RUN_LABEL=<short-label>}"
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh

deployment_prefix="${GRID_TDSE_ENV_ROOT}/${GRID_TDSE_DEPLOYMENT}"
manifest_root="${GRID_TDSE_ROOT}/manifests/${GRID_TDSE_DEPLOYMENT}"
run_root="${GRID_TDSE_WORK_ROOT}/runs/${GRID_TDSE_RUN_LABEL}-${SLURM_JOB_ID}"
provenance_root="${run_root}/provenance"

test -d "${deployment_prefix}"
test -s "${manifest_root}/source-commits.txt"
mkdir -p "${provenance_root}" "${run_root}/results"

export PATH="${deployment_prefix}/bin:${PATH}"
hash -r

printf '%s\n' "${GRID_TDSE_DEPLOYMENT}" \
    > "${provenance_root}/deployment-id.txt"
printf '%s\n' "${SLURM_JOB_ID}" \
    > "${provenance_root}/job-id.txt"
printf '%s\n' "${deployment_prefix}" \
    > "${provenance_root}/deployment-prefix.txt"
cp "${manifest_root}/source-commits.txt" \
    "${provenance_root}/source-commits.txt"
cp "$0" "${provenance_root}/submitted-job-script.sh"

date --iso-8601=seconds > "${provenance_root}/started-at.txt"
hostname --fqdn > "${provenance_root}/launch-host.txt"
module --terse list 2> "${provenance_root}/modules.txt"
scontrol show job -d "${SLURM_JOB_ID}" \
    > "${provenance_root}/slurm-job-start.txt"
scontrol show hostnames "${SLURM_JOB_NODELIST}" \
    > "${provenance_root}/nodes.txt"
lscpu > "${provenance_root}/lscpu.txt"
taskset -pc $$ > "${provenance_root}/launch-affinity.txt"
```

Copying the submitted job script records its `#SBATCH` settings and all
hard-coded physical parameters. If the calculation reads separate input
files, copy them as well:

```bash
mkdir -p "${provenance_root}/inputs"
cp path/to/input-file "${provenance_root}/inputs/"
```

Replace `path/to/input-file`, or omit these two commands when every input is
present in the job script and command line.

### 19.3 Record software, MPI, and the Gaunt backend

Still inside the batch script, record the installed packages and the runtime
libraries seen through the deployment wrapper:

```bash
python --version > "${provenance_root}/python-version.txt" 2>&1
python -m pip freeze --all > "${provenance_root}/pip-freeze.txt"
python -m pip check > "${provenance_root}/pip-check.txt"

python - <<'PY' > "${provenance_root}/runtime-software.txt"
import mpi4py
import numba
import numpy
import scipy
from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend
from mpi4py import MPI

print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Numba:", numba.__version__)
print("mpi4py:", mpi4py.__version__)
print("Gaunt backend:", get_gaunt_backend(resolve=True))
print("MPI library:")
print(MPI.Get_library_version().rstrip("\x00"))
PY

python -m threadpoolctl -i numpy scipy \
    > "${provenance_root}/threadpools.txt"
```

### 19.4 Record task, thread, MPI, and cache settings

Record a safe, explicit allowlist instead of dumping the complete environment,
which could expose credentials or proxy settings:

```bash
provenance_variables=(
    SLURM_JOB_ID
    SLURM_JOB_NAME
    SLURM_JOB_NODELIST
    SLURM_NNODES
    SLURM_NTASKS
    SLURM_CPUS_PER_TASK
    OMP_NUM_THREADS
    OPENBLAS_NUM_THREADS
    MKL_NUM_THREADS
    NUMBA_NUM_THREADS
    OMPI_MCA_pml
    OMPI_MCA_btl
    OMPI_MCA_mtl
    OMPI_MCA_mtl_ofi_av
    NUMBA_CACHE_DIR
    MPLCONFIGDIR
    PYTHONPYCACHEPREFIX
    APPTAINER_CACHEDIR
    APPTAINER_TMPDIR
)

for variable in "${provenance_variables[@]}"
do
    printf '%s=%q\n' "${variable}" "${!variable-}"
done > "${provenance_root}/execution-environment.txt"
```

Add a variable to this allowlist when it affects numerical behavior,
placement, communication, or performance. Do not add tokens, passwords,
private keys, or complete proxy URLs containing credentials.

### 19.5 Record and run the exact scientific command

Put the complete invocation in a Bash array. The following is a structural
example; replace its contents with the real executable and all physical
inputs, tolerances, sampling options, and output paths:

```bash
simulation_command=(
    python
    path/to/grid-tdse-example.py
    --time-step 0.25
    --final-time 2.0
    --output-json "${run_root}/results/summary.json"
)

{
    printf '%q ' "${simulation_command[@]}"
    printf '\n'
} > "${provenance_root}/simulation-command.sh"

start_epoch=$(date +%s)
set +e
"${simulation_command[@]}"
simulation_status=$?
set -e
end_epoch=$(date +%s)

printf '%s\n' "${simulation_status}" \
    > "${provenance_root}/simulation-exit-code.txt"
printf '%s\n' "$((end_epoch - start_epoch))" \
    > "${provenance_root}/simulation-wall-seconds.txt"
date --iso-8601=seconds > "${provenance_root}/finished-at.txt"
```

For MPI, make the entire `srun` invocation part of the array, including task
count, CPU binding, process grid, and Python command. The saved `%q` form is a
shell-escaped record suitable for inspection and reuse.

### 19.6 Generate input and output checksums

After the simulation command returns, calculate checksums relative to the run
directory. Exclude the provenance directory from the output checksum list so
that writing the list does not checksum itself:

```bash
(
    cd "${run_root}"

    if [[ -d provenance/inputs ]]
    then
        find provenance/inputs -type f -print0 |
        LC_ALL=C sort -z |
        xargs -0 -r sha256sum
    fi
) > "${provenance_root}/input-sha256sums.txt"

(
    cd "${run_root}"
    find . \
        -path ./provenance -prune -o \
        -type f -print0 |
    LC_ALL=C sort -z |
    xargs -0 -r sha256sum
) > "${provenance_root}/output-sha256sums.txt"

sha256sum \
    "${provenance_root}/submitted-job-script.sh" \
    "${provenance_root}/source-commits.txt" \
    "${provenance_root}/simulation-command.sh" \
    > "${provenance_root}/provenance-sha256sums.txt"

exit "${simulation_status}"
```

Checksumming very large wavefunction archives adds storage I/O. It is still
appropriate for an archival run, but a performance-only benchmark may instead
checksum its JSON, CSV, log, and final-state products explicitly. Document any
excluded output class in the provenance directory.

### 19.7 Capture Slurm accounting after the job finishes

The final accounting record must be collected from the login node because
`sacct` is complete only after the batch job exits. Reconstruct the run path
from the label and recorded job ID:

```bash
source /cluster/projects/<project-id>/<username>/grid-tdse/olivia_grid_tdse_paths.sh

run_label="h2p-production"
job_id=$(cat \
    "${GRID_TDSE_WORK_ROOT}/jobs/latest-${run_label}-job-id.txt")
run_root="${GRID_TDSE_WORK_ROOT}/runs/${run_label}-${job_id}"
provenance_root="${run_root}/provenance"

sacct -X \
    -j "${job_id}" \
    --format=JobID,JobName%32,State,ExitCode,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS,NodeList \
    > "${provenance_root}/sacct.txt"

cat "${provenance_root}/sacct.txt"
cat "${provenance_root}/simulation-exit-code.txt"

(
    cd "${run_root}"
    sha256sum -c provenance/output-sha256sums.txt
)
```

Finally, verify that the minimum provenance record is complete:

```bash
required_provenance=(
    job-id.txt
    deployment-id.txt
    deployment-prefix.txt
    source-commits.txt
    submitted-job-script.sh
    modules.txt
    slurm-job-start.txt
    nodes.txt
    runtime-software.txt
    execution-environment.txt
    simulation-command.sh
    simulation-exit-code.txt
    simulation-wall-seconds.txt
    output-sha256sums.txt
    sacct.txt
)

for file in "${required_provenance[@]}"
do
    test -s "${provenance_root}/${file}" || {
        printf 'MISSING: %s\n' "${file}" >&2
        provenance_status=1
    }
done

test "${provenance_status:-0}" -eq 0 &&
    echo "Per-job provenance: COMPLETE"
```

Do not compare Olivia and Saga timings unless commits, numerical inputs,
tolerances, accepted work, rank/thread placement, and recorded output are
identical.

## 20. Known uncertainties for the first Olivia installation

The first qualification must resolve these site-specific questions:

1. whether the non-slim HPC-container-wrapper build exposes Olivia's `mpicc`
   and runtime Open MPI libraries during the `mpi4py` source build;
2. whether all host Open MPI and libfabric libraries are visible when the
   generated Python wrapper runs under `srun`;
3. whether `py3nj 0.2.1` builds with the selected Python and Fortran toolchain;
4. whether the documented single-node Open MPI settings are already injected
   by the current `NRIS/CPU` module;
5. whether a dedicated wheelhouse is preferable to Git installations for the
   project's access policy.

These are deployment checks, not known grid-tdse portability failures. Stop
at the first failed gate, preserve the complete log and module list, and avoid
substituting a Conda MPI implementation.
