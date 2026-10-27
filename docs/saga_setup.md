# Setting up grid-tdse on Saga

This guide describes a project-local development installation of `grid-tdse`
and its HyQD dependencies on Saga. It also explains how to update an existing
installation safely.

The commands and module names were last verified on 2026-10-27.

The commands use the following placeholders:

- `<project-id>`: the Sigma2 project ID, for example `nn1234k`;
- `<username>`: the Saga username.

Replace both placeholders before running the commands.

> **Warning:** Text such as `<project-id>`, `<username>`, and `...` denotes a
> placeholder in this guide. Do not type the angle brackets or a literal
> ellipsis into a shell command.

The setup keeps source repositories and the Conda environment under
`/cluster/projects`. This avoids consuming home-directory quota and makes the
installation available from both login and compute nodes. Consult the
[official Saga documentation](https://documentation.sigma2.no/hpc_machines/saga.html)
and the
[cluster-storage documentation](https://documentation.sigma2.no/files_storage/clusters.html)
for current machine and storage policies.

## 1. Choose the installation layout

This guide uses:

```text
/cluster/projects/<project-id>/<username>/
├── .conda-envs/
│   └── grid_tdse/
├── grid_tdse_paths.sh
├── jobs/
│   └── grid-tdse/
└── src/
    └── HyQD/
        ├── electromagnetic-fields/
        ├── grid-lib/
        ├── grid-tdse/
        ├── integrator-lib/
        └── matter_field_params/
```

All commands use absolute paths, so the initial setup may be run from the home
directory or another convenient directory.

## 2. Create a reusable paths file

Create
`/cluster/projects/<project-id>/<username>/grid_tdse_paths.sh` with:

```bash
#!/bin/bash

export PROJECT_ROOT="/cluster/projects/<project-id>/<username>"
export ENV_ROOT="${PROJECT_ROOT}/.conda-envs"
export ENV_PREFIX="${ENV_ROOT}/grid_tdse"
export SRC_ROOT="${PROJECT_ROOT}/src/HyQD"
```

Load it with:

```bash
source /cluster/projects/<project-id>/<username>/grid_tdse_paths.sh
```

Use `source`, rather than `bash grid_tdse_paths.sh`, because the variables
must remain in the current shell. Sourcing this file defines paths only; it
does not activate the Conda environment.

The variables mean:

- `PROJECT_ROOT`: the user's persistent directory inside the Sigma2 project;
- `ENV_ROOT`: the directory holding project-local Conda environments;
- `ENV_PREFIX`: the complete path of the `grid_tdse` Conda environment;
- `SRC_ROOT`: the directory holding the five HyQD source checkouts.

There is no need to define a separate `user` shell variable. Replace
`<username>` in the paths file with the actual Saga username once.

Then create the directories:

```bash
mkdir -p "${ENV_ROOT}"
mkdir -p "${SRC_ROOT}"
mkdir -p "${PROJECT_ROOT}/jobs/grid-tdse"
```

> **Warning:** Do not assign `CONDA_PREFIX` in `grid_tdse_paths.sh` and do not
> prepend the environment to `PATH` manually. `CONDA_PREFIX` and the associated
> `PATH` entries are managed by `conda activate`. Setting only `CONDA_PREFIX`
> can make the shell appear to use the project environment while `python`
> still comes from the Miniconda base installation.

## 3. Load and initialize Conda

First inspect the available modules:

```bash
module spider Miniconda3
module spider Mamba
```

The setup tested for this guide used `Miniconda3/25.7.0-2`. If that version is
no longer available, use a current Miniconda module and adjust the path to
`conda.sh` accordingly.

```bash
module purge
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
source /cluster/projects/<project-id>/<username>/grid_tdse_paths.sh
```

It is normal for `which conda` or `type conda` to report a shell function. The
underlying executable can be inspected with:

```bash
echo "${CONDA_EXE}"
```

## 4. Create the environment

Two supported approaches are described below. Use one approach consistently.

### 4.1 Reproducible pinned environment

Use this approach to reproduce the development environment based on Python
3.14, NumPy 2.5.2, SciPy 1.18.0, and Numba 0.67.0 as closely as possible.

```bash
conda create --prefix "${ENV_PREFIX}" --channel conda-forge \
    python=3.14 \
    pip \
    setuptools \
    wheel \
    numpy=2.5.2 \
    scipy=1.18.0 \
    numba=0.67.0 \
    opt_einsum=3.4.* \
    sympy=1.14.* \
    packaging \
    tqdm \
    matplotlib \
    periodictable=2.1.* \
    pytest \
    pytest-cov
```

`py3nj` is an optional accelerated angular-momentum backend in `grid-lib`.
Install it when a compatible build is available:

```bash
conda install --prefix "${ENV_PREFIX}" --channel conda-forge py3nj
```

If `py3nj` is unavailable, `grid-lib` falls back to its SymPy backend.

The pinned approach is the recommended starting point for reproducible
benchmarks. When dependency bounds change in `pyproject.toml`, update the
pinned environment deliberately rather than running an unrestricted upgrade.

### 4.2 Requirements-driven environment

Use this approach when the current dependency declarations in the HyQD
repositories should control resolution. It is more convenient during active
development but less reproducible over time.

Create only the Python/build-tool foundation initially:

```bash
conda create --prefix "${ENV_PREFIX}" --channel conda-forge \
    python=3.14 \
    pip \
    setuptools \
    wheel
```

After cloning the repositories in step 6, install them in the order shown in
step 7 without `--no-deps`. Pip will resolve the external requirements from
the repositories' current metadata.

> **Clarification:** `grid-tdse`'s `pyproject.toml` is authoritative for its
> supported Python, NumPy, SciPy, Numba, and other runtime-version ranges. A
> requirements-driven installation may therefore change versions when those
> ranges or the package index change. Record the resolved environment if the
> calculation must be reproducible.

## 5. Activate and verify the interpreter

Activate the environment explicitly:

```bash
conda activate base
conda activate "${ENV_PREFIX}"
hash -r
```

Verify the complete interpreter path, not only `CONDA_PREFIX`:

```bash
echo "ENV_PREFIX=${ENV_PREFIX}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-not activated}"
command -v python
python -c 'import sys; print(sys.executable)'
python -m pip --version
```

All Python and pip paths must begin with:

```text
/cluster/projects/<project-id>/<username>/.conda-envs/grid_tdse/
```

For the pinned environment, verify the numerical stack:

```bash
python -c 'import numpy, scipy, numba; print("NumPy:", numpy.__version__); print("SciPy:", scipy.__version__); print("Numba:", numba.__version__)'
```

Expected versions are:

```text
NumPy: 2.5.2
SciPy: 1.18.0
Numba: 0.67.0
```

> **Warning:** A correct-looking `CONDA_PREFIX` is not sufficient. If
> `command -v python` reports `/cluster/software/Miniconda3/.../bin/python`,
> the base interpreter is still active. Re-run the `conda.sh` initialization,
> activate `base`, activate `${ENV_PREFIX}`, and run `hash -r`.

The environment interpreter can always be addressed without activation:

```bash
"${ENV_PREFIX}/bin/python" -c 'import sys; print(sys.executable)'
```

Always prefer `python -m pip` to a bare `pip` command. This guarantees that pip
belongs to the Python interpreter that will run the calculation.

## 6. Configure GitHub access and clone the repositories

Test SSH authentication:

```bash
ssh -T git@github.com
```

GitHub may return a nonzero status even after printing a successful
authentication message because it does not provide shell access.

Clone all five repositories over SSH:

```bash
cd "${SRC_ROOT}"

git clone git@github.com:HyQD/grid-lib.git
git clone git@github.com:HyQD/integrator-lib.git
git clone git@github.com:HyQD/electromagnetic-fields.git
git clone git@github.com:HyQD/matter_field_params.git
git clone git@github.com:HyQD/grid-tdse.git
```

> **Warning:** GitHub does not support account-password authentication for Git
> operations over HTTPS. A private repository cloned with an HTTPS URL will
> require an appropriate token. SSH URLs are recommended for this development
> setup.

If a repository already has an unsuitable remote URL, replace it from within
that repository, for example:

```bash
git remote set-url origin git@github.com:HyQD/grid-tdse.git
```

## 7. Install the repositories in editable mode

Editable installations make imports point directly to the project source
trees. Pulling a pure-Python change therefore updates subsequent Python
processes without copying the package into `site-packages`.

### 7.1 Pinned environment

The external dependencies are already installed by Conda, so install the
local packages without dependency resolution:

```bash
python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/grid-lib"

python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/integrator-lib"

python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/electromagnetic-fields"

python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/matter_field_params"

python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/grid-tdse[dev]"
```

> **Warning:** Use `--no-deps` only after installing the complete external
> numerical stack. Using it in an otherwise empty environment can make
> `grid-tdse` importable while NumPy, SciPy, and its other requirements remain
> absent.

### 7.2 Requirements-driven environment

Allow pip to resolve dependencies from the current package metadata:

```bash
python -m pip install --editable "${SRC_ROOT}/grid-lib[fast]"
python -m pip install --editable "${SRC_ROOT}/integrator-lib"
python -m pip install --editable "${SRC_ROOT}/electromagnetic-fields"
python -m pip install --editable "${SRC_ROOT}/matter_field_params"
python -m pip install --editable "${SRC_ROOT}/grid-tdse[dev]"
```

Install `grid-tdse` last so the four local HyQD requirements are already
registered in the environment.

If the optional `grid-lib[fast]` installation cannot obtain a compatible
`py3nj` build, install `${SRC_ROOT}/grid-lib` without the `[fast]` extra. The
SymPy angular-momentum backend remains available.

## 8. Verify the installation

Check dependency metadata:

```bash
python -m pip check
```

Check the import locations:

```bash
python -c 'import grid_tdse, grid_lib, integrator_lib, electromagnetic_fields, matter_field_params; print(grid_tdse.__file__); print(grid_lib.__file__); print(integrator_lib.__file__); print(electromagnetic_fields.__file__); print(matter_field_params.__file__)'
```

Each path should point into `${SRC_ROOT}`, not the Miniconda base installation
or an unrelated user-site directory.

Inspect editable-install metadata when troubleshooting:

```bash
python -m pip show HyQD-grid-tdse
python -m pip show HyQD-grid-lib
python -m pip show HyQD-integrator-lib
python -m pip show HyQD-electromagnetic-fields
python -m pip show HyQD-matter-field-params
```

Run a short API check:

```bash
python -c 'import inspect; from grid_tdse.spherical.two_radial import NonBornOppenheimerDiatomicIonGroundState as C; assert "radial_coulomb_method" in inspect.signature(C).parameters; print("Updated grid-tdse API found")'
```

## Optional MPI setup on Saga

Distributed calculations require `mpi4py` to be built against the same MPI
implementation loaded by the Slurm job. On Saga, use a cluster Open MPI module
rather than installing Open MPI from Conda.

### Select the cluster MPI implementation

First confirm that the project environment does not already contain `mpi4py`
or a Conda-provided MPI implementation:

```bash
command -v python
python -m pip show mpi4py
conda list | grep -E '^(mpi4py|openmpi|mpich|mpi)[[:space:]]'
```

The Python path must point into `${ENV_PREFIX}`. It is fine for the latter two
commands to report no matching packages at this stage.

Inspect the available Open MPI modules and the prerequisites for the selected
version:

```bash
module spider OpenMPI
module spider OpenMPI/5.0.10-GCC-15.2.0
```

The setup verified for this guide uses the standard Saga CPU environment and:

```bash
StdEnv
OpenMPI/5.0.10-GCC-15.2.0
```

`Zen2Env` is not needed for the normal Saga CPU nodes used here. Record the
chosen Open MPI module in `grid_tdse_paths.sh` by adding:

```bash
export GRID_TDSE_MPI_MODULE="OpenMPI/5.0.10-GCC-15.2.0"
```

Do not add MPI directories to `PATH` or `LD_LIBRARY_PATH` manually. The module
system manages those paths.

### Load Open MPI and activate the Conda environment

Use the following order in an interactive shell and in every MPI job script:

```bash
module --quiet reset
module load StdEnv
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
source /cluster/projects/<project-id>/<username>/grid_tdse_paths.sh
module load "${GRID_TDSE_MPI_MODULE}"

conda activate base
conda activate "${ENV_PREFIX}"
hash -r
```

Confirm that `python` comes from the project environment and `mpicc` comes
from the selected cluster module:

```bash
module list
command -v python
command -v mpicc
mpicc --showme:version
```

The expected paths begin with `${ENV_PREFIX}/bin/python` and
`/cluster/software/OpenMPI/5.0.10-GCC-15.2.0/bin/mpicc`, respectively.

### Build mpi4py against Saga's Open MPI

If `mpi4py` was installed previously against another MPI implementation,
remove it first:

```bash
python -m pip uninstall -y mpi4py
```

Build `mpi4py` from source with the currently loaded `mpicc` wrapper:

```bash
MPICC="$(command -v mpicc)" \
python -m pip install --no-cache-dir --no-binary=mpi4py "mpi4py>=4,<5"
```

Verify the linked library:

```bash
python -c 'import mpi4py; from mpi4py import MPI; print("mpi4py:", mpi4py.__version__); print(MPI.Get_library_version())'
python -m pip check
```

The reported library must be Open MPI 5.0.10. If it reports a Conda Open MPI
or MPICH build, remove `mpi4py`, restore the module setup above, and rebuild
it.

### Test mpi4py through Slurm

MPI programs should run on compute nodes assigned by Slurm, not directly on a
login node. Create
`${PROJECT_ROOT}/jobs/grid-tdse/test_mpi.sh` with:

```bash
#!/bin/bash

#SBATCH --account=<project-id>
#SBATCH --job-name=test-mpi
#SBATCH --qos=devel
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --output=test-mpi-%j.out

set -euo pipefail

project_root="/cluster/projects/<project-id>/<username>"

module --quiet reset
module load StdEnv
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
source "${project_root}/grid_tdse_paths.sh"
module load "${GRID_TDSE_MPI_MODULE}"

conda activate base
conda activate "${ENV_PREFIX}"
hash -r

if [[ "$(command -v python)" != "${ENV_PREFIX}/bin/python" ]]; then
    echo "Failed to activate ${ENV_PREFIX}" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# HCOLL is an optional HPC-X collective component. It is not needed for this
# mpi4py validation and can emit harmless multicast/hierarchy warnings.
export OMPI_MCA_coll="^hcoll"

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Python:       $(command -v python)"
echo "MPI compiler: $(command -v mpicc)"

srun --mpi=pmix \
    python -c 'from mpi4py import MPI; c=MPI.COMM_WORLD; print(f"rank {c.rank} of {c.size}; sum={c.allreduce(c.rank + 1)}", flush=True)'
```

On Saga, `devel` is a quality of service, not a partition. Therefore use
`#SBATCH --qos=devel`, not `#SBATCH --partition=devel`. The normal CPU
partition is selected by default.

Submit and inspect the test with:

```bash
cd "${PROJECT_ROOT}/jobs/grid-tdse"
sbatch test_mpi.sh
squeue --me
cat test-mpi-<job-id>.out
```

The rank lines may appear in either order, but both ranks must report
`sum=3`. The job must exit normally.

Load the same `GRID_TDSE_MPI_MODULE` in every subsequent MPI batch job. Use
`srun`, not `mpiexec` or `mpirun`, on Saga. For Open MPI, Sigma2 recommends:

```bash
srun --mpi=pmix python mpi_program.py
```

Do not use the laptop-only option `--mca btl self,sm` on Saga. It restricts
Open MPI to local shared-memory communication and is unsuitable for
multi-node jobs.

If HCOLL emits messages such as `No MCAST components selected` or
`hcoll_mcast_base_select`, the MPI program can still complete using fallback
collectives. Keep `OMPI_MCA_coll="^hcoll"` in the job scripts used for these
benchmarks to disable that optional component and avoid the warnings.

Ready-to-edit serial and MPI job scripts, together with a 1-, 2-, 4-, and
8-rank submission helper, are provided in
[`examples/Saga`](../examples/Saga/README.md).

## 9. Run the test suite through Slurm

Do not run sustained computations on a login node. Create
`${PROJECT_ROOT}/jobs/grid-tdse/job_grid_tdse_tests.sh`:

```bash
#!/bin/bash

#SBATCH --job-name=grid_tdse_tests
#SBATCH --account=<project-id>
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=grid_tdse_tests-%j.out

set -euo pipefail

module purge
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
source "/cluster/projects/<project-id>/<username>/grid_tdse_paths.sh"
conda activate base
conda activate "${ENV_PREFIX}"
hash -r

if [[ "$(command -v python)" != "${ENV_PREFIX}/bin/python" ]]; then
    echo "Failed to activate the grid_tdse environment" >&2
    exit 1
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
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores python -m pytest
```

Submit it with:

```bash
cd "${PROJECT_ROOT}/jobs/grid-tdse"
sbatch job_grid_tdse_tests.sh
```

The output file is created in the submission directory. The directory must
exist before `sbatch` is called because Slurm opens the output file before the
script body runs.

## 10. Record the resolved environment

For an exact platform-specific Conda record:

```bash
conda list --explicit > "${PROJECT_ROOT}/grid_tdse-conda-explicit.txt"
```

For a human-readable Conda history:

```bash
conda env export --from-history > "${PROJECT_ROOT}/grid_tdse-conda-history.yml"
```

Record pip/editable metadata as well:

```bash
python -m pip freeze > "${PROJECT_ROOT}/grid_tdse-pip-freeze.txt"
```

These records are especially important for the requirements-driven setup.

## Updating an existing installation

### 11. Update only grid-tdse

Activate and verify the environment first:

```bash
module purge
module load Miniconda3/25.7.0-2
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
source /cluster/projects/<project-id>/<username>/grid_tdse_paths.sh
conda activate base
conda activate "${ENV_PREFIX}"
hash -r

command -v python
python -c 'import sys; print(sys.executable)'
```

Update the checkout:

```bash
cd "${SRC_ROOT}/grid-tdse"
git status --short --branch
git switch main
git pull --ff-only origin main
```

Review any tracked local modifications before pulling. Do not use
`git reset --hard` or discard files merely to make the update succeed. A
feature branch should be merged before updating `main`, or checked out
explicitly when testing that branch.

For a pure-Python source change, an existing editable installation uses the
new code in the next Python process. Re-run the editable installation anyway
when `pyproject.toml`, entry points, package layout, or dependency metadata may
have changed:

```bash
python -m pip install --no-deps --no-build-isolation --editable .
```

Then verify and test:

```bash
python -m pip check
python -c 'import grid_tdse; print(grid_tdse.__file__)'

cd "${PROJECT_ROOT}/jobs/grid-tdse"
sbatch job_grid_tdse_tests.sh
```

### 12. Update all HyQD libraries

Update shared libraries before `grid-tdse`, which consumes them. In each
repository, inspect the working tree and then fast-forward `main`:

```bash
cd "${SRC_ROOT}/grid-lib"
git status --short --branch
git switch main
git pull --ff-only origin main

cd "${SRC_ROOT}/integrator-lib"
git status --short --branch
git switch main
git pull --ff-only origin main

cd "${SRC_ROOT}/electromagnetic-fields"
git status --short --branch
git switch main
git pull --ff-only origin main

cd "${SRC_ROOT}/matter_field_params"
git status --short --branch
git switch main
git pull --ff-only origin main

cd "${SRC_ROOT}/grid-tdse"
git status --short --branch
git switch main
git pull --ff-only origin main
```

Refresh all editable metadata in the same order:

```bash
python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/grid-lib"
python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/integrator-lib"
python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/electromagnetic-fields"
python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/matter_field_params"
python -m pip install --no-deps --no-build-isolation \
    --editable "${SRC_ROOT}/grid-tdse[dev]"
```

For a requirements-driven environment, omit `--no-deps` so pip can reconcile
new dependency metadata. For a pinned environment, inspect changed version
bounds and update the corresponding Conda packages deliberately before using
`--no-deps`.

Finally run:

```bash
python -m pip check
python -c 'import grid_tdse, grid_lib, integrator_lib, electromagnetic_fields, matter_field_params; print("All HyQD imports succeeded")'
```

Submit the Slurm test job after every coordinated update.

### 13. Common problems

#### `CONDA_PREFIX` is correct but `python` is from Miniconda base

The environment variable was set without a corresponding `PATH` update, or
the shell retained stale command lookup state. Do not set `CONDA_PREFIX`
manually. Reinitialize and activate:

```bash
source /cluster/software/Miniconda3/25.7.0-2/etc/profile.d/conda.sh
conda activate base
conda activate "${ENV_PREFIX}"
hash -r
command -v python
```

#### `pip check` says every dependency is missing

This almost always means `python -m pip` is running under the wrong Python.
Check:

```bash
python -c 'import sys; print(sys.executable)'
python -m pip --version
```

Do not install a second numerical stack into Miniconda base. Activate the
project environment and repeat the check.

#### A local HyQD import resolves to the wrong package

Print the module path:

```bash
python -c 'import matter_field_params; print(matter_field_params.__file__)'
```

Reinstall the intended checkout in editable mode with the environment's
Python. Restart Python afterward; an already-running interpreter does not
reload pulled source automatically.

#### GitHub rejects an HTTPS password

Use an SSH remote or a supported token. For SSH:

```bash
git remote set-url origin git@github.com:HyQD/grid-tdse.git
```

#### A batch job cannot find `grid_tdse_paths.sh`

Use the project path, not a home-directory path:

```bash
source /cluster/projects/<project-id>/<username>/grid_tdse_paths.sh
```

#### `LOCALSCRATCH` is unset

Do not assume a site-specific scratch variable exists. Use a fallback such as:

```bash
cache_root="${TMPDIR:-/tmp}/grid-tdse-${SLURM_JOB_ID}"
```

#### Matplotlib reports that its configuration directory is not writable

Set a writable, job-local configuration directory before importing packages
that may import Matplotlib:

```bash
cache_root="${TMPDIR:-/tmp}/grid-tdse-${SLURM_JOB_ID}"
mkdir -p "${cache_root}/matplotlib"
export MPLCONFIGDIR="${cache_root}/matplotlib"
```

#### Python reports invalid escape sequences in docstrings

A `SyntaxWarning` about an invalid escape such as `\i` or `\P` usually comes
from a mathematical expression in a Python docstring. It is not evidence that
the environment is broken, although the upstream docstring should be changed
to a raw string or use escaped backslashes.

#### Python 3.14 reports that py3nj enables the GIL

The compiled `py3nj` extension may not declare that it is safe under
free-threaded Python. Python then enables the global interpreter lock while
loading it and emits a `RuntimeWarning`. This is not an import failure. Do not
set `PYTHON_GIL=0` merely to suppress the warning; free-threaded execution
should only be enabled after all compiled dependencies have been validated.

#### CPU utilization exceeds 100 percent

Multiple numerical thread pools are probably oversubscribing the allocation.
Choose one parallel layer, make the other libraries single-threaded, and bind
the Slurm step. For the Numba-parallel Coulomb kernel:

```bash
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    python calculation.py
```

Small symmetry-reduced calculations may still use only one or two cores
effectively. Benchmark `--cpus-per-task=1`, `2`, `4`, and `8` rather than
assuming that more CPUs reduce wall time.

#### Memory usage is much lower than the request

Slurm resource requests should be based on measured peak usage with a safety
margin. Excess memory requests may increase queueing or billing. Increase the
request again when moving to larger radial grids, larger angular bases, or
full angular storage.

### 14. Related Sigma2 documentation

- [Saga machine overview](https://documentation.sigma2.no/hpc_machines/saga.html)
- [Storage areas on HPC clusters](https://documentation.sigma2.no/files_storage/clusters.html)
- [Software modules](https://documentation.sigma2.no/software/modulescheme.html)
- [Installing software as a user](https://documentation.sigma2.no/software/userinstallsw.html)
- [Running jobs](https://documentation.sigma2.no/jobs/overview.html)
