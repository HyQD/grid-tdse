# Installation

`grid-tdse` requires Python 3.12 through 3.14. Install the package and its
declared runtime dependencies with:

```bash
python -m pip install .
```

For development, install the repository in editable mode with its test
dependencies:

```bash
python -m pip install -e ".[dev]"
```

Install the optional MPI dependency for distributed calculations with:

```bash
python -m pip install -e ".[mpi]"
```

The `mpi4py` installation must match the MPI implementation used to launch
the calculation. On an HPC system, load the intended compiler and MPI modules
before installing this optional dependency.

When developing several HyQD repositories together, install their local
checkouts in editable mode before installing `grid-tdse`. For example, adjust
the paths to match the workspace and run:

```bash
python -m pip install -e ../grid-lib
python -m pip install -e ../electromagnetic-fields
python -m pip install -e ../matter_field_params
python -m pip install -e ../INTEGRATOR-LIB/integrator-lib
python -m pip install -e ".[dev]"
```

Run the test suite from the repository root:

```bash
python -m pytest
```

For a project-local development environment on Saga, including cloning,
editable installation, Slurm testing, troubleshooting, and update procedures,
see [Setting up grid-tdse on Saga](saga_setup.md).
