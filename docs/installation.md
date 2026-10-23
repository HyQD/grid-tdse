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
