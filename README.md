# grid-tdse

`grid-tdse` provides direct grid-based tools for solving few-body
time-dependent Schrödinger equations,

$$
i\frac{\partial}{\partial t}\Psi(\mathbf r_1,\ldots,\mathbf r_N;t)
=\hat H(t)\Psi(\mathbf r_1,\ldots,\mathbf r_N;t).
$$

The package owns coordinate-specific wavefunction representations,
Hamiltonians, observables, and calculation interfaces. Spatial grids and their
numerical operators come from `grid-lib`; coordinate-independent time
integrators come from `integrator-lib`; temporal electromagnetic fields come
from `electromagnetic-fields`; and particle, isotope, and charge data come from
`matter-field-params`.

The project is under active development. Its public API may change while the
initial one- and two-coordinate solvers are organized and validated.

## Documentation

- [Documentation overview](docs/index.md)
- [Runtime dependencies](docs/runtime_dependencies.md)
- [Installation](docs/installation.md)
- [Two-radial spherical foundation](docs/two_radial_foundation.md)
- [Imaginary-time ground-state iteration](docs/ground_state_iteration.md)
- [Non-Born--Oppenheimer diatomic-ion ground states](docs/non_bo_diatomic_ground_state.md)
- [Two-electron atomic ground states](docs/two_electron_atomic_ground_state.md)

Feature documentation will be added alongside each implemented and tested
solver.

## Installation

Install the package and its runtime dependencies:

```bash
python -m pip install .
```

For development, install it in editable mode with the test dependencies:

```bash
python -m pip install -e ".[dev]"
```
