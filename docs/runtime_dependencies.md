# Runtime dependencies

`grid-tdse` uses the shared HyQD packages for infrastructure that is not
specific to a TDSE representation:

- [`grid-lib`](https://github.com/HyQD/grid-lib) supplies spatial grids,
  quadrature rules, differentiation matrices, spherical-coordinate angular
  algebra, and grid-based Coulomb utilities.
- `integrator-lib` supplies coordinate-independent real- and imaginary-time
  integration algorithms.
- `electromagnetic-fields` supplies temporal electromagnetic-field profiles
  and their gauge conventions.
- `matter-field-params` supplies particle charges, isotope masses, reduced
  masses, and related physical parameters.

The numerical runtime also uses NumPy, SciPy, opt-einsum, Numba, SymPy,
Packaging, tqdm, and Matplotlib. Supported Python and numerical-library
versions are declared in [`pyproject.toml`](../pyproject.toml), which is the
authoritative dependency specification.

Distributed calculations additionally require `mpi4py` and a compatible MPI
implementation. This dependency is optional so that serial installations do
not require an MPI runtime.

Spatial discretizations should be implemented in `grid-lib` and consumed by
`grid-tdse`. General-purpose integration algorithms should similarly be added
to `integrator-lib`; `grid-tdse` is responsible for adapting these components
to its wavefunction representations and Hamiltonians.
