# Two-radial state archives

Ground-state results can be stored in a self-describing NumPy archive for
later inspection or use as the initial state of a time-dependent calculation.
The user supplies the complete output path:

```python
result = calculation.solve()
result.save("my_ground_state", overwrite=False)
```

`grid-tdse` uses the path exactly as given and does not add an extension. An
existing file is protected unless `overwrite=True` is requested explicitly.

The archive contains the wavefunction coefficients, both radial grids and
quadrature weights, the ordered angular state tables, the grid and angular
specifications, the Coulomb-multipole truncation, physical parameters and
symmetries, and ground-state convergence diagnostics. Representation and
coefficient checksums detect inconsistent or modified archive contents.
Python objects are not pickled, and loading always uses
`numpy.load(..., allow_pickle=False)`.

Load an archive with:

```python
from grid_tdse.spherical.two_radial import load_two_radial_state

stored = load_two_radial_state("my_ground_state")
psi = stored.coefficients
```

The typed grid, angular-basis, and Coulomb-multipole specifications are
available as `grid1_specification`, `grid2_specification`, `basis1`, `basis2`,
and `multipoles`.

## Compatibility and embedding

An exact restart requires the stored and target representations to have the
same radial grid specifications, realized points, quadrature weights, and
ordered angular state tables:

```python
psi = stored.state_for(target_representation)
```

For field-driven calculations, a symmetry-reduced ground state may instead
be embedded into a larger angular basis:

```python
psi = stored.embed_into(target_representation)
```

Every stored `(l, m)` channel must occur in the target. Matching components
are copied without modification and additional target channels are initialized
to zero. Both radial grids must still match exactly, so embedding preserves the
two-radial quadrature norm.

No radial interpolation or projection is performed implicitly. Moving a state
to a different radial grid is a separate numerical operation and is rejected
by the archive compatibility checks.

The lower-level `save_two_radial_state` function is available for custom
states. It requires a prepared `TwoRadialSystem` and an explicit physical-state
description so the resulting file retains the information needed to interpret
the coefficients.
