# PySCF Becke grid with a user-owned radial rule

The optional PySCF backend uses PySCF for atom-centered Lebedev grids,
angular pruning, Becke partitioning, and spatial grouping.  The radial points
and weights remain completely controlled by `grid-tdse`.

Install the optional dependency and run the H2+ example with

```console
python -m pip install -e '.[pyscf]'
python examples/one_radial_atomic_h2plus_pyscf.py
```

The general interface is

```python
from grid_tdse.spherical.one_radial_atomic import build_pyscf_becke_grid

def radial_rule(atom_index, atomic_number):
    # Any positive physical radii and one-dimensional quadrature weights.
    # The number of points may differ between atoms.
    return radii[atom_index], radial_weights[atom_index]

grid = build_pyscf_becke_grid(
    centers,
    atomic_numbers,
    radial_rule,
    angular_points=590,
    prune="nwchem",
)

coordinates = grid.coordinates
weights = grid.weights
```

The callback owns its output length.  PySCF's nominal radial count is not part
of the callback contract.  Unique internal atom labels ensure that the radial
callback is evaluated separately for atoms of the same element.

Two radial helpers are included:

- `composite_gauss_legendre` uses interior points in specified finite-element
  intervals.  It is useful when FEM basis gradients are evaluated from only
  the Cartesian coordinates because no point lies on a shared boundary.
- `fem_lobatto` reproduces the element-local radial rule used by the existing
  prototype, including duplicate internal boundaries and excluding the
  zero-volume origin.

For the current H2+ basis at `R = 2` bohr, the example uses radial integration
spacing `0.5`, 12 Gauss-Legendre points per subinterval, Lebedev order 47, and
NWChem pruning.  A representative run gives approximately

```text
electronic energy       -1.102633801 Eh
reference error           0.413 uEh
quadrature points        295224
assembly time             0.75 s
```

Timings vary between runs.  The corresponding native unpruned calculation has
368060 points and an error around 0.27 microhartree.  The main result is that
the PySCF route already preserves sub-microhartree accuracy while reducing the
grid by about 20 percent.  More aggressive pruning should be validated for
strong-field and long-range states rather than inherited blindly from DFT.
