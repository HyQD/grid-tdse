# Midpoint basis on a multicenter Becke grid

This diagnostic holds the original midpoint-centered FEM-DVR basis fixed and
changes only the matrix integration to the atom-centered Becke grid.  It is
therefore independent of the redundancy in the new atom-centered basis.

The fixed H2+ calculation uses `R = 2` bohr, midpoint radial boundaries
`(0, 1, ..., 8)`, eight Lobatto basis points per element, and `l_max = 2`.
The original structured solver gives

```text
E = -1.083680184786 Eh
```

With integration spacing `0.25` bohr and 16 radial integration points per
subinterval, changing only the Lebedev order gives

| Lebedev order | Points | Becke minus structured energy (microhartree) |
|---:|---:|---:|
| 29 | 347300 | -17.925 |
| 35 | 499100 | -2.998 |
| 41 | 678500 | -1.621 |
| 47 | 885500 | -5.414 |
| 53 | 1120100 | -5.477 |
| 59 | 1382300 | -0.740 |
| 65 | 1672100 | -0.754 |
| 71 | 1989500 | -2.225 |
| 77 | 2334500 | -4.765 |
| 83 | 2707100 | -1.358 |

The angular sequence is non-monotonic even though the basis is unchanged.
At Lebedev order 59, four proper rotations of the same angular rule gave
energy differences between `-0.740` and `-8.190` microhartree.  At order 83,
three rotations gave `-3.061`, `-0.740`, and `-0.332` microhartree.

The absolute difference from the structured solver also contains the change
from its DVR mass-lumped matrix integration to continuum-like numerical
integration.  Rotation sensitivity at otherwise identical settings does not:
it is a direct measure of angular quadrature error.

## Matrix-component diagnosis

The order-59 rotation experiment was repeated while replacing only one matrix
component at a time.  Rotation zero is the existing generic rotation.  For
each other rotation, the table reports the ground-state shift obtained with
all rotated matrices and with only the rotated overlap, kinetic, or potential
matrix combined with the rotation-zero matrices.

| Rotation | All matrices | Overlap only | Kinetic only | Potential only |
|---:|---:|---:|---:|---:|
| 1 | -2.949 | -0.035 | -2.963 | +0.033 |
| 2 | -1.755 | -0.091 | -1.766 | +0.015 |
| 3 | -7.450 | -0.009 | -7.449 | +0.014 |

All values are in microhartree relative to rotation zero.  The rotation
sensitivity is almost entirely in the kinetic matrix.  The nuclear potential
is stable at a few hundredths of a microhartree, so the displaced Coulomb
singularity is not the dominant error in this experiment.  Pruning is also
excluded: all grids in this diagnostic are unpruned.

The midpoint FEM-DVR basis is continuous across radial finite-element
boundaries, but its gradient is element-local and is generally discontinuous
there.  Atom-centered quadrature shells cut through the midpoint-centered
element surfaces.  Rotating the Lebedev directions changes how these
piecewise-smooth kinetic-energy integrands are sampled, explaining the much
larger sensitivity of the gradient term.

The result demonstrates that irregular microhartree behavior exists when the
original one-center basis is integrated on the multicenter Becke grid.  It is
therefore not caused solely by redundancy in the atom-centered basis.  In the
atom-centered representation, an A-centered integration grid is aligned with
the radial elements of the A basis but not with those of the B basis.  The
next diagnostic should split the atom-centered kinetic matrix into A--A,
B--B, and A--B blocks under grid rotation.  The A--B block is the leading
candidate for the remaining angular sensitivity.

Possible remedies are to assemble same-center kinetic blocks with the native
FEM weak form, use element-aware or adaptive quadrature for cross-center
blocks, or use a radially smoother basis.  Angular pruning cannot repair the
loss of smoothness and should remain disabled while this is diagnosed.

Run the diagnostic with

```console
conda run -n haakon_2026 \
  python examples/one_radial_midpoint_becke_quadrature.py

conda run -n haakon_2026 \
  python examples/one_radial_midpoint_rotation_components.py
```
