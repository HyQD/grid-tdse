# C1 B-spline radial prototype

## Purpose

This prototype tests one narrow hypothesis: the anomalous Lebedev-rotation
sensitivity of cross-center kinetic matrix elements is caused by C0 radial
FEM element boundaries from one center cutting through the quadrature shells
of another center.

The implementation adds `radial_basis="bspline-c1"` to
`solve_atomic_h2plus()`. It changes only the radial basis evaluator. Becke
weights, radial integration, Lebedev grids, point grouping, matrix assembly,
and the generalized eigensolver are unchanged. The existing FEM-DVR basis
remains the default.

The experimental basis is an open clamped B-spline basis. Interior knots have
multiplicity `degree - 1`, which gives C1 continuity. The first and last
B-splines are removed to impose the same zero endpoint values as the finite
FEM-DVR basis. Degree 7 gives 36 radial functions per center, compared with
41 for the default FEM-DVR basis.

## Default benchmark

Run both bases with

```console
conda run -n haakon_2026 \
  python examples/one_radial_atomic_rotation_components.py \
  --radial-basis both
```

The fixed calculation uses `R = 2` bohr, `r_max = 8` bohr, `l_max = 2`, ten
radial integration points, and Lebedev order 41.

| Quantity | C0 FEM-DVR | C1 B-spline |
|---|---:|---:|
| Total basis size | 246 | 216 |
| Grid points | 470820 | 470820 |
| Overlap condition number | 4.81e7 | 3.91e7 |
| Unrotated energy error | +0.203 microhartree | +3.378 microhartree |
| Maximum rotated energy shift | 7.220 microhartree | 2.000 microhartree |
| Mean cross/same kinetic sensitivity | 15.92 | 6.59 |
| Largest relative cross-center T change | 1.28e-1 | 2.87e-3 |
| Assembly time per rotation | about 1.2 s | about 1.3 s |

The largest relative change in the A--B kinetic block is reduced by roughly a
factor of 40. The total rotation envelope is reduced by a factor of 3.6.

## Larger quadrature

With 14 radial integration points and Lebedev order 59:

| Quantity | C0 FEM-DVR | C1 B-spline |
|---|---:|---:|
| Unrotated energy error | +2.283 microhartree | +3.437 microhartree |
| Maximum rotated energy shift | 0.587 microhartree | 0.641 microhartree |
| Mean cross/same kinetic sensitivity | 12.06 | 4.32 |
| Largest relative cross-center T change | 6.17e-2 | 1.74e-3 |

The C1 basis still reduces the cross-center kinetic matrix sensitivity by
more than an order of magnitude, but the total energy envelope is now similar
for the two bases. This is consistent with correlated error cancellation in
the nonorthogonal atom-centered representation: a more stable individual
matrix block does not guarantee that one finite-grid total energy is closer
to the reference.

## Interpretation

The prototype supports the smoothness hypothesis. Removing radial gradient
jumps strongly stabilizes the A--B kinetic block without changing the grid.
The test does not yet show a uniformly better total energy because the first
spline basis has a different and slightly smaller radial variational space.
Its unrotated error is about 3 microhartree and changes slowly with spline
degree, so basis convergence and quadrature convergence are not yet cleanly
separated.

This is therefore a useful positive diagnostic, not a production replacement
for FEM-DVR. The next controlled experiment should converge the C1 radial
space at fixed quadrature, then repeat the rotation envelope at increasing
Lebedev order. Origin regularity should also be made angular-momentum aware
before using a more flexible radial space in production.

## Implementation boundary

The prototype confirms the feasibility assessment:

- the Becke and PySCF grid paths both work unchanged;
- the grouped dense assembly accepts the wider local B-spline support;
- the C0 implementation remains the default and its regression tests pass;
- C1 continuity and endpoint constraints have focused unit tests.

The remaining work is mathematical convergence and production design, not a
large grid-code refactor.
