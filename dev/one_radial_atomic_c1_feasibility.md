# Feasibility of a C1 radial basis in `one_radial_atomic`

> **Implementation status:** A first experimental C1 B-spline basis has now
> been implemented. See `dev/one_radial_atomic_c1_prototype.md` for measured
> results. The feasibility discussion below records the design reasoning that
> preceded the prototype.

## Conclusion

A C1 prototype is a local-to-moderate change, not a rewrite of the Becke
grid. The current matrix assembly already consumes arbitrary basis values,
gradients, and active global indices at each group of points. The radial
basis evaluator is the part that assumes C0 FEM-DVR cardinal functions.

The implemented prototype uses degree-7 B-splines because they give a radial
basis dimension close to the current high-order FEM-DVR basis on the same
element boundaries. A piecewise Hermite basis remains a simpler low-order
alternative if a more local diagnostic is needed later.

## What is coupled to C0 FEM-DVR today

`_RadialBasis` in `one_radial_atomic/h2plus.py` constructs `grid-lib`'s
`FEMDVR`. The relevant assumptions are:

1. Adjacent elements share one value degree of freedom at their common
   endpoint. This makes the radial function C0 but generally not C1.
2. The first and last global nodes are removed to impose the finite radial
   boundary conditions.
3. `evaluate_element()` evaluates local values and first derivatives and
   returns their global basis indices.
4. `_local_basis_values()` combines those radial quantities with spherical
   harmonics and Cartesian angular gradients.

Only items 1-3 are basis-specific. The Becke partition, radial integration
rule, Lebedev rule, point grouping, local dense products, generalized
eigensolver, and rotation diagnostic do not require a DVR basis.

The current `grid-lib.FEMDVR` cannot be changed from C0 to C1 by an option. Its
interface matrix identifies the endpoint value from adjacent elements, but
there is no shared derivative degree of freedom. It does, however, expose the
element topology and weak-form matrices needed by the existing C0 method.

## Minimal refactoring boundary

Extract a radial-basis interface with the following information:

```text
size
r_max
locate(radii) -> interval indices
evaluate_interval(interval, radii) -> values, first derivatives, global indices
```

The current `_RadialBasis` becomes the C0 implementation of this interface.
`_local_basis_values()` and the grouped matrix assembly can remain unchanged.
The basis specification and integration specification must remain separate;
both may use the same radial breakpoints without sharing basis functions.

This extraction is useful even if the C1 experiment fails, because it makes
the present basis/quadrature separation explicit in the code.

## Recommended prototype: Hermite elements

A cubic Hermite element has value and derivative degrees of freedom at each
endpoint. Sharing both quantities between neighboring elements gives a C1
radial function with compact two-element support around an interior node.
Its values and first derivatives are analytic and cheap to evaluate at
arbitrary atom-centered grid points.

For the first diagnostic:

- retain the current radial element boundaries and radial extent;
- impose the same homogeneous outer value condition;
- handle regularity at the origin explicitly for the reduced radial function
  `u_l(r)`;
- use the existing Gauss-Legendre radial integration and Lebedev grids;
- compare basis sizes at similar stationary H2+ accuracy, rather than at the
  same number of polynomial degrees of freedom.

Origin conditions deserve care. The present basis removes the endpoint at
`r=0`, which enforces `u_l(0)=0`. A Hermite basis also has an endpoint
derivative degree of freedom. Its admissible value depends on angular
momentum and should be constrained consistently with
`u_l(r) ~ r^(l+1)`. This is the only part that is not a purely mechanical
replacement of the C0 evaluator.

## B-spline alternative

B-splines provide controllable polynomial degree and continuity. Degree
`p >= 3` with suitable knot multiplicities gives at least C1 continuity.
They have compact support over several radial intervals, so the active-index
list is slightly wider than for FEM-DVR or cubic Hermite elements, but the
existing grouped assembly can still use it.

A B-spline implementation requires knot construction, origin and outer
boundary constraints, values and derivatives at arbitrary radii, and local
support lookup. It is therefore somewhat more infrastructure than the
Hermite diagnostic, but likely the more flexible long-term representation.

## What should not change in the first experiment

- Becke partitioning and Becke weights;
- the native and PySCF grid adapters;
- the Lebedev rotations used by the benchmark;
- the generalized overlap treatment and cutoff;
- spatial basis localization or outer-region partitioning.

Holding these fixed ensures that a reduction in rotation sensitivity can be
attributed to radial smoothness.

## Acceptance test

Run `examples/one_radial_atomic_rotation_components.py` for the C0 reference
and the C1 prototype, then repeat at several Lebedev orders. Record:

1. absolute H2+ energy error;
2. maximum energy change under the fixed rotations;
3. relative changes in same-center and cross-center kinetic blocks;
4. overlap condition number;
5. basis size, grid size, and assembly time.

The C1 route is promising if the cross-center kinetic variation and total
rotation spread decrease regularly with Lebedev order without a damaging
increase in basis size or overlap conditioning. Monotonicity of every finite
grid value is not required, but the rotation envelope should provide a clear
convergence axis.

## Estimated implementation scope

For a self-contained H2+ diagnostic, the change is small-to-moderate:

- extract the radial interface and retain the current C0 adapter;
- add one Hermite evaluator and focused unit tests;
- parameterize `solve_atomic_h2plus()` by radial-basis type;
- run the existing benchmark unchanged apart from selecting the basis.

Moving the successful basis into `grid-lib`, generalizing all angular
momenta, and optimizing propagation would be a later production step. The
prototype should first answer the narrow numerical question: does C1 radial
smoothness remove the anomalous rotation sensitivity of the A-B kinetic
blocks?
