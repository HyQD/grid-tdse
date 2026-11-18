# Rotation sensitivity of atomic-center matrix blocks

This diagnostic tests the hypothesis raised by the midpoint-basis experiment:
an A-centered angular grid is aligned with the radial FEM elements of basis A,
but cuts through the element surfaces of basis B.  The cross-center blocks
should therefore be more sensitive to Lebedev rotation than same-center
blocks.

The fixed H2+ calculation uses the default atom-centered basis (`R = 2` bohr,
`r_max = 8` bohr, eight radial basis points per element, and `l_max = 2`).
The unpruned native quadrature uses integration spacing `0.25` bohr, 14 radial
points per subinterval, and Lebedev order 59.  Four proper rotations of the
same angular rule were compared.

Using the retained metric subspace from rotation zero, the energy shifts were

| Rotation | All matrices | Overlap only | Potential only | Kinetic only |
|---:|---:|---:|---:|---:|
| 1 | -0.043 | -0.094 | +0.119 | -0.055 |
| 2 | +0.587 | +0.054 | -0.084 | +0.599 |
| 3 | -0.070 | -0.048 | +0.057 | -0.057 |

All values are in microhartree relative to rotation zero.  Unlike the
midpoint-basis experiment, the complete atom-centered energy is already
stable to better than one microhartree at these deliberately large quadrature
settings.  No single matrix component dominates every rotation.

The relative Frobenius changes in the kinetic blocks were

| Rotation | Same-center blocks | Cross-center blocks |
|---:|---:|---:|
| 1 | 3.63e-3 | 6.17e-2 |
| 2 | 4.80e-3 | 4.64e-2 |
| 3 | 4.22e-3 | 4.02e-2 |

The A--B kinetic block is about an order of magnitude more rotation-sensitive
than the A--A and B--B blocks.  This supports the element-boundary hypothesis.
The block changes cannot be converted independently into reliable energy
contributions: the atom-centered basis is strongly nonorthogonal, and
replacing only one block destroys correlated cancellations between the
blocks.

The result suggests two separate requirements:

1. Cross-center kinetic integration needs element-aware quadrature, angular
   adaptivity near foreign FEM surfaces, or a radially smoother basis.
2. Basis localization is needed so that small matrix errors are not amplified
   by asymptotic near-linear dependencies.

Run with

```console
conda run -n haakon_2026 \
  python examples/one_radial_atomic_rotation_components.py
```

The script is also the canonical small benchmark for this diagnostic. It uses
the cheaper default quadrature (10 radial integration points and Lebedev order
41), for which each assembly takes about 1.2 seconds on the development
machine. The unrotated energy error is only +0.203 microhartree, but the fixed
rotations move the energy by as much as 7.220 microhartree. The mean ratio of
cross-center to same-center kinetic sensitivity is 15.92. This exposes both
the favorable unrotated value and its hidden quadrature uncertainty in about
five seconds.

The benchmark prints its full fixed configuration, reference-energy error,
basis and grid sizes, overlap condition number, assembly time for every
rotation, matrix-component shifts, and block sensitivity. Its settings should
remain fixed when comparing changes to the radial basis or matrix assembly.
The larger order-59 calculation documented above is retained as a separate
confirmation that the complete energy envelope eventually falls below one
microhartree while the A--B kinetic block remains the most sensitive block.
