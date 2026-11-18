# Smooth B-spline basis convergence

## Target notion of systematic convergence

The desired numerical hierarchy has two separate limits:

1. At sufficiently converged quadrature, increasing the basis should lower
   the variational energy towards a basis-set limit.
2. For every fixed basis, increasing the quadrature should make the energy
   independent of the Lebedev orientation.

The earlier C1 prototype greatly reduced the second error, but richer C1
bases again required expensive angular grids. B-splines allow the continuity
at radial knots to be raised without changing the Becke or PySCF grid code.

## Continuity sweep

The spline degree and continuity were increased together so that all four
bases had similar dimensions. The unpruned PySCF grid used Lebedev order 65
and four fixed proper rotations.

| Degree | Continuity | Basis size | Retained rank | cond(S) | Energy error | Rotation envelope |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | C1 | 216 | 216 | 3.91e7 | +3.386 microhartree | 0.112 microhartree |
| 8 | C2 | 222 | 222 | 3.78e7 | +3.458 microhartree | 0.055 microhartree |
| 9 | C3 | 228 | 228 | 3.60e7 | +3.463 microhartree | 0.015 microhartree |
| 10 | C4 | 234 | 234 | 3.41e7 | +3.467 microhartree | 0.002 microhartree |

Higher continuity removes the remaining orientation sensitivity very
efficiently. C4 is effectively quadrature-converged at this grid for the
tested basis size. This clarifies the terminology: a B-spline is not
intrinsically C1. Its continuity is selected through the interior knot
multiplicity; C1 was only the first pilot choice.

Run with

```console
conda run -n haakon_2026 \
  python examples/one_radial_atomic_spline_continuity.py
```

## Radial basis increase at fixed C4

Increasing the spline degree at fixed C4 gives

| Degree | Basis size | Retained rank | cond(S) | Energy error | Rotation envelope |
|---:|---:|---:|---:|---:|---:|
| 6 | 90 | 90 | 2.47e7 | +3.498 microhartree | 0.000 microhartree |
| 8 | 162 | 162 | 3.04e7 | +3.467 microhartree | 0.000 microhartree |
| 10 | 234 | 234 | 3.41e7 | +3.467 microhartree | 0.002 microhartree |
| 12 | 306 | 306 | 3.73e7 | +3.456 microhartree | 0.055 microhartree |
| 14 | 378 | 378 | 8.63e7 | +3.323 microhartree | 0.688 microhartree |

The radial space is already effectively converged by degree 8 for
`l_max = 2`. At degrees 12 and 14 the fixed Lebedev-65 grid is no longer
converged, illustrating that basis and quadrature still have to be increased
in a nested way. No radial functions are removed by the overlap cutoff in
this sweep.

Run the C4 sweep above with

```console
conda run -n haakon_2026 \
  python examples/one_radial_atomic_c1_basis_convergence.py \
  --continuity 4 --degrees 6 8 10 12 14
```

## Angular basis increase at fixed smooth radial basis

The remaining +3.47 microhartree error is therefore angular-basis error. With
the degree-8 C4 radial basis fixed:

| l_max | Basis size | Retained rank | cond(S) | Energy error | Rotation envelope |
|---:|---:|---:|---:|---:|---:|
| 2 | 162 | 162 | 3.04e7 | +3.467 microhartree | 0.000 microhartree |
| 3 | 216 | 214 | 7.54e10 | +0.185 microhartree | 0.001 microhartree |
| 4 | 270 | 264 | 1.51e14 | +0.122 microhartree | 0.000 microhartree |
| 5 | 324 | 313 | 1.03e16 | +0.119 microhartree | 0.000 microhartree |

The energy now decreases systematically while the quadrature remains
rotation-invariant. At the same time, the overlap matrix becomes rapidly
singular and the metric cutoff removes 2, 6, and 11 functions. This is the
previously identified atom-centered redundancy appearing directly in the
basis convergence sequence.

Run with

```console
conda run -n haakon_2026 \
  python examples/one_radial_atomic_c4_angular_basis_convergence.py
```

## Conclusion

The pilot now demonstrates the intended separation:

- smooth B-splines, especially C3-C4, make the tested Becke/PySCF quadrature
  effectively converged at moderate angular order;
- increasing the radial and angular basis lowers the basis error from
  hundreds of microhartree to about 0.12 microhartree;
- the next limiting error is not quadrature but near-linear dependence in the
  overlapping atom-centered basis.

The remaining basis error may contain radial-extent and metric-cutoff effects,
so 0.12 microhartree is not claimed as the final basis limit. The important
result is that the two error sources are now experimentally separated, and
the redundancy problem can be studied without the earlier grid-rotation
noise.
