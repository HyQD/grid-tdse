# PySCF quadrature convergence of the C1 pilot

## Question

Does the C1 B-spline pilot converge systematically when only the angular
quadrature is increased, and what error is introduced by standard PySCF
pruning?

The basis, radial integration rule, Becke scheme, geometry, and four proper
Lebedev rotations were held fixed. Only the Lebedev order and pruning scheme
were changed. Håkon's radial rule was used through the PySCF adapter; no GTO
basis entered the calculation.

Run with

```console
conda run -n haakon_2026 \
  python examples/one_radial_atomic_c1_pyscf_convergence.py
```

All energies below are errors relative to the H2+ electronic reference in
microhartree. The envelope is the maximum energy minus the minimum energy
among the four fixed rotations.

## Unpruned angular grids

| Lebedev order | Grid points | Unrotated error | Rotation minimum | Rotation maximum | Envelope |
|---:|---:|---:|---:|---:|---:|
| 29 | 241600 | -2.989 | -9.042 | +3.188 | 12.230 |
| 41 | 472000 | +3.392 | +1.451 | +3.392 | 1.941 |
| 53 | 779200 | +3.362 | +3.318 | +3.362 | 0.044 |
| 59 | 961600 | +3.434 | +2.804 | +3.434 | 0.630 |
| 65 | 1163200 | +3.386 | +3.353 | +3.464 | 0.112 |

The scalar sequence is not strictly monotone: order 53 is unusually
favorable, and order 59 has a larger rotation envelope. Nevertheless, the
envelope falls by roughly two orders of magnitude from order 29 to order 65,
and all rotations at order 65 lie within 0.112 microhartree. The unrotated
energy has settled near a fixed basis error of +3.38 microhartree.

This is practical systematic convergence if the rotation envelope, rather
than one grid orientation, is used as the error estimate. It is not monotone
quadrature convergence in the strict one-number sense.

## PySCF NWChem pruning

| Lebedev order | Grid points | Unrotated error | Rotation minimum | Rotation maximum | Envelope |
|---:|---:|---:|---:|---:|---:|
| 29 | 213952 | -4.053 | -9.946 | +1.390 | 11.337 |
| 41 | 365728 | +3.214 | +1.107 | +3.214 | 2.107 |
| 53 | 633952 | +2.895 | +2.659 | +3.332 | 0.673 |
| 59 | 795856 | +2.895 | +2.626 | +3.335 | 0.709 |
| 65 | 976288 | +2.696 | +2.696 | +3.457 | 0.762 |

NWChem pruning reduces the point count by about 11-23 percent in this setup.
It is useful at moderate accuracy, but it leaves a high-order rotation
envelope of about 0.7 microhartree and shifts the unrotated order-65 energy by
-0.690 microhartree relative to the unpruned grid. Increasing only the nominal
maximum Lebedev order does not remove this floor because the pruning function
continues to assign smaller angular grids to selected radial shells.

## Interpretation

The PySCF adapter reproduces the native-grid result closely. At order 41 the
unpruned PySCF C1 error and envelope are +3.392 and 1.941 microhartree,
compared with +3.378 and 2.000 microhartree for the native grid. At order 59
the corresponding values are +3.434 and 0.630 microhartree, compared with
+3.437 and 0.641 microhartree.

The C1 pilot therefore supports controlled angular convergence to a fixed
basis limit. For sub-microhartree work, the current standard pruning is not
yet harmless. The natural next pruning experiment is a less aggressive
custom PySCF pruning function that retains high angular order in the radial
regions contributing most strongly to cross-center kinetic matrix elements.
