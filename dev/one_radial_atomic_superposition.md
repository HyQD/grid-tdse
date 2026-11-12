# Unweighted atomic superposition prototype

## Question

The original `one_radial_becke` prototype uses Becke functions twice:

1. `q_C(x)` partitions atom-centered numerical integration;
2. `p_A(x)` is included in each orbital basis function,
   `chi_A(x) = p_A(x) phi_A(x)`.

The second use is a partition-of-unity representation choice, not a
requirement of Becke integration.  It introduces gradient terms involving
`grad p_A` and changes the approximation space.

This alternative asks whether an ordinary QC-style superposition is simpler
and sufficient:

```text
chi_A(x) = phi_A(x)
W_Ci = omega_Ci q_C(x_Ci)
```

All atomic basis functions are evaluated on every relevant atomic integration
grid.  Becke weights occur only in the final integration weights.

## Implementation

The prototype is in `grid_tdse.spherical.one_radial_atomic`.  It is separate
from `one_radial_becke`; no existing representation or test has been changed.

The H2+ implementation:

- uses unweighted atom-centered spherical FEM-DVR functions;
- uses the existing Becke partition only for integration;
- extends every atomic integration grid to cover the union of finite basis
  supports;
- groups points by their active radial FEM element on every center;
- evaluates and assembles only locally active basis functions;
- uses real arithmetic for the collinear `m = 0` block;
- solves the resulting generalized Hermitian eigenproblem with the existing
  canonical overlap orthogonalization.

Run it with:

```bash
conda run -n haakon_2026 python examples/one_radial_atomic_h2plus.py
```

## Initial H2+ result

For `R = 2` bohr, atomic radial boundaries `(0, 1, 2, 3, 4, 6, 8)`, eight
FEM-DVR points per element, `l_max = 2`, integration spacing `0.25`, ten
radial integration points per subelement, and Lebedev order 41:

```text
electronic energy       -1.102634011684 Eh
reference energy        -1.102634214000 Eh
absolute error           0.202 microhartree
basis dimension          246
quadrature points        470820
active element groups    24
overlap condition        4.81e7
retained rank             246 at cutoff 1e-8
assembly time             about 1.3 seconds
```

The energy is unchanged to displayed precision for relative overlap cutoffs
from `1e-8` through `1e-12`.  At `1e-6`, one direction is discarded without a
material energy change.

The comparison is not yet a general replacement for `one_radial_becke`.
Current scope is one electron, two equal collinear centers, and `m = 0`.

## Next checks

1. Add an independent dense-assembly comparison for a small basis.
2. Run separate radial, angular, Lebedev, and integration-spacing sweeps.
3. Generalize from H2+ to arbitrary collinear centers and charges.
4. Decide whether the overlap spectrum remains acceptable for three centers
   and larger radial boxes.
5. Only then expose a stable public system-level API.
