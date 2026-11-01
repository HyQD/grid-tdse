# Davidson ground-state archives

`examples/non_bo_diatomic_davidson_ground_state.py` writes converged CODATA
2018 H2+, HD+, and D2+ Davidson states to this directory.

Archive names encode the system, symmetry sector, fixed
`8e21-4e21_16-8` radial representation, angular cutoff, solver, CODATA set,
and residual tolerance. For an even `l_max >= 2`, the example automatically
uses the corresponding `l_max - 2` archive from this directory when present.
It falls back to the product state when that archive is unavailable.

Generated `.npz` state archives and `__log.txt` progress logs are calculation
outputs and are ignored by Git. The fifteen local archives from the validation
campaign occupy approximately 129 MiB; keeping them outside ordinary Git
history avoids permanently inflating every clone.

## CODATA 2018 validation

The serial Davidson implementation was validated on 2026-11-01 for all three
systems and even angular cutoffs through `l_max = 8`. The `l_max = 0` states
used the product guess; every subsequent state continued from `l_max - 2`.
All calculations used the one-body spectral preconditioner, a 30-vector
subspace with four retained Ritz vectors, and residual tolerance `1e-8`.

| System | `l_max` | Energy (Eh) | Residual | H actions | Restarts |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2+ | 0 | -0.515658722200 | 7.000e-09 | 134 | 4 |
| H2+ | 2 | -0.578170599182 | 8.982e-09 | 49 | 1 |
| H2+ | 4 | -0.590684018037 | 7.728e-09 | 46 | 1 |
| H2+ | 6 | -0.594298725872 | 9.406e-09 | 45 | 1 |
| H2+ | 8 | -0.595655496947 | 9.669e-09 | 44 | 1 |
| HD+ | 0 | -0.510263922769 | 8.982e-09 | 142 | 5 |
| HD+ | 2 | -0.572121746804 | 9.461e-09 | 55 | 1 |
| HD+ | 4 | -0.588358315536 | 9.609e-09 | 54 | 1 |
| HD+ | 6 | -0.593669719570 | 9.266e-09 | 55 | 1 |
| HD+ | 8 | -0.595720116307 | 8.038e-09 | 55 | 1 |
| D2+ | 0 | -0.517754846498 | 9.594e-09 | 146 | 5 |
| D2+ | 2 | -0.580083223498 | 9.125e-09 | 59 | 2 |
| D2+ | 4 | -0.592453574203 | 7.810e-09 | 55 | 1 |
| D2+ | 6 | -0.596007546297 | 9.669e-09 | 52 | 1 |
| D2+ | 8 | -0.597337475006 | 6.319e-09 | 53 | 1 |

All norms were one to twelve decimal places, inversion expectations were one,
and pair-exchange expectations were one where applicable. All fifteen energies
round to the supplied co-worker references at eight decimal places, including
the corrected HD+ `l_max = 8` reference value `-0.59572012` Eh.
