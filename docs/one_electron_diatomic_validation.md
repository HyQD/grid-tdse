# One-electron diatomic ground-state validation

The `grid_tdse.spherical.one_radial` package represents the clamped-nuclei
one-electron Hamiltonian

```math
H=-\frac{1}{2\mu}\nabla^2
-\frac{Z_1}{|\mathbf r-\mathbf R_A|}
-\frac{Z_2}{|\mathbf r-\mathbf R_B|},
```

where

```math
\mathbf R_A=(0,0,-a),\qquad
\mathbf R_B=(0,0,a),\qquad
R=2a.
```

This document validates the initial spherical implementation for the H2+
ground state at `R=2` bohr, with `Z1=Z2=1`, `a=1`, `mu=1`, and `m=0`.
The clamped-nuclei total energy is

```math
E_{\mathrm{BO}}=\epsilon_0+\frac{1}{R},
```

where `epsilon_0` is the electronic eigenvalue.

## Reference energy

Kullie reports the nonrelativistic electronic energy at `R=2` as

```math
\epsilon_{\mathrm{ref}}
=-1.10263421449494646150896894154\ E_h.
```

The corresponding clamped-nuclei total energy used below is therefore

```math
E_{\mathrm{ref}}
=-0.60263421449494646150896894154\ E_h.
```

Reference: O. Kullie, "High-precision minmax solution of the two-center
Dirac equation," *Physical Review A* **110**, 052808 (2024),
[doi:10.1103/PhysRevA.110.052808](https://doi.org/10.1103/PhysRevA.110.052808).
The reference calculation uses a two-centre finite-element representation
and reports the nonrelativistic limit separately from the relativistic
energy.

## Reproducing the validation

Run the complete validation with

```bash
python examples/one_electron_diatomic_validation.py
```

The default calculation uses `p=21` for the radial-box and angular studies,
extends the orbital expansion through `l_max=60`, performs an independent
prolate-spheroidal calculation, and saves
`H2plus_spherical_convergence_R2.pdf`. The controls are listed by

```bash
python examples/one_electron_diatomic_validation.py --help
```

The independent radial polynomial-order calculation is

```bash
python examples/one_electron_diatomic_p_convergence.py \
    --l-max 4 \
    --p-max 41
```

The polynomial-order script uses the Poisson nuclear construction. The main
validation uses the analytic multipoles and separately compares them with the
Poisson construction.

## Radial polynomial order

For `l_max=4`, `L_max=8`, `r_max=16`, and element boundaries
`[0, 1, 2, 4, 8, 16]`, selected values from the odd-`p` sequence are

| `p` | Total energy (`E_h`) |
|---:|---:|
| 5  | -0.596366482310 |
| 7  | -0.596437582233 |
| 9  | -0.596437507539 |
| 13 | -0.596437507156 |
| 21 | -0.596437507153 |
| 41 | -0.596437507024 |

The radial error decreases rapidly through about `p=9`. At larger orders the
energy fluctuates on a scale of approximately `1e-10 E_h`; this is the
floating-point and conditioning floor of the high-order FEM-DVR and Poisson
matrices, not an angular-convergence result. The final value in the plot is a
finite-grid reference, so the relative-error curve becomes irregular after
reaching this floor.

## Radial box

With `p=21`, `l_max=12`, and the inner boundaries held fixed, extending the
outer boundary gives

| `r_max` (bohr) | Total energy (`E_h`) |
|---:|---:|
| 4  | -0.595920971981484 |
| 6  | -0.602080756714767 |
| 8  | -0.602136091231865 |
| 12 | -0.602136431169690 |
| 16 | -0.602136431169690 |

Thus, `r_max=12` is sufficient at the displayed precision for this fixed
angular cutoff.

## Radial element size

The element-size study keeps `r_max=16`, `p=7`, and `l_max=12`, while refining
the element layout without moving the nuclear boundary at `r=a=1`.

| Maximum element width (bohr) | Number of elements | Total energy (`E_h`) |
|---:|---:|---:|
| 8 | 5  | -0.602122627400778 |
| 4 | 10 | -0.602136270840137 |
| 2 | 20 | -0.602136431056216 |

The refined low-order sequence approaches the same fixed-`l_max` energy as
the polynomial-order study.

## Orbital angular cutoff

The midpoint one-centre expansion must represent electron-nuclear cusps at
two displaced points. Radial refinement cannot eliminate the resulting
partial-wave truncation error, and convergence with `l_max` is algebraic.
With `p=21`, `r_max=16`, and `L_max=2*l_max`, the principal results are

| `l_max` | Total energy (`E_h`) | Absolute reference error (`E_h`) |
|---:|---:|---:|
| 0  | -0.518502331446033 | 8.4132e-2 |
| 4  | -0.596437507162098 | 6.1967e-3 |
| 8  | -0.601263740383875 | 1.3705e-3 |
| 12 | -0.602136431169690 | 4.9778e-4 |
| 20 | -0.602507081701287 | 1.2713e-4 |
| 30 | -0.602593192801485 | 4.1022e-5 |
| 40 | -0.602616147687976 | 1.8067e-5 |
| 50 | -0.602624721678397 | 9.4928e-6 |
| 60 | -0.602628625001089 | 5.5895e-6 |

The sequence is monotone toward the high-accuracy result. At `l_max=60`, the
remaining error is about `9.3e-6` relative. This slow angular convergence is a
property of the midpoint spherical representation, rather than evidence of a
radial or nuclear-multipole error.

## Nuclear multipole cutoff

For orbital functions truncated at `l_max`, Gaunt selection rules imply that
nuclear multipoles above `L_max=2*l_max` have no projected action. At
`l_max=12`, the energy converges as follows:

| `L_max` | Total energy (`E_h`) |
|---:|---:|
| 0  | -0.518502331531511 |
| 4  | -0.594593284155860 |
| 8  | -0.600871971269661 |
| 12 | -0.601999880755671 |
| 16 | -0.602123119032873 |
| 20 | -0.602135177214181 |
| 22 | -0.602136186608644 |
| 24 | -0.602136431169690 |
| 26 | -0.602136431169690 |

Increasing the nuclear cutoff beyond `2*l_max=24` leaves the projected
Hamiltonian and energy unchanged.

## Analytic and Poisson nuclear potentials

For `p=21`, `l_max=12`, and `L_max=24`, the two constructions give

| Construction | Total energy (`E_h`) |
|---|---:|
| Analytic multipoles | -0.602136431169690 |
| Poisson solve | -0.602136431247889 |

Their difference is `7.82e-11 E_h`, below the angular truncation error by
more than six orders of magnitude.

## Independent prolate-spheroidal cross-check

The validation script also constructs the same Hamiltonian using grid-lib's
prolate-spheroidal weak-form operators. The nuclei coincide with the two
coordinate foci, so this calculation neither uses the midpoint spherical
partial-wave expansion nor its nuclear multipole representation. The default
prolate calculation gives

```text
Computed total energy:  -0.602634214495260 Eh
Published total energy: -0.602634214494946 Eh
Absolute difference:     3.140e-13 Eh
```

This independent coordinate representation confirms the Hamiltonian
conventions, the nuclear-repulsion shift, and the published benchmark. The
spherical sequence approaches the same value with its expected slower
angular convergence.
