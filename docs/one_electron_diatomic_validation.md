# One-electron diatomic stationary-state validation

The `grid_tdse.spherical.one_radial` and
`grid_tdse.prolate_spheroidal` packages represent the clamped-nuclei
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

This document validates the spherical and prolate-spheroidal implementations
for H2+ stationary states at `R=2` bohr, with `Z1=Z2=1`, `a=1`, `mu=1`.
The clamped-nuclei total energy is

```math
E_{\mathrm{BO}}=\epsilon_0+\frac{1}{R},
```

where `epsilon_0` is the electronic eigenvalue.

## High-precision ground-state reference

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

## Spherical one-radial ground-state validation

## Reproducing the validation

Run the complete validation with

```bash
python examples/spherical/one_radial/one_electron_diatomic_validation.py
```

The default calculation uses `p=21` for the radial-box and angular studies,
extends the orbital expansion through `l_max=60`, performs an independent
prolate-spheroidal calculation, and saves
`H2plus_spherical_convergence_R2.pdf`. The controls are listed by

```bash
python examples/spherical/one_radial/one_electron_diatomic_validation.py --help
```

The independent radial polynomial-order calculation is

```bash
python examples/spherical/one_radial/one_electron_diatomic_p_convergence.py \
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

## Prolate-spheroidal stationary-state validation

The prolate-spheroidal implementation places the two nuclei at the coordinate
foci. It therefore represents both electron--nuclear singularities directly,
without the midpoint spherical partial-wave expansion or a nuclear
multipole cutoff.

### Davidson ground state

Run the user-facing Davidson calculation with

```bash
python examples/prolate_spheroidal/one_electron_diatomic_ground_state.py
```

The development calculation uses

```text
R:                            2 bohr
xi element boundaries:       [1, 4, 14, 24]
xi points per element:       14
eta Gauss--Legendre points:  9
m:                            0
active state shape:           (39, 9)
```

This compact grid gives

| Quantity | Calculated (`E_h`) | Kullie reference (`E_h`) | Difference (`E_h`) |
|---|---:|---:|---:|
| Electronic energy | -1.102634214485352 | -1.102634214494946 | 9.594e-12 |
| Total energy | -0.602634214485352 | -0.602634214494946 | 9.594e-12 |

The Davidson residual norm is `8.264e-11` for this run. The agreement should
be interpreted as a validation of the Hamiltonian, metric, boundary
conditions, and eigensolver on this grid. It is not a general convergence
claim for other bond lengths, charges, or excited states.

### Excitation energies from Tao, McCurdy, and Rescigno

Table II of Tao, McCurdy, and Rescigno reports selected H2+ electronic and
excitation energies at `R=2` bohr. Their fixed-nuclei time-dependent method
uses a finite-element DVR in prolate spheroidal coordinates. The present
comparison is reproduced with

```bash
python examples/prolate_spheroidal/one_electron_diatomic_excited_states.py
```

The comparison grid follows the published Table-I discretization used by the
grid-lib validation:

```text
xi elements:                  one element from 1 to 2,
                              ten equal elements from 2 to 80
xi points per element:       15
eta Gauss--Legendre points:  9
active state shape:           (154, 9) in each fixed-m sector
sectors:                      m=0 and m=1
```

The direct solver is used here as a deterministic validation oracle for the
low-lying spectrum. Davidson remains the default stationary-state solver.
Excitation energies are formed from the more precise calculated ground-state
energy,

```math
\Delta E_n = E_n-E_0,
\qquad
E_0=-1.102634214493190\ E_h.
```

The constant nuclear repulsion cancels in every excitation energy. The
published values are rounded to the digits shown in Table II.

| Table-II state | `m` | Calculated symmetry | Calculated `E_n` (`E_h`) | Table-II `E_n` (`E_h`) | Calculated `Delta E_n` (`E_h`) | Table-II `Delta E_n` (`E_h`) |
|---|---:|:---:|---:|---:|---:|---:|
| `2p sigma_u` | 0 | `u` | -0.667534392201 | -0.667534 | 0.435099822292 | 0.43510 |
| `3p sigma_u` | 0 | `u` | -0.255413165086 | -0.255413 | 0.847221049407 | 0.84722 |
| `4p sigma_u` entry | 0 | `g` | -0.177681045126 | -0.177681 | 0.924953169367 | 0.92495 |
| `4f sigma_u` | 0 | `u` | -0.126643870150 | -0.126644 | 0.975990344344 | 0.97599 |
| `2p pi_u` | 1 | `u` | -0.428771819896 | -0.428772 | 0.673862394597 | 0.67386 |
| `3p pi_u` | 1 | `u` | -0.200864829911 | -0.200865 | 0.901769384582 | 0.90177 |
| `4f pi_u` | 1 | `u` | -0.126198920530 | -0.126199 | 0.976435293964 | 0.97643 |

The largest difference from a printed Table-II excitation energy is
`5.3e-6 E_h`. This is consistent with the limited precision of the published
entries and the use of the more precise calculated ground-state energy.

There is one important state-label qualification. The numerical eigenstate
at `-0.177681045126 E_h`, which matches the energy printed for
`4p sigma_u`, has inversion expectation

```math
\langle P\rangle=+1,
```

and is therefore gerade. The nearest computed ungerade `Sigma` root instead
has

```text
E_n       = -0.137312924277 E_h
Delta E_n =  0.965321290216 E_h.
```

The example reports this as an apparent reference-label inconsistency rather
than silently assigning the published label by numerical proximity. Every
state in the comparison is classified using inversion of the computed
wavefunction. For a fixed magnetic quantum number, inversion combines
reflection of the prolate `eta` coordinate with the azimuthal phase
`(-1)^m`.

Reference: L. Tao, C. W. McCurdy, and T. N. Rescigno, "Grid-based methods
for diatomic quantum scattering problems. II. Time-dependent treatment of
single- and two-photon ionization of H2+," *Physical Review A* **80**,
013402 (2009),
[doi:10.1103/PhysRevA.80.013402](https://doi.org/10.1103/PhysRevA.80.013402).

### Heteronuclear two-centre validation

The same Hamiltonian implementation supports unequal positive charges. The
heteronuclear comparison is reproduced with

```bash
python \
    examples/prolate_spheroidal/one_electron_diatomic_heteronuclear_validation.py
```

Every case uses `m=0` and the same validation grid:

```text
xi element boundaries:       [1, 4, 14, 24]
xi points per element:       18
eta Gauss--Legendre points:  24
```

Campos et al. tabulate nonrelativistic electronic eigenvalues for HeH2+,
LiH3+, and BeH4+ at fixed internuclear distances. Their convention excludes
the nuclear repulsion. The clamped-nuclei total energy reported here is

```math
E_{\mathrm{BO}}=E_{\mathrm{el}}+\frac{Z_1Z_2}{R}.
```

The direct eigensolver is used as a deterministic validation oracle. The
results are

| System | State | `R` (bohr) | Root in `m=0` sector | Calculated `E_el` (`E_h`) | Reference `E_el` (`E_h`) | Absolute error (`E_h`) | Calculated `E_BO` (`E_h`) |
|---|---|---:|---:|---:|---:|---:|---:|
| HeH2+ | `1s sigma` | 2 | 0 | -2.512193016590 | -2.512193016590 | 3.14e-13 | -1.512193016590 |
| LiH3+ | `1s sigma` | 2 | 0 | -5.002042016848 | -5.002042016849 | 8.29e-13 | -3.502042016848 |
| LiH3+ | `1s sigma` | 4 | 0 | -4.750111784802 | -4.750111784789 | 1.34e-11 | -4.000111784802 |
| BeH4+ | `1s sigma` | 2 | 0 | -8.500592345518 | -8.500592345526 | 7.99e-12 | -6.500592345518 |
| BH5+ | `3p sigma` | `sqrt(10)/3` | 4 | -2.000000000000 | -2 exactly | 1.32e-13 | 2.743416490252 |

The largest electronic-energy discrepancy is `1.34e-11 E_h`, while the
largest eigenstate residual in the recorded run is `1.05e-12`. These results
validate unequal charges, the charge-dependent nuclear-repulsion shift, and
low excited-state ordering in the fixed-`m` representation.

The BH5+ entry is qualitatively different from the other entries. Khoma
reports Demkov's exact analytic solution for `Z1=1`, `Z2=5`,
`R=sqrt(10)/3`, and the `3p sigma` state:

```math
E_{\mathrm{el}}=-2\ E_h,
\qquad
\lambda=-\frac{10}{3}.
```

The calculated state is root 4 when the `m=0` eigenvalues are sorted in
ascending order. It provides an exact heteronuclear excited-state benchmark,
whereas the other four entries test tabulated ground-state energies.

References:

- J. A. Campos, D. L. Nascimento, D. T. Cavalcante, A. L. A. Fonseca,
  and A. O. C. Nunes, "Determination of Electronic Energy Levels for the
  Heteromolecular Ions HeH2+, LiH3+, and BeH4+ From the Hamilton--Jacobi
  Equation," *International Journal of Quantum Chemistry* **106**, 2587--2596
  (2006),
  [doi:10.1002/qua.21015](https://doi.org/10.1002/qua.21015).
- M. V. Khoma, "Generalized spheroidal wave equation for real and complex
  valued parameters. An algorithm based on the analytic derivatives for the
  eigenvalues," [arXiv:2503.18280](https://arxiv.org/abs/2503.18280).
