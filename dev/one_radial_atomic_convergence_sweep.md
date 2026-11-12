# One-radial atomic H2+ convergence sweep

## Status

This is a provisional diagnostic report for the unweighted atom-centered
H2+ prototype introduced in `grid_tdse.spherical.one_radial_atomic`. The
initial sweep was run on 2026-11-12 at commit `ba60576` on the
`becke_partitioning` branch. The radial-extent follow-up was run at commit
`2a493f6`; the solver code was unchanged between the two commits. This report
is intended to identify the dominant error sources; it is not yet a
production convergence study.

## Question

The default native calculation agrees with the accurate H2+ benchmark to
about `0.203 microhartree`. This sweep asks whether that remaining difference
is primarily caused by

1. the radial FEM-DVR basis resolution;
2. the spherical-harmonic cutoff `l_max`; or
3. the radial or angular integration grid; or
4. the finite atom-centered basis radius.

## Reference and error convention

The published nonrelativistic electronic energy at an internuclear distance
of 2 bohr is

```text
E_ref = -1.1026342144949464615 Eh.
```

This is the nonrelativistic limit reported by O. Kullie, "High-precision
minmax solution of the two-center Dirac equation," *Physical Review A*
**110**, 052808 (2024),
[doi:10.1103/PhysRevA.110.052808](https://doi.org/10.1103/PhysRevA.110.052808).
The corresponding clamped-nuclei total energy is
`-0.6026342144949464615 Eh` after adding the nuclear repulsion `1/R = 0.5 Eh`.

All errors below are signed and reported in microhartree:

```text
error = 1e6 * (E - E_ref).
```

A positive error is above the benchmark. A negative error is below the
benchmark and therefore warns that numerical integration has broken the
expected variational trend.

## Default native calculation

Unless a parameter is varied explicitly, the native calculations use

```text
radial element boundaries       (0, 1, 2, 3, 4, 6, 8) bohr
radial points per element       8
l_max                           2
integration subinterval width   0.25 bohr
integration points/subinterval  10
Lebedev order                   41
Becke switch iterations         3
relative overlap cutoff         1e-8
```

The default result is

```text
E = -1.102634011684481 Eh
error = +0.2028 microhartree
basis dimension = 246
quadrature points = 470820
```

## Native-grid sweeps

### Radial extent

The initial sweeps fixed the atom-centered basis radius at 8 bohr. A
follow-up extended the basis by appending outer elements without changing the
inner mesh:

```text
r_max =  8: (0, 1, 2, 3, 4, 6, 8)
r_max = 15: (0, 1, 2, 3, 4, 6, 8, 10, 12, 15)
r_max = 20: (0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 17.5, 20)
```

For H2+ at `R = 2` bohr, the corresponding integration grids extend to
`r_max + R`, namely 10, 17, and 22 bohr. The basis used eight radial points
per element and `l_max = 2` throughout.

The default native quadrature gives

| Basis `r_max` | Integration `r_max` | Basis size | Retained rank | `cond(S)` | Quadrature points | Electronic energy (Eh) | Error (microhartree) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 10 | 246 | 246/246 | 4.810e7 | 470820 | -1.102634011684481 | +0.20281 |
| 15 | 17 | 372 | 370/372 | 2.487e10 | 801220 | -1.102634134912691 | +0.07958 |
| 20 | 22 | 456 | 453/456 | 3.763e11 | 1037220 | -1.102634134912475 | +0.07958 |

Increasing the basis radius from 8 to 15 bohr lowers the energy by
`0.12323 microhartree`. Increasing it further from 15 to 20 bohr has no
resolved effect on the energy.

The extent shift is reproducible with two different integration settings:

| Grid backend and settings | Error at 8 bohr | Error at 15 bohr | Error at 20 bohr | 8-to-15 energy shift |
|---|---:|---:|---:|---:|
| Native, 10 radial points and Lebedev 41 | +0.20281 | +0.07958 | +0.07958 | -0.12323 |
| Native, 14 radial points and Lebedev 47 | +3.22924 | +3.10641 | +3.10641 | -0.12283 |
| PySCF/NWChem-pruned, 14 radial points and Lebedev 53 | +0.43001 | +0.30729 | +0.30729 | -0.12273 |

All values and shifts in this table are in microhartree. Although the
absolute errors depend on the integration grid, all three grids give an
8-to-15-bohr correction of approximately `-0.123 microhartree`. This makes
the finite-box effect much more robust than the absolute sub-microhartree
error of any one grid.

The larger boxes introduce severe near-linear dependencies. At the default
relative overlap cutoff of `1e-8`, two directions are discarded for
`r_max = 15` and three for `r_max = 20`. A cutoff check gave

| Basis `r_max` | Relative cutoff | Retained rank | Retained condition number | Electronic energy (Eh) |
|---:|---:|---:|---:|---:|
| 15 | 1e-8 | 370/372 | 2.117e7 | -1.102634134912691 |
| 15 | 1e-10 | 371/372 | 3.977e8 | -1.102634134912610 |
| 15 | 1e-12 | failed | -- | -- |
| 20 | 1e-8 | 453/456 | 3.319e7 | -1.102634134912475 |
| 20 | 1e-10 | failed | -- | -- |
| 20 | 1e-12 | failed | -- | -- |

The `r_max = 15` energy changes by only about `8e-14 Eh` when the cutoff is
tightened from `1e-8` to `1e-10`. The failed cases exceed the
anti-Hermitian tolerance after canonical orthogonalization because the
near-singular overlap amplifies roundoff. There is therefore no observed
ground-state benefit from extending the basis beyond 15 bohr, while the
conditioning worsens by more than an order of magnitude.

### Angular basis cutoff

Only `l_max` changes in this table.

| `l_max` | Basis size | Electronic energy (Eh) | Error (microhartree) |
|---:|---:|---:|---:|
| 0 | 82 | -1.090951942101 | +11682.2724 |
| 1 | 164 | -1.102427510295 | +206.7042 |
| 2 | 246 | -1.102634011684 | +0.2028 |
| 3 | 328 | -1.102638007795 | -3.7933 |
| 4 | 410 | -1.102638936961 | -4.7225 |

Increasing `l_max` from 1 to 2 removes about `206.5 microhartree` of error.
The values for `l_max >= 3` fall below the benchmark, so they cannot be
interpreted as clean variational convergence of the angular basis. At that
point the fixed integration grid is no longer sufficiently accurate.

### Radial FEM-DVR basis resolution

The varied value is `radial_n_points`, the number of Lobatto points in every
radial finite element. Here `l_max = 2` and the integration grid remains at
its default settings.

| Radial points/element | Basis size | Electronic energy (Eh) | Error (microhartree) |
|---:|---:|---:|---:|
| 5 | 138 | -1.102630419706 | +3.7948 |
| 6 | 174 | -1.102632448275 | +1.7662 |
| 7 | 210 | -1.102633324914 | +0.8896 |
| 8 | 246 | -1.102634011684 | +0.2028 |
| 9 | 282 | -1.102635467242 | -1.2527 |
| 10 | 318 | -1.102636150221 | -1.9357 |

Radial basis refinement from five to eight points per element improves the
energy by about `3.6 microhartree`. Further refinement again exposes
integration error and drives the result below the benchmark.

### Radial quadrature resolution

This sweep changes only the number of radial Lobatto integration points per
subinterval. The maximum integration-subinterval width remains 0.25 bohr.

| Integration points/subinterval | Quadrature points | Electronic energy (Eh) | Error (microhartree) |
|---:|---:|---:|---:|
| 6 | 282020 | -1.102641868974 | -7.6545 |
| 8 | 376420 | -1.102633138272 | +1.0762 |
| 10 | 470820 | -1.102634011684 | +0.2028 |
| 12 | 565220 | -1.102631541144 | +2.6734 |
| 14 | 659620 | -1.102632593039 | +1.6215 |

The sequence is non-monotonic and varies by several microhartree. The
default `+0.203 microhartree` result is therefore not a reliable estimate of
the radial-basis error by itself.

### Lebedev angular quadrature

This sweep changes only the native Lebedev order.

| Lebedev order | Quadrature points | Electronic energy (Eh) | Error (microhartree) |
|---:|---:|---:|---:|
| 29 | 240996 | -1.102642490472 | -8.2760 |
| 35 | 346332 | -1.102634821937 | -0.6074 |
| 41 | 470820 | -1.102634011684 | +0.2028 |
| 47 | 614460 | -1.102633610016 | +0.6045 |
| 53 | 777252 | -1.102637873706 | -3.6592 |

This sequence is also non-monotonic. Increasing the formal Lebedev order does
not guarantee a smaller error for the off-center Coulomb cusp sampled on
atom-centered shells.

## PySCF-grid cross-check

An independent sweep used the optional PySCF grid backend. PySCF supplied the
original three-iteration Becke partition, Lebedev grids, NWChem angular
pruning, and spatial grouping. `grid-tdse` supplied an interior composite
Gauss-Legendre radial rule.

For the `l_max` sweep, the common settings were

```text
radial basis points/element     8
integration subinterval width   0.25 bohr
integration points/subinterval  14
Lebedev order                   53
angular pruning                 NWChem
quadrature points               888392
```

| `l_max` | Basis size | Electronic energy (Eh) | Error (microhartree) |
|---:|---:|---:|---:|
| 1 | 164 | -1.102427072712 | +207.1418 |
| 2 | 246 | -1.102633784481 | +0.4300 |
| 3 | 328 | -1.102638729867 | -4.5154 |
| 4 | 410 | -1.102640668548 | -6.4541 |

The PySCF backend confirms the main native-grid pattern: `l_max = 1` is about
`207 microhartree` too high, `l_max = 2` reaches sub-microhartree agreement,
and larger angular bases expose several microhartree of integration error.

At fixed `l_max = 3`, Lebedev order 53, and the same integration settings,
the radial-basis sweep was

| Radial points/element | Basis size | Electronic energy (Eh) | Error (microhartree) |
|---:|---:|---:|---:|
| 6 | 232 | -1.102636507562 | -2.2931 |
| 8 | 328 | -1.102638729867 | -4.5154 |
| 10 | 424 | -1.102641880652 | -7.6662 |

At fixed `l_max = 3` and eight radial points per element, the pruned angular
grid sweep was

| Lebedev order | Quadrature points | Electronic energy (Eh) | Error (microhartree) |
|---:|---:|---:|---:|
| 41 | 512456 | -1.102654865798 | -20.6513 |
| 47 | 687440 | -1.102639991702 | -5.7772 |
| 53 | 888392 | -1.102638729867 | -4.5154 |
| 59 | 1115312 | -1.102639535850 | -5.3214 |

The PySCF results do not remove the nonvariational behavior. Interior radial
points avoid evaluating directly on shared FEM boundaries, but angular
pruning and the numerical treatment of the off-center nuclear singularity
still require further study.

## Interpretation

The sweep supports four conclusions.

1. **Angular-basis truncation dominates at low `l_max`.** Increasing
   `l_max` from 1 to 2 reduces the error by roughly `206.5 microhartree`, much
   more than the few-microhartree improvement obtained by increasing the
   radial basis from five to eight points per element.

2. **The default sub-microhartree agreement is partly error cancellation.**
   Changes in radial quadrature and Lebedev order move the result by several
   microhartree, which is larger than the reported default error of
   `0.203 microhartree`.

3. **The current limiting issue is integration rather than a demonstrated
   `l_max = 2` basis limit.** Enlarging either the radial or angular basis
   drives the energy below the benchmark on both grid backends. A converged
   integration must be established before the residual radial- and
   angular-basis errors can be separated reliably.

4. **A basis radius of 15 bohr is sufficient for the present H2+ ground-state
   calculation.** Extending the radius from 8 to 15 bohr lowers the energy by
   a reproducible `0.123 microhartree`, whereas extending it from 15 to 20
   bohr produces no resolved improvement and substantially worsens the
   overlap conditioning. For the default native grid, the extent correction
   reduces the signed benchmark error from `+0.203` to
   `+0.080 microhartree`, but this absolute agreement still contains
   quadrature-error cancellation.

The most likely difficult term is the multicenter nuclear attraction: a grid
centered on one nucleus must integrate the Coulomb cusp at the other nucleus.
This produces slow or irregular angular convergence on shells that pass near
the off-center nucleus.

## Recommended follow-up

1. Converge radial and angular quadrature at a fixed small basis before
   increasing either radial basis size or `l_max`.
2. Record overlap, kinetic, and nuclear-attraction energies separately to
   identify which matrix causes the nonvariational shift.
3. Compare pruned and unpruned PySCF angular grids using identical radial
   points.
4. Test a singularity-aware treatment of the nuclear-attraction integrals,
   such as center-local analytic terms or a dedicated partition/integration
   treatment around each nucleus.
5. Repeat the radial and `l_max` sweeps only after quadrature changes affect
   the energy by substantially less than `0.1 microhartree`.
6. Use `r_max = 15` rather than 20 for subsequent ground-state convergence
   work unless a more stable treatment of the outer-box overlap spectrum is
   introduced.

## Completion criterion

This diagnostic becomes a completed convergence study when independent
radial-quadrature and angular-quadrature refinements change the electronic
energy by less than `0.1 microhartree` for every radial-basis and `l_max`
value used to infer basis convergence.
