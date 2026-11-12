# One-radial atomic H2+ convergence sweep

## Status

This is a provisional diagnostic report for the unweighted atom-centered
H2+ prototype introduced in `grid_tdse.spherical.one_radial_atomic`. The
sweep was run on 2026-11-12 at commit `ba60576` on the
`becke_partitioning` branch. It is intended to identify the dominant error
sources; it is not yet a production convergence study.

## Question

The default native calculation agrees with the accurate H2+ benchmark to
about `0.203 microhartree`. This sweep asks whether that remaining difference
is primarily caused by

1. the radial FEM-DVR basis resolution;
2. the spherical-harmonic cutoff `l_max`; or
3. the radial or angular integration grid.

## Reference and error convention

The published nonrelativistic electronic energy at an internuclear distance
of 2 bohr is

```text
E_ref = -1.1026342144949464615 Eh.
```

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

The sweep supports three conclusions.

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

## Completion criterion

This diagnostic becomes a completed convergence study when independent
radial-quadrature and angular-quadrature refinements change the electronic
energy by less than `0.1 microhartree` for every radial-basis and `l_max`
value used to infer basis convergence.
