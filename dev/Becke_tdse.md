# Time-dependent extension of the atom-centered Becke representation

## Status

This document is a design assessment, not an implementation plan. It
evaluates whether the unweighted atom-centered representation in
`grid_tdse.spherical.one_radial_atomic` can be extended to time-dependent
one-electron calculations and identifies the numerical work required before
such an implementation should begin.

The discussion initially assumes fixed nuclei. Moving nuclei introduce a
time-dependent basis, overlap matrix, and basis-connection terms and should
be treated as a separate project.

## Current representation

The new stationary prototype uses ordinary unweighted atom-centered
spherical FEM-DVR functions,

```math
\phi_{A k\ell m}(\mathbf r)
=
\frac{L_k^A(r_A)}{r_A}Y_{\ell m}(\Omega_A),
```

and uses Becke weights only to partition numerical integration. The basis is
therefore an overlapping, nonorthogonal superposition of functions centered
on the nuclei. This differs from the earlier prototype in which the orbital
functions themselves were multiplied by Becke partition functions.

This representation is highly effective in the molecular inner region. Each
nuclear cusp is represented in its natural local coordinates, avoiding the
very long midpoint-centered partial-wave expansion otherwise required for a
two-center Coulomb problem. The stationary convergence evidence is recorded
in [one_radial_atomic_convergence_sweep.md](one_radial_atomic_convergence_sweep.md).

## Fixed-basis time-dependent equation

For fixed nuclei and a time-independent spatial basis, expansion coefficients
obey the generalized TDSE

```math
iS\dot{\mathbf c}(t)
=
\left[H_0+V_{\mathrm{field}}(t)\right]\mathbf c(t),
```

where

```math
S_{\mu\nu}=\langle\phi_\mu|\phi_\nu\rangle.
```

For a spatially uniform field in the length gauge,

```math
V_{\mathrm{field}}(t)
=
\sum_{q=-1}^{1}E_q(t)D_q.
```

The field-free Hamiltonian, overlap matrix, and three dipole components can
all be assembled once. Time dependence then enters only through scalar field
amplitudes. Becke quadrature would not need to be repeated at every time
step for this one-electron problem.

A generalized Crank--Nicolson method can be written as

```math
\left(S+\frac{i\Delta t}{2}H_{n+1/2}\right)\mathbf c_{n+1}
=
\left(S-\frac{i\Delta t}{2}H_{n+1/2}\right)\mathbf c_n.
```

When the linear systems are solved accurately and the matrices are mutually
consistent, this form preserves the discrete `S` norm. Metric-aware Krylov
and Magnus methods are alternatives. Explicitly forming `S^{-1}H` should be
avoided because it can magnify overlap-conditioning errors and obscure
Hermiticity.

## Angular couplings from an external field

For a collinear molecule driven by a field polarized along the molecular
axis, signed `m` remains conserved. In the length gauge, the local factor
`r_A cos(theta_A)` gives the usual

```math
\Delta\ell=\pm1,\qquad \Delta m=0
```

couplings. The current real `m = 0` prototype is therefore conceptually well
matched to this restricted case.

For arbitrary polarization, the spherical dipole components also produce
`Delta m = +/-1` couplings. The representation must then contain complete
angular tables rather than one fixed-`m` block. Per center, the angular size
changes from approximately

```text
l_max + 1
```

to

```text
(l_max + 1)^2.
```

This is a major increase in basis size, matrix storage, and quadrature cost.
The static cusp advantage remains, but it does not eliminate the angular
complexity of a continuum wavepacket driven far from the molecule.

Length- and velocity-gauge calculations should ultimately agree in a
converged representation. Their agreement will be an important diagnostic:
the length gauge stresses the large-radius basis through the unbounded
position operator, while the velocity gauge stresses accurate derivative
and surface terms.

## Why simply increasing the atomic radii is problematic

The principal obstacle is not the formal time-dependent equation. It is the
asymptotic redundancy of overlapping atomic bases.

Far from the molecule,

```math
r_A\simeq r_B,
\qquad
Y_{\ell m}(\Omega_A)\simeq Y_{\ell m}(\Omega_B),
```

so functions belonging to different centers increasingly describe the same
physical degrees of freedom. The stationary overlap spectra already show
this trend:

| Atomic basis radius | Untruncated `cond(S)` | Rank at cutoff `1e-8` |
|---:|---:|---:|
| 8 bohr | `4.81e7` | 246/246 |
| 15 bohr | `2.49e10` | 370/372 |
| 20 bohr | `3.76e11` | 453/456 |

For a stationary ground state, removing a few near-null directions does not
materially change the energy. For time propagation, the same operation is
more consequential:

- the discarded combinations may contain continuum motion;
- small overlap errors can be amplified over many time steps;
- canonical orthogonalization mixes distant basis functions and destroys
  locality;
- a cutoff that is harmless for the ground state may alter ionization or
  high-energy observables;
- a dense `S^(-1/2)` transformation makes large-scale propagation expensive.

Strong-field calculations can require radial boxes of tens or hundreds of
bohr. Extending both atomic grids to those distances would likely make the
overlap problem much worse than the `r_max = 20` result already indicates.

The present prototype also assembles dense matrices. Long radial grids,
larger angular cutoffs, or complete `m` spaces require a sparse, local, or
matrix-free operator representation. Time propagation should not be built
around the current dense prototype and then expected to scale by increasing
the box size.

## Outer-boundary complications

Ionizing calculations require a CAP, mask, or exterior complex scaling. Two
overlapping atom-centered exterior regions create several difficulties:

- the same physical outgoing wavepacket is represented on both grids;
- two center-specific CAPs overlap and can absorb duplicated components
  inconsistently;
- exterior complex scaling about two distinct centers is not geometrically
  consistent in their overlap;
- the union of two finite atomic spheres has a non-spherical exterior
  boundary;
- the most redundant part of the basis is also the region where absorption
  must be cleanest.

A physical-coordinate CAP can formally be integrated using Becke weights,
but it does not by itself remove basis redundancy or guarantee reflectionless
absorption.

## Recommended inner--outer architecture

The most credible time-dependent design is a hybrid domain decomposition.

### Inner region

Use the unweighted atom-centered basis in a finite molecular region enclosing
the nuclei and the chemically important part of the wavefunction. This
retains the principal advantage demonstrated by the stationary prototype:
rapid cusp convergence at small local `l_max`.

### Matching region

Couple the overlapping atomic representation to one common exterior basis.
Possible coupling mechanisms include

- a finite-element interface with shared trace degrees of freedom;
- a smooth partition-of-unity transition;
- a mortar or discontinuous-Galerkin interface;
- an `R`-matrix-style inner--outer boundary.

A smooth attenuation of the atomic functions is possible, but it reintroduces
partition-gradient contributions to the kinetic energy. Those terms would be
confined to the matching shell rather than affecting the entire molecular
representation and must be retained exactly in the weak form.

### Outer region

Use one nonredundant grid centered on the molecular origin. A midpoint
spherical FEM-DVR expansion is a natural first candidate. It converges slowly
when it must represent displaced nuclear cusps, but the inner atomic region
has already removed that task. In the far field, the molecular potential is
smooth and admits a rapidly convergent multipole expansion.

The CAP or exterior-complex-scaling region should live exclusively on this
outer grid. This gives one global outgoing-wave representation and one
well-defined absorbing boundary.

## Relation to published work

There is published precedent for using Becke-type multicenter quadrature in
continuum and time-dependent molecular calculations, although none of the
work cited below is identical to the present unweighted atom-centered
FEM-DVR representation.

The closest direct precedent is the real-time one-electron TDSE framework of
Hamer *et al.* [1]. It combines a hybrid Gaussian--FEDVR basis with
Becke-type multicenter quadrature and demonstrates bound-state calculations,
high-harmonic generation, photoionization cross sections, and photoelectron
momentum distributions for H and H2+. The propagation is performed in the
length gauge, and a complex absorbing potential is placed on an external
FEDVR region. This establishes that multicenter Becke quadrature can be part
of a practical real-time strong-field method, but it does not remove the need
for basis orthogonalization, treatment of near-linear dependencies, and a
separate long-range representation.

Gharibnejad *et al.* [2] analyze the continuum limitation of the original
Becke construction particularly directly. Ordinary atom-centered Becke
weights do not switch off asymptotically. Each atom-centered grid must
therefore cover essentially the complete continuum quantization volume,
leading to redundant quadrature points and awkward boundary behavior. Their
modified scheme smoothly extinguishes the atomic weights within a molecular
region and introduces a central master weight that tends to unity in the far
field. Small atom-centered grids then resolve the nuclear regions, while one
central grid represents the interstitial and asymptotic regions. Although
that paper develops and tests a quadrature scheme rather than a full
real-time propagator, its analysis gives a concrete basis for the
inner--outer architecture recommended here.

Earlier multicenter B-spline work supplies additional dynamical precedent.
Awasthi *et al.* [3] solve a single-active-electron molecular TDSE for H2 in
ultrashort laser pulses. Stener, Fronzoni, and Decleva [4] use a multicenter
B-spline basis in time-dependent density-functional photoionization theory,
and Toffoli and Decleva [5] extend the same general continuum machinery to
perturbative molecular multiphoton ionization. These studies demonstrate the
usefulness of multicenter local representations for molecular continua, but
the latter two are response or perturbative calculations rather than direct
strong-field real-time propagation. They should therefore be regarded as
supporting numerical context, not as demonstrations of the exact proposed
TDSE representation.

The literature supports the following design conclusions:

- Becke partitioning is realistic for molecular dynamics and has now been
  demonstrated in real-time one-electron strong-field calculations.
- Simply extending every atom-centered grid through a large ionization box
  preserves the asymptotic redundancy of the original Becke construction.
- Atomic grids and weights should be localized to the molecular region, with
  one central or otherwise nonredundant outer grid taking over smoothly.
- The absorber should act only in that common outer region.
- Orthonormalization thresholds and near-linear dependencies must be tested
  against continuum observables, not selected only from bound-state energy
  convergence.
- Existing results validate the general architecture, but not the specific
  matrices, weak form, or interface proposed for `grid-tdse`; those still
  require the validation gates below.

## Quadrature requirements before time propagation

The stationary prototype has not yet established fully converged
microhartree quadrature. Radial and Lebedev refinements are non-monotonic, and
some enlarged bases produce energies below the variational benchmark. The
most difficult contribution appears to be the off-center nuclear attraction:
an angular shell centered on one nucleus must integrate the Coulomb cusp at
the other nucleus.

Time propagation raises the standard. Small inconsistencies can accumulate
into norm drift, phase error, or spurious high-frequency radiation. Before a
TDSE implementation is justified, the following matrices should be
independently converged and checked:

- overlap;
- weak-form kinetic energy;
- nuclear attraction;
- all required dipole or momentum components;
- any matching-region and absorbing-boundary operators.

Matrix Hermiticity alone is insufficient if it is imposed by explicit
symmetrization after inaccurate quadrature. Convergence of physically
relevant matrix elements and commutator identities is also required.

## Validation gates

The following checks should precede production time propagation.

1. **Metric stability.** Map the overlap spectrum versus atomic radius,
   radial resolution, angular cutoff, number of centers, and matching radius.
   Demonstrate that observables are insensitive to the rank cutoff.
2. **Field-free propagation.** Propagate stationary eigenstates for many
   optical cycles and verify `S`-norm, energy, and phase conservation.
3. **Weak-field response.** Compare excitation energies and oscillator
   strengths with stationary eigenvalue calculations and established H2+
   references.
4. **Gauge comparison.** Demonstrate convergence between length- and
   velocity-gauge observables.
5. **Interface tests.** Send wavepackets across the inner--outer boundary and
   measure reflection, norm transfer, and flux conservation.
6. **Absorber tests.** Quantify CAP or ECS reflection over the momentum range
   relevant to the intended pulse.
7. **Continuum benchmark.** Compare ionization probabilities, photoelectron
   spectra, or harmonic spectra against a prolate-spheroidal or other
   independently converged calculation.
8. **Scaling assessment.** Measure memory and operator-application costs
   without relying on dense matrices or a dense global orthogonalization.

## Feasibility by target regime

| Target regime | Assessment |
|---|---|
| Bound-state dynamics and weak-field spectroscopy | Realistic after metric and quadrature validation |
| Linear polarization along the molecular axis | Most favorable because fixed `m` remains available |
| Moderate strong-field dynamics with limited ionization | Realistic with a hybrid outer region |
| Long-time ionization, ATI, or HHG | Not advisable with two long overlapping atomic grids alone |
| Arbitrary polarization | Feasible but substantially more expensive because `m` sectors couple |
| Moving nuclei | Separate major project requiring basis-connection terms |
| Multi-electron time-dependent mean fields | Possible long term, but repeated dynamic integration and scaling require a different performance assessment |

## References

1. K. A. Hamer, H. Gharibnejad, L. Argenti, and N. Douguet,
   ["Hybrid Basis and Multi-Center Grid Method for Strong-Field
   Processes," *Atoms* **13**, 92
   (2025)](https://www.mdpi.com/2218-2004/13/11/92).
2. H. Gharibnejad, N. Douguet, B. I. Schneider, J. Olsen, and L. Argenti,
   ["A Multi-Center Quadrature Scheme for the Molecular Continuum,"
   *Computer Physics Communications* **263**, 107889
   (2021)](https://doi.org/10.1016/j.cpc.2021.107889).
3. M. Awasthi, Y. V. Vanne, A. Saenz, A. Castro, and P. Decleva,
   ["Single-active-electron approximation for describing molecules in
   ultrashort laser pulses and its application to molecular hydrogen,"
   *Physical Review A* **77**, 063403
   (2008)](https://doi.org/10.1103/PhysRevA.77.063403).
4. M. Stener, G. Fronzoni, and P. Decleva,
   ["Time-dependent density-functional theory for molecular photoionization
   with noniterative algorithm and multicenter B-spline basis set: CS2 and
   C6H6 case studies," *The Journal of Chemical Physics* **122**, 234301
   (2005)](https://doi.org/10.1063/1.1937367).
5. D. Toffoli and P. Decleva,
   ["Density functional theory for molecular multiphoton ionization in the
   perturbative regime," *The Journal of Chemical Physics* **137**, 134103
   (2012)](https://doi.org/10.1063/1.4754820).

## Critical conclusion

The atom-centered Becke representation is a strong candidate for the inner
molecular region of a time-dependent method. Its stationary advantage is
real: nuclear cusps are represented accurately with a very small local
angular cutoff. That advantage does not make the same overlapping basis a
good continuum representation.

A direct extension to moderately larger boxes may be useful for bound or
weakly ionizing dynamics, especially for axial linear polarization. A robust
strong-field method should instead combine the atom-centered inner basis with
a nonredundant single-center exterior grid. The main research risks are the
inner--outer interface, overlap conditioning, and accurate multicenter
quadrature. These should be resolved before choosing a production time
integrator.
