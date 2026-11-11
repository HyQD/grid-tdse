# Coupled-angular implementation plan for non-Born-Oppenheimer diatomic ions

## Status and scope

This document is a design, implementation, and validation plan. Steps 1-7 are
implemented and covered by focused tests; Steps 8-17 have not started.

The primary target systems are H2+, HD+, and D2+. The first coupled-angular
representation will describe one electronic Jacobi coordinate and one
internuclear Jacobi coordinate in a space-fixed bipolar spherical-harmonic
basis. It must support field-free ground states and laser-driven dynamics while
preserving the existing uncoupled representation as a reference implementation
and regression oracle.

The principal decisions are:

1. use space-fixed bipolar spherical harmonics with explicit total angular
   momentum quantum numbers `(L, M)`;
2. support inversion parity for all three isotopologues and nuclear-exchange
   parity for homonuclear H2+ and D2+;
3. retain the existing two-grid FEMDVR Poisson construction for the radial
   Coulomb multipoles;
4. replace only the angular Coulomb coupling layer by a coupled-basis layer;
5. validate every coupled operator against an explicit unitary transformation
   of the existing uncoupled representation before using it in production;
6. add serial field-free functionality first, followed by length-gauge
   dynamics and finally MPI integration.

Neutral H2, electron spin, nuclear spin, hyperfine structure, relativistic
corrections, and body-fixed nonadiabatic representations are outside the first
implementation.

## 1. Assessment of the FEMDVR Coulomb construction

### 1.1 Thesis evidence

Chapter 3 of Denis Jelovina's thesis develops the electron-nuclear interaction
from the multipole expansion

```math
\frac{1}{|\mathbf r-c\mathbf R|}
=
4\pi\sum_{\lambda=0}^{\infty}\frac{1}{2\lambda+1}
\frac{r_<^\lambda}{r_>^{\lambda+1}}
\sum_{\mu=-\lambda}^{\lambda}
Y_{\lambda\mu}^{*}(\hat{\mathbf r})
Y_{\lambda\mu}(\widehat{c\mathbf R}).
```

The thesis points out that direct Gauss-Lobatto evaluation of
`r_<^lambda / r_>^(lambda+1)` is inaccurate near its derivative discontinuity.
It instead obtains the radial multipoles from a Poisson equation in the FEMDVR
basis. The derivation permits separate electronic and internuclear FEMDVR grids
and interpolates between them. The resulting electron-nuclear operator is
diagonal in the two radial DVR indices. See Section 3.7.3, especially equations
3.81-3.98 and the comparison in Figure 3.4.

This evidence favors the Poisson construction over direct DVR sampling of the
piecewise analytic kernel. It does not imply that a coupled-angular basis must
use the same angular specialization as the thesis.

Two qualifications are important:

- the coupled spherical harmonics introduced in the thesis are used to couple
  the angular coordinates of the two electrons in neutral H2, not the
  electronic and internuclear coordinates of H2+;
- parts of the H2+ treatment align or freeze the molecular axis, reducing the
  angular expansion to `mu=0`. That specialization must not be carried into a
  full space-fixed rotational treatment.

The radial Poisson rationale remains applicable because it depends on the
radial multipole kernel, not on whether the angular functions are coupled or
uncoupled.

### 1.2 Current grid-tdse separation

The current implementation already separates the interaction into:

- `RadialCoulombMultipoles`, built from the two radial grids and the retained
  `lambda` values; and
- sparse angular coupling records, currently built as products of one-coordinate
  Gaunt coefficients.

For FEMDVR grids, `build_radial_coulomb_multipoles` calls the two-grid Poisson
implementation in `grid-lib`. The returned data are physical pointwise values
`W_lambda(r_i, R_j)` with source quadrature weights removed. The Hamiltonian
then combines these values with sparse angular coefficients.

This is the correct boundary between reusable and new work. The coupled basis
changes the angular coefficients but not `W_lambda`.

### 1.3 Recommendation

Retain the FEMDVR Poisson radial construction as the production method. This is
not a drawback for the coupled representation:

- it avoids the known DVR error at `r = |c|R`;
- it already supports independent electronic and internuclear grids;
- it supports the unequal center scales required by HD+;
- it preserves a radial operator that is diagonal at each `(r_i, R_j)` point;
- its output is independent of the chosen angular channel basis;
- it is already exercised by the serial and distributed Hamiltonian paths.

Keep the analytic multipole implementation as a small-grid and convergence
oracle. Do not make direct Gauss-Lobatto sampling of the Coulomb singularity a
production alternative.

The coupled implementation must nevertheless revalidate the Poisson path for
all center scales and retained multipoles. In particular, H2+ and D2+ cancel
odd `lambda` contributions, whereas HD+ generally retains both even and odd
`lambda` contributions in nuclear-center-of-mass Jacobi coordinates.

## 2. Coordinates and field-free Hamiltonian

Let particles `A` and `B` be the two nuclei and let `e` denote the electron.
Use the same translationally invariant Jacobi coordinates as the existing
non-Born-Oppenheimer implementation:

```math
\mathbf R = \mathbf R_B-\mathbf R_A,
\qquad
\mathbf r = \mathbf r_e-
\frac{m_A\mathbf R_A+m_B\mathbf R_B}{m_A+m_B}.
```

The nuclear positions relative to their center of mass are

```math
\mathbf R_A=c_A\mathbf R,
\qquad
\mathbf R_B=c_B\mathbf R,
\qquad
c_A=-\frac{m_B}{m_A+m_B},
\qquad
c_B= \frac{m_A}{m_A+m_B}.
```

After removal of the total center-of-mass motion, the field-free Hamiltonian is

```math
\hat H_0=
-\frac{1}{2\mu_r}\nabla_{\mathbf r}^{2}
-\frac{1}{2\mu_R}\nabla_{\mathbf R}^{2}
+\frac{Z_A Z_B}{R}
+\sum_{a\in\{A,B\}}
\frac{q_e Z_a}{|\mathbf r-c_a\mathbf R|}.
```

Masses, reduced masses, effective field charges, center scales, and CODATA
selection remain the responsibility of `matter-field-params` and the existing
diatomic parameter layer. The coupled representation must not duplicate or
special-case those numerical constants.

For the reduced radial wavefunction used by grid-tdse, the radial kinetic
actions contain `d^2/dr^2` and `d^2/dR^2`, while the angular kinetic terms are

```math
\frac{l_1(l_1+1)}{2\mu_r r^2}
+\frac{l_2(l_2+1)}{2\mu_R R^2}.
```

Here `l_1` is the electronic orbital angular momentum, `l_2` is the
internuclear rotational angular momentum, and `L` is their coupled total
angular momentum.

## 3. Explicit coupled wavefunction parametrization

### 3.1 Bipolar harmonics

Adopt Condon-Shortley spherical harmonics and the Clebsch-Gordan convention
used by the selected Wigner-symbol backend. Define

```math
\mathcal Y_{l_1l_2}^{LM}(\hat{\mathbf r},\hat{\mathbf R})
=
\sum_{m_1=-l_1}^{l_1}\sum_{m_2=-l_2}^{l_2}
\langle l_1m_1,l_2m_2|LM\rangle
Y_{l_1m_1}(\hat{\mathbf r})Y_{l_2m_2}(\hat{\mathbf R}).
```

Only terms satisfying

```math
m_1+m_2=M,
\qquad
|l_1-l_2|\le L\le l_1+l_2
```

are present.

### 3.2 Wavefunction

The full space-fixed wavefunction is parametrized as

```math
\Psi(\mathbf r,\mathbf R,t)
=
\frac{1}{rR}
\sum_{\alpha}
\psi_{\alpha}(r,R,t)
\mathcal Y_{l_1l_2}^{LM}(\hat{\mathbf r},\hat{\mathbf R}),
\qquad
\alpha=(l_1,l_2,L,M).
```

With orthonormal angular functions and the current reduced FEMDVR radial
convention,

```math
\langle\Psi|\Psi\rangle
=
\sum_{\alpha}\int_0^\infty dr\int_0^\infty dR
|\psi_{\alpha}(r,R,t)|^2,
```

where the implementation evaluates the radial integrals with the existing
FEMDVR quadrature and bridge-point conventions.

The in-memory serial shape should be

```text
(coupled_channel, electronic_radial_point, internuclear_radial_point)
```

with an explicit immutable channel table containing `(l_1, l_2, L, M)` for every
channel. MPI layouts continue to distribute only the two radial axes initially;
all coupled channels remain local to a rank.

### 3.3 Truncations

Expose separate truncations:

- `l1_max`: maximum electronic orbital angular momentum `l_1`;
- `l2_max`: maximum internuclear rotational angular momentum `l_2`;
- `L_max`: maximum coupled total angular momentum `L`;
- `M_tot`: selected total projection `M`;
- `lambda_max`: Coulomb multipole cutoff.

The public keyword names and mathematical indices deliberately follow the
coupling pattern `l_1 + l_2 -> L`:

```text
l1_max <-> l_1,max
l2_max <-> l_2,max
L_max  <-> L_max
M_tot  <-> M
```

Do not silently set all angular cutoffs equal. A convenience policy may propose
defaults, but the representation metadata must record every realized cutoff.
For a complete projection of the Coulomb interaction within finite `l_1` and `l_2`
spaces, no multipoles beyond

```math
\lambda_{\mathrm{complete}}=2\min(l_{1,\max},l_{2,\max})
```

can contribute. A lower `lambda_max` is an additional physical/numerical
approximation and must be explicit.

For fixed `M_tot`, retaining every allowed `L` is a unitary recoupling of the
existing uncoupled channel pairs satisfying `m_1+m_2=M`. This equality of spaces
is central to validation.

## 4. Symmetry sectors for H2+, HD+, and D2+

### 4.1 Field-free rotation

The field-free Hamiltonian is a rotational scalar. It is block diagonal in
`L` and `M`, and independent of `M` within numerical precision. Field-free
ground-state calculations should therefore operate in one `(L, M)` block. The
isotropic rovibronic ground state is naturally represented in `L=0`, `M=0`.

A linearly polarized field along the laboratory `z` axis preserves `M` but
mixes allowed `L` values. General polarization may mix `M` sectors and is a
later extension.

### 4.2 Inversion

Simultaneous inversion of both Jacobi vectors gives

```math
\mathcal P\mathcal Y_{l_1l_2}^{LM}
=(-1)^{l_1+l_2}\mathcal Y_{l_1l_2}^{LM}.
```

All three isotopologues may therefore be organized by total spatial inversion
parity in the field-free problem. An electric-dipole interaction connects
opposite inversion parities.

### 4.3 Nuclear exchange

For H2+ and D2+, exchange of the identical nuclei maps
`\mathbf R -> -\mathbf R` while leaving the electronic Jacobi coordinate
unchanged. The spatial exchange eigenvalue is

```math
p_{AB}=(-1)^{l_2}.
```

The coupled basis should be constructed directly in a requested exchange
sector rather than projecting a larger state after every operation. Combining
exchange and inversion fixes the allowed parity of `l_1`.

HD+ has no nuclear-exchange symmetry and must retain both parities of `l_2`.
Because its two center scales have unequal magnitudes, its Coulomb interaction
also retains odd multipoles. This makes HD+ an essential validation case, not a
minor parameter variation.

Nuclear-spin statistical weights are outside the first implementation. The
archive and user interface should call this quantity *spatial nuclear-exchange
parity* to avoid implying that nuclear spin is represented.

## 5. Coulomb interaction in the coupled basis

### 5.1 Radial representation

For one center `a`, define

```math
r_{<,a}=\min(r,|c_a|R),
\qquad
r_{>,a}=\max(r,|c_a|R).
```

The two-center radial multipole is

```math
W_\lambda(r,R)
=
\frac{4\pi}{2\lambda+1}
\sum_{a\in\{A,B\}}
q_eZ_a\,\operatorname{sgn}(c_a)^\lambda
\frac{r_{<,a}^{\lambda}}{r_{>,a}^{\lambda+1}}.
```

The production implementation should continue to obtain the discretized
`W_lambda(r_i,R_j)` through the two-grid FEMDVR Poisson solver rather than by
sampling this expression directly.

The full electron-nuclear interaction is

```math
\hat V_{eN}
=
\sum_{\lambda=0}^{\lambda_{\max}}
W_\lambda(r,R)
\sum_{\mu=-\lambda}^{\lambda}
Y_{\lambda\mu}^{*}(\hat{\mathbf r})
Y_{\lambda\mu}(\hat{\mathbf R}).
```

For H2+ and D2+, equal charges and center scales `c_A=-1/2`, `c_B=1/2`
eliminate all odd `lambda`. No such reduction is valid for HD+ in the
nuclear-center-of-mass coordinates.

### 5.2 Coupled angular coefficients

Use the current one-coordinate Gaunt conventions as the primitive definition:

```math
g^{(r)}_{l_1'm_1',l_1m_1}(\lambda,\mu)
=
\langle l_1'm_1'|Y_{\lambda\mu}^{*}|l_1m_1\rangle,
```

```math
g^{(R)}_{l_2'm_2',l_2m_2}(\lambda,\mu)
=
\langle l_2'm_2'|Y_{\lambda\mu}|l_2m_2\rangle.
```

The coupled coefficient is defined unambiguously by

```math
C^{\lambda}_{\alpha'\alpha}
=
\sum_{m_1',m_2',m_1,m_2,\mu}
\langle l_1'm_1',l_2'm_2'|L'M'\rangle
\langle l_1m_1,l_2m_2|LM\rangle
g^{(r)}_{l_1'm_1',l_1m_1}(\lambda,\mu)
g^{(R)}_{l_2'm_2',l_2m_2}(\lambda,\mu).
```

All coefficients are real for the adopted conventions, but the implementation
should verify this rather than assume it during derivation.

The same expression may be reduced analytically to products of reduced matrix
elements and a Wigner 6-j symbol. The optimized builder should use the reduced
formula only after it has been checked coefficient by coefficient against the
Clebsch-Gordan sum above. The direct sum is the test oracle and fixes all phase
and normalization conventions.

Because the Coulomb operator is a rotational scalar,

```math
C^{\lambda}_{\alpha'\alpha}
\propto\delta_{L'L}\delta_{M'M}.
```

The remaining selection rules include the triangle conditions for
`(l_1,l_1',lambda)` and `(l_2,l_2',lambda)` and the parity rules

```math
l_1+l_1'+\lambda\ \text{even},
\qquad
l_2+l_2'+\lambda\ \text{even}.
```

At each radial point, the action is

```math
(V_{eN}\psi)_{\alpha'}(r_i,R_j)
=
\sum_{\lambda,\alpha}
C^{\lambda}_{\alpha'\alpha}
W_\lambda(r_i,R_j)
\psi_{\alpha}(r_i,R_j).
```

This has the same computational form as the current sparse, output-grouped
Coulomb action. The new coupling table should therefore reuse the existing
kernel data model where possible: output channel, input channel, radial
multipole row, coefficient, and output-group offsets.

### 5.3 Other field-free terms

The electronic and internuclear kinetic operators are diagonal in
`(l_1,l_2,L,M)`
apart from their radial derivative actions. Nuclear repulsion `Z_A Z_B/R` is
diagonal in every angular channel. Radial masks and CAPs are also angularly
diagonal and require no new angular algebra.

## 6. Electromagnetic interaction

The first dynamics milestone should support a laboratory-frame, linearly
`z`-polarized field in the length gauge. In Jacobi coordinates the total
dipole has electronic and internuclear rank-one contributions,

```math
\hat d_q=\xi_r r\,C_q^{(1)}(\hat{\mathbf r})
+\xi_R R\,C_q^{(1)}(\hat{\mathbf R}),
```

where `C_q^(1) = sqrt(4*pi/3) Y_(1q)` is the Racah-normalized spherical
tensor and the effective charges are supplied by the existing diatomic
parameters. The
internuclear contribution vanishes for homonuclear H2+ and D2+ but is generally
nonzero for HD+.

For `q=0`, `M` is conserved and the rank-one operator connects the allowed
`L` blocks. The angular coefficients should again be constructed in two ways:
an explicit Clebsch-Gordan transform as an oracle and a reduced Wigner-symbol
formula for production.

Velocity gauge should follow only after length-gauge field-free and real-time
equivalence tests pass. Arbitrary polarization and multiple `M` sectors are
later extensions.

## 7. Proposed software structure

### 7.1 Preserve the current representation

Do not replace or reinterpret `AngularBasisSpec` or `AngularPairBasis`. They
remain the uncoupled reference representation and a supported public path.

The existing `TwoRadialSystem` should remain the prepared uncoupled
representation so that current callers, state layouts, and archives do not
change meaning. Introduce a separate prepared coupled representation,
provisionally named `CoupledTwoRadialSystem`, rather than adding pervasive
`if coupled` branches to `TwoRadialSystem`.

Introduce a distinct immutable coupled-channel specification, provisionally
named `CoupledAngularBasisSpec`, with:

- explicit ordered states `(l_1, l_2, L, M)`;
- separate `l1_max`, `l2_max`, and `L_max`;
- selected `M_tot` sector;
- optional inversion and spatial exchange sectors;
- deterministic lookup from a state tuple to its channel index;
- complete metadata serialization.

The exact class name is less important than keeping coupled and uncoupled
quantum numbers impossible to confuse.

Both prepared representations should satisfy a small shared protocol used by
high-level orchestration. That protocol should expose only genuinely common
operations and metadata, such as:

- `psi_shape` and an ordered channel description;
- radial grids and quadrature-aware `inner_product`;
- state validation and allocation;
- Hamiltonian construction;
- observable and archive metadata hooks.

The angular basis, operator builders, symmetry rules, and state conversion
remain representation-specific objects. Polymorphism should be resolved when
the calculation is configured, not repeatedly inside radial or Coulomb hot
kernels.

### 7.2 Transformation oracle

Provide a sparse or structured transform between the fixed-`M` uncoupled
product basis and the coupled basis:

```math
\psi_{l_1l_2LM}
=
\sum_{m_1,m_2}
\langle l_1m_1,l_2m_2|LM\rangle
\psi^{(\mathrm{unc})}_{l_1m_1,l_2m_2}.
```

This transform is required for tests, archive conversion tools, and diagnostic
comparisons. It must not be applied inside every production Hamiltonian action.

### 7.3 Generic channel coupling storage

Generalize the sparse Coulomb coupling container only as far as necessary to
accept one channel index rather than two uncoupled coordinate-channel indices.
Keep the radial multipole object and element-local radial operators unchanged.

Avoid constructing:

- the dense coupled Coulomb matrix for every `lambda`;
- a dense uncoupled-to-coupled transform in production runs;
- separate radial multipoles for each `(L,M)` block;
- expanded angular-pair arrays on every MPI rank when sparse transition lists
  are sufficient.

### 7.4 High-level interface

Add an explicit representation selector rather than changing defaults
silently. Use the same keyword in the ground-state and real-time interfaces:

```text
wf_representation="uncoupled" | "coupled"
l1_max=...
l2_max=...
L_max=...
M_tot=...
inversion_parity=...
pair_exchange_parity=...
lambda_max=...
```

`wf_representation="uncoupled"` must be the default for backward
compatibility. The high-level constructor should validate representation-
specific options and dispatch once to an uncoupled or coupled representation
factory. Downstream ground-state and propagation orchestration should then
operate on the returned representation protocol rather than inspect the
selector repeatedly.

The existing `l_max`, `m_max`, and `M_tot` options remain valid for the
uncoupled path. The coupled path uses explicit `l1_max`, `l2_max`, `L_max`,
and `M_tot` options. Because `M_tot` is meaningful in both representations,
it retains the same physical definition: the total laboratory-frame angular
momentum projection. Passing other coupled-only
options with `wf_representation="uncoupled"`, or uncoupled-only options with
`wf_representation="coupled"`, should raise a clear error instead of being
silently ignored. Deprecation of the uncoupled option, if ever appropriate, is
a separate decision after production validation.

### 7.5 State archives

Extend representation metadata with:

- `wf_representation: "coupled"`;
- `angular_basis: "bipolar-spherical-harmonics"`;
- the ordered `(l_1,l_2,L,M)` table;
- all angular cutoffs and symmetry sectors;
- spherical-harmonic, Clebsch-Gordan, and phase conventions;
- the radial Coulomb representation and multipole truncation.

Never interpret an uncoupled archive as coupled merely because its array shape
matches. Conversion must be explicit, validated, and checksum-protected.

### 7.6 MPI compatibility

The Cartesian radial decomposition can initially be reused unchanged because
the coupled channel axis remains local. Distributed norms, reductions, radial
halos, element-local derivatives, CAPs, and rolling checkpoints should require
only representation-aware metadata and channel iteration.

Angular storage and Coulomb transition counts may become the next per-rank
memory bottleneck. Measure them before considering angular-channel
distribution across MPI ranks.

## 8. Dynamics-first step-by-step implementation plan

### 8.1 First production target

The first end-to-end target is deliberately narrower than the complete
coupled-angular formulation:

- H2+, HD+, and D2+ in nuclear-center-of-mass Jacobi coordinates;
- an isotropic field-free initial state in `L=0`, `M=0`;
- a spatially uniform, linearly `z`-polarized electric field;
- the electric-dipole approximation in the length gauge;
- a fixed `M_tot` sector, initially `M_tot=0`;
- the existing FEMDVR grids, Poisson Coulomb multipoles, radial CAPs, and SIA
  propagation;
- serial correctness first, followed by the established Cartesian radial MPI
  decomposition.

The time-dependent Hamiltonian for this milestone is

```math
\hat H(t)
=
\hat H_0+\hat V_{\mathrm{CAP}}
-E_z(t)\hat d_0,
```

with

```math
\hat d_0
=
\xi_r r C_0^{(1)}(\hat{\mathbf r})
+\xi_R R C_0^{(1)}(\hat{\mathbf R}).
```

The reduced charges `xi_r` and `xi_R` must come from
`DiatomicIonParameters`. For H2+ and D2+, `xi_R=0`; for HD+, both terms are
generally present. The interaction sign must remain identical to the existing
uncoupled `TwoRadialAxialDipoleInteraction` convention.

The implementation dependency order is

```text
conventions -> coupled channels -> transformation oracle
            -> field-free action -> ground state
            -> length-gauge dipole -> serial pulse propagation
            -> observables/restart -> isotope qualification
            -> velocity gauge -> Cartesian MPI -> production benchmarks
```

Each numbered step below should normally be a separate reviewable change. A
step is complete only when its exit gate passes; later steps must not be used
to conceal a failure in an earlier layer.

### Step 0: freeze the dynamics contract and reference cases

Before adding classes, record the exact spherical-harmonic,
Clebsch-Gordan, Wigner-symbol, field-sign, and gauge conventions in the module
docstrings and tests. Fix a small reference grid and angular space that can be
run in both representations. Define reference cases for:

1. zero field without a CAP;
2. a weak, short length-gauge pulse without a CAP;
3. a finite pulse with electronic and internuclear CAPs;
4. H2+, D2+, and HD+ with the same numerical truncations.

Store the expected uncoupled matrices or actions rather than only final
energies. Set numerical tolerances separately for coefficient comparisons,
Hamiltonian actions, propagated states, norms, and sampled observables.

Exit gate: the reference inputs, expected metadata, comparison norms, and
tolerances are explicit and do not depend on the future coupled code.

### Step 1: implement coupled-channel enumeration

Status: implemented and covered by focused unit tests.

Add `CoupledAngularBasisSpec` and an immutable channel table with states
`(l_1,l_2,L,M)`. Its public inputs are `l1_max`, `l2_max`, `L_max`, `M_tot`,
inversion parity, and optional spatial nuclear-exchange parity. Enumeration
must apply, in this order:

1. the individual `l_1` and `l_2` cutoffs;
2. `|l_1-l_2| <= L <= l_1+l_2`;
3. `L <= L_max` and `|M_tot| <= L`;
4. inversion parity `(-1)^(l_1+l_2)` when selected;
5. homonuclear exchange parity `(-1)^l_2` when selected.

HD+ must reject a requested exchange sector. H2+ and D2+ must support both
spatial exchange sectors even though the lowest physical rotational state is
normally sought in the even sector.

Exit gate: ordering, dimensions, lookup, symmetry filtering, serialization,
and invalid-input behavior are exhaustively unit tested at small cutoffs.

### Step 2: build the coupled/uncoupled transformation oracle

Status: implemented and covered by focused unit tests.

Implement a structured fixed-`M` transformation using the
Clebsch-Gordan coefficients, without introducing it into any hot Hamiltonian
path. Support vectors and batches over arbitrary trailing radial axes.

Use it to verify:

- orthonormality and completeness when both bases span the same finite space;
- forward/inverse round trips;
- preservation of norm, inner products, radial marginal densities, inversion,
  and exchange expectation values;
- correct projection behavior when the coupled space is deliberately
  truncated in `L`.

Exit gate: full equivalent spaces transform unitarily to machine precision,
and projected spaces report rather than hide discarded norm.

The implementation is `CoupledAngularTransform`. Its uncoupled ordering
matches the existing compact `AngularPairBasis` ordering, its dense
coefficient matrix is confined to the validation oracle, and
`CoupledAngularProjection` reports retained and discarded norm explicitly for
an `L`-truncated space.

### Step 3: introduce the coupled prepared representation

Status: implemented and covered by focused unit tests.

Add a separate `CoupledTwoRadialSystem`; do not change the meaning or storage
of `TwoRadialSystem`. Define the smallest shared representation protocol
needed by ground-state and real-time orchestration. Replace strict
`isinstance(TwoRadialSystem)` checks only where the operation is genuinely
representation-independent.

The coupled serial state layout is

```text
(coupled_channel, electronic_radial_point, internuclear_radial_point).
```

The object owns the coupled channel table, radial grids, weights, radial
multipoles, state validation, allocation, and inner product. It must not own a
dense uncoupled transformation or dense angular matrices per radial point.

Exit gate: zero states, random states, norms, copies, metadata, and storage
estimates work without constructing a Hamiltonian.

`TwoRadialSystemSpec.build_coupled()` prepares the radial grids and reuses the
existing FEMDVR or analytic radial Coulomb construction. The returned
`CoupledTwoRadialSystem` implements the minimal `TwoRadialRepresentation`
protocol and retains no Hamiltonian, dense angular transform, or uncoupled
coupling table.

### Step 4: implement diagonal field-free terms

Status: implemented and covered by focused unit tests.

Implement the electronic and internuclear radial kinetic actions, centrifugal
terms, and nuclear repulsion. The angular diagonal for channel
`(l_1,l_2,L,M)` is fixed by `l_1(l_1+1)` and `l_2(l_2+1)`; it must not depend
on `L` except through channel availability.

Start with a simple global radial test oracle if useful, but make the
element-local FEMDVR action the production path. Reuse the existing radial
operator builders rather than copying their numerical formulas.

Exit gate: every term agrees with the transformed uncoupled action on random
states for multiple cutoffs and independent electronic/internuclear grids.

`CoupledTwoRadialOneBodyHamiltonian` provides separate radial-kinetic,
centrifugal, and angular-independent radial-potential actions for both
coordinates. The production path uses the element-local FEMDVR derivative;
the optional dense global path is retained only as a small-system oracle.
Angular-dependent uncoupled one-body terms are rejected explicitly because
individual magnetic projections are not quantum numbers in this basis.

### Step 5: implement the coupled Coulomb action

Status: implemented and covered by focused unit tests.

Reuse the existing `RadialCoulombMultipoles` data unchanged. First implement
the direct Clebsch-Gordan sum in Section 5.2 as a coefficient oracle. Then add
the reduced Wigner-symbol builder and sparse output-grouped transition table.

Test H2+ and D2+ for exact removal of odd `lambda` rows. Test HD+ with unequal
center scales and nonzero allowed odd multipoles. The scalar Coulomb operator
must preserve `L`, `M`, inversion parity, and homonuclear exchange parity.

Exit gate: direct and reduced coefficients agree, the table is Hermitian, and
the complete Coulomb action agrees with the transformed uncoupled action.

`CoupledTwoRadialCoulombHamiltonian` reuses the prepared
`RadialCoulombMultipoles` without modification. Its production coupling
builder evaluates the scalar angular coefficient through reduced
spherical-harmonic matrix elements and a Wigner 6-j symbol, while the direct
Clebsch-Gordan sum remains a public validation oracle. The resulting real
transition table is sparse and output-grouped, and its point-local action does
not retain dense angular matrices. A coupled scalar interaction requires all
magnetic components through the largest retained `lambda`; truncated `mu`
expansions are rejected explicitly.

### Step 6: complete and validate the field-free Hamiltonian

Status: implemented and covered by focused unit tests.

Compose the terms from Steps 4 and 5 behind the same callable Hamiltonian
contract used by real-time propagation. Add field-free energy evaluation and
the coupled versions of the preconditioner inputs required by Davidson and
SIA. Do not add the laser interaction yet.

Required tests include:

- Hermiticity without a CAP;
- the expected non-Hermiticity with a CAP;
- invariance of the spectrum with respect to `M` for a complete fixed-`L`
  space;
- term-by-term and total-action agreement with the uncoupled representation;
- global versus element-local radial-action agreement.

Exit gate: the field-free action is a validated replacement for the uncoupled
action at equivalent truncations.

`CoupledTwoRadialHamiltonian` composes the one-body and Coulomb actions behind
the standard callable contract. It distinguishes the physical field-free
action from the static propagation action when a radial CAP is configured, so
physical energy evaluation always excludes the CAP. Separate component
actions and the complete coefficient-space diagonal are available for
diagnostics and preconditioning. `davidson_diagonal()` follows the existing
one-body default and can optionally include the point-local Coulomb diagonal.
SIA consumes the complete callable action directly and requires no separate
preconditioner object.

### Step 7: connect field-free initial states

Status: implemented and covered by focused unit tests.

Extend the Davidson ground-state path to the coupled representation. Begin
with `L=0`, `M=0` and no CAP. Reuse isotope parameters and archive
compatibility checks from `NonBornOppenheimerDiatomicIonGroundState`.

Keep this step a fixed-space implementation and representation-validation
task. Do not attempt physical convergence with respect to either radial grid,
`l1_max`, or `l2_max`. Systematic radial and angular convergence belongs to a
later qualification step after the field interaction and propagation path are
available.

Use the same deliberately small FEMDVR grids as the existing uncoupled
Davidson tests:

```python
electronic_grid = FEMDVRGridSpec([0.0, 2.0, 5.0], [3, 3])
internuclear_grid = FEMDVRGridSpec([0.0, 1.5, 4.0], [3, 3])
```

Use fixed `l1_max = l2_max = 2`, `L_max = 0`, `M_tot = 0`, and
`lambda_max = 4` for the initial validation cases. These choices include the
complete Coulomb projection within the retained `(l_1,l_2)` rectangle but
deliberately select only its `L=0` block. Use positive inversion parity for
all three isotopes, positive spatial nuclear-exchange parity for H2+ and D2+,
and no exchange restriction for HD+.

For each isotope, compare energy, Davidson residual, norm, radial densities,
`L^2`, `L_z`, inversion, and spatial exchange where defined. Compare the
coupled state with the explicit `L=0` projection of the uncoupled
`M_tot=0` state, including the retained and discarded projection norms. The
uncoupled ground state is expected to have zero discarded norm within the
solver tolerance because the documented ground states have `L=0`. Do not
describe agreement on these small grids as radial or angular convergence.

The finite-space equivalence conditions, and the distinction between the
complete uncoupled `M_tot=0` space and its coupled `L=0` block, are specified
in [`coupled_vs_uncoupled.md`](../docs/coupled_vs_uncoupled.md). Tests in this
step must follow those conditions rather than comparing coefficient arrays
without the Clebsch--Gordan transformation.

Add
`examples/spherical/two_radial/coupled_gs_nbo_diatomics.py` as a compact
serial example. It must demonstrate Davidson ground-state calculations for
H2+, HD+, and D2+. Unlike the deliberately tiny unit-test grids above, use
four 11-point electronic elements with nodes `[0, 4, 8, 12, 16]` and two
15-point internuclear elements with nodes `[0, 4, 8]`. Retain the fixed
`l1_max = l2_max = 2`, `L_max = 0`, `M_tot = 0`, and `lambda_max = 4`
angular setup. Print at least the isotope, energy, residual norm, wavefunction
norm, iteration count, termination reason, and solver-convergence status.
State prominently that the example is not a radial or angular convergence
study and that its finite-basis energies are not convergence claims.

Ground-state archives must record the coupled representation metadata,
isotope and symmetry selections, all fixed cutoffs, grid specifications,
Davidson tolerances, residual, and solver-convergence status. A dynamics
calculation must refuse an archive whose Davidson solve did not converge
unless the caller explicitly selects the existing unsafe override policy.
This check concerns eigensolver convergence only; it must not imply that the
fixed radial or angular basis is physically converged.

Exit gate: the coupled Davidson interface and the example run successfully
for H2+, HD+, and D2+; every solve satisfies the established residual and
wavefunction tolerances; and the coupled results agree with the transformed
or explicitly `L=0`-projected uncoupled results at the documented numerical
tolerances. No radial-grid or `l1_max`/`l2_max` convergence claim is required.

`NonBornOppenheimerDiatomicIonGroundState` now dispatches once from the
explicit `wf_representation` selector while preserving `"uncoupled"` as the
default. The coupled path uses `CoupledTwoRadialSystem`, the generic Davidson
solver, and the field-free diagonal supplied in Step 6. Coupled archives use a
distinct checksum-protected format and expose an explicit solver-convergence
gate, so their arrays cannot be interpreted as uncoupled states accidentally.
The three-isotope tests use the small uncoupled-test grids to compare the
coupled state with the transformed `L=0` uncoupled result and verify
negligible discarded projection norm. The public example uses the larger
fixed grids specified above.

### Step 8: implement the length-gauge angular dipole builders

Create separate electronic and internuclear rank-one coupling tables. The
field-independent matrix elements are

```math
D^{(r,q)}_{\alpha'\alpha}
=
\xi_r\left\langle(l_1'l_2')L'M'
\middle|C_q^{(1)}(\hat{\mathbf r})
\middle|(l_1l_2)LM\right\rangle,
```

```math
D^{(R,q)}_{\alpha'\alpha}
=
\xi_R\left\langle(l_1'l_2')L'M'
\middle|C_q^{(1)}(\hat{\mathbf R})
\middle|(l_1l_2)LM\right\rangle.
```

The coordinate-specific selection rules are

```math
\begin{aligned}
\text{electronic:}\quad &l_2'=l_2, &l_1'&=l_1\pm1,\\
\text{internuclear:}\quad &l_1'=l_1, &l_2'&=l_2\pm1.
\end{aligned}
```

Both terms obey `M'=M+q`, the rank-one triangle rule in `L`, exclusion of an
`L=0 -> L'=0` transition, and reversal of total inversion parity. For `q=0`,
`M` is conserved.

For H2+ and D2+, the internuclear reduced charge is zero, so the field changes
`l_1` while preserving `l_2` and spatial exchange parity. For HD+, both
coordinate terms must be retained and tested independently.

As with the Coulomb coefficients, use an explicit Clebsch-Gordan transform as
the normative oracle before enabling a reduced Wigner-symbol builder.

Exit gate: both dipole tables agree coefficient by coefficient and as actions
with transformed uncoupled reference operators for all three isotopologues.

This step is implemented in `coupled_dipole.py`. The direct builder evaluates
the defining Clebsch--Gordan sums and remains the convention oracle. The
production builder uses coordinate-specific reduced matrix elements and a
Wigner 6-j symbol. Both support `q=-1,0,1`, distinct input and output
fixed-`M_tot` bases, effective-charge scaling, sparse output-grouped storage,
and an angular action on arrays with arbitrary trailing axes. Electronic and
internuclear tables are built independently with `coordinate=1` and
`coordinate=2`, respectively. The radial factors and the overall
length-gauge interaction sign remain intentionally deferred to Step 9.

The Step 8 tests compare the direct and reduced coefficients and complete
sparse tables for every spherical component, verify the rank-one, inversion,
and spectator-coordinate selection rules, and compare radialized `q=0`
actions with transformed uncoupled `AxialCoordinateDipole` actions. H2+ and
D2+ exercise an exactly empty internuclear table because their effective
internuclear charge vanishes; HD+ exercises both nonzero coordinate tables.

### Step 9: implement the serial length-gauge interaction

Add a coupled counterpart of `TwoRadialAxialDipoleInteraction` with a
field-independent `spatial_action(state)` and a cheap scalar field-amplitude
evaluation. Apply the radial factors `r` and `R` only to their corresponding
coupling tables. Preserve the current convention for multiple axial field
profiles and the sign of `-E_z(t)d_0`.

Keep the interaction representation-specific, but expose it through the same
time-dependent Hamiltonian protocol. Do not branch on `wf_representation`
inside its action kernel.

Exit gate: zero-field, electronic-only, internuclear-only, combined, and
multiple-field actions match explicit reference matrices and the transformed
uncoupled action.

This step is implemented by `CoupledAxialCoordinateDipole` and
`CoupledTwoRadialAxialDipoleInteraction` in `coupled_dipole.py`. Each
coordinate operator combines its Step 8 sparse angular table with only the
matching radial DVR coordinate and applies the established minus sign. The
combined interaction exposes the same `field_amplitude(time)`,
`spatial_action(state)`, and time-dependent `apply(state, time)` protocol as
the uncoupled interaction, including summation of multiple axial field
profiles. It accepts length gauge explicitly and rejects velocity gauge until
the corresponding coupled momentum operator is implemented.

The Step 9 tests cover zero field, electronic-only, internuclear-only,
combined, and multiple-field actions. They compare coordinate actions with
explicit sparse matrices and compare the complete action with transformed
uncoupled operators for H2+, HD+, and D2+. Field-coupled bases must retain
both inversion parities; a nonzero internuclear term must also retain both
pair-exchange parities.

### Step 10: expose the coupled high-level dynamics interface

Add `wf_representation="uncoupled" | "coupled"` to
`NonBornOppenheimerDiatomicIonTDSE` and its ground-state initializer. Dispatch
once during construction. The coupled path accepts `l1_max`, `l2_max`,
`L_max`, `M_tot`, `lambda_max`, inversion metadata, and exchange parity; it
must reject ambiguous mixtures of coupled-only and uncoupled-only options.

The first supported field combination is length gauge with axial profiles.
Validate that `M_tot` is compatible with every retained channel and that a
field-coupled space contains both inversion parities. A symmetry-restricted
ground-state archive may be embedded into the larger dynamics space, but the
embedding must be explicit and norm preserving.

Exit gate: existing uncoupled callers are unchanged, while a coupled
calculation can be configured entirely through the public high-level API.

### Step 11: validate serial pulse propagation with SIA

Use SIA as the first propagator because a CAP makes the Hamiltonian
non-Hermitian. Exercise the following sequence before any performance work:

1. zero-field propagation without a CAP: norm and field-free energy remain
   constant and the eigenstate changes only by phase;
2. weak-pulse propagation without a CAP: coupled and uncoupled states agree
   after transforming to the same basis;
3. finite-pulse propagation without a CAP: dipole and angular populations
   agree throughout the trajectory;
4. finite-pulse propagation with CAPs: raw norm loss and conditional
   observables agree;
5. time-step and Krylov-tolerance convergence.

After SIA passes, connect CF4-SIA using its linear-combination factory and the
field-independent dipole action. Compare SIA and CF4-SIA only after each is
independently time-step converged.

Exit gate: all three isotopologues pass state-level and observable-level
comparisons for the serial reference pulses.

### Step 12: add dynamics observables and angular populations

Generalize `DiatomicIonDynamicsObservableEvaluator` through the shared
representation protocol. Sample both raw numerators and explicitly named
conditional expectations when a CAP is active. At minimum provide:

- norm, CAP norm loss, and field-free energy numerator;
- electronic, internuclear, and total dipoles;
- `<L^2>`, `<L_z>`, `<l_1^2>`, and `<l_2^2>`;
- populations resolved by `L`, `l_1`, and `l_2`;
- inversion expectation and homonuclear spatial exchange expectation;
- existing radial moments, alignment moments, and `P(r)`, `P(R)` densities.

The sampled total dipole must be directly usable for later spectral or HHG
post-processing. Dipole acceleration should be a later, separately validated
observable rather than a numerical second derivative hidden in the propagator.

Exit gate: every observable agrees with its uncoupled transformed reference
at the initial state and at selected times during a pulse.

### Step 13: make archives, snapshots, and restart representation-safe

Extend ground-state archives, wavefunction snapshots, and rolling checkpoints
with the coupled channel table, all four angular cutoffs, `M_tot`, symmetry
sectors, field/gauge metadata, and phase conventions. Keep only the newest
rolling checkpoint according to the existing retention behavior; requested
scientific snapshots remain independent files.

Test uninterrupted versus restarted trajectories, restarting on a different
compatible radial process grid, and rejection of incompatible coupled or
uncoupled archives. A restart must reconstruct the same field phase and
accepted-step count.

Exit gate: restarted serial trajectories reproduce uninterrupted states and
time series to the established tolerance.

### Step 14: qualify isotope-specific laser physics

Run one controlled convergence matrix for each isotope:

- H2+: electronic dipole only, even/odd inversion mixing, fixed exchange
  sector;
- D2+: the same symmetry structure as H2+ but with the deuteron reduced mass;
- HD+: electronic plus internuclear dipoles, no exchange restriction, and
  even plus odd Coulomb multipoles.

For each case vary `l1_max`, `l2_max`, `L_max`, `lambda_max`, radial boxes,
FEMDVR order, CAP onset/strength, time step, and Krylov tolerance. Compare
absorbed norm, final angular populations, radial densities, and dipole time
series—not only total energy.

Exit gate: the implementation has documented convergence envelopes and no
isotope relies accidentally on a symmetry valid only for another isotope.

### Step 15: add velocity gauge as a cross-check

Only after length-gauge qualification, implement the coupled electronic and
internuclear momentum actions using the same reduced velocity couplings as the
uncoupled path. Preserve the existing convention that omits the spatially
uniform `A(t)^2` scalar term, which affects only global phase. Require a pulse
with compatible initial vector potential.

Exit gate: length- and velocity-gauge observables converge toward one another
as angular and radial spaces and time resolution are increased. Do not use
gauge agreement to replace comparison against the uncoupled implementation.

### Step 16: integrate Cartesian radial MPI

Keep the coupled channel axis local initially and reuse the permanent
two-dimensional radial ownership model. Add channel-aware local kinetic,
Coulomb, dipole, observable, and checkpoint actions. Reuse the batched Krylov
reductions already established for SIA and CF4-SIA.

Validation order is one MPI rank, multiple ranks on one node, and then
multiple nodes. Compare local term actions before full trajectories. Inventory
channel tables and transition arrays retained on every rank before increasing
angular cutoffs.

Exit gate: serial and MPI states, samples, CAP loss, and restarts agree, and
one-node qualification passes before any multi-node scaling claim.

### Step 17: optimize and qualify for production

Benchmark coupled and uncoupled representations at equivalent finite spaces.
Measure channel and transition counts, construction time, peak RSS,
Hamiltonian time by term, Krylov time, collective count, observables,
snapshots, and checkpoint overhead. Optimize only a measured bottleneck, while
retaining reference kernels behind test-only or diagnostic paths.

Archive small, production, and stress results on Olivia. The coupled
representation becomes a production recommendation only after demonstrating
a computational or scientific advantage without loss of accuracy or restart
reliability.

## 9. Required validation matrix

### Algebraic tests

- channel enumeration and lookup;
- Clebsch-Gordan orthogonality and transform unitarity;
- inversion and exchange eigenvalues;
- kinetic-energy diagonal structure;
- direct-sum versus reduced Wigner Coulomb coefficients;
- Hermiticity and selection rules;
- electronic and internuclear dipole rank-one selection rules;
- direct-sum versus reduced Wigner dipole coefficients;
- coupled dipole actions versus transformed uncoupled actions.

### Radial Coulomb tests

- preserve the current Poisson-versus-analytic tests;
- add unequal center scales representative of HD+;
- compare increasingly refined independent electronic and nuclear grids;
- test both even and odd multipoles;
- verify boundary behavior and convergence when `r_max` and `R_max` differ;
- verify that the coupled and uncoupled paths consume identical
  `W_lambda(r_i,R_j)` arrays.

### Physical tests

- H2+, HD+, and D2+ field-free ground states;
- unrestricted versus symmetry-adapted homonuclear calculations;
- degeneracy across `M` in the field-free problem;
- zero-field, weak-pulse, and finite-pulse coupled versus uncoupled
  trajectories;
- electronic, internuclear, and total dipoles;
- `L`-, `l_1`-, and `l_2`-resolved populations;
- CAP and no-CAP norm histories;
- SIA versus independently converged CF4-SIA trajectories;
- restart equivalence;
- length- versus velocity-gauge convergence after velocity gauge is added.

### Distributed tests

- thread-emulated ownership tests without MPI;
- two-rank and four-rank action equivalence;
- process-grid independence;
- one-node Olivia qualification before multi-node scaling;
- checkpoint portability and latest-only retention.

## 10. Risks and controls

### Phase conventions

Risk: a sign or normalization mismatch between Clebsch-Gordan, Gaunt, and
Wigner backends can produce plausible but incorrect matrices.

Control: make the explicit Clebsch-Gordan sum normative and compare every
optimized coefficient against it at small cutoffs.

### Coupled basis is not automatically smaller

Risk: fixed `M` with all allowed `L` is only a unitary basis change and has the
same dimension as the equivalent uncoupled space.

Control: claim memory or speed reductions only for justified `L`, parity, or
exchange restrictions and report comparisons at equivalent finite spaces.

### Truncation coupling

Risk: using one cutoff value for `l1_max`, `l2_max`, `L_max`, and `lambda_max`
can hide
convergence errors.

Control: represent and converge the four cutoffs independently.

### Field-driven symmetry space

Risk: a symmetry-adapted field-free ground state contains only one inversion
parity, while an electric-dipole interaction must access the opposite parity.
Using the ground-state channel table directly for dynamics would suppress the
laser response.

Control: construct the dynamics representation independently, retain every
field-connected inversion sector, and embed the ground state through an
explicit norm-preserving channel map.

### Strong-field angular truncation

Risk: a space adequate for the ground state may be inadequate during the
pulse. Population can accumulate at `L_max`, `l1_max`, or `l2_max` and produce
apparently converged radial observables with incorrect angular dynamics.

Control: sample boundary-channel populations and converge all angular cutoffs
against the complete dipole time series, absorbed norm, and final angular
populations.

### HD+ odd multipoles

Risk: carrying the homonuclear even-`lambda` restriction into HD+ removes
physical terms.

Control: make multipole parity a consequence of the center terms and validate
nonzero odd multipoles for HD+.

### HD+ internuclear dipole

Risk: reusing the homonuclear assumption `xi_R=0` for HD+ removes a physical
field coupling even if the asymmetric field-free Coulomb operator is correct.

Control: construct electronic and internuclear dipole tables separately from
the isotope parameters and test both contributions independently for HD+.

### CAP and propagator compatibility

Risk: using a Hermitian-only Lanczos path with a CAP can invalidate the Krylov
projection while still producing a smooth-looking trajectory.

Control: make SIA or CF4-SIA the default coupled dynamics path whenever a CAP
is active, retain the existing compatibility validation, and test raw norm
loss rather than renormalizing the propagated state.

### Coulomb construction cost

Risk: although the radial Poisson data are reusable, angular recoupling may
produce large transition tables at high cutoffs.

Control: group sparse transitions by output and `lambda`, use Wigner selection
rules before coefficient evaluation, cache reusable angular tables, and record
construction memory in benchmarks.

### Archive ambiguity

Risk: coupled and compact uncoupled states can both be three-dimensional.

Control: require explicit representation type and ordered channel metadata;
never infer representation from array rank or shape.

### Replicated angular storage

Risk: keeping all coupled channels and transition tables on every MPI rank may
become the dominant per-rank memory cost even after radial arrays are
distributed.

Control: record channel-table and transition-table bytes separately in every
scaling benchmark. Consider distributing angular channels only after measured
production cases show that replicated angular storage is limiting.

## 11. Definition of done

The first coupled-angular implementation is complete when:

1. the wavefunction and all angular conventions are explicit in code and
   archive metadata;
2. H2+, HD+, and D2+ field-free Hamiltonians agree with transformed uncoupled
   references;
3. the existing FEMDVR Poisson multipoles are reused without duplication and
   pass coupled/uncoupled equivalence tests;
4. homonuclear inversion and spatial exchange sectors are supported directly;
5. HD+ correctly retains asymmetric-center and odd-multipole physics as well
   as its nonzero internuclear dipole coupling;
6. serial Davidson ground states and length-gauge SIA pulse dynamics are
   validated against the uncoupled representation for all three isotopologues;
7. angular populations, coupled observables, snapshots, and rolling restart
   checkpoints are
   portable and self-describing;
8. the Cartesian-MPI path agrees with serial execution;
9. local and Olivia qualification results are archived;
10. no existing uncoupled public workflow regresses.

## 12. References and implementation anchors

- D. Jelovina, *A full dimensional Discrete Variable Representation of H2+ and
  H2 photoionization*, Chapter 3. The coupled two-electron harmonics are
  introduced in Section 3.2; the electron-nuclear FEMDVR Poisson construction
  is developed in Section 3.7.3, equations 3.81-3.98.
- `grid_tdse/spherical/two_radial/angular.py`: current uncoupled channel and
  multipole specifications.
- `grid_tdse/spherical/two_radial/coulomb.py`: reusable radial multipole
  construction.
- `grid_tdse/spherical/two_radial/couplings.py`: current sparse Gaunt-product
  coupling builder and a useful action-format reference.
- `grid_tdse/spherical/two_radial/dipole.py`: current length- and
  velocity-gauge axial dipole conventions, effective couplings, and
  field-independent spatial action.
- `grid_tdse/spherical/two_radial/system.py`: current separation of grids,
  radial multipoles, angular couplings, and Hamiltonian-ready data.
- `grid_tdse/spherical/two_radial/diatomic.py` and `tdse.py`: isotope,
  symmetry, ground-state, field, and high-level interface behavior that the
  coupled representation must preserve.
- `grid_tdse/spherical/two_radial/dynamics_sampling.py`: current raw and
  conditional dynamics sampling contract.
- `grid_tdse/spherical/two_radial/distributed_2d_real_time.py`: Cartesian
  radial SIA/CF4-SIA integration target after serial qualification.
- `grid_tdse/real_time/propagation.py`: propagator, snapshot, checkpoint, and
  CF4 linear-combination contracts shared by both representations.
