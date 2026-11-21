# Spherical one-radial off-axis implementation plan

## 1. Objective and status

This document plans the extension of the clamped-nuclei spherical
one-radial implementation from an axial, fixed-`m` diatomic model to a
general two-centre geometry in a complete complex spherical-harmonic basis.
The first scientific target is H2+ with a molecular axis tilted relative to
a linearly polarized laboratory-`z` field. The physical-system interface
must nevertheless accept two arbitrary Cartesian nuclear position vectors
and arbitrary positive nuclear charges, so that heteronuclear one-electron
systems and displaced geometries do not require another data model.

The first milestone must provide:

- explicit Cartesian positions for both clamped nuclei;
- the full complex `(l,m)` expansion through user-selected `l_max` and
  `m_max`;
- complete nuclear Coulomb multipoles through `L_max` and `M_max`;
- matrix-free Davidson stationary states in the full angular space;
- length- and velocity-gauge dynamics for a laboratory-`z` field;
- radial masks and cubic CAPs through the existing absorber interface;
- Crank--Nicolson (CN), short-iterative Lanczos (SIL), and short-iterative
  Arnoldi (SIA) propagation where their current compatibility conditions
  permit;
- raw norm, field-free energy, radial probability, and all three Cartesian
  components of the electronic, nuclear, total, and induced dipoles;
- all three Cartesian components of canonical momentum and, in velocity
  gauge, mechanical momentum;
- orientation-complete metadata, checkpoints, and state archives;
- field-free rotational-covariance and driven-dynamics validation.

The external field remains polarized parallel to laboratory `z` during this
milestone. Arbitrary field polarization, symmetry-adapted reductions, and
MPI distribution are later extensions.

The preliminary grid-lib work is complete. Its public spherical reduced-
radial operators provide Cartesian coordinate and momentum actions, and its
spherical-harmonic rotation object supplies active rotations of complete
complex `(l,m)` states. This plan therefore treats those operations as
dependencies rather than reimplementing their angular algebra in grid-tdse.

Steps 1--10 are complete on the development branch. The explicit-position
system model, arbitrary-position Coulomb construction, full-angular
Hamiltonian utilities, matrix-free Davidson stationary solver, finite-basis
rotation qualification, and analytic-versus-Poisson cross-check are
implemented and tested. Representation-level full-angular laboratory-`z`
field interactions, projected gauge transformations, radial absorbers,
time-dependent Hamiltonian actions, and raw vector observables are also
implemented. The completed on-axis implementation remains the production
high-level dynamics path until Steps 11--16 extend and qualify
preconditioning, persistence, and the public calculation interface.

## 2. Relationship to existing work

The extension builds on:

- `dev/spherical_one_radial_dynamics_implementation_plan.md` for the axial
  real-time implementation;
- `docs/spherical_one_radial_dynamics.md` for its public conventions;
- `docs/spherical_one_radial_rotation_covariance.md` for the finite-basis
  rotation argument and its limits;
- `grid_tdse.spherical.one_radial.OneElectronDiatomicHamiltonian` for the
  existing full-angular field-free action;
- `grid_tdse.spherical.one_radial.OneElectronDiatomicTDSE` for the fixed-`m`
  high-level workflow;
- the representation-independent Davidson and block-Davidson solvers in
  `grid_tdse.ground_state`;
- `grid_tdse.real_time.RealTimePropagator` for all time integration,
  progress, sampling, checkpoint, restart, and continuation machinery.

The existing fixed-`m` classes and defaults must remain backward compatible.
The full-angular implementation should use separate representation-specific
Hamiltonian, dynamics, observable, and preconditioner objects behind the same
high-level calculation interface. It must not add conditionals to every
fixed-`m` inner loop.

## 3. Physical system and geometry contract

### 3.1 General two-centre Hamiltonian

For an electron of user-specified mass `mu` and charge `q_e = -1`, use

```math
H_0
=
-\frac{1}{2\mu}\nabla^2
-\frac{Z_1}{|\mathbf r-\mathbf R_1|}
-\frac{Z_2}{|\mathbf r-\mathbf R_2|}.
```

The public geometry consists of

```text
nuclear_position1 = (x1, y1, z1)
nuclear_position2 = (x2, y2, z2)
```

in bohr. Both vectors must contain finite real values and must be distinct.
The charges and particle mass remain finite and positive. Derived quantities
are

```math
R=|\mathbf R_2-\mathbf R_1|,
\qquad
\mathbf R_c=\frac{\mathbf R_1+\mathbf R_2}{2},
\qquad
E_{\mathrm{nn}}=\frac{Z_1Z_2}{R}.
```

The spherical-expansion origin is not silently moved to `R_c`. Explicit
positions are interpreted in the same Cartesian frame as the spherical
grid, field, and sampled vector observables.

### 3.2 Backward-compatible construction

Retain the current axial constructor:

```python
OneElectronDiatomicSystem(
    nuclear_charge1,
    nuclear_charge2,
    half_bond_length,
    particle_mass=1.0,
    name=None,
)
```

It continues to place the nuclei at `(0,0,-a)` and `(0,0,a)`. Add the
explicit-position constructor

```python
OneElectronDiatomicSystem.from_nuclear_positions(
    nuclear_charge1,
    nuclear_charge2,
    nuclear_position1,
    nuclear_position2,
    particle_mass=1.0,
    name=None,
)
```

Internally, explicit read-only Cartesian positions become the canonical
geometry. `internuclear_distance`, `half_bond_length`, midpoint, bond-axis
unit vector, nuclear repulsion, and nuclear dipoles are derived from them.
This avoids two conflicting sources of geometry truth.

### 3.3 Origin and translation policy

Initial orientation tests use midpoint-centred positions

```math
\mathbf R_1=-\frac{R}{2}\hat{\mathbf n},
\qquad
\mathbf R_2=+\frac{R}{2}\hat{\mathbf n},
```

with `R = 2` bohr for H2+. This isolates rotation from translation.

Arbitrary translated positions are supported by the data model and Coulomb
construction, but no finite-`l_max` translation equivalence is claimed. A
translation mixes unbounded angular momenta about a fixed spherical origin.
The exact energy is origin independent, while finite-cutoff errors generally
are not. Translation tests must therefore check convergence rather than
equality at a fixed cutoff.

For a charged system, the total dipole depends on the chosen origin. Use a
public Cartesian `origin=(0,0,0)` for vector observables and length-gauge
nuclear scalar terms. Preserve the current `origin_z` spelling as a
backward-compatible axial alias, and reject mutually inconsistent values.

## 4. Wavefunction representation

Use the complete complex spherical-harmonic expansion

```math
\Psi(\mathbf r,t)
=
\frac{1}{r}
\sum_{l=0}^{l_{\max}}
\sum_{m=-\min(l,m_{\max})}^{\min(l,m_{\max})}
u_{lm}(r,t)Y_{lm}(\hat{\mathbf r}).
```

The public radial-first trailing state shape remains

```text
(n_r, n_lm)
```

using the channel ordering exposed by grid-lib. With `m_max = l_max`,

```math
n_{lm}=(l_{\max}+1)^2.
```

This complete space is closed under rotations. A smaller `m_max` may remain
available as an explicit convergence experiment, but it is not rotation
covariant and must never be selected implicitly for an off-axis system.

Use grid-lib's active-rotation convention throughout:

```math
[U(\mathcal R)\Psi](\mathbf r)
=
\Psi(\mathcal R^{-1}\mathbf r).
```

Thus, the same rotation maps each nuclear position according to

```math
\mathbf R_A'=\mathcal R\mathbf R_A
```

and maps the molecular wavefunction according to the preceding equation,
while the laboratory axes remain fixed.

The high-level interface should select the representation explicitly:

```text
angular_representation = "fixed_m" | "full"
```

- `"fixed_m"` preserves the existing optimized axial path and requires `m`;
- `"full"` uses the complete radial-first `(l,m)` state and normally sets
  `m_max = l_max`;
- an off-axis potential must reject `"fixed_m"` because it mixes laboratory
  magnetic channels;
- the default remains `"fixed_m"` for backward compatibility.

## 5. Nuclear Coulomb representation

Expand each point-charge potential about the spherical origin,

```math
V_{\mathrm{nuc}}(\mathbf r)
=
\sum_{L=0}^{L_{\max}}
\sum_{M=-\min(L,M_{\max})}^{\min(L,M_{\max})}
V_{LM}(r)Y_{LM}(\hat{\mathbf r}).
```

For a wavefunction truncated at `l_max`, the complete projected potential
requires

```text
nuclear_L_max = 2*l_max
```

by default. For the retained `m_max`, a complete global magnetic cutoff is

```text
nuclear_M_max = min(nuclear_L_max, 2*m_max).
```

Thus the rotation-complete default `m_max = l_max` gives
`nuclear_M_max = nuclear_L_max = 2*l_max`. `M_max = 0` remains only an axial
optimization. User-supplied smaller cutoffs must be recorded as additional
approximations.

Use grid-lib's analytic point-charge construction first. Retain the Poisson
construction as an independent cross-check after the analytic path passes
rotation tests.

Every distinct nonzero nuclear radius `|R_A|` must coincide with an internal
FEM-DVR element boundary. A nucleus at the spherical origin naturally lies
on the radial endpoint. This generalizes the existing common
`half_bond_length` boundary rule. Error messages must list every missing
radius and the supplied element boundaries.

The potential action, metadata, and validation must preserve complex
coefficients. Hermiticity follows from the point-charge reality relation and
must be tested in the physical FEM-DVR metric rather than enforced by
discarding imaginary parts.

## 6. Field coupling and gauge conventions

### 6.1 Initial field scope

The first off-axis milestone retains a spatially uniform field polarized
along laboratory `z`. Tilting the nuclei therefore changes the physical
laser--molecule alignment; it is not merely a coordinate rotation.

The electronic length-gauge interaction is

```math
V_{L,e}(t)=-E_z(t)q_e(z-z_0),
```

and the optional clamped-nuclear scalar is

```math
V_{L,\mathrm{nuc}}(t)
=
-E_z(t)
\sum_A Z_A(R_{A,z}-z_0).
```

The electronic velocity-gauge interaction remains

```math
V_V(t)=-A_z(t)\frac{q_e}{\mu}p_z,
```

with the spatially uniform `A_z(t)^2` term omitted as the same documented
global-phase convention used by the axial implementation.

Laboratory `z` operators preserve laboratory `m`, but the off-axis nuclear
potential does not. The complete Hamiltonian must therefore act on the full
state even though the field interaction itself is block diagonal in `m`.

### 6.2 Gauge transformation

Extend the projected axial gauge transformation to full-angular states. The
phase `exp(i q_e A_z z)` preserves laboratory `m`, so each retained `m` block
may be projected independently while using the common full-state channel
ordering. Report the finite-space projection norm defect exactly as for the
fixed-`m` path.

Gauge comparisons must use the same physical molecular orientation and must
not compare in-pulse states without applying this transformation.

### 6.3 Future arbitrary field vectors

The completed grid-lib `x`, `y`, `z`, `p_x`, `p_y`, and `p_z` actions make a
future vector-polarized interaction possible. That extension is deliberately
outside this milestone so that failures in molecular orientation are not
confounded with a simultaneous generalization of the driving field.

## 7. Absorbers and integrators

The radial mask and cubic CAP depend only on spherical `r` and therefore act
identically on every `(l,m)` channel. Generalize their state validation and
application from fixed-`m` shapes to the full angular shape without changing
the profiles or their physical conventions.

Preserve the established compatibility table:

| Integrator | No absorber | Radial mask | Cubic CAP |
| --- | ---: | ---: | ---: |
| Crank--Nicolson | yes | yes | deferred |
| SIL | yes | yes | no |
| SIA | yes | yes | yes |

SIL must reject a CAP because the generator is non-Hermitian. CN+CAP remains
deferred until separately qualified. The CAP is excluded from sampled
physical energy, while mask or CAP loss remains visible in all raw
observables.

The full-angular CN preconditioner should retain radial kinetic and
centrifugal blocks independently for every `(l,m)` channel. Since the
preconditioner omits the molecular potential and field, extending its channel
indexing should not require new physics.

## 8. Stationary states

### 8.1 Direct oracle

Add full-angular metric weights, physical-to-Euclidean transformations, and
an explicit Euclidean-Hermitian matrix only for tiny validation systems. The
direct solver is an oracle, not the production path.

### 8.2 Davidson production path

Use the existing representation-independent Davidson solvers with the
matrix-free full-angular Hamiltonian action and FEM-DVR inner product. A
diagonal preconditioner is the initial production choice. Its diagonal must
include the radial kinetic and centrifugal contributions and may optionally
include the diagonal nuclear-potential contribution when that improves the
iteration count without excessive construction cost.

The public stationary-state entry point should be analogous to

```python
diagonalize_full_angular(
    hamiltonian,
    n_states=1,
    solver="davidson",
    initial_states=None,
    solver_options=None,
)
```

For a rigidly rotated midpoint-centred geometry, a converged on-axis state
rotated with grid-lib is the exact eigenstate of the correspondingly rotated
finite-basis Hamiltonian, up to the residual of the original eigensolve and
floating-point rotation error. It is therefore a reference solution, not
merely a high-quality Davidson guess. The independent off-axis Davidson
qualification must start from a generic internally generated guess and
converge to the rotated reference. For arbitrary geometries that are not
related by a rigid rotation about the spherical origin, the internally
generated or user-supplied guess remains the normal production path.

This exact statement concerns the selected radial discretization and complete
finite angular space. It does not assert that the finite-grid state is the
continuum-limit physical ground state. For degenerate excited levels, compare
the complete rotated eigenspace rather than individual eigenvectors.

Return immutable electronic energies, total energies, orbitals, residuals,
iteration history, solver metadata, and the full representation metadata.
Do not label full-angular states by laboratory `m` when the potential is not
axially symmetric.

## 9. Vector observables

All sampled expectation values remain raw, unnormalized numerators. The norm
is stored separately so conditional values can be formed in postprocessing.

For Cartesian origin `r_0`, sample the electronic dipole vector

```math
\mathbf D_e(t)
=
q_e\langle\Psi(t)|\mathbf r-\mathbf r_0|\Psi(t)\rangle,
```

the clamped-nuclear dipole

```math
\mathbf d_{\mathrm{nuc}}
=
\sum_A Z_A(\mathbf R_A-\mathbf r_0),
```

and the raw total numerator

```math
\mathbf D(t)
=
\mathbf D_e(t)+\mathbf d_{\mathrm{nuc}}N(t).
```

Define the induced vector numerator by

```math
\mathbf D^{\mathrm{ind}}(t)
=
\mathbf D(t)-\mathbf d(0)N(t),
```

where `d(0) = D(0)/N(0)`. This prevents absorber-induced norm loss from
creating a false static contribution.

Sample canonical electronic momentum

```math
\mathbf P(t)
=
\mathrm{Re}\langle\Psi(t)|\mathbf p|\Psi(t)\rangle.
```

For the initial velocity-gauge field `A(t) = (0,0,A_z(t))`, the mechanical
momentum numerator is

```math
\boldsymbol{\Pi}(t)
=
\mathbf P(t)-q_e\mathbf A(t)N(t).
```

Consequently, the `x` and `y` mechanical components equal their canonical
components while the `z` component carries the existing field correction.

Also sample:

- the wavefunction norm;
- raw field-free energy numerator;
- optional reference-state populations;
- unnormalized
  `P(r,t) = sum_lm |u_lm(r,t)|^2` at the user-selected cadence;
- integrator diagnostics.

The FEM-DVR integral of every stored `P(r,t)` must equal the corresponding
sampled norm.

## 10. High-level and persistence design

Extend `OneElectronDiatomicTDSE` rather than introduce an unrelated public
workflow. A representative full-angular calculation is

```python
calculation = OneElectronDiatomicTDSE(
    system=system,
    nodes=nodes,
    n_points=n_points,
    l_max=l_max,
    angular_representation="full",
    m_max=l_max,
    nuclear_L_max=2 * l_max,
    nuclear_M_max=2 * l_max,
    initial_state=initial_state,
    fields=laser,
    field_gauge="length",
    mask=mask,
    integrator="sia",
    final_time=final_time,
    time_step=time_step,
)
```

Checkpoint compatibility and archives must include at least:

- both nuclear positions and charges in a defined order;
- spherical origin and observable origin;
- internuclear distance, midpoint, and bond-axis unit vector;
- particle mass;
- `angular_representation`, channel ordering, `l_max`, and `m_max`;
- `nuclear_L_max`, `nuclear_M_max`, and Coulomb method;
- radial nodes and points per element;
- gauge, fields, absorber, integrator, and preconditioner;
- state representation and normalization policy;
- vector-observable component ordering.

An archive with the same bond length but a different orientation must not be
accepted silently. A user-requested rotation of a compatible state should be
an explicit operation that produces new metadata and a new state identity.

## 11. Step-by-step implementation sequence

### Step 1: freeze geometry and representation contracts

Turn Sections 3--10 into executable validation tests. Define the active-
rotation convention, nuclear ordering, origins, state shape, channel order,
cutoff defaults, and metadata schema before altering the Hamiltonian.

Acceptance gate:

- old axial system construction and fixed-`m` metadata are unchanged;
- explicit positions have one unambiguous interpretation;
- off-axis geometry cannot enter the fixed-`m` path accidentally.

Status: complete. The contracts in this plan fix the explicit-position
constructor, active-rotation convention, state layout, cutoff defaults,
origin policy, and persistence requirements.

### Step 2: generalize the physical system

Add explicit-position construction and derived geometry properties to
`OneElectronDiatomicSystem`. Preserve the axial constructor and all existing
system names and mass behavior.

Acceptance gate:

- axial construction and equivalent explicit positions compare identically;
- nuclear distance, repulsion, midpoint, axis, and dipoles match analytic
  values for homonuclear, heteronuclear, tilted, and translated examples;
- coordinate-specific consumers such as the prolate-spheroidal Hamiltonian
  continue to reject geometries that are incompatible with their axes;
- invalid, coincident, complex, or non-finite positions are rejected.

Status: complete. Explicit Cartesian positions are canonical internal data;
the legacy axial constructor produces the same immutable system as equivalent
explicit positions. Prolate-spheroidal consumers reject incompatible
geometries explicitly.

### Step 3: generalize Coulomb cutoffs and radial-boundary validation

Expose `nuclear_M_max`, select the complete full-angular default, and require
every distinct nonzero nuclear radius to be an FEM-DVR element boundary.
Build the analytic arbitrary-position potential through grid-lib.

Acceptance gate:

- the axial `M_max=0` action is unchanged;
- a tilted potential contains the expected nonzero-`M` coefficients;
- its full action is Hermitian in the physical metric;
- increasing cutoffs beyond `L_max=2*l_max` and
  `M_max=min(L_max,2*m_max)` does not change the projected matrix.

Status: complete. Non-axial systems select the complete magnetic multipole
range by default, axial systems retain the `M_max=0` optimization, and every
distinct nonzero nuclear radius is validated against FEM-DVR boundaries.

### Step 4: complete full-angular Hamiltonian utilities

Add public full-state validation, metric weights, Euclidean transformations,
explicit tiny-system matrices, Davidson diagonals, and full Cartesian
coordinate and momentum delegations to grid-lib.

Acceptance gate:

- full actions match explicit matrices on tiny grids;
- `x`, `y`, `z`, `p_x`, `p_y`, and `p_z` are Hermitian in the physical
  metric to the documented boundary tolerance;
- the existing fixed-`m` action equals the corresponding axial full-state
  block.

Status: complete. Full strong and Euclidean-Hermitian matrices, metric
weights and transforms, Davidson diagonals, state validation, and all six
Cartesian coordinate and momentum delegations are public.

### Step 5: add full-angular stationary-state solvers

Implement the direct oracle and matrix-free Davidson path, including the
initial-guess policy and immutable result object.

Acceptance gate:

- direct and Davidson energies and residuals agree on tiny grids;
- Davidson converges the on-axis and tilted H2+ ground states without
  constructing the full matrix;
- an independently converged tilted ground state agrees, up to global phase,
  with the exact rotated on-axis finite-basis state.

Status: complete. The direct solver is retained as a small-grid oracle.
Matrix-free Davidson and block Davidson use the physical FEM-DVR metric,
batched overlaps, and a diagonal preconditioner.

### Step 6: establish field-free rotational covariance

At `R = 2` bohr, compare full-angular H2+ Hamiltonians for molecular-axis
angles `0`, `30`, `45`, and `90` degrees in the `x`-`z` plane. Use several
small `l_max` values with complete `m` and nuclear multipoles.

Acceptance gate:

- complete low-energy spectra agree to numerical precision at each fixed
  cutoff;
- a rotated on-axis eigenvector satisfies the tilted Hamiltonian residual to
  the original eigensolver and floating-point tolerances;
- vector dipoles and momenta transform covariantly;
- intentionally omitting magnetic channels or nuclear `M` components makes
  the test fail clearly.

Status: complete for the complete-space invariants. H2+ spectra at `30`,
`45`, and `90` degrees agree with the axial spectrum, rotated eigenstates
satisfy the independently assembled tilted Hamiltonian, and Hamiltonian,
coordinate, and momentum actions transform covariantly.

### Step 7: cross-check the analytic potential with Poisson construction

Generalize the Poisson path to the same arbitrary positions and compare its
coefficients, actions, and low energies with the analytic construction on a
compatible grid.

Acceptance gate:

- analytic and Poisson discrepancies decrease with radial refinement;
- both constructions satisfy the same rotational tests within their stated
  numerical tolerances.

Status: complete. For tilted H2+, analytic-versus-Poisson coefficient,
Hamiltonian-action, and ground-state-energy discrepancies decrease with
radial-order refinement.

### Step 8: add full-angular field interactions and gauge transforms

Add full-state laboratory-`z` length- and velocity-gauge interactions,
optional nuclear scalar handling, field validation, and the projected
full-state axial gauge transformation.

Acceptance gate:

- axial full-state interactions reproduce every fixed-`m` block;
- zero fields produce exactly zero interaction;
- the heteronuclear nuclear scalar has the analytic value;
- the gauge transformation preserves `m` blocks and reports its projection
  norm defect.

Status: complete. Dedicated full-angular length- and velocity-gauge
interactions use the laboratory-`z` coordinate and momentum actions. Tests
show exact agreement with every retained fixed-`m` block, exact zero-field
actions, the analytic heteronuclear nuclear scalar, and independent
projected gauge transformations in every signed-`m` block. The transformed
state norm remains available to quantify the finite-space projection defect.

### Step 9: add full-angular absorbers and time-dependent Hamiltonian

Generalize mask and CAP state handling, add a dedicated full-angular
time-dependent Hamiltonian, and retain the current integrator compatibility
rules.

Acceptance gate:

- field-free eigenstate propagation preserves norm and phase without an
  absorber;
- the radial mask acts identically on every angular channel;
- SIA+CAP produces controlled norm loss and SIL+CAP is rejected before
  propagation.

Status: complete. Masks and CAPs broadcast over every full-angular channel,
and the dedicated full-angular time-dependent Hamiltonian combines the
field-free, field, and absorber actions. A full-angular field-free
eigenstate retains its phase and norm under SIA, a CAP produces norm loss,
and incompatible SIL+CAP propagation is rejected before the first step.

### Step 10: add vector observables

Implement the complete raw vector observable contract from Section 9 and
full-angular radial probability sampling.

Acceptance gate:

- radial probability integrates to the sampled norm;
- midpoint-centred H2+ has zero nuclear dipole for every orientation;
- a heteronuclear or translated test gives the analytic nuclear vector;
- momentum and dipole vectors rotate with the state and geometry;
- velocity-gauge mechanical momentum differs from canonical momentum by
  exactly `-q_e*A_z(t)*N(t)` in the `z` component only.

Status: complete. The full-angular observable object samples raw norm,
field-free energy, Cartesian electronic/nuclear/total/induced dipoles,
Cartesian canonical momentum, velocity-gauge mechanical momentum, state
populations, and cadence-controlled radial probability. Tests cover radial
normalization, midpoint-centred homonuclear and off-axis heteronuclear
nuclear dipoles, rotational covariance, and the exact mechanical-momentum
shift in only the laboratory-`z` component.

### Step 11: generalize CN support

Extend the radial kinetic-plus-centrifugal preconditioner to full channel
ordering and preserve per-step linear-solver diagnostics.

Acceptance gate:

- preconditioned and unpreconditioned CN agree on a short full-angular
  propagation;
- the preconditioner reduces total iterations on the development problem;
- mask-based CN is reliable and CN+CAP remains rejected.

### Step 12: extend the high-level interface and persistence

Add `angular_representation`, full-state preparation, vector observables,
orientation-complete metadata, checkpoint validation, restart, and
continuation to `OneElectronDiatomicTDSE`.

Acceptance gate:

- changing only the representation selects the appropriate internal path;
- all existing fixed-`m` interface tests remain unchanged;
- a full-angular run restarts exactly;
- incompatible geometry, origin, cutoff, or channel metadata is rejected.

### Step 13: validate the three integrators

Compare CN, SIL, and SIA on a small tilted, field-free problem and on a short
weak laboratory-`z` pulse. Use a fine-step SIA calculation or a tiny explicit
matrix as the oracle.

Acceptance gate:

- all supported integrators converge toward the same state and vector
  observables under time-step refinement;
- field-free norm and energy behavior matches the on-axis qualification;
- mask and CAP behavior satisfies the existing compatibility contract.

### Step 14: establish off-axis gauge equivalence

Propagate the same tilted H2+ state and weak pulse in length and velocity
gauge. Compare endpoints directly when `A_z = 0` and compare in-pulse states
only after the projected gauge transformation.

Acceptance gate:

- each gauge is independently stable under radial and time-step refinement;
- endpoint and transformed in-pulse discrepancies decrease with angular
  refinement;
- all three dipole and momentum components show the same convergence trend.

### Step 15: validate physical alignment behavior

Compare parallel, intermediate, and perpendicular molecular alignments in
the same fixed laboratory-`z` pulse. These dynamics are physically distinct;
the test checks symmetry and consistency rather than equality.

Acceptance gate:

- an `x`-`z`-plane geometry retains the expected reflection result
  `D_y(t) = P_y(t) = 0` within numerical tolerance;
- changing the azimuth about laboratory `z` rotates transverse vector
  observables without changing scalar observables;
- simultaneous rigid rotation of molecule, state, and a future vector field
  is identified as the stronger covariance test, but is not required until
  arbitrary field polarization is implemented.

### Step 16: add a public example and documentation

Add a short H2+ example that accepts either two Cartesian nuclear positions
or bond-orientation angles, computes or loads a full-angular initial state,
and drives it with a laboratory-`z` pulse. Document geometry, cutoffs,
origins, state layout, gauge conventions, vector observables, convergence
responsibilities, and the rotation-versus-translation distinction.

Acceptance gate:

- the example uses only public APIs;
- its defaults are explicitly labelled as development inputs;
- output archives contain enough metadata to reconstruct the orientation and
  representation exactly.

### Step 17: qualify performance before MPI work

Measure Hamiltonian-action, Davidson, observable, and propagation costs for
increasing `l_max`. Record active radial points, channel counts, potential
component counts, memory, wall time, Krylov dimensions, and CN iteration
counts. Test controlled BLAS/thread settings, but keep the implementation
serial.

Acceptance gate:

- the dominant kernels and memory consumers are measured rather than
  inferred;
- the results identify whether later distribution should prioritize radial
  points, angular channels, or a two-dimensional layout.

## 12. Validation matrix

| Area | Primary comparison | Required trend or invariant |
| --- | --- | --- |
| Geometry | axial versus explicit positions | identical derived data |
| Coulomb | action versus tiny explicit matrix | metric-Hermitian agreement |
| Magnetic cutoff | complete versus truncated `M` | complete path restores covariance |
| Stationary states | Davidson versus direct | matching energies and residuals |
| Rotation | `0`, `30`, `45`, `90` degree H2+ | identical field-free spectra |
| State rotation | rotated on-axis eigenstate | small tilted residual |
| Translation | displaced midpoint | convergence, not fixed-cutoff equality |
| Analytic/Poisson | same arbitrary geometry | radial-refinement agreement |
| Operators | grid-tdse wrapper versus grid-lib | identical Cartesian actions |
| Vector observables | rotated geometry and state | Cartesian covariance |
| Integrators | CN, SIL, SIA | common fine-step limit |
| Gauge | length versus velocity | angular-refinement agreement |
| Reflection | molecule in `x`-`z` plane | vanishing `y` response |
| Persistence | uninterrupted versus restart | identical final state and samples |
| Backward compatibility | existing fixed-`m` suite | no regression |

The compact development grids qualify implementation consistency only. They
must not be presented as radial, angular, time-step, pulse, or absorber
convergence for a scientific calculation.

## 13. Initial development cases

Use H2+ at `R = 2` bohr and particle mass `1` a.u. for the primary gates.
Keep the radial grid small enough for direct-matrix oracles, but include the
nuclear radius `1` bohr as an FEM-DVR boundary. Suggested midpoint-centred
orientations are

```text
parallel:       n = (0, 0, 1)
tilted 30 deg:  n = (1/2, 0, sqrt(3)/2)
tilted 45 deg:  n = (1/sqrt(2), 0, 1/sqrt(2))
perpendicular:  n = (1, 0, 0)
```

At each orientation,

```text
R1 = -n
R2 = +n
```

in bohr. Start field-free covariance tests at `l_max = 1, 2, 3`, where tiny
direct matrices remain practical, then use `l_max = 4, 6, 8` for Davidson,
gauge, and short-dynamics refinement tests. These values test code paths and
trends, not convergence of the H2+ ground state.

Add at least one heteronuclear two-centre test with unequal charges to ensure
that odd multipoles and the nuclear dipole are exercised. Add a translated
geometry only after the rotation gates pass, and assess it through cutoff
convergence rather than exact equality at fixed `l_max`.

## 14. Performance and scaling expectations

For the rotation-complete basis,

```math
n_{lm}=(l_{\max}+1)^2,
```

whereas the on-axis `m=0` path contains only `l_max + 1` channels. The full
state is therefore intrinsically larger by another factor of approximately
`l_max + 1` before considering the denser angular coupling graph.

The number of nominal nuclear multipole labels through
`L_max = M_max = 2*l_max` is

```math
\sum_{L=0}^{2l_{\max}}(2L+1)=(2l_{\max}+1)^2.
```

Do not form a dense `(n_r*n_lm)^2` production matrix. Cache sparse angular
couplings and radial coefficient arrays, keep the kinetic action batched over
channels, and use matrix-free Davidson and Krylov propagation. Performance
work should separately time:

- radial kinetic and centrifugal action;
- nuclear multipole action;
- Cartesian operator actions;
- Davidson preconditioning and orthogonalization;
- one real-time Hamiltonian action;
- one observable sample;
- checkpoint serialization.

Serial profiling precedes MPI design. Symmetry adaptation, real-harmonic
bases, and distributed layouts must be justified by those measurements.

## 15. Deferred extensions

The following are explicitly outside the first milestone:

- arbitrary laser-polarization vectors and multiple polarization axes;
- real spherical harmonics or reflection-adapted angular bases;
- homonuclear inversion or other discrete-symmetry reductions;
- MPI distribution of radial or angular dimensions;
- moving nuclei or coupling to the two-radial non-Born--Oppenheimer model;
- production claims for translated geometries at finite `l_max`;
- CN with a CAP;
- direct-force acceleration and acceleration-form HHG for arbitrary nuclear
  positions;
- orientation averaging over a molecular ensemble.

These extensions should reuse the explicit geometry, complete angular state,
vector-operator, and metadata contracts established here.

## 16. Expected deliverables

The milestone is complete only when it provides:

1. a backward-compatible explicit-position physical-system model;
2. arbitrary-position analytic and Poisson nuclear Coulomb construction;
3. complete full-angular Hamiltonian utilities and Davidson stationary
   states;
4. full-angular length- and velocity-gauge laboratory-`z` dynamics;
5. full-angular radial absorbers and practical CN preconditioning;
6. raw Cartesian dipole and momentum observables plus radial probability;
7. orientation-complete archives, checkpoints, restart, and continuation;
8. automated field-free rotation, direct/Davidson, analytic/Poisson,
   integrator, gauge, symmetry, and persistence qualifications;
9. a public off-axis H2+ example and user documentation;
10. a serial performance report that informs later MPI work.
