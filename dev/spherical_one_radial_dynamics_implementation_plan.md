# Spherical one-radial laser-dynamics implementation plan

## 1. Objective and status

This document plans a reusable real-time implementation for clamped-nuclei,
one-electron diatomic systems in the spherical one-radial representation. The
initial scientific target is laser-induced dynamics in H2+ with the two nuclei
on the laboratory `z` axis and a linearly polarized electric field parallel to
that axis.

The first implementation must support:

- the existing spherical FEM-DVR one-radial field-free Hamiltonian;
- a user-supplied particle mass;
- arbitrary compatible initial states, not only a computed ground state;
- length- and velocity-gauge axial fields;
- a radial cosine mask or cubic complex absorbing potential (CAP);
- Crank--Nicolson (CN), short-iterative Lanczos (SIL), and short-iterative
  Arnoldi (SIA) propagation;
- standard time-dependent observables, including the radial probability
  distribution;
- the generic progress, checkpoint, restart, and sampling facilities in
  `grid_tdse.real_time.RealTimePropagator`.

The reference development system is clamped H2+ at internuclear distance
`R = 2` bohr with particle mass `1` a.u. The particle mass must remain a public
input, so reduced-mass and other effective-mass calculations do not require a
new implementation.

Steps 2--6 are complete: the axial position and momentum operators, the
fixed-`m` length- and velocity-gauge interactions, the spherical radial
absorber wrapper, and the fixed-`m` time-dependent Hamiltonian are implemented
and validated. The remaining implementation steps have not started. This
document records the settled design and the validation sequence. It does not
claim radial, angular, time-step, pulse, or absorber convergence.

## 2. Existing foundation and present limitation

`grid_tdse.spherical.one_radial.OneElectronDiatomicHamiltonian` already
provides:

- a spherical FEM-DVR radial grid;
- a two-centre Coulomb potential represented through spherical multipoles;
- full-angular and fixed-`m` field-free Hamiltonian actions;
- the physical FEM-DVR inner product;
- explicit strong and Euclidean-Hermitian fixed-`m` matrices;
- direct fixed-`m` stationary-state diagonalization.

`grid_tdse.real_time.RealTimePropagator` is independent of the spatial
representation and already provides CN, SIL, SIA, CF4-SIL, and CF4-SIA. It
also provides sampling, progress reporting, post-step transformations,
rolling checkpoints, exact restart, and continuation. The present work must
reuse that propagator rather than introduce a second time-integration layer.

The one-radial field-free action is already propagation-capable. A regression
test constructs a local length-gauge `z` action and uses the generic SIA
propagator to compare weak-pulse H2+ dynamics with the prolate-spheroidal
representation. That test demonstrates feasibility, but the spherical
one-radial package still lacks reusable field operators, absorbers, a
time-dependent Hamiltonian wrapper, a high-level dynamics interface, and a
production example.

The older `examples/spherical/one_radial/tdse_h_atom_femdvr.py` example uses
`grid-lib` operators directly. It is a useful velocity-gauge and CN reference,
but it is not the public interface to extend.

## 3. Initial physical representation

### 3.1 Axial clamped-nuclei geometry

For the first milestone, place the nuclei symmetrically about the origin:

```math
\mathbf R_1=(0,0,-R/2),
\qquad
\mathbf R_2=(0,0,R/2).
```

The spherical expansion is centered at the bond midpoint. For one conserved
magnetic quantum number `m`, write

```math
\Psi(\mathbf r,t)
=
\frac{1}{r}
\sum_{l=|m|}^{l_{\max}}
u_{lm}(r,t)Y_{lm}(\hat{\mathbf r}).
```

The first dynamics calculations use `m = 0`. Both the axial field-free
Hamiltonian and a laboratory-`z`-polarized field preserve `m`, so a fixed-`m`
state is sufficient.

The field-free electronic Hamiltonian is

```math
H_0
=
-\frac{1}{2\mu}\nabla^2
-\frac{Z_1}{|\mathbf r-\mathbf R_1|}
-\frac{Z_2}{|\mathbf r-\mathbf R_2|},
```

where `mu` is the user-supplied particle mass. Nuclear repulsion is a
separate constant and is not part of the electronic propagation generator.

### 3.2 Initial-state policy

Real-time propagation must accept any finite, nonzero state compatible with
the configured radial grid, angular cutoff, and fixed `m`. The initial state
may be:

- the ground state from the existing fixed-`m` diagonalizer;
- an excited stationary state;
- a coherent superposition;
- a compatible state archive;
- a state constructed by user code.

Ground-state computation and archive loading are convenience paths, not
requirements of the time-dependent Hamiltonian or propagator. The high-level
interface must validate the state shape, fixed-`m` sector, finite values,
nonzero norm, and any available archive metadata. It must not normalize the
state silently.

## 4. Field interactions and gauge conventions

### 4.1 Length gauge

For a spatially uniform electric field parallel to `z`, the interaction is

```math
V_L(t)=-E_z(t)d_z.
```

About axial origin `z_0`, the electronic dipole is

```math
d_{e,z}=q_e(z-z_0),
```

so the electronic interaction is

```math
V_{L,e}(t)=-E_z(t)q_e(z-z_0).
```

For an electron, `q_e = -1`, and the midpoint-origin spatial action is
therefore `+z` multiplied by `E_z(t)`.

The clamped-nuclei dipole is

```math
d_{\mathrm{nuc},z}
=
\sum_A Q_A(R_{A,z}-z_0).
```

Its length-gauge interaction is a scalar multiplying the complete state and
therefore changes only the global phase. For homonuclear nuclei placed
symmetrically about the chosen origin, this contribution is exactly zero,
including when the molecular axis is later tilted away from `z`.

Support the nuclear scalar as an explicit option for completeness and future
heteronuclear systems, but use

```text
include_nuclear_scalar = False
```

by default. Record its value and the omitted scalar in archive metadata.

### 4.2 Velocity gauge

For a spatially uniform axial vector potential, use the linear
minimal-coupling interaction

```math
V_V(t)
=
-A_z(t)\frac{q_e}{\mu}p_z.
```

With `q_e = -1`, the spatial action is `p_z / mu`. The spatially constant
quadratic term

```math
\frac{q_e^2 A_z(t)^2}{2\mu}
```

is omitted, consistent with the prolate-spheroidal implementation. It changes
only the global phase. Metadata and state-comparison tests must record this
convention explicitly.

Temporal profiles must remain separate from spatial operators. A length-gauge
profile returns `E_z(t)`, while a velocity-gauge profile returns `A_z(t)`.
When a field object exposes a `gauge` attribute, the interaction must validate
it against the requested gauge.

### 4.3 Required spatial operators

Add reusable operators for both the full angular expansion and a fixed-`m`
restriction:

```text
apply_z(state)
apply_z_fixed_m(state, m)
apply_pz(state)
apply_pz_fixed_m(state, m)
```

The position operator follows

```math
z=r\cos\theta
```

and couples `(l,m)` only to `(l+1,m)` and `(l-1,m)`. The momentum operator
contains radial-derivative and angular terms and obeys the same `Delta m = 0`
selection rule.

Use a direct spherical derivative representation for the production `p_z`
action. Validate it independently against the discrete local-potential
commutator relation

```math
p_z \simeq i\mu[H_0,z],
```

allowing for finite-basis and boundary errors. Building full-angular
operators now avoids duplicating the spatial algebra when arbitrary molecular
orientations are added later.

The production `z` and `p_z` actions delegate their reduced-radial angular
algebra and FEM-DVR first-derivative action to the public grid-lib
`SphericalReducedRadialOperators` primitive. Grid-tdse retains the physical
Hamiltonian, fixed-system policy, field coupling, observables, and metadata.

## 5. Absorbing boundaries

Reuse the public `grid-lib` radial profiles:

- `RadialCosineMask(r0, n)`;
- `CubicCAP(alpha, eta)`.

Both profiles are evaluated directly on the spherical radial coordinate, so
no coordinate-specific reformulation is required. Wrap them in a
representation-aware class that validates the radial grid, caches read-only
profile values, applies a mask after accepted steps, and adds a CAP to the
Hamiltonian action.

Mask and CAP configuration are mutually exclusive. An absent absorber must
be represented explicitly in metadata rather than inferred from missing
fields.

The initial compatibility contract is:

| Integrator | No absorber | Radial mask | Cubic CAP |
| --- | ---: | ---: | ---: |
| Crank--Nicolson | yes | yes | deferred |
| SIL | yes | yes | no |
| SIA | yes | yes | yes |

SIL must reject a CAP because Lanczos propagation assumes a Hermitian
generator. CN can in principle propagate a non-Hermitian CAP Hamiltonian, but
CN+CAP is deferred until the iterative solver and preconditioner have a
dedicated validation. The first guaranteed CAP path is SIA.

The mask is supplied to `RealTimePropagator` as a post-step transformation.
It is applied after each accepted step and before norm, samples, and
checkpoints are recorded. The CAP is part of the propagation Hamiltonian but
must be excluded from the physical-energy observable.

## 6. Proposed package architecture

Add these representation-specific modules:

```text
grid_tdse/spherical/one_radial/
    dipole.py
    absorbing_boundaries.py
    dynamics.py
    observables.py
```

The initial public classes should be analogous to:

```text
SphericalOneRadialAxialLengthGaugeInteraction
SphericalOneRadialAxialVelocityGaugeInteraction
SphericalOneRadialAbsorbingBoundaries
SphericalOneRadialFixedMTimeDependentHamiltonian
```

The time-dependent Hamiltonian wrapper combines:

1. the physical field-free fixed-`m` action;
2. zero or more compatible temporal field profiles;
3. the selected length- or velocity-gauge spatial interaction;
4. an optional CAP.

It must provide:

```text
action(state, time)
inner_product(left, right)
inner_products(left_states, right)
propagate(propagator, initial_state, ...)
archive_metadata()
```

The wrapper enforces absorber/integrator compatibility and installs a mask as
the propagator's `post_step` callback. Integration, progress reporting,
sampling, checkpointing, restart, and continuation remain responsibilities of
`RealTimePropagator`.

## 7. High-level interface

The high-level interface should be similar in spirit to the two-radial
non-Born--Oppenheimer dynamics interface. A representative form is:

```python
calculation = prepare_spherical_one_radial_dynamics(
    system=system,
    nodes=nodes,
    n_points=n_points,
    l_max=l_max,
    m=0,
    initial_state=initial_state,
    fields=laser,
    field_gauge="length",
    absorber=absorber,
    integrator="sia",
    initial_time=0.0,
    final_time=final_time,
    time_step=time_step,
    radial_probability_interval=1,
)

result = calculation.propagate()
```

The final function and result names should follow established package naming
after inspecting the two-radial interface. The important contract is the
separation of system, representation, initial state, field, absorber,
integrator, sampling, and persistence settings.

The public system already accepts `particle_mass`; the high-level preparation
path must preserve it in Hamiltonian construction, velocity-gauge coupling,
metadata, checkpoints, and compatibility checks.

## 8. Observable and sampling contract

Sample the following by default:

- wavefunction norm;
- raw physical field-free energy numerator, excluding any CAP;
- raw electronic and total axial dipole numerators;
- raw induced axial dipole numerator relative to the initial sample;
- ground-state population when a reference ground state is supplied;
- unnormalized radial probability distribution `P(r,t)`;
- integrator diagnostics supplied by `RealTimePropagator`.

Do not normalize observables during sampling. Conditional expectations remain
postprocessing quantities obtained from the raw numerator and sampled norm.

For fixed `m`, define

```math
P(r,t)=\sum_{l=|m|}^{l_{\max}}|u_{lm}(r,t)|^2.
```

Its FEM-DVR quadrature integral equals the instantaneous wavefunction norm,
including loss from a mask or CAP.

Use a configurable accepted-step interval:

```text
radial_probability_interval = positive integer or None
```

- a positive integer samples every specified number of accepted steps;
- `None` disables radial-probability sampling;
- when enabled, always retain the initial and final distributions;
- store the corresponding accepted-step indices and physical times;
- use interval `1` by default for short development calculations.

The final result must make the lower-cadence radial samples distinguishable
from scalar observables sampled at every accepted step.

## 9. Crank--Nicolson support

The generic propagator already implements CN with GMRES or BiCGSTAB. Establish
correctness first on small grids without a specialized preconditioner. Then
provide or adapt a one-radial preconditioner before presenting CN as a
production option.

The older standalone hydrogen example uses the `grid-lib` `M2Psi`
preconditioner and is an implementation reference, not an API to expose
unchanged. A package-level preconditioner must accept the new radial-first
state layout, preserve the physical metric, support the configured particle
mass, and record iteration diagnostics.

CN must report linear-solver failures without returning a partial result.
Linear tolerance, maximum iteration count, solver choice, and preconditioner
metadata must be part of the reproducibility record.

## 10. Step-by-step implementation sequence

### Step 1: freeze conventions and tests

Turn Sections 3--9 into executable API contracts and small analytic tests.
Record field signs, origin, mass, omitted scalar phases, state layout,
absorber semantics, normalization policy, and metadata schema before adding
the high-level wrapper.

Acceptance gate:

- the initial axial scope and future full-angular scope are unambiguous;
- no test depends on an undocumented field or phase convention.

### Step 2: implement and validate the position action

Add full-angular and fixed-`m` `z` actions. Validate angular coefficients,
metric Hermiticity, batched leading dimensions, complex input, and agreement
between full and restricted actions.

Acceptance gate:

- the new fixed-`m` action reproduces the independent `z` action currently
  defined in the prolate-versus-spherical dynamics test.

### Step 3: implement and validate the momentum action

Add full-angular and fixed-`m` `p_z` actions using the FEM-DVR first derivative
and spherical angular couplings. Compare with the discrete commutator oracle
and verify metric Hermiticity on states satisfying the radial boundary
conditions.

Acceptance gate:

- direct and commutator actions agree within a documented discretization
  tolerance on successively refined development grids.

Implemented validation uses the H2+ ground state on the fixed radial extent
`[0, 1, 3, 6]` bohr and the combined refinement sequence
`(points_per_element, l_max) = (4, 2), (6, 4), (8, 6)`. The relative
FEM-DVR norm difference between the direct action and
`i*mu*(H_0*z - z*H_0)` decreases from `1.36e-1` through `4.16e-2` to
`1.86e-2`. The nonzero difference is expected from finite angular projection
and radial discretization; the acceptance test requires monotonic decrease
and a final relative difference below `2.0e-2`.

### Step 4: add axial length- and velocity-gauge interactions

Add reusable temporal/spatial interaction classes, field-sequence handling,
gauge validation, origin handling, optional nuclear scalar phase, electronic
and total dipoles, momentum expectation, and JSON-compatible metadata.

Acceptance gate:

- zero field produces exactly zero interaction;
- field summation and gauge rejection are tested;
- the midpoint H2+ nuclear scalar is verified to vanish.

### Step 5: add spherical radial absorbers

Wrap `RadialCosineMask` and `CubicCAP`, enforce mutual exclusion, cache radial
values, and implement post-step mask and CAP actions.

Acceptance gate:

- the mask is unity before its onset and reaches the documented outer value;
- the CAP has the correct onset, sign, and strength;
- absorber-free actions are unchanged.

### Step 6: add the fixed-`m` time-dependent Hamiltonian

Compose the field-free, field-interaction, and CAP actions. Provide the
physical inner products, propagation convenience method, compatibility
validation, and archive metadata. Keep the field-free physical action
available separately for energy sampling.

Acceptance gate:

- field-free propagation of a stationary state preserves norm and acquires
  the expected phase;
- invalid absorber/integrator combinations fail before propagation.

### Step 7: add observables and cadence-aware radial sampling

Implement raw norm, energy, dipole, induced-dipole, optional reference-state
population, and `P(r)` sampling. Ensure that CAP and mask loss remain visible
and that the physical energy excludes the CAP.

Acceptance gate:

- the quadrature integral of every stored `P(r)` equals the corresponding
  sampled norm;
- interval and final-sample behavior are tested.

### Step 8: validate the three initial integrators

Compare CN, SIL, and SIA on small field-free and short-pulse calculations.
Use an explicit small Hamiltonian or converged fine-step calculation as an
oracle. Verify time-step convergence, norm behavior, and observable agreement.

Acceptance gate:

- all three integrators agree within their documented numerical tolerances
  without an absorber and with a mask;
- SIA+CAP gives controlled monotonic outer-region loss;
- SIL+CAP is rejected.

### Step 9: complete practical CN support

Measure unpreconditioned iteration counts, implement or adapt a compatible
preconditioner, and validate GMRES and BiCGSTAB behavior. Treat CN+CAP as a
separate later gate.

Acceptance gate:

- mask-based CN completes the development pulse reliably with recorded
  iteration diagnostics.

### Step 10: add the high-level preparation interface

Create a calculation object or preparation function aligned with the
two-radial high-level conventions. Support arbitrary compatible initial
states, user-specified mass, field and absorber selection, all three initial
integrators, samplers, progress, and persistence.

Acceptance gate:

- the same initial state and physical inputs can be propagated with CN, SIL,
  or SIA by changing only integrator configuration.

### Step 11: establish gauge equivalence

Propagate one weak short pulse in both gauges. At pulse endpoints where the
vector potential vanishes, compare norms, populations, dipoles, energies, and
phase-aligned states. During the pulse, compare states only after applying the
appropriate gauge transformation.

Acceptance gate:

- length- and velocity-gauge differences decrease with radial, angular, and
  time-step refinement.

### Step 12: add the development example

Add

```text
examples/spherical/one_radial/h2plus_laser_dynamics.py
```

with a short, weak default pulse suitable for local development. Expose at
least:

```text
--gauge length|velocity
--integrator crank-nicolson|sil|sia
--absorber none|mask|cap
--particle-mass
--field-strength
--frequency or wavelength
--cycles
--time-step
--r-max
--l-max
--radial-probability-interval
```

The example must report the effective grid, pulse, gauge, integrator,
absorber, state source, mass, norm, and output locations. Defaults are smoke
test inputs, not convergence recommendations.

### Step 13: document and qualify the initial implementation

Document the wavefunction, operators, gauge conventions, scalar phases,
absorber compatibility, initial-state contract, observables, persistence,
example usage, and convergence responsibilities. Run the complete local test
suite before cluster performance or production studies.

Acceptance gate:

- documentation and examples use only the public interface;
- the implementation is validated against the existing spherical stationary
  solver and the independent prolate representation.

## 11. Validation matrix

The initial test matrix must include:

1. analytic `z` angular selection rules;
2. `z` and `p_z` metric Hermiticity;
3. direct `p_z` versus discrete-commutator comparison;
4. full-angular versus fixed-`m` operator agreement;
5. zero-field stationary-state phase and norm;
6. explicit-matrix small-grid propagation;
7. CN, SIL, and SIA time-step convergence;
8. mask behavior with all three integrators;
9. SIA+CAP norm loss and reflection reduction;
10. invalid SIL+CAP rejection;
11. length-versus-velocity gauge equivalence;
12. midpoint H2+ nuclear-scalar cancellation;
13. nonunit particle-mass propagation and velocity coupling;
14. arbitrary compatible initial-state propagation;
15. `P(r)` normalization and sampling cadence;
16. checkpoint, exact restart, and continuation equivalence;
17. short-pulse comparison with the prolate-spheroidal representation.

Correctness tests must use deliberately small grids. Physical production
calculations require independent convergence in radial extent, FEM-DVR order,
angular cutoff, time step, absorber onset and strength, and pulse duration.

## 12. Future arbitrary molecular orientation

The near-future target permits nuclei in, for example, the laboratory
`x`-`z` plane while retaining a laboratory-`z`-polarized field. Even though
the field interaction still has `Delta m = 0`, a tilted two-centre field-free
potential contains multipoles with nonzero magnetic projection and therefore
mixes different `m` channels.

That extension requires:

- a physical-system geometry accepting explicit Cartesian nuclear positions;
- a clearly selected spherical-expansion origin;
- nuclear multipoles with configurable nonzero `M_max`;
- full `(l,m)` field-free stationary states and propagation;
- full-angular inner products and observables in the high-level interface;
- orientation metadata and rotated-geometry regression tests;
- reconsideration of parity and reflection symmetries for each geometry.

The first milestone does not generalize
`OneElectronDiatomicSystem.nuclear_positions`. Instead it prepares for that
extension by implementing full-angular `z` and `p_z` actions, keeping
absorbers independent of `m`, separating laboratory field direction from
molecular-axis metadata, and containing the fixed-`m` assumption within the
initial dynamics wrapper.

Do not rotate every tilted molecule back to the `z` axis inside the
implementation. The physical purpose of the future extension is to vary the
molecular orientation relative to a fixed laboratory field.

## 13. Expected deliverables

The initial axial implementation should deliver:

1. tested full-angular and fixed-`m` `z` and `p_z` actions;
2. reusable axial length- and velocity-gauge interactions;
3. reusable radial mask and CAP support;
4. a fixed-`m` time-dependent Hamiltonian wrapper;
5. raw standard observables and cadence-controlled `P(r)` sampling;
6. validated CN, SIL, and SIA paths with the compatibility matrix in this
   document;
7. a high-level dynamics interface accepting arbitrary compatible initial
   states and user-specified particle mass;
8. checkpoint-compatible system, representation, field, gauge, absorber,
   integrator, sampling, and state metadata;
9. a short-pulse H2+ development example;
10. length-versus-velocity and spherical-versus-prolate validation evidence;
11. user documentation and explicit convergence responsibilities.

CN+CAP, arbitrary nuclear orientation, full-`m` dynamics, non-axial field
polarization, moving nuclei, and non-Born--Oppenheimer dynamics are later
milestones. The existing spherical two-radial implementation remains the
appropriate representation for quantum nuclear motion.
