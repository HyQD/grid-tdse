# Prolate-spheroidal one-electron TDSE implementation plan

## 1. Objective and status

This document assesses and plans a prolate-spheroidal implementation in
`grid-tdse` for clamped two-centre, one-electron systems. H2+ is the primary
target. The same field-free formulation must support arbitrary positive
nuclear charges, including LiH3+ (`Z1 = 3`, `Z2 = 1`) and BH5+
(`Z1 = 5`, `Z2 = 1`).

This is a design document only. Implementation has not started.

The recommended first production scope is:

- two clamped nuclei at fixed internuclear distance `R = 2a`;
- one dynamical electron;
- prolate coordinates tied to the fixed molecular axis;
- field-free eigenstates and real-time propagation;
- a linearly polarized electric field parallel to the molecular axis;
- the length gauge first;
- fixed magnetic quantum number `m`, initially `m = 0`;
- a post-step mask for initial absorbed propagation tests, followed by a
  cubic complex absorbing potential near the outer `xi` boundary;
- Davidson as the primary ground-state and stationary-state eigensolver;
- serial execution first, followed by threaded and MPI assessment only after
  the operator has been validated.

Moving nuclei, rotational dynamics, arbitrary polarization, and a velocity-
gauge implementation are deliberate later extensions. In particular, this
plan does not replace the existing space-fixed spherical two-radial
non-Born--Oppenheimer representation.

## 2. Assessment

### 2.1 Suitability of prolate spheroidal coordinates

For nuclei at `z = -a` and `z = +a`, use

```math
x = a\sqrt{(\xi^2-1)(1-\eta^2)}\cos\phi,
\qquad
y = a\sqrt{(\xi^2-1)(1-\eta^2)}\sin\phi,
\qquad
z = a\xi\eta,
```

with `1 <= xi < infinity` and `-1 <= eta <= 1`. The distances from the two
foci are

```math
r_1=a(\xi+\eta),
\qquad
r_2=a(\xi-\eta).
```

The electron--nuclear attraction is therefore local:

```math
V_{\mathrm{en}}(\xi,\eta)
=
-\frac{Z_1}{a(\xi+\eta)}
-\frac{Z_2}{a(\xi-\eta)}
=
-\frac{(Z_1+Z_2)\xi+(Z_2-Z_1)\eta}
       {a(\xi^2-\eta^2)}.
```

This representation is especially attractive for a two-centre one-electron
problem because both Coulomb singularities coincide with coordinate foci. It
avoids the slowly convergent spherical multipole representation of displaced
nuclei used by the current one-radial spherical implementation. It should
therefore reach accurate H2+ field-free states with substantially fewer
angular degrees of freedom.

The same formula already supports unequal charges. LiH3+ and BH5+ require no
new field-free Coulomb derivation; the term proportional to
`(Z2 - Z1) eta` represents their heteronuclear asymmetry.

### 2.2 Existing `grid-lib` foundation

The current `grid_lib.prolate_spheroidal_coordinates` package already
provides the essential field-free numerical primitives:

- `ProlateSpheroidalGrid`;
- a focus-safe first right-Radau `xi` element followed by Lobatto elements;
- a Gauss--Legendre `eta` grid without axis endpoint nodes;
- parity-adapted axis regularity for odd absolute `m`;
- separable weak-form kinetic matrices;
- the diagonal overlap metric;
- the arbitrary-`Z1`, arbitrary-`Z2` two-centre Coulomb matrix;
- active-node selection and outer Dirichlet removal;
- validated `m = 0` and nonzero-`m` field-free calculations.

The H2 Hartree--Fock demo is not itself the foundation that should be copied
into `grid-tdse`. It demonstrates how the public grid and operator primitives
can be composed. For the present one-electron target, the more direct
reference is `dev/reproduce_tao_table1_h2plus.py`, which validates H2+ states
against published prolate-DVR energies.

The electron--electron Neumann and Poisson solvers are not required for this
one-electron implementation.

### 2.3 Initial development grid

Use a grid comparable to the H2 calculation in
`grid-lib/dev/compare_haxton_poisson_h2.py` for initial development:

```text
xi element boundaries:       (1.0, 4.0, 14.0, 24.0)
xi points per element:       14
eta Gauss-Legendre points:   9
m:                           0
outer xi boundary:           Dirichlet
```

The assembled `xi` grid has 40 unique nodes. Together with the 9 `eta` nodes,
this gives 360 product nodes and 351 active nodes after the outer `xi` row is
removed.

Use two geometries for distinct purposes:

1. `R = 1.4` bohr reproduces the exact grid geometry in the existing Haxton
   H2 development example and provides a direct cross-package assembly and
   metric regression test;
2. `R = 2.0` bohr with `Z1 = Z2 = 1` is the initial H2+ development problem,
   using the same dimensionless grid and discretization.

For H2+ at `R = 2`, `a = 1` and the outer semi-major coordinate is
`q_max = a xi_max = 24` bohr. For the exact Haxton geometry at `R = 1.4`, it
is 16.8 bohr.

This is a development and regression grid, not an HHG production grid and not
a radial or angular convergence claim. Later field-driven calculations must
increase the outer extent according to the pulse excursion and independently
converge the `xi` and `eta` resolutions.

### 2.4 What is missing for TDSE propagation

The existing package does not yet provide a complete time-dependent
representation. The missing pieces are:

1. a public constructor for nonuniform `xi` element boundaries, avoiding the
   manual reconstruction of `ProlateSpheroidalGrid` used by development
   demos;
2. reusable physical coordinate arrays such as `z = a xi eta`, cylindrical
   radius, and Cartesian radius;
3. reusable local-potential/weak-diagonal construction;
4. a propagation-facing fixed-`m` discretization with active indices,
   overlap metric, and strong/symmetric transformations;
5. length-gauge dipole operators;
6. an absorber defined consistently on the prolate outer domain;
7. TDSE observables, archive metadata, snapshots, and restart validation;
8. later, a Hermitian velocity-gauge momentum operator and couplings between
   different `m` sectors for non-axial polarization.

The representation-independent `grid_tdse.real_time.RealTimePropagator`
already supplies CN, SIL, SIA, and CF4 time integration. It accepts a
representation-specific Hamiltonian action and physical inner product, so a
new integrator is not needed.

### 2.5 Important limitations

The prolate coordinates are tied to a fixed internuclear distance and a
body-fixed molecular axis. Promoting `R` to a dynamical coordinate is not a
small extension: the coordinate scale `a = R/2` then depends on a quantum
coordinate, and the nuclear kinetic energy generates additional derivative
and nonadiabatic coupling terms. Full non-Born--Oppenheimer H2+ should remain
in the existing spherical two-radial representation unless a separate
body-fixed derivation is undertaken.

Similarly, an isotropic rotational ground state cannot be represented by one
fixed prolate axis. Isotropic or aligned ensembles can initially be treated
as orientation averages of clamped-axis calculations. A quantum rotational
wavepacket belongs to the space-fixed spherical implementation.

The outer surface `xi = xi_max` is an ellipsoid, not a sphere. At large `xi`
it becomes nearly spherical, but CAP placement and radial observables must be
defined using explicit physical coordinates rather than treating `xi` as a
Cartesian radius.

## 3. Package responsibilities

The implementation should preserve the current package separation:

- `grid-lib` owns coordinate grids, quadrature, differentiation, weak-form
  geometric operators, active-node maps, and generic coordinate functions;
- `integrator-lib` owns representation-independent propagators;
- `electromagnetic-fields` owns temporal field profiles;
- `grid-tdse` owns physical systems, wavefunction representations,
  Hamiltonian assembly, field coupling, CAP policy, observables, ground-state
  workflows, checkpoints, and user-facing calculations.

Do not move pulse handling, H2+-specific Hamiltonian policy, or propagation
workflows into `grid-lib`.

## 4. Recommended `grid-lib` prerequisite

A `grid-lib` update is advisable but should be a small modularization, not a
monolithic prolate Hamiltonian framework. The current low-level operators are
already appropriately separated.

### 4.1 Required before production implementation

Add and test:

1. `setup_prolate_spheroidal_grid_from_boundaries(...)`, accepting explicit
   strictly increasing `xi` boundaries, per-element or uniform point counts,
   and an explicit `eta` degree;
2. coordinate helpers returning `x`, `y`, `z`, cylindrical radius, Cartesian
   radius, and optionally the two focus distances;
3. a generic helper that forms a weak diagonal from a pointwise local
   potential by multiplying by the overlap weights;
4. explicit transforms between nodal coefficients with metric `S` and the
   Euclidean representation using `S^(1/2)`;
5. tests of coordinate identities, metric transformations, weak local
   operators, and custom-boundary setup.

The custom-boundary constructor is the most immediate modularization need.
Both existing development demos currently perform this assembly themselves.

### 4.2 Add when the corresponding TDSE feature is reached

Add generic prolate spatial operators only when needed:

- axial length operator `z = a xi eta`;
- transverse coordinate factors that couple `m` to `m +/- 1`;
- a weak-form Hermitian representation of `p_z` for velocity gauge;
- transverse momentum operators for arbitrary polarization.

These are coordinate-level operators and belong in `grid-lib`. Their temporal
field amplitudes and charge-dependent physical prefactors belong in
`grid-tdse`.

### 4.3 What does not require `grid-lib` changes

A minimal field-free H2+ solver and an axial length-gauge proof of concept can
be assembled from the current API. Therefore, a `grid-lib` update is not a
strict mathematical prerequisite. It is nevertheless recommended before a
production interface so that `grid-tdse` does not duplicate grid assembly,
metric transformations, and coordinate geometry.

## 5. Wavefunction representation

For a fixed magnetic quantum number `m`, use

```math
\Psi_m(\xi,\eta,\phi,t)
=
\psi_m(\xi,\eta,t)
\frac{e^{im\phi}}{\sqrt{2\pi}}.
```

The nodal state has shape `(n_xi_active, n_eta)` or an equivalent flattened
active-space layout. The physical norm is

```math
\langle\Psi|\Psi\rangle
=
\mathbf c^\dagger S\mathbf c,
```

where `S` is the diagonal prolate overlap metric.

The weak field-free equation is

```math
A_m\mathbf c = E S\mathbf c.
```

For propagation, either of the following equivalent representations may be
used:

1. nodal coefficients with the strong action `S^(-1) A_m c` and the physical
   `S`-weighted inner product;
2. Euclidean coefficients `y = S^(1/2)c` with Hamiltonian
   `S^(-1/2) A_m S^(-1/2)`.

Use nodal coefficients as the public representation because local potentials
and observables are direct at DVR nodes. Retain the Euclidean transform for
Hermiticity tests, eigensolvers, and independent validation. Confirm with
small matrices that SIL sees a self-adjoint action in the supplied physical
inner product.

For an axial linearly polarized field, `m` is conserved. The H2+ ground state
therefore remains in `m = 0`, although its homonuclear `eta` parity alternates
under successive dipole interactions. Do not restrict real-time dynamics to
one `eta` parity.

## 6. Field-free Hamiltonian

For particle mass `mu`, construct

```math
H_0
=
-\frac{1}{2\mu}\nabla^2
-\frac{Z_1}{r_1}
-\frac{Z_2}{r_2}.
```

The constant nuclear repulsion

```math
V_{\mathrm{nn}}=\frac{Z_1Z_2}{R}
```

must be exposed separately and included in reported clamped-nuclei total
energies, but it may be omitted from propagation because it contributes only
a global phase.

Create a coordinate-independent physical system description rather than
duplicating `OneElectronDiatomicSystem`. That class currently lives under
`grid_tdse.spherical.one_radial` even though its charges, geometry, and mass
are not spherical-representation concepts. Move it to a shared system module
with a compatibility re-export from the existing import path.

The prolate Hamiltonian should expose:

- weak and strong field-free actions;
- physical and Euclidean inner products/transforms;
- explicit small-system matrices for testing;
- fixed-`m` active state shape;
- electronic and total energy evaluation;
- serial batched actions for Krylov vectors.

### 6.1 Target public ground-state interface

The user-facing workflow must closely match
`examples/spherical/one_radial/one_electron_diatomic_ground_state.py`:

1. construct a representation-independent `OneElectronDiatomicSystem`;
2. construct a coordinate-specific Hamiltonian;
3. call `diagonalize_fixed_m(hamiltonian, m=..., n_states=...)`;
4. inspect the same result properties used by the spherical one-radial path.

The intended prolate usage is:

```python
from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
)

system = OneElectronDiatomicSystem(
    nuclear_charge1=1.0,
    nuclear_charge2=1.0,
    half_bond_length=1.0,
    particle_mass=1.0,
    name="H2+",
)

hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
    system,
    xi_boundaries=[1.0, 4.0, 14.0, 24.0],
    xi_points_per_element=14,
    eta_degree=9,
    m_max=0,
)

result = diagonalize_fixed_m(
    hamiltonian,
    m=0,
    n_states=1,
    residual_tolerance=1.0e-10,
)

print(result.electronic_energy)
print(result.total_energy)
orbital = result.orbital
```

This is the initial Haxton-comparable development grid defined in Section
2.3, not a convergence recommendation.

`diagonalize_fixed_m` should select Davidson by default. The ordinary call
therefore remains as compact as the current spherical interface. An optional
`solver="direct"` path may be exposed for tiny validation grids, but it must
not be the production default. Solver-specific expert options may be accepted
as explicit keyword arguments or a typed options object without changing the
basic call pattern.

The result contract should retain:

- `hamiltonian` and `m`;
- `electronic_energies` and `total_energies`;
- `orbitals`;
- lowest-root conveniences `electronic_energy`, `total_energy`, and
  `orbital`.

Add Davidson information without removing those properties:

- convergence and termination status for each requested root;
- residual norms;
- iteration and Hamiltonian-action counts;
- restart and preconditioner diagnostics;
- optional iteration histories.

For `n_states = 1`, route the call through the existing single-root Davidson
solver. For `n_states > 1`, route it through the planned generic block-
Davidson implementation. The public function and result layout should not
change when the multi-root implementation becomes available.

## 7. Electric-field interaction

### 7.1 First implementation: axial length gauge

For a field parallel to the molecular axis, the electronic length operator is
local because

```math
z=a\xi\eta.
```

Use the charge and sign convention already established by the existing
`grid-tdse` dipole interfaces. Keep the temporal electric field separate from
the field-independent spatial dipole action.

For unequal charges, the clamped nuclei have a constant dipole about the bond
midpoint. In a prescribed uniform field this adds a time-dependent scalar to
the Hamiltonian and hence only a global phase for fixed nuclei. Record the
chosen origin and nuclear contribution in metadata and include it where a
reported total dipole requires it.

### 7.2 Later: velocity gauge

Derive and implement `p_z` in the prolate basis in weak form. Validate metric
Hermiticity and compare length- and velocity-gauge dynamics under systematic
grid, box, and time-step refinement. Do not construct `p_z` by an unverified
pointwise transformation of first-derivative matrices.

### 7.3 Later: non-axial and arbitrary polarization

Perpendicular field components couple `m` to `m +/- 1` and require the
odd-absolute-`m` axis-regular basis already supported by `grid-lib`. Introduce
an explicit ordered table of retained signed `m` sectors and sparse coupling
records. Validate the coupled action against direct coordinate-space
quadrature on small grids.

## 8. Absorber and outer-domain policy

Use the physical prolate semi-major coordinate

```math
q=a\xi
```

to parameterize the first absorber. Constant `q`, like constant `xi`, defines
an ellipsoid, but using `q` gives the absorber parameters dimensions of length
and makes their interpretation less dependent on the chosen bond length.
Document the corresponding `xi` value and the minimum and maximum Cartesian
radii of the onset surface.

### 8.1 Initial mask option

A post-step mask is an acceptable first absorber. It can use the same cosine
profile as the existing spherical implementation,

```math
M(q)=
\begin{cases}
1, & q < q_0,\\
\cos\!\left[
\dfrac{\pi(q-q_0)}{2(q_{\max}-q_0)}
\right]^{1/n}, & q_0\le q\le q_{\max}.
\end{cases}
```

The representation-independent real-time propagator already provides a
post-step transformation hook suitable for applying this diagonal mask. This
is the shortest path to testing outgoing-wave absorption while retaining a
Hermitian field-free Hamiltonian inside each propagation step.

A fixed mask applied after every step has an effective absorption strength
that depends on the time step and application frequency. It must therefore
not be treated as a converged production absorber without explicit time-step
and mask-parameter studies. A more controlled split absorber can instead be
derived from a nonnegative rate `W(q)` through

```math
M_{\Delta t}(q)=\exp[-\Delta t\,W(q)].
```

This makes the intended continuous-time absorption rate explicit, although a
post-step application still introduces operator-splitting error.

### 8.2 Recommended production CAP

Use a cubic CAP consistent with the established spherical convention:

```math
V_{\mathrm{CAP}}(q)
=
-i\eta\,\max(q-q_{\mathrm{CAP}},0)^3.
```

Equivalently, specify `q_CAP = alpha * q_max` with `0 <= alpha < 1`. The CAP
is diagonal at the prolate DVR nodes. Its weak matrix is the pointwise
potential multiplied by the overlap diagonal.

Including the CAP directly in the Hamiltonian avoids the first-order
post-step splitting associated with a simple mask and allows the requested
integrator accuracy to apply to the complete non-Hermitian Hamiltonian. A CAP
makes the Hamiltonian non-Hermitian, so real-time production runs should use
SIA or CF4-SIA rather than SIL.

The recommended progression is therefore:

1. no absorber for field-free and short norm-conservation tests;
2. a `q`-based cosine mask for the first outgoing-wave smoke tests;
3. a `q`-based cubic CAP for production dynamics;
4. comparison with a Cartesian-radius CAP if ellipsoidal-onset effects are
   measurable.

A Cartesian-radius CAP has spherical onset but varies over both `xi` and
`eta`; it nevertheless remains diagonal in the product DVR.

## 9. Observables and archives

The initial observable set should include:

- norm and norm loss;
- electronic and total field-free energy;
- axial electronic dipole and total clamped-nuclei dipole;
- probability marginals in `xi` and `eta` with the full metric;
- population in selected field-free states;
- probability near and inside the CAP;
- optional dipole velocity and acceleration only after their operators have
  independent validation.

HHG spectra should not initially rely only on a numerically delicate Coulomb-
force acceleration at the foci. Begin with the dipole time series and add
velocity/acceleration forms as explicit gauge and observable validation
steps.

Checkpoint and result archives must record at least:

- `Z1`, `Z2`, particle mass, `R`, and coordinate origin;
- `m` sectors and any symmetry reduction;
- `xi` boundaries, points per element, `eta` degree, and active-node policy;
- CAP definition;
- field and gauge metadata;
- metric/representation convention;
- source package versions and integrator settings.

## 10. Step-by-step implementation plan

### Step 1: freeze the initial scope -- complete

The decisions in Section 12 are resolved. The initial scope is clamped H2+
at `R = 2` bohr, axial linear polarization, length gauge, conserved `m = 0`,
one fixed orientation per run, arbitrary positive nuclear charges in the
general API, Davidson-first stationary-state solution, an initial post-step
mask followed by a production cubic CAP, and the Haxton-comparable
development grid from Section 2.3. Velocity gauge, non-axial polarization,
and quantum nuclear motion are later milestones.

### Step 2: modularize the required `grid-lib` setup

Add the custom-boundary grid constructor, coordinate helpers, weak local
diagonal helper, and metric transforms described in Section 4. Add unit tests
without introducing TDSE-specific policy.

### Step 3: make the physical system representation-independent

Move the current `OneElectronDiatomicSystem` implementation to a shared
`grid-tdse` system module. Preserve its existing public import as a
compatibility re-export. Add named constructors only if they clarify, rather
than hide, the charges of H2+, LiH3+, and BH5+.

### Step 4: implement the prolate fixed-`m` representation

Create the grid, active map, overlap, inner product, nodal/Euclidean
transforms, state validation, and archive metadata. Test normalization and
round-trip transformations for even and odd absolute `m`. Use the
Haxton-comparable grid from Section 2.3 as the primary `m = 0` development
fixture.

### Step 5: implement the field-free Hamiltonian

Compose the existing prolate kinetic and electron--nuclear weak matrices.
Expose weak, strong, and symmetric actions. Test metric Hermiticity,
matrix/action agreement, and independent direct quadrature of local terms.

### Step 6: implement Davidson ground-state and stationary-state workflows

Use the existing representation-independent `DavidsonGroundStateSolver` as
the starting point, not dense diagonalization. Supply the prolate physical
inner product and a preconditioner that acts in the same nodal
representation.

The existing solver targets the lowest root. Use it directly for the ground
state. For several stationary states, extend the representation-independent
ground-state eigensolver layer with a multi-root or block-Davidson algorithm,
rather than implementing repeated prolate-specific solves. The extension
should support an initial vector block, root-wise residual tests, locking or
deflation of converged roots, metric orthogonalization, thick restart, and a
Davidson correction for each unconverged Ritz pair. Preserve the existing
single-root API as a compatibility path.

The first preconditioner should be the safeguarded diagonal Davidson
preconditioner constructed from the physical diagonal of the strong or
Euclidean-symmetric Hamiltonian. The two diagonals agree under the diagonal
metric similarity transform. Record regularized denominators and fallback
counts using the existing Davidson diagnostics.

After the diagonal path is validated, assess a stronger preconditioner based
on separable prolate kinetic pieces or an approximate shifted sparse solve.
Do not delay the initial implementation on the stronger preconditioner.

Retain a small explicit generalized `eigh` or sparse `eigsh` solve only as a
test oracle for tiny grids. Validate Davidson energies, residual norms, and
phase-aligned eigenvectors against that oracle and the published H2+
references. Return electronic and total energies separately. Provide an
example such as
`examples/prolate_spheroidal/one_electron_diatomic_ground_state.py`.

Make that example intentionally parallel to the existing spherical
one-radial example. It should differ only in the Hamiltonian/grid
construction and in the additional Davidson convergence diagnostics, not in
the overall user workflow.

### Step 7: establish field-free validation gates

Before real-time work, require:

1. reproduction of the existing Tao H2+ `1s sigma_g`, `2p sigma_u`, and
   selected `pi` reference states at `R = 2`;
2. agreement between matrix and action residuals;
3. convergence trends in `xi_max`, `xi` element order, and `eta` degree;
4. agreement with the existing spherical one-radial implementation in a
   jointly converged finite box;
5. independent one-electron references for at least one heteronuclear system;
6. sensible united-atom and separated-centre limiting trends.

The published excited-state gate includes the `Sigma_u` and `Pi_u` entries
from Table II of Tao, McCurdy, and Rescigno, Phys. Rev. A 80, 013402 (2009),
at `R = 2`. Compare both eigenvalues and excitation energies, and classify
every numerical root by inversion rather than assigning states solely by
energy order. The table's `4p sigma_u` energy, `-0.177681` hartree, must be
reported as a reference-label inconsistency: the corresponding numerical
eigenstate is gerade. Preserve this distinction instead of silently
relabelling the computed state.

Provide the executable comparison in
`examples/prolate_spheroidal/one_electron_diatomic_excited_states.py`. It may
use the direct solver as a deterministic reference oracle; Davidson remains
the production default and is demonstrated by the ground-state example.

Provide the heteronuclear gate in
`examples/prolate_spheroidal/one_electron_diatomic_heteronuclear_validation.py`.
It must compare the HeH2+, LiH3+, and BeH4+ tabulated ground-state energies
of Campos et al. and Demkov's exact BH5+ `3p sigma` solution. Report
electronic and clamped-nuclei total energies separately so that the nuclear
repulsion convention is unambiguous.

Do not use the H2 Hartree--Fock iteration as the primary one-electron
validation oracle.

### Step 8: implement axial length-gauge coupling

Add the field-independent `z` action and a temporal-field wrapper compatible
with `electromagnetic-fields`. Test selection rules, metric Hermiticity,
homonuclear parity changes, and zero-field equivalence.

### Step 9: add real-time propagation

Connect the representation to `RealTimePropagator`. Qualify field-free phase
evolution and norm conservation with SIL/SIA, then a weak short pulse without
an absorber. Compare with the spherical one-radial implementation.

### Step 10: implement and qualify mask and CAP absorption

First add the `q`-based cosine mask through the existing post-step hook and
use it for short outgoing-wave smoke tests. Then add the `q`-based cubic CAP,
require SIA or CF4-SIA for its non-Hermitian propagation, and make the CAP the
production default. Test reflection, norm loss, time-step sensitivity, onset
sensitivity, absorber strength, and box enlargement. Preserve only the latest
restart checkpoint according to the current checkpoint policy.

### Step 11: implement dynamics observables and HHG output

Add dipole sampling, state populations, CAP diagnostics, and windowed Fourier
analysis inputs. Validate sampling and archive restart behavior before making
an HHG production example.

### Step 12: validate the target system family

Run matched field-free and short-dynamics tests for:

- H2+ (`Z1 = Z2 = 1`);
- LiH3+ (`Z1 = 3`, `Z2 = 1`);
- BH5+ (`Z1 = 5`, `Z2 = 1`).

For heteronuclear systems, explicitly test loss of `eta` parity and the
origin-dependent static nuclear dipole bookkeeping.

### Step 13: add velocity gauge

After the length-gauge path is stable, add the verified prolate `p_z`
operator and require converged length-versus-velocity comparisons for
field-free energy, norm, final-state overlap, dipole history, and HHG spectra.

### Step 14: add non-axial polarization if required

Introduce multiple signed-`m` sectors and transverse dipole couplings. Test
against rotated weak-field calculations and direct quadrature. This step is
not needed for a first parallel-aligned H2+ HHG implementation.

### Step 15: performance qualification

Profile sparse-matrix, separable-Kronecker, and matrix-free actions. Begin
with one process and controlled BLAS/Numba threading. Add `xi`-domain MPI
partitioning only if production grid sizes or repeated HHG propagation justify
it. A two-dimensional one-electron state is much smaller than the existing
two-radial non-BO wavefunction, so MPI should not be assumed necessary.

### Step 16: production convergence study

Converge `xi_max`, CAP onset, `xi` element boundaries and order, `eta` degree,
time step, Krylov tolerance, and pulse sampling. For HHG, ensure the CAP lies
outside the relevant returning long trajectories and compare dipole,
velocity, and acceleration spectra when all three observables are available.

## 11. Expected benefits and risks

Expected benefits are:

- direct and local representation of both nuclear Coulomb singularities;
- substantially smaller angular representations for clamped diatomics;
- natural treatment of homonuclear and heteronuclear one-electron systems;
- simple axial length-gauge coupling;
- likely feasibility on a single node for large HHG boxes.

Primary risks are:

- incorrect handling of the generalized overlap metric during propagation;
- loss of Hermiticity in hand-constructed momentum operators;
- CAP reflections from an ellipsoidal outer boundary;
- focus and axis convergence problems for high charges or nonzero `m`;
- conflating a fixed molecular axis with isotropic or rotational dynamics;
- premature MPI complexity for a problem that may be efficiently threaded.

## 12. Resolved design decisions

The initial implementation decisions are:

1. nuclei are clamped at a user-specified `R`; nuclear vibration and rotation
   are outside the initial scope;
2. the first laser is linearly polarized parallel to the molecular axis, and
   the H2+ ground-state dynamics begin in conserved `m = 0`;
3. length gauge is implemented first;
4. HHG is the primary real-time target;
5. one fixed molecular orientation is represented per run, with isotropic or
   aligned ensemble averaging handled by a separate driver;
6. the bond midpoint is the coordinate and dipole origin, and heteronuclear
   archives record that origin and the constant nuclear dipole explicitly;
7. the field-free API accepts arbitrary positive `Z1` and `Z2` directly;
8. a `q = a xi` cosine mask is permitted for initial absorption tests, while
   a cubic CAP in the same coordinate is the production target;
9. velocity gauge and non-axial polarization are later milestones;
10. Davidson is the primary ground-state and stationary-state solver from the
    start. The existing single-root solver is used for the ground state, and
    a generic multi-root/block-Davidson extension supplies excited stationary
    states. Both use a diagonal preconditioner first and retain explicit
    diagonalization only as a small-grid validation oracle. The public
    `diagonalize_fixed_m` workflow and eigenstate result properties mirror the
    existing spherical one-radial interface.
11. the initial development discretization uses `xi` boundaries
    `(1, 4, 14, 24)`, 14 points per `xi` element, and 9 Gauss--Legendre `eta`
    points. The exact Haxton geometry at `R = 1.4` is a regression case, while
    H2+ development uses the same discretization at `R = 2.0`.
