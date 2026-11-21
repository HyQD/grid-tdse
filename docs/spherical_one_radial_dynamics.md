# Spherical one-radial laser dynamics

`grid_tdse.spherical.one_radial.OneElectronDiatomicTDSE` is the public
high-level interface for real-time, clamped-nuclei, one-electron diatomic
calculations in spherical coordinates. The backward-compatible `"fixed_m"`
representation targets a molecule on laboratory `z`. The `"full"`
representation accepts explicit Cartesian nuclear positions and retains all
configured complex `(l,m)` channels for a linearly polarized laboratory-`z`
field. Both paths support unequal charges and a user-selected particle mass.

The interface prepares the spatial Hamiltonian, field interaction, absorber,
integrator, observables, progress reporting, and checkpoint policy. It accepts
an arbitrary compatible initial state; a stationary ground state is a common
choice, not a requirement of the time-dependent implementation.

## Representation and field-free Hamiltonian

For `angular_representation="fixed_m"`, the nuclei are fixed at

```math
\mathbf R_1=(0,0,-R/2),
\qquad
\mathbf R_2=(0,0,R/2).
```

For one conserved signed magnetic quantum number `m`, the wavefunction is

```math
\Psi(\mathbf r,t)
=
\frac{1}{r}
\sum_{l=|m|}^{l_{\max}}
u_{lm}(r,t)Y_{lm}(\hat{\mathbf r}).
```

The public state is a radial-first array with shape
`(n_r, l_max - abs(m) + 1)`. The radial values are physical FEM-DVR nodal
coefficients on the active grid; homogeneous Dirichlet conditions remove the
first and last radial nodes. The physical norm is

```math
N(t)
=
\sum_{i,l} w_i |u_{lm}(r_i,t)|^2,
```

where `w_i` are the FEM-DVR radial weights.

For `angular_representation="full"`, the state is

```math
\Psi(\mathbf r,t)
=
\frac{1}{r}
\sum_{l=0}^{l_{\max}}
\sum_{m=-\min(l,m_{\max})}^{\min(l,m_{\max})}
u_{lm}(r,t)Y_{lm}(\hat{\mathbf r}),
```

with shape `(n_r, n_lm)` in the public grid-lib channel ordering. The
rotation-complete default is `m_max=l_max`. An off-axis geometry requires
this full representation; the fixed-`m` path rejects it explicitly.

The electronic field-free Hamiltonian is

```math
H_0
=
-\frac{1}{2\mu}\nabla^2
-\frac{Z_1}{|\mathbf r-\mathbf R_1|}
-\frac{Z_2}{|\mathbf r-\mathbf R_2|}.
```

Here `mu` is `system.particle_mass`. The static nuclear-repulsion energy is
not part of the propagation generator. Add it separately when reporting a
clamped-nuclei total stationary energy.

The two-centre Coulomb potential is projected into spherical multipoles. For
an orbital cutoff `l_max`, `nuclear_L_max=2*l_max` retains every multipole
that can couple two retained orbital channels. This is an exact cutoff for
the projected finite angular space, but it does not remove the need to
converge `l_max`.

## Axial fields and gauge conventions

Temporal fields come from `electromagnetic-fields`. A length-gauge profile
returns the electric field `E_z(t)`; a velocity-gauge profile returns the
vector potential `A_z(t)`. The field object's `gauge` must agree with
`field_gauge`.

### Length gauge

About an axial origin `z_0`, the retained interaction is

```math
V_L(t)
=
-E_z(t)q_e(z-z_0)
-E_z(t)d_{\mathrm{nuc},z}(z_0),
```

with electron charge `q_e` and

```math
d_{\mathrm{nuc},z}(z_0)
=
\sum_A Z_A(R_{A,z}-z_0).
```

The nuclear term is spatially constant and changes only the global phase.
It is omitted by default and may be retained with
`include_nuclear_scalar=True`. It vanishes for homonuclear nuclei placed
symmetrically about the midpoint origin.

### Velocity gauge

The retained linear minimal-coupling interaction is

```math
V_V(t)
=
-A_z(t)\frac{q_e}{\mu}p_z.
```

The spatially constant term

```math
\frac{q_e^2 A_z(t)^2}{2\mu}
```

is omitted because it contributes only a global phase. A velocity-gauge run
requires `A_z(initial_time)=0`; otherwise the caller must first transform the
initial state explicitly.

The public `transform_axial_gauge_state()` helper uses

```math
\Psi_L(\mathbf r,t)
=
\exp[-i q_e A_z(t)z]\Psi_V(\mathbf r,t)
```

and projects the transformed state back into the finite angular space. The
`transform_full_angular_axial_gauge_state()` variant applies the same
projection independently in every signed-`m` block. At finite `l_max`, the
retained space is not closed under the exact phase.
The resulting norm defect and length--velocity difference are therefore
useful angular-convergence diagnostics, not evidence that the transform
silently preserves information.

## Initial-state contract

`initial_state` may be a stationary state, an excited state, a coherent
superposition, or a compatible state prepared elsewhere. The high-level
interface requires the exact state shape for the selected representation,
finite numeric values, and a positive finite norm. It copies the state but
does **not** normalize it.
Consequently, absorber-induced norm loss and deliberately unnormalized input
remain visible.

For an orientation-dependent ground-state campaign, solve Davidson with the
full-angular field-free Hamiltonian at each requested nuclear geometry. Do
not silently obtain tilted initial states by rotating one axial solution.
Rotation remains an exact finite-space covariance check when all magnetic
channels and nuclear multipoles are present, but geometry-local stationary
solves provide an independent preparation record for every dynamics run.

The package-level interface accepts an array. Archive selection and loading
are application policies; the H2+ example accepts NPY files and NPZ files
containing `final_state`, `state`, or `initial_state`.

## Integrators and absorbing boundaries

The supported initial combinations are:

| Boundary treatment | Crank--Nicolson | SIL | SIA |
|---|:---:|:---:|:---:|
| none | yes | yes | yes |
| radial cosine mask | yes | yes | yes |
| cubic CAP | no | no | yes |

A `RadialCosineMask` is applied after every accepted step. A `CubicCAP`
contributes a non-Hermitian term to the Hamiltonian and therefore uses SIA in
the public interface. Configuring both simultaneously is rejected. CAP
support for CN and SIL has not been qualified.

Crank--Nicolson uses a mass-aware, channelwise radial
kinetic-plus-centrifugal preconditioner by default. It can be disabled or
replaced through `integrator_options`. SIL and SIA use the physical FEM-DVR
inner product throughout their Krylov construction.

No propagation path renormalizes the wavefunction after a step.

## Observables

Scalar observables are sampled at every accepted step. They preserve the
propagated norm: expectation values are raw, unnormalized numerators, and
reference populations are raw squared overlaps with normalized references:

- `norm`: `N(t)`;
- `field_free_energy_numerator`: the real part of
  `<Psi(t)|H_0|Psi(t)>`, excluding any CAP;
- `electronic_dipole_numerator`: `<Psi|q_e(z-z_0)|Psi>`;
- `total_dipole_numerator`: electronic plus clamped-nuclear dipole;
- `induced_dipole_numerator`: total dipole relative to its initial
  conditional value;
- `canonical_momentum_numerator`: the real part of `<Psi|p_z|Psi>`;
- `mechanical_momentum_numerator`, in velocity gauge only:

```math
\langle p_z-q_eA_z(t)\rangle_{\mathrm{raw}}
=
\langle p_z\rangle_{\mathrm{raw}}-q_eA_z(t)N(t);
```

- `state_populations`, the raw squared overlaps when normalized reference
  states are supplied.

In the full representation, the electronic, nuclear, total, induced-dipole,
and canonical-momentum samplers return Cartesian vectors ordered `(x,y,z)`.
The velocity-gauge mechanical momentum is also a Cartesian vector; only its
laboratory-`z` component receives the `-q_e A_z(t) N(t)` correction.

Conditional expectation values and conditional reference populations can be
recovered in postprocessing by dividing the raw value by `norm`. This
convention is essential when a mask or CAP removes probability.

The radial distribution is

```math
P(r_i,t)=\sum_{lm} |u_{lm}(r_i,t)|^2,
```

and satisfies `sum_i w_i P(r_i,t)=N(t)`. It is stored separately at the
user-selected `radial_probability_interval`, always including the initial
and final states and any checkpoint step. Set the interval to `None` to
disable radial-distribution storage.

## Persistence and continuation

The calculation records the system, geometry, Cartesian origin, grid,
particle mass, angular representation, channel ordering, angular cutoffs,
field and gauge, absorber, integrator, preconditioner, initial-state identity,
and observable policy in checkpoint metadata.

- `run()` or `propagate()` starts from a copy of `initial_state`.
- `resume()` exactly resumes an interrupted compatible propagation segment.
- `continue_from()` starts a new compatible segment from a checkpoint.

Use `checkpoint_path` and `checkpoint_interval` to enable rolling
checkpoints. The generic real-time layer validates representation and
dynamics metadata before restart. See [Real-time propagation](real_time_propagation.md)
for the underlying checkpoint semantics.

## Public-interface example

The compact development calculation is

```bash
python examples/spherical/one_radial/h2plus_laser_dynamics.py
```

Its defaults are smoke-test inputs, not convergence recommendations. For an
illustrative velocity-gauge calculation with an absorber:

```bash
python examples/spherical/one_radial/h2plus_laser_dynamics.py \
    --gauge velocity \
    --integrator sia \
    --absorber cap \
    --wavelength-nm 800 \
    --field-strength 0.05 \
    --r-max 64 \
    --l-max 12 \
    --time-step 0.03125 \
    --cap-alpha 0.8 \
    --cap-eta 1.0
```

The example uses only public `grid-tdse` interfaces. It computes a projected
ground state unless `--initial-state` is supplied, prints the effective
setup, and writes a JSON summary plus a compressed NPZ archive containing the
final state, raw observables, integrator diagnostics, and sampled radial
distributions. Use `--help` for all controls.

## Qualification

The stationary Hamiltonian and spherical multipole construction are
qualified separately in
[One-electron diatomic stationary-state validation](one_electron_diatomic_validation.md).
That validation includes high-precision H2+ references and an independent
prolate-spheroidal comparison.

The real-time regression suite adds three complementary qualifications:

1. Paired length- and velocity-gauge H2+ calculations use a one-cycle
   `SineSquareLaser` with `E0=0.05` and `omega=4`. Endpoint state, dipole,
   population, energy, in-pulse transformed-state, and projection-defect
   discrepancies all decrease across `l_max=2,4,6,8`, by more than one order
   of magnitude from the first to the last cutoff. Separate radial and
   time-step cases ensure that those errors are controlled before the angular
   trend is interpreted. Full-angular orientation cases compute an
   independent Davidson ground state from each geometry's Hamiltonian rather
   than rotating an axial state.
2. An independent prolate-spheroidal weak-pulse calculation is compared with
   the public spherical `OneElectronDiatomicTDSE` workflow for clamped H2+ at
   `R=2`. The pulse has `E0=0.01`, `omega=2`, one cycle, length gauge,
   `dt=0.1`, and SIA propagation. The spherical calculation uses radial nodes
   `[0,1,2,4,8]`, seven points per element, `l_max=8`, and `L_max=16`; the
   prolate calculation uses `xi` boundaries `[1,2,4,8]`, seven points per
   element, eta degree nine, and `m=0`. The regression requires electronic
   ground-state energies within `2e-3` hartree, final norms within `2e-11` of
   unity, survival probabilities within `5e-8`, and full dipole histories
   within `2e-5` atomic units.
3. A full-angular H2+ alignment sweep uses polar angles `0`, `30`, `45`,
   `60`, `90`, `120`, `135`, `150`, and `180` degrees relative to the fixed
   laboratory-`z` field. The dynamics are physically different across
   inequivalent alignments. Reflection forces the `y` dipole and momentum to
   roundoff in the `x`-`z` plane, homonuclear mirrored-axis pairs transform
   consistently, and an independent `phi=60` degrees calculation verifies
   that azimuthal rotation preserves scalar and longitudinal observables
   while rotating transverse vectors.

These are compact implementation regressions. They do not establish
production convergence.

## Convergence responsibilities and present limits

Before interpreting a scientific observable, converge at least:

- radial extent, element boundaries, and points per element;
- `l_max` and the nuclear multipole cutoff;
- time step and Krylov or linear-solver tolerances;
- pulse duration, carrier parameters, and post-pulse propagation time;
- absorber onset, strength, and total box size;
- length--velocity agreement after radial and temporal errors are controlled;
- the particle mass and stationary initial state used by the calculation.

The H2+ ground state is slowly convergent in a midpoint-centered spherical
expansion. The small angular cutoffs in the regression suite test the
convergence mechanism; they are not recommendations. Scientific work should
begin around `l_max=12` and test at least `16` and `20`, extending farther
when the target observable requires it.

The full-angular implementation supports an arbitrary clamped two-centre
orientation with a laboratory-`z` field. The optimized fixed-`m` path remains
restricted to an axial molecule. General field polarization is not yet
supported. See [Rotation covariance of the spherical one-radial
representation](spherical_one_radial_rotation_covariance.md) for the
equivalence and truncation requirements. Direct-force acceleration sampling
for HHG is a later implementation step.
