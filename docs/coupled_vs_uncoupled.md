# Coupled and uncoupled two-radial angular representations

This document states when the coupled and uncoupled angular parametrizations
of a two-radial wavefunction are mathematically equivalent, how their finite
truncations must be matched, and what changes during laser-driven dynamics.
The discussion applies in particular to the non-Born--Oppenheimer treatment
of H2+, HD+, and D2+.

The distinction between the representations is a choice of angular basis. It
does not introduce a different physical Hamiltonian. Differences in computed
results should arise only from unequal finite spaces, unequal operator
truncations, different symmetry restrictions, or numerical errors.

## Wavefunction parametrizations

Let `l_1` denote the electronic orbital angular momentum and `l_2` the
internuclear rotational angular momentum. In the uncoupled representation,

```math
\Psi(\mathbf r,\mathbf R,t)
=
\frac{1}{rR}
\sum_{l_1m_1l_2m_2}
\psi_{l_1m_1l_2m_2}(r,R,t)
Y_{l_1m_1}(\hat{\mathbf r})
Y_{l_2m_2}(\hat{\mathbf R}).
```

For a selected total magnetic projection `M_tot`, only channels satisfying

```math
m_1+m_2=M_{\mathrm{tot}}
```

are retained.

The coupled representation uses bipolar spherical harmonics,

```math
\mathcal Y_{l_1l_2}^{LM}
(\hat{\mathbf r},\hat{\mathbf R})
=
\sum_{m_1m_2}
\langle l_1m_1,l_2m_2\mid LM\rangle
Y_{l_1m_1}(\hat{\mathbf r})
Y_{l_2m_2}(\hat{\mathbf R}),
```

and parametrizes the wavefunction as

```math
\Psi(\mathbf r,\mathbf R,t)
=
\frac{1}{rR}
\sum_{l_1l_2LM}
\psi_{l_1l_2LM}(r,R,t)
\mathcal Y_{l_1l_2}^{LM}
(\hat{\mathbf r},\hat{\mathbf R}).
```

The coupled channels obey

```math
|l_1-l_2|\le L\le l_1+l_2,
\qquad
M=M_{\mathrm{tot}}.
```

At fixed `M_tot`, the two coefficient vectors are related by a
time-independent Clebsch--Gordan transformation `U`:

```math
\boldsymbol\psi_{\mathrm{coupled}}(t)
=U^\dagger\boldsymbol\psi_{\mathrm{uncoupled}}(t),
\qquad
\boldsymbol\psi_{\mathrm{uncoupled}}(t)
=U\boldsymbol\psi_{\mathrm{coupled}}(t).
```

The coefficient arrays therefore should not be compared channel by channel
without first applying this transformation.

## Conditions for exact finite-space equivalence

The transformation is unitary only when the two calculations span the same
finite angular space. Matching `l1_max`, `l2_max`, and the radial grids is not
enough by itself. All of the following conditions are required.

1. Both representations use the same electronic cutoff `l1_max` and
   internuclear cutoff `l2_max`.
2. The uncoupled representation retains every `(m_1, m_2)` pair compatible
   with each retained `(l_1, l_2)` and the selected `M_tot`.
3. For every retained `(l_1, l_2)`, the coupled representation retains every
   allowed `L` compatible with `M_tot`.
4. Both representations use identical radial grids, masses, charges,
   boundary conditions, and radial operators.
5. Both use the same Coulomb multipole cutoff, external-field operators,
   CAPs, and other operator approximations.
6. Both impose equivalent inversion and nuclear-exchange restrictions.

The largest possible coupled angular momentum in the rectangular
`(l_1,l_2)` truncation is

```math
L_{\mathrm{complete}}=l_{1,\max}+l_{2,\max}.
```

Consequently,

```math
L_{\max}=l_{1,\max}+l_{2,\max}
```

is sufficient to avoid an additional total-angular-momentum truncation,
provided all allowed intermediate `L` values are actually retained. A smaller
`L_max` defines a projected subspace and is not generally equivalent to the
complete uncoupled `M_tot` sector.

When the finite spaces match, every represented operator must satisfy

```math
H_{\mathrm{coupled}}(t)
=U^\dagger H_{\mathrm{uncoupled}}(t)U.
```

This relation applies to the complete Hamiltonian and separately to the
one-body kinetic terms, electron--nuclear Coulomb interaction, field
interaction, and radial CAPs.

## Coulomb multipole truncation

The radial FEMDVR Coulomb multipoles are independent of the angular channel
parametrization and can be reused in both representations. Their sparse
angular coupling layers are related by the same Clebsch--Gordan
transformation.

For finite `l1_max` and `l2_max`, no Coulomb multipoles above

```math
\lambda_{\mathrm{complete}}
=2\min(l_{1,\max},l_{2,\max})
```

can couple two channels within the retained rectangular angular space. Using
this value includes the full Coulomb projection in that space. A lower
`lambda_max` is an additional approximation. Coupled and uncoupled results
can still agree with a lower value if exactly the same truncated Coulomb
operator is used in both calculations, but that agreement does not establish
convergence with respect to `lambda_max`.

The same center scales must be used in both representations. Odd multipoles
cancel for homonuclear H2+ and D2+ in nuclear-center-of-mass Jacobi
coordinates, whereas HD+ generally retains both even and odd multipoles.

## Field-free ground states

The field-free Hamiltonian is rotationally invariant and commutes with
`L^2` and `L_z`. The rovibronic ground states considered for H2+, HD+, and
D2+ have

```math
L=0,
\qquad
M=0,
\qquad
\langle L^2\rangle=0.
```

A coupled calculation restricted to `L_max=0` is therefore an appropriate
ground-state calculation in the known symmetry sector. For `L=0`, the
triangle rule requires `l_1=l_2`.

The full uncoupled `M_tot=0` space also contains states with `L>0`. Thus an
`L_max=0` coupled calculation is not a unitary reparametrization of that
entire uncoupled space; it is the `L=0` block of it. The two ground-state
energies are expected to agree when the lowest state of the uncoupled
calculation belongs entirely to this block, as it should for the systems and
symmetry sector above.

The strongest finite-space ground-state validation is therefore to compare
the coupled state with the explicit `L=0` projection of the uncoupled state.
Useful comparisons include:

- the ground-state energy;
- the Davidson residual norm;
- the wavefunction norm;
- radial probability densities;
- `L^2` and `L_z` expectation values;
- inversion parity; and
- nuclear-exchange parity for H2+ and D2+.

With matched spaces and solver tolerances, discrepancies should be limited by
floating-point error and the Davidson stopping tolerance. These comparisons
validate the representation change; they do not demonstrate convergence of
the radial or angular truncations.

### Fixed-grid Davidson validation

The coupled and uncoupled field-free implementations were compared on the
same fixed FEMDVR grids. The electronic coordinate used four 4-bohr elements,
with 11 points in every element,

```python
electronic_grid = FEMDVRGridSpec(
    nodes=[0.0, 4.0, 8.0, 12.0, 16.0],
    n_points=[11, 11, 11, 11],
)
```

and the internuclear coordinate used two 4-bohr elements, with 15 points in
every element,

```python
internuclear_grid = FEMDVRGridSpec(
    nodes=[0.0, 4.0, 8.0],
    n_points=[15, 15],
)
```

After removal of the two radial boundary points, these specifications give
39 electronic and 27 internuclear radial coefficients per angular channel.
The calculations used CODATA 2022 particle data, the FEMDVR Poisson Coulomb
construction, element-local radial Hamiltonian actions, and the one-body
spectral Davidson preconditioner. The Davidson residual tolerance was
`1.0e-8`.

The coupled calculations selected `L_max=0`, `M_tot=0`, and
`l1_max=l2_max=l_max`. The uncoupled calculations used `m_max=l_max` and
`M_tot=0`. At every cutoff, the complete Coulomb projection was retained by
setting `lambda_max=2*l_max` in the coupled calculation and
`lambda_max=mu_max=2*l_max` in the uncoupled calculation. Positive inversion
parity was selected for all systems. Positive spatial nuclear-exchange parity
was selected for H2+ and D2+; HD+ has no exchange restriction.

| `l_max` | System | Coupled energy (Eh) | Uncoupled energy (Eh) | Absolute difference (Eh) |
| ---: | :--- | ---: | ---: | ---: |
| 2 | H2+ | -0.576977161109 | -0.576977161109 | 2.0e-13 |
| 2 | HD+ | -0.571842318284 | -0.571842318284 | 2.5e-13 |
| 2 | D2+ | -0.578789888172 | -0.578789888172 | 2.3e-13 |
| 4 | H2+ | -0.588407314504678 | -0.588407314504696 | 1.8e-14 |
| 4 | HD+ | -0.588467969903139 | -0.588467969903120 | 1.9e-14 |
| 4 | D2+ | -0.590072023205223 | -0.590072023205240 | 1.7e-14 |
| 6 | H2+ | -0.591382793725416 | -0.591382793725429 | 1.3e-14 |
| 6 | HD+ | -0.593908932988414 | -0.593908932988385 | 2.9e-14 |
| 6 | D2+ | -0.592892162324956 | -0.592892162324955 | 1.2e-15 |

All 18 solves converged below the requested residual tolerance. The
uncoupled solutions had numerical `L^2` expectation values below `4e-17`,
confirming that they remained in the same `L=0` block as
the coupled calculations. The agreement therefore validates the coupled
Hamiltonian and Davidson path on these fixed spaces. The changes in energy
with `l_max` are substantial, so this table must not be interpreted as an
angular- or radial-convergence result.

## Time-dependent propagation

Because `U` is time independent, equivalent finite-space Hamiltonians have
equivalent propagators:

```math
\mathcal U_{\mathrm{coupled}}(t,t_0)
=
U^\dagger
\mathcal U_{\mathrm{uncoupled}}(t,t_0)
U.
```

An initial state transformed consistently between the two bases must
therefore produce the same physical wavefunction, norm loss, expectation
values, radial densities, and final-state probabilities, up to the numerical
errors of the propagator.

This finite-space equivalence applies in velocity gauge as well as length
gauge. The coupled and uncoupled momentum actions must use the same reduced
velocity couplings, radial first-derivative convention, angular momentum
coefficients, and treatment of the spatially uniform quadratic
vector-potential term. The implementation omits that `A(t)^2` scalar term in
both representations because it changes only the global phase. A field-free
initial state is used directly only when the supplied vector potential
satisfies `A(t_0)=0`; otherwise an explicit gauge transformation of the
initial state would be required.

Short SIA trajectories on deliberately small, complete matched spaces have
been compared for H2+, HD+, and D2+. After applying the exact
Clebsch--Gordan transformation, the coupled and uncoupled velocity-gauge
states and sampled observables agree to the propagation tolerance. This is a
representation-equivalence test, not a claim that the finite radial or
angular spaces are physically converged and not a substitute for a
length-versus-velocity gauge convergence study.

The comparison used the same electronic and internuclear FEMDVR
specification, `FEMDVRGridSpec([0, 1, 2], [3, 3])`, with
`l1_max=l2_max=1`, `L_max=2`, `M_tot=0`, and `lambda_max=2`. The radial
Coulomb construction was analytic and radial differential actions were
element-local. A normalized arbitrary state was propagated from `t=0` to
`t=0.006` with SIA and `dt=0.002`, using the compatible vector potential

```math
A(t)=0.02\sin^2(\pi t/0.006),
```

which has `A(0)=0`. Using an arbitrary state rather than only the ground
state exercises every retained angular channel. The phase-aligned final
states and all sampled representation-independent observables agree within
`3e-10`; the underlying velocity-operator actions agree within `8e-13`.

The fact that the initial ground state has `L=0` does not justify retaining
only `L=0` during a laser pulse. The electric-dipole interaction is a
rank-one spherical tensor. It couples total angular momenta satisfying

```math
|L-L'|\le 1,
```

with the `L=0` to `L'=0` transition excluded. Starting from `L=0`, the first
dipole interaction therefore populates `L=1`; repeated interactions can
populate progressively higher `L` sectors. A coupled calculation with an
insufficient `L_max` and a complete uncoupled calculation are different
projected dynamics problems.

For a field polarized along the laboratory `z` axis, only the `q=0`
spherical component is present and

```math
\Delta M=0.
```

An initial `M_tot=0` state can then remain in a single `M_tot=0` block.
Circular, elliptical, or non-axis-aligned fields contain `q=+1` and/or
`q=-1` components and generally require multiple `M` sectors.

The electric-dipole interaction is odd under spatial inversion, so both
inversion parities must be available during field-driven dynamics. For the
homonuclear ions H2+ and D2+, exchange of the identical nuclei remains a
valid symmetry and the selected nuclear-exchange sector can be retained.
HD+ has no identical-nucleus exchange symmetry.

## Consequences of unequal truncations

Several comparisons can look plausible without testing equivalent spaces:

- Equal radial grids and equal `l1_max`, `l2_max` do not guarantee
  equivalence if `L_max` removes allowed coupled channels.
- Equal `L_max` values do not imply equal spaces when the underlying
  `l_1,l_2` channel sets differ.
- Restricting the coupled calculation to `L=0` is valid for the field-free
  ground state but not for dipole-driven dynamics.
- Restricting the uncoupled calculation to too small an `m_1` or `m_2` range
  prevents the Clebsch--Gordan transformation from being unitary.
- Different inversion or exchange sectors compare different physical
  subspaces.
- Different `lambda_max` values compare different Coulomb projections.

For Hermitian field-free calculations, restricting the coupled space further
usually gives a variational energy no lower than the ground-state energy in
the larger matching space. Equality indicates that the omitted channels have
no component in the ground state; it does not establish equivalence for
excited states or time-dependent dynamics.

## Recommended equivalence tests

Representation validation should use deliberately small fixed radial and
angular spaces. It should not be combined with a physical convergence study.
For each supported isotope and symmetry sector:

1. construct matching coupled and uncoupled finite spaces;
2. verify that both channel counts equal the rank of the
   Clebsch--Gordan transformation;
3. check `U^dagger U` and `U U^dagger` on the represented spaces;
4. compare individual Hamiltonian component actions after transformation;
5. compare complete field-free Hamiltonian actions;
6. compare the `L=0` Davidson ground state with the projected uncoupled
   result;
7. compare axial length-gauge dipole and velocity-gauge momentum actions;
8. propagate short length- and velocity-gauge pulses in both representations
   and compare transformed
   states and representation-independent observables; and
9. repeat the propagation with a deliberately incomplete `L_max` to confirm
   that the validation detects the resulting projection error.

Production convergence with respect to the radial grids, `l1_max`,
`l2_max`, `L_max`, and `lambda_max` is a separate activity. Agreement between
the two parametrizations is an implementation check, not evidence that either
finite calculation is physically converged.
