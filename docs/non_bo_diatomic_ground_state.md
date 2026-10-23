# Non-Born--Oppenheimer diatomic-ion ground states

`NonBornOppenheimerDiatomicIonGroundState` constructs and solves the
field-free three-body Hamiltonian for H2+, HD+, and D2+ without making the
Born--Oppenheimer approximation. Particle masses and charges are obtained
from `matter-field-params` for a selected CODATA release.

The first implementation uses FEM-DVR for both radial coordinates and the
representation-independent imaginary-time solver in `grid-tdse`.

## Coordinates and Hamiltonian

The vector `R` joins the two positive particles. The electronic vector `r`
is measured from their center of mass. With positive-particle masses `M_a`
and `M_b`, their positions relative to that center are

$$
\mathbf R_a=-\frac{M_b}{M_a+M_b}\mathbf R,
\qquad
\mathbf R_b=+\frac{M_a}{M_a+M_b}\mathbf R.
$$

The internal Hamiltonian is

$$
H=-\frac{1}{2\mu_e}\nabla_r^2
  -\frac{1}{2\mu_R}\nabla_R^2
  +\frac{q_aq_b}{R}
  +\frac{q_eq_a}{|\mathbf r-\mathbf R_a|}
  +\frac{q_eq_b}{|\mathbf r-\mathbf R_b|}.
$$

The electron--particle attractions are translated into the signed
`CoulombTerm` radial scales used by the two-radial foundation. FEM-DVR radial
multipoles are evaluated by the two-grid Poisson implementation in
`grid-lib`.

## Symmetry inputs

The native interface uses numerical physical parities rather than compact
`sp`, `ap`, `so`, or `ao` labels:

```python
inversion_parity=+1
pair_exchange_parity=+1
symmetry_adapted_basis=True
```

Total inversion and identical-pair exchange act as

$$
I=(-1)^{l_e+l_R},
\qquad
P_{12}=(-1)^{l_R}.
$$

For H2+ and D2+, `pair_exchange_parity` is required and must be `+1` or
`-1`. When `symmetry_adapted_basis=True`, the corresponding coordinate
parities are removed from the angular bases. Setting it to `False` retains
the complete coordinate bases while the initial state is still projected
onto the requested physical sector.

For HD+, the proton and deuteron are distinguishable. Its
`pair_exchange_parity` must therefore be `None`; supplying `+1` or `-1`
raises an error. Both coordinate parities are retained because odd Coulomb
multipoles couple even-even and odd-odd channels within a fixed total
inversion sector.

The `lambda_max` and `mu_max` arguments truncate the Coulomb expansion. They
are separate from the coordinate `l_max`, `m_max` and from total-system
angular momentum.

## Initial state

`initial_state="product"` constructs a deterministic product of the lowest
one-coordinate radial eigenstates in the lowest angular channels permitted
by the requested parities. The resulting state is normalized with the
two-radial FEM-DVR quadrature.

For symmetric inversion and antisymmetric pair exchange, the lowest angular
channels have `l_e=l_R=1`. The initial state is the pure
`L_total=0`, `M_total=0` Clebsch--Gordan combination

$$
\frac{1}{\sqrt{3}}
\left(
|1,-1;1,1\rangle
-|1,0;1,0\rangle
+|1,1;1,-1\rangle
\right).
$$

It is not initialized as the mixed-total-angular-momentum product
`|1,0;1,0>`. Field-free imaginary-time propagation therefore remains in the
selected total-angular-momentum sector.

A NumPy array with the representation's `psi_shape` may be supplied instead.
The interface projects it onto the requested inversion and pair-exchange
sector, rejects a zero projected state, and normalizes the result.

## Solver and result

The calculation uses matrix-free Crank--Nicolson imaginary-time steps. By
default, the Krylov solve is preconditioned with the inverse of

$$
I+\frac{\Delta\tau}{2}(h_e+h_R),
$$

where the two one-coordinate Hamiltonians are diagonalized in their
quadrature-weighted Hermitian representations. The two-coordinate Coulomb
interaction is not included in the preconditioner.

The current energy, energy change, phase-invariant state change, and
eigenstate residual are printed every ten steps by default. Set
`progress_interval` to another positive integer to change the reporting
frequency, or to `None` to disable progress output.

`DiatomicIonGroundStateResult` contains the particle parameters, prepared
two-radial representation, normalized initial state, complete ground-state
iteration result, final symmetry expectations, and initial and final values
of `<L_total^2>` and `<L_total,z>`.

See
[`examples/non_bo_diatomic_ground_state.py`](../examples/non_bo_diatomic_ground_state.py)
for a complete calculation with every grid, angular, symmetry, and solver
option shown explicitly. Radial-grid, angular, Coulomb-multipole, and
imaginary-time-step convergence must all be checked before treating an
energy as converged.
