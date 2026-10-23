# Two-electron atomic ground states

`TwoElectronAtomGroundState` constructs and solves field-free He and Be2+ in
the infinite-nuclear-mass approximation. Both electrons are represented
directly; this is not a Hartree--Fock or independent-particle model. The first
implementation uses the same FEM-DVR specification for both electron radial
coordinates and the representation-independent imaginary-time solver in
`grid-tdse`.

## Hamiltonian and representation

For nuclear charge `Z`, the atomic-unit Hamiltonian is

$$
H = \sum_{i=1}^{2}\left(-\frac{1}{2}\nabla_i^2-\frac{Z}{r_i}\right)
  + \frac{1}{|\mathbf r_1-\mathbf r_2|}.
$$

The fixed nucleus has no coordinate or kinetic-energy term. The interface
currently provides the following presets:

| System | `Z` | Net ionic charge |
| --- | ---: | ---: |
| `"He"` | 2 | 0 |
| `"Be2+"` | 4 | 2 |

The nuclear charges are exact defining data owned by these atomic presets.
`matter-field-params` is not consulted because finite nuclear masses are not
part of this approximation.

The state is stored as `psi[a, b, i, j]`, where `a` and `b` enumerate the
uncoupled electron angular states `(l1, m1)` and `(l2, m2)`, and `i`, `j`
enumerate their FEM-DVR radial points. Electron exchange therefore swaps both
the angular and radial axes:

$$
(P_{12}\Psi)_{ab}(r_1,r_2)=\Psi_{ba}(r_2,r_1).
$$

The full coordinate angular basis is retained. Reducing each coordinate to
only even or only odd `l` would omit valid correlated components such as an
odd-odd contribution to an even-inversion state.

## Symmetry inputs

The native inputs are explicit physical parities:

```python
inversion_parity=+1
electron_exchange_parity=+1
```

Total inversion acts on an angular product as

$$
I=(-1)^{l_1+l_2}.
$$

For two electrons, `electron_exchange_parity=+1` selects a symmetric spatial
state and hence the spin-singlet sector. A value of `-1` selects an
antisymmetric spatial state and hence the spin-triplet sector. Spin is not an
explicit coordinate; `spin_multiplicity` reports the multiplicity implied by
the spatial exchange symmetry.

The rotationally invariant interface requires `m_max=l_max`. The
`lambda_max` and `mu_max` arguments independently truncate the
electron--electron Coulomb expansion. They must not be confused with total
system angular momentum.

## Initial state

`initial_state="product"` constructs a deterministic state in the requested
sector:

- even inversion, symmetric exchange: `1s(1) 1s(2)`;
- even inversion, antisymmetric exchange: the antisymmetrized `1s 2s`
  product;
- odd inversion: the symmetric or antisymmetric combination of `1s 1p` and
  `1p 1s`.

These choices have `M_total=0`. The even guesses have `L_total=0`; the odd
guesses have `L_total=1`.

A NumPy array with the representation's `psi_shape` may be supplied instead.
The interface projects it onto the requested total-inversion and
electron-exchange sector, rejects a zero projected state, and normalizes it
with the two-radial quadrature.

## Solver and result

The calculation uses matrix-free Crank--Nicolson imaginary-time steps. By
default, the Krylov solve is preconditioned with the inverse of

$$
I+\frac{\Delta\tau}{2}(h_1+h_2),
$$

where the one-electron Hamiltonians are diagonalized in their
quadrature-weighted Hermitian representations. Electron--electron repulsion
is omitted from the preconditioner but retained in every Hamiltonian action.

The current energy, energy change, phase-invariant state change, and
eigenstate residual are printed every ten steps by default. Set
`progress_interval` to another positive integer to change the reporting
frequency, or to `None` to disable progress output.

`TwoElectronAtomGroundStateResult` contains the atomic parameters, prepared
two-radial representation, normalized initial state, complete ground-state
iteration result, spin multiplicity, symmetry expectations, and initial and
final values of `<L_total^2>` and `<L_total,z>`.

See
[`examples/two_electron_atom_ground_state.py`](../examples/two_electron_atom_ground_state.py)
for a complete calculation with every grid, angular, symmetry, and solver
option shown explicitly. Radial-grid, angular, Coulomb-multipole, and
imaginary-time-step convergence must all be checked before treating an
energy as converged.
