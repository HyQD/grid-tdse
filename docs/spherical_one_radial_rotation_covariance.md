# Rotational covariance of spherical one-radial diatomic calculations

## Scope

This note describes when a clamped-nuclei, one-electron diatomic calculation
in a finite single-centre spherical-harmonic basis is independent of the
orientation of the molecular axis. The primary example is H2+, but the
rotation argument applies to any two-centre potential that is rigidly rotated
about the spherical-expansion origin.

The result concerns the field-free Hamiltonian. A molecule rotated relative
to a fixed laboratory electric field represents a physically different
laser--molecule alignment and does not, in general, have equivalent dynamics.

## Finite angular space

Write the one-electron wavefunction as

```math
\Psi(\mathbf r)
=
\frac{1}{r}
\sum_{l=0}^{l_{\max}}
\sum_{m=-l}^{l}
u_{lm}(r)Y_{lm}(\hat{\mathbf r}).
```

The complete angular space at cutoff `l_max` is

```math
\mathcal H_{l_{\max}}
=
\bigoplus_{l=0}^{l_{\max}}
\mathrm{span}
\{Y_{lm}:-l\le m\le l\}.
```

A spatial rotation mixes magnetic components belonging to the same `l`,

```math
U(\mathcal R)Y_{lm}
=
\sum_{m'=-l}^{l}
D_{m'm}^{l}(\mathcal R)Y_{lm'},
```

but does not mix different `l` values. The finite space is therefore closed
under rotations only when every `m=-l,...,l` is retained for every included
`l`.

Let `P_lmax` project onto this complete finite space. It commutes with a
rotation:

```math
[P_{l_{\max}},U(\mathcal R)]=0.
```

If `H_z` is the field-free Hamiltonian for a molecule on the `z` axis and the
nuclei are rigidly rotated by `R`, then

```math
H_{\mathcal R}
=
U(\mathcal R)H_zU(\mathcal R)^\dagger.
```

The projected finite-basis Hamiltonians consequently satisfy

```math
P_{l_{\max}}H_{\mathcal R}P_{l_{\max}}
=
U(\mathcal R)
P_{l_{\max}}H_zP_{l_{\max}}
U(\mathcal R)^\dagger.
```

They are unitarily equivalent and must have the same eigenvalues, apart from
floating-point and numerical assembly errors.

## Required Coulomb multipoles

Expand the two-centre nuclear potential about the same spherical origin:

```math
V(\mathbf r)
=
\sum_{L=0}^{\infty}
\sum_{M=-L}^{L}
V_{LM}(r)Y_{LM}(\hat{\mathbf r}).
```

Its angular matrix elements contain Gaunt coefficients,

```math
\langle lm|V|l'm'\rangle
=
\sum_{LM}
V_{LM}(r)
\int
Y_{lm}^{*}
Y_{LM}
Y_{l'm'}
\,d\Omega.
```

The triangle condition is

```math
|l-l'|\le L\le l+l'.
```

For `l,l' <= l_max`, no matrix element in the finite orbital space can depend
on a multipole with

```math
L>2l_{\max}.
```

The complete projection of the nuclear Coulomb potential therefore requires

```text
L = 0, ..., 2*l_max
M = -L, ..., L.
```

Multipoles beyond `2*l_max` are present in the exact point-charge potential,
but have exactly zero angular matrix elements within this finite wavefunction
space. Truncating the potential at `2*l_max` does not introduce an additional
angular approximation beyond the selected orbital cutoff.

For homonuclear nuclei at opposite positions about the origin, inversion
symmetry cancels all odd-`L` nuclear multipoles for every orientation. An
implementation may exploit this cancellation, although retaining the zero
odd multipoles is harmless.

## Expected energy equivalence

Consider two H2+ geometries with identical bond length and radial grid:

1. nuclei on the laboratory `z` axis;
2. the same nuclei rigidly rotated about the spherical origin.

At a fixed `l_max`, their field-free energies should agree to numerical
precision when:

- all `m=-l,...,l` are retained for every `l`;
- `nuclear_L_max = 2*l_max`;
- all `M=-L,...,L` are retained for every nuclear multipole;
- the radial grid, radial boundary, masses, charges, and bond length are
  identical;
- the molecular midpoint remains at the spherical origin;
- both Hamiltonians use the same numerical Coulomb construction.

The complete on-axis full-`m` spectrum and the complete off-axis full-`m`
spectrum should be identical. The on-axis H2+ ground state lies in the
body-axis `m=0` sector, so its energy should also equal the lowest eigenvalue
of the off-axis full-`m` Hamiltonian.

## Comparing excited states

Care is required when comparing restricted and complete spectra. For the
on-axis geometry, axial symmetry separates the Hamiltonian into independent
`m` blocks. After rotating the molecule, the corresponding body-axis magnetic
quantum number remains meaningful, but its eigenstates are superpositions of
laboratory-frame `m` components.

Consequently:

- every on-axis body-axis `m=0` eigenvalue must occur in the off-axis full-`m`
  spectrum;
- the off-axis full-`m` spectrum also contains the rotated on-axis
  `m != 0` sectors;
- comparing the sorted on-axis fixed-`m=0` spectrum directly with the sorted
  off-axis full-`m` spectrum can pair different physical states.

The strongest spectral comparison is full-`m` against full-`m`. Individual
states may instead be matched by energy, symmetry, overlaps after applying
the rotation matrix, or other physical observables.

## Rotation is not translation

The finite single-centre expansion is rotation covariant because a rotation
does not mix different `l` spaces. It is not translation invariant at finite
`l_max`.

If the molecular midpoint is displaced relative to the spherical origin, the
translated wavefunction and potential generally require an unbounded range of
angular momenta about the original origin. Two translated calculations may
therefore have different finite-`l_max` errors even though the exact physical
energy is translation invariant.

For orientation tests, rotate the two nuclei rigidly while keeping their
midpoint at the expansion origin.

## Consequences for laser-driven dynamics

Field-free rotational equivalence does not imply that differently aligned
molecules have identical dynamics in a fixed laboratory field. With an
electric field polarized along laboratory `z`:

- an on-axis molecule experiences parallel polarization;
- a tilted molecule experiences a mixture of parallel and perpendicular
  polarization relative to its molecular axis;
- transition strengths, ionization, excitation, and high-harmonic spectra may
  differ.

The two dynamics calculations become rotationally equivalent only if both the
molecule and the external field are rotated together. This gives another
useful implementation test: simultaneous rotation of the field and molecular
geometry should leave physical observables unchanged when the full required
angular space is retained.

Although a laboratory-`z` dipole interaction preserves laboratory `m`, an
off-axis field-free nuclear potential contains `M != 0` multipoles and mixes
laboratory `m` channels. Off-axis propagation must therefore use the complete
angular state rather than one fixed-`m` block.

## Current implementation limitation

The current
`grid_tdse.spherical.one_radial.OneElectronDiatomicHamiltonian` defaults to
`m_max = l_max` and `nuclear_L_max = 2*l_max`, but explicitly sets

```python
self.nuclear_M_max = 0
```

because the current `OneElectronDiatomicSystem` places both nuclei on the
`z` axis. The present public Hamiltonian therefore represents only an axial
nuclear potential even though the underlying `grid-lib` multipole machinery
can construct nonzero-`M` components.

The future arbitrary-orientation extension must:

- accept explicit Cartesian nuclear positions or an equivalent orientation;
- expose the complete required nuclear `M` range;
- build and diagonalize the full `(l,m)` Hamiltonian;
- preserve a common spherical origin and radial grid;
- retain orientation information in archives and checkpoint compatibility
  metadata.

For `m_max = l_max` and `nuclear_L_max = 2*l_max`, allowing the multipole
constructor to choose its complete default magnetic range gives
`nuclear_M_max = 2*l_max`. Passing `M_max = 0` must remain only an axial
optimization.

## Recommended validation tests

The arbitrary-orientation implementation should include the following gates.

### Field-free spectrum

For several `l_max` values:

1. construct the on-axis full-`m` Hamiltonian;
2. rigidly rotate the nuclei to one or more angles in the `x`-`z` plane;
3. construct the off-axis full-`m` Hamiltonian with
   `L_max = M_max = 2*l_max`;
4. compare their complete low-energy spectra;
5. verify that discrepancies are consistent with floating-point assembly and
   eigensolver tolerances.

### Explicit rotation covariance

Build the block-diagonal angular rotation

```math
U_{l_{\max}}(\mathcal R)
=
\bigoplus_{l=0}^{l_{\max}}D^l(\mathcal R)
```

and extend it identically over every radial point. Verify

```math
H_{\mathrm{rotated}}
\simeq
U_{l_{\max}}(\mathcal R)
H_z
U_{l_{\max}}(\mathcal R)^\dagger.
```

This test exercises nuclear geometry, multipole coefficients, spherical-
harmonic phases, angular ordering, Gaunt couplings, and the full-`M` potential
action simultaneously.

### Rotated states and observables

Rotate selected on-axis eigenvectors and verify that they are eigenvectors of
the off-axis Hamiltonian with the same energies. Compare rotationally
invariant observables directly and rotate vector or tensor observables before
comparison.

### Simultaneously rotated field and molecule

After general field directions are supported, rotate both the molecular
geometry and the field. Propagated norms, populations, and consistently
rotated observables should agree. Rotating only the molecule is a physical
alignment study and is not an equivalence test.

## Summary

At fixed `l_max`, an on-axis and rigidly rotated field-free H2+ Hamiltonian
have the same finite-basis energies when the wavefunction retains every
`m=-l,...,l` and the nuclear potential retains every
`L=0,...,2*l_max`, `M=-L,...,L`. This is an exact rotational-covariance
property of the complete finite spherical-harmonic space, not merely a
large-`l_max` expectation.

Any orientation dependence of the field-free energies under these conditions
indicates an incomplete magnetic space, an inconsistent multipole truncation,
a geometry or origin mismatch, or a numerical implementation defect.
