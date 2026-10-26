# Two-radial spherical foundation

The `grid_tdse.spherical.two_radial` package prepares Hamiltonians for
wavefunctions represented as

$$
\Psi_{ij}(r_1,r_2),
$$

where $i$ and $j$ enumerate explicit, uncoupled spherical-harmonic states
$(l_1,m_1)$ and $(l_2,m_2)$. It is the shared numerical foundation for the
two-electron atomic and non-Born--Oppenheimer diatomic interfaces. A
ground-state iteration driver is not part of this layer.

## Radial grids

Two grid families are supported:

- `FEMDVRGridSpec(nodes, n_points)` constructs a finite-element DVR and uses
  its interior points. Coulomb radial multipoles are obtained from the
  two-grid Poisson implementation in `grid-lib`.
- `RadialSincDVRGridSpec(r_max, n_points)` constructs the parity-adapted radial
  sinc DVR from `grid-lib`. Its derivative matrix depends on $l$, and Coulomb
  radial multipoles are evaluated analytically.

Both coordinates must use the same grid family. Their extents and point
counts may differ.

### FEM-DVR radial operator actions

`TwoRadialSystem.hamiltonian()` accepts
`radial_operator_mode="global"` or `"element-local"`. The global mode uses the
existing dense radial derivative matrices and remains the default reference
path. The element-local mode uses the finite-element maps and local stiffness
actions supplied by `grid-lib`, inserts the omitted homogeneous Dirichlet
boundary values, and scatter-adds the element contributions without applying
a dense global radial matrix.

The centrifugal and other one-coordinate potentials remain diagonal in the
radial DVR. Both modes use the same optimized Coulomb action and produce the
same physical Hamiltonian. In velocity gauge, the same selection also applies
the radial first derivative in the dipole interaction element by element.
Length-gauge multiplication and the Coulomb action are already radial-local.
Element-local mode currently requires FEM-DVR; radial sinc-DVR calculations
must use global mode.

## Angular specifications

`AngularBasisSpec` defines the coordinate quantum numbers $l$ and $m$.
Setting `coordinate_parity=1` retains even $l$, while `coordinate_parity=-1`
retains odd $l$. Leaving it unset retains both parities.

`CoulombMultipoleSpec` independently defines the expansion quantum numbers
$\lambda$ and $\mu$. These are Coulomb multipoles and are not the total
angular and magnetic quantum numbers of the complete system.

### Conserved total magnetic quantum number

In the uncoupled angular basis, every product function
$Y_{l_1m_1}(\Omega_1)Y_{l_2m_2}(\Omega_2)$ is an eigenfunction of the total
orbital $z$-projection,

$$
L_{z,\mathrm{tot}}=L_{z,1}+L_{z,2},
$$

with eigenvalue $M_{\mathrm{tot}}=m_1+m_2$ in units where $\hbar=1$. If the
Hamiltonian commutes with $L_{z,\mathrm{tot}}$, angular products with different
values of $M_{\mathrm{tot}}$ do not couple. A state initialized in one such
sector consequently remains in it, so angular blocks belonging to all other
sectors need neither be stored nor evaluated.

The field-free atomic and non-Born--Oppenheimer diatomic Hamiltonians used
here are rotational scalars and therefore conserve both total orbital angular
momentum and its magnetic projection. The conservation of $M_{\mathrm{tot}}$
can also be seen directly from the implemented selection rules:

- A central one-coordinate kinetic or potential operator is diagonal in $m_i$
  and cannot change $m_1+m_2$.
- In a Coulomb multipole $(\lambda,\mu)$, the first coordinate changes its
  magnetic quantum number by the opposite amount to the second coordinate.
  Thus every nonzero coupling satisfies $m_1^{\mathrm{in}}+m_2^{\mathrm{in}}=m_1^{\mathrm{out}}+m_2^{\mathrm{out}}$ and preserves the sum of the two magnetic quantum numbers.
- The rank-one spherical dipole tensor has three components labeled by
  $\tilde{m}\in\{-1,0,+1\}$. A spatially uniform field polarized along the
  laboratory $z$ axis selects $\tilde{m}=0$. It therefore has $\Delta m=0$ and
  preserves the same `M_tot` sector in both length and velocity gauge.

This reduction is not a restriction to a particular value of total orbital
angular momentum. For example, an `M_tot=0` basis can contain contributions
from $L_{\mathrm{tot}}=0,1,2,\ldots$. It is also independent of inversion and,
when defined, identical-pair exchange parity; those symmetries may be imposed
in addition. Spin is not part of the present spatial representation, so
`M_tot` refers only to orbital angular momentum.

Set `M_tot` on `TwoRadialSystemSpec` to retain only angular pairs satisfying
$m_1+m_2=M_{\mathrm{tot}}$. The high-level atomic and non-Born--Oppenheimer
diatomic interfaces expose the same option. `M_tot=None`, which is the
default, keeps the full angular-product representation for legacy use and
comparisons. A fixed sector must not be used when the complete Hamiltonian
breaks axial symmetry. Examples include an $x$- or $y$-polarized field, a
non-axisymmetric external potential, and any interaction containing spherical
dipole components with $\tilde{m}\ne 0$ without retaining all sectors to which
they couple.

#### Representation in the code

The compact representation is assembled and used as follows:

1. `AngularBasisSpec` enumerates the one-coordinate states $(l_i,m_i)$.
2. `AngularPairBasis(..., M_tot=M)` enumerates only index pairs `(i, j)` for
   which `basis1.states[i][1] + basis2.states[j][1] == M` and assigns each pair
   a compact index `p`.
3. `TwoRadialSystemSpec.build()` passes this table to the Coulomb-coupling
   builder. Both the input and output angular pairs of every stored coupling
   must occur in the retained table. The Coulomb selection rules guarantee
   that this is a closed Hamiltonian block when $M_{\mathrm{tot}}$ is
   conserved.
4. A full state has shape `(n1, n2, nr1, nr2)`. A fixed-sector state instead
   has shape `(n_pairs, nr1, nr2)`, where the first axis follows
   `system.angular_pairs.pairs`.
5. The one-body action selects the appropriate coordinate blocks for each
   compact pair. The optimized Coulomb kernel uses compact input and output
   pair indices, so it does not temporarily reconstruct the full angular
   product.
6. The axial dipole action constructs only transitions whose output pair is
   retained. Because its angular action has $\Delta m_i=0$, these transitions
   stay in the same compact sector.

Schematically, the mapping is

```python
pairs = tuple(
    (i, j)
    for i, (_, m1) in enumerate(basis1.states)
    for j, (_, m2) in enumerate(basis2.states)
    if m1 + m2 == M_tot
)

# Full storage:    psi[i, j, r1, r2]
# Compact storage: psi[p, r1, r2], where pairs[p] == (i, j)
```

`TwoRadialSystem.pack_state()` extracts a fixed sector from a full
angular-product array, while `unpack_state()` restores the full shape and
fills absent sectors with zeros. State archives record `M_tot` as part of the
representation metadata, preventing an exact reload into an incompatible
storage sector. The full and compact Hamiltonian actions are tested against
one another, and `M_tot=None` remains useful as a reference implementation.

The initial state must have a nonzero component in the selected sector. The
built-in product guesses belong to `M_tot=0`; selecting another sector
therefore requires a custom initial state. When a full custom array is
provided to a compact high-level calculation, `pack_state()` projects it by
retaining the requested angular blocks, after which the high-level interface
applies its other symmetry projections and normalizes the state.

For example, the following table assumes that both coordinates use complete,
identical angular bases with `m_max=l_max` and no coordinate-parity
reduction. It compares the full product with the `M_tot=0` sector:

| `l_max` | Full angular pairs | `M_tot=0` pairs |
| ---: | ---: | ---: |
| 0 | 1 | 1 |
| 2 | 81 | 19 |
| 4 | 625 | 85 |
| 6 | 2,401 | 231 |
| 8 | 6,561 | 489 |
| 10 | 14,641 | 891 |
| 12 | 28,561 | 1,469 |
| 14 | 50,625 | 2,255 |
| 16 | 83,521 | 3,281 |

The full count is $(l_{\max}+1)^4$. For `M_tot=0`, the number of retained
pairs is $(l_{\max}+1)^2+l_{\max}(l_{\max}+1)(2l_{\max}+1)/3$. At
`l_max=16`, compact storage therefore uses about 25.5 times fewer angular
blocks. Symmetry-adapted coordinate bases have different counts because some
$l$ values have already been removed.

## Coulomb action

The setup evaluates scalar Gaunt coefficients and stores only nonzero angular
coupling pairs. It never constructs the full dense `y` and `y_bar` tensors.
The production Hamiltonian uses the optimized output-blocked Coulomb action;
the old dense contraction is retained only conceptually as a test oracle and
is not exposed as a runtime option.

## Preparing a system

```python
from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    RadialSincDVRGridSpec,
    TwoRadialSystemSpec,
    radial_potential,
)

electron = CoordinateSpec(
    grid=RadialSincDVRGridSpec(r_max=30.0, n_points=200),
    basis=AngularBasisSpec(l_max=2, m_max=2),
    kinetic_prefactor=-0.5,
    one_body_terms=(radial_potential(lambda r: -2.0 / r),),
)

system = TwoRadialSystemSpec(
    coordinate1=electron,
    coordinate2=electron,
    multipoles=CoulombMultipoleSpec(lambda_max=4, mu_max=4),
    coulomb_terms=(CoulombTerm(radial_scale=1.0, charge_product=1.0),),
    M_tot=0,
).build()

hamiltonian = system.hamiltonian(radial_operator_mode="global")
```

The atomic and molecular-ion interfaces own physical validation, particle
parameters, symmetry defaults, initial states, and imaginary-time
ground-state propagation. This foundation owns only the reusable
discretization and Hamiltonian action.
