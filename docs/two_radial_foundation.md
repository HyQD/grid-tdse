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

## Angular specifications

`AngularBasisSpec` defines the coordinate quantum numbers $l$ and $m$.
Setting `coordinate_parity=1` retains even $l$, while `coordinate_parity=-1`
retains odd $l$. Leaving it unset retains both parities.

`CoulombMultipoleSpec` independently defines the expansion quantum numbers
$\lambda$ and $\mu$. These are Coulomb multipoles and are not the total
angular and magnetic quantum numbers of the complete system.

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
).build()

hamiltonian = system.hamiltonian()
```

The atomic and molecular-ion interfaces own physical validation, particle
parameters, symmetry defaults, initial states, and imaginary-time
ground-state propagation. This foundation owns only the reusable
discretization and Hamiltonian action.
