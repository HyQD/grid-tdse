# Axial two-radial dipole interactions

`grid-tdse` provides matrix-free length- and velocity-gauge dipole actions for
two-radial FEM-DVR states. The initial implementation is restricted to a
spatially uniform field polarized along the laboratory `z` axis.

For coordinate `i`, the length-gauge action is

$$
V_i^{(L)}(t)\Psi=-\xi_i E_z(t)z_i\Psi,
$$

and the velocity-gauge action is

$$
V_i^{(V)}(t)\Psi=-\nu_i A_z(t)p_{z,i}\Psi,
\qquad
\nu_i=\frac{\xi_i}{\mu_i}.
$$

The spatially uniform term quadratic in the vector potential is omitted. It
is a time-dependent scalar and therefore changes only the global phase.

## Spatial representation

`AxialCoordinateDipole` applies one coordinate's field-independent operator to
a state ordered as

```text
(angular1, angular2, radial1, radial2).
```

The velocity-gauge implementation reuses `MomentumOperators` from `grid-lib`,
including both the reduced-radial derivative and angular derivative terms in
`p_z`. No dense operator over the complete two-radial state is constructed.

The `z`-polarized interaction conserves `m` and couples only angular channels
with `l` differing by one. A field-coupled coordinate must therefore contain
both even- and odd-`l` channels. A coordinate whose physical coupling is
exactly zero is omitted and may remain parity reduced. In particular, the
internuclear coordinate of a homonuclear molecular ion may retain its pair-
exchange parity reduction.

Only FEM-DVR is supported initially. Radial sinc-DVR requires a separate
velocity implementation because its first-derivative matrix depends on `l`.

## Temporal fields

`TwoRadialAxialDipoleInteraction` accepts one scalar callable or an iterable
of scalar callables. Multiple profiles are summed. A profile from
`electromagnetic-fields` returns `E_z(t)` when configured for length gauge and
`A_z(t)` when configured for velocity gauge. Its declared gauge is checked
against the interaction gauge.

```python
from electromagnetic_fields.dipole_fields import SineSquareLaser
from grid_tdse.spherical.two_radial import (
    TwoRadialAxialDipoleInteraction,
)

pulse = SineSquareLaser(
    E0=0.05,
    omega=0.057,
    ncycles=4,
    gauge="length",
)

interaction = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
    representation,
    parameters,
    gauge="length",
    fields=pulse,
)

field_action = interaction(state, time)
```

This object supplies only the field-interaction action. Combining it with the
field-free Hamiltonian and advancing the real-time TDSE belongs to the later
propagation interface.

## Physical coupling factories

For a two-electron atom, `for_two_electron_atom` applies the electron charge
in length gauge or the charge-to-mass ratio in velocity gauge to both electron
coordinates.

For a non-Born--Oppenheimer diatomic ion, `for_diatomic_ion` uses the electronic
and internuclear reduced charges or velocity couplings stored in
`DiatomicIonParameters`. Homonuclear H2+ and D2+ have zero direct
internuclear coupling, while a heteronuclear ion such as HD+ generally has
both coordinate contributions.

`AxialDipoleCouplings` can be supplied directly for a custom two-radial
system.
