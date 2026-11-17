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

## Analytical axial delta kick

`AxialDeltaKick` implements the instantaneous state transformation described
below. The name deliberately uses *kick* rather than *pulse*: it acts once on
an initial state and is not a temporal field sampled by the propagator.

Let

$$
K=\int_{-\epsilon}^{+\epsilon} E_z(t)\,dt
$$

be the integrated electric field of an impulse polarized along the laboratory
`z` axis. For a non-Born--Oppenheimer diatomic ion, define the internal dipole
coordinate

$$
D_z=\xi_e r\cos\theta_e+\xi_R R\cos\theta_R.
$$

Here `r` is the electronic Jacobi coordinate measured from the nuclear center
of mass, `R` is the internuclear coordinate, and `xi_e` and `xi_R` are the
reduced charges stored in `DiatomicIonParameters`. The length-gauge
interaction is

$$
H_{\mathrm{int}}(t)=-E_z(t)D_z.
$$

Integrating the time-dependent Schrodinger equation across an ideal impulse
gives the exact full-space transformation

$$
\Psi(0^+)=\exp(iK D_z)\Psi(0^-).
$$

Because the two coordinate operators commute in the full product space, this
may also be written as

$$
\Psi(0^+)=
\exp(iK\xi_e r\cos\theta_e)
\exp(iK\xi_R R\cos\theta_R)
\Psi(0^-).
$$

For homonuclear H2+ and D2+, `xi_R=0`, so only the electronic factor remains.
For HD+, both factors generally contribute. With the package convention that
the field-independent length-gauge action is `W=-D_z`, the same kick is

$$
\Psi(0^+)=\exp(-iK W)\Psi(0^-).
$$

### Definition in a finite angular space

Unlike the prolate-spheroidal nodal representation, the two-radial
representation does not store the angular coordinates directly. Multiplying
by `exp(i kappa r cos(theta))` couples an unbounded sequence of angular
momenta, so a pointwise phase followed by projection is not closed in a
finite spherical-harmonic basis.

Let `P` project onto the retained radial and angular representation. The
recommended discrete kick is therefore

$$
U_K^{(P)}=\exp(iK P D_z P),
$$

rather than `P exp(i K D_z) P`. The first definition is unitary within the
finite representation when `P D_z P` is Hermitian. The second can discard
amplitude generated above the angular cutoff and consequently need not
preserve the represented norm. Numerically evaluating the exponential action
does not require resolving a finite pulse in time: it is one instantaneous
operator application, with only the tolerance of the exponential algorithm.

A `z`-polarized kick preserves `M_tot`. It mixes adjacent one-coordinate
angular momenta at first order and, through the exponential, can populate many
retained `l_1`, `l_2`, and `L` sectors. Both inversion parities must therefore
be present. The identical-nucleus exchange sector remains conserved for H2+
and D2+ because their direct internuclear dipole coupling vanishes.

### Initial-state workflow

The kick should be implemented as a state transformation, not as part of the
ground-state solver:

1. Prepare or load a normalized initial state.
2. Embed that state in the dynamics representation containing both inversion
   parities and all intended angular sectors.
3. Apply `U_K^(P)` once.
4. Propagate the kicked state with the field-free Hamiltonian.

The ground state is the usual initial state for absorption spectroscopy and
is the natural default, but it is not a restriction. The same operator may be
applied to an excited eigenstate, a thermal sample, or a prepared
superposition. Keeping preparation and kicking separate also allows the same
ground-state archive to be reused for several kick strengths and for paired
`+K` and `-K` linear-response calculations.

The kick itself must not renormalize the state. For a Hermitian projected
dipole, its norm should be preserved to the exponential-action tolerance.
Any later norm loss should arise from the absorber during field-free
propagation, not from the delta kick.

### Length and velocity gauge

The physical electric-field impulse is not restricted to one gauge, but the
instantaneous state phase is naturally its length-gauge representation. With

$$
E_z(t)=-\frac{dA_z(t)}{dt},
$$

the same delta electric field appears in velocity gauge as a discontinuity

$$
\Delta A_z=-K.
$$

The canonical velocity-gauge wavefunction may remain continuous across that
step, after which propagation must retain the shifted vector potential. If a
gauge transformation immediately returns `A_z` to zero, the wavefunction
acquires exactly the length-gauge phase `exp(i K D_z)`. Thus a kick followed
by an ordinary field-free propagation is most directly and unambiguously
implemented using the length-gauge transformation above.

Exponentiating the velocity-gauge momentum interaction for a delta-shaped
vector potential is a different perturbation. Its electric field is
proportional to the derivative of a delta function and it must not be labeled
as the velocity-gauge form of the electric-field delta kick.

Validation conditions for the implementation are norm
preservation, `+K`/`-K` reversibility, convergence to a narrow finite-area
pulse, conservation of `M_tot`, coupled--uncoupled equivalence in matched
finite spaces, angular-cutoff convergence, and agreement between serial and
distributed radial layouts.

The high-level diatomic interface accepts the kick separately from ordinary
fields:

```python
from grid_tdse.spherical.two_radial import (
    AxialDeltaKick,
    NonBornOppenheimerDiatomicIonTDSE,
)

dynamics = NonBornOppenheimerDiatomicIonTDSE(
    "H2+",
    electronic_grid=electronic_grid,
    internuclear_grid=internuclear_grid,
    initial_state=None,
    ground_state_path=ground_state_path,
    initial_state_policy="compute-if-missing",
    pair_exchange_parity=1,
    initial_kick=AxialDeltaKick(integrated_field=1.0e-3),
    initial_kick_options={
        "max_krylov_dimension": 20,
        "rtol": 1.0e-12,
        "atol": 0.0,
        "max_refinements": 8,
    },
    fields=(),
    final_time=500.0,
    time_step=0.0625,
)
```

`initial_state` is the post-kick state after construction. Initial observable
samples and wavefunction snapshots consequently describe the kicked state.
The ground-state archive is not modified. Kick metadata records the
integrated field, application time, resolved exponential options, and norms
before and after the transformation. An exact `resume()` must retain the same
`initial_kick` declaration so its dynamics metadata matches the interrupted
segment; the constructor's transformed copy is not used, because propagation
starts from the checkpointed state. A compatible `continue_from()` must omit
`initial_kick`, since that checkpoint already contains the transformed state.
To apply a second physical kick between stages, start a new calculation from
the archived state explicitly rather than treating it as a continuation.

The Cartesian two-radial solver exposes the same operation at run time:

```python
result = solver.run(
    distributed_initial_state,
    initial_kick=AxialDeltaKick(1.0e-3),
    initial_kick_options={"rtol": 1.0e-12, "atol": 0.0},
)
```

Its projected dipole action is radial-tile local. The exponential algorithm
uses distributed physical inner products, so the resulting kick is
independent of the radial process layout to its requested tolerance.

### Finite-width discrete delta pulse

`electromagnetic-fields` provides `DiscreteDeltaPulse`, a right-sided
rectangular approximation to the same electric-field impulse:

$$
E_z(t)=\frac{K}{\Delta t},
\qquad t_0\le t<t_0+\Delta t.
$$

Its integrated field is exactly `K`. When the pulse width equals the
propagation time step and `t0` is a step boundary, it is a one-step
rectangular impulse:

```python
from electromagnetic_fields import DiscreteDeltaPulse

time_step = 0.002
pulse = DiscreteDeltaPulse(
    F_str=1.0e-3,
    dt=time_step,
    t0=0.0,
    gauge="length",
)

dynamics = NonBornOppenheimerDiatomicIonTDSE(
    # physical, grid, and angular options omitted
    initial_state=ground_state,
    pair_exchange_parity=1,
    fields=(pulse,),
    field_gauge="length",
    initial_time=0.0,
    final_time=500.0,
    time_step=time_step,
)
```

At finite `dt`, the field-free Hamiltonian acts during this pulse, so its
one-step evolution is not identical to the instantaneous `AxialDeltaKick`.
The discrete result must converge to the analytical kick as `dt` decreases at
fixed `F_str`.

`DiscreteDeltaPulse` also supplies the consistent velocity-gauge vector
potential. It ramps from zero to `-F_str` during the pulse and remains at that
constant afterward. Consequently, a length--velocity comparison must retain
the nonzero post-pulse vector potential or perform the corresponding gauge
transformation; it must not switch directly to a zero-vector-potential
field-free Hamiltonian.

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
