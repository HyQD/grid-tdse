# Real-time propagation

`grid-tdse` provides a representation-independent propagator for a linear,
possibly time-dependent Schrödinger equation. Spatial representations supply
three objects:

- an initial NumPy state;
- a Hamiltonian action `hamiltonian(state, time)`;
- the physical inner product for that representation.

The propagator advances the equation `i dPsi/dt = H(t) Psi` using the
matrix-free implicit midpoint/Crank--Nicolson implementation from
`integrator-lib`. The Hamiltonian is evaluated at the midpoint of every time
step, which is required when fields make it explicitly time dependent.

## Propagation interval

```python
from grid_tdse.real_time import RealTimePropagator

propagator = RealTimePropagator(
    initial_time=0.0,
    final_time=20.0,
    time_step=0.1,
    linear_solver="gmres",
    linear_rtol=0.0,
    linear_atol=1.0e-12,
    progress_interval=10,
)

result = propagator.propagate(
    initial_state,
    hamiltonian,
    inner_product,
    preconditioner=preconditioner,
)
```

The requested time step is treated as an upper bound. The interval is split
into uniformly sized steps and the actual step is adjusted downward when
necessary so that `final_time` is reached exactly.

The Hamiltonian callable must be linear in its state argument. For a field-free
Hamiltonian, its time argument can simply be ignored.

Two-radial FEM-DVR calculations can use
`TwoRadialRealTimePreconditioner`. It approximates the real-time
Crank--Nicolson inverse with the separable one-coordinate Hamiltonians while
omitting the Coulomb and explicitly time-dependent interactions.

## Results and diagnostics

`RealTimePropagationResult` contains:

- a copy of the final state;
- the requested and actual time steps;
- the initial and final times;
- the initial norm;
- one diagnostic record for every accepted step.

The `times` and `norm_history` properties include the initial point. Linear
iteration counts and residual norms are recorded for every step. The norm is
computed as `inner_product(state, state)`, following the established dynamics
convention in which the norm is the total represented probability rather than
its square root. This is also the denominator needed to form conditional
expectation values in post-processing.

Optional named samplers receive a read-only state view and the corresponding
time at the initial point and after every accepted step. Their outputs are
stored in `result.samples`. The propagation layer validates that every output
is finite, numeric, and retains a constant shape, but it does not normalize
sampled expectation values.

```python
result = propagator.propagate(
    initial_state,
    hamiltonian,
    inner_product,
    samplers={"expec_z": total_z_expectation},
)

times = result.times
expec_z = result.samples["expec_z"]
norm_t = result.norm_history
```

Real-time propagation never normalizes the state. An initial state with norm
different from one retains that normalization under a Hermitian Hamiltonian.

## High-level two-radial interfaces

`TwoElectronAtomTDSE` and `NonBornOppenheimerDiatomicIonTDSE` assemble the
FEM-DVR representation, field-free Hamiltonian, z-polarized dipole
interaction, integrator, preconditioner, built-in sampling, and checkpoint
metadata. Their `run(progress=True)` method returns the same
`RealTimePropagationResult` used by the representation-independent layer.

The preferred initial-state input is a loaded `TwoRadialState`. The caller
still specifies the grid and angular truncations, and the interface validates
the archived physical parameters and radial representation before using the
coefficients. A NumPy array is also accepted for custom states; in that case
the caller is responsible for ensuring that the array describes the requested
physical setup. States are copied but are not normalized.

```python
from grid_tdse.spherical.two_radial import (
    TwoElectronAtomTDSE,
    load_two_radial_state,
)

initial_state = load_two_radial_state("helium-ground-state.npz")
calculation = TwoElectronAtomTDSE(
    "He",
    grid=initial_state.grid1_specification,
    initial_state=initial_state,
    fields=[(pulse, (0.0, 0.0, 1.0))],
    field_gauge="length",
    l_max=initial_state.basis1.l_max,
    m_max=initial_state.basis1.m_max,
    lambda_max=initial_state.multipoles.lambda_max,
    mu_max=initial_state.multipoles.mu_max,
    M_tot=0,
    radial_coulomb_method="poisson",
    integrator="crank-nicolson",
    integrator_options={
        "solver": "bicgstab",
        "rtol": 0.0,
        "atol": 1.0e-10,
        "preconditioner": True,
    },
    time_step=0.5,
    final_time=10.0,
)
result = calculation.run(progress=True)
```

Only `crank-nicolson` is currently implemented. The separate `integrator` and
`integrator_options` arguments provide the dispatch point for future methods
such as SIL without changing either physical-system interface.

FEM-DVR uses `radial_coulomb_method="poisson"` by default. Setting it to
`"analytic"` evaluates the radial multipole expression directly on the two
radial grids. This selection concerns the radial Coulomb kernel; both paths
use the same angular multipole expansion and sparse angular contractions.

For a homonuclear non-Born--Oppenheimer ion, a z-polarized field preserves
pair-exchange parity but couples even and odd electronic partial waves. The
diatomic interface therefore retains the selected parity of the internuclear
coordinate while using both parities for the electronic coordinate. A
symmetry-reduced archived ground state is embedded into this larger angular
space automatically. The archive's complete particle parameters, including
its CODATA set, must match the requested real-time calculation.

## Absorbing boundaries

`grid-lib` provides reusable radial profiles through `RadialCosineMask` and
`CubicCAP`:

```python
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

mask = RadialCosineMask(r0=24.0, n=4.0)
cap = CubicCAP(alpha=0.75, eta=0.01)
```

Both are callables with the signature `profile(radial_grid, r_max)`. A finite
one-dimensional array evaluated on the representation's radial nodes can be
supplied instead. Mask values must be real and lie between zero and one. The
imaginary part of every CAP value must be non-positive.

The atomic interface accepts `mask` or `cap` and applies the same radial
profile to both electron coordinates. One mask step therefore multiplies the
state by $M(r_1)M(r_2)$, while a CAP contributes $W(r_1)+W(r_2)$ to the
Hamiltonian.

The non-Born--Oppenheimer diatomic interface accepts the coordinate-specific
`mask_r`, `mask_R`, `cap_r`, and `cap_R` arguments. Its corresponding factors
are $M_r(r)M_R(R)$ and $W_r(r)+W_R(R)$. This permits different absorber onset
and strength parameters for the electronic and internuclear grids.

A mask and CAP on the same coordinate are mutually exclusive and are rejected
at construction. It is valid for a diatomic calculation to use a mask on one
coordinate and a CAP on the other.

A mask is applied once after every accepted time step, before the norm,
observables, and checkpoint are recorded. It is not applied to the initial
state. A CAP is instead included as a diagonal, non-Hermitian term in every
real-time Hamiltonian action. CAP terms are excluded from ground-state
construction and from the current separable preconditioner. The latter remains
an approximation to the Hermitian one-coordinate part of the
Crank--Nicolson system.

Neither absorber renormalizes the state. Consequently, norm loss represents
absorbed probability and all recorded expectation values remain unnormalized.
The realized absorber parameters and value digests are stored in the dynamics
metadata. Exact checkpoint resume therefore verifies the same absorber setup,
whereas starting a new propagation segment may select another absorber.

## Checkpoints and recovery

Pass an exact output path and a positive interval to save a rolling checkpoint
after a specified number of accepted steps. The final state is always saved,
even when the final step does not coincide with the interval.

```python
result = propagator.propagate(
    initial_state,
    hamiltonian,
    inner_product,
    samplers={"expec_z": total_z_expectation},
    checkpoint_path="helium.chkpt",
    checkpoint_interval=100,
    compatibility_metadata=compatibility_metadata,
    dynamics_metadata=field_metadata,
)
```

Checkpoint replacement is atomic: a complete previous checkpoint remains
available if writing the next one fails. A pre-existing file is protected for
a new propagation unless `checkpoint_overwrite=True` is explicit. No filename
extension is added.

The archive contains the current state, time and completed-step count, norm
history, named samples, linear-solver diagnostics, integrator configuration,
and user-supplied metadata. State and trajectory checksums are verified while
loading, and loading never enables NumPy pickle support.

Use `resume` for exact recovery of an interrupted propagation:

```python
from grid_tdse.real_time import load_real_time_checkpoint

checkpoint = load_real_time_checkpoint("helium.chkpt")
propagator = RealTimePropagator(
    initial_time=checkpoint.current_time,
    final_time=checkpoint.latest_segment.target_final_time,
    time_step=checkpoint.latest_segment.requested_time_step,
    # Use the same linear-solver settings as the interrupted segment.
)
result = propagator.resume(
    checkpoint,
    hamiltonian,
    inner_product,
    samplers={"expec_z": total_z_expectation},
    compatibility_metadata=compatibility_metadata,
    dynamics_metadata=field_metadata,
)
```

Exact resume requires the original target time, time step, integrator options,
dynamics metadata, and sampler names. The checkpoint source path is reused by
default. Supply `checkpoint_interval` again to retain periodic output during
the resumed run; otherwise only its final state is saved. The recovered samples
and diagnostics are appended to the interrupted segment.

## Continuing with another field

`continue_from` starts a new propagation segment from the stored state. It
requires matching compatibility metadata but deliberately permits a different
Hamiltonian, external field, gauge, integrator configuration, final time, or
set of sampled observables.

```python
second_propagator = RealTimePropagator(
    initial_time=checkpoint.current_time,
    final_time=second_final_time,
    time_step=second_time_step,
)
second_result = second_propagator.continue_from(
    checkpoint,
    second_hamiltonian,
    inner_product,
    samplers={"expec_z": total_z_expectation},
    compatibility_metadata=compatibility_metadata,
    dynamics_metadata=second_field_metadata,
)
```

Earlier segments and their observables remain in the updated archive. Each
segment records its own dynamics and integrator metadata. Times remain on one
continuous global axis; a second pulse that starts at the checkpoint time can
use that time as its field `t0`.

For two-radial calculations, use
`two_radial_compatibility_metadata(representation, physical_state=...)`. It
records the physical system, reconstructible grid and angular specifications,
state shape, and a representation identifier formed from the realized radial
quadratures and ordered angular channel tables.

## Helium example

[`examples/tdse_he_atom.py`](../examples/tdse_he_atom.py) first computes the
infinite-nuclear-mass helium ground state and then propagates it in a
z-polarized, one-cycle sine-square laser pulse. The example exposes length and
velocity gauges through one setting and demonstrates composition of the
field-free two-radial Hamiltonian with the axial dipole interaction. It samples
the unnormalized total electronic position expectation and probability norm as
`expec_z` and `norm_t`, respectively. It also demonstrates periodic checkpoint
output with explicit physical, representation, laser, gauge, and CODATA
metadata.

[`examples/tdse_he_atom_high_level.py`](../examples/tdse_he_atom_high_level.py)
loads an archived helium ground state and performs the same type of calculation
through `TwoElectronAtomTDSE`. The corresponding non-Born--Oppenheimer H2+
workflow is shown in
[`examples/tdse_non_bo_h2p.py`](../examples/tdse_non_bo_h2p.py).
