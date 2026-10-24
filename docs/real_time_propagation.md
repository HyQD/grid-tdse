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

## Current absorption scope

This layer currently applies no mask, complex absorbing potential, or other
post-step transformation. The returned state is exactly the state accepted by
the time integrator. Absorption will be added as a separate propagation policy
before the two-radial observable interface is introduced.

## Helium example

[`examples/tdse_he_atom.py`](../examples/tdse_he_atom.py) first computes the
infinite-nuclear-mass helium ground state and then propagates it in a
z-polarized, one-cycle sine-square laser pulse. The example exposes length and
velocity gauges through one setting and demonstrates composition of the
field-free two-radial Hamiltonian with the axial dipole interaction. It samples
the unnormalized total electronic position expectation and probability norm as
`expec_z` and `norm_t`, respectively.
