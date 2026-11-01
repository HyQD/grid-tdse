# Imaginary-time ground-state iteration

`grid_tdse.ground_state.ImaginaryTimeGroundStateSolver` finds the lowest
energy eigenstate represented in a supplied initial wavefunction. It is
independent of spatial grids and physical-system construction.

The solver requires three representation-level objects:

- an initial state represented by a nonempty NumPy array;
- a time-independent Hamiltonian action `hamiltonian(state)`;
- a physical inner product `inner_product(left, right)`.

The Hamiltonian and inner product are explicit inputs so the same iteration
can be used with different coordinate representations and quadrature rules.

## Algorithm

Imaginary-time evolution obeys

```math
\frac{\partial}{\partial\tau}\Psi(\tau)=-H\Psi(\tau).
```

The solver supports four matrix-free integrators from `integrator-lib`:

- `integrator="crank-nicolson"` applies the Cayley step below and remains the
  default;
- `integrator="sil"` uses a Hermitian Lanczos projection to approximate the
  imaginary-time exponential;
- `integrator="sia"` uses a general Arnoldi projection as a validation and
  fallback path;
- `integrator="exponential-euler"` applies a Hermitian Lanczos `phi_1`
  action to the energy-shifted stationary residual.

The Crank--Nicolson step is

```math
\left(I+\frac{\Delta\tau}{2}H\right)\Psi_{n+1}
=
\left(I-\frac{\Delta\tau}{2}H\right)\Psi_n.
```

SIL and SIA instead approximate `exp(-Delta tau H) Psi_n`. For SIL,
`grid-tdse` configures `ShortIterativeLanczos` with
`generator_factor=-1.0`. SIA consumes the generator `-H` directly. SIL is the
preferred exponential method for the ordinary Hermitian ground-state problem
because its projected matrix is tridiagonal and it requires less storage than
Arnoldi.

Exponential Euler uses

```math
\Psi_{n+1}
=
\Psi_n
+
\Delta\tau\,\varphi_1(-\Delta\tau H)
\left[-(H-E_n)\Psi_n\right],
\qquad
\varphi_1(z)=\frac{e^z-1}{z}.
```

The energy shift makes every normalized eigenstate an exact fixed point of
the discrete iteration: its stationary residual is zero independently of the
chosen time step. This update is not an SIL or SIA exponential step. It uses
the same Hermitian Lanczos machinery for the matrix-free `phi_1` action, but
applies that action to the full stationary residual rather than directly to
the state.

The updated state is normalized after every step using the supplied physical
inner product. No Euclidean coefficient norm is assumed.

The energy and normalized eigenstate residual are

```math
E
=
\frac{\langle\Psi|H\Psi\rangle}
{\langle\Psi|\Psi\rangle},
```

```math
\epsilon_r
=
\frac{\left\|H\Psi-E\Psi\right\|}{\left\|\Psi\right\|}.
```

The energy and residual convergence conditions are

```math
|E_n-E_{n-1}|\leq\epsilon_E
```

and

```math
\epsilon_r\leq\epsilon_{\mathrm{res}}.
```

The energy criterion alone is insufficient because it can become stationary
before the wavefunction is an accurate eigenstate.

The normalized state change is evaluated after aligning the arbitrary global
phase of consecutive states:

```math
\Delta_\Psi
=
\min_\phi
\left\|e^{i\phi}\Psi_{n+1}-\Psi_n\right\|.
```

For normalized states this is equivalently

```math
\Delta_\Psi
=
\sqrt{2-2\left|\langle\Psi_n|\Psi_{n+1}\rangle\right|}.
```

For real states, phase alignment reduces to invariance under a global sign
change. For complex states it provides invariance under the complete global
phase. The implementation evaluates the norm of the aligned state difference
using the supplied physical inner product.

Convergence requires the energy change, state change, and eigenstate residual
to satisfy their respective tolerances. State change is not used alone
because a sufficiently small time step can make consecutive states nearly
identical before the eigenstate residual is small.

## Basic usage

```python
import numpy as np

from grid_tdse.ground_state import ImaginaryTimeGroundStateSolver

hamiltonian_matrix = np.array(
    [
        [0.5, 0.1],
        [0.1, 1.5],
    ]
)

solver = ImaginaryTimeGroundStateSolver(
    integrator="sil",
    time_step=0.2,
    max_iterations=500,
    energy_tolerance=1.0e-12,
    state_tolerance=1.0e-10,
    residual_tolerance=1.0e-10,
    max_krylov_dimension=20,
    krylov_rtol=1.0e-12,
)

result = solver.solve(
    initial_state=np.array([1.0, 1.0]),
    hamiltonian=lambda state: hamiltonian_matrix @ state,
    inner_product=np.vdot,
)
```

`result.converged` reports whether all three tolerances were met. Reaching the
maximum number of iterations returns a result with
`termination_reason == "maximum_iterations"`; it does not silently label the
state as converged.

Failure of the inner Crank--Nicolson solve raises
`integrator_lib.LinearSolveError`. Failure of an SIL or SIA exponential after
the configured refinements raises `integrator_lib.KrylovConvergenceError`.
Failure of an exponential-Euler `phi_1` action to meet its tolerance raises
the same exception. The current `phi_1` implementation does not subdivide a
step, so `max_krylov_refinements` does not affect exponential Euler.

Crank--Nicolson is not strongly damping for arbitrarily large imaginary-time
steps. The time step should therefore be included in numerical convergence
tests; increasing it without bound does not necessarily accelerate removal
of high-energy components.

The same time-step convergence requirement applies to the exponential
methods. Although SIL and SIA accurately approximate the exponential for one
chosen step, normalization and finite-precision convergence diagnostics are
still evaluated only at step boundaries.

## Weighted two-radial states

`TwoRadialSystem.inner_product` supplies the correct quadrature-weighted
inner product for a prepared two-radial system:

```math
\langle\Phi|\Psi\rangle
=
\sum_{i,j,a,b}
\Phi_{ijab}^{*}\Psi_{ijab}
w_a^{(1)}w_b^{(2)}.
```

It can be passed directly to the representation-independent solver:

```python
result = solver.solve(
    initial_state=psi0,
    hamiltonian=system.hamiltonian(),
    inner_product=system.inner_product,
    preconditioner=None,
)
```

The construction of `psi0` belongs to the atomic and molecular high-level
interfaces. A two-radial preconditioner may be supplied only when
`integrator="crank-nicolson"`; SIL, SIA, and exponential Euler reject it.

## Diagnostics

`GroundStateResult` contains:

- the normalized final `state`;
- its `energy`, `state_change`, and `residual_norm`;
- `converged`, `iterations`, `imaginary_time`, and `termination_reason`;
- a tuple of `GroundStateIteration` records;
- convenience energy, state-change, and residual history arrays.

Each iteration record includes the energy, energy change, phase-invariant
state change, eigenstate residual, and normalized state norm. CN records its
linear iteration count and final residual. SIL, SIA, and exponential Euler
record Hamiltonian calls, Krylov attempts, refinement depth, Krylov dimension,
and the exponential or `phi_1` residual estimate.

## Progress reporting

By default, the solver prints the current energy and three convergence
measures every ten accepted steps. The converged step or final allowed step
is also printed:

```text
ITP     10: E=-0.590000000000  |dE|=1.234e-05  dPsi=2.345e-04  residual=3.456e-04
```

Set a different interval with `progress_interval`, or disable printing with
`progress_interval=None`:

```python
solver = ImaginaryTimeGroundStateSolver(progress_interval=25)
quiet_solver = ImaginaryTimeGroundStateSolver(progress_interval=None)
```
