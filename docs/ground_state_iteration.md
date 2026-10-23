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

Each iteration uses the matrix-free Crank--Nicolson implementation from
`integrator-lib`:

```math
\left(I+\frac{\Delta\tau}{2}H\right)\Psi_{n+1}
=
\left(I-\frac{\Delta\tau}{2}H\right)\Psi_n.
```

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
    time_step=0.2,
    max_iterations=500,
    energy_tolerance=1.0e-12,
    state_tolerance=1.0e-10,
    residual_tolerance=1.0e-10,
    linear_solver="gmres",
    linear_rtol=1.0e-12,
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

Failure of the inner Crank--Nicolson Krylov solve raises
`integrator_lib.LinearSolveError`.

Crank--Nicolson is not strongly damping for arbitrarily large imaginary-time
steps. The time step should therefore be included in numerical convergence
tests; increasing it without bound does not necessarily accelerate removal
of high-energy components.

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
    preconditioner=preconditioner,
)
```

The construction of `psi0` and a two-radial preconditioner belongs to the
future atomic and molecular high-level interfaces.

## Diagnostics

`GroundStateResult` contains:

- the normalized final `state`;
- its `energy`, `state_change`, and `residual_norm`;
- `converged`, `iterations`, `imaginary_time`, and `termination_reason`;
- a tuple of `GroundStateIteration` records;
- convenience energy, state-change, and residual history arrays.

Each iteration record includes the energy, energy change, phase-invariant
state change, eigenstate residual, normalized state norm, and the iteration
count and final residual reported by the inner Krylov solve.
