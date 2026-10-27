# Real-time propagation

`grid-tdse` provides a representation-independent propagator for a linear,
possibly time-dependent Schrödinger equation. Spatial representations supply
three objects:

- an initial NumPy state;
- a Hamiltonian action `hamiltonian(state, time)`;
- the physical inner product for that representation.

The propagator advances the equation `i dPsi/dt = H(t) Psi` using one of three
matrix-free methods from `integrator-lib`: Crank--Nicolson, short-iterative
Lanczos (SIL), or short-iterative Arnoldi (SIA). The Hamiltonian is evaluated
at the midpoint of every time step, which is required when fields make it
explicitly time dependent.

## Propagation interval

```python
from grid_tdse.real_time import RealTimePropagator

propagator = RealTimePropagator(
    initial_time=0.0,
    final_time=20.0,
    time_step=0.1,
    integrator="crank-nicolson",
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

Two-radial FEM-DVR calculations provide several Crank--Nicolson
preconditioners. All currently approximate the inverse of the separable
one-coordinate part of the CN matrix. They omit the Coulomb interaction,
external field, and CAP; those terms remain present in every Hamiltonian
action and therefore in the converged solution.

## Crank--Nicolson preconditioners

For a CN matrix written as $A=I+i\Delta t H/2$, the available high-level
selectors are:

- `"spectral"`: the historical `TwoRadialRealTimePreconditioner`. It applies
  the exact inverse of the separable one-body approximation by global weighted
  eigenvector transforms in both coordinates. It is the serial reference and
  remains the default, but its global transform in the partitioned coordinate
  is unsuitable for a permanently distributed state.
- `"diagonal"`: pointwise Jacobi using the diagonal of both one-coordinate
  Hamiltonians. Its setup and action are local, but the weaker approximation
  generally requires more Krylov iterations.
- `"element-block-jacobi"`: non-overlapping blocks formed from FEM-DVR degree-
  of-freedom ownership. Each block retains the full unpartitioned coordinate
  and inverts the separable one-body approximation over its owned elements.
- `"additive-schwarz"`: overlapping element neighborhoods. Local corrections
  are combined with inverse-multiplicity weights, so the scatter weights form
  a partition of unity.

The element methods partition coordinate 2 by default. For a non-BO diatomic
calculation this is the internuclear coordinate $R$, matching the distributed
state layout in which a rank owns an element-aligned $R$ interval but retains
the full electronic coordinate $r$ and every angular channel. The state layer
now records neighboring bridge halos. Distributed Hamiltonian and
preconditioner actions remain future work; the present preconditioner
implementation is serial and establishes the numerical algorithm and
comparison baseline.

Select a method through the CN integrator options:

```python
calculation = NonBornOppenheimerDiatomicIonTDSE(
    # physical, grid, field, and time arguments omitted
    integrator="crank-nicolson",
    integrator_options={
        "solver": "bicgstab",
        "atol": 1.0e-10,
        "preconditioner": "additive-schwarz",
        "preconditioner_options": {
            "partition_coordinate": 2,
            "overlap_elements": 1,
        },
    },
)
```

`preconditioner=True` is retained for backward compatibility and selects
`"spectral"`; `False` or `None` disables preconditioning. The block-Jacobi
method accepts `partition_coordinate`, while additive Schwarz also accepts
`overlap_elements`. Unsupported method-specific options are rejected.

The reproducible comparison driver
[`non_bo_diatomic_cn_preconditioner.py`](../benchmarks/non_bo_diatomic_cn_preconditioner.py)
reports setup time, CN solve time, Krylov iterations, Hamiltonian calls,
residuals, norm changes, and final-state differences for every selected
method. For example:

```bash
python benchmarks/non_bo_diatomic_cn_preconditioner.py \
    --preconditioners spectral diagonal element-block-jacobi additive-schwarz none \
    --partition-coordinate 2 --overlap-elements 1
```

## Integrator selection

Use `integrator="crank-nicolson"` for the implicit Cayley step. Its linear
solver is controlled by `linear_solver`, `linear_rtol`, `linear_atol`,
`linear_max_iterations`, and `gmres_restart`. An optional preconditioner is
accepted only by this method.

Use `integrator="sil"` for a Hermitian Hamiltonian without a CAP:

```python
propagator = RealTimePropagator(
    final_time=20.0,
    time_step=0.1,
    integrator="sil",
    max_krylov_dimension=20,
    krylov_rtol=1.0e-10,
    krylov_atol=0.0,
    max_krylov_refinements=8,
    krylov_reorthogonalize=True,
    krylov_breakdown_tolerance=1.0e-14,
)
```

SIL builds a tridiagonal projection by exploiting Hermiticity. The two-radial
high-level interfaces reject SIL when a CAP is present. A mask is compatible
because it is applied after the otherwise Hermitian propagation step rather
than included in the Hamiltonian.

Use `integrator="sia"` with the same Krylov options for a general generator.
SIA uses an Arnoldi projection and supports the non-Hermitian Hamiltonian
created by a CAP. SIA may also be used without a CAP, although SIL is the more
specialized method for a Hermitian problem.

Both exponential propagators increase the Krylov dimension up to
`max_krylov_dimension`. If the requested tolerance is not reached, they bisect
the requested interval recursively up to `max_krylov_refinements`. Every
accepted or refined substep evaluates the Hamiltonian at its own temporal
midpoint.

## Results and diagnostics

`RealTimePropagationResult` contains:

- a copy of the final state;
- the requested and actual time steps;
- the initial and final times;
- the initial norm;
- one diagnostic record for every accepted step.

The `times` and `norm_history` properties include the initial point. CN records
linear iteration counts and residual norms for every step. SIL and SIA instead
record Hamiltonian calls, accepted and rejected Krylov substeps, maximum
refinement depth, maximum accepted Krylov dimension, and an accumulated error
estimate. Non-applicable CN or Krylov diagnostics are zero. The norm is
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
    radial_operator_mode="element-local",
    integrator="sil",
    integrator_options={
        "max_krylov_dimension": 20,
        "rtol": 1.0e-10,
        "atol": 0.0,
        "max_refinements": 8,
        "reorthogonalize": True,
        "breakdown_tolerance": 1.0e-14,
    },
    time_step=0.5,
    final_time=10.0,
)
result = calculation.run(progress=True)
```

The high-level `integrator` choices are `"crank-nicolson"`, `"sil"`, and
`"sia"`. The `rtol` and `atol` entries control the selected method. CN accepts
the additional `solver`, `max_iterations`, `restart`, `preconditioner`, and
`preconditioner_options` options. SIL and SIA accept `max_krylov_dimension`, `max_refinements`,
`reorthogonalize`, and `breakdown_tolerance`. Options belonging only to another
method are rejected rather than silently ignored.

FEM-DVR uses `radial_coulomb_method="poisson"` by default. Setting it to
`"analytic"` evaluates the radial multipole expression directly on the two
radial grids. This selection concerns the radial Coulomb kernel; both paths
use the same angular multipole expansion and sparse angular contractions.

The independent `radial_operator_mode` option selects how the one-coordinate
kinetic operators are applied. `"global"` uses the existing dense matrices,
while `"element-local"` uses FEM-DVR element actions from `grid-lib`. The
global mode remains the compatibility default and reference implementation.
The selection covers the kinetic second derivatives and the velocity-gauge
radial first derivative. Element-local mode currently requires FEM-DVR.

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
construction and from the current separable CN preconditioner. The latter
remains an approximation to the Hermitian one-coordinate part of the
Crank--Nicolson system. A CAP also makes SIL inapplicable; select SIA or CN
instead.

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
history, named samples, method-specific CN or Krylov diagnostics, integrator
configuration, and user-supplied metadata. State and trajectory checksums are
verified while loading, and loading never enables NumPy pickle support.
Checkpoint format version 2 adds the Krylov diagnostics; version 1 CN
checkpoints remain readable.

Use `resume` for exact recovery of an interrupted propagation:

```python
from grid_tdse.real_time import load_real_time_checkpoint

checkpoint = load_real_time_checkpoint("helium.chkpt")
propagator = RealTimePropagator(
    initial_time=checkpoint.current_time,
    final_time=checkpoint.latest_segment.target_final_time,
    time_step=checkpoint.latest_segment.requested_time_step,
    # Use the same integrator and method-specific options as the interrupted
    # segment.
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
through `TwoElectronAtomTDSE` and demonstrates SIL with a post-step mask. The
corresponding non-Born--Oppenheimer H2+ workflow is shown in
[`examples/tdse_non_bo_h2p.py`](../examples/tdse_non_bo_h2p.py), which uses SIA
because its CAP makes the Hamiltonian non-Hermitian.
