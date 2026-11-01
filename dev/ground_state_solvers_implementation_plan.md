# Ground-state eigensolvers implementation plan

Status: in progress; serial phases 1--5 complete

Last updated: 2026-11-01

## 1. Purpose

`grid-tdse` currently obtains non-Born--Oppenheimer diatomic ground states by
imaginary-time propagation (ITP). The existing SIL, SIA, Crank--Nicolson, and
exponential-Euler paths provide valuable, independently testable reference
implementations, but they solve a sequence of propagation problems rather than
the Hermitian eigenvalue problem directly.

This plan adds two matrix-free ground-state eigensolvers:

1. a thick-restarted Hermitian Krylov/Rayleigh--Ritz solver, referred to below
   as the thick-restarted Lanczos solver; and
2. a preconditioned Davidson solver built on the same subspace and Ritz
   infrastructure.

The Lanczos solver is the first implementation target because it requires only
Hamiltonian actions and physical inner products. Davidson follows because it
can reduce Hamiltonian calls substantially when a useful preconditioner is
available. ITP remains supported as the robust numerical reference and as a
fallback when direct eigensolver convergence is unsatisfactory.

The first scientific target is the lowest state in a selected symmetry sector
of the non-BO one-electron diatomic Hamiltonian. The numerical abstractions
should nevertheless be representation-independent enough to support other
Hermitian `grid-tdse` Hamiltonians later.

## 2. Motivation

The converged ground-state chains on Saga show two related costs at increasing
angular cutoff:

- the number of Krylov Hamiltonian actions required by every ITP step grows;
- tighter stable or efficient ITP time steps can be required as the spectral
  width grows; and
- each outer ITP step discards most of the spectral information constructed by
  the inner exponential action.

For the H2+ electronic-11-point chain, the measured Krylov Hamiltonian calls
per ITP step increased from approximately 10.6 at `l_max = 0` to 107.9 at
`l_max = 10`. A direct Hermitian eigensolver instead keeps a subspace across
iterations and targets the lowest Ritz pair explicitly.

The primary question is therefore not whether the new solvers reproduce the
ITP energy, but whether they reach the same residual tolerance with fewer
Hamiltonian actions and lower elapsed time at acceptable memory use.

## 3. Scope

### 3.1 Included

- Lowest-eigenpair solution for real-symmetric and complex-Hermitian operators.
- Matrix-free Hamiltonian actions.
- User-supplied physical inner products and batched inner products.
- Serial and distributed-array state vectors.
- Full and compact fixed-total-magnetic-quantum-number angular storage.
- Existing H2+, HD+, and D2+ non-BO diatomic representations.
- Continuation from a lower-`l_max` state archive.
- Residual-, energy-, and state-based diagnostics.
- Restart and memory controls suitable for large wavefunctions.
- Root-rank progress output, log files, state archives, observables, and
  profiling consistent with the current ground-state examples.

### 3.2 Deferred

- Excited-state locking and general multi-root calculations.
- Block Davidson and LOBPCG.
- Chebyshev-filtered subspace iteration.
- Shift-invert methods and sparse direct factorization.
- PETSc/SLEPc integration.
- Replacement or removal of any ITP solver.
- Production optimization before correctness and convergence are established.

LOBPCG should be reconsidered after Davidson is working. Its block formulation
is attractive for excited states, but it adds complexity without an immediate
advantage for the single ground-state target.

## 4. Mathematical contract

Let `H` be self-adjoint under the physical weighted inner product. For vectors
`x` and `y`, write

```math
\langle x,y\rangle_W = x^\dagger W y.
```

The Rayleigh quotient and residual are

```math
\theta(x) = \frac{\langle x,Hx\rangle_W}
                  {\langle x,x\rangle_W},
\qquad
r(x) = Hx - \theta(x)x.
```

Convergence is governed primarily by the physical residual norm,

```math
\|r\|_W \leq \varepsilon_r,
```

with energy and phase-aligned state changes retained as secondary diagnostics.
This avoids declaring convergence merely because an iteration has stagnated.

For a physical-orthonormal basis `V`, Rayleigh--Ritz extraction forms

```math
T = V^\dagger W H V
```

and solves the small dense Hermitian problem. All orthogonalization, projected
matrices, residuals, and convergence tests must use the same physical inner
product. Euclidean orthogonality is not a valid substitute for FEM-DVR states
unless the representation has already absorbed the quadrature weights.

## 5. Architectural decisions

### 5.1 Package ownership

The eigensolvers belong in `grid_tdse.ground_state`, not `integrator-lib`.
They solve eigenvalue problems rather than integration problems and need
ground-state-specific result data and convergence semantics.

`integrator-lib` remains responsible for reusable exponential actions used by
ITP. `grid-tdse` supplies physical inner products, Hamiltonian adapters,
preconditioners, serial/MPI representations, observables, and persistence.

### 5.2 Layering

Use three layers:

1. **Generic subspace layer**
   - weighted orthogonalization;
   - projected Hermitian matrices;
   - Ritz extraction;
   - residual construction;
   - restart selection; and
   - common result and history records.
2. **Algorithm layer**
   - thick-restarted Lanczos/Hermitian Krylov expansion; and
   - Davidson correction expansion.
3. **Representation adapters**
   - serial two-radial state handling;
   - one-dimensional distributed layout, where retained for comparison; and
   - Cartesian two-dimensional distributed layout.

The generic layers operate on local coefficient arrays. Global semantics enter
through callbacks for inner products and batched reductions. This matches the
current distributed ground-state design and avoids coupling the generic solver
to MPI types.

### 5.3 Common API

Introduce a shared result type and compatible solver interfaces. A provisional
shape is:

```python
result = solver.solve(
    initial_state,
    hamiltonian_action,
    inner_product,
    inner_products=batched_inner_products,
    preconditioner=preconditioner,
    iteration_observer=observer,
)
```

The exact API should be finalized in phase 1, but it must support:

- local-array input and output;
- real and complex dtypes;
- in-place-safe operator wrappers;
- optional batched inner products;
- an optional eigenvalue-dependent preconditioner;
- deterministic iteration callbacks; and
- cancellation through controlled exceptions rather than MPI divergence.

The common result should record at least:

- eigenvalue and eigenvector;
- residual norm;
- convergence flag and termination reason;
- outer iterations and restarts;
- Hamiltonian actions;
- preconditioner applications;
- scalar values and collective reductions;
- maximum and final subspace dimensions;
- elapsed solve time; and
- per-iteration diagnostic history when requested.

Avoid a deep class hierarchy initially. Shared functions and dataclasses are
preferable until the two solvers reveal a stable common protocol.

## 6. Shared numerical infrastructure

Implement and test the following before either production solver:

### 6.1 Physical orthogonalization

- Modified Gram--Schmidt under the physical inner product.
- Mandatory second pass when the post-projection norm indicates loss of
  orthogonality.
- Optional always-reorthogonalize mode for reference tests.
- Batched overlaps against the complete basis so MPI uses one collective per
  pass rather than one collective per basis vector.
- A configurable breakdown threshold.

### 6.2 Ritz extraction

- Incremental construction of `V^* W H V` from cached `V` and `HV`.
- Explicit Hermitian symmetrization of the small projected matrix to suppress
  roundoff asymmetry.
- Dense solution with `scipy.linalg.eigh` or `numpy.linalg.eigh`.
- Reconstruction of Ritz vectors, Hamiltonian images, and residuals without an
  additional Hamiltonian action when `HV` is cached.
- Phase alignment for state-change diagnostics.

### 6.3 Subspace storage

Store basis vectors and, by default, their Hamiltonian images. Caching `HV`
reduces Hamiltonian calls but approximately doubles subspace storage. Provide a
documented low-memory mode that recomputes `HV` during restart if future state
sizes make the cache prohibitive.

Memory estimates must be printed before a solve when requested:

```text
state bytes x (basis vectors + cached H vectors + work vectors)
```

The configured subspace dimension is therefore a memory limit as well as a
convergence parameter.

### 6.4 Diagnostics

Track:

- Ritz energy;
- energy change;
- residual norm;
- phase-aligned state change;
- basis orthogonality error;
- accepted and rejected expansion vectors;
- subspace dimension and restart count;
- Hamiltonian and preconditioner calls;
- time in Hamiltonian actions, reductions, orthogonalization, Ritz solution,
  preconditioning, and restart; and
- solver wall time.

## 7. Thick-restarted Lanczos solver

### 7.1 Initial implementation

Begin with an unrestarted Hermitian Krylov solver on small problems:

1. normalize the initial vector physically;
2. apply the Hamiltonian;
3. expand with the residual/search direction;
4. physically orthogonalize and reorthogonalize;
5. update the projected Hermitian problem;
6. extract the lowest Ritz pair; and
7. stop on the residual criterion.

Although the implementation is called Lanczos, robust full
reorthogonalization and thick restart make the internal representation closer
to a general Hermitian Krylov/Rayleigh--Ritz method than to a bare three-term
recurrence. The code and documentation must not assume a tridiagonal projected
matrix after restart unless the mathematically correct thick-restart recurrence
has been implemented and tested.

### 7.2 Thick restart

When the basis reaches `maximum_subspace_dimension`:

- retain a configurable number of the lowest Ritz vectors;
- retain enough residual/search information to continue expansion;
- rebuild a physical-orthonormal basis;
- transform or reconstruct the cached Hamiltonian images consistently; and
- verify the Ritz residual before and after restart.

Suggested initial defaults for evaluation, not permanent API promises:

- maximum subspace dimension: 30;
- retained Ritz vectors: 4;
- residual tolerance: `1e-8`;
- energy tolerance: `1e-10` Eh; and
- full two-pass reorthogonalization.

The single-root solver does not require locking. Retaining more than one Ritz
vector is still useful when low-lying states are clustered.

### 7.3 Breakdown handling

Distinguish:

- **happy breakdown**: the residual is converged;
- **invariant-subspace breakdown**: the new direction vanishes but the target
  is not converged; and
- **numerical breakdown**: non-finite values, negative norm beyond roundoff, or
  severe loss of orthogonality.

Each must produce a clear termination reason. Random perturbations must not be
introduced silently.

## 8. Davidson solver

### 8.1 Shared subspace machinery

Davidson should reuse the physical orthogonalization, `V/HV` storage, Ritz
extraction, restart, diagnostics, and result types developed for Lanczos. The
algorithm-specific expansion is a preconditioned residual.

For Ritz value `theta` and residual `r`, form approximately

```math
t \approx -(H-\theta I)^{-1}r.
```

Project `t` against the current basis, reorthogonalize, normalize, and add it to
the subspace. If preconditioning produces a nearly dependent or invalid vector,
fall back to an unpreconditioned residual direction and record that event.

The preconditioner protocol must receive the current Ritz value:

```python
correction = preconditioner(residual, ritz_value)
```

It may also receive iteration metadata later, but the first interface should
remain small.

### 8.2 Preconditioner sequence

Implement increasingly physical preconditioners in separate steps:

1. **Identity preconditioner**
   - validates Davidson orchestration independently of preconditioning;
   - provides a subspace-iteration baseline.
2. **Coefficient-diagonal/Jacobi preconditioner**
   - local and MPI-friendly;
   - uses a safeguarded approximation to `diag(H) - theta`;
   - applies a denominator floor and sign-preserving level shift.
3. **Separable one-body spectral preconditioner**
   - approximates `H` by the electronic and internuclear one-body terms;
   - applies an approximate inverse of `H_1 + H_2 - theta I`;
   - prototypes in serial before distributed use; and
   - uses explicit regularization near small denominators.
4. **Distributed block-Jacobi or additive-Schwarz preconditioner**
   - considered only if the separable preconditioner is insufficient or too
     communication-heavy.

The existing `TwoRadialOneBodyPreconditioner` is designed for the
Crank--Nicolson system `I + (dt/2)H_0`. It must not be reused unchanged for
Davidson. Its eigensystem machinery may be factored into a shared component,
but Davidson needs an eigenvalue-dependent approximation to
`H - theta I` with different singularity safeguards.

### 8.3 Restart

Initially use the same thick-restart policy as the Hermitian Krylov solver.
Retain the lowest Ritz vectors and selected recent correction vectors only when
they add independent information. Avoid solver-specific restart optimization
until the reference implementation is stable.

## 9. Serial implementation phases

### Phase 0: freeze benchmarks and acceptance criteria

- Record the ITP reference commands, commits, grids, symmetry sectors, initial
  states, energies, residuals, Hamiltonian calls, and timings.
- Use H2+ electronic-11-point `l_max = 0, 2, 4, 6` as the first progression.
- Preserve the current converged state archives for continuation and
  cross-validation.
- Define a small dense reference problem for every new numerical primitive.

Completion gate: the reference dataset and comparison script are reproducible
without changing solver code.

Status: complete. Existing converged H2+, HD+, and D2+ chains and their Saga
provenance provide the serial reference data; the broader equal-hardware
performance comparison remains phase 8.

### Phase 1: weighted subspace primitives

- Add physical normalization and batched orthogonalization utilities.
- Add projected Hermitian matrix and Ritz extraction utilities.
- Add phase alignment and orthogonality diagnostics.
- Add common result, history, and termination records.

Completion gate: dense real and complex weighted problems reproduce direct
eigenpairs, and restart transformations preserve the represented subspace.

Status: complete.

### Phase 2: unrestarted Hermitian Krylov solver

- Implement the non-restarted lowest-Ritz-pair path.
- Add happy-breakdown and invalid-state handling.
- Validate against exact dense spectra and the smallest serial diatomic cases.

Completion gate: the solver reaches the requested residual for exact-test and
small physical Hamiltonians without relying on Euclidean inner products.

Status: complete.

### Phase 3: thick restart

- Add maximum and retained subspace dimensions.
- Preserve cached Hamiltonian images correctly through restart.
- Test clustered spectra, frequent forced restarts, and memory accounting.
- Add restart diagnostics and reproducibility tests.

Completion gate: restarted and unrestarted solutions agree within tolerance,
including when the restart dimension is intentionally small.

Status: complete.

### Phase 4: Davidson

- Implement identity-preconditioned Davidson.
- Add the diagonal/Jacobi preconditioner.
- Prototype the separable one-body spectral preconditioner.
- Compare convergence histories and Hamiltonian calls against Lanczos.

Completion gate: Davidson is never less robust than its documented fallback,
and at least one physical preconditioner demonstrably reduces Hamiltonian calls
on a representative `l_max >= 4` problem.

Status: complete.

Local phase-4 validation on 2026-11-01 used the deterministic small-grid H2+
`l_max = 4` test problem with a 16-vector subspace and four retained Ritz
vectors. Identity Davidson required 28 Hamiltonian actions, while the
separable one-body spectral correction required 15 and reached the same energy
and residual tolerance. This is a regression/performance gate, not a
production benchmark; the standard-grid comparison remains a later phase.

### Phase 5: serial non-BO application integration

- Add high-level serial solver selection without duplicating representation
  setup.
- Support lower-`l_max` state continuation.
- Reuse existing progress logging, observables, state archive metadata, and
  terminal summaries.
- Record the solver and all numerical options in saved provenance.

Completion gate: a serial example can select ITP, thick-restarted Lanczos, or
Davidson and produces compatible result summaries and archives.

Status: complete. The high-level serial non-BO interface now selects all three
solver families without rebuilding the representation, accepts compatible
lower-`l_max` archives directly, and reuses observable sampling, terminal and
file logging, result summaries, and state persistence. Direct-solver archives
record the solver, restart controls, preconditioner and safeguards,
Hamiltonian actions, subspace dimensions, and continuation identifiers.
Serial integration tests cover H2+, D2+, and HD+ as well as identity,
diagonal, and separable one-body Davidson preconditioning.

## 10. MPI implementation phases

### Phase 6: distributed inner products and subspace operations

- Adapt the shared solver to local coefficient arrays.
- Use `TwoRadialDistributed2DReductions` for physical norms and overlaps.
- Batch all overlaps with the current basis into one collective per
  orthogonalization pass.
- Build projected-matrix updates from batched reductions.
- Ensure every rank makes identical convergence, rejection, and restart
  decisions.
- Keep progress output and file writing on the root rank.

Completion gate: `1x1`, `1x2`, `2x1`, and `2x2` process grids reproduce the
serial Ritz value, residual, and phase-aligned state within tolerance.

### Phase 7: distributed Lanczos and Davidson integration

- Connect both solvers to `TwoRadialDistributed2DLayout` and the distributed
  Hamiltonian action.
- Add distributed Jacobi preconditioning.
- Add the separable one-body preconditioner only after its communication and
  memory behavior are understood.
- Profile Hamiltonian work, collectives, orthogonalization, Ritz work, and
  preconditioning separately.

Completion gate: full converged H2+ runs complete on Saga with deterministic
results and no rank-dependent termination.

### Phase 8: Saga comparison campaign

Run, at minimum:

- H2+, electronic-11-point grid, `l_max = 0, 2, 4, 6`;
- continuation from the same lower-`l_max` archives;
- ITP SIL and exponential Euler references;
- thick-restarted Lanczos; and
- Davidson with identity, diagonal, and accepted physical preconditioners.

Repeat selected configurations to separate solver differences from node
variation. Extend to electronic-21-point H2+ and then HD+/D2+ only after the
standard-grid conclusions are clear.

Completion gate: archived results identify the preferred default by problem
size and document any regimes in which ITP remains superior.

## 11. Test plan

### 11.1 Unit tests

- Real symmetric dense matrices.
- Complex Hermitian dense matrices.
- Nontrivial positive quadrature weights.
- Diagonal matrices with exact happy breakdown.
- Clustered lowest eigenvalues.
- Initial states with small ground-state overlap.
- Deliberately orthogonal initial states, with documented behavior.
- Forced restart at small subspace dimensions.
- Repeated eigenvalues and near-degeneracy.
- Orthogonalization rejection and fallback paths.
- Non-finite operator, inner-product, and preconditioner output.
- Preconditioner denominator regularization.
- Result counters and termination reasons.

Weighted dense tests should also be constructed through a similarity transform
so the expected physical eigenvectors and eigenvalues are known independently.

### 11.2 Representation tests

- Serial Hamiltonian action and physical inner product.
- Compact versus full angular storage.
- Real versus complex coefficient arrays.
- Lower-`l_max` continuation and symmetry compatibility checks.
- Phase-aligned comparison with ITP states.
- Archive round-trip and provenance.

### 11.3 MPI tests

- Thread-communicator tests for reduction ordering and restart decisions.
- Real MPI tests for `1x2`, `2x1`, and `2x2` grids.
- Uneven radial partitions.
- Batched overlap equivalence to scalar reductions.
- Serial/distributed energy, residual, and state agreement.
- Root-only logging and failure propagation.

## 12. Benchmark metrics and acceptance criteria

### 12.1 Primary metrics

- Hamiltonian actions to reach `||r||_W <= 1e-8`.
- Total elapsed solve time.
- Peak memory per rank.
- Collective reduction calls and reduced scalar values.

### 12.2 Secondary metrics

- Final energy and energy error relative to the converged ITP reference.
- Phase-aligned state difference.
- Observable differences.
- Restart count and achieved subspace dimension.
- Orthogonalization and small dense eigensolve time.
- Preconditioner applications and time.
- Run-to-run timing spread and CPU utilization on Saga.

### 12.3 Numerical acceptance

Initial targets, subject to refinement from the reference dataset:

- energy agreement no worse than `1e-10` Eh;
- physical residual no worse than `1e-8`;
- norm equal to one within `1e-12`;
- phase-aligned state difference no worse than `1e-8` for like-for-like
  representations; and
- symmetry and sampled observable values consistent with the ITP state within
  their established tolerances.

Performance claims require equal Hamiltonian, grid, angular storage, symmetry,
initial state, residual tolerance, and hardware allocation. A lower energy with
a larger residual is not a successful comparison.

## 13. Memory and scalability considerations

For local state size `N_local`, scalar size `b`, basis dimension `m`, and cached
Hamiltonian images, the leading storage is approximately

```math
2mN_{\mathrm{local}}b
```

before work vectors and representation data. This makes unbounded subspace
growth unsuitable for high `l_max` dynamics-sized states.

The implementation must therefore:

- expose a hard subspace-dimension cap;
- report estimated solver workspace;
- support a low-memory `HV` recomputation policy if needed;
- avoid root-rank gathers during iterations;
- retain distributed vectors in the active Cartesian layout; and
- measure reduction latency as basis size grows.

Pure MPI was fastest in the current 2D Hamiltonian benchmarks up to the tested
rank counts. The eigensolver should initially use one CPU thread per MPI rank,
while preserving thread controls for later experiments.

## 14. Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Incorrect Euclidean orthogonalization | Wrong Ritz problem and false convergence | Require physical-inner-product tests at every layer |
| Loss of basis orthogonality | Duplicate Ritz values or stalled convergence | Batched two-pass reorthogonalization and explicit diagnostics |
| Too-small restart space | Slow or erratic convergence | Retain several low Ritz vectors and benchmark restart parameters |
| Large subspace memory | Per-rank memory exhaustion | Hard cap, memory estimate, optional `HV` recomputation |
| Excess MPI reductions | Poor scaling despite fewer Hamiltonian calls | Batch basis overlaps and projected-matrix updates |
| Weak Davidson preconditioner | No gain over Lanczos | Identity baseline, fallback direction, staged preconditioners |
| Near-singular Davidson inverse | Exploding correction vectors | Denominator floor, level shift, finite/norm checks |
| Low initial ground-state overlap | Convergence to another represented state or stagnation | Continuation archives, overlap diagnostics, documented behavior |
| Near-degenerate symmetry sector | Unstable state comparisons | Compare invariant subspaces/observables and retain multiple Ritz vectors |
| Rank-dependent control flow | MPI deadlock | Synchronize all accept/restart/stop decisions and test failures |

## 15. Recommended pull-request sequence

1. **Weighted Ritz core**
   - shared dataclasses, orthogonalization, projected matrices, and dense tests.
2. **Serial thick-restarted Lanczos**
   - generic solver, restart, diagnostics, and small physical integration.
3. **Serial Davidson**
   - identity and diagonal preconditioners plus common restart reuse.
4. **Two-radial physical preconditioner**
   - factored one-body eigensystem support and safeguarded Davidson inverse.
5. **High-level serial application path**
   - examples, logging, observables, archives, and comparison tooling.
6. **Cartesian MPI eigensolvers**
   - batched reductions, distributed preconditioner, and MPI tests.
7. **Saga benchmark and archive**
   - reproducible scripts, raw summaries, analysis, and default recommendation.

Each PR should be independently testable and should not combine a new solver,
a new preconditioner, and MPI integration in one review unit.

## 16. Decision gates

### Gate A: shared Ritz machinery

Proceed only when weighted dense tests and restart transformations are correct.

### Gate B: serial Lanczos viability

Proceed to Davidson when restarted Lanczos reliably converges the standard
serial H2+ cases and gives a trustworthy Hamiltonian-call baseline.

### Gate C: Davidson value

Promote Davidson to the distributed path only if a tested preconditioner lowers
Hamiltonian actions or elapsed time without reducing robustness.

### Gate D: MPI correctness

Run production Saga comparisons only after all tested process grids reproduce
the serial eigenpair and terminate consistently.

### Gate E: default selection

Choose a default solver from measured regimes rather than globally. A likely
outcome is:

- Lanczos as the no-preconditioner direct solver;
- Davidson when a validated physical preconditioner is available; and
- ITP as the robust reference and fallback.

No existing ITP default should change until the archived Saga comparison
supports that decision.

## 17. Definition of completion

This implementation path is complete when:

- serial and Cartesian-MPI thick-restarted Lanczos are available;
- serial and Cartesian-MPI Davidson are available with at least one useful,
  safeguarded preconditioner;
- both use the physical weighted inner product consistently;
- both support continuation, logging, observables, profiling, and archives;
- correctness is established against dense spectra and converged ITP states;
- Saga results compare Hamiltonian calls, collectives, memory, and elapsed time
  on equal numerical problems; and
- the user documentation states when to use Lanczos, Davidson, or ITP.

## 18. Current status

| Item | Status |
| --- | --- |
| ITP SIL/SIA reference path | available |
| ITP exponential-Euler path | available |
| Serial and Cartesian-MPI physical reductions | available |
| Ground-state continuation and archives | available |
| Hamiltonian/reduction profiling | available |
| Weighted generic Ritz primitives | implemented; phase-1 tests pass locally |
| Unrestarted Hermitian Krylov solver | implemented; phase-2 tests pass locally |
| Thick-restarted Lanczos | implemented; phase-3 tests pass locally |
| Davidson | implemented; phase-4 tests pass locally |
| Davidson-specific preconditioner | diagonal and separable one-body prototypes implemented |
| Distributed direct eigensolvers | not started |
| Saga comparison campaign | not started |
