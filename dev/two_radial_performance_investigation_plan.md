# Two-radial performance investigation plan

## Status and purpose

This is a collaborative draft for a systematic performance investigation of
the non-Born--Oppenheimer two-radial implementation on Olivia. It translates
the broad benchmark phases in `improved_parallelization_plan.md` into concrete
scientific workloads, controlled measurements, recorded metrics, and decision
gates.

The **initial investigation is now fixed as a non-MPI, coupled-angular study**.
It uses analytical `AxialDeltaKick` dynamics for H2+ and HD+ on one fixed
radial grid while increasing the angular cutoff. Its principal performance
question is whether the coupled kernels and SIA propagation benefit from
additional threads, especially at `l_max=8` and `10`.

MPI scaling, uncoupled comparisons, driven laser dynamics, and CF4-versus-SIA
remain later stages. They must not enlarge or delay the initial campaign.

This document plans measurements; it does not authorize implementation of new
propagators, kernels, I/O formats, or job scripts. A specialized field-free
propagator is explicitly out of scope.

The initial physical and performance configuration is settled. Later MPI,
integrator, and application studies remain intentionally deferred.

## Questions for the initial investigation

1. How do construction time, memory, kick time, Hamiltonian time, Krylov work,
   and total propagation time grow for `l_max=2,4,6,8,10`?
2. How much more expensive is HD+ than H2+ when HD+ cannot use nuclear-exchange
   symmetry?
3. Which coupled Hamiltonian component dominates at each angular cutoff?
4. Do the parallel Numba Coulomb and dipole kernels benefit from additional
   CPU threads?
5. Do threaded BLAS operations in the radial kinetic actions provide an
   independent benefit?
6. What thread count minimizes wall time, and what thread count minimizes
   allocated CPU time, for each high-angular-cutoff case?
7. At what cutoff does memory, rather than runtime, become the practical
   serial-process limit?

The initial result should be a recommended non-MPI Olivia request for each
system and angular cutoff, together with evidence about which threading
mechanism provides the benefit.

## Fixed initial configuration

The first campaign uses:

- one process and no MPI launch;
- the coupled angular representation;
- H2+ and HD+;
- `l_max=2,4,6,8,10`, mapped to the coupled inputs described below;
- an analytical, laboratory-z-polarized `AxialDeltaKick` applied to the
  initial state;
- integrated kick strength `K=1e-3` a.u.;
- no subsequent external field;
- independent electronic and internuclear
  `CubicCAP(alpha=0.8, eta=1.0)` absorbers;
- SIA propagation;
- `dt=1/16` a.u.;
- `initial_time=0` and `final_time=10` a.u., giving 160 requested propagation
  steps;
- norm, physical-energy numerator, and total induced z-dipole numerator
  sampling after every accepted step;
- no periodic full-state snapshots or checkpoints;
- one final-state archive written after the timed propagation region;
- identical numerical tolerances at every thread count;
- at least three repetitions of every accepted timing point;
- untimed Numba and library warm-up before the measured propagation;
- retained outliers and recorded Olivia node placement;
- no specialized field-free propagation path.

The fixed radial grids are:

| coordinate | maximum | element width | points per element | elements | retained points |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronic `r` | 64 bohr | 4 bohr | 11 | 16 | 159 |
| internuclear `R` | 16 bohr | 4 bohr | 15 | 4 | 55 |

The radial coefficient grid therefore contains `159 * 55 = 8745` points per
retained coupled angular channel.

For both systems, a requested cutoff called `l_max` in this plan maps to

```text
l1_max = l2_max = l_max
L_max = 2*l_max
M_tot = 0
lambda_max = 2*l_max.
```

The coupled public interface does not expose independent `m_max` or `mu_max`
arguments. The underlying one-coordinate bases retain the complete magnetic
ranges through `abs(m_i) <= l_i <= l_max`, which is the coupled equivalent of
`m_max=l_max`. Internally the Coulomb multipole specification sets
`mu_max=lambda_max`. Thus the requested `m_max`, `lambda_max`, and `mu_max`
truncation is retained without passing nonexistent coupled-interface options.

The dynamics basis must include both inversion parities because the kick is a
rank-one odd-parity operator. Use:

- H2+: `pair_exchange_parity=+1`, `inversion_parity=None`;
- HD+: `pair_exchange_parity=None`, `inversion_parity=None`.

At the stated cutoffs, the current basis construction gives the following
complex128 state sizes:

| `l_max` | H2+ channels | H2+ coefficients | H2+ state | HD+ channels | HD+ coefficients | HD+ state |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 12 | 104940 | 1.60 MiB | 19 | 166155 | 2.54 MiB |
| 4 | 49 | 428505 | 6.54 MiB | 85 | 743325 | 11.34 MiB |
| 6 | 128 | 1119360 | 17.08 MiB | 231 | 2020095 | 30.82 MiB |
| 8 | 265 | 2317425 | 35.36 MiB | 489 | 4276305 | 65.25 MiB |
| 10 | 476 | 4162620 | 63.52 MiB | 891 | 7791795 | 118.89 MiB |

These are complex wavefunction coefficients only; SIA Krylov vectors,
Hamiltonian work arrays, angular tables, CAP arrays, and temporary construction
data increase the actual memory requirement substantially.

## Scientific workload families

### A. Hamiltonian-action benchmark

Apply the complete Hamiltonian repeatedly to a fixed normalized complex state.
For the initial non-MPI workload this isolates radial work, Coulomb work, and
CAP work from Krylov adaptivity. Time the projected dipole action separately
as part of the analytical kick; it is absent from the subsequent field-free
propagation.

For every case:

- perform untimed construction and compilation warm-up;
- verify the candidate action against the reference action;
- time enough actions to make the timed interval substantially longer than
  launch and timer noise;
- record radial, Coulomb, CAP, and complete-action profiles;
- record the projected axial dipole action independently.

This is a diagnostic benchmark, not an application performance result. Run it
inside the same one-process allocations as the dynamics calculations so that
component timings can explain the observed thread scaling.

### B. Analytical delta-kick response

Apply

```python
AxialDeltaKick(integrated_field=1.0e-3)
```

to a ground state and propagate with no subsequent external field.
This is a reproducible physical workload for absorption and linear-response
studies. It avoids pulse-shape evaluation while retaining the complete
two-radial field-free Hamiltonian, Krylov orthogonalization, absorber behavior,
and observable sampling.

Measure the kick separately from subsequent propagation. The initial campaign
uses the CAP/SIA variant exclusively and propagates all 160 requested steps to
`tfinal=10` a.u. A CAP-free SIL comparison is deferred.

### C. Driven laser dynamics

**Deferred until after the initial serial/threading campaign.**

Use the existing three-cycle, 800-nm, z-polarized `SineSquareLaser` H2+
workload in length gauge. This includes the time-dependent dipole action and is
representative of the established HHG use case.

The scaling pilot uses a fixed contiguous set of accepted steps rather than a
complete pulse. At least one final application run should cover the complete
pulse and requested post-pulse interval after its resource configuration has
been selected.

### D. Ground-state preparation

Davidson preparation is scientifically necessary but must not be mixed into
the real-time scaling measurements. A separate study should record:

- archive load and validation time;
- Hamiltonian and preconditioner construction time;
- Davidson solve time and iteration count;
- peak memory;
- archive write time.

The primary production workflow may continue to use serial Davidson unless a
measured case shows that it is a significant fraction of total campaign cost.
The timed dynamics region must begin after the initial state has been prepared
and the analytical kick has been applied. Construction, ground-state, and kick
times are nevertheless retained as separate measurements.

### E. Coupled-versus-uncoupled equivalence workload

**Deferred until after the initial serial/threading campaign.**

Use H2+ with matched radial grids and angular spaces. For the coupled basis set

```text
L_max = l1_max + l2_max
M_tot = 0
```

and retain the same physical inversion and nuclear-exchange sectors as in the
uncoupled calculation. Compare:

- retained angular-channel count;
- angular-table construction time and memory;
- complete Hamiltonian action time;
- kick time;
- accepted propagation-step time;
- total Hamiltonian calls and Krylov dimensions;
- per-rank and aggregate memory;
- final states after transformation into a common representation;
- physical observable time series.

This comparison must use equivalent finite spaces. An angularly truncated
coupled calculation is a separate scientific approximation and must not be
reported as a representation-level performance comparison.

## Initial calculation matrix

The scientific matrix contains ten configurations:

```text
systems = H2+, HD+
l_max   = 2, 4, 6, 8, 10
```

Every configuration uses the fixed grids, angular mapping, CAP, kick, SIA
integrator, time step, and final time specified above. Changes of radial box,
radial order, integrator, absorber, or representation do not belong in this
matrix.

Run every configuration first with all numerical thread pools restricted to
one thread. Thread sweeps are then concentrated where sufficient work exists:

| cutoff | required thread counts |
| ---: | --- |
| 2 | 1, 2, 4 |
| 4 | 1, 2, 4 |
| 6 | 1, 2, 4, 8 |
| 8 | 1, 2, 4, 8, 16 |
| 10 | 1, 2, 4, 8, 16; add 32 only if 16 threads still gives a material gain |

The thread count denotes allocated CPU cores for one process, not MPI ranks.
Each timing point must request a matching `--cpus-per-task` and must not
oversubscribe the allocation.

## Numerical controls

For thread comparisons, all thread counts must use identical physical and
numerical inputs:

- identical initial-state archive for a given system and cutoff;
- identical angular and radial bases;
- identical analytical kick;
- identical time step and final time;
- identical Krylov dimension, tolerances, refinement limit, and
  reorthogonalization setting;
- identical absorber;
- identical sampling policy;
- identical `radial_operator_mode="element-local"` setting;
- identical package commits and deployment.

The initial draft uses:

```text
delta-kick strength K = 1e-3 a.u.
time step             = 1/16 a.u.
final time            = 10 a.u.
maximum Krylov size   = 20
Krylov rtol           = 1e-10
Krylov atol           = 0
maximum refinements   = 8
```

The CAP candidate follows the previous H2+ studies:

```text
CubicCAP(alpha=0.8, eta=1.0)
```

on both radial coordinates. These settings are fixed for the initial matrix.

These are performance inputs, not asserted convergence parameters.

Use `K=1e-3` a.u. for every kick. This is intended as a weak linear-response
perturbation and does not change the operator dimensions across the matrix.

Use one converged coupled Davidson ground-state archive for every system and
`l_max`. Construct it in the `L=0`, `M_tot=0`, even-inversion sector and embed
it into the full-parity dynamics basis. The job may use a
`compute-if-missing` policy, but ground-state construction must finish before
the timed kick and propagation regions. Record whether the state was loaded or
computed, together with Davidson convergence, residual, iteration, time, and
memory data.

At every accepted propagation step, record at least:

- time and accepted-step index;
- norm `N(t)`;
- the raw total z-dipole numerator;
- the raw induced total z-dipole numerator relative to its post-kick value at
  `t=0`;
- the raw physical-energy numerator;
- SIA step and Krylov diagnostics.

Never normalize an observable during sampling. The CAP is part of the
propagation generator but not the physical energy operator, so the stored
energy numerator is

```text
energy_numerator(t) = <psi(t)|H_0|psi(t)>,
```

with the CAP excluded. Because no external field follows the kick, `H_0` is
also the gauge-covariant physical energy operator for this propagation stage.

The stored induced dipole numerator is

```text
d_ind,z_numerator(t) = d_z,numerator(t) - d_z,numerator(0).
```

Conditional quantities are postprocessing results obtained from the sampled
numerators and norm. For example,

```text
d_ind,z,conditional(t)
    = d_z,numerator(t)/N(t) - d_z,numerator(0)/N(0).
energy_conditional(t) = energy_numerator(t)/N(t).
```

This preserves unnormalized CAP-loss semantics in the primary archive while
retaining everything required to construct conditional observables later.

Do not write periodic snapshots or checkpoints. After propagation timing has
stopped, write one final-state archive with its physical time, accepted-step
index, norm, representation and system metadata, numerical settings, kick and
CAP metadata, code provenance, and checksum. Record final-state write time and
file size separately from propagation time.

## Experimental sequence

### Stage 0: freeze provenance and create a preflight report

Before submitting timed jobs, record:

- grid-tdse, grid-lib, integrator-lib, electromagnetic-fields, and parameter
  package commits;
- Python, NumPy, SciPy, Numba, threadpoolctl, and relevant compiled-library
  versions;
- Olivia modules and container deployment identifier;
- CPU model, node memory, CPU binding, NUMA placement, and thread-pool state;
- state shapes and coefficient counts;
- angular-channel and coupling-table counts;
- the relation between every coupling-table count and the configured
  `max_coupling_pairs` safety limit;
- estimated state, Krylov work-buffer, and final-state archive sizes;
- construction RSS and process peak;
- active Numba threading layer and `numba.get_num_threads()`;
- effective BLAS/OpenMP thread pools reported by `threadpoolctl`.

Reject a configuration before propagation if its measured construction peak
leaves insufficient memory headroom for Krylov vectors and communication
buffers.

### Stage 1: numerical equivalence gate

For every system and angular cutoff:

1. verify the ground-state archive and embedding metadata;
2. verify the initial norm and the norm after the analytical kick;
3. compare a complete Hamiltonian action with its component sum;
4. verify one accepted SIA step;
5. verify the complete 160-step final state, norm, CAP loss, and observables;
6. verify that repeated one-thread runs agree numerically;
7. verify that threaded results agree with the one-thread result within a
   documented tolerance.

Performance data are accepted only after this gate passes.

### Stage 2: one-thread angular-growth baseline

Run all ten system/cutoff configurations with one allocated CPU core and every
numerical thread pool restricted to one. Record:

- representation and coupling-table construction time;
- construction RSS and process peak;
- initial-state load and embedding time;
- analytical-kick time and Krylov diagnostics;
- Hamiltonian component times;
- total propagation time and time per accepted step;
- Hamiltonian calls, Krylov dimensions, refinements, and error estimates;
- sampling time;
- final norm, CAP loss, and validation observables.

The coupled interface currently uses its reference kernels and rejects the
uncoupled-only Coulomb, radial, and dipole kernel selectors. Do not include
those selectors in this campaign.

### Stage 3: one-process thread investigation

Separate the available threading mechanisms rather than changing all thread
variables at once:

| mode | `NUMBA_NUM_THREADS` | BLAS/OpenMP library threads | purpose |
| --- | ---: | ---: | --- |
| single | 1 | 1 | reference |
| Numba-only | `T` | 1 | coupled Coulomb and dipole `prange` kernels |
| BLAS-only | 1 | `T` | radial dense/library operations |
| combined | `T` | `T` | end-to-end candidate after isolated tests |

For each `T`, request one task and exactly `T` CPUs per task. Record the
effective pools inside the job. The combined mode does not request `2*T`
cores: the Numba and BLAS regions are expected to execute sequentially, but
the run must check for nested or accidental oversubscription.

Use the full thread-count matrix for the Numba-only mode. Test BLAS-only and
combined modes first at `l_max=6,8,10`; omit lower cutoffs if the pilot shows
no measurable work to amortize thread startup. If BLAS-only is ineffective
with the element-local radial operator, retain that negative result rather
than repeating it throughout the matrix.

For thread count `T`, report

```text
speedup(T)    = median_time(1) / median_time(T)
efficiency(T) = speedup(T) / T
```

Report both wall time and allocated CPU time. A faster run can still be a poor
resource recommendation if its core-time cost rises disproportionately.

### Stage 4: initial recommendations

For every system and cutoff, select:

- the lowest-wall-time thread count;
- the lowest-core-time thread count;
- a balanced recommended Olivia request;
- expected memory and time limit with safety margins.

The recommendation must identify whether its speedup comes from Numba, BLAS,
or both. It must also say when one thread remains preferable.

## Deferred MPI and extended-application investigation

The following stages remain part of the longer performance roadmap but are
not part of the initial job matrix.

### Later stage A: process-grid shape study

At fixed rank counts, compare every practical factorization

```text
P_r x P_R, with P_r <= number of r elements
             and P_R <= number of R elements.
```

Begin with 4, 8, and 16 ranks where supported. Record:

- local element and coefficient imbalance;
- radial halo and weak-contribution exchange time in each direction;
- Coulomb and angular work;
- Krylov reduction time;
- total propagation time.

Choose the process grid by measured propagation time, not by geometrical
squareness alone. Retain one deliberately unfavorable layout as evidence of
the shape sensitivity.

### Later stage B: single-node MPI strong scaling

Use fixed problem definitions and rank sequences selected from

```text
1, 2, 4, 8, 16, 32, 64, 128
```

subject to element topology, memory, and useful local tile size. Each point
uses the same fixed number of accepted steps and at least three repetitions.

For rank count `P`, report

```text
speedup(P)    = median_time(1) / median_time(P)
efficiency(P) = speedup(P) / P
```

and also report core-hours per accepted step. The recommended rank count is
the smallest allocation near the minimum wall time unless a larger allocation
provides a material wall-time reduction at acceptable billing cost.

### Later stage C: multi-node decision gate

Do not automatically extend every single-node curve across nodes. Select at
most two scientifically representative future workloads. A multi-node run
proceeds only if:

- the process topology exposes enough useful radial tiles;
- per-node memory or single-node throughput justifies another node;
- the best single-node run has not already saturated in collectives;
- the projected billed cost is recorded before submission.

Run a network correctness point first. Then run only rank counts with a
credible scaling hypothesis. Record actual allocation, ranks per node, NUMA
placement, and Slurm billing information.

### Later stage D: computational weak scaling

Keep element width, points per element, angular cutoff, and approximately the
same number of radial element tiles per rank while increasing both the radial
domain and process grid. This is a computational weak-scaling experiment; it
changes the physical box and must not be presented as a convergence series.

Report time per Hamiltonian action, time per accepted step, per-rank memory,
collective time, and halo exchange time. A separate angular-growth experiment
holds the radial grids fixed and varies `l_max`.

### Later stage E: coupled-versus-uncoupled comparison

After choosing one good MPI layout for each representation, compare matched
coupled and uncoupled spaces on selected small and production grids. Use the
same radial state after an explicit representation transformation, the same
analytical kick, and the same propagation settings.

This stage determines whether either representation has a practical advantage
from channel ordering, sparse coupling structure, memory access, or setup
cost. It must not assume an advantage from the outset.

### Later stage F: extended application validation

Run complete calculations only after selecting configurations from the pilot
stages:

1. one delta-kick response calculation long enough to resolve the intended
   spectral range;
2. one complete three-cycle 800-nm H2+ calculation;
3. one HD+ short calculation to exercise the heteronuclear dipole and absence
   of nuclear-exchange adaptation;
4. one D2+ short calculation to verify mass-dependent reuse.

Report scientific checks and total resource use. These runs validate the job
recommendations; they are not additional points on a clean scaling curve.

## Separate CF4-versus-SIA investigation

CF4-SIA and SIA must not be compared merely at the same time step. The
fourth-order method should be tested separately at matched physical error and
at matched computational budget.

Use one representative driven-pulse calculation and construct a fine
reference. For each method and candidate time step record:

- final phase-aligned state error at a pulse endpoint where comparison is
  meaningful;
- norm and CAP loss;
- gauge-covariant energy and dipole errors;
- Hamiltonian calls;
- accepted and rejected Krylov refinements;
- wall time and billed core time.

Compare:

1. SIA versus CF4-SIA at matched final-state or observable error;
2. SIA versus CF4-SIA at matched Hamiltonian-call budget;
3. sensitivity to Krylov tolerance;
4. serial and one selected MPI configuration.

This study should select an accuracy/cost envelope, not declare one integrator
universally superior.

This comparison is explicitly deferred until after the initial coupled SIA
threading investigation.

## Deferred additional sampling and I/O experiments

The initial baseline samples norm, raw total-dipole and induced-dipole
numerators, the raw physical-energy numerator, and SIA diagnostics every
accepted step. It writes one final-state archive outside the propagation timing
and performs no periodic full-state output. Later studies may compare:

- the initial baseline cadence with reduced dipole-sampling cadence;
- the complete physical observable contract;
- radial marginals on selected cadences;
- rolling checkpoints at a realistic interval;
- periodic wavefunction snapshots.

Current Cartesian-MPI full-state output gathers to rank zero. Until scalable
distributed I/O is implemented, do not run production-sized high-frequency
snapshot tests that can exceed root memory. Record file sizes, write bandwidth,
maximum root RSS, synchronization time, and restart-read time.

## Measurements and statistics

Each accepted run should produce one machine-readable record containing at
least the applicable fields below. MPI-only fields remain empty in the initial
non-MPI investigation:

```text
campaign_id
timestamp
commit and deployment identifiers
system and CODATA set
representation and symmetry sector
radial grid definitions
angular cutoffs and channel counts
integrator and Krylov settings
absorber and field or kick metadata
kernel selectors
nodes, ranks, ranks per node, process grid, and thread settings
node list and binding
global/local coefficient counts
setup, warm-up, kick, propagation, sampling, and I/O times
time per accepted step
Hamiltonian calls and component times
halo, weak-exchange, and collective times and counts
Krylov dimensions, refinements, and error estimates
per-rank and aggregate memory
norm, energy, state, and observable validation errors
Slurm elapsed time, allocated CPU time, and billed resource use
status and notes
```

For repeated points report the median, minimum, maximum, and median absolute
deviation. Do not silently discard an outlier. A point may be marked as a node
outlier only when repeated evidence identifies anomalous node performance and
the numerical result remains correct.

Timed jobs should be serialized unless simultaneous execution is itself the
subject of the experiment.

## Data and artifact layout

This investigation belongs in the private data workspace, separate from both
the public repository and the physics-oriented `nbo_hhg` archive. Use the
following local archive root after synchronization from Olivia:

```text
grid-tdse-data/
    nbo_performance/
        coupled_axial_delta_kick/
            olivia/
                <campaign_id>/
```

Use an immutable campaign identifier that describes the fixed matrix, for
example

```text
coupled-kick-r64-R16-l2-10-YYYYMMDD-HHMMSS
```

On Olivia, keep small control files and scripts under

```text
${GRID_TDSE_ROOT}/studies/nbo_performance/<campaign_id>/
```

and runtime output under

```text
${GRID_TDSE_WORK_ROOT}/studies/nbo_performance/<campaign_id>/
```

The synchronized local campaign should retain:

```text
<campaign_id>/
    README.md
    manifest/
    configurations/
    scripts/
    submissions/
    logs/
    ground_states/
        H2p/lmax-XX/
        HDp/lmax-XX/
    runs/
        <system>/
            lmax-XX/
                <thread-mode>/
                    threads-XX/
                        repetition-XX/
                            observables.npz
                            final-state.npz
                            result.json
                            slurm.out
    summaries/
    analysis/
    checksums/
```

Ground-state archives are shared by all thread repetitions at the same system
and cutoff. Every run directory is immutable once its checksum has been
recorded. Generate a campaign-level `SHA256SUMS` after synchronization and
verify it before analysis.

Private Olivia paths, project identifiers, raw scheduler output, and exact job
scripts remain in this private archive. They must not be added directly to the
public repository.

The public repository should receive only reusable generic drivers, sanitized
machine-readable summaries, and a written benchmark report. Every copied
artifact must be checked for usernames, project numbers, private paths, and
other site-specific information.

## Decision gates and stopping rules

A thread series stops when any of the following applies:

- median wall time no longer improves materially;
- thread efficiency has fallen below the campaign threshold for two
  successive thread-count increases;
- aggregate billed CPU time increases sharply without a justified reduction
  in turnaround time;
- construction or Krylov memory lacks safe headroom;
- numerical equivalence fails;
- node placement makes the measurement incomparable and a clean repeat is
  unavailable.

Stop after two consecutive thread-count increases below 50% efficiency or
below 10% incremental wall-time improvement. The conditional 32-thread
`l_max=10` point requires at least a 10% gain from 8 to 16 threads.

## Expected deliverables

The initial investigation delivers:

1. validated H2+ and HD+ manifests for every angular cutoff;
2. one-thread angular-growth, component, and memory profiles;
3. Numba-only, BLAS-only, and combined thread-scaling measurements;
4. wall-time, efficiency, core-time, and memory comparisons;
5. recommended Olivia CPU, memory, and time requests for every system/cutoff;
6. one validated final-state archive for every accepted timing run;
7. sanitized machine-readable summaries and a written interpretation.

Later campaigns may add MPI process-grid recommendations, multi-node curves,
weak scaling, uncoupled comparisons, I/O measurements, driven laser dynamics,
and the CF4-SIA accuracy/cost report.

## Initial-plan readiness

The integrated kick, per-cutoff ground-state policy, sampling contract,
final-state retention, and thread-series stopping rules are settled. The next
revision can specify the serial benchmark driver, Olivia job-array structure,
resource preflight, and machine-readable result schema without changing the
scientific matrix.
