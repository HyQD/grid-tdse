# Spherical one-radial off-axis performance qualification

## Purpose

`benchmarks/spherical_one_radial_off_axis_performance.py` qualifies the
serial kernels used by arbitrary-orientation, clamped-nuclei H2+ dynamics
before an MPI decomposition is designed. Its default geometry has `R=2` bohr
and a 45-degree molecular axis relative to the laboratory `z` field. The
compact defaults are performance-development inputs, not scientific
convergence parameters.

For each requested `(r_max,l_max)` pair the benchmark records:

- Hamiltonian construction and repeated field-free action timings;
- full-angular Davidson time, iterations, actions, and residual;
- every default scalar or Cartesian-vector observable separately;
- short SIA Krylov dimensions and operator-call counts;
- short preconditioned Crank--Nicolson iteration counts and residuals;
- active radial points, ordered angular-channel count, potential components,
  sparse-coupling nonzeros, complex state bytes, operator-array bytes, RSS,
  CPU affinity, thread-pool state, package versions, and Git revision.

## Thread qualification

Run each thread setting in a separate serial process:

```console
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/spherical_one_radial_off_axis_performance.py \
    --l-max 2 4 6 8 \
    --threads 1 \
    --result-json one-radial-t1.json

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python benchmarks/spherical_one_radial_off_axis_performance.py \
    --l-max 2 4 6 8 \
    --threads 2 \
    --result-json one-radial-t2.json

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python benchmarks/spherical_one_radial_off_axis_performance.py \
    --l-max 2 4 6 8 \
    --threads 4 \
    --result-json one-radial-t4.json
```

`--threads` applies a runtime `threadpoolctl` limit and records the effective
pools. Setting the environment as shown also controls libraries before Python
imports them. The benchmark remains one Python process and does not initialize
MPI.

## Radial and angular sweep

Use more than one radial extent to separate radial from angular growth:

```console
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/spherical_one_radial_off_axis_performance.py \
    --r-max 8 16 32 \
    --l-max 2 4 6 8 \
    --threads 1 \
    --result-json one-radial-axis-sweep.json
```

Compare action, observable, Davidson, and propagation growth at fixed
`active_radial_points` and at fixed `angular_channels`. Interpret timings only
within the same reserved hardware and software environment. Do not compare
overlapping jobs or combine results collected with different CPU affinity.

The first local smoke sweep used 29 active radial points. From `l_max=2` to
`l_max=8`, angular channels grew from 9 to 81, Hamiltonian-action time grew
about 16-fold, and short SIA time grew about 17-fold. At `l_max=8`, increasing
the radial grid from 29 to 101 points increased action time about 2.3-fold and
SIA time about 2.4-fold. Two and four BLAS threads provided no measurable
benefit on these compact cases. These local results identify angular work as
the first distribution target, while the multiplicative state shape means a
two-dimensional radial/angular layout should be reconsidered for production
grids. Olivia measurements, rather than these smoke timings, are the decision
gate.
