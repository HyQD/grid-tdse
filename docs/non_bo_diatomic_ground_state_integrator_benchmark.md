# Non-BO diatomic ground-state integrator comparison

The serial benchmark
[`non_bo_diatomic_ground_state_integrators.py`](../benchmarks/non_bo_diatomic_ground_state_integrators.py)
compares Crank--Nicolson, short-iterative Lanczos, and short-iterative Arnoldi
for normalized imaginary-time propagation.

The supported systems are H2+, HD+, and D2+. Every method starts from the
same deterministic product state and uses the same FEM-DVR, angular,
Coulomb, symmetry, `M_tot=0`, and CODATA configuration.

Run the lightweight grid locally with:

```bash
python benchmarks/non_bo_diatomic_ground_state_integrators.py --grid small
```

Run the moderate H2+ case with:

```bash
python benchmarks/non_bo_diatomic_ground_state_integrators.py \
    --systems H2+ --grid moderate
```

The benchmark records:

- convergence status, iterations, and accumulated imaginary time;
- energy, energy change, state change, and eigenstate residual;
- Hamiltonian calls and integrator-specific iteration diagnostics;
- elapsed wall-clock time;
- state-array storage and process peak resident memory;
- inversion, pair-exchange, total-angular-momentum expectations;
- energy and phase-aligned state differences relative to CN when CN is run.

Use `--json-output PATH` to save the complete configuration and results. The
file is not overwritten unless `--overwrite` is supplied.

Process peak resident memory is cumulative when several methods are run in
one Python process. For independent memory measurements, invoke the benchmark
separately for each integrator, for example:

```bash
python benchmarks/non_bo_diatomic_ground_state_integrators.py \
    --systems H2+ --integrators sil --grid moderate
```

The small preset is intended for development and regression checks. The
moderate preset is a serial performance reference, not a converged production
calculation. Radial extent, FEM-DVR order, angular truncation, imaginary-time
step, and all convergence tolerances must be varied before drawing physical
conclusions.
