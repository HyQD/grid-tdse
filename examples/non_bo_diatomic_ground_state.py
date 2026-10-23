"""Non-Born--Oppenheimer H2+ ground state by imaginary-time propagation."""

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
)

# The electron coordinate is measured from the nuclear center of mass.
electronic_grid = FEMDVRGridSpec(
    nodes=[0.0, 4.0, 8.0, 16.0],
    n_points=[11, 11, 11],
)

# The internuclear coordinate has its own extent and resolution, but uses the
# same FEM-DVR grid family.
internuclear_grid = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 4.0, 6.0, 8.0],
    n_points=[21, 21, 21, 21],
)

calculation = NonBornOppenheimerDiatomicIonGroundState(
    "H2+",
    electronic_grid=electronic_grid,
    internuclear_grid=internuclear_grid,
    # Native symmetry inputs: symmetric total inversion and symmetric
    # proton-pair exchange.  For HD+, pair_exchange_parity must be None.
    inversion_parity=+1,
    pair_exchange_parity=+1,
    symmetry_adapted_basis=True,
    # Coordinate angular basis and independent Coulomb-multipole truncation.
    l_max=2,
    m_max=2,
    lambda_max=4,
    mu_max=4,
    # Reproducible particle masses and deterministic product-state guess.
    codata_set="2022",
    initial_state="product",
    # Imaginary-time Crank--Nicolson iteration.
    time_step=0.25,
    max_iterations=2000,
    energy_tolerance=1.0e-10,
    state_tolerance=1.0e-8,
    residual_tolerance=1.0e-8,
    linear_solver="bicgstab",
    linear_rtol=0.0,
    linear_atol=1.0e-12,
    progress_interval=10,
    use_preconditioner=True,
)

result = calculation.solve()
last = result.history[-1]

print(f"System:                   {result.parameters.name}")
print(f"Particles:                {result.parameters.particles}")
print(f"Converged:                {result.converged}")
print(f"Iterations:               {result.iterations}")
print(f"Energy:                   {result.energy:.12f} Eh")
print(f"Energy change:            {last.energy_change:.3e} Eh")
print(f"State change:             {last.state_change:.3e}")
print(f"Eigenstate residual:      {last.residual_norm:.3e}")
print(f"<L_total^2>:              {result.angular_momentum.total_l_squared:.12f}")
print(f"<L_total,z>:              {result.angular_momentum.total_l_z:.12f}")
print(f"<inversion>:              {result.inversion_expectation:.12f}")
print(f"<pair exchange>:          {result.pair_exchange_expectation:.12f}")
