"""Infinite-nuclear-mass helium ground state by imaginary-time propagation."""

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    TwoElectronAtomGroundState,
)

radial_grid = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    n_points=[11, 11, 11, 11, 11],
)

calculation = TwoElectronAtomGroundState(
    "He",  # Use "Be2+" for the beryllium dication.
    grid=radial_grid,
    # Symmetric spatial exchange corresponds to the spin-singlet sector.
    inversion_parity=+1,
    electron_exchange_parity=+1,
    # Coordinate angular basis and independent Coulomb-multipole truncation.
    l_max=2,
    m_max=2,
    lambda_max=4,
    mu_max=4,
    # Deterministic symmetry-adapted product-state guess.
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
    linear_max_iterations=None,
    gmres_restart=None,
    progress_interval=10,
    use_preconditioner=True,
    tile_size=2048,
)

result = calculation.solve()
last = result.history[-1]

print(f"System:                   {result.parameters.name}")
print(f"Nuclear charge:           {result.parameters.nuclear_charge:.0f}")
print(f"Spin multiplicity:        {result.spin_multiplicity}")
print(f"Converged:                {result.converged}")
print(f"Iterations:               {result.iterations}")
print(f"Energy:                   {result.energy:.12f} Eh")
print(f"Energy change:            {last.energy_change:.3e} Eh")
print(f"State change:             {last.state_change:.3e}")
print(f"Eigenstate residual:      {last.residual_norm:.3e}")
print(f"<L_total^2>:              {result.angular_momentum.total_l_squared:.12f}")
print(f"<L_total,z>:              {result.angular_momentum.total_l_z:.12f}")
print(f"<inversion>:              {result.inversion_expectation:.12f}")
print(f"<electron exchange>:      {result.electron_exchange_expectation:.12f}")
