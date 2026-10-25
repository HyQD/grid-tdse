"""Infinite-nuclear-mass helium ground state by imaginary-time propagation."""

from pathlib import Path

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    TwoElectronAtomGroundState,
)

ATOM = "He"  # Use "Be2+" for the beryllium dication.
INVERSION_PARITY = +1
ELECTRON_EXCHANGE_PARITY = +1
INITIAL_STATE = "product"
STATE_TOLERANCE = 1.0e-8
l_max = 2

radial_grid = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    n_points=[11, 11, 11, 11, 11],
)

# Retain only m1 + m2 = M_tot. Use None for the full legacy representation.
M_tot = 0

# Follow the legacy ground.py filename convention. The archive itself stores
# the complete grid, angular representation, physical parameters, and
# diagnostics; the compact filename is only a human-readable summary.
atom_label = ATOM.replace("+", "p")
symmetry_label = "s" if INVERSION_PARITY == 1 else "a"
symmetry_label += "p" if ELECTRON_EXCHANGE_PARITY == 1 else "o"
grid_label = f"{len(radial_grid.n_points)}e{radial_grid.n_points[0]}"
extent_label = f"{radial_grid.nodes[-1]:g}"
initial_state_label = "prod" if INITIAL_STATE == "product" else INITIAL_STATE
GROUND_STATE_PATH = Path(__file__).resolve().parent / "ground_states" / (
    f"{atom_label}__{symmetry_label}_{grid_label}_{extent_label}_{l_max}__"
    f"{initial_state_label}_{STATE_TOLERANCE:.0e}.npz"
)
GROUND_STATE_OVERWRITE = False

calculation = TwoElectronAtomGroundState(
    ATOM,
    grid=radial_grid,
    # Symmetric spatial exchange corresponds to the spin-singlet sector.
    inversion_parity=INVERSION_PARITY,
    electron_exchange_parity=ELECTRON_EXCHANGE_PARITY,
    # Coordinate angular basis and independent Coulomb-multipole truncation.
    l_max=l_max,
    m_max=l_max,
    lambda_max=2 * l_max,
    mu_max=2 * l_max,
    M_tot=M_tot,
    # Deterministic symmetry-adapted product-state guess.
    initial_state=INITIAL_STATE,
    # Imaginary-time Crank--Nicolson iteration.
    time_step=0.25,
    max_iterations=2000,
    energy_tolerance=1.0e-10,
    state_tolerance=STATE_TOLERANCE,
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

if not result.converged:
    raise RuntimeError("refusing to save an unconverged ground state")
saved_path = result.save(
    GROUND_STATE_PATH,
    overwrite=GROUND_STATE_OVERWRITE,
)

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
print(f"Saved state:              {saved_path}")
