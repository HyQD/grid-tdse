"""Non-Born--Oppenheimer H2+ ground state by imaginary-time propagation."""

from pathlib import Path

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
)

SYSTEM = "H2+"
INVERSION_PARITY = +1
PAIR_EXCHANGE_PARITY = +1
INITIAL_STATE = "product"
STATE_TOLERANCE = 1.0e-8
RADIAL_COULOMB_METHOD = "poisson"
RADIAL_OPERATOR_MODE = "element-local"
INTEGRATOR = "crank-nicolson"  # Choose "crank-nicolson", "sil", or "sia".
USE_PRECONDITIONER = INTEGRATOR == "crank-nicolson"

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

# Retain only m_e + m_R = M_tot. Use None for the full legacy representation.
M_tot = 0
l_max = 2

# Follow the legacy ground.py filename convention. The archive itself stores
# the complete grid, angular representation, physical parameters, and
# diagnostics; the compact filename is only a human-readable summary.
system_label = SYSTEM.replace("+", "p")
symmetry_label = "s" if INVERSION_PARITY == 1 else "a"
if PAIR_EXCHANGE_PARITY is not None:
    symmetry_label += "p" if PAIR_EXCHANGE_PARITY == 1 else "o"
grid_label = (
    f"{len(electronic_grid.n_points)}e{electronic_grid.n_points[0]}-"
    f"{len(internuclear_grid.n_points)}e{internuclear_grid.n_points[0]}"
)
extent_label = (
    f"{electronic_grid.nodes[-1]:g}-{internuclear_grid.nodes[-1]:g}"
)
initial_state_label = "prod" if INITIAL_STATE == "product" else INITIAL_STATE
GROUND_STATE_PATH = Path(__file__).resolve().parent / "ground_states" / (
    f"{system_label}__{symmetry_label}_{grid_label}_{extent_label}_{l_max}__"
    f"{initial_state_label}_{STATE_TOLERANCE:.0e}.npz"
)
GROUND_STATE_OVERWRITE = False

calculation = NonBornOppenheimerDiatomicIonGroundState(
    SYSTEM,
    electronic_grid=electronic_grid,
    internuclear_grid=internuclear_grid,
    # Native symmetry inputs: symmetric total inversion and symmetric
    # proton-pair exchange.  For HD+, pair_exchange_parity must be None.
    inversion_parity=INVERSION_PARITY,
    pair_exchange_parity=PAIR_EXCHANGE_PARITY,
    symmetry_adapted_basis=True,
    # Coordinate angular basis and independent Coulomb-multipole truncation.
    l_max=l_max,
    m_max=l_max,
    lambda_max=2*l_max,
    mu_max=2*l_max,
    M_tot=M_tot,
    radial_coulomb_method=RADIAL_COULOMB_METHOD,
    radial_operator_mode=RADIAL_OPERATOR_MODE,
    # Reproducible particle masses and deterministic product-state guess.
    codata_set="2022",
    initial_state=INITIAL_STATE,
    # Representation-independent imaginary-time iteration.
    time_step=0.25,
    max_iterations=15000,
    energy_tolerance=1.0e-10,
    state_tolerance=STATE_TOLERANCE,
    residual_tolerance=1.0e-8,
    integrator=INTEGRATOR,
    # Crank--Nicolson linear-solver options.
    linear_solver="bicgstab",
    linear_rtol=0.0,
    linear_atol=1.0e-12,
    linear_max_iterations=None,
    gmres_restart=None,
    # SIL/SIA exponential-Krylov options.
    max_krylov_dimension=20,
    krylov_rtol=1.0e-10,
    krylov_atol=0.0,
    max_krylov_refinements=8,
    krylov_reorthogonalize=True,
    krylov_breakdown_tolerance=1.0e-14,
    progress_interval=10,
    use_preconditioner=USE_PRECONDITIONER,
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
print(f"Particles:                {result.parameters.particles}")
print(f"ITP integrator:           {result.integrator}")
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
print(f"Saved state:              {saved_path}")
