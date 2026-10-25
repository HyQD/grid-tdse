"""Propagate helium in a one-cycle, z-polarized sine-square laser pulse."""

from dataclasses import asdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields import SineSquareLaser
from matter_field_params import wavelength_nm_to_omega_au

from grid_tdse.real_time import RealTimePropagator
from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    TwoElectronAtomGroundState,
    TwoRadialAxialDipoleInteraction,
    TwoRadialRealTimePreconditioner,
    two_radial_compatibility_metadata,
)


# Select either "length" or "velocity". The laser and spatial interaction
# must use the same gauge.
FieldGauge = Literal["length", "velocity"]
FIELD_GAUGE: FieldGauge = "length"

# Five 4-bohr finite elements, each with 11 Gauss--Lobatto points.
RADIAL_GRID = FEMDVRGridSpec(
    nodes=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
    n_points=[11, 11, 11, 11, 11],
)

# Shared angular basis and Coulomb-multipole truncation.
L_MAX = 2
M_MAX = L_MAX
LAMBDA_MAX = 2 * L_MAX
MU_MAX = LAMBDA_MAX

# Retain only m1 + m2 = M_tot. Use None for the full legacy representation.
M_tot = 0

# One-cycle sine-square pulse. The field vanishes after the optical cycle,
# while propagation continues until FINAL_TIME.
FIELD_STRENGTH = 0.001
WAVELENGTH_NM = 400.0
FREQUENCY_CODATA_SET = "2022"
OMEGA = wavelength_nm_to_omega_au(
    WAVELENGTH_NM,
    codata_set=FREQUENCY_CODATA_SET,
)
N_CYCLES = 1.0
PHASE = 0.0
PULSE_START = 0.0
PULSE_DURATION = 2.0 * np.pi * N_CYCLES / OMEGA

# Real-time propagation and Krylov-solver controls.
INITIAL_TIME = 0.0
FINAL_TIME = 15.0
TIME_STEP = 0.25
REAL_TIME_LINEAR_SOLVER = "bicgstab"
REAL_TIME_LINEAR_RTOL = 0.0
REAL_TIME_LINEAR_ATOL = 1.0e-10
REAL_TIME_LINEAR_MAX_ITERATIONS = None
REAL_TIME_GMRES_RESTART = None
REAL_TIME_PROGRESS_INTERVAL = 10
CHECKPOINT_PATH = "tdse_he_atom.chkpt"
CHECKPOINT_INTERVAL = 10
CHECKPOINT_OVERWRITE = True

# Ground-state imaginary-time propagation controls.
GROUND_STATE_TIME_STEP = 0.25
GROUND_STATE_MAX_ITERATIONS = 2_000
GROUND_STATE_ENERGY_TOLERANCE = 1.0e-10
GROUND_STATE_STATE_TOLERANCE = 1.0e-8
GROUND_STATE_RESIDUAL_TOLERANCE = 1.0e-8
GROUND_STATE_LINEAR_SOLVER = "bicgstab"
GROUND_STATE_LINEAR_RTOL = 0.0
GROUND_STATE_LINEAR_ATOL = 1.0e-12
GROUND_STATE_LINEAR_MAX_ITERATIONS = None
GROUND_STATE_GMRES_RESTART = None
GROUND_STATE_PROGRESS_INTERVAL = 10
USE_GROUND_STATE_PRECONDITIONER = True
TILE_SIZE = 2048


ground_state_calculation = TwoElectronAtomGroundState(
    "He",
    grid=RADIAL_GRID,
    # The singlet ground state has even inversion and symmetric spatial
    # electron exchange. The full one-coordinate angular bases nevertheless
    # retain both even and odd l channels for the subsequent dipole coupling.
    inversion_parity=+1,
    electron_exchange_parity=+1,
    l_max=L_MAX,
    m_max=M_MAX,
    lambda_max=LAMBDA_MAX,
    mu_max=MU_MAX,
    M_tot=M_tot,
    initial_state="product",
    time_step=GROUND_STATE_TIME_STEP,
    max_iterations=GROUND_STATE_MAX_ITERATIONS,
    energy_tolerance=GROUND_STATE_ENERGY_TOLERANCE,
    state_tolerance=GROUND_STATE_STATE_TOLERANCE,
    residual_tolerance=GROUND_STATE_RESIDUAL_TOLERANCE,
    linear_solver=GROUND_STATE_LINEAR_SOLVER,
    linear_rtol=GROUND_STATE_LINEAR_RTOL,
    linear_atol=GROUND_STATE_LINEAR_ATOL,
    linear_max_iterations=GROUND_STATE_LINEAR_MAX_ITERATIONS,
    gmres_restart=GROUND_STATE_GMRES_RESTART,
    progress_interval=GROUND_STATE_PROGRESS_INTERVAL,
    use_preconditioner=USE_GROUND_STATE_PRECONDITIONER,
    tile_size=TILE_SIZE,
)
ground_state = ground_state_calculation.solve()
if not ground_state.converged:
    raise RuntimeError(
        "helium ground-state calculation did not converge: "
        f"{ground_state.termination_reason}"
    )

laser = SineSquareLaser(
    E0=FIELD_STRENGTH,
    omega=OMEGA,
    ncycles=N_CYCLES,
    gauge=FIELD_GAUGE,
    phase=PHASE,
    t0=PULSE_START,
)
dipole_interaction = TwoRadialAxialDipoleInteraction.for_two_electron_atom(
    ground_state.representation,
    ground_state.parameters,
    gauge=FIELD_GAUGE,
    fields=laser,
)

# A unit-amplitude length-gauge interaction for two electrons applies the
# total position operator z1 + z2. It is independent of the propagation gauge.
total_position_z = TwoRadialAxialDipoleInteraction.for_two_electron_atom(
    ground_state.representation,
    ground_state.parameters,
    gauge="length",
    fields=lambda time: 1.0,
)


def time_dependent_hamiltonian(state, time):
    """Apply the field-free and z-polarized laser Hamiltonian."""

    return ground_state_calculation.hamiltonian(state) + dipole_interaction(
        state,
        time,
    )


def total_z_expectation(state, time):
    """Return the unnormalized expectation of z1 + z2."""

    del time
    action = total_position_z(state, 0.0)
    value = ground_state.representation.inner_product(state, action)
    if abs(value.imag) > 1.0e-10 * max(1.0, abs(value.real)):
        raise RuntimeError(
            "the total z expectation acquired a significant imaginary part"
        )
    return value.real


propagator = RealTimePropagator(
    initial_time=INITIAL_TIME,
    final_time=FINAL_TIME,
    time_step=TIME_STEP,
    linear_solver=REAL_TIME_LINEAR_SOLVER,
    linear_rtol=REAL_TIME_LINEAR_RTOL,
    linear_atol=REAL_TIME_LINEAR_ATOL,
    linear_max_iterations=REAL_TIME_LINEAR_MAX_ITERATIONS,
    gmres_restart=REAL_TIME_GMRES_RESTART,
    progress_interval=REAL_TIME_PROGRESS_INTERVAL,
)
real_time_preconditioner = TwoRadialRealTimePreconditioner(
    ground_state.representation,
    propagator.time_step,
)
compatibility_metadata = two_radial_compatibility_metadata(
    ground_state.representation,
    physical_state={
        "system_family": "two_electron_atom",
        "system": ground_state.parameters.name,
        "parameters": asdict(ground_state.parameters),
    },
)
dynamics_metadata = {
    "interaction": "electric_dipole",
    "polarization_axis": "z",
    "gauge": FIELD_GAUGE,
    "field": {
        "type": "SineSquareLaser",
        "field_strength": FIELD_STRENGTH,
        "angular_frequency": OMEGA,
        "wavelength_nm": WAVELENGTH_NM,
        "frequency_codata_set": FREQUENCY_CODATA_SET,
        "n_cycles": N_CYCLES,
        "phase": PHASE,
        "start_time": PULSE_START,
    },
}
result = propagator.propagate(
    ground_state.state,
    time_dependent_hamiltonian,
    ground_state.representation.inner_product,
    preconditioner=real_time_preconditioner,
    samplers={"expec_z": total_z_expectation},
    checkpoint_path=CHECKPOINT_PATH,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    checkpoint_overwrite=CHECKPOINT_OVERWRITE,
    compatibility_metadata=compatibility_metadata,
    dynamics_metadata=dynamics_metadata,
)

# Both arrays include the initial time and every accepted propagation step.
# These are unnormalized quantities; no division by norm_t is performed.
times = result.times
expec_z = result.samples["expec_z"]
norm_t = result.norm_history

print(f"System:                   {ground_state.parameters.name}")
print(f"Ground-state energy:      {ground_state.energy:.12f} Eh")
print(f"Field gauge:              {FIELD_GAUGE}")
print(f"Field strength:           {FIELD_STRENGTH:.6f} a.u.")
print(f"Wavelength:               {WAVELENGTH_NM:.1f} nm")
print(f"Angular frequency:        {OMEGA:.12f} a.u.")
print(f"Pulse duration:           {PULSE_DURATION:.12f} a.u.")
print(f"Final propagation time:   {result.final_time:.12f} a.u.")
print(f"Time step:                {result.time_step:.12f} a.u.")
print(f"Number of steps:          {result.n_steps}")
print(f"Checkpoint path:          {CHECKPOINT_PATH}")
print(f"Initial norm:             {result.initial_norm:.12f}")
print(f"Final norm:               {result.norm:.12f}")
print(f"Final <z1 + z2>:          {expec_z[-1]:.12f} bohr")
print(
    "Maximum norm change:      "
    f"{np.max(np.abs(result.norm_history - result.initial_norm)):.3e}"
)

figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 7.0))
axes[0].plot(times, expec_z)
axes[0].set_ylabel(r"$\langle z_1 + z_2 \rangle$ (bohr)")
axes[0].grid(alpha=0.3)

axes[1].semilogy(times, np.abs(1.0 - norm_t))
axes[1].set_xlabel("Time (a.u.)")
axes[1].set_ylabel(r"$|1 - \langle \Psi | \Psi \rangle|$")
axes[1].grid(alpha=0.3)

figure.suptitle(f"Helium in a 400 nm pulse ({FIELD_GAUGE} gauge)")
figure.tight_layout()
plt.show()
