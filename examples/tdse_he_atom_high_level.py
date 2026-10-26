"""Propagate an archived helium ground state with the high-level TDSE API."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields import SineSquareLaser
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask
from matter_field_params import wavelength_nm_to_omega_au

from grid_tdse.spherical.two_radial import (
    TwoElectronAtomTDSE,
    load_two_radial_state,
)


FieldGauge = Literal["length", "velocity"]
FIELD_GAUGE: FieldGauge = "length"

GROUND_STATE_PATH = Path(__file__).resolve().parent / "ground_states" / (
    "He__sp_5e11_32_2__prod_1e-08.npz"
)

FIELD_STRENGTH = 0.001
WAVELENGTH_NM = 400.0
CODATA_SET = "2022"
OMEGA = wavelength_nm_to_omega_au(WAVELENGTH_NM, codata_set=CODATA_SET)
N_CYCLES = 1.0
PHASE = 0.0

TIME_STEP = 0.5
FINAL_TIME = 10.0
# SIL applies because the Hamiltonian remains Hermitian; the mask is applied
# only after each accepted propagation step.
INTEGRATOR = "sil"

# Select at most one absorber. For example, replace the mask by
# CubicCAP(alpha=0.75, eta=0.01) to use a CAP instead.
MASK = RadialCosineMask(r0=24.0, n=4.0)
CAP: CubicCAP | None = None


initial_state = load_two_radial_state(GROUND_STATE_PATH)
laser = SineSquareLaser(
    E0=FIELD_STRENGTH,
    omega=OMEGA,
    ncycles=N_CYCLES,
    gauge=FIELD_GAUGE,
    phase=PHASE,
)

calculation = TwoElectronAtomTDSE(
    "He",
    grid=initial_state.grid1_specification,
    initial_state=initial_state,
    fields=[(laser, (0.0, 0.0, 1.0))],
    field_gauge=FIELD_GAUGE,
    l_max=initial_state.basis1.l_max,
    m_max=initial_state.basis1.m_max,
    lambda_max=initial_state.multipoles.lambda_max,
    mu_max=initial_state.multipoles.mu_max,
    M_tot=0,
    integrator=INTEGRATOR,
    integrator_options={
        "max_krylov_dimension": 20,
        "rtol": 1.0e-10,
        "atol": 0.0,
        "max_refinements": 8,
        "reorthogonalize": True,
        "breakdown_tolerance": 1.0e-14,
    },
    time_step=TIME_STEP,
    final_time=FINAL_TIME,
    mask=MASK,
    cap=CAP,
    observables=("expec_z",),
)
result = calculation.run(progress=True)

expec_z = result.samples["expec_z"]
norm_t = result.norm_history

print(f"System:                 {calculation.atom_name}")
print(f"Field gauge:            {FIELD_GAUGE}")
print(f"Integrator:             {calculation.integrator}")
print(f"Time step:              {result.time_step:.12f} a.u.")
print(f"Number of steps:        {result.n_steps}")
print(f"Final norm:             {result.norm:.12f}")
print(f"Final <z1 + z2>:        {expec_z[-1]:.12f} bohr")

figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 7.0))
axes[0].plot(result.times, expec_z)
axes[0].set_ylabel(r"$\langle z_1 + z_2 \rangle$ (bohr)")
axes[0].grid(alpha=0.3)
axes[1].semilogy(result.times, np.abs(1.0 - norm_t))
axes[1].set_xlabel("Time (a.u.)")
axes[1].set_ylabel(r"$|1 - \langle \Psi | \Psi \rangle|$")
axes[1].grid(alpha=0.3)
figure.tight_layout()
plt.show()
