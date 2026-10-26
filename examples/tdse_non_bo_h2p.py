"""Propagate an archived non-BO H2+ ground state with the high-level API."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields import SineSquareLaser
from grid_lib.spherical_coordinates import CubicCAP
from matter_field_params import wavelength_nm_to_omega_au

from grid_tdse.spherical.two_radial import (
    NonBornOppenheimerDiatomicIonTDSE,
    load_two_radial_state,
)


FieldGauge = Literal["length", "velocity"]
FIELD_GAUGE: FieldGauge = "length"

GROUND_STATE_PATH = Path(__file__).resolve().parent / "ground_states" / (
    "H2p__sp_3e11-4e21_16-8_2__prod_1e-08.npz"
)

FIELD_STRENGTH = 0.001
WAVELENGTH_NM = 400.0
N_CYCLES = 1.0
PHASE = 0.0

TIME_STEP = 0.5
FINAL_TIME = 10.0
RADIAL_OPERATOR_MODE = "element-local"

# Each CAP begins at 80% of its coordinate's radial interval. Thus the
# electronic and internuclear onsets are 0.8 * r_max and 0.8 * R_max.
MASK_r = None
MASK_R = None
CAP_r = CubicCAP(alpha=0.8, eta=1.0)
CAP_R = CubicCAP(alpha=0.8, eta=1.0)


initial_state = load_two_radial_state(GROUND_STATE_PATH)
CODATA_SET = initial_state.physical_state["parameters"]["codata_set"]
OMEGA = wavelength_nm_to_omega_au(WAVELENGTH_NM, codata_set=CODATA_SET)
laser = SineSquareLaser(
    E0=FIELD_STRENGTH,
    omega=OMEGA,
    ncycles=N_CYCLES,
    gauge=FIELD_GAUGE,
    phase=PHASE,
)

calculation = NonBornOppenheimerDiatomicIonTDSE(
    "H2+",
    electronic_grid=initial_state.grid1_specification,
    internuclear_grid=initial_state.grid2_specification,
    initial_state=initial_state,
    pair_exchange_parity=+1,
    fields=[(laser, (0.0, 0.0, 1.0))],
    field_gauge=FIELD_GAUGE,
    l_max=initial_state.basis1.l_max,
    m_max=initial_state.basis1.m_max,
    lambda_max=initial_state.multipoles.lambda_max,
    mu_max=initial_state.multipoles.mu_max,
    M_tot=0,
    codata_set=CODATA_SET,
    radial_operator_mode=RADIAL_OPERATOR_MODE,
    integrator="crank-nicolson",
    integrator_options={
        "solver": "bicgstab",
        "rtol": 0.0,
        "atol": 1.0e-10,
        "preconditioner": True,
    },
    time_step=TIME_STEP,
    final_time=FINAL_TIME,
    mask_r=MASK_r,
    mask_R=MASK_R,
    cap_r=CAP_r,
    cap_R=CAP_R,
    observables=("expec_z",),
)
result = calculation.run(progress=True)

expec_z = result.samples["expec_z"]
norm_t = result.norm_history

print(f"System:                 {calculation.system_name}")
print(f"CODATA set:             {calculation.parameters.codata_set}")
print(f"Field gauge:            {FIELD_GAUGE}")
print(f"Radial operator mode:   {calculation.radial_operator_mode}")
print(f"Integrator:             {calculation.integrator}")
print(f"Time step:              {result.time_step:.12f} a.u.")
print(f"Number of steps:        {result.n_steps}")
print(f"Final norm:             {result.norm:.12f}")
print(f"Final <z>:              {expec_z[-1]:.12f} bohr")

figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 7.0))
axes[0].plot(result.times, expec_z)
axes[0].set_ylabel(r"$\langle z \rangle$ (bohr)")
axes[0].grid(alpha=0.3)
axes[1].semilogy(result.times, np.abs(1.0 - norm_t))
axes[1].set_xlabel("Time (a.u.)")
axes[1].set_ylabel(r"$|1 - \langle \Psi | \Psi \rangle|$")
axes[1].grid(alpha=0.3)
figure.tight_layout()
plt.show()
