import sys
import time
from pathlib import Path
from tracemalloc import stop

import numpy as np
import scipy
import scipy as sp
from opt_einsum import contract
from packaging import version
from scipy.sparse.linalg import LinearOperator, bicgstab

from grid_lib.pseudospectral_grids.sinc_dvr import RadialSincDVR
from grid_lib.spherical_coordinates.angular_momentum import (
    LM_to_I,
    get_y,
    get_ybar,
    number_of_lm_states,
)

from grid_tdse.spherical.two_radial.hamiltonian import (
    Hamiltonian,
)
from grid_tdse.spherical.two_radial.system import TwoRadialSystem

from matter_field_params.particle_params_manager import (
    get_reduced_masses_and_charges_3body,
    predefined_3body_systems,
)

SCIPY_VERSION = version.parse(sp.__version__)
if SCIPY_VERSION >= version.parse("1.12.0"):
    tol_kwargs = {"rtol": 0.0}
else:
    tol_kwargs = {"tol": 0.0}


class Counter:
    def __init__(self):
        self.counter = 0

    def __call__(self, _):
        self.counter += 1


class PreCond:
    def __init__(self, He, Hp, alpha):
        self.He = He
        self.Hp = Hp
        self.alpha = alpha

        self.n_lme = He.shape[0]
        self.n_lmp = Hp.shape[0]
        self.n_r = He.shape[1]
        self.n_R = Hp.shape[1]

        self.Ue = np.zeros_like(self.He)
        self.Up = np.zeros_like(self.Hp)

        self.eps_e = np.zeros((self.n_lme, self.n_r), dtype=np.float64)
        self.eps_p = np.zeros((self.n_lmp, self.n_R), dtype=np.float64)

        self.Ue_inv = np.zeros_like(self.Ue)
        self.Up_inv = np.zeros_like(self.Up)

        for I in range(self.n_lme):
            self.eps_e[I], self.Ue[I] = np.linalg.eig(self.He[I])
            self.Ue_inv[I] = np.linalg.inv(self.Ue[I])
        for J in range(self.n_lmp):
            self.eps_p[J], self.Up[J] = np.linalg.eig(self.Hp[J])
            self.Up_inv[J] = np.linalg.inv(self.Up[J])

    def __call__(self, psi):
        psi_tilde = np.zeros_like(psi)
        for I in range(self.n_lme):
            Ue_inv_I = self.Ue_inv[I]
            for J in range(self.n_lmp):
                Up_inv_J = self.Up_inv[J]
                psi_matrix = psi[I, J]

                tmp = Ue_inv_I @ psi_matrix @ Up_inv_J.T

                lambda_IJ = np.ones_like(tmp)
                lambda_IJ += self.alpha * (
                    self.eps_e[I][:, np.newaxis] + self.eps_p[J][np.newaxis, :]
                )

                tmp /= lambda_IJ
                psi_tilde[I, J] = self.Ue[I] @ tmp @ self.Up[J].T

        return psi_tilde


def norm(Psi, w_r, w_R):
    norm_sq = contract("IJab,IJab,a,b->", Psi.conj(), Psi, w_r, w_R)
    return np.sqrt(norm_sq.real)


def rayleigh_energy(Psi, H0, w_r, w_R):
    H_Psi = H0(Psi)
    energy = contract("IJab,IJab,a,b->", Psi.conj(), H_Psi, w_r, w_R)
    if abs(energy.imag) > 1e-10:
        print(
            f"Warning: energy has imaginary part {energy.imag:.3e}", flush=True
        )
    return energy.real


particles = predefined_3body_systems("H2+")
mu_p, mu_e, xi_p, xi_e, M_ref, m_p, m_e, Q_ref, q_p, q_e = (
    get_reduced_masses_and_charges_3body(particles=particles, codata_set="2022")
)
Mn = M_ref + m_p
cR_1 = m_p / Mn
cR_2 = (Mn - m_p) / Mn
print(f"c1: {cR_1}, c2: {cR_2}")
cR = [
    (-cR_1, Q_ref, q_e),
    (cR_2, q_p, q_e),
]  # The coefficient and charges, (c,q1,q2), in the Coulomb interaction q1*q2/|r-c*R|
################################################################################
"""
Define radial-angular grids and setup matrices/arrays for the Hamiltonian.
"""
from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.radial_Coulomb import radial_Coulomb, radial_Coulomb_femdvr_two_grid
# Define the radial grids for (r,R). Keep these identical to two_radial_sphc.py.
grid = "elements"

if grid=="uniform":

    r_max = 10.0
    n_rpoints = 50

    R_max = 5.0
    n_Rpoints = 50

    rDVR = RadialSincDVR(r_max=r_max, N=n_rpoints)
    RDVR = RadialSincDVR(r_max=R_max, N=n_Rpoints)

    r = rDVR.r
    D1_r = rDVR.D1
    D2_r = rDVR.D2
    w_r = rDVR.weights

    R = RDVR.r
    D1_R = RDVR.D1
    D2_R = RDVR.D2
    w_R = RDVR.weights

elif grid == "elements":

    r_max = 12.0
    n_elements_r = 4
    element_size_r = r_max / n_elements_r

    nodes_r = np.arange(0.0, r_max + element_size_r, element_size_r)

    n_points_pr_element_r = 15
    n_points_r = np.ones((n_elements_r,), dtype=int) * n_points_pr_element_r
    
    rDVR = FEMDVR(nodes_r, n_points_r, Linear_map, GaussLegendreLobatto)

    r = rDVR.r[1:-1]
    D1_r = rDVR.D1[1:-1, 1:-1]
    D2_r = rDVR.D2[1:-1, 1:-1]
    w_r = rDVR.weights[1:-1]

    R_max = 5.0
    n_elements_R = 5
    element_size_R = R_max / n_elements_R

    nodes_R = np.arange(0.0, R_max + element_size_R, element_size_R)

    n_points_pr_element_R = 15
    n_points_R = np.ones((n_elements_R,), dtype=int) * n_points_pr_element_R

    RDVR = FEMDVR(nodes_R, n_points_R, Linear_map, GaussLegendreLobatto)

    R = RDVR.r[1:-1]
    D1_R = RDVR.D1[1:-1, 1:-1]
    D2_R = RDVR.D2[1:-1, 1:-1]

    w_R = RDVR.weights[1:-1]

    n_rpoints = len(r)
    n_Rpoints = len(R)

# Define angular momentum cut-offs. Keep these identical to two_radial_sphc.py.
l_max_e = 2
l_max_p = 2

m_max_e = l_max_e
m_max_p = l_max_p

# Cut-off values (L_max, M_max) in the multipole expansion of the Coulomb interaction.
L_max = l_max_e + l_max_p
M_max = L_max

n_LM = number_of_lm_states(L_max, M_max)

n_l_e = l_max_e + 1
n_lm_e = number_of_lm_states(l_max_e, m_max_e)

n_l_p = l_max_p + 1
n_lm_p = number_of_lm_states(l_max_p, m_max_p)

# Setup system.
Te = np.zeros((n_lm_e, n_rpoints, n_rpoints))
Hp = np.zeros((n_lm_p, n_Rpoints, n_Rpoints))
for m in range(-m_max_e, m_max_e + 1):
    for l in range(abs(m), l_max_e + 1):
        nabla_sq_r = D2_r - np.diag(l * (l + 1) / r**2)
        Te[LM_to_I(l, m, l_max_e, m_max_e)] = -1 / (2 * mu_e) * nabla_sq_r

for m in range(-m_max_p, m_max_p + 1):
    for l in range(abs(m), l_max_p + 1):
        nabla_sq_R = D2_R - np.diag(l * (l + 1) / R**2)
        Hp[LM_to_I(l, m, l_max_p, m_max_p)] = -1 / (
            2 * mu_p
        ) * nabla_sq_R + np.diag(1 / R)

from grid_tdse.spherical.two_radial.coulomb import (
    CoulombInteraction,
)
from matplotlib import pyplot as plt

radial_coulomb_rep = "Poisson" # or Poisson
if grid == "elements":


    if radial_coulomb_rep == "Poisson":
        W = radial_Coulomb_femdvr_two_grid(rDVR, RDVR, L_max+1, cR)

    elif radial_coulomb_rep == "quadrature":
        coulomb = CoulombInteraction(
            r, R, cR, l_max_e, m_max_e, l_max_p, m_max_p, L_max, M_max
        )
        W = coulomb.W
        W *= w_R[None, None, :]



if grid == "uniform":
    coulomb = CoulombInteraction(
        r, R, cR, l_max_e, m_max_e, l_max_p, m_max_p, L_max, M_max
    )

    W = coulomb.W
    W *= w_R[None, None, :]

W_LM = np.zeros((n_LM, n_rpoints, n_Rpoints))
for M in range(-M_max, M_max + 1):
    for L in range(abs(M), L_max + 1):
        I_LM = LM_to_I(L, M, L_max, M_max)
        W_LM[I_LM] = W[L]

tic = time.time()
yp = get_y(l_max_p, m_max_p, L_max, M_max)
ye_bar = get_ybar(l_max_e, m_max_e, L_max, M_max)
toc = time.time()
print("Time setup y and ybar:", toc - tic, flush=True)
###########################################################################


H0_opt = Hamiltonian(
    Te,
    Hp,
    W_LM,
    yp,
    ye_bar,
    L_max,
    M_max,
    w_r,
    w_R,
    OPT_SWITCH=True,
)

# Imaginary-time propagation parameters, following XHplus/3D/ground.py.
dt = 0.25
n_steps = 100
ctol_Psi = 1e-6
report_interval = 10
bicgstab_atol = 1e-12
RNG_SEED = 42

Psi = np.zeros((n_lm_e, n_lm_p, n_rpoints, n_Rpoints))
rng = np.random.default_rng(RNG_SEED)
Psi[LM_to_I(0, 0, l_max_e, m_max_e), LM_to_I(0, 0, l_max_p, m_max_p)] = (
    rng.random((n_rpoints, n_Rpoints))
)
Psi /= norm(Psi, w_r, w_R)
print(f"RNG seed: {RNG_SEED}", flush=True)
print(f"Initial norm: {norm(Psi, w_r, w_R):.15f}", flush=True)
print(f"Size of Psi: {Psi.size * Psi.itemsize / 1e9:.2g} GB", flush=True)

tic = time.time()
initial_energy = rayleigh_energy(Psi, H0_opt, w_r, w_R)
toc = time.time()


precond = PreCond(Te, Hp, alpha=0.5 * dt)
M_linear = LinearOperator(
    (Psi.size, Psi.size),
    matvec=lambda x: precond(x.reshape(Psi.shape)).reshape(-1),
    dtype=Psi.dtype,
)


def H0_flat(x):
    Psi_in = np.ascontiguousarray(x.reshape(Psi.shape))
    return H0_opt(Psi_in).reshape(-1)


def A_flat(x):
    return x + 0.5 * dt * H0_flat(x)


A_linear = LinearOperator((Psi.size, Psi.size), matvec=A_flat, dtype=Psi.dtype)

print("Starting imaginary-time propagation with CN + BiCGSTAB", flush=True)

t = 0.0
energy_old = initial_energy
Psi_old = Psi.copy()
converged = False

tic_total = time.time()
for step in range(1, n_steps + 1):

    tic = time.time()
    HPsi = H0_opt(Psi)
    z = Psi - 0.5 * dt * HPsi
    toc = time.time()
    time_CN_z = toc - tic

    local_counter = Counter()
    tic = time.time()
    Psi_new, info = bicgstab(
        A_linear,
        z.reshape(-1),
        x0=Psi.reshape(-1),
        atol=bicgstab_atol,
        M=M_linear,
        callback=local_counter,
        **tol_kwargs,
    )
    toc = time.time()
    time_bicgstab = toc - tic

    if info != 0:
        print(
            f"Warning: BiCGSTAB did not converge at step {step}. Info: {info}"
        )

    tic = time.time()
    Psi = np.ascontiguousarray(Psi_new.reshape(Psi.shape))
    Psi /= norm(Psi, w_r, w_R)
    t += dt

    dPsi = np.linalg.norm(Psi.reshape(-1) - Psi_old.reshape(-1), ord=np.inf)
    Psi_old = Psi.copy()

    energy = rayleigh_energy(Psi, H0_opt, w_r, w_R)
    dE = energy - energy_old
    energy_old = energy
    toc = time.time()
    time_ex_vals = toc - tic

    if step == 1 or step % report_interval == 0 or dPsi < ctol_Psi:
        print(
            f"step {step:2d} "
            f"E={energy: .6f}  dE={dE: .2g}  dPsi={dPsi:.2g}  "
            f"bicgstab_it={local_counter.counter:4d}  solve_time={time_bicgstab:.2g}s  CN_z_time={time_CN_z:.2g}s  ex_vals_time={time_ex_vals:.2g}s",
            flush=True,
        )

    if dPsi < ctol_Psi:
        converged = True
        break


print()
sum_norm_psi_IJ = 0
for le in range(0, l_max_e + 1):
    for me in range(-le, le + 1):
        Ie = LM_to_I(le, me, l_max_e, m_max_e)
        for lp in range(0, l_max_p + 1):
            for mp in range(-lp, lp + 1):
                Ip = LM_to_I(lp, mp, l_max_p, m_max_p)
                norm_psi_IJ = contract("ab,a,b->", Psi[Ie, Ip] ** 2, w_r, w_R)
                if norm_psi_IJ > 1e-12:
                    sum_norm_psi_IJ += norm_psi_IJ
                    print(
                        f"||Psi[Ie={Ie}, Ip={Ip}]|| (le={le}, me={me}, lp={lp}, mp={mp}) = {norm_psi_IJ:.6f}"
                    )
print()
print(f"sum_norm_psi_IJ = {sum_norm_psi_IJ:.15f}")
print()
toc_total = time.time()

print(flush=True)
print("Imaginary-time propagation finished", flush=True)
print("converged:", converged, flush=True)
print("steps:", step, flush=True)
print(f"final t/i: {t:.6f}", flush=True)
print(f"final energy: {energy_old:.15f}", flush=True)
print(f"final norm: {norm(Psi, w_r, w_R):.15f}", flush=True)
print(f"total time: {toc_total - tic_total:.2f} s", flush=True)
