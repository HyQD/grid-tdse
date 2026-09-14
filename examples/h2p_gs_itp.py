import sys
import time
from pathlib import Path

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


# Define the radial grids for (r,R). Keep these identical to two_radial_sphc.py.
r_max = 10.0
n_rpoints = 50
rDVR = RadialSincDVR(r_max, n_rpoints)
r = rDVR.r
D1_r = rDVR.D1
D2_r = rDVR.D2
w_r = rDVR.weights

R_max = 5.0
n_Rpoints = 50
RDVR = RadialSincDVR(R_max / 2, n_Rpoints)
R = RDVR.r
D1_R = RDVR.D1
D2_R = RDVR.D2
w_R = RDVR.weights

# Define angular momentum cut-offs. Keep these identical to two_radial_sphc.py.
l_max_e = 2
l_max_p = 2

m_max_e = l_max_e
m_max_p = l_max_p

n_l_e = l_max_e + 1
n_lm_e = number_of_lm_states(l_max_e, m_max_e)

n_l_p = l_max_p + 1
n_lm_p = number_of_lm_states(l_max_p, m_max_p)

particles = predefined_3body_systems("H2+")
mu_p, mu_e, xi_p, xi_e, M_ref, m_p, m_e, Q_ref, q_p, q_e = (
    get_reduced_masses_and_charges_3body(particles=particles, codata_set="2022")
)

# Imaginary-time propagation parameters, following XHplus/3D/ground.py.
dt = 0.25
n_steps = int(1e8)
ctol_Psi = 1e-8
report_interval = 10
bicgstab_atol = 1e-12


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
            8 * mu_p
        ) * nabla_sq_R + np.diag(1 / (2 * R))

L_max = l_max_e + l_max_p
M_max = L_max

L_vec = np.arange(L_max + 1)[:, None, None]
r_grid = r[:, None]
R_grid = R[None, :]
r_min = np.minimum(r_grid, R_grid)[None, :, :]
r_max_grid = np.maximum(r_grid, R_grid)[None, :, :]

W_rR = (
    (4 * np.pi / (2 * L_vec + 1)) * (r_min**L_vec) / (r_max_grid ** (L_vec + 1))
)
W = W_rR + (-1) ** L_vec * W_rR
W *= w_R[None, None, :]

W_LM = np.zeros((number_of_lm_states(L_max, M_max), n_rpoints, n_Rpoints))
for M in range(-M_max, M_max + 1):
    for L in range(abs(M), L_max + 1):
        I_LM = LM_to_I(L, M, L_max, M_max)
        W_LM[I_LM] = W[L]

tic = time.time()
yp = get_y(l_max_p, m_max_p, L_max, M_max)
ye_bar = get_ybar(l_max_e, m_max_e, L_max, M_max)
toc = time.time()
print("Time setup y and ybar:", toc - tic, flush=True)

Psi = np.random.random((n_lm_e, n_lm_p, n_rpoints, n_Rpoints))
Psi /= norm(Psi, w_r, w_R)
print(f"Initial norm: {norm(Psi, w_r, w_R):.15f}", flush=True)
print(f"Size of Psi: {Psi.size * Psi.itemsize / 1e9:.2g} GB", flush=True)


# Allow running this example directly from the examples/ folder.
try:
    from grid_tdse.coordinates.spherical.two_radial.hamiltonians import (
        Hamiltonian,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    nonbo_src = repo_root.parent / "nonBO_Hamiltonian" / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if nonbo_src.exists() and str(nonbo_src) not in sys.path:
        sys.path.insert(0, str(nonbo_src))
    from grid_tdse.coordinates.spherical.two_radial.hamiltonians import (
        Hamiltonian,
    )


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

tic = time.time()
initial_energy = rayleigh_energy(Psi, H0_opt, w_r, w_R)
toc = time.time()
print("Time for initial H0_opt(Psi):", toc - tic, flush=True)
print(f"Initial energy: {initial_energy:.12f}", flush=True)
print("SciPy version:", scipy.__version__, flush=True)
print(f"dt: {dt}", flush=True)
print(f"ctol_Psi: {ctol_Psi:.1e}", flush=True)
print(f"max steps: {n_steps}", flush=True)
print(f"progress interval: {report_interval} steps", flush=True)
print(flush=True)

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
    HPsi = H0_opt(Psi)
    z = Psi - 0.5 * dt * HPsi

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

    if info != 0:
        print(
            f"Warning: BiCGSTAB did not converge at step {step}. Info: {info}"
        )

    Psi = np.ascontiguousarray(Psi_new.reshape(Psi.shape))
    Psi /= norm(Psi, w_r, w_R)
    t += dt

    dPsi = np.linalg.norm(Psi.reshape(-1) - Psi_old.reshape(-1), ord=np.inf)
    Psi_old = Psi.copy()

    energy = rayleigh_energy(Psi, H0_opt, w_r, w_R)
    dE = energy - energy_old
    energy_old = energy

    if step == 1 or step % report_interval == 0 or dPsi < ctol_Psi:
        print(
            f"step {step:8d}  t/i={t:12.6f}  "
            f"E={energy: .12f}  dE={dE: .3e}  dPsi={dPsi:.3e}  "
            f"bicgstab_it={local_counter.counter:4d}  solve_time={toc - tic:.2f}s",
            flush=True,
        )

    if dPsi < ctol_Psi:
        converged = True
        break

toc_total = time.time()

print(flush=True)
print("Imaginary-time propagation finished", flush=True)
print("converged:", converged, flush=True)
print("steps:", step, flush=True)
print(f"final t/i: {t:.6f}", flush=True)
print(f"final energy: {energy_old:.15f}", flush=True)
print(f"final norm: {norm(Psi, w_r, w_R):.15f}", flush=True)
print(f"total time: {toc_total - tic_total:.2f} s", flush=True)
