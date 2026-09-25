import sys
import time
from pathlib import Path

import numpy as np
import scipy
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

SCIPY_VERSION = version.parse(scipy.__version__)
if SCIPY_VERSION >= version.parse("1.12.0"):
    bicgstab_tol_kwargs = {"rtol": 1e-10}
else:
    bicgstab_tol_kwargs = {"tol": 1e-10}


class Counter:
    def __init__(self):
        self.counter = 0

    def __call__(self, _):
        self.counter += 1


class ShiftInvertPreCond:
    def __init__(self, Te, Hp, sigma):
        self.Te = Te
        self.Hp = Hp
        self.sigma = sigma

        self.n_lme = Te.shape[0]
        self.n_lmp = Hp.shape[0]
        self.n_r = Te.shape[1]
        self.n_R = Hp.shape[1]

        self.Ue = np.zeros_like(self.Te)
        self.Up = np.zeros_like(self.Hp)
        self.Ue_inv = np.zeros_like(self.Ue)
        self.Up_inv = np.zeros_like(self.Up)
        self.eps_e = np.zeros((self.n_lme, self.n_r), dtype=np.float64)
        self.eps_p = np.zeros((self.n_lmp, self.n_R), dtype=np.float64)

        for I in range(self.n_lme):
            self.eps_e[I], self.Ue[I] = np.linalg.eigh(self.Te[I])
            self.Ue_inv[I] = self.Ue[I].T
        for J in range(self.n_lmp):
            self.eps_p[J], self.Up[J] = np.linalg.eigh(self.Hp[J])
            self.Up_inv[J] = self.Up[J].T

    def __call__(self, psi):
        psi_tilde = np.zeros_like(psi)
        for I in range(self.n_lme):
            Ue_inv_I = self.Ue_inv[I]
            for J in range(self.n_lmp):
                Up_inv_J = self.Up_inv[J]
                tmp = Ue_inv_I @ psi[I, J] @ Up_inv_J.T
                lambda_IJ = (
                    self.eps_e[I][:, None] + self.eps_p[J][None, :] - self.sigma
                )
                tmp /= lambda_IJ
                psi_tilde[I, J] = self.Ue[I] @ tmp @ self.Up[J].T

        return psi_tilde


def norm(Psi, w_r, w_R):
    norm_sq = contract("IJab,IJab,a,b->", Psi.conj(), Psi, w_r, w_R)
    return np.sqrt(norm_sq.real)


def normalize(Psi, w_r, w_R):
    return Psi / norm(Psi, w_r, w_R)


def rayleigh_energy(Psi, H0, w_r, w_R):
    H_Psi = H0(np.ascontiguousarray(Psi))
    energy = contract(
        "IJab,IJab,a,b->", Psi.conj(), H_Psi, w_r, w_R
    ) / contract("IJab,IJab,a,b->", Psi.conj(), Psi, w_r, w_R)
    if abs(energy.imag) > 1e-10:
        print(
            f"Warning: energy has imaginary part {energy.imag:.3e}", flush=True
        )
    return energy.real, H_Psi


def weighted_residual_norm(Psi, H_Psi, energy, w_r, w_R):
    residual = H_Psi - energy * Psi
    residual_norm_sq = contract(
        "IJab,IJab,a,b->",
        residual.conj(),
        residual,
        w_r,
        w_R,
    )
    return np.sqrt(residual_norm_sq.real)


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
RDVR = RadialSincDVR(R_max, n_Rpoints)
R = RDVR.r
D1_R = RDVR.D1
D2_R = RDVR.D2
w_R = RDVR.weights

# Define angular momentum cut-offs. Keep these identical to two_radial_sphc.py.
l_max_e = 4
l_max_p = 4

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
Mn = M_ref + m_p
cR_1 = m_p / Mn
cR_2 = (Mn - m_p) / Mn
print(f"c1: {cR_1}, c2: {cR_2}")

# Shift-invert inverse iteration parameters.
sigma = -0.60
n_iterations = 200
ctol_energy = 1e-10
ctol_residual = 1e-8
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
            2 * mu_p
        ) * nabla_sq_R + np.diag(1 / R)

L_max = l_max_e + l_max_p
M_max = L_max

L_vec = np.arange(L_max + 1)[:, None, None]
r_grid = r[:, None]
R_grid = R[None, :]

r_min_1 = np.minimum(r_grid, cR_1 * R_grid)[None, :, :]
r_max_grid_1 = np.maximum(r_grid, cR_1 * R_grid)[None, :, :]

r_min_2 = np.minimum(r_grid, cR_2 * R_grid)[None, :, :]
r_max_grid_2 = np.maximum(r_grid, cR_2 * R_grid)[None, :, :]

W1_rR = (
    (4 * np.pi / (2 * L_vec + 1))
    * (r_min_1**L_vec)
    / (r_max_grid_1 ** (L_vec + 1))
)

W2_rR = (
    (4 * np.pi / (2 * L_vec + 1))
    * (r_min_2**L_vec)
    / (r_max_grid_2 ** (L_vec + 1))
)

W = (-1) ** L_vec * W1_rR + W2_rR
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

Psi = np.zeros((n_lm_e, n_lm_p, n_rpoints, n_Rpoints))
Psi[
    LM_to_I(0, 0, l_max_e, m_max_e),
    LM_to_I(0, 0, l_max_p, m_max_p),
] = np.random.random((n_rpoints, n_Rpoints))
Psi = np.ascontiguousarray(normalize(Psi, w_r, w_R))
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

initial_energy, H0_Psi = rayleigh_energy(Psi, H0_opt, w_r, w_R)
initial_residual = weighted_residual_norm(Psi, H0_Psi, initial_energy, w_r, w_R)

print("SciPy version:", scipy.__version__, flush=True)
print(f"shift-invert sigma: {sigma}", flush=True)
print(f"max inverse iterations: {n_iterations}", flush=True)
print(f"energy convergence tolerance: {ctol_energy:.1e}", flush=True)
print(f"residual convergence tolerance: {ctol_residual:.1e}", flush=True)
print(f"Initial energy: {initial_energy:.15f}", flush=True)
print(f"Initial residual norm: {initial_residual:.3e}", flush=True)
print(flush=True)


def H0_flat(x):
    Psi_in = np.ascontiguousarray(x.reshape(Psi.shape))
    return H0_opt(Psi_in).reshape(-1)


def shifted_H0_flat(x):
    return H0_flat(x) - sigma * x


shifted_H0_linear = LinearOperator(
    (Psi.size, Psi.size),
    matvec=shifted_H0_flat,
    dtype=Psi.dtype,
)

precond = ShiftInvertPreCond(Te, Hp, sigma=sigma)
precond_linear = LinearOperator(
    (Psi.size, Psi.size),
    matvec=lambda x: precond(x.reshape(Psi.shape)).reshape(-1),
    dtype=Psi.dtype,
)

print("Starting explicit shift-invert inverse iteration", flush=True)

energy_old = initial_energy
converged = False
tic_total = time.time()

for iteration in range(1, n_iterations + 1):
    local_counter = Counter()
    tic = time.time()
    y, info = bicgstab(
        shifted_H0_linear,
        Psi.reshape(-1),
        atol=bicgstab_atol,
        M=precond_linear,
        callback=local_counter,
        **bicgstab_tol_kwargs,
    )
    toc = time.time()

    if info != 0:
        print(
            f"Warning: BiCGSTAB did not converge at inverse iteration "
            f"{iteration}. Info: {info}",
            flush=True,
        )

    Psi_new = np.ascontiguousarray(y.reshape(Psi.shape))
    Psi_new = np.ascontiguousarray(normalize(Psi_new, w_r, w_R))

    # Fix the arbitrary sign/phase so dPsi tracks vector convergence.
    overlap = contract("IJab,IJab,a,b->", Psi.conj(), Psi_new, w_r, w_R)
    if overlap.real < 0:
        Psi_new *= -1

    dPsi = np.linalg.norm((Psi_new - Psi).reshape(-1), ord=np.inf)
    Psi = Psi_new

    energy, H0_Psi = rayleigh_energy(Psi, H0_opt, w_r, w_R)
    dE = energy - energy_old
    energy_old = energy
    residual_norm = weighted_residual_norm(Psi, H0_Psi, energy, w_r, w_R)

    print(
        f"iter {iteration:5d}  "
        f"E={energy: .15f}  dE={dE: .3e}  "
        f"res={residual_norm:.3e}  dPsi={dPsi:.3e}  "
        f"bicgstab_it={local_counter.counter:4d}  solve_time={toc - tic:.2f}s",
        flush=True,
    )

    if abs(dE) < ctol_energy and residual_norm < ctol_residual:
        converged = True
        break

toc_total = time.time()

print(flush=True)
print("Shift-invert inverse iteration finished", flush=True)
print("converged:", converged, flush=True)
print("iterations:", iteration, flush=True)
print(f"final energy: {energy_old:.15f}", flush=True)
print(f"final norm: {norm(Psi, w_r, w_R):.15f}", flush=True)
print(f"total time: {toc_total - tic_total:.2f} s", flush=True)
