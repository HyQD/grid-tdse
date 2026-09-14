import numpy as np
import sys
import scipy
from pathlib import Path
from packaging import version
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    bicgstab,
    eigs,
)
from grid_lib.pseudospectral_grids.sinc_dvr import RadialSincDVR
from matter_field_params.particle_params_manager import (
    get_reduced_masses_and_charges_3body,
    predefined_3body_systems,
)
from grid_lib.spherical_coordinates.angular_momentum import (
    number_of_lm_states,
    LM_to_I,
    setup_y_and_ybar_sympy,
    get_y,
    get_ybar,
)
import time
from opt_einsum import contract

run_noopt_timing = False

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


# Define the radial grids for (r,R)
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

# Define angular momentum cut-offs
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

# Setup system

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
r_max = np.maximum(r_grid, R_grid)[None, :, :]

W_rR = (4 * np.pi / (2 * L_vec + 1)) * (r_min**L_vec) / (r_max ** (L_vec + 1))
W = W_rR + (-1) ** L_vec * W_rR
W *= w_R[None, None, :]

W_LM = np.zeros((number_of_lm_states(L_max, M_max), n_rpoints, n_Rpoints))
for M in range(-M_max, M_max + 1):
    for L in range(abs(M), L_max + 1):
        I_LM = LM_to_I(L, M, L_max, M_max)
        W_LM[I_LM] = W[L]

# for L in range(1, L_max + 1, 2):
#     print(np.linalg.norm(W[L].ravel()))

# for L in range(0, L_max + 1, 2):
#     print(np.linalg.norm(W[L].ravel()))

import time

tic = time.time()
yp, ye_bar = get_y(l_max_p, m_max_p, L_max, M_max), get_ybar(
    l_max_e, m_max_e, L_max, M_max
)
toc = time.time()
print("Time setup y and ybar:", toc - tic, flush=True)

Psi = np.random.random((n_lm_e, n_lm_p, n_rpoints, n_Rpoints))
norm_Psi = np.sqrt(contract("IJab,IJab,a,b", Psi.conj(), Psi, w_r, w_R))
Psi /= norm_Psi
norm_Psi = np.sqrt(contract("IJab,IJab,a,b", Psi.conj(), Psi, w_r, w_R))
print(f"norm_psi: {norm_Psi}")


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
    Te, Hp, W_LM, yp, ye_bar, L_max, M_max, w_r, w_R, OPT_SWITCH=True
)

tic = time.time()
H0_psi = H0_opt(Psi)
toc = time.time()
time_opt_H0 = toc - tic
print("Time for H0_opt(Psi):", time_opt_H0, flush=True)

if run_noopt_timing:
    H0_noopt = Hamiltonian(
        Te, Hp, W_LM, yp, ye_bar, L_max, M_max, w_r, w_R, OPT_SWITCH=False
    )

    tic = time.time()
    H0_noopt_psi = H0_noopt(Psi)
    toc = time.time()
    time_noopt_H0 = toc - tic
    print("Time for H0_noopt(Psi):", time_noopt_H0, flush=True)
    print(flush=True)
    print(
        "Speedup factor (noopt / opt):",
        time_noopt_H0 / time_opt_H0,
        flush=True,
    )
    print(np.allclose(H0_psi, H0_noopt_psi), flush=True)

# Solve for the eigenvalue nearest sigma using shift-invert mode.
eigs_tol = 1e-10
eigs_which = "LM"
eigs_num_states = 1
eigs_maxiter = 500
eigs_report_interval = 10
eigs_matvec_calls = 0
opinv_calls = 0
opinv_inner_iterations = []
opinv_sigma = -0.60
opinv_atol = 1e-12

print(flush=True)
print(
    "Solving H0_opt eigenproblem with scipy.sparse.linalg.eigs shift-invert",
    flush=True,
)
print("SciPy version:", scipy.__version__, flush=True)
print(
    "eigs tolerance:",
    eigs_tol,
    "(relative eigenvalue accuracy)",
    flush=True,
)
print("eigs which:", eigs_which, flush=True)
print("shift-invert sigma:", opinv_sigma, flush=True)
print(
    "eigs maxiter:",
    eigs_maxiter,
    "(None uses SciPy/ARPACK default)",
    flush=True,
)
print(
    "Rayleigh quotient report interval:",
    eigs_report_interval,
    "matvec calls",
    flush=True,
)


def H0_opt_matvec(x):
    global eigs_matvec_calls
    eigs_matvec_calls += 1

    x_psi = np.ascontiguousarray(x.reshape(Psi.shape))
    Hx = H0_opt(x_psi).reshape(-1)

    if eigs_report_interval and eigs_matvec_calls % eigs_report_interval == 0:
        Hx_psi = Hx.reshape(Psi.shape)
        rayleigh = np.einsum(
            "IJab,IJab,a,b->", x_psi.conj(), Hx_psi, w_r, w_R
        ) / np.einsum("IJab,IJab,a,b->", x_psi.conj(), x_psi, w_r, w_R)
        assert np.abs(rayleigh.imag) < 1e-12
        print(
            f"matvec {eigs_matvec_calls}: "
            f"DVR-weighted Rayleigh quotient = {rayleigh.real:.6f}",
            flush=True,
        )

    return Hx


def shifted_H0_matvec(x):
    return H0_opt_matvec(x) - opinv_sigma * x


H0_opt_linear = LinearOperator(
    (Psi.size, Psi.size),
    matvec=H0_opt_matvec,
    dtype=Psi.dtype,
)

shifted_H0_linear = LinearOperator(
    (Psi.size, Psi.size),
    matvec=shifted_H0_matvec,
    dtype=Psi.dtype,
)

opinv_precond = ShiftInvertPreCond(Te, Hp, sigma=opinv_sigma)
opinv_precond_linear = LinearOperator(
    (Psi.size, Psi.size),
    matvec=lambda x: opinv_precond(x.reshape(Psi.shape)).reshape(-1),
    dtype=Psi.dtype,
)


def opinv_matvec(x):
    global opinv_calls
    opinv_calls += 1

    local_counter = Counter()
    y, info = bicgstab(
        shifted_H0_linear,
        x,
        atol=opinv_atol,
        M=opinv_precond_linear,
        callback=local_counter,
        **bicgstab_tol_kwargs,
    )
    opinv_inner_iterations.append(local_counter.counter)

    if info != 0:
        print(
            f"Warning: shift-invert BiCGSTAB solve {opinv_calls} "
            f"did not converge. Info: {info}",
            flush=True,
        )

    if eigs_report_interval and opinv_calls % eigs_report_interval == 0:
        print(
            f"OPinv call {opinv_calls}: "
            f"BiCGSTAB iterations = {local_counter.counter}",
            flush=True,
        )

    return y


opinv_linear = LinearOperator(
    (Psi.size, Psi.size),
    matvec=opinv_matvec,
    dtype=Psi.dtype,
)

tic = time.time()
try:
    eigenvalues, eigenvectors = eigs(
        H0_opt_linear,
        k=eigs_num_states,
        sigma=opinv_sigma,
        which=eigs_which,
        OPinv=opinv_linear,
        v0=Psi.reshape(-1),
        tol=eigs_tol,
        maxiter=eigs_maxiter,
    )
    eigs_converged = True
except ArpackNoConvergence as err:
    eigenvalues = err.eigenvalues
    eigenvectors = err.eigenvectors
    eigs_converged = False
toc = time.time()

print("Time for eigs(H0_opt):", toc - tic, flush=True)
print("H0_opt matvec calls:", eigs_matvec_calls, flush=True)
print("shift-invert OPinv calls:", opinv_calls, flush=True)
if opinv_inner_iterations:
    print(
        "shift-invert BiCGSTAB iterations "
        f"(min/mean/max): {np.min(opinv_inner_iterations)} / "
        f"{np.mean(opinv_inner_iterations):.1f} / {np.max(opinv_inner_iterations)}",
        flush=True,
    )
print("eigs converged:", eigs_converged, flush=True)

if eigenvalues is None or eigenvectors is None or len(eigenvalues) == 0:
    raise RuntimeError("eigs did not return any converged eigenvalues")

ground_state_index = np.argmin(eigenvalues.real)
ground_state_energy = eigenvalues[ground_state_index]
Psi_gs = eigenvectors[:, ground_state_index].reshape(Psi.shape)

norm_Psi_gs = np.einsum("IJab,IJab,a,b->", Psi_gs.conj(), Psi_gs, w_r, w_R)
Psi_gs /= np.sqrt(norm_Psi_gs.real)
norm_Psi_gs_after = np.einsum(
    "IJab,IJab,a,b->", Psi_gs.conj(), Psi_gs, w_r, w_R
)
H0_Psi_gs = H0_opt(Psi_gs)
rayleigh_gs = (
    np.einsum("IJab,IJab,a,b->", Psi_gs.conj(), H0_Psi_gs, w_r, w_R)
    / norm_Psi_gs_after
)
residual_gs = H0_Psi_gs - ground_state_energy * Psi_gs
weighted_residual_norm = np.sqrt(
    np.einsum(
        "IJab,IJab,a,b->",
        residual_gs.conj(),
        residual_gs,
        w_r,
        w_R,
    ).real
)

print(
    "Shift-invert eigenvalue nearest sigma:",
    f"{ground_state_energy.real:.15f}{ground_state_energy.imag:+.3e}j",
    flush=True,
)
print(
    "DVR-weighted Rayleigh quotient:",
    f"{rayleigh_gs.real:.15f}{rayleigh_gs.imag:+.3e}j",
    flush=True,
)
print(
    "DVR-weighted residual norm:",
    f"{weighted_residual_norm:.3e}",
    flush=True,
)
print(
    "DVR-weighted norm of eigenvector:",
    f"{norm_Psi_gs_after.real:.15f}",
    flush=True,
)
print(
    "DVR-weighted norm imaginary part:",
    f"{norm_Psi_gs_after.imag:.3e}",
    flush=True,
)
