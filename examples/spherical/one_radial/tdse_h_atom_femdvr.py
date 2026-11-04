"""Propagate hydrogen on a finite-element GLL grid in velocity gauge.

The wavefunction is expanded as

    Psi(r, Omega, t) = sum_l u_l(r, t) Y_l0(Omega) / r.

A linearly z-polarized field conserves the magnetic quantum number, so only
the m=0 partial waves are included.  All quantities are in atomic units.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields.dipole_fields import SineSquareLaser
from scipy.sparse.linalg import LinearOperator, bicgstab
from tqdm.auto import tqdm

from grid_lib.pseudospectral_grids.femdvr import FEMDVR
from grid_lib.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)
from grid_lib.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_l,
)
from grid_lib.spherical_coordinates.preconditioners import M2Psi
from grid_lib.spherical_coordinates.properties import compute_norm, expec_x_i
from grid_lib.spherical_coordinates.rhs import H0Psi, HtPsi
from grid_lib.spherical_coordinates.time_dependent_field_interaction import (
    V_psi_velocity_z,
)
from grid_lib.spherical_coordinates.utils import Counter


def cosine_mask(r, r0, r_max, exponent):
    """Return a cosine mask that is one below ``r0`` and zero at ``r_max``."""
    if not 0.0 <= r0 < r_max:
        raise ValueError("Require 0 <= mask_r0 < r_max.")
    if exponent <= 0.0:
        raise ValueError("mask_exponent must be positive.")

    mask = np.ones_like(r)
    active = r > r0
    argument = np.pi * (r[active] - r0) / (2.0 * (r_max - r0))
    mask[active] = np.cos(argument) ** exponent
    return mask


def radial_probability_density(psi):
    r"""Return the radial probability density.

    Orthogonality of the spherical harmonics gives
    ``P(r) = r**2 integral |Psi(r, Omega)|**2 dOmega = sum_l |u_l(r)|**2``
    for the radial convention used here.  Its quadrature integral equals the
    instantaneous wavefunction norm, including loss caused by the mask.
    """
    return np.sum(np.abs(psi) ** 2, axis=0)


# Finite-element Gauss-Legendre-Lobatto grid.
r_max = 320.0
n_elements = 80
points_per_element = 11
nodes = np.linspace(0.0, r_max, n_elements + 1)
n_points = np.full(n_elements, points_per_element, dtype=int)

# The z-polarized field preserves m=0; l_max controls the partial-wave basis.
l_max = 4

# Three-cycle sine-square pulse polarized parallel to the z axis.
field_gauge = "velocity"
E0 = 0.029
omega = 0.057
phase = 0.0
ncycles = 3
pulse = SineSquareLaser(
    E0=E0,
    omega=omega,
    ncycles=ncycles,
    gauge=field_gauge,
    phase=phase,
)

# Crank-Nicolson and iterative-solver controls.  The time step is adjusted
# slightly so the last time point coincides exactly with the end of the pulse.
requested_time_step = 1/16
linear_atol = 1e-13
linear_maxiter = 500
duration = 2.0 * np.pi * ncycles / omega
n_steps = int(np.ceil(duration / requested_time_step))
time_step = duration / n_steps
times = np.linspace(0.0, duration, n_steps + 1)

# Cosine mask M(r) = cos[pi(r-r0)/(2(r_max-r0))]**exponent for r > r0.
# Set mask_r0 closer to r_max for a narrower absorber, or set mask_enabled to
# False to disable absorption.  The mask is applied after every time step.
mask_enabled = False
mask_r0 = 70.0
mask_exponent = 0.25


# Build the FEM-DVR and impose homogeneous radial boundary conditions by
# removing the first and last GLL points.
femdvr = FEMDVR(nodes, n_points, Linear_map, GaussLegendreLobatto)
r = femdvr.r[1:-1]
weights = femdvr.weights[1:-1]

# The grid-lib TDSE operators consume this small radial-data interface.  The
# FEMDVR differentiation matrices are already assembled across all elements.
radial_matrix_elements = SimpleNamespace(
    r=r,
    r_inv=1.0 / r,
    D1=femdvr.D1[1:-1, 1:-1],
    D2=femdvr.D2[1:-1, 1:-1],
    weights=weights,
    nr=r.size,
)

angular_matrix_elements = AngularMatrixElements_l(
    arr_to_calc=["z_Omega", "H_z_beta"],
    l_max=l_max,
    m=0,
)
n_channels = angular_matrix_elements.n_lm


# Compute the numerical hydrogen 1s state.  D2_symmetric is the FEM-DVR
# kinetic-energy matrix in the quadrature-orthonormal representation.
potential = -1.0 / r
ground_state_hamiltonian = (
    -0.5 * femdvr.D2_symmetric[1:-1, 1:-1] + np.diag(potential)
)
ground_state_energies, ground_state_vectors = np.linalg.eigh(
    ground_state_hamiltonian
)
ground_state_energy = ground_state_energies[0]
radial_1s = ground_state_vectors[:, 0] / np.sqrt(weights)
radial_1s /= np.sqrt(np.sum(weights * np.abs(radial_1s) ** 2))

psi_t = np.zeros((n_channels, r.size), dtype=np.complex128)
psi_t[0] = radial_1s

if mask_enabled:
    mask_r = cosine_mask(r, mask_r0, r_max, mask_exponent)
else:
    mask_r = np.ones_like(r)


# H(t) = H0 + A_z(t) p_z.  The spatially uniform A_z(t)^2/2 term only adds
# a global phase and is therefore omitted.
H0_psi = H0Psi(angular_matrix_elements, radial_matrix_elements, potential)
Vt_psi = V_psi_velocity_z(
    angular_matrix_elements,
    radial_matrix_elements,
    a_field_z=pulse,
)
hamiltonian = HtPsi(
    angular_matrix_elements,
    radial_matrix_elements,
    H0_psi,
    [Vt_psi],
)

preconditioner = M2Psi(
    angular_matrix_elements,
    radial_matrix_elements,
    time_step,
)
dimension = n_channels * r.size
preconditioner_operator = LinearOperator(
    (dimension, dimension),
    matvec=preconditioner,
    dtype=np.complex128,
)


# Sample the norm and z dipole throughout the propagation.
z_angular = angular_matrix_elements("z_Omega")
norms = np.empty(n_steps + 1)
dipole_z = np.empty(n_steps + 1)
radial_probability = np.empty((n_steps + 1, r.size))
linear_iterations = np.empty(n_steps, dtype=int)
norms[0] = compute_norm(psi_t, weights).real
dipole_z[0] = expec_x_i(psi_t, weights, r, z_angular).real
radial_probability[0] = radial_probability_density(psi_t)

for step in tqdm(range(n_steps), desc="Crank-Nicolson"):
    midpoint = 0.5 * (times[step] + times[step + 1])

    def crank_nicolson_left(vector):
        return vector + 0.5j * time_step * hamiltonian(vector, midpoint)

    left_operator = LinearOperator(
        (dimension, dimension),
        matvec=crank_nicolson_left,
        dtype=np.complex128,
    )
    right_hand_side = psi_t.ravel() - 0.5j * time_step * hamiltonian(
        psi_t, midpoint
    )

    counter = Counter()
    propagated, info = bicgstab(
        left_operator,
        right_hand_side,
        x0=psi_t.ravel(),
        M=preconditioner_operator,
        rtol=0.0,
        atol=linear_atol,
        maxiter=linear_maxiter,
        callback=counter,
    )
    if info != 0:
        raise RuntimeError(
            f"Crank-Nicolson solve failed at step {step + 1} "
            f"(t={times[step + 1]:.8f}, bicgstab info={info})."
        )

    linear_iterations[step] = counter.counter
    psi_t = propagated.reshape((n_channels, r.size))
    psi_t *= mask_r[None, :]

    norms[step + 1] = compute_norm(psi_t, weights).real
    dipole_z[step + 1] = expec_x_i(
        psi_t, weights, r, z_angular
    ).real
    radial_probability[step + 1] = radial_probability_density(psi_t)


title = "TDSE propagation of H(1s)"
print(f"\n{title}")
print("=" * len(title))
print(f"{'Field gauge':<24}: {field_gauge}")
print(f"{'Integrator':<24}: Crank-Nicolson")
print(f"{'Element boundaries':<24}: {nodes.tolist()} bohr")
print(f"{'Points per element':<24}: {points_per_element}")
print(f"{'Number of radial points':<24}: {r.size}")
print(f"{'l_max':<24}: {l_max}")
print(f"{'Ground-state energy':<24}: {ground_state_energy:.12f} a.u.")
print(f"{'Pulse duration':<24}: {duration:.12f} a.u.")
print(f"{'Time step':<24}: {time_step:.12f} a.u.")
print(f"{'Number of steps':<24}: {n_steps}")
print(f"{'Mask enabled':<24}: {mask_enabled}")
if mask_enabled:
    print(f"{'Mask start':<24}: {mask_r0} bohr")
    print(f"{'Mask exponent':<24}: {mask_exponent}")
print(f"{'Final norm':<24}: {norms[-1]:.12f}")
print(f"{'Maximum linear iterations':<24}: {linear_iterations.max()}")

vector_potential = pulse(times)
figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
axes[0].plot(times, vector_potential)
axes[0].set_ylabel(r"$A_z(t)$ (a.u.)")
axes[0].set_title(title)
axes[1].plot(times, dipole_z)
axes[1].set_ylabel(r"$\langle z\rangle$ (bohr)")
axes[2].semilogy(times, np.abs(1-norms))
axes[2].set_xlabel("Time (a.u.)")
axes[2].set_ylabel(r"$|1-\langle\Psi|\Psi\rangle|$")
figure.tight_layout()

radial_figure, radial_axes = plt.subplots(figsize=(8, 4.5))
#radial_axes.semilogy(r, radial_probability[0], label="Initial")
radial_axes.semilogy(r, radial_probability[-1], label="Final")
radial_axes.axhline(1e-15, color='red', linestyle='--')
radial_axes.set_ylim(1e-16, 1)
radial_axes.set_xlabel("Radius (bohr)")
radial_axes.set_ylabel(r"$P(r)=r^2\int |\Psi(r,\Omega)|^2\,d\Omega$")
radial_axes.set_title("Radial probability density")
radial_axes.legend()
radial_figure.tight_layout()
plt.show()
