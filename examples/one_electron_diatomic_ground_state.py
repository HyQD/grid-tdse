"""Diagonalize a clamped-nuclei one-electron diatomic Hamiltonian.

All quantities are in atomic units. H2+ is selected below, but ``Z1`` and
``Z2`` may be changed to represent any positive two-centre charges acting on
one electron. The spherical expansion origin is the bond midpoint.
"""

import numpy as np

from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    diagonalize_fixed_m,
)


# Physical system: nuclei at z=-a and z=+a, with R12=2*a.
name = "H2+"
Z1 = 1.0
Z2 = 1.0
a = 1.0
particle_mass = 1.0

# The nuclear radius a is an internal FEM element boundary.
nodes = [0.0, a, 2*a, 4.0, 8.0, 16.0]

n_points = 15
n_points_list = [n_points] * (len(nodes) - 1)

# Axial nuclei conserve m. The ground state belongs to m=0.
m = 0
l_max = 10
m_max = 0
nuclear_L_max = 2 * l_max
nuclear_method = "poisson"
n_states = 1


system = OneElectronDiatomicSystem(
    nuclear_charge1=Z1,
    nuclear_charge2=Z2,
    half_bond_length=a,
    particle_mass=particle_mass,
    name=name,
)
hamiltonian = OneElectronDiatomicHamiltonian(
    system,
    nodes=nodes,
    n_points=n_points_list,
    l_max=l_max,
    m_max=m_max,
    nuclear_L_max=nuclear_L_max,
    nuclear_method=nuclear_method,
)
result = diagonalize_fixed_m(
    hamiltonian,
    m=m,
    n_states=n_states,
)

residual = (
    hamiltonian.apply_fixed_m(result.orbital, m)
    - result.electronic_energy * result.orbital
)
residual_norm = np.sqrt(hamiltonian.inner_product(residual, residual).real)
orbital_norm = hamiltonian.inner_product(result.orbital, result.orbital).real

print(f"\n** One-electron diatomic ground state: {name} **")
print(f"Nuclear charges:            ({Z1:g}, {Z2:g})")
print(f"Internuclear distance:      {system.internuclear_distance:.8f} bohr")
print(f"FEM element nodes:          {nodes}")
print(f"Points per element:         {n_points}")
print(f"Orbital angular cutoff:     l_max={l_max}, m={m}")
print(f"Nuclear multipole cutoff:   L_max={nuclear_L_max}, M_max=0")
print(f"Nuclear-potential method:   {nuclear_method}")
print(f"Electronic energy:          {result.electronic_energy:.15f} Eh")
print(f"Nuclear repulsion:          {system.nuclear_repulsion_energy:.15f} Eh")
print(f"Clamped-nuclei total:       {result.total_energy:.15f} Eh")
print(f"Orbital norm:               {orbital_norm:.15f}")
print(f"Eigenstate residual norm:   {residual_norm:.3e}")
