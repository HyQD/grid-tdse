"""Compute a clamped one-electron diatomic ground state with Davidson.

All quantities are in atomic units. The development calculation uses H2+ at
R=2 on the Haxton-comparable prolate grid documented by ``grid-lib``. Change
``Z1`` and ``Z2`` to describe other positive two-centre charges.
"""

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
)


# Physical system: nuclei at z=-a and z=+a, with R12=2*a.
name = "H2+"
Z1 = 1.0
Z2 = 1.0
a = 1.0
particle_mass = 1.0

# Haxton-comparable development grid; it is not a convergence prescription.
xi_boundaries = [1.0, 4.0, 14.0, 24.0]
xi_points_per_element = 14
eta_degree = 9

# An axial field-free Hamiltonian conserves signed m.
m = 0
m_max = 0
n_states = 1


system = OneElectronDiatomicSystem(
    nuclear_charge1=Z1,
    nuclear_charge2=Z2,
    half_bond_length=a,
    particle_mass=particle_mass,
    name=name,
)
hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
    system,
    xi_boundaries=xi_boundaries,
    xi_points_per_element=xi_points_per_element,
    eta_degree=eta_degree,
    m_max=m_max,
)
result = diagonalize_fixed_m(
    hamiltonian,
    m=m,
    n_states=n_states,
    residual_tolerance=1.0e-10,
)

orbital_norm = hamiltonian.inner_product(
    result.orbital, result.orbital, m
).real

print(f"\n** Prolate one-electron diatomic ground state: {name} **")
print(f"Nuclear charges:            ({Z1:g}, {Z2:g})")
print(f"Internuclear distance:      {system.internuclear_distance:.8f} bohr")
print(f"xi element boundaries:      {xi_boundaries}")
print(f"xi points per element:      {xi_points_per_element}")
print(f"eta Gauss-Legendre points:  {eta_degree}")
print(f"Magnetic quantum number:    m={m}")
print(f"Active state shape:         {hamiltonian.fixed_m_shape(m)}")
print(f"Electronic energy:          {result.electronic_energy:.15f} Eh")
print(f"Nuclear repulsion:          {system.nuclear_repulsion_energy:.15f} Eh")
print(f"Clamped-nuclei total:       {result.total_energy:.15f} Eh")
print(f"Orbital norm:               {orbital_norm:.15f}")
print(f"Eigenstate residual norm:   {result.residual_norm:.3e}")
print(f"Davidson converged:         {result.converged}")
print(f"Termination reason:         {result.termination_reason}")
print(f"Davidson iterations:        {result.iterations}")
print(f"Hamiltonian actions:        {result.hamiltonian_actions}")
print(f"Davidson restarts:          {result.restarts}")
print(f"Preconditioner fallbacks:   {result.preconditioner_fallbacks}")
print(f"Regularized denominators:   {result.regularized_denominators}")
