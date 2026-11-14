"""Validate heteronuclear one-electron two-centre energies.

The numerical references are from Tables I, II, VI, and IX of

J. A. Campos et al., Int. J. Quantum Chem. 106, 2587 (2006),
https://doi.org/10.1002/qua.21015.

The exact BH5+ Demkov solution is reported by

M. V. Khoma, arXiv:2503.18280,
https://arxiv.org/abs/2503.18280.

All quantities are in atomic units. Published values are electronic
eigenvalues: the nuclear repulsion ``Z1*Z2/R`` is added separately when the
clamped-nuclei total energy is printed.
"""

from dataclasses import dataclass
from math import sqrt

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
)


@dataclass(frozen=True)
class ReferenceCase:
    """One published fixed-nuclei electronic energy."""

    system: str
    state: str
    Z1: float
    Z2: float
    R: float
    root: int
    electronic_energy: float
    source: str


REFERENCE_CASES = (
    ReferenceCase(
        "HeH2+",
        "1s sigma",
        1.0,
        2.0,
        2.0,
        0,
        -2.512193016590,
        "Campos Table II",
    ),
    ReferenceCase(
        "LiH3+",
        "1s sigma",
        1.0,
        3.0,
        2.0,
        0,
        -5.002042016849,
        "Campos Table VI",
    ),
    ReferenceCase(
        "LiH3+",
        "1s sigma",
        1.0,
        3.0,
        4.0,
        0,
        -4.750111784789,
        "Campos Table VI",
    ),
    ReferenceCase(
        "BeH4+",
        "1s sigma",
        1.0,
        4.0,
        2.0,
        0,
        -8.500592345526,
        "Campos Table IX",
    ),
    ReferenceCase(
        "BH5+",
        "3p sigma",
        1.0,
        5.0,
        sqrt(10.0) / 3.0,
        4,
        -2.0,
        "Khoma exact Demkov solution",
    ),
)


# A common validation grid is used for every charge and bond length. It is
# deliberately more resolved than the compact H2+ development example.
xi_boundaries = [1.0, 4.0, 14.0, 24.0]
xi_points_per_element = 18
eta_degree = 24
maximum_allowed_error = 5.0e-10


print("\n** Heteronuclear one-electron fixed-nuclei validation **")
print(f"xi element boundaries:      {xi_boundaries}")
print(f"xi points per element:      {xi_points_per_element}")
print(f"eta Gauss-Legendre points:  {eta_degree}")
print(f"maximum allowed error:      {maximum_allowed_error:.1e} Eh")
print(
    f"\n{'system':7s} {'state':9s} {'R':>12s} {'root':>4s} "
    f"{'calculated Eel':>18s} {'reference Eel':>18s} "
    f"{'abs. error':>11s} {'total E':>18s}"
)

errors = []
residuals = []
for reference in REFERENCE_CASES:
    system = OneElectronDiatomicSystem(
        nuclear_charge1=reference.Z1,
        nuclear_charge2=reference.Z2,
        half_bond_length=reference.R / 2.0,
        particle_mass=1.0,
        name=reference.system,
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=xi_boundaries,
        xi_points_per_element=xi_points_per_element,
        eta_degree=eta_degree,
        m_max=0,
    )
    result = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=reference.root + 1,
        solver="direct",
    )
    electronic_energy = float(
        result.electronic_energies[reference.root]
    )
    total_energy = electronic_energy + system.nuclear_repulsion_energy
    error = abs(electronic_energy - reference.electronic_energy)
    residual = float(result.residual_norms[reference.root])
    errors.append(error)
    residuals.append(residual)

    print(
        f"{reference.system:7s} {reference.state:9s} "
        f"{reference.R:12.9f} {reference.root:4d} "
        f"{electronic_energy:18.12f} "
        f"{reference.electronic_energy:18.12f} "
        f"{error:11.3e} {total_energy:18.12f}"
    )

largest_error = max(errors)
largest_residual = max(residuals)
print(f"\nLargest absolute energy error: {largest_error:.3e} Eh")
print(f"Largest eigenstate residual:    {largest_residual:.3e}")

if largest_error > maximum_allowed_error:
    raise RuntimeError(
        "heteronuclear validation exceeded the allowed energy error: "
        f"{largest_error:.3e} > {maximum_allowed_error:.3e} Eh"
    )

print("Heteronuclear validation: PASS")
