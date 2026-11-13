"""Compare fixed-R H2+ excitation energies with Tao et al. Table II.

The reference is Table II of Phys. Rev. A 80, 013402 (2009),
https://doi.org/10.1103/PhysRevA.80.013402. All quantities are in atomic
units. The table comparison deliberately uses the direct validation solver
to obtain deterministic low-lying spectra and symmetry labels. The separate
ground-state example demonstrates the production-default Davidson workflow.

The calculation also exposes an apparent state-label inconsistency in the
published table: the eigenvalue printed for ``4p sigma_u`` has gerade
inversion symmetry on this grid.
"""

from dataclasses import dataclass

import numpy as np

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
)


@dataclass(frozen=True)
class ReferenceState:
    """One rounded entry from Tao et al. Table II."""

    label: str
    m: int
    inversion: int
    electronic_energy: float
    excitation_energy: float


REFERENCE_STATES = (
    ReferenceState("2p sigma_u", 0, -1, -0.667534, 0.43510),
    ReferenceState("3p sigma_u", 0, -1, -0.255413, 0.84722),
    ReferenceState("4p sigma_u", 0, -1, -0.177681, 0.92495),
    ReferenceState("4f sigma_u", 0, -1, -0.126644, 0.97599),
    ReferenceState("2p pi_u", 1, -1, -0.428772, 0.67386),
    ReferenceState("3p pi_u", 1, -1, -0.200865, 0.90177),
    ReferenceState("4f pi_u", 1, -1, -0.126199, 0.97643),
)


# H2+ at the internuclear distance used in Table II.
system = OneElectronDiatomicSystem(
    nuclear_charge1=1.0,
    nuclear_charge2=1.0,
    half_bond_length=1.0,
    particle_mass=1.0,
    name="H2+",
)

# Tao et al. Table-I grid: one xi element from 1 to 2, ten equal elements
# from 2 to 80, a 15-point DVR in each element, and 9 eta points.
xi_boundaries = np.concatenate(([1.0], np.linspace(2.0, 80.0, 11)))
hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
    system,
    xi_boundaries=xi_boundaries,
    xi_points_per_element=15,
    eta_degree=9,
    m_max=1,
)

# Nine Sigma roots reach 4f sigma_u; six Pi roots reach 4f pi_u.
sectors = {
    0: diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=9,
        solver="direct",
    ),
    1: diagonalize_fixed_m(
        hamiltonian,
        m=1,
        n_states=6,
        solver="direct",
    ),
}
ground_energy = sectors[0].electronic_energy

print("\n** H2+ prolate excitation energies at R=2 **")
print(f"Ground electronic energy: {ground_energy:.12f} Eh")
print(
    f"{'Table-II state':16s} {'m':>2s} {'root':>4s} {'sym':>3s} "
    f"{'calculated En':>16s} {'calculated dE':>16s} "
    f"{'Table-II dE':>12s} {'dE error':>11s}"
)

symmetry_mismatches = []
for reference in REFERENCE_STATES:
    result = sectors[reference.m]
    root = int(
        np.argmin(
            np.abs(
                result.electronic_energies
                - reference.electronic_energy
            )
        )
    )
    electronic_energy = float(result.electronic_energies[root])
    excitation_energy = electronic_energy - ground_energy
    inversion = hamiltonian.inversion_expectation(
        result.orbitals[root], reference.m
    )
    inversion_sign = 1 if inversion > 0.0 else -1
    symmetry = "g" if inversion_sign == 1 else "u"
    if inversion_sign != reference.inversion:
        symmetry_mismatches.append((reference, root, inversion))

    print(
        f"{reference.label:16s} {reference.m:2d} {root:4d} "
        f"{symmetry:>3s} {electronic_energy:16.12f} "
        f"{excitation_energy:16.12f} "
        f"{reference.excitation_energy:12.5f} "
        f"{excitation_energy - reference.excitation_energy:+11.3e}"
    )

print("\nLargest eigenstate residual:", end=" ")
print(
    f"{max(np.max(result.residual_norms) for result in sectors.values()):.3e}"
)

for reference, root, inversion in symmetry_mismatches:
    result = sectors[reference.m]
    matching_roots = [
        index
        for index, orbital in enumerate(result.orbitals)
        if np.sign(
            hamiltonian.inversion_expectation(orbital, reference.m)
        )
        == reference.inversion
    ]
    nearest_matching_root = min(
        matching_roots,
        key=lambda index: abs(
            result.electronic_energies[index]
            - reference.electronic_energy
        ),
    )
    matching_energy = float(
        result.electronic_energies[nearest_matching_root]
    )
    print(
        "\nSymmetry warning: "
        f"the eigenvalue nearest the Table-II {reference.label} entry "
        f"is root {root} with <P>={inversion:+.12f}."
    )
    print(
        "The nearest computed root with the table's ungerade symmetry is "
        f"root {nearest_matching_root}: En={matching_energy:.12f} Eh, "
        f"dE={matching_energy - ground_energy:.12f} Eh."
    )
