"""Study radial polynomial-order convergence for a one-electron diatomic.

The geometry and radial element boundaries match
``one_electron_diatomic_ground_state.py``.  All quantities are in atomic
units.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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

# Keep the geometry and radial element boundaries aligned with the ground-state
# example.  The nuclear radius a is an internal FEM element boundary.
nodes = [0.0, a, 2 * a, 4.0, 8.0, 16.0]

# Axial nuclei conserve m.  The ground state belongs to m=0.
m = 0
m_max = 0
nuclear_method = "poisson"


def parse_arguments():
    """Parse command-line convergence parameters."""

    parser = argparse.ArgumentParser(
        description=(
            "Test FEM-DVR p-convergence of the clamped-nuclei H2+ "
            "ground-state energy."
        )
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=10,
        help="orbital angular cutoff (default: 10)",
    )
    parser.add_argument(
        "--p-max",
        type=int,
        default=21,
        help="largest odd number of points per element (default: 21)",
    )
    arguments = parser.parse_args()
    if arguments.l_max < 0:
        parser.error("--l-max must be non-negative")
    if arguments.p_max < 7 or arguments.p_max % 2 == 0:
        parser.error("--p-max must be an odd integer greater than or equal to 7")
    return arguments


def ground_state_energy(system, *, p, l_max):
    """Return the clamped-nuclei ground-state energy for one radial order."""

    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=nodes,
        n_points=[p] * (len(nodes) - 1),
        l_max=l_max,
        m_max=m_max,
        nuclear_L_max=2 * l_max,
        nuclear_method=nuclear_method,
    )
    result = diagonalize_fixed_m(hamiltonian, m=m, n_states=1)
    return result.total_energy


def output_path(*, l_max):
    """Return a descriptive PDF filename for the convergence figure."""

    label = name.replace("+", "plus").replace("-", "minus")
    r_max = nodes[-1]
    return Path(
        f"{label}_p_convergence_R{2 * a:g}_rmax{r_max:g}_lmax{l_max}.pdf"
    )


def main():
    """Run the convergence sequence and save its relative-energy plot."""

    arguments = parse_arguments()
    p_values = np.arange(5, arguments.p_max + 1, 2, dtype=int)
    system = OneElectronDiatomicSystem(
        nuclear_charge1=Z1,
        nuclear_charge2=Z2,
        half_bond_length=a,
        particle_mass=particle_mass,
        name=name,
    )

    energies = np.empty(p_values.size)
    print(f"\n** One-electron diatomic p-convergence: {name} **")
    print(f"Internuclear distance:    {system.internuclear_distance:g} bohr")
    print(f"FEM element nodes:        {nodes}")
    print(f"Orbital angular cutoff:   l_max={arguments.l_max}, m={m}")
    print(f"Nuclear multipole cutoff: L_max={2 * arguments.l_max}, M_max=0")
    print(f"Nuclear-potential method: {nuclear_method}")
    print("\n   p       Clamped-nuclei energy (Eh)")
    print("  ---      ---------------------------")
    for index, p in enumerate(p_values):
        energies[index] = ground_state_energy(
            system,
            p=int(p),
            l_max=arguments.l_max,
        )
        print(f"  {p:3d}      {energies[index]: .12f}")

    reference_energy = energies[-1]
    relative_errors = np.abs(energies - reference_energy) / abs(reference_energy)
    print(f"\nReference energy E(p_max): {reference_energy:.15f} Eh")
    print("\n   p       |E(p)-E(p_max)|/|E(p_max)|")
    print("  ---      ---------------------------")
    for p, relative_error in zip(p_values, relative_errors, strict=True):
        print(f"  {p:3d}      {relative_error:.6e}")

    positive = relative_errors > 0.0
    figure, axes = plt.subplots(figsize=(6.4, 4.4))
    axes.semilogy(
        p_values[positive],
        relative_errors[positive],
        "o-",
        label=(
            rf"$E(p_{{\max}}={arguments.p_max})"
            rf"={reference_energy:.15f}\,E_h$"
        ),
    )
    axes.set_xlabel("Points per finite element, $p$")
    axes.set_ylabel(r"$|E(p)-E(p_{\max})|/|E(p_{\max})|$")
    axes.set_xticks(p_values)
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()

    filename = output_path(l_max=arguments.l_max)
    figure.savefig(filename)
    plt.close(figure)
    print(f"\nSaved convergence figure to {filename}")


if __name__ == "__main__":
    main()
