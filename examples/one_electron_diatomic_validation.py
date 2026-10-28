"""Validate the spherical one-electron H2+ ground-state calculation.

The calculation uses fixed nuclei at ``z = +/-1`` bohr.  It studies the
radial box, radial element layout, orbital angular cutoff, and nuclear
multipole cutoff independently.  A prolate-spheroidal calculation provides
an independent coordinate-system cross-check.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy.sparse.linalg import eigsh

from grid_lib.prolate_spheroidal_coordinates import (
    active_indices,
    electron_nuclear_coulomb_matrix,
    kinetic_energy_matrix,
    overlap_matrix,
    restrict_matrix,
    setup_prolate_spheroidal_grid,
)
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    diagonalize_fixed_m,
)


REFERENCE_ELECTRONIC_ENERGY = -1.1026342144949464615
REFERENCE_TOTAL_ENERGY = REFERENCE_ELECTRONIC_ENERGY + 0.5

SYSTEM = OneElectronDiatomicSystem(1.0, 1.0, 1.0, name="H2+")
BASE_NODES = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
M = 0


def parse_arguments():
    """Parse validation controls."""

    parser = argparse.ArgumentParser(
        description="Validate spherical H2+ energies at R=2 bohr."
    )
    parser.add_argument(
        "--angular-l-max",
        type=int,
        default=60,
        help="largest orbital angular cutoff in the angular study (default: 60)",
    )
    parser.add_argument(
        "--points-per-element",
        type=int,
        default=21,
        help="radial order used outside the element-size study (default: 21)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("H2plus_spherical_convergence_R2.pdf"),
        help="output PDF path",
    )
    parser.add_argument(
        "--skip-prolate",
        action="store_true",
        help="skip the independent prolate-spheroidal cross-check",
    )
    arguments = parser.parse_args()
    if arguments.angular_l_max < 4:
        parser.error("--angular-l-max must be at least 4")
    if arguments.points_per_element < 5:
        parser.error("--points-per-element must be at least 5")
    return arguments


def spherical_energy(
    *,
    nodes,
    points_per_element,
    l_max,
    nuclear_L_max=None,
    nuclear_method="analytic",
):
    """Return one spherical clamped-nuclei H2+ ground-state energy."""

    hamiltonian = OneElectronDiatomicHamiltonian(
        SYSTEM,
        nodes=nodes,
        n_points=[points_per_element] * (len(nodes) - 1),
        l_max=l_max,
        m_max=0,
        nuclear_L_max=(2 * l_max if nuclear_L_max is None else nuclear_L_max),
        nuclear_method=nuclear_method,
    )
    return diagonalize_fixed_m(hamiltonian, m=M).total_energy


def prolate_reference_energy():
    """Return an independent prolate-spheroidal H2+ energy."""

    grid = setup_prolate_spheroidal_grid(
        internuclear_distance=2.0,
        xi_max=20.0,
        xi_elements=8,
        points_per_element=10,
        eta_degree=32,
    )
    indices = active_indices(grid, m=0)
    hamiltonian = kinetic_energy_matrix(grid, m=0, active_indices=indices)
    hamiltonian += restrict_matrix(
        electron_nuclear_coulomb_matrix(grid), indices
    )
    overlap = restrict_matrix(overlap_matrix(grid), indices)
    electronic_energy = eigsh(
        hamiltonian,
        k=1,
        M=overlap,
        sigma=-1.1,
        which="LM",
        return_eigenvectors=False,
        tol=1.0e-12,
    )[0]
    return float(electronic_energy + SYSTEM.nuclear_repulsion_energy)


def angular_cutoffs(maximum):
    """Return a compact angular convergence sequence including ``maximum``."""

    candidates = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60]
    values = [value for value in candidates if value <= maximum]
    if values[-1] != maximum:
        values.append(maximum)
    return np.asarray(values, dtype=int)


def print_table(title, labels, energies, *, label_header):
    """Print one convergence table."""

    print(f"\n{title}")
    print(f"{label_header:>12s}       Energy (Eh)")
    print(f"{'-' * len(label_header):>12s}       ------------------")
    for label, energy in zip(labels, energies, strict=True):
        print(f"{str(label):>12s}      {energy: .15f}")


def positive_error(values, reference):
    """Return values and errors for points representable on a log axis."""

    values = np.asarray(values)
    errors = np.abs(values - reference)
    mask = errors > 0.0
    return mask, errors


def main():
    """Run all convergence studies and save the summary figure."""

    arguments = parse_arguments()
    radial_l_max = min(12, arguments.angular_l_max)
    p = arguments.points_per_element

    box_nodes = [
        [0.0, 1.0, 2.0, 4.0],
        [0.0, 1.0, 2.0, 4.0, 6.0],
        [0.0, 1.0, 2.0, 4.0, 8.0],
        [0.0, 1.0, 2.0, 4.0, 8.0, 12.0],
        BASE_NODES,
    ]
    box_energies = np.asarray(
        [
            spherical_energy(
                nodes=nodes,
                points_per_element=p,
                l_max=radial_l_max,
            )
            for nodes in box_nodes
        ]
    )

    element_nodes = [
        BASE_NODES,
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
        ],
    ]
    element_p = 7
    element_energies = np.asarray(
        [
            spherical_energy(
                nodes=nodes,
                points_per_element=element_p,
                l_max=radial_l_max,
            )
            for nodes in element_nodes
        ]
    )

    l_values = angular_cutoffs(arguments.angular_l_max)
    angular_energies = np.asarray(
        [
            spherical_energy(
                nodes=BASE_NODES,
                points_per_element=p,
                l_max=int(l_max),
            )
            for l_max in l_values
        ]
    )

    nuclear_l_max = radial_l_max
    maximum_nuclear_L = 2 * nuclear_l_max
    nuclear_L_values = np.unique(
        np.asarray(
            [
                0,
                2,
                4,
                8,
                12,
                16,
                20,
                maximum_nuclear_L - 2,
                maximum_nuclear_L,
                maximum_nuclear_L + 2,
            ],
            dtype=int,
        )
    )
    nuclear_L_values = nuclear_L_values[nuclear_L_values >= 0]
    nuclear_energies = np.asarray(
        [
            spherical_energy(
                nodes=BASE_NODES,
                points_per_element=p,
                l_max=nuclear_l_max,
                nuclear_L_max=int(L_max),
            )
            for L_max in nuclear_L_values
        ]
    )

    poisson_energy = spherical_energy(
        nodes=BASE_NODES,
        points_per_element=p,
        l_max=radial_l_max,
        nuclear_method="poisson",
    )

    print("\n** Spherical one-electron H2+ validation at R=2 bohr **")
    print(f"Published total energy: {REFERENCE_TOTAL_ENERGY:.15f} Eh")
    print_table(
        "Radial-box convergence",
        [nodes[-1] for nodes in box_nodes],
        box_energies,
        label_header="r_max",
    )
    print_table(
        f"Element-size convergence (p={element_p})",
        [max(np.diff(nodes)) for nodes in element_nodes],
        element_energies,
        label_header="max(h)",
    )
    print_table(
        "Orbital angular convergence",
        l_values,
        angular_energies,
        label_header="l_max",
    )
    print_table(
        f"Nuclear multipole convergence (l_max={nuclear_l_max})",
        nuclear_L_values,
        nuclear_energies,
        label_header="L_max",
    )
    analytic_energy = angular_energies[np.where(l_values == radial_l_max)[0][0]]
    print("\nNuclear-potential construction")
    print(f"Analytic: {analytic_energy: .15f} Eh")
    print(f"Poisson:  {poisson_energy: .15f} Eh")
    print(f"Difference: {abs(analytic_energy - poisson_energy):.3e} Eh")

    if not arguments.skip_prolate:
        prolate_energy = prolate_reference_energy()
        print("\nIndependent prolate-spheroidal cross-check")
        print(f"Computed total energy:  {prolate_energy:.15f} Eh")
        print(f"Published total energy: {REFERENCE_TOTAL_ENERGY:.15f} Eh")
        prolate_error = abs(prolate_energy - REFERENCE_TOTAL_ENERGY)
        print(f"Absolute difference: {prolate_error:.3e} Eh")

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))

    r_max = np.asarray([nodes[-1] for nodes in box_nodes])
    mask, errors = positive_error(box_energies, box_energies[-1])
    axes[0, 0].semilogy(r_max[mask], errors[mask], "o-")
    axes[0, 0].set_xlabel(r"$r_{\max}$ (bohr)")
    axes[0, 0].set_ylabel(r"$|E-E(r_{\max}=16)|$ ($E_h$)")
    axes[0, 0].set_title("Radial box")

    widths = np.asarray([max(np.diff(nodes)) for nodes in element_nodes])
    mask, errors = positive_error(element_energies, element_energies[-1])
    axes[0, 1].loglog(widths[mask], errors[mask], "o-")
    axes[0, 1].invert_xaxis()
    axes[0, 1].set_xticks([8, 4], ["8", "4"])
    axes[0, 1].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 1].set_xlabel("Maximum element width (bohr)")
    axes[0, 1].set_ylabel(r"$|E-E_{\mathrm{finest}}|$ ($E_h$)")
    axes[0, 1].set_title(f"Element size ($p={element_p}$)")

    angular_errors = np.abs(angular_energies - REFERENCE_TOTAL_ENERGY)
    axes[1, 0].loglog(l_values[1:], angular_errors[1:], "o-")
    angular_ticks = [
        value
        for value in (2, 4, 8, 16, 32, 60)
        if value <= arguments.angular_l_max
    ]
    axes[1, 0].set_xticks(angular_ticks, [str(value) for value in angular_ticks])
    axes[1, 0].xaxis.set_minor_formatter(NullFormatter())
    axes[1, 0].set_xlabel(r"Orbital cutoff $l_{\max}$")
    axes[1, 0].set_ylabel(r"$|E-E_{\mathrm{ref}}|$ ($E_h$)")
    axes[1, 0].set_title("Angular expansion")

    saturated = nuclear_energies[nuclear_L_values == maximum_nuclear_L][0]
    mask, errors = positive_error(nuclear_energies, saturated)
    axes[1, 1].semilogy(nuclear_L_values[mask], errors[mask], "o-")
    axes[1, 1].axvline(maximum_nuclear_L, color="black", linestyle="--")
    axes[1, 1].set_xlabel(r"Nuclear cutoff $L_{\max}$")
    axes[1, 1].set_ylabel(r"$|E-E(L_{\max}=2l_{\max})|$ ($E_h$)")
    axes[1, 1].set_title("Nuclear multipoles")

    for axis in axes.ravel():
        axis.grid(True, which="both", alpha=0.3)
    figure.suptitle(
        rf"H$_2^+$ at $R=2$: $E_{{\mathrm{{ref}}}}="
        rf"{REFERENCE_TOTAL_ENERGY:.15f}\,E_h$"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(arguments.output)
    plt.close(figure)
    print(f"\nSaved validation figure to {arguments.output}")


if __name__ == "__main__":
    main()
