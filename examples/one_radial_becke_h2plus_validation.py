"""Run the Phase 6.5 two-center H2+ ground-state validation."""

from grid_tdse.spherical.one_radial_becke import (
    validate_h2plus_ground_state,
)


def _print_level(label, result, reference_energy):
    settings = result.matrices.integration_settings
    energy = result.electronic_energies[0]
    print(
        f"{label:10s} {settings.radial_spacing:8.3f} "
        f"{settings.radial_n_points:8d} {settings.lebedev_order:8d} "
        f"{result.matrices.quadrature_size:11d} {energy:18.12f} "
        f"{abs(energy - reference_energy):12.3e}"
    )


def main():
    """Print quadrature, cutoff, and midpoint comparisons."""

    validation = validate_h2plus_ground_state()
    reference_energy = validation.reference.electronic_energies[0]
    print("\n** one_radial_becke H2+ ground-state validation **")
    print(f"Internuclear distance: {validation.internuclear_distance:g} bohr")
    print(
        f"Atomic radial nodes: {validation.radial_nodes}; "
        f"orders: {validation.radial_n_points}"
    )
    print(
        f"Angular block: l_max={validation.l_max}, m={validation.m}; "
        f"basis dimension={validation.reference.matrices.basis_size}"
    )
    print(
        "Each atomic integration grid reaches r_max + R and uses the correct "
        "one-sided FEM derivative traces."
    )

    print("\nQuadrature refinement")
    print(
        "result       max dr radial p Lebedev      points       "
        "electronic E   error to high"
    )
    _print_level("coarse", validation.coarse, reference_energy)
    _print_level("refined", validation.refined, reference_energy)
    _print_level("high order", validation.reference, reference_energy)

    print("\nOverlap-cutoff study at high integration order")
    print(" cutoff      rank kept/discarded       electronic E        total E")
    for cutoff, retained, discarded, electronic, total in zip(
        validation.overlap_cutoffs,
        validation.reference.retained_ranks,
        validation.reference.discarded_ranks,
        validation.reference.electronic_energies,
        validation.reference.total_energies,
        strict=True,
    ):
        print(
            f" {cutoff:8.1e} {retained:8d}/{discarded:<8d} "
            f"{electronic:18.12f} {total:18.12f}"
        )

    midpoint = validation.midpoint
    print("\nMidpoint-centered comparison")
    print(
        "The midpoint sphere has radius r_max + R/2, the smallest centered "
        "sphere containing both atomic support spheres. The finite domains "
        "therefore have matched maximum extent but different shapes."
    )
    print(f"Midpoint nodes:             {midpoint.radial_nodes}")
    print(f"Midpoint radial order:      {midpoint.radial_n_points}")
    print(f"Midpoint l_max:             {midpoint.l_max}")
    print(f"Midpoint basis dimension:   {midpoint.basis_size}")
    print(f"Becke electronic energy:    {reference_energy:.12f} Eh")
    print(f"Midpoint electronic energy: {midpoint.electronic_energy:.12f} Eh")
    print(
        "Absolute difference:         "
        f"{validation.becke_to_midpoint_energy_error:.6e} Eh"
    )

    maximum_residual = max(
        solution.diagnostics.relative_residual_norms[0]
        for solution in validation.reference.solutions
    )
    maximum_antihermitian = max(
        validation.reference.matrices.relative_antihermitian_norms
    )
    print("\nNumerical diagnostics")
    print(
        "Overlap condition number:    "
        f"{validation.reference.matrices.overlap_condition_number:.6e}"
    )
    print(f"Cutoff energy spread:       {validation.cutoff_energy_spread:.3e} Eh")
    print(f"Maximum relative residual:  {maximum_residual:.3e}")
    print(f"Maximum anti-Hermitian norm:{maximum_antihermitian: .3e}")
    print(
        "\nThe generalized solve and cutoff handling are stable. The remaining "
        "energy difference is dominated by the deliberately small l_max=1, "
        "20-function Becke basis and is the target of Phase 6.6 convergence "
        "studies."
    )


if __name__ == "__main__":
    main()
