"""Run the Phase 6.6 independent H2+ convergence studies."""

from grid_tdse.spherical.one_radial_becke import (
    run_h2plus_convergence_study,
)


def _print_sweep(title, parameter, points):
    print(f"\n{title}")
    print(
        f"{parameter:>27s} {'N':>5s} {'rank':>9s} {'cond(S)':>12s} "
        f"{'electronic E':>18s} {'|E-E_mid|':>13s}"
    )
    for point in points:
        rank = f"{point.retained_rank}/{point.basis_size}"
        print(
            f"{point.label:>27s} {point.basis_size:5d} {rank:>9s} "
            f"{point.overlap_condition_number:12.4e} "
            f"{point.electronic_energy:18.12f} "
            f"{point.midpoint_energy_error:13.4e}"
        )


def main():
    """Print all one-parameter-at-a-time energy and rank sweeps."""

    study = run_h2plus_convergence_study()
    settings = study.integration_settings
    print("\n** one_radial_becke independent H2+ convergence study **")
    print(f"Internuclear distance: {study.internuclear_distance:g} bohr")
    print(
        "Fixed integration controls except in the Lebedev sweep: "
        f"max dr={settings.radial_spacing:g}, "
        f"radial p={settings.radial_n_points}, "
        f"Lebedev={settings.lebedev_order}"
    )
    print(
        "Fixed midpoint reference: "
        f"E={study.midpoint.electronic_energy:.12f} Eh, "
        "atomic-equivalent extent="
        f"{study.midpoint.radial_extent - 0.5 * study.internuclear_distance:g}, "
        f"p={study.midpoint.radial_n_points}, "
        f"l_max={study.midpoint.l_max}, N={study.midpoint.basis_size}"
    )

    _print_sweep(
        "Radial polynomial order",
        "points / element",
        study.radial_order,
    )
    _print_sweep(
        "Finite-element layout",
        "radial nodes",
        study.element_layout,
    )
    _print_sweep("Radial extent", "r_max", study.radial_extent)
    _print_sweep("Lebedev order", "Lebedev order", study.lebedev_order)
    _print_sweep("Angular cutoff", "l_max", study.angular_cutoff)
    _print_sweep(
        "Orbital Becke smoothing order",
        "iterations",
        study.becke_iterations,
    )
    _print_sweep(
        "Overlap cutoff at baseline",
        "relative cutoff",
        study.overlap_cutoff,
    )
    _print_sweep(
        "Overlap cutoff at l_max=11",
        "relative cutoff",
        study.near_dependent_overlap_cutoff,
    )

    radial_extent_change = abs(
        study.radial_extent[-1].electronic_energy
        - study.radial_extent[-2].electronic_energy
    )
    lebedev_change = abs(
        study.lebedev_order[-1].electronic_energy
        - study.lebedev_order[-2].electronic_energy
    )
    angular_change = abs(
        study.angular_cutoff[-1].electronic_energy
        - study.angular_cutoff[-2].electronic_energy
    )
    near_cutoff_change = abs(
        study.near_dependent_overlap_cutoff[0].electronic_energy
        - study.near_dependent_overlap_cutoff[-1].electronic_energy
    )
    print("\nAssessment")
    print(f"Change from r_max=6 to 8:     {radial_extent_change:.3e} Eh")
    print(f"Change from Lebedev 29 to 41: {lebedev_change:.3e} Eh")
    print(f"Change from l_max=3 to 4:     {angular_change:.3e} Eh")
    print(
        "High-l cutoff shift, 1e-8 to 1e-12: "
        f"{near_cutoff_change:.3e} Eh"
    )
    print(
        "The box extent and angular quadrature are close to stable at the "
        "reported resolution, but radial polynomial/layout and angular-basis "
        "convergence are not complete. At l_max=11, cutoff 1e-8 removes one "
        "direction and changes the energy materially; 1e-10 and 1e-12 agree."
    )


if __name__ == "__main__":
    main()
