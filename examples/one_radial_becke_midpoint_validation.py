"""Validate Becke integration of the existing midpoint-centered basis.

This Phase 6.2 calculation separates multicenter quadrature error from the
finite-order FEM-DVR mass-lumping and pointwise-potential approximations used
by the structured ``one_radial`` representation.
"""

from grid_tdse.spherical.one_radial_becke import validate_midpoint_l0


def _print_errors(label, errors):
    print(f"{label:28s} {errors.overlap:11.3e} {errors.kinetic:11.3e} "
          f"{errors.potential:11.3e} {errors.hamiltonian:11.3e}")


def main():
    """Run coarse and refined partitioned integrations."""

    coarse = validate_midpoint_l0(
        integration_radial_spacing=1.0,
        integration_n_points=6,
        lebedev_order=17,
    )
    refined = validate_midpoint_l0()

    print("\n** one_radial_becke midpoint-basis validation **")
    print(f"Midpoint nodes: {refined.midpoint_nodes}")
    print(f"Midpoint FEM orders: {refined.midpoint_n_points}")
    print("\nRelative Frobenius matrix errors")
    print("                                      S           T           V           H")
    _print_errors("coarse Becke / continuum", coarse.partitioned_to_continuum)
    _print_errors("refined Becke / continuum", refined.partitioned_to_continuum)
    _print_errors("structured DVR / continuum", refined.structured_to_continuum)

    print("\nRefined integration settings")
    print(
        f"Atomic radial spacing:       {refined.integration_radial_spacing:g} bohr"
    )
    print(f"Points per radial element:  {refined.integration_n_points}")
    print(f"Lebedev order:              {refined.lebedev_order}")
    print(f"Total quadrature points:    {refined.partitioned_quadrature_size}")
    print(
        "\nThe structured weak kinetic matrix is the same continuum Galerkin "
        "matrix to roundoff. The finite-order structured overlap and potential "
        "use the intended DVR diagonal quadrature; their difference from the "
        "continuum matrices is therefore not a Becke-integration error."
    )


if __name__ == "__main__":
    main()
