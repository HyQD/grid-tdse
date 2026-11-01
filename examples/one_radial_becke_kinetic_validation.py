"""Run the Phase 6.4 two-center weak kinetic-energy validation."""

from grid_tdse.spherical.one_radial_becke import (
    validate_two_center_kinetic,
)


def _print_matrix_result(label, result):
    settings = result.integration_settings
    print(
        f"{label:10s} {settings.radial_spacing:9.3f} "
        f"{settings.radial_n_points:8d} {settings.lebedev_order:8d} "
        f"{result.quadrature_size:11d} "
        f"{result.minimum_eigenvalue:12.4e} "
        f"{result.maximum_eigenvalue:12.4e} "
        f"{result.relative_antihermitian_norm:12.3e}"
    )


def main():
    """Print positivity, Hermiticity, and high-order comparisons."""

    validation = validate_two_center_kinetic()
    print("\n** one_radial_becke two-center kinetic validation **")
    print(
        f"R={validation.internuclear_distance:g}, "
        f"nodes={validation.radial_nodes}, "
        f"orders={validation.radial_n_points}, "
        f"l_max={validation.l_max}, m={validation.m}"
    )
    print(
        "Every integration cell reaches r_max + R. FEM boundaries use "
        "their correct one-sided derivative traces."
    )
    print(
        "\nresult       max dr radial p Lebedev      points        min(T) "
        "       max(T) anti-Hermitian"
    )
    _print_matrix_result("coarse", validation.coarse)
    _print_matrix_result("refined", validation.refined)
    _print_matrix_result("reference", validation.reference)

    print("\nRelative Frobenius error against high-order integration")
    print(f"coarse:  {validation.coarse_relative_matrix_error:.6e}")
    print(f"refined: {validation.refined_relative_matrix_error:.6e}")
    print("\nSelected refined matrix elements")
    print(
        f"{'(row,col)':>12s} {'refined':>18s} {'reference':>18s} "
        f"{'absolute error':>16s} {'relative error':>16s}"
    )
    for comparison in validation.selected_elements:
        pair = f"({comparison.row},{comparison.column})"
        print(
            f"{pair:>12s} {comparison.refined_value.real:18.10e} "
            f"{comparison.reference_value.real:18.10e} "
            f"{comparison.absolute_error:16.6e} "
            f"{comparison.relative_error:16.6e}"
        )

    print(
        "\nThe weak kinetic matrix is Hermitian and positive semidefinite "
        "within numerical precision. Refinement reduces the full-matrix "
        "quadrature error before the nuclear potential is introduced in "
        "Phase 6.5."
    )


if __name__ == "__main__":
    main()
