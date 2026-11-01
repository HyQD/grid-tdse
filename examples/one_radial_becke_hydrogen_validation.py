"""Validate the single-center hydrogen limit of ``one_radial_becke``.

The calculation uses one orbital partition and one independent integration
partition.  Both are identically one for a single center.  It reports radial
convergence of the 1s and 2s energies, the lowest circular state in the
``s``, ``p``, ``d``, and ``f`` channels, kinetic positivity, metric
normalization, and centered angular selection-rule errors.
"""

import numpy as np

from grid_tdse.spherical.one_radial_becke import (
    BeckeMulticenterQuadrature,
    BeckePartition,
    BeckeSphericalBasis,
    FEMDVRQuadratureSpec,
    assemble_becke_matrices,
    fixed_m_channels,
    full_lm_channels,
    solve_becke_eigenproblem,
)


RADIAL_NODES = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
RADIAL_ORDERS = (4, 5, 6)
OVERLAP_CUTOFF = 1.0e-12


def hydrogen_channel(ell, radial_order, *, number_of_states=1):
    """Assemble and solve one hydrogen channel with ``m = ell``."""

    n_elements = len(RADIAL_NODES) - 1
    orbital_partition = BeckePartition([[0.0, 0.0, 0.0]])
    integration_partition = BeckePartition([[0.0, 0.0, 0.0]])
    basis = BeckeSphericalBasis(
        orbital_partition,
        FEMDVRQuadratureSpec(
            RADIAL_NODES, [radial_order] * n_elements
        ),
        fixed_m_channels(ell, ell),
    )
    quadrature = BeckeMulticenterQuadrature(
        integration_partition,
        FEMDVRQuadratureSpec(
            RADIAL_NODES, [2 * radial_order] * n_elements
        ),
        17,
    )
    matrices = assemble_becke_matrices(basis, quadrature, [1.0])
    solution = solve_becke_eigenproblem(
        matrices,
        relative_cutoff=OVERLAP_CUTOFF,
        number_of_states=number_of_states,
    )
    return basis, quadrature, matrices, solution


def angular_selection_errors():
    """Return relative off-channel matrix norms for a centered atom."""

    orbital_partition = BeckePartition([[0.0, 0.0, 0.0]])
    integration_partition = BeckePartition([[0.0, 0.0, 0.0]])
    basis = BeckeSphericalBasis(
        orbital_partition,
        FEMDVRQuadratureSpec([0.0, 1.0, 3.0, 7.0], [4, 4, 4]),
        full_lm_channels(2),
    )
    quadrature = BeckeMulticenterQuadrature(
        integration_partition,
        FEMDVRQuadratureSpec([0.0, 1.0, 3.0, 7.0], [8, 8, 8]),
        17,
    )
    matrices = assemble_becke_matrices(basis, quadrature, [1.0])
    channels = np.asarray(
        [(function.ell, function.m) for function in basis.functions]
    )
    off_channel = np.any(
        channels[:, np.newaxis, :] != channels[np.newaxis, :, :], axis=-1
    )
    return {
        name: float(np.linalg.norm(matrix[off_channel]) / np.linalg.norm(matrix))
        for name, matrix in (
            ("overlap", matrices.overlap),
            ("kinetic", matrices.kinetic),
            ("potential", matrices.potential),
            ("hamiltonian", matrices.hamiltonian),
        )
    }


def main():
    """Run and print the Phase 6.1 validation."""

    print("\n** one_radial_becke single-center hydrogen validation **")
    print(f"Radial nodes: {RADIAL_NODES}")
    print(f"Overlap cutoff: {OVERLAP_CUTOFF:.1e}")
    print("\nRadial convergence in the l=0 channel")
    print(
        " order  basis  quadrature          E(1s)       error       "
        "E(2s)       error"
    )
    final_matrices = None
    final_solution = None
    for radial_order in RADIAL_ORDERS:
        basis, quadrature, matrices, solution = hydrogen_channel(
            0, radial_order, number_of_states=2
        )
        errors = solution.energies - np.array([-0.5, -0.125])
        print(
            f" {radial_order:5d} {basis.size:6d} {quadrature.point_count:11d} "
            f"{solution.energies[0]: .10f} {errors[0]: .2e} "
            f"{solution.energies[1]: .10f} {errors[1]: .2e}"
        )
        final_matrices = matrices
        final_solution = solution

    print("\nLowest hydrogenic circular states")
    print(" ell   m   n          energy          exact       error")
    for ell in range(4):
        _, _, _, solution = hydrogen_channel(
            ell, RADIAL_ORDERS[-1], number_of_states=1
        )
        exact_energy = -0.5 / (ell + 1) ** 2
        print(
            f" {ell:3d} {ell:3d} {ell + 1:3d} "
            f"{solution.energies[0]: .12f} {exact_energy: .12f} "
            f"{solution.energies[0] - exact_energy: .2e}"
        )

    minimum_kinetic_eigenvalue = np.linalg.eigvalsh(
        final_matrices.kinetic
    )[0]
    print("\nNumerical diagnostics")
    print(
        "Minimum kinetic eigenvalue:       "
        f"{minimum_kinetic_eigenvalue:.6e}"
    )
    print(
        "Metric normalization error:       "
        f"{final_solution.diagnostics.orthonormality_error:.3e}"
    )
    print(
        "Maximum relative residual:        "
        f"{np.max(final_solution.diagnostics.relative_residual_norms):.3e}"
    )
    print("Centered angular off-channel norms")
    for name, error in angular_selection_errors().items():
        print(f"  {name:11s}: {error:.3e}")


if __name__ == "__main__":
    main()
