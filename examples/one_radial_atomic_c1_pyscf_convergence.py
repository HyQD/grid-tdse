"""Sweep unpruned PySCF angular grids for the fixed C1 H2+ pilot."""

from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


REFERENCE_ELECTRONIC_ENERGY = -1.102634214494946
LEBEDEV_ORDERS = (29, 41, 53, 59, 65)
PRUNING_SCHEMES = ((None, "unpruned"), ("nwchem", "nwchem"))
ROTATION_VECTORS = (
    None,
    (0.2, 0.3, 0.4),
    (0.7, -0.2, 0.3),
    (-0.4, 0.6, 0.1),
)


def rotation_matrix(rotation_vector):
    if rotation_vector is None:
        return None
    return Rotation.from_rotvec(rotation_vector).as_matrix()


def main():
    print("PySCF angular convergence for the C1 B-spline H2+ pilot")
    print("Fixed radial rule: spacing=0.25 bohr, 10 points per subinterval")
    print("Four fixed proper rotations at every Lebedev order")

    for pruning, label in PRUNING_SCHEMES:
        print()
        print(f"pruning={label}")
        print(
            "order   points  basis  unrotated_error  energy_min  energy_max  "
            "envelope  max_shift  time_s"
        )
        print(
            "                          "
            "(all energies and spreads in microhartree)"
        )

        for order in LEBEDEV_ORDERS:
            started = perf_counter()
            results = [
                solve_atomic_h2plus(
                    radial_basis="bspline-c1",
                    grid_backend="pyscf",
                    pyscf_prune=pruning,
                    lebedev_order=order,
                    lebedev_rotation=rotation_matrix(rotation_vector),
                )
                for rotation_vector in ROTATION_VECTORS
            ]
            energies = np.array(
                [result.electronic_energy for result in results]
            )
            errors = 1.0e6 * (energies - REFERENCE_ELECTRONIC_ENERGY)
            shifts = 1.0e6 * (energies - energies[0])
            elapsed = perf_counter() - started
            matrices = results[0].matrices
            print(
                f"{order:5d} {matrices.quadrature_size:8d} "
                f"{matrices.basis_size:6d} {errors[0]:16.3f} "
                f"{errors.min():11.3f} {errors.max():11.3f} "
                f"{np.ptp(errors):9.3f} {np.max(np.abs(shifts)):10.3f} "
                f"{elapsed:7.2f}"
            )


if __name__ == "__main__":
    main()
