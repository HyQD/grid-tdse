"""Increase l_max for a radially converged smooth B-spline pilot."""

from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


REFERENCE_ELECTRONIC_ENERGY = -1.102634214494946
L_MAX_VALUES = (2, 3, 4, 5)
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
    print("Angular basis convergence for smooth atom-centered H2+")
    print("Fixed degree-8 C4 B-spline radial basis")
    print("Fixed unpruned PySCF grid: Lebedev order 65")
    print()
    print(
        "l_max  basis  rank   cond(S)  unrotated_error  energy_min  "
        "energy_max  envelope  time_s"
    )
    print("                             (energies in microhartree)")

    for l_max in L_MAX_VALUES:
        started = perf_counter()
        results = [
            solve_atomic_h2plus(
                radial_basis="bspline",
                radial_spline_degree=8,
                radial_spline_continuity=4,
                l_max=l_max,
                grid_backend="pyscf",
                pyscf_prune=None,
                lebedev_order=65,
                lebedev_rotation=rotation_matrix(rotation_vector),
            )
            for rotation_vector in ROTATION_VECTORS
        ]
        energies = np.array(
            [result.electronic_energy for result in results]
        )
        errors = 1.0e6 * (energies - REFERENCE_ELECTRONIC_ENERGY)
        matrices = results[0].matrices
        metric = results[0].solution.metric
        print(
            f"{l_max:5d} {matrices.basis_size:6d} "
            f"{metric.retained_rank:5d} "
            f"{matrices.overlap_condition_number:9.3e} "
            f"{errors[0]:16.3f} {errors.min():11.3f} "
            f"{errors.max():11.3f} {np.ptp(errors):9.3f} "
            f"{perf_counter() - started:7.2f}"
        )


if __name__ == "__main__":
    main()
