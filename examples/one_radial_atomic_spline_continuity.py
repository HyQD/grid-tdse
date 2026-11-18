"""Compare B-spline continuity at similar atom-centered basis sizes."""

from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


REFERENCE_ELECTRONIC_ENERGY = -1.102634214494946
DEGREE_CONTINUITY_PAIRS = ((7, 1), (8, 2), (9, 3), (10, 4))
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
    print("B-spline continuity sweep for H2+")
    print("Fixed unpruned PySCF grid: Lebedev order 65")
    print("Degree is increased with continuity to keep basis sizes similar")
    print()
    print(
        "degree  continuity  basis  rank   cond(S)  unrotated_error  "
        "energy_min  energy_max  envelope  time_s"
    )
    print("                                          (energies in microhartree)")

    for degree, continuity in DEGREE_CONTINUITY_PAIRS:
        started = perf_counter()
        results = [
            solve_atomic_h2plus(
                radial_basis="bspline",
                radial_spline_degree=degree,
                radial_spline_continuity=continuity,
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
            f"{degree:6d} {continuity:11d} {matrices.basis_size:6d} "
            f"{metric.retained_rank:5d} "
            f"{matrices.overlap_condition_number:9.3e} "
            f"{errors[0]:16.3f} {errors.min():11.3f} "
            f"{errors.max():11.3f} {np.ptp(errors):9.3f} "
            f"{perf_counter() - started:7.2f}"
        )


if __name__ == "__main__":
    main()
