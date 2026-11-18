"""Increase a smooth B-spline space at fixed PySCF quadrature."""

import argparse
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


REFERENCE_ELECTRONIC_ENERGY = -1.102634214494946
SPLINE_DEGREES = (3, 5, 7, 9, 11)
SPLINE_CONTINUITY = 1
LEBEDEV_ORDER = 65
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


def main(
    spline_degrees=SPLINE_DEGREES,
    spline_continuity=SPLINE_CONTINUITY,
    lebedev_order=LEBEDEV_ORDER,
):
    print("B-spline basis convergence for H2+")
    print(f"Spline continuity: C{spline_continuity}")
    print(f"Fixed unpruned PySCF grid: Lebedev order {lebedev_order}")
    print("Four fixed proper rotations for every basis")
    print()
    print(
        "degree  basis  rank   cond(S)  unrotated_error  energy_min  "
        "energy_max  envelope  time_s"
    )
    print("                              (energies in microhartree)")

    for degree in spline_degrees:
        started = perf_counter()
        results = [
            solve_atomic_h2plus(
                radial_basis="bspline",
                radial_spline_degree=degree,
                radial_spline_continuity=spline_continuity,
                grid_backend="pyscf",
                pyscf_prune=None,
                lebedev_order=lebedev_order,
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
            f"{degree:6d} {matrices.basis_size:6d} "
            f"{metric.retained_rank:5d} "
            f"{matrices.overlap_condition_number:9.3e} "
            f"{errors[0]:16.3f} {errors.min():11.3f} "
            f"{errors.max():11.3f} {np.ptp(errors):9.3f} "
            f"{perf_counter() - started:7.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--degrees", nargs="+", type=int, default=SPLINE_DEGREES
    )
    parser.add_argument(
        "--continuity", type=int, default=SPLINE_CONTINUITY
    )
    parser.add_argument(
        "--lebedev-order", type=int, default=LEBEDEV_ORDER
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(
        spline_degrees=arguments.degrees,
        spline_continuity=arguments.continuity,
        lebedev_order=arguments.lebedev_order,
    )
