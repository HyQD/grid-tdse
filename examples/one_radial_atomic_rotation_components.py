"""Reproducible H2+ benchmark for atom-centered rotation sensitivity.

The benchmark holds the basis and every radial setting fixed, then applies
four proper rotations to the same Lebedev rule. It reports the absolute
ground-state error, assembly cost, matrix-component energy shifts, and the
relative rotation sensitivity of same-center and cross-center kinetic blocks.
"""

import argparse

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


REFERENCE_ELECTRONIC_ENERGY = -1.102634214494946
ROTATION_VECTORS = (
    None,
    (0.2, 0.3, 0.4),
    (0.7, -0.2, 0.3),
    (-0.4, 0.6, 0.1),
)
SOLVER_OPTIONS = {
    "integration_radial_n_points": 10,
    "lebedev_order": 41,
}
RADIAL_SPLINE_DEGREE = 7


def rotation_matrix(rotation_vector):
    if rotation_vector is None:
        return None
    return Rotation.from_rotvec(rotation_vector).as_matrix()


def project(matrix, transformation):
    return transformation.conj().T @ matrix @ transformation


def lowest_projected_energy(overlap, kinetic, potential, transformation):
    return float(
        eigh(
            project(kinetic + potential, transformation),
            project(overlap, transformation),
            subset_by_index=(0, 0),
            check_finite=False,
        )[0][0]
    )


def center_blocks(matrix, center_size):
    same_center = np.zeros_like(matrix)
    same_center[:center_size, :center_size] = matrix[
        :center_size, :center_size
    ]
    same_center[center_size:, center_size:] = matrix[
        center_size:, center_size:
    ]
    return same_center, matrix - same_center


def run_benchmark(radial_basis="fem-dvr"):
    results = [
        solve_atomic_h2plus(
            **SOLVER_OPTIONS,
            lebedev_rotation=rotation_matrix(rotation_vector),
            radial_basis=radial_basis,
            radial_spline_degree=RADIAL_SPLINE_DEGREE,
        )
        for rotation_vector in ROTATION_VECTORS
    ]
    reference = results[0].matrices
    transformation = results[0].solution.metric.transformation
    center_size = reference.basis_size // 2
    reference_energy = lowest_projected_energy(
        reference.overlap,
        reference.kinetic,
        reference.potential,
        transformation,
    )
    reference_same, reference_cross = center_blocks(
        reference.kinetic, center_size
    )

    print("H2+ atom-centered Becke rotation benchmark")
    print(f"radial basis={radial_basis}")
    print("R=2 bohr, r_max=8 bohr, l_max=2")
    if radial_basis == "fem-dvr":
        print("radial_n_points=8")
    else:
        print(f"radial_spline_degree={RADIAL_SPLINE_DEGREE}")
    print(
        "integration_spacing=0.25 bohr, "
        "integration_radial_n_points="
        f"{SOLVER_OPTIONS['integration_radial_n_points']}, "
        f"Lebedev order={SOLVER_OPTIONS['lebedev_order']}"
    )
    print(
        f"basis={reference.basis_size}, grid={reference.quadrature_size}, "
        f"overlap condition={reference.overlap_condition_number:.3e}"
    )
    print(
        "baseline electronic energy="
        f"{reference_energy:.15f} Eh, error="
        f"{1.0e6 * (reference_energy - REFERENCE_ELECTRONIC_ENERGY):+.3f} microEh"
    )
    print()
    print(
        "rotation    all       overlap  potential  kinetic   assembly_s "
        "(energy shifts in microEh)"
    )

    norm_rows = []
    all_shifts = []
    for index, result in enumerate(results):
        matrices = result.matrices
        shifts = 1.0e6 * np.array(
            [
                lowest_projected_energy(
                    matrices.overlap,
                    matrices.kinetic,
                    matrices.potential,
                    transformation,
                ),
                lowest_projected_energy(
                    matrices.overlap,
                    reference.kinetic,
                    reference.potential,
                    transformation,
                ),
                lowest_projected_energy(
                    reference.overlap,
                    reference.kinetic,
                    matrices.potential,
                    transformation,
                ),
                lowest_projected_energy(
                    reference.overlap,
                    matrices.kinetic,
                    reference.potential,
                    transformation,
                ),
            ]
        )
        shifts -= 1.0e6 * reference_energy
        all_shifts.append(shifts[0])
        print(
            f"{index:8d}"
            + "".join(f" {value:10.3f}" for value in shifts)
            + f" {matrices.assembly_seconds:12.3f}"
        )

        same_change, cross_change = center_blocks(
            matrices.kinetic - reference.kinetic, center_size
        )
        norm_rows.append(
            (
                np.linalg.norm(same_change) / np.linalg.norm(reference_same),
                np.linalg.norm(cross_change) / np.linalg.norm(reference_cross),
            )
        )

    print()
    print("rotation  relative same-center T change  relative cross T change")
    for index, row in enumerate(norm_rows):
        print(f"{index:8d} {row[0]:30.3e} {row[1]:24.3e}")

    nonzero_rows = np.asarray(norm_rows[1:])
    print()
    print(
        "maximum absolute rotated energy shift="
        f"{np.max(np.abs(all_shifts[1:])):.3f} microEh"
    )
    print(
        "mean cross/same kinetic sensitivity ratio="
        f"{np.mean(nonzero_rows[:, 1] / nonzero_rows[:, 0]):.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radial-basis",
        choices=("fem-dvr", "bspline-c1", "both"),
        default="fem-dvr",
    )
    return parser.parse_args()


if __name__ == "__main__":
    selected_basis = parse_args().radial_basis
    basis_names = (
        ("fem-dvr", "bspline-c1")
        if selected_basis == "both"
        else (selected_basis,)
    )
    for basis_index, basis_name in enumerate(basis_names):
        if basis_index:
            print("\n" + "=" * 79 + "\n")
        run_benchmark(basis_name)
