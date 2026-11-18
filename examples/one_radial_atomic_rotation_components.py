"""Split atomic-basis rotation sensitivity into matrix and center blocks."""

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


rotations = [
    None,
    Rotation.from_rotvec([0.2, 0.3, 0.4]).as_matrix(),
    Rotation.from_rotvec([0.7, -0.2, 0.3]).as_matrix(),
    Rotation.from_rotvec([-0.4, 0.6, 0.1]).as_matrix(),
]
results = [
    solve_atomic_h2plus(
        integration_radial_n_points=14,
        lebedev_order=59,
        lebedev_rotation=rotation,
    )
    for rotation in rotations
]
reference = results[0].matrices
transformation = results[0].solution.metric.transformation
center_size = reference.basis_size // 2


def project(matrix):
    return transformation.conj().T @ matrix @ transformation


def lowest_projected_energy(overlap, kinetic, potential):
    return float(
        eigh(
            project(kinetic + potential),
            project(overlap),
            subset_by_index=(0, 0),
            check_finite=False,
        )[0][0]
    )


reference_energy = lowest_projected_energy(
    reference.overlap, reference.kinetic, reference.potential
)

print(
    "rotation    all       overlap  potential  kinetic (microhartree)"
)
norm_rows = []
for index, result in enumerate(results):
    matrices = result.matrices
    shifts = 1.0e6 * np.array(
        [
            lowest_projected_energy(
                matrices.overlap, matrices.kinetic, matrices.potential
            ),
            lowest_projected_energy(
                matrices.overlap, reference.kinetic, reference.potential
            ),
            lowest_projected_energy(
                reference.overlap, reference.kinetic, matrices.potential
            ),
            lowest_projected_energy(
                reference.overlap, matrices.kinetic, reference.potential
            ),
        ]
    )
    shifts -= 1.0e6 * reference_energy
    print(f"{index:8d}" + "".join(f" {value:10.3f}" for value in shifts))
    kinetic_change = matrices.kinetic - reference.kinetic
    same_change = np.zeros_like(kinetic_change)
    same_change[:center_size, :center_size] = kinetic_change[
        :center_size, :center_size
    ]
    same_change[center_size:, center_size:] = kinetic_change[
        center_size:, center_size:
    ]
    cross_change = kinetic_change - same_change
    reference_same = np.zeros_like(reference.kinetic)
    reference_same[:center_size, :center_size] = reference.kinetic[
        :center_size, :center_size
    ]
    reference_same[center_size:, center_size:] = reference.kinetic[
        center_size:, center_size:
    ]
    reference_cross = reference.kinetic - reference_same
    norm_rows.append(
        (
            np.linalg.norm(same_change) / np.linalg.norm(reference_same),
            np.linalg.norm(cross_change) / np.linalg.norm(reference_cross),
        )
    )

print("\nrotation  relative same-center T change  relative cross T change")
for index, row in enumerate(norm_rows):
    print(f"{index:8d} {row[0]:30.3e} {row[1]:24.3e}")
