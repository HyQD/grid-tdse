"""Diagnose rotation sensitivity of midpoint-basis Becke matrices."""

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation

from grid_tdse.spherical.one_radial_atomic import (
    solve_midpoint_h2plus_on_becke_grid,
)


def lowest_energy(overlap, kinetic, potential):
    return float(
        eigh(
            kinetic + potential,
            overlap,
            subset_by_index=(0, 0),
            check_finite=False,
        )[0][0]
    )


rotations = [
    None,
    Rotation.from_rotvec([0.2, 0.3, 0.4]).as_matrix(),
    Rotation.from_rotvec([0.7, -0.2, 0.3]).as_matrix(),
    Rotation.from_rotvec([-0.4, 0.6, 0.1]).as_matrix(),
]
results = [
    solve_midpoint_h2plus_on_becke_grid(
        integration_spacing=0.25,
        integration_radial_n_points=16,
        lebedev_order=59,
        rotation=rotation,
    )
    for rotation in rotations
]
reference = results[0]
reference_energy = reference.becke_energy

print("rotation    all       overlap    kinetic    potential (microhartree)")
for index, result in enumerate(results):
    energy = lowest_energy(result.overlap, result.kinetic, result.potential)
    overlap_energy = lowest_energy(
        result.overlap, reference.kinetic, reference.potential
    )
    kinetic_energy = lowest_energy(
        reference.overlap, result.kinetic, reference.potential
    )
    potential_energy = lowest_energy(
        reference.overlap, reference.kinetic, result.potential
    )
    shifts = 1.0e6 * np.array(
        [energy, overlap_energy, kinetic_energy, potential_energy]
    )
    shifts -= 1.0e6 * reference_energy
    print(f"{index:8d}" + "".join(f" {value:10.3f}" for value in shifts))
