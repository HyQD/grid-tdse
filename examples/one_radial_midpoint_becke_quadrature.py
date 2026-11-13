"""Hold the midpoint H2+ basis fixed and sweep its Becke quadrature."""

from scipy.spatial.transform import Rotation

from grid_tdse.spherical.one_radial_atomic import (
    solve_midpoint_h2plus_on_becke_grid,
)


print("Lebedev  points       E(Becke)       difference")
for lebedev_order in (29, 41, 53, 65, 83):
    result = solve_midpoint_h2plus_on_becke_grid(
        integration_spacing=0.25,
        integration_radial_n_points=16,
        lebedev_order=lebedev_order,
    )
    print(
        f"{lebedev_order:7d} {result.quadrature_size:7d} "
        f"{result.becke_energy:16.10f} "
        f"{1e6 * result.energy_difference:12.3f} uEh"
    )
print(f"structured reference: {result.structured_energy:.12f} Eh")

print("\nLebedev 59 rotation sensitivity")
for rotation_vector in ([0.2, 0.3, 0.4], [0.7, -0.2, 0.3], [-0.4, 0.6, 0.1]):
    result = solve_midpoint_h2plus_on_becke_grid(
        integration_spacing=0.25,
        integration_radial_n_points=16,
        lebedev_order=59,
        rotation=Rotation.from_rotvec(rotation_vector).as_matrix(),
    )
    print(
        f"{rotation_vector!s:>18} "
        f"{1e6 * result.energy_difference:12.3f} uEh"
    )
