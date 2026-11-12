"""Run the atom-centered H2+ prototype on a custom-radial PySCF grid."""

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


REFERENCE_ELECTRONIC_ENERGY = -1.102634214

result = solve_atomic_h2plus(
    grid_backend="pyscf",
    integration_spacing=0.5,
    integration_radial_n_points=12,
    lebedev_order=47,
)

print("\n** custom-radial PySCF Becke grid for H2+ **")
print(f"electronic energy       {result.electronic_energy: .12f} Eh")
print(f"total energy            {result.total_energy: .12f} Eh")
print(
    "reference error         "
    f"{1.0e6 * (result.electronic_energy - REFERENCE_ELECTRONIC_ENERGY): .3f} uEh"
)
print(f"basis size              {result.matrices.basis_size:8d}")
print(f"quadrature points       {result.matrices.quadrature_size:8d}")
print(f"element groups          {result.matrices.element_group_count:8d}")
print(f"assembly time           {result.matrices.assembly_seconds:8.3f} s")
