"""Run the unweighted atom-centered H2+ reference implementation."""

from grid_tdse.spherical.one_radial_atomic import solve_atomic_h2plus


def main():
    result = solve_atomic_h2plus()
    matrices = result.matrices
    reference_energy = -1.102634214
    print("\nH2+ atomic superposition; Becke integration weights only")
    print(f"Electronic energy:       {result.electronic_energy:.12f} Eh")
    print(
        "Error to reference:      "
        f"{1.0e6 * abs(result.electronic_energy - reference_energy):.3f} uEh"
    )
    print(f"Total energy:            {result.total_energy:.12f} Eh")
    print(f"Basis size:              {matrices.basis_size}")
    print(f"Quadrature points:       {matrices.quadrature_size}")
    print(f"Active element groups:   {matrices.element_group_count}")
    print(f"Overlap condition:       {matrices.overlap_condition_number:.3e}")
    print(f"Retained overlap rank:   {result.solution.metric.retained_rank}")
    print(f"Assembly time:           {matrices.assembly_seconds:.3f} s")


if __name__ == "__main__":
    main()
