"""Small coupled-angular Davidson ground states for H2+, HD+, and D2+.

This example demonstrates the coupled ground-state path on moderate fixed
FEMDVR grids. The calculation is not a radial or angular convergence study,
and the reported finite-basis energies are not claimed to be converged.
"""

from time import perf_counter

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
)


SYSTEMS = ("H2+", "HD+", "D2+")
ELECTRONIC_GRID = FEMDVRGridSpec(
    nodes=[0.0, 4.0, 8.0, 12.0, 16.0],
    n_points=[11] * 4,
)
INTERNUCLEAR_GRID = FEMDVRGridSpec(
    nodes=[0.0, 4.0, 8.0],
    n_points=[15] * 2,
)

L1_MAX = 2
L2_MAX = 2
L_MAX = 0
M_TOT = 0
LAMBDA_MAX = 4
RESIDUAL_TOLERANCE = 1.0e-8


print("Coupled-angular Davidson ground-state validation")
print("The following finite-basis energies are not convergence claims.\n")

for system in SYSTEMS:
    pair_exchange_parity = None if system == "HD+" else +1
    calculation = NonBornOppenheimerDiatomicIonGroundState(
        system,
        wf_representation="coupled",
        electronic_grid=ELECTRONIC_GRID,
        internuclear_grid=INTERNUCLEAR_GRID,
        inversion_parity=+1,
        pair_exchange_parity=pair_exchange_parity,
        l1_max=L1_MAX,
        l2_max=L2_MAX,
        L_max=L_MAX,
        M_tot=M_TOT,
        lambda_max=LAMBDA_MAX,
        radial_coulomb_method="poisson",
        radial_operator_mode="element-local",
        max_iterations=200,
        residual_tolerance=RESIDUAL_TOLERANCE,
        maximum_subspace_dimension=30,
        retained_ritz_vectors=4,
        progress_interval=None,
    )

    started = perf_counter()
    result = calculation.solve()
    wall_time = perf_counter() - started

    print(f"System:              {system}")
    print(f"Energy:              {result.energy:.12f} Eh")
    print(f"Residual norm:       {result.residual_norm:.3e}")
    print(f"Wavefunction norm:   {result.norm:.12f}")
    print(f"Iterations:          {result.iterations}")
    print(f"Termination reason:  {result.termination_reason}")
    print(f"Solver converged:    {result.converged}")
    print(f"<L^2>:               {result.observables.total_l_squared:.12f}")
    print(f"<L_z>:               {result.observables.total_l_z:.12f}")
    print(f"Solve wall time:     {wall_time:.6f} s\n")

    if not result.converged:
        raise RuntimeError(f"Davidson did not converge for {system}")
