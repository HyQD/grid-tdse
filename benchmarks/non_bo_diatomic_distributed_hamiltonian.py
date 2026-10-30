"""Validate distributed non-BO diatomic Hamiltonian actions without MPI.

Use a reproducible complex state on a small H2+ representation with::

    python benchmarks/non_bo_diatomic_distributed_hamiltonian.py --ranks 2

Use a compatible archived ground state with::

    python benchmarks/non_bo_diatomic_distributed_hamiltonian.py \
        --archive path/to/ground_state.npz --ranks 4

The rank-local action follows the distributed element and bridge-assembly
path, but all ranks are emulated in one process.  This is an equivalence and
memory-layout reference, not an MPI scaling benchmark.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributedHamiltonian,
    TwoRadialDistributedLayout,
    TwoRadialDistributedReductions,
    load_two_radial_state,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), default="H2+")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument(
        "--partition-coordinate",
        choices=("r", "R"),
        default="R",
    )
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--codata-set", default="2022")
    return parser.parse_args()


def calculation_from_arguments(args, archived):
    if archived is None:
        system = args.system
        electronic_grid = FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4])
        internuclear_grid = FEMDVRGridSpec(
            [0.0, 1.0, 3.0, 6.0, 10.0],
            [4, 5, 3, 4],
        )
        inversion_parity = 1
        pair_exchange_parity = None if system == "HD+" else 1
        symmetry_adapted_basis = system != "HD+"
        l_max = args.l_max
        m_max = args.l_max
        lambda_max = 2 * args.l_max
        mu_max = lambda_max
        M_tot = 0
        codata_set = args.codata_set
        radial_coulomb_method = "poisson"
    else:
        physical = archived.physical_state
        symmetry = physical["symmetry"]
        system = physical["system"]
        electronic_grid = archived.grid1_specification
        internuclear_grid = archived.grid2_specification
        inversion_parity = symmetry["inversion_parity"]
        pair_exchange_parity = symmetry["pair_exchange_parity"]
        symmetry_adapted_basis = symmetry["symmetry_adapted_basis"]
        l_max = archived.basis1.l_max
        m_max = archived.basis1.m_max
        lambda_max = archived.multipoles.lambda_max
        mu_max = archived.multipoles.mu_max
        M_tot = archived.M_tot
        codata_set = physical["parameters"]["codata_set"]
        representation = archived.metadata["representation"]
        radial_coulomb_method = (
            "poisson"
            if representation["radial_coulomb_representation"]
            == "femdvr-poisson"
            else "analytic"
        )

    return NonBornOppenheimerDiatomicIonGroundState(
        system,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        inversion_parity=inversion_parity,
        pair_exchange_parity=pair_exchange_parity,
        symmetry_adapted_basis=symmetry_adapted_basis,
        l_max=l_max,
        m_max=m_max,
        lambda_max=lambda_max,
        mu_max=mu_max,
        M_tot=M_tot,
        radial_coulomb_method=radial_coulomb_method,
        codata_set=codata_set,
        initial_state="product",
        time_step=0.25,
        max_iterations=1,
        energy_tolerance=1.0e-10,
        state_tolerance=1.0e-8,
        residual_tolerance=1.0e-8,
        integrator="sil",
        max_krylov_dimension=20,
        krylov_rtol=1.0e-10,
        krylov_atol=0.0,
        max_krylov_refinements=8,
        krylov_reorthogonalize=True,
        progress_interval=None,
        use_preconditioner=False,
        radial_operator_mode="element-local",
        tile_size=2048,
    )


def timed(function, repeats):
    function()
    values = np.empty(repeats)
    result = None
    for repeat in range(repeats):
        start = perf_counter()
        result = function()
        values[repeat] = perf_counter() - start
    return result, float(values.min()), float(np.median(values))


args = parse_arguments()
if args.ranks < 1:
    raise ValueError("ranks must be positive")
if args.l_max < 0:
    raise ValueError("l-max must be non-negative")
if args.repeats < 1:
    raise ValueError("repeats must be positive")

archived = load_two_radial_state(args.archive) if args.archive else None
calculation = calculation_from_arguments(args, archived)
representation = calculation.representation
if archived is None:
    rng = np.random.default_rng(args.seed)
    state = rng.normal(size=representation.psi_shape) + 1j * rng.normal(
        size=representation.psi_shape
    )
else:
    state = archived.state_for(representation)

layout = TwoRadialDistributedLayout(
    representation,
    args.ranks,
    partitioned_coordinate=1 if args.partition_coordinate == "r" else 2,
)
distributed_hamiltonian = TwoRadialDistributedHamiltonian(layout)
reductions = TwoRadialDistributedReductions(layout)
local_state = layout.distribute(state)
serial_hamiltonian = representation.hamiltonian(
    radial_operator_mode="element-local"
)

serial_action, serial_min, serial_median = timed(
    lambda: serial_hamiltonian(state),
    args.repeats,
)
local_action, distributed_min, distributed_median = timed(
    lambda: distributed_hamiltonian.apply_all(local_state),
    args.repeats,
)
assembled_action = layout.assemble(local_action)
difference = assembled_action - serial_action
difference_norm = np.sqrt(
    representation.inner_product(difference, difference).real
)
serial_norm = np.sqrt(representation.inner_product(state, state).real)
distributed_norm = np.sqrt(
    reductions.inner_product_all(local_state, local_state).real
)
serial_energy = representation.inner_product(state, serial_action)
distributed_energy = reductions.inner_product_all(local_state, local_action)

print(f"System:                     {calculation.system_name}")
print(f"State source:               {args.archive or 'reproducible complex state'}")
print(f"Emulated ranks:             {args.ranks}")
print(f"Partitioned coordinate:    {args.partition_coordinate}")
print(f"Global state shape:         {representation.psi_shape}")
print(
    f"Owned {args.partition_coordinate} points/rank:     "
    + ", ".join(str(part.n_owned) for part in layout.partitions)
)
print(f"Serial minimum:             {serial_min:.6f} s")
print(f"Serial median:              {serial_median:.6f} s")
print(f"Distributed-path minimum:   {distributed_min:.6f} s")
print(f"Distributed-path median:    {distributed_median:.6f} s")
print(f"Maximum action difference:  {np.max(np.abs(difference)):.3e}")
print(f"Action-difference norm:     {difference_norm:.3e}")
print(f"Norm difference:            {abs(distributed_norm - serial_norm):.3e}")
print(f"Energy difference:          {abs(distributed_energy - serial_energy):.3e}")
print(f"Global state storage:       {state.nbytes / 1024.0**2:.3f} MiB")
print(
    "Local state storage/rank: "
    + ", ".join(
        f"{local.coefficients.nbytes / 1024.0**2:.3f}"
        for local in local_state
    )
    + " MiB"
)
