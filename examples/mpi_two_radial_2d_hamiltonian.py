"""Validate Cartesian two-radial Hamiltonian actions with MPI.

For a four-rank H2+ comparison against the serial element-local action, run::

    mpiexec -n 4 python examples/mpi_two_radial_2d_hamiltonian.py \
        --process-grid 2 2 --l-max 1

The example scatters a deterministic complex state, applies the two radial
terms and Coulomb interaction on the Cartesian layout, and compares every
component with the serial implementation on rank zero.
"""

import argparse

import numpy as np
from mpi4py import MPI

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributed2DHamiltonian,
    TwoRadialDistributed2DLayout,
    gather_two_radial_2d_state,
    scatter_two_radial_2d_state,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--system",
    choices=("H2+", "HD+", "D2+"),
    default="H2+",
)
parser.add_argument(
    "--l-max",
    type=int,
    default=1,
    help="one-coordinate angular cutoff (default: 1)",
)
parser.add_argument(
    "--process-grid",
    type=int,
    nargs=2,
    metavar=("P_r", "P_R"),
    help="explicit P_r x P_R process grid; default: cost-selected",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1729,
    help="deterministic complex-state seed (default: 1729)",
)
arguments = parser.parse_args()

if arguments.l_max < 0:
    parser.error("--l-max must be nonnegative")

communicator = MPI.COMM_WORLD
identical_pair = arguments.system != "HD+"
l_max = arguments.l_max
calculation = NonBornOppenheimerDiatomicIonGroundState(
    arguments.system,
    electronic_grid=FEMDVRGridSpec(
        nodes=[0.0, 2.0, 5.0],
        n_points=[5, 5],
    ),
    internuclear_grid=FEMDVRGridSpec(
        nodes=[0.0, 1.5, 4.0],
        n_points=[5, 5],
    ),
    inversion_parity=+1,
    pair_exchange_parity=+1 if identical_pair else None,
    symmetry_adapted_basis=identical_pair,
    l_max=l_max,
    m_max=l_max,
    lambda_max=2 * l_max,
    mu_max=2 * l_max,
    M_tot=0,
    radial_coulomb_method="analytic",
    radial_operator_mode="element-local",
    codata_set="2018",
    initial_state="product",
)
representation = calculation.representation
process_grid = (
    None if arguments.process_grid is None else tuple(arguments.process_grid)
)
layout = TwoRadialDistributed2DLayout(
    representation,
    communicator.size,
    process_grid=process_grid,
)

state = None
if communicator.rank == 0:
    generator = np.random.default_rng(arguments.seed)
    state = generator.normal(size=representation.psi_shape)
    state = state + 1j * generator.normal(size=representation.psi_shape)

local = scatter_two_radial_2d_state(layout, communicator, state)
hamiltonian = TwoRadialDistributed2DHamiltonian(
    layout,
    communicator,
    profile=True,
)
components = hamiltonian.components(local)
action = components.total

coordinate1 = gather_two_radial_2d_state(
    components.coordinate1,
    communicator,
)
coordinate2 = gather_two_radial_2d_state(
    components.coordinate2,
    communicator,
)
coulomb = gather_two_radial_2d_state(components.coulomb, communicator)
distributed_action = gather_two_radial_2d_state(action, communicator)
local_coefficient_counts = communicator.gather(
    local.coefficients.size,
    root=0,
)

if communicator.rank == 0:
    serial = representation.hamiltonian(radial_operator_mode="element-local")
    expected_coordinate1 = serial.apply_one_body_coordinate1(state)
    expected_coordinate2 = serial.apply_one_body_coordinate2(state)
    expected_coulomb = serial.apply_coulomb(state)
    expected_action = expected_coordinate1 + expected_coordinate2 + expected_coulomb
    counts = np.asarray(local_coefficient_counts)

    print(f"MPI ranks:                   {communicator.size}")
    print(f"System:                      {arguments.system}")
    print(f"l_max:                       {l_max}")
    print(f"Process grid:                {layout.process_grid}")
    print(f"Global coefficients:         {np.prod(layout.global_shape)}")
    print(f"Local coefficients:          {counts.min()}--{counts.max()}")
    print(
        "Coordinate-1 max error:     "
        f"{np.max(np.abs(coordinate1 - expected_coordinate1)):.3e}"
    )
    print(
        "Coordinate-2 max error:     "
        f"{np.max(np.abs(coordinate2 - expected_coordinate2)):.3e}"
    )
    print(
        "Coulomb max error:          "
        f"{np.max(np.abs(coulomb - expected_coulomb)):.3e}"
    )
    print(
        "Hamiltonian max error:      "
        f"{np.max(np.abs(distributed_action - expected_action)):.3e}"
    )
