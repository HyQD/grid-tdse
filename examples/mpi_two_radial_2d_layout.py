"""Validate the Cartesian two-radial layout with MPI collectives.

For a four-rank smoke test that also redistributes the state, run::

    mpiexec -n 4 python examples/mpi_two_radial_2d_layout.py \
        --process-grid 2 2 --redistribute-process-grid 1 4

The calculation constructs a non-BO H2+ representation and product state, but
does not apply the Hamiltonian.  Distributed Hamiltonian support belongs to
the next implementation phase.
"""

import argparse

import numpy as np
from mpi4py import MPI

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributed2DLayout,
    gather_two_radial_2d_state,
    phase_aligned_two_radial_state_difference,
    redistribute_two_radial_2d_state,
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
    default=2,
    help="one-coordinate angular cutoff (default: 2)",
)
parser.add_argument(
    "--process-grid",
    type=int,
    nargs=2,
    metavar=("P_r", "P_R"),
    help="explicit P_r x P_R process grid; default: cost-selected",
)
parser.add_argument(
    "--redistribute-process-grid",
    type=int,
    nargs=2,
    metavar=("P_r", "P_R"),
    help="optional same-size target process grid",
)
arguments = parser.parse_args()

if arguments.l_max < 0:
    parser.error("--l-max must be nonnegative")

communicator = MPI.COMM_WORLD
l_max = arguments.l_max
identical_pair = arguments.system != "HD+"
calculation = NonBornOppenheimerDiatomicIonGroundState(
    arguments.system,
    electronic_grid=FEMDVRGridSpec(
        nodes=[0.0, 4.0, 8.0, 12.0, 16.0],
        n_points=[11, 11, 11, 11],
    ),
    internuclear_grid=FEMDVRGridSpec(
        nodes=[0.0, 2.0, 4.0, 6.0, 8.0],
        n_points=[21, 21, 21, 21],
    ),
    inversion_parity=+1,
    pair_exchange_parity=+1 if identical_pair else None,
    symmetry_adapted_basis=identical_pair,
    l_max=l_max,
    m_max=l_max,
    lambda_max=2 * l_max,
    mu_max=2 * l_max,
    M_tot=0,
    radial_coulomb_method="poisson",
    radial_operator_mode="element-local",
    codata_set="2022",
    initial_state="product",
)
process_grid = (
    None
    if arguments.process_grid is None
    else tuple(arguments.process_grid)
)
layout = TwoRadialDistributed2DLayout(
    calculation.representation,
    communicator.size,
    process_grid=process_grid,
)
local = scatter_two_radial_2d_state(
    layout,
    communicator,
    calculation.initial_state if communicator.rank == 0 else None,
)
active_layout = layout

if arguments.redistribute_process_grid is not None:
    target = TwoRadialDistributed2DLayout(
        calculation.representation,
        communicator.size,
        process_grid=tuple(arguments.redistribute_process_grid),
    )
    local = redistribute_two_radial_2d_state(
        local,
        target,
        communicator,
    )
    active_layout = target

gathered = gather_two_radial_2d_state(local, communicator)
local_coefficient_counts = communicator.gather(
    local.coefficients.size,
    root=0,
)

if communicator.rank == 0:
    local_coefficient_counts = np.asarray(local_coefficient_counts)
    difference = phase_aligned_two_radial_state_difference(
        calculation.initial_state,
        gathered,
        calculation.representation.inner_product,
    )
    print(f"MPI ranks:                {communicator.size}")
    print(f"System:                   {arguments.system}")
    print(f"Initial process grid:     {layout.process_grid}")
    print(f"Final process grid:       {active_layout.process_grid}")
    print(f"Global coefficients:      {np.prod(layout.global_shape)}")
    print(
        "Local coefficients:       "
        f"{local_coefficient_counts.min()}--{local_coefficient_counts.max()}"
    )
    print(f"Phase-aligned difference: {difference:.3e}")
    exact_match = np.array_equal(gathered, calculation.initial_state)
    print(f"Exact coefficient match:  {exact_match}")
