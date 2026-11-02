"""Saga deployment validation for non-BO dynamics initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    TwoRadialDistributed2DLayout,
    gather_two_radial_2d_state,
    load_two_radial_state,
    phase_align_two_radial_state,
    phase_aligned_two_radial_state_difference,
    prepare_distributed_non_bo_diatomic_dynamics_initial_state,
)


def parse_process_grid(value: str) -> tuple[int, int]:
    first, second = value.lower().split("x", maxsplit=1)
    return int(first), int(second)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("seed", "mpi"), required=True)
parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), required=True)
parser.add_argument(
    "--policy",
    choices=("require", "compute-if-missing", "recompute"),
    default="require",
)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--seed-archive", type=Path)
parser.add_argument("--process-grid", default="2x2")
parser.add_argument("--expect-source", choices=("archive", "computed"))
parser.add_argument("--result", type=Path, required=True)
arguments = parser.parse_args()

communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
size = communicator.Get_size()

pair_exchange_parity = None if arguments.system == "HD+" else +1

electronic_grid = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 4.0],
    n_points=[5, 5],
)
internuclear_grid = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 4.0],
    n_points=[5, 5],
)

ground_state_options = {
    "max_iterations": 500,
    "maximum_subspace_dimension": 30,
    "retained_ritz_vectors": 4,
    "residual_tolerance": 1.0e-8,
    "progress_interval": None,
    "sample_interval": None,
}

tdse_options = {
    "system": arguments.system,
    "electronic_grid": electronic_grid,
    "internuclear_grid": internuclear_grid,
    "pair_exchange_parity": pair_exchange_parity,
    "l_max": 1,
    "m_max": 1,
    "lambda_max": 2,
    "mu_max": 2,
    "M_tot": 0,
    "codata_set": "2022",
    "final_time": 0.01,
    "time_step": 0.01,
    "integrator": "sil",
    "integrator_options": {
        "max_krylov_dimension": 10,
        "rtol": 1.0e-10,
    },
    "observables": (),
    "progress_interval": None,
}

if arguments.mode == "seed":
    if size != 1:
        raise RuntimeError("seed mode requires exactly one MPI rank")

    calculation = NonBornOppenheimerDiatomicIonTDSE(
        **tdse_options,
        initial_state_policy=arguments.policy,
        ground_state_path=arguments.archive,
        ground_state_options=ground_state_options,
    )
    initialization = calculation.initialization

    class NonzeroVectorPotential:
        gauge = "velocity"

        def __call__(self, time):
            return 0.1

    velocity_guard = False
    try:
        NonBornOppenheimerDiatomicIonTDSE(
            **tdse_options,
            initial_state=initialization.archive,
            fields=NonzeroVectorPotential(),
            field_gauge="velocity",
        )
    except ValueError as error:
        velocity_guard = "A(initial_time)=0" in str(error)

    if not velocity_guard:
        raise RuntimeError("velocity-gauge A(initial_time)=0 guard failed")

    representation = calculation.representation
    norm = representation.inner_product(
        calculation.initial_state,
        calculation.initial_state,
    ).real
    norm = float(np.asarray(norm))

    values = {
        "mode": "seed",
        "system": arguments.system,
        "policy": arguments.policy,
        "source": initialization.source,
        "archive": str(initialization.archive_path),
        "state_data_id": initialization.archive.state_data_id,
        "representation_id": initialization.archive.representation_id,
        "setup_wall_time": initialization.setup_wall_time,
        "ground_state_wall_time": initialization.ground_state_wall_time,
        "M_tot": representation.M_tot,
        "electronic_coordinate_parity": (
            representation.basis1.coordinate_parity
        ),
        "internuclear_coordinate_parity": (
            representation.basis2.coordinate_parity
        ),
        "state_shape": list(calculation.initial_state.shape),
        "norm": norm,
        "velocity_initial_condition_guard": velocity_guard,
    }

    if initialization.source != "computed":
        raise RuntimeError(
            f"seed initialization used {initialization.source!r}, "
            "expected 'computed'"
        )
    if abs(norm - 1.0) > 1.0e-12:
        raise RuntimeError(f"seed state norm is {norm}")

    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    arguments.result.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name, value in values.items():
        print(f"{name}: {value}", flush=True)

    raise SystemExit(0)

if arguments.seed_archive is None:
    raise ValueError("--seed-archive is required in MPI mode")
if not arguments.seed_archive.is_file():
    raise FileNotFoundError(arguments.seed_archive)

# The seed is loaded here only to construct an identical target dynamics
# representation on every rank. The initializer under test independently
# loads or computes the requested archive on rank zero only.
seed = load_two_radial_state(arguments.seed_archive)
template = NonBornOppenheimerDiatomicIonTDSE(
    **tdse_options,
    initial_state=seed,
)

process_grid = parse_process_grid(arguments.process_grid)
if process_grid[0] * process_grid[1] != size:
    raise ValueError(
        f"process grid {process_grid} requires "
        f"{process_grid[0] * process_grid[1]} ranks, received {size}"
    )

layout = TwoRadialDistributed2DLayout(
    template.representation,
    size,
    process_grid=process_grid,
)

initialization = (
    prepare_distributed_non_bo_diatomic_dynamics_initial_state(
        layout,
        communicator,
        system=arguments.system,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        pair_exchange_parity=pair_exchange_parity,
        archive_path=arguments.archive,
        policy=arguments.policy,
        codata_set="2022",
        ground_state_options=ground_state_options,
    )
)

if (
    arguments.expect_source is not None
    and initialization.source != arguments.expect_source
):
    raise RuntimeError(
        f"initialization source was {initialization.source!r}; "
        f"expected {arguments.expect_source!r}"
    )

local_coefficients = int(initialization.state.coefficients.size)
local_sizes = communicator.gather(local_coefficients, root=0)

assembled = gather_two_radial_2d_state(
    initialization.state,
    communicator,
    root=0,
)

if rank == 0:
    archived = load_two_radial_state(arguments.archive)
    expected = archived.embed_into(template.representation)
    expected /= np.sqrt(
        template.representation.inner_product(expected, expected).real
    )

    difference = phase_aligned_two_radial_state_difference(
        expected,
        assembled,
        template.representation.inner_product,
    )
    aligned = phase_align_two_radial_state(
        expected,
        assembled,
        template.representation.inner_product,
    )
    max_error = float(np.max(np.abs(aligned - expected)))
    norm = float(
        template.representation.inner_product(assembled, assembled).real
    )

    values = {
        "mode": "mpi",
        "system": arguments.system,
        "policy": arguments.policy,
        "source": initialization.source,
        "archive": str(initialization.archive_path),
        "state_data_id": initialization.state_data_id,
        "representation_id": initialization.representation_id,
        "setup_wall_time": initialization.setup_wall_time,
        "ground_state_wall_time": initialization.ground_state_wall_time,
        "ranks": size,
        "process_grid": list(layout.process_grid),
        "M_tot": template.representation.M_tot,
        "electronic_coordinate_parity": (
            template.representation.basis1.coordinate_parity
        ),
        "internuclear_coordinate_parity": (
            template.representation.basis2.coordinate_parity
        ),
        "local_coefficients": local_sizes,
        "norm": norm,
        "serial_state_difference": difference,
        "maximum_coefficient_error": max_error,
    }

    if abs(norm - 1.0) > 1.0e-12:
        raise RuntimeError(f"distributed state norm is {norm}")
    if difference > 1.0e-11:
        raise RuntimeError(
            f"serial/distributed state difference is {difference:.3e}"
        )

    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    arguments.result.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name, value in values.items():
        print(f"{name}: {value}", flush=True)
