"""Benchmark the rank-local non-BO diatomic Coulomb kernel.

The benchmark times only the compact angular-coupling kernel on each rank's
owned radial tile.  Hamiltonian construction, Numba compilation, MPI state
distribution, result gathering, and the serial correctness comparison are
outside the timed interval.  For example::

    mpiexec -n 4 python -u benchmarks/non_bo_diatomic_coulomb_execution.py \
        --process-grid 2 2 --l-max 2 --state-dtype complex \
        --warmup-actions 2 --actions 20

Use the dedicated Saga scripts in ``examples/Saga/coulomb_kernel`` for
controlled tile-size and fixed-core MPI/Numba-thread studies.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import resource
from time import perf_counter, process_time

from mpi4py import MPI
from numba import get_num_threads, threading_layer
import numpy as np

from grid_tdse.benchmarking import (
    THREAD_ENVIRONMENT_VARIABLES,
    default_result_path,
    runtime_environment,
    write_json_exclusive,
)
from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributed2DHamiltonian,
    TwoRadialDistributed2DLayout,
    gather_two_radial_2d_state,
    scatter_two_radial_2d_state,
)


SCHEMA = "grid-tdse-non-bo-coulomb-execution-v1"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=("H2+", "HD+", "D2+"),
        default="H2+",
    )
    parser.add_argument("--l-max", type=int, default=8)
    parser.add_argument(
        "--state-dtype",
        choices=("real", "complex"),
        default="complex",
    )
    parser.add_argument(
        "--process-grid",
        type=int,
        nargs=2,
        metavar=("P_r", "P_R"),
        help="explicit MPI process grid; default: cost-selected",
    )
    parser.add_argument(
        "--electronic-nodes",
        type=float,
        nargs="+",
        default=(0.0, 4.0, 8.0, 12.0, 16.0),
    )
    parser.add_argument("--electronic-points", type=int, default=11)
    parser.add_argument(
        "--internuclear-nodes",
        "--R-nodes",
        dest="internuclear_nodes",
        type=float,
        nargs="+",
        default=(0.0, 2.0, 4.0, 6.0, 8.0),
    )
    parser.add_argument(
        "--internuclear-points",
        "--R-points",
        dest="internuclear_points",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--radial-coulomb-method",
        choices=("poisson", "analytic"),
        default="poisson",
    )
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=26111)
    parser.add_argument("--warmup-actions", type=int, default=2)
    parser.add_argument("--actions", type=int, default=20)
    parser.add_argument(
        "--no-serial-check",
        action="store_true",
        help="skip the untimed gathered comparison with the serial action",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--result-json", type=Path)
    output.add_argument("--result-directory", type=Path)
    parser.add_argument("--overwrite-result", action="store_true")
    return parser.parse_args()


def _positive_integer(value, name):
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _grid(nodes, n_points, name):
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError(f"{name} nodes must contain at least two values")
    if not np.all(np.isfinite(nodes)) or not np.all(np.diff(nodes) > 0.0):
        raise ValueError(f"{name} nodes must be finite and strictly increasing")
    n_points = _positive_integer(n_points, f"{name} points")
    return FEMDVRGridSpec(nodes, [n_points] * (nodes.size - 1))


def _configuration_kind(ranks, threads):
    if ranks == 1 and threads == 1:
        return "serial"
    if ranks == 1:
        return "shared-memory"
    if threads == 1:
        return "pure-mpi"
    return "hybrid"


def _allocated_threads():
    return _positive_integer(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "allocated threads per rank",
    )


def _configured_numba_threads():
    value = os.environ.get("NUMBA_NUM_THREADS")
    if value is None:
        return None
    return _positive_integer(int(value), "NUMBA_NUM_THREADS")


def _validate_thread_environment(allocated_threads, numba_threads):
    if numba_threads is not None and numba_threads > allocated_threads:
        raise RuntimeError(
            "NUMBA_NUM_THREADS exceeds the CPUs allocated to each MPI rank"
        )
    if os.environ.get("GRID_TDSE_VALIDATE_THREADS") != "1":
        return
    expected = {
        "NUMBA_NUM_THREADS": str(numba_threads),
        "OMP_NUM_THREADS": str(numba_threads),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(
                f"{name}={os.environ.get(name)!r}; expected {value!r}"
            )


def _oversubscribed_pools(environment, allocated_threads):
    pools = environment["thread_pools"]
    if not isinstance(pools, list):
        return []
    return [
        {
            "internal_api": pool.get("internal_api"),
            "num_threads": pool.get("num_threads"),
        }
        for pool in pools
        if isinstance(pool.get("num_threads"), int)
        and pool["num_threads"] > allocated_threads
    ]


def _calculation(arguments):
    identical_pair = arguments.system != "HD+"
    return NonBornOppenheimerDiatomicIonGroundState(
        arguments.system,
        electronic_grid=_grid(
            arguments.electronic_nodes,
            arguments.electronic_points,
            "electronic",
        ),
        internuclear_grid=_grid(
            arguments.internuclear_nodes,
            arguments.internuclear_points,
            "internuclear",
        ),
        inversion_parity=1,
        pair_exchange_parity=1 if identical_pair else None,
        symmetry_adapted_basis=identical_pair,
        l_max=arguments.l_max,
        m_max=arguments.l_max,
        lambda_max=2 * arguments.l_max,
        mu_max=2 * arguments.l_max,
        M_tot=0,
        radial_coulomb_method=arguments.radial_coulomb_method,
        radial_operator_mode="element-local",
        codata_set="2022",
        initial_state="product",
        max_iterations=1,
        sample_interval=None,
        use_preconditioner=False,
        tile_size=arguments.tile_size,
    )


def _random_state(generator, shape, dtype):
    values = generator.normal(size=shape)
    if dtype == np.dtype(np.complex128):
        values = values + 1j * generator.normal(size=shape)
    return np.ascontiguousarray(values, dtype=dtype)


def _peak_resident_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _output_lambda_groups(couplings):
    groups = 0
    for start, stop in zip(couplings.offsets[:-1], couplings.offsets[1:]):
        rows = couplings.radial_index[start:stop]
        if rows.size:
            groups += 1 + int(np.count_nonzero(rows[1:] != rows[:-1]))
    return groups


def main():
    arguments = parse_arguments()
    arguments.l_max = int(arguments.l_max)
    if arguments.l_max < 0:
        raise ValueError("l-max must be nonnegative")
    tile_size = _positive_integer(arguments.tile_size, "tile-size")
    actions = _positive_integer(arguments.actions, "actions")
    if arguments.warmup_actions < 0:
        raise ValueError("warmup-actions must be nonnegative")
    warmup_actions = int(arguments.warmup_actions)
    dtype = np.dtype(
        np.float64 if arguments.state_dtype == "real" else np.complex128
    )

    communicator = MPI.COMM_WORLD
    rank = communicator.rank
    ranks = communicator.size
    allocated_threads = _allocated_threads()
    numba_threads = _configured_numba_threads()
    _validate_thread_environment(allocated_threads, numba_threads)
    expected_total = os.environ.get("GRID_TDSE_TOTAL_CORES")
    if expected_total is not None and int(expected_total) != (
        ranks * allocated_threads
    ):
        raise RuntimeError(
            "GRID_TDSE_TOTAL_CORES does not equal MPI ranks times "
            "SLURM_CPUS_PER_TASK"
        )

    environment = runtime_environment(include_build_configuration=rank == 0)
    local_oversubscription = (
        _oversubscribed_pools(environment, allocated_threads)
        if os.environ.get("GRID_TDSE_VALIDATE_THREADS") == "1"
        else []
    )
    oversubscription = communicator.allgather(local_oversubscription)
    if any(oversubscription):
        raise RuntimeError(
            "a detected thread pool exceeds the CPUs allocated per MPI rank: "
            f"{oversubscription}"
        )

    setup_started = perf_counter()
    calculation = _calculation(arguments)
    representation = calculation.representation
    process_grid = (
        None
        if arguments.process_grid is None
        else tuple(arguments.process_grid)
    )
    layout = TwoRadialDistributed2DLayout(
        representation,
        ranks,
        process_grid=process_grid,
    )
    global_state = None
    if rank == 0:
        global_state = _random_state(
            np.random.default_rng(arguments.seed),
            representation.psi_shape,
            dtype,
        )
    local_state = scatter_two_radial_2d_state(
        layout,
        communicator,
        global_state,
    )
    hamiltonian = TwoRadialDistributed2DHamiltonian(
        layout,
        communicator,
        tile_size=tile_size,
        profile=False,
    )
    local_radial = hamiltonian._local_radial_coulomb[rank]
    local_kernel = hamiltonian.serial_hamiltonian._apply_coulomb_values
    output = np.zeros_like(local_state.coefficients)
    setup_seconds = perf_counter() - setup_started

    for _ in range(warmup_actions):
        local_kernel(
            local_state.coefficients,
            local_radial,
            out=output,
        )
    output.fill(0)
    actual_numba_threads = get_num_threads()
    actual_threading_layer = threading_layer()
    if (
        numba_threads is not None
        and actual_numba_threads != numba_threads
    ):
        raise RuntimeError(
            f"Numba uses {actual_numba_threads} threads; expected "
            f"{numba_threads}"
        )

    communicator.Barrier()
    wall_started = perf_counter()
    cpu_started = process_time()
    for _ in range(actions):
        local_kernel(
            local_state.coefficients,
            local_radial,
            out=output,
        )
    local_wall_seconds = perf_counter() - wall_started
    local_cpu_seconds = process_time() - cpu_started
    communicator.Barrier()

    wall_seconds = communicator.gather(local_wall_seconds, root=0)
    cpu_seconds = communicator.gather(local_cpu_seconds, root=0)
    setup_times = communicator.gather(setup_seconds, root=0)
    peak_memory = communicator.gather(_peak_resident_mib(), root=0)
    local_coefficients = communicator.gather(
        local_state.coefficients.size,
        root=0,
    )
    environments = communicator.gather(environment, root=0)

    gathered_action = None
    if not arguments.no_serial_check:
        checked_action = hamiltonian.apply_coulomb(local_state)
        gathered_action = gather_two_radial_2d_state(
            checked_action,
            communicator,
        )

    if rank != 0:
        return

    maximum_wall = max(wall_seconds)
    total_cpu = sum(cpu_seconds)
    total_cores = ranks * allocated_threads
    allocated_core_seconds = maximum_wall * total_cores
    cpu_utilization = 100.0 * total_cpu / allocated_core_seconds
    correctness = None
    if gathered_action is not None:
        serial = representation.hamiltonian(
            tile_size=tile_size,
            radial_operator_mode="element-local",
        )
        reference = serial.apply_coulomb(global_state)
        difference = gathered_action - reference
        maximum_error = float(np.max(np.abs(difference)))
        reference_scale = max(float(np.max(np.abs(reference))), 1.0)
        correctness = {
            "maximum_absolute_error": maximum_error,
            "maximum_relative_error": maximum_error / reference_scale,
        }

    couplings = representation.couplings
    output_lambda_groups = _output_lambda_groups(couplings)
    result = {
        "schema": SCHEMA,
        "problem": {
            "system": arguments.system,
            "l_max": arguments.l_max,
            "m_max": arguments.l_max,
            "lambda_max": 2 * arguments.l_max,
            "mu_max": 2 * arguments.l_max,
            "M_tot": 0,
            "state_dtype": dtype.name,
            "electronic_nodes": list(arguments.electronic_nodes),
            "electronic_points_per_element": arguments.electronic_points,
            "internuclear_nodes": list(arguments.internuclear_nodes),
            "internuclear_points_per_element": arguments.internuclear_points,
            "radial_coulomb_method": arguments.radial_coulomb_method,
            "global_shape": list(representation.psi_shape),
            "global_coefficients": int(np.prod(representation.psi_shape)),
            "angular_pairs": representation.angular_pairs.n_pairs,
            "coulomb_couplings": couplings.n_pairs,
            "output_lambda_groups": output_lambda_groups,
            "couplings_per_output_lambda_group": (
                couplings.n_pairs / output_lambda_groups
                if output_lambda_groups
                else 0.0
            ),
            "seed": arguments.seed,
            "tile_size": tile_size,
        },
        "execution": {
            "kind": _configuration_kind(ranks, allocated_threads),
            "mpi_ranks": ranks,
            "threads_per_rank": allocated_threads,
            "numba_threads_per_rank": actual_numba_threads,
            "numba_threading_layer": actual_threading_layer,
            "total_allocated_cores": total_cores,
            "process_grid": list(layout.process_grid),
            "warmup_actions": warmup_actions,
            "timed_actions": actions,
            "local_coefficients_minimum": min(local_coefficients),
            "local_coefficients_maximum": max(local_coefficients),
        },
        "timing": {
            "setup_seconds_maximum": max(setup_times),
            "wall_seconds_by_rank": wall_seconds,
            "wall_seconds_maximum": maximum_wall,
            "wall_seconds_per_action": maximum_wall / actions,
            "process_cpu_seconds_by_rank": cpu_seconds,
            "process_cpu_seconds_sum": total_cpu,
            "allocated_core_seconds": allocated_core_seconds,
            "allocated_core_utilization_percent": cpu_utilization,
            "peak_resident_mib_by_rank": peak_memory,
            "peak_resident_mib_maximum": max(peak_memory),
        },
        "correctness": correctness,
        "runtime_by_rank": environments,
        "mpi_library": MPI.Get_library_version().rstrip("\x00"),
        "thread_environment_variables": list(
            THREAD_ENVIRONMENT_VARIABLES
        ),
    }

    print(f"Benchmark component:        Coulomb kernel")
    print(f"State dtype:               {dtype.name}")
    print(f"MPI ranks:                  {ranks}")
    print(f"Threads per rank:           {allocated_threads}")
    print(f"Numba threads per rank:     {actual_numba_threads}")
    print(f"Numba threading layer:      {actual_threading_layer}")
    print(f"Total allocated cores:      {total_cores}")
    print(f"Process grid:               {layout.process_grid}")
    print(
        "Node placement:             "
        + ", ".join(item["hostname"] for item in environments)
    )
    print(f"System:                     {arguments.system}")
    print(f"l_max:                      {arguments.l_max}")
    print(f"Angular pairs:              {representation.angular_pairs.n_pairs}")
    print(f"Coulomb couplings:          {couplings.n_pairs}")
    print(f"Output-lambda groups:       {output_lambda_groups}")
    print(f"Global coefficients:        {result['problem']['global_coefficients']}")
    print(
        "Local coefficients:         "
        f"{min(local_coefficients)}--{max(local_coefficients)}"
    )
    print(f"Tile size:                  {tile_size}")
    print(f"Warmup actions:             {warmup_actions}")
    print(f"Timed actions:              {actions}")
    print(f"Coulomb wall time:          {maximum_wall:.6f} s")
    print(f"Coulomb time per action:    {maximum_wall / actions:.9f} s")
    print(f"Process CPU time (sum):     {total_cpu:.6f} s")
    print(f"Allocated-core utilization: {cpu_utilization:.2f} %")
    print(f"Peak memory/rank:           {max(peak_memory):.3f} MiB")
    if correctness is not None:
        print(
            "Coulomb max error:          "
            f"{correctness['maximum_absolute_error']:.3e}"
        )
    print("Thread pools:")
    pools = environments[0]["thread_pools"]
    if isinstance(pools, list):
        for pool in pools:
            print(
                f"  {pool.get('internal_api', 'unknown')}: "
                f"{pool.get('num_threads', 'unknown')} threads "
                f"({pool.get('version', 'unknown')})"
            )
    else:
        print("  unavailable")

    result_path = arguments.result_json
    if result_path is None and arguments.result_directory is not None:
        result_path = default_result_path(
            arguments.result_directory,
            stem="non-bo-coulomb-execution",
        )
    if result_path is not None:
        destination = write_json_exclusive(
            result_path,
            result,
            overwrite=arguments.overwrite_result,
        )
        print(f"Result JSON:                {destination}")


if __name__ == "__main__":
    main()
