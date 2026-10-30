"""Benchmark serial, threaded, MPI, and hybrid 2D Hamiltonian execution.

The workload is a fixed number of field-free Hamiltonian actions on a
deterministic complex non-BO diatomic state.  It is a performance benchmark,
not an imaginary-time convergence calculation.  For example, a local
four-rank smoke run is::

    mpiexec -n 4 python -u benchmarks/non_bo_diatomic_2d_execution.py \
        --process-grid 2 2 --l-max 1 --warmup-actions 1 --actions 3

On Saga, use the phase-6 job and submission scripts in ``examples/Saga`` so
rank placement and thread counts are controlled consistently.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from mpi4py import MPI

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


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=("H2+", "HD+", "D2+"),
        default="H2+",
    )
    parser.add_argument("--l-max", type=int, default=2)
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
    parser.add_argument(
        "--electronic-points",
        type=int,
        default=11,
        help="FEM-DVR points in each electronic element (default: 11)",
    )
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
        help="FEM-DVR points in each internuclear element (default: 21)",
    )
    parser.add_argument(
        "--radial-coulomb-method",
        choices=("poisson", "analytic"),
        default="poisson",
    )
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=26031)
    parser.add_argument("--warmup-actions", type=int, default=2)
    parser.add_argument("--actions", type=int, default=20)
    parser.add_argument(
        "--no-serial-check",
        action="store_true",
        help="skip the untimed gathered comparison with the serial action",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--result-json", type=Path)
    output.add_argument(
        "--result-directory",
        type=Path,
        help="write a Slurm-named JSON result below this directory",
    )
    parser.add_argument("--overwrite-result", action="store_true")
    return parser.parse_args()


def _positive_integer(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive")
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


def _requested_threads():
    value = os.environ.get(
        "SLURM_CPUS_PER_TASK",
        os.environ.get("OMP_NUM_THREADS", "1"),
    )
    return _positive_integer(int(value), "threads per rank")


def _validate_thread_environment(threads):
    if os.environ.get("GRID_TDSE_VALIDATE_THREADS") != "1":
        return
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        value = os.environ.get(name)
        if value != str(threads):
            raise RuntimeError(
                f"{name}={value!r}; expected {threads} from cpus-per-task"
            )
    if os.environ.get("NUMBA_NUM_THREADS") is None:
        raise RuntimeError("NUMBA_NUM_THREADS must be set independently")


def _oversubscribed_pools(environment, threads):
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
        and pool["num_threads"] > threads
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


def _profile_summary(profiles):
    fields = asdict(profiles[0])
    return {
        name: {
            "minimum": min(getattr(profile, name) for profile in profiles),
            "maximum": max(getattr(profile, name) for profile in profiles),
        }
        for name in fields
    }


def _maximum_profile_value(summary, name):
    return summary[name]["maximum"]


def _peak_resident_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    arguments = parse_arguments()
    arguments.l_max = int(arguments.l_max)
    if arguments.l_max < 0:
        raise ValueError("l-max must be nonnegative")
    actions = _positive_integer(arguments.actions, "actions")
    if arguments.warmup_actions < 0:
        raise ValueError("warmup-actions must be nonnegative")
    warmup_actions = int(arguments.warmup_actions)

    communicator = MPI.COMM_WORLD
    rank = communicator.rank
    ranks = communicator.size
    threads = _requested_threads()
    _validate_thread_environment(threads)
    expected_total = os.environ.get("GRID_TDSE_TOTAL_CORES")
    if expected_total is not None and int(expected_total) != ranks * threads:
        raise RuntimeError(
            "GRID_TDSE_TOTAL_CORES does not equal MPI ranks times "
            "threads per rank"
        )

    environment = runtime_environment(
        include_build_configuration=rank == 0
    )
    if (
        os.environ.get("GRID_TDSE_VALIDATE_THREADS") == "1"
        and not isinstance(environment["thread_pools"], list)
    ):
        raise RuntimeError(
            "threadpoolctl is required when thread validation is enabled"
        )
    local_oversubscription = _oversubscribed_pools(environment, threads)
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
        generator = np.random.default_rng(arguments.seed)
        global_state = generator.normal(size=representation.psi_shape)
        global_state = global_state + 1j * generator.normal(
            size=representation.psi_shape
        )
    local_state = scatter_two_radial_2d_state(
        layout,
        communicator,
        global_state,
    )
    hamiltonian = TwoRadialDistributed2DHamiltonian(
        layout,
        communicator,
        tile_size=arguments.tile_size,
        profile=True,
    )
    setup_seconds = perf_counter() - setup_started

    for _ in range(warmup_actions):
        hamiltonian(local_state)
    communicator.Barrier()
    hamiltonian.reset_profile()

    wall_started = perf_counter()
    cpu_started = process_time()
    action = None
    for _ in range(actions):
        action = hamiltonian(local_state)
    communicator.Barrier()
    local_wall_seconds = perf_counter() - wall_started
    local_cpu_seconds = process_time() - cpu_started

    profiles = communicator.gather(hamiltonian.profile, root=0)
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
        gathered_action = gather_two_radial_2d_state(action, communicator)

    if rank != 0:
        return

    maximum_wall = max(wall_seconds)
    total_cpu = sum(cpu_seconds)
    total_cores = ranks * threads
    allocated_core_seconds = maximum_wall * total_cores
    cpu_utilization = 100.0 * total_cpu / allocated_core_seconds
    profile = _profile_summary(profiles)
    correctness = None
    if gathered_action is not None:
        serial = representation.hamiltonian(
            tile_size=arguments.tile_size,
            radial_operator_mode="element-local",
        )
        reference = serial(global_state)
        difference = gathered_action - reference
        maximum_error = float(np.max(np.abs(difference)))
        reference_scale = max(float(np.max(np.abs(reference))), 1.0)
        correctness = {
            "maximum_absolute_error": maximum_error,
            "maximum_relative_error": maximum_error / reference_scale,
        }

    result = {
        "schema": "grid-tdse-non-bo-2d-execution-v1",
        "problem": {
            "system": arguments.system,
            "l_max": arguments.l_max,
            "m_max": arguments.l_max,
            "lambda_max": 2 * arguments.l_max,
            "mu_max": 2 * arguments.l_max,
            "M_tot": 0,
            "electronic_nodes": list(arguments.electronic_nodes),
            "electronic_points_per_element": arguments.electronic_points,
            "internuclear_nodes": list(arguments.internuclear_nodes),
            "internuclear_points_per_element": arguments.internuclear_points,
            "radial_coulomb_method": arguments.radial_coulomb_method,
            "global_shape": list(representation.psi_shape),
            "global_coefficients": int(np.prod(representation.psi_shape)),
            "seed": arguments.seed,
            "tile_size": arguments.tile_size,
        },
        "execution": {
            "kind": _configuration_kind(ranks, threads),
            "mpi_ranks": ranks,
            "threads_per_rank": threads,
            "numba_threads": os.environ.get("NUMBA_NUM_THREADS"),
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
        "profile": profile,
        "correctness": correctness,
        "runtime_by_rank": environments,
        "mpi_library": MPI.Get_library_version().rstrip("\x00"),
        "thread_environment_variables": list(
            THREAD_ENVIRONMENT_VARIABLES
        ),
    }

    print(f"Execution kind:             {result['execution']['kind']}")
    print(f"MPI ranks:                  {ranks}")
    print(f"Threads per rank:           {threads}")
    print(f"Total allocated cores:      {total_cores}")
    print(f"Process grid:               {layout.process_grid}")
    print(
        "Node placement:             "
        + ", ".join(item["hostname"] for item in environments)
    )
    print(
        "CPU affinity/rank:          "
        + "; ".join(
            ",".join(str(cpu) for cpu in item["cpu"]["affinity"])
            if item["cpu"]["affinity"] is not None
            else "unavailable"
            for item in environments
        )
    )
    print(f"System:                     {arguments.system}")
    print(f"l_max:                      {arguments.l_max}")
    print(f"Global coefficients:        {result['problem']['global_coefficients']}")
    print(
        "Local coefficients:         "
        f"{min(local_coefficients)}--{max(local_coefficients)}"
    )
    print(f"Warmup actions:             {warmup_actions}")
    print(f"Timed actions:              {actions}")
    print(f"Wall time:                  {maximum_wall:.6f} s")
    print(f"Wall time per action:       {maximum_wall / actions:.6f} s")
    print(f"Process CPU time (sum):     {total_cpu:.6f} s")
    print(f"Allocated-core utilization: {cpu_utilization:.2f} %")
    print(f"Peak memory/rank:           {max(peak_memory):.3f} MiB")
    print(
        "Coordinate-1 work (max):   "
        f"{_maximum_profile_value(profile, 'coordinate1_seconds'):.6f} s"
    )
    print(
        "Coordinate-2 work (max):   "
        f"{_maximum_profile_value(profile, 'coordinate2_seconds'):.6f} s"
    )
    print(
        "Coulomb work (max):        "
        f"{_maximum_profile_value(profile, 'coulomb_seconds'):.6f} s"
    )
    print(
        "Coordinate-1 halo (max):   "
        f"{_maximum_profile_value(profile, 'coordinate1_halo_seconds'):.6f} s"
    )
    print(
        "Coordinate-2 halo (max):   "
        f"{_maximum_profile_value(profile, 'coordinate2_halo_seconds'):.6f} s"
    )
    print(
        "Coordinate-1 weak (max):   "
        f"{_maximum_profile_value(profile, 'coordinate1_weak_seconds'):.6f} s"
    )
    print(
        "Coordinate-2 weak (max):   "
        f"{_maximum_profile_value(profile, 'coordinate2_weak_seconds'):.6f} s"
    )
    if correctness is not None:
        print(
            "Hamiltonian max error:      "
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
            stem="non-bo-2d-execution",
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
