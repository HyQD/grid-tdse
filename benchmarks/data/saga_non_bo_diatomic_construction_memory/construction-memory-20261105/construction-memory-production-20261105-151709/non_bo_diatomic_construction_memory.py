"""Measure retained memory while constructing a distributed two-radial problem.

This diagnostic deliberately performs no Hamiltonian action or propagation.  Each
MPI rank independently constructs the same global representation and then the
rank-local Cartesian Hamiltonian, matching the current production setup path.
It records both process RSS and a byte inventory of the principal NumPy arrays.

For a small local smoke test::

    mpiexec -n 1 python benchmarks/non_bo_diatomic_construction_memory.py \
        --l-max 2 --electronic-elements 2 --electronic-points 5 \
        --internuclear-elements 2 --internuclear-points 5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
from time import perf_counter

from mpi4py import MPI
import numpy as np

from grid_lib.spherical_coordinates.angular_momentum import get_gaunt_backend
from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    FEMDVRGridSpec,
    TwoRadialDistributed2DHamiltonian,
    TwoRadialDistributed2DLayout,
    TwoRadialSystemSpec,
    radial_potential,
)
from grid_tdse.spherical.two_radial.diatomic import _diatomic_parameters


SCHEMA = "grid-tdse-two-radial-construction-memory-v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), default="H2+")
    parser.add_argument("--l-max", type=int, default=8)
    parser.add_argument("--codata-set", choices=("2018", "2022"), default="2018")
    parser.add_argument("--electronic-elements", type=int, default=4)
    parser.add_argument("--electronic-rmax", type=float, default=16.0)
    parser.add_argument("--electronic-points", type=int, default=11)
    parser.add_argument("--internuclear-elements", type=int, default=4)
    parser.add_argument("--internuclear-rmax", type=float, default=8.0)
    parser.add_argument("--internuclear-points", type=int, default=21)
    parser.add_argument(
        "--process-grid",
        type=int,
        nargs=2,
        metavar=("P_r", "P_R"),
        help="explicit Cartesian process grid; default: automatic",
    )
    parser.add_argument(
        "--layout-ranks",
        type=int,
        help=(
            "logical ranks represented by the layout; with one MPI process, "
            "rank zero is emulated without allocating a state"
        ),
    )
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument(
        "--profile-name",
        default="custom",
        help="descriptive label stored in the result",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    for name in (
        "l_max",
        "electronic_elements",
        "electronic_points",
        "internuclear_elements",
        "internuclear_points",
        "tile_size",
    ):
        value = getattr(arguments, name)
        minimum = 0 if name == "l_max" else 1
        if value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be at least {minimum}")
    for name in ("electronic_rmax", "internuclear_rmax"):
        if not np.isfinite(getattr(arguments, name)) or getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    layout_ranks = (
        MPI.COMM_WORLD.size
        if arguments.layout_ranks is None
        else arguments.layout_ranks
    )
    if layout_ranks < 1:
        parser.error("--layout-ranks must be positive")
    if MPI.COMM_WORLD.size not in {1, layout_ranks}:
        parser.error("MPI size must be one or equal --layout-ranks")
    arguments.layout_ranks = layout_ranks
    if arguments.process_grid is not None:
        if any(value <= 0 for value in arguments.process_grid):
            parser.error("--process-grid dimensions must be positive")
        if int(np.prod(arguments.process_grid)) != layout_ranks:
            parser.error("--process-grid dimensions must multiply to layout ranks")
    return arguments


def _grid(elements: int, maximum: float, points: int) -> FEMDVRGridSpec:
    nodes = np.linspace(0.0, maximum, elements + 1)
    return FEMDVRGridSpec(nodes, [points] * elements)


def _specification(arguments: argparse.Namespace) -> TwoRadialSystemSpec:
    parameters = _diatomic_parameters(arguments.system, arguments.codata_set)
    identical_pair = parameters.has_identical_pair
    electronic_grid = _grid(
        arguments.electronic_elements,
        arguments.electronic_rmax,
        arguments.electronic_points,
    )
    internuclear_grid = _grid(
        arguments.internuclear_elements,
        arguments.internuclear_rmax,
        arguments.internuclear_points,
    )
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=electronic_grid,
            basis=AngularBasisSpec(arguments.l_max, arguments.l_max),
            kinetic_prefactor=-0.5 / parameters.electronic_reduced_mass,
        ),
        coordinate2=CoordinateSpec(
            grid=internuclear_grid,
            basis=AngularBasisSpec(
                arguments.l_max,
                arguments.l_max,
                coordinate_parity=1 if identical_pair else None,
            ),
            kinetic_prefactor=-0.5 / parameters.internuclear_reduced_mass,
            one_body_terms=(
                radial_potential(
                    lambda radius: (
                        parameters.reference_charge
                        * parameters.partner_charge
                        / radius
                    )
                ),
            ),
        ),
        multipoles=CoulombMultipoleSpec(
            2 * arguments.l_max,
            2 * arguments.l_max,
            lambda_parity=1 if identical_pair else None,
        ),
        coulomb_terms=(
            CoulombTerm(
                parameters.reference_center_scale,
                parameters.electron_charge * parameters.reference_charge,
            ),
            CoulombTerm(
                parameters.partner_center_scale,
                parameters.electron_charge * parameters.partner_charge,
            ),
        ),
        M_tot=0,
        radial_coulomb_method="poisson",
    )


def _linux_process_memory_mib() -> dict[str, float | None]:
    values: dict[str, float] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as stream:
            for line in stream:
                name, separator, remainder = line.partition(":")
                if separator and name in {"VmRSS", "VmHWM"}:
                    values[name] = float(remainder.split()[0]) / 1024.0
    except OSError:
        pass

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resource_peak = (
        maximum / 1024.0
        if sys.platform.startswith("linux")
        else maximum / 2**20
    )
    return {
        "current_rss_mib": values.get("VmRSS"),
        "peak_rss_mib": values.get("VmHWM", resource_peak),
        "resource_peak_rss_mib": resource_peak,
    }


def _snapshot(
    communicator: MPI.Comm,
    name: str,
    *,
    elapsed_seconds: float | None = None,
) -> dict[str, float | str | None]:
    gc.collect()
    communicator.Barrier()
    result: dict[str, float | str | None] = {"stage": name}
    result.update(_linux_process_memory_mib())
    result["elapsed_seconds"] = elapsed_seconds
    return result


def _root_array(array: np.ndarray) -> np.ndarray:
    root = np.asarray(array)
    while isinstance(root.base, np.ndarray):
        root = root.base
    return root


def _array_inventory(representation, hamiltonian) -> dict:
    entries: list[tuple[str, str, np.ndarray]] = []

    def add(category: str, name: str, value) -> None:
        if isinstance(value, np.ndarray):
            entries.append((category, name, value))

    add("dense_one_body", "representation.h1", representation.h1)
    add("dense_one_body", "representation.h2", representation.h2)
    add(
        "dense_one_body",
        "serial_hamiltonian.h2_transpose",
        hamiltonian.serial_hamiltonian.h2_transpose,
    )
    add(
        "expanded_pair_operators",
        "serial_hamiltonian._h1_pairs",
        getattr(hamiltonian.serial_hamiltonian, "_h1_pairs", None),
    )
    add(
        "expanded_pair_operators",
        "serial_hamiltonian._h2_pairs_transpose",
        getattr(hamiltonian.serial_hamiltonian, "_h2_pairs_transpose", None),
    )
    add(
        "global_radial_multipoles",
        "representation.radial_multipoles.values",
        representation.radial_multipoles.values,
    )
    add(
        "global_radial_multipoles",
        "serial_hamiltonian._radial_flat",
        hamiltonian.serial_hamiltonian._radial_flat,
    )
    add(
        "local_radial_multipoles",
        "distributed_hamiltonian._local_radial_coulomb",
        hamiltonian._local_radial_coulomb[hamiltonian.rank],
    )

    for name in (
        "output_1",
        "output_2",
        "offsets",
        "input_1",
        "input_2",
        "radial_index",
        "coefficient",
    ):
        add(
            "global_couplings",
            f"representation.couplings.{name}",
            getattr(representation.couplings, name),
        )
    compact = getattr(hamiltonian.serial_hamiltonian, "_compact_couplings", ())
    for index, value in enumerate(compact):
        add(
            "compact_couplings",
            f"serial_hamiltonian._compact_couplings[{index}]",
            value,
        )

    for name, value in (
        ("representation.h1_diagonal", representation.h1_diagonal),
        ("representation.h2_diagonal", representation.h2_diagonal),
        (
            "serial_hamiltonian._h1_diagonal_pairs",
            getattr(hamiltonian.serial_hamiltonian, "_h1_diagonal_pairs", None),
        ),
        (
            "serial_hamiltonian._h2_diagonal_pairs",
            getattr(hamiltonian.serial_hamiltonian, "_h2_diagonal_pairs", None),
        ),
        ("distributed_hamiltonian._h1_diagonal", hamiltonian._h1_diagonal),
        ("distributed_hamiltonian._h2_diagonal", hamiltonian._h2_diagonal),
    ):
        add("one_body_diagonals", name, value)

    for coordinate, matrices in hamiltonian._element_matrices.items():
        for index, matrix in enumerate(matrices):
            add("element_matrices", f"second_derivative[{coordinate}][{index}]", matrix)
    for coordinate, matrices in hamiltonian._first_derivative_element_matrices.items():
        for index, matrix in enumerate(matrices):
            add("element_matrices", f"first_derivative[{coordinate}][{index}]", matrix)
    roots: dict[int, dict] = {}
    named_entries = []
    for category, name, array in entries:
        root = _root_array(array)
        key = id(root)
        owns_unique_storage = key not in roots
        if owns_unique_storage:
            roots[key] = {
                "category": category,
                "nbytes": int(root.nbytes),
            }
        named_entries.append(
            {
                "category": category,
                "name": name,
                "shape": list(array.shape),
                "dtype": array.dtype.name,
                "view_nbytes": int(array.nbytes),
                "root_nbytes": int(root.nbytes),
                "owns_unique_storage_in_inventory": owns_unique_storage,
            }
        )

    category_bytes: defaultdict[str, int] = defaultdict(int)
    for item in roots.values():
        category_bytes[item["category"]] += item["nbytes"]
    return {
        "named_entries": named_entries,
        "unique_bytes_by_category": dict(sorted(category_bytes.items())),
        "unique_bytes_total": int(sum(category_bytes.values())),
    }


class _LayoutCommunicator:
    """Expose rank/size for construction without performing communication."""

    def __init__(self, rank: int, size: int) -> None:
        self.rank = rank
        self.size = size


def _backend_information() -> dict[str, object]:
    specification = importlib.util.find_spec("py3nj")
    information: dict[str, object] = {
        "configured": get_gaunt_backend(resolve=False),
        "resolved": get_gaunt_backend(resolve=True),
        "py3nj_available": specification is not None,
        "py3nj_path": None if specification is None else specification.origin,
        "py3nj_version": None,
    }
    if specification is not None:
        try:
            information["py3nj_version"] = importlib.metadata.version("py3nj")
        except importlib.metadata.PackageNotFoundError:
            module = importlib.import_module("py3nj")
            information["py3nj_version"] = getattr(module, "__version__", None)
    return information


def _write_json(path: Path, result: dict, *, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def _stage_value(stages: list[dict], name: str, field: str) -> float | None:
    for stage in stages:
        if stage["stage"] == name:
            return stage[field]
    raise KeyError(name)


def main() -> None:
    arguments = _arguments()
    communicator = MPI.COMM_WORLD
    rank = communicator.rank
    backend = _backend_information()
    stages = [_snapshot(communicator, "imports_complete")]

    started = perf_counter()
    representation = _specification(arguments).build()
    stages.append(
        _snapshot(
            communicator,
            "representation_built",
            elapsed_seconds=perf_counter() - started,
        )
    )

    started = perf_counter()
    layout = TwoRadialDistributed2DLayout(
        representation,
        arguments.layout_ranks,
        process_grid=(
            None
            if arguments.process_grid is None
            else tuple(arguments.process_grid)
        ),
    )
    stages.append(
        _snapshot(
            communicator,
            "layout_built",
            elapsed_seconds=perf_counter() - started,
        )
    )

    started = perf_counter()
    operator_communicator = (
        communicator
        if communicator.size == arguments.layout_ranks
        else _LayoutCommunicator(rank=0, size=arguments.layout_ranks)
    )
    hamiltonian = TwoRadialDistributed2DHamiltonian(
        layout,
        operator_communicator,
        tile_size=arguments.tile_size,
        profile=False,
    )
    stages.append(
        _snapshot(
            communicator,
            "distributed_hamiltonian_built",
            elapsed_seconds=perf_counter() - started,
        )
    )

    inventory = _array_inventory(representation, hamiltonian)
    layout_rank = rank if communicator.size == arguments.layout_ranks else 0
    owned_shape = layout.owned_shape(layout_rank)
    owned_complex_state_bytes = int(np.prod(owned_shape, dtype=np.int64)) * 16

    local = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "backend": backend,
        "stages": stages,
        "layout_rank": layout_rank,
        "owned_shape": list(owned_shape),
        "owned_complex_state_bytes": owned_complex_state_bytes,
        "array_inventory": inventory,
    }
    by_rank = communicator.gather(local, root=0)
    if rank != 0:
        return

    result = {
        "schema": SCHEMA,
        "problem": {
            "profile_name": arguments.profile_name,
            "system": arguments.system,
            "codata_set": arguments.codata_set,
            "l_max": arguments.l_max,
            "m_max": arguments.l_max,
            "lambda_max": 2 * arguments.l_max,
            "mu_max": 2 * arguments.l_max,
            "M_tot": 0,
            "electronic_elements": arguments.electronic_elements,
            "electronic_rmax": arguments.electronic_rmax,
            "electronic_points_per_element": arguments.electronic_points,
            "internuclear_elements": arguments.internuclear_elements,
            "internuclear_rmax": arguments.internuclear_rmax,
            "internuclear_points_per_element": arguments.internuclear_points,
            "radial_points": [
                representation.grid1.r.size,
                representation.grid2.r.size,
            ],
            "angular_pairs": representation.angular_pairs.n_pairs,
            "coulomb_couplings": representation.couplings.n_pairs,
            "global_shape": list(representation.psi_shape),
        },
        "execution": {
            "mpi_processes": communicator.size,
            "layout_ranks": arguments.layout_ranks,
            "process_grid": list(layout.process_grid),
            "tile_size": arguments.tile_size,
        },
        "backend": backend,
        "ranks": by_rank,
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "mpi_library": MPI.Get_library_version().rstrip("\x00"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }

    final_rss = [
        _stage_value(
            item["stages"],
            "distributed_hamiltonian_built",
            "current_rss_mib",
        )
        for item in by_rank
    ]
    final_peak = [
        _stage_value(
            item["stages"],
            "distributed_hamiltonian_built",
            "peak_rss_mib",
        )
        for item in by_rank
    ]
    print("Two-radial construction-memory diagnostic")
    print(f"Profile:                    {arguments.profile_name}")
    print(f"System:                     {arguments.system}")
    print(f"l_max:                      {arguments.l_max}")
    print(f"MPI processes:              {communicator.size}")
    print(f"Logical layout ranks:       {arguments.layout_ranks}")
    print(f"Process grid:               {layout.process_grid}")
    print(
        "Radial points:              "
        f"{representation.grid1.r.size} x {representation.grid2.r.size}"
    )
    print(f"Angular pairs:              {representation.angular_pairs.n_pairs}")
    print(f"Coulomb couplings:          {representation.couplings.n_pairs}")
    print(f"Gaunt backend:              {backend['resolved']}")
    print(f"py3nj available:            {backend['py3nj_available']}")
    print(f"py3nj path:                 {backend['py3nj_path']}")
    print(f"Final RSS/rank:             {min(final_rss):.1f}--{max(final_rss):.1f} MiB")
    print(
        "Peak RSS/rank:              "
        f"{min(final_peak):.1f}--{max(final_peak):.1f} MiB"
    )
    maximum_state_mib = max(
        item["owned_complex_state_bytes"] for item in by_rank
    ) / 2**20
    print(
        f"Logical state tile/rank:    {maximum_state_mib:.1f} MiB "
        "(not allocated)"
    )
    print("Rank-0 unique array inventory:")
    categories = by_rank[0]["array_inventory"]["unique_bytes_by_category"]
    for category, nbytes in categories.items():
        print(f"  {category:29s} {nbytes / 2**20:12.1f} MiB")
    print(
        "  "
        f"{'total':29s} "
        f"{by_rank[0]['array_inventory']['unique_bytes_total'] / 2**20:12.1f} MiB"
    )

    if arguments.result_json is not None:
        destination = _write_json(
            arguments.result_json,
            result,
            overwrite=arguments.overwrite,
        )
        print(f"Result JSON:                {destination}")


if __name__ == "__main__":
    main()
