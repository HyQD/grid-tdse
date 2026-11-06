"""Summarize Phase 9i fixed-action and production-SIA baseline records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


HAMILTONIAN_SCHEMA = "grid-tdse-non-bo-2d-execution-v1"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="default: RUN_ROOT",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _spread(values):
    center = median(values)
    return 0.0 if center == 0.0 else 100.0 * (max(values) - min(values)) / center


def _format_grid(values):
    return "x".join(str(value) for value in values)


def _kernel_label(*values):
    return "/".join(values)


def _hamiltonian_rows(run_root):
    paths = sorted((run_root / "hamiltonian" / "results").glob("*.json"))
    records = []
    for path in paths:
        record = _load(path)
        if record.get("schema") != HAMILTONIAN_SCHEMA:
            raise ValueError(f"unsupported Hamiltonian record: {path}")
        records.append((path, record))
    grouped = {}
    for path, record in records:
        problem = record["problem"]
        execution = record["execution"]
        key = (
            problem["system"],
            problem["l_max"],
            tuple(problem["global_shape"]),
            problem["tile_size"],
            problem.get("coulomb_kernel", "reference"),
            problem.get("radial_kernel", "reference"),
            execution["timed_actions"],
            execution["mpi_ranks"],
            tuple(execution["process_grid"]),
        )
        grouped.setdefault(key, []).append((path, record))

    rows = []
    for key, repetitions in sorted(grouped.items()):
        (
            system,
            l_max,
            shape,
            tile_size,
            coulomb_kernel,
            radial_kernel,
            actions,
            ranks,
            process_grid,
        ) = key
        times = [
            record["timing"]["wall_seconds_per_action"]
            for _, record in repetitions
        ]
        cpu_utilizations = [
            record["timing"]["allocated_core_utilization_percent"]
            for _, record in repetitions
        ]
        memories = [
            record["timing"]["peak_resident_mib_maximum"]
            for _, record in repetitions
        ]

        def component(name):
            return median(
                record["profile"][name]["maximum"] / actions
                for _, record in repetitions
            )

        errors = [
            record["correctness"]["maximum_absolute_error"]
            for _, record in repetitions
            if record["correctness"] is not None
        ]
        rows.append(
            {
                "workload": "hamiltonian",
                "system": system,
                "l_max": l_max,
                "global_shape": _format_grid(shape),
                "tile_size": tile_size,
                "coulomb_kernel": coulomb_kernel,
                "radial_kernel": radial_kernel,
                "timed_actions": actions,
                "mpi_ranks": ranks,
                "process_grid": _format_grid(process_grid),
                "repetitions": len(repetitions),
                "median_seconds_per_action": median(times),
                "minimum_seconds_per_action": min(times),
                "maximum_seconds_per_action": max(times),
                "spread_percent": _spread(times),
                "median_coordinate1_seconds_per_action": component(
                    "coordinate1_seconds"
                ),
                "median_coordinate2_seconds_per_action": component(
                    "coordinate2_seconds"
                ),
                "median_coulomb_seconds_per_action": component(
                    "coulomb_seconds"
                ),
                "median_cpu_utilization_percent": median(cpu_utilizations),
                "median_peak_resident_mib_per_rank": median(memories),
                "maximum_absolute_error": max(errors) if errors else None,
                "source_files": ";".join(str(path) for path, _ in repetitions),
            }
        )
    return rows


def _dynamics_rows(run_root):
    paths = sorted((run_root / "dynamics" / "runs").glob("*/summary.json"))
    records = []
    for path in paths:
        record = _load(path)
        if record.get("phase") != "9h" or record.get("execution") != "cartesian-mpi":
            raise ValueError(f"unsupported dynamics record: {path}")
        if "propagation_diagnostics" not in record:
            raise ValueError(f"dynamics record lacks Phase 9i diagnostics: {path}")
        records.append((path, record))
    grouped = {}
    for path, record in records:
        configuration = record["configuration"]
        key = (
            configuration["system"],
            configuration["l_max"],
            configuration["field"]["gauge"],
            configuration["time_step"],
            configuration["final_time"],
            configuration.get("coulomb_kernel", "reference"),
            configuration.get("radial_kernel", "reference"),
            configuration.get("dipole_kernel", "reference"),
            record["mpi_ranks"],
            tuple(record["process_grid"]),
        )
        grouped.setdefault(key, []).append((path, record))

    rows = []
    for key, repetitions in sorted(grouped.items()):
        (
            system,
            l_max,
            gauge,
            time_step,
            final_time,
            coulomb_kernel,
            radial_kernel,
            dipole_kernel,
            ranks,
            process_grid,
        ) = key
        wall_times = [record["wall_time_seconds"] for _, record in repetitions]
        steps = [record["accepted_steps_this_segment"] for _, record in repetitions]
        operator_calls = [
            record["propagation_diagnostics"]["operator_calls_total"]
            for _, record in repetitions
        ]
        wall_per_step = [wall / count for wall, count in zip(wall_times, steps)]
        wall_per_call = [
            wall / count for wall, count in zip(wall_times, operator_calls)
        ]
        norm_losses = [
            abs(record["initial_norm"] - record["final_norm"])
            for _, record in repetitions
        ]
        rows.append(
            {
                "workload": "production-sia",
                "system": system,
                "l_max": l_max,
                "field_gauge": gauge,
                "time_step": time_step,
                "final_time": final_time,
                "coulomb_kernel": coulomb_kernel,
                "radial_kernel": radial_kernel,
                "dipole_kernel": dipole_kernel,
                "accepted_steps": steps[0],
                "mpi_ranks": ranks,
                "process_grid": _format_grid(process_grid),
                "repetitions": len(repetitions),
                "median_wall_seconds": median(wall_times),
                "minimum_wall_seconds": min(wall_times),
                "maximum_wall_seconds": max(wall_times),
                "spread_percent": _spread(wall_times),
                "median_seconds_per_step": median(wall_per_step),
                "median_operator_calls_total": median(operator_calls),
                "median_seconds_per_operator_call": median(wall_per_call),
                "median_krylov_dimension_mean": median(
                    record["propagation_diagnostics"]["krylov_dimension_mean"]
                    for _, record in repetitions
                ),
                "maximum_krylov_dimension": max(
                    record["propagation_diagnostics"]["krylov_dimension_maximum"]
                    for _, record in repetitions
                ),
                "maximum_rejected_substeps": max(
                    record["propagation_diagnostics"]["rejected_substeps_total"]
                    for _, record in repetitions
                ),
                "maximum_norm_loss": max(norm_losses),
                "source_files": ";".join(str(path) for path, _ in repetitions),
            }
        )
    return rows


def _write_csv(path, rows, overwrite):
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _print_hamiltonian(rows):
    if not rows:
        print("No Hamiltonian records found.")
        return
    print("Hamiltonian baseline")
    print(
        "kernels                           ranks  grid  reps  action(s)  spread(%)  "
        "r(s)     R(s)     Vc(s)    RSS(MiB)"
    )
    for row in rows:
        print(
            f"{_kernel_label(row['coulomb_kernel'], row['radial_kernel']):<33} "
            f"{row['mpi_ranks']:5d}  {row['process_grid']:<4}  "
            f"{row['repetitions']:4d}  "
            f"{row['median_seconds_per_action']:9.6f}  "
            f"{row['spread_percent']:9.2f}  "
            f"{row['median_coordinate1_seconds_per_action']:7.4f}  "
            f"{row['median_coordinate2_seconds_per_action']:7.4f}  "
            f"{row['median_coulomb_seconds_per_action']:7.4f}  "
            f"{row['median_peak_resident_mib_per_rank']:8.1f}"
        )


def _print_dynamics(rows):
    if not rows:
        print("\nNo production-SIA records found.")
        return
    print("\nProduction-SIA baseline")
    print(
        "kernels                                            "
        "ranks  grid  reps  wall(s)  step(s)  "
        "H calls  s/H-call  Krylov  rejected"
    )
    for row in rows:
        kernel_label = _kernel_label(
            row["coulomb_kernel"],
            row["radial_kernel"],
            row["dipole_kernel"],
        )
        print(
            f"{kernel_label:<43} "
            f"{row['mpi_ranks']:5d}  {row['process_grid']:<4}  "
            f"{row['repetitions']:4d}  "
            f"{row['median_wall_seconds']:7.2f}  "
            f"{row['median_seconds_per_step']:7.4f}  "
            f"{row['median_operator_calls_total']:7.0f}  "
            f"{row['median_seconds_per_operator_call']:8.5f}  "
            f"{row['median_krylov_dimension_mean']:6.2f}  "
            f"{row['maximum_rejected_substeps']:8d}"
        )


def main():
    arguments = parse_arguments()
    run_root = arguments.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {run_root}")
    output_directory = (
        run_root
        if arguments.output_directory is None
        else arguments.output_directory.resolve()
    )
    hamiltonian_rows = _hamiltonian_rows(run_root)
    dynamics_rows = _dynamics_rows(run_root)
    if not hamiltonian_rows and not dynamics_rows:
        raise ValueError(f"no Phase 9i result records found below {run_root}")
    _print_hamiltonian(hamiltonian_rows)
    _print_dynamics(dynamics_rows)
    for name, rows in (
        ("hamiltonian_summary.csv", hamiltonian_rows),
        ("dynamics_summary.csv", dynamics_rows),
    ):
        destination = _write_csv(
            output_directory / name,
            rows,
            arguments.overwrite,
        )
        if destination is not None:
            print(f"Summary CSV: {destination}")


if __name__ == "__main__":
    main()
