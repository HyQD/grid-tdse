"""Summarize isolated non-BO Coulomb-kernel benchmark records.

Pass one or more JSON files or result directories. For example::

    python benchmarks/summarize_non_bo_diatomic_coulomb_execution.py \
        coulomb-kernel/results --csv coulomb-kernel/summary.csv

The output groups identical physical problems while retaining tile size and
MPI/Numba execution configuration as explicit comparison columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


SCHEMA = "grid-tdse-non-bo-coulomb-execution-v1"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--csv", type=Path, dest="csv_path")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _input_paths(inputs):
    paths = []
    for value in inputs:
        if value.is_dir():
            paths.extend(sorted(value.glob("*.json")))
        elif value.is_file():
            paths.append(value)
        else:
            raise FileNotFoundError(f"benchmark input does not exist: {value}")
    unique = tuple(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValueError("no JSON benchmark records were found")
    return unique


def _load_records(paths):
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        if record.get("schema") != SCHEMA:
            raise ValueError(f"unsupported benchmark schema in {path}")
        records.append((path, record))
    return records


def _problem_key(record):
    problem = record["problem"]
    retained = {
        name: problem[name]
        for name in (
            "system",
            "l_max",
            "m_max",
            "lambda_max",
            "mu_max",
            "M_tot",
            "state_dtype",
            "electronic_nodes",
            "electronic_points_per_element",
            "internuclear_nodes",
            "internuclear_points_per_element",
            "radial_coulomb_method",
            "global_shape",
            "coulomb_couplings",
            "seed",
        )
    }
    retained["timed_actions"] = record["execution"]["timed_actions"]
    return json.dumps(retained, sort_keys=True, separators=(",", ":"))


def _execution_key(record):
    problem = record["problem"]
    execution = record["execution"]
    return (
        problem["tile_size"],
        execution["total_allocated_cores"],
        execution["mpi_ranks"],
        execution["threads_per_rank"],
        execution["numba_threads_per_rank"],
        execution["kind"],
        tuple(execution["process_grid"]),
    )


def summarize(records):
    grouped = {}
    for path, record in records:
        key = (_problem_key(record), _execution_key(record))
        grouped.setdefault(key, []).append((path, record))

    by_problem = {}
    for (problem_key, execution_key), repetitions in grouped.items():
        times = [
            record["timing"]["wall_seconds_per_action"]
            for _, record in repetitions
        ]
        utilizations = [
            record["timing"]["allocated_core_utilization_percent"]
            for _, record in repetitions
        ]
        memories = [
            record["timing"]["peak_resident_mib_maximum"]
            for _, record in repetitions
        ]
        errors = [
            record["correctness"]["maximum_absolute_error"]
            for _, record in repetitions
            if record["correctness"] is not None
        ]
        (
            tile_size,
            total_cores,
            ranks,
            threads,
            numba_threads,
            kind,
            process_grid,
        ) = execution_key
        median_time = median(times)
        row = {
            "tile_size": tile_size,
            "total_cores": total_cores,
            "mpi_ranks": ranks,
            "threads_per_rank": threads,
            "numba_threads_per_rank": numba_threads,
            "kind": kind,
            "process_grid": f"{process_grid[0]}x{process_grid[1]}",
            "repetitions": len(repetitions),
            "median_seconds_per_action": median_time,
            "minimum_seconds_per_action": min(times),
            "maximum_seconds_per_action": max(times),
            "spread_percent": 100.0 * (max(times) - min(times)) / median_time,
            "median_allocated_core_utilization_percent": median(utilizations),
            "median_peak_resident_mib_per_rank": median(memories),
            "maximum_absolute_error": max(errors) if errors else None,
            "speedup_from_fixed_core_pure_mpi": None,
            "job_ids": ";".join(
                str(record["runtime_by_rank"][0]["slurm"]["SLURM_JOB_ID"])
                for _, record in repetitions
            ),
            "source_files": ";".join(str(path) for path, _ in repetitions),
        }
        by_problem.setdefault(problem_key, []).append(row)

    for rows in by_problem.values():
        baselines = {
            (row["tile_size"], row["total_cores"]): row[
                "median_seconds_per_action"
            ]
            for row in rows
            if row["threads_per_rank"] == 1
            and row["numba_threads_per_rank"] == 1
        }
        for row in rows:
            baseline = baselines.get((row["tile_size"], row["total_cores"]))
            if baseline is not None:
                row["speedup_from_fixed_core_pure_mpi"] = (
                    baseline / row["median_seconds_per_action"]
                )
        rows.sort(
            key=lambda row: (
                row["tile_size"],
                row["total_cores"],
                -row["mpi_ranks"],
            )
        )
    return by_problem


def _format_optional(value, format_spec):
    return "N/A" if value is None else format(value, format_spec)


def _print_summary(by_problem):
    for index, (problem_key, rows) in enumerate(by_problem.items(), start=1):
        problem = json.loads(problem_key)
        print(
            f"Problem {index}: {problem['system']}, l_max={problem['l_max']}, "
            f"dtype={problem['state_dtype']}, "
            f"shape={tuple(problem['global_shape'])}, "
            f"couplings={problem['coulomb_couplings']}, "
            f"actions={problem['timed_actions']}"
        )
        print(
            "tile  cores  ranks  nthr  grid  reps  median(s)  spread(%)  "
            "vs-mpi  cpu-util(%)  max-error"
        )
        for row in rows:
            print(
                f"{row['tile_size']:4d}  "
                f"{row['total_cores']:5d}  "
                f"{row['mpi_ranks']:5d}  "
                f"{row['numba_threads_per_rank']:4d}  "
                f"{row['process_grid']:<5}  "
                f"{row['repetitions']:4d}  "
                f"{row['median_seconds_per_action']:9.6f}  "
                f"{row['spread_percent']:9.2f}  "
                f"{_format_optional(row['speedup_from_fixed_core_pure_mpi'], '.3f'):>6}  "
                f"{row['median_allocated_core_utilization_percent']:11.2f}  "
                f"{_format_optional(row['maximum_absolute_error'], '.3e')}"
            )
        print()


def _write_csv(path, by_problem, *, overwrite=False):
    rows = []
    for problem_key, problem_rows in by_problem.items():
        problem = json.loads(problem_key)
        for row in problem_rows:
            rows.append(
                {
                    "system": problem["system"],
                    "l_max": problem["l_max"],
                    "state_dtype": problem["state_dtype"],
                    "global_shape": "x".join(
                        str(value) for value in problem["global_shape"]
                    ),
                    "coulomb_couplings": problem["coulomb_couplings"],
                    "timed_actions": problem["timed_actions"],
                }
                | row
            )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main():
    arguments = parse_arguments()
    paths = _input_paths(arguments.inputs)
    records = _load_records(paths)
    by_problem = summarize(records)
    _print_summary(by_problem)
    if arguments.csv_path is not None:
        destination = _write_csv(
            arguments.csv_path,
            by_problem,
            overwrite=arguments.overwrite,
        )
        print(f"Summary CSV: {destination}")


if __name__ == "__main__":
    main()
