import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "summarize_non_bo_diatomic_2d_execution.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_non_bo_diatomic_2d_execution",
    SCRIPT_PATH,
)
SUMMARY_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY_MODULE)


def _record(
    *,
    seconds_per_action,
    total_cores,
    mpi_ranks,
    threads_per_rank,
    job_id,
):
    return {
        "schema": SUMMARY_MODULE.SCHEMA,
        "problem": {
            "system": "H2+",
            "l_max": 2,
            "m_max": 2,
            "lambda_max": 2,
            "mu_max": 4,
            "M_tot": 0,
            "electronic_nodes": [0.0, 4.0, 8.0],
            "electronic_points_per_element": 5,
            "internuclear_nodes": [0.0, 2.0, 4.0],
            "internuclear_points_per_element": 5,
            "radial_coulomb_method": "poisson",
            "global_shape": [8, 9, 8],
            "seed": 1729,
            "tile_size": 16,
        },
        "execution": {
            "timed_actions": 20,
            "total_allocated_cores": total_cores,
            "mpi_ranks": mpi_ranks,
            "threads_per_rank": threads_per_rank,
            "kind": (
                "serial"
                if total_cores == 1
                else "shared-memory"
                if mpi_ranks == 1
                else "pure-mpi"
                if threads_per_rank == 1
                else "hybrid"
            ),
            "process_grid": [1, mpi_ranks],
        },
        "timing": {
            "wall_seconds_per_action": seconds_per_action,
            "allocated_core_utilization_percent": 95.0,
            "peak_resident_mib_maximum": 256.0,
        },
        "correctness": {"maximum_absolute_error": 2.0e-13},
        "runtime_by_rank": [
            {"slurm": {"SLURM_JOB_ID": str(job_id)}}
        ],
    }


def test_summarize_reports_repetition_statistics_and_scaling(tmp_path):
    records = [
        (
            tmp_path / "serial.json",
            _record(
                seconds_per_action=4.0,
                total_cores=1,
                mpi_ranks=1,
                threads_per_rank=1,
                job_id=100,
            ),
        ),
        (
            tmp_path / "mpi-1.json",
            _record(
                seconds_per_action=2.0,
                total_cores=2,
                mpi_ranks=2,
                threads_per_rank=1,
                job_id=101,
            ),
        ),
        (
            tmp_path / "mpi-2.json",
            _record(
                seconds_per_action=3.0,
                total_cores=2,
                mpi_ranks=2,
                threads_per_rank=1,
                job_id=102,
            ),
        ),
    ]

    by_problem = SUMMARY_MODULE.summarize(records)

    assert len(by_problem) == 1
    rows = next(iter(by_problem.values()))
    assert len(rows) == 2
    serial, mpi = rows
    assert serial["median_seconds_per_action"] == 4.0
    assert serial["speedup_from_one_core"] == 1.0
    assert serial["parallel_efficiency"] == 1.0
    assert mpi["repetitions"] == 2
    assert mpi["median_seconds_per_action"] == 2.5
    assert mpi["spread_percent"] == pytest.approx(40.0)
    assert mpi["speedup_from_one_core"] == pytest.approx(1.6)
    assert mpi["parallel_efficiency"] == pytest.approx(0.8)
    assert mpi["maximum_absolute_error"] == 2.0e-13
    assert mpi["job_ids"] == "101;102"
