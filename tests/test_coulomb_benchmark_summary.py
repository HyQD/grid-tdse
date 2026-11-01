import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "summarize_non_bo_diatomic_coulomb_execution.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_non_bo_diatomic_coulomb_execution",
    SCRIPT_PATH,
)
SUMMARY_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY_MODULE)


def _record(
    *,
    seconds_per_action,
    tile_size,
    mpi_ranks,
    numba_threads,
    job_id,
):
    total_cores = mpi_ranks * numba_threads
    return {
        "schema": SUMMARY_MODULE.SCHEMA,
        "problem": {
            "system": "H2+",
            "l_max": 8,
            "m_max": 8,
            "lambda_max": 16,
            "mu_max": 16,
            "M_tot": 0,
            "state_dtype": "complex128",
            "electronic_nodes": [0.0, 4.0, 8.0],
            "electronic_points_per_element": 5,
            "internuclear_nodes": [0.0, 2.0, 4.0],
            "internuclear_points_per_element": 5,
            "radial_coulomb_method": "poisson",
            "global_shape": [29, 9, 9],
            "coulomb_couplings": 63673,
            "seed": 26111,
            "tile_size": tile_size,
        },
        "execution": {
            "timed_actions": 2000,
            "total_allocated_cores": total_cores,
            "mpi_ranks": mpi_ranks,
            "threads_per_rank": numba_threads,
            "numba_threads_per_rank": numba_threads,
            "kind": "pure-mpi" if numba_threads == 1 else "hybrid",
            "process_grid": [2, mpi_ranks // 2],
        },
        "timing": {
            "wall_seconds_per_action": seconds_per_action,
            "allocated_core_utilization_percent": 97.0,
            "peak_resident_mib_maximum": 320.0,
        },
        "correctness": {"maximum_absolute_error": 3.0e-13},
        "runtime_by_rank": [
            {"slurm": {"SLURM_JOB_ID": str(job_id)}}
        ],
    }


def test_summarize_groups_repetitions_and_uses_fixed_core_mpi_baseline(
    tmp_path,
):
    records = [
        (
            tmp_path / "mpi-1.json",
            _record(
                seconds_per_action=4.0,
                tile_size=256,
                mpi_ranks=16,
                numba_threads=1,
                job_id=101,
            ),
        ),
        (
            tmp_path / "mpi-2.json",
            _record(
                seconds_per_action=5.0,
                tile_size=256,
                mpi_ranks=16,
                numba_threads=1,
                job_id=102,
            ),
        ),
        (
            tmp_path / "hybrid.json",
            _record(
                seconds_per_action=3.0,
                tile_size=256,
                mpi_ranks=8,
                numba_threads=2,
                job_id=103,
            ),
        ),
    ]

    by_problem = SUMMARY_MODULE.summarize(records)

    assert len(by_problem) == 1
    mpi, hybrid = next(iter(by_problem.values()))
    assert mpi["repetitions"] == 2
    assert mpi["median_seconds_per_action"] == 4.5
    assert mpi["spread_percent"] == pytest.approx(100.0 / 4.5)
    assert mpi["speedup_from_fixed_core_pure_mpi"] == 1.0
    assert mpi["job_ids"] == "101;102"
    assert hybrid["median_seconds_per_action"] == 3.0
    assert hybrid["speedup_from_fixed_core_pure_mpi"] == 1.5
    assert hybrid["maximum_absolute_error"] == 3.0e-13
