import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "summarize_non_bo_diatomic_phase9i_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("phase9i_baseline_summary", SCRIPT_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_phase9i_summarizes_hamiltonian_and_dynamics_records(tmp_path):
    hamiltonian = {
        "schema": SUMMARY.HAMILTONIAN_SCHEMA,
        "problem": {
            "system": "H2+",
            "l_max": 8,
            "global_shape": [145, 39, 79],
            "tile_size": 2048,
        },
        "execution": {
            "timed_actions": 10,
            "mpi_ranks": 4,
            "process_grid": [2, 2],
        },
        "timing": {
            "wall_seconds_per_action": 0.5,
            "allocated_core_utilization_percent": 95.0,
            "peak_resident_mib_maximum": 400.0,
        },
        "profile": {
            "coordinate1_seconds": {"maximum": 1.0},
            "coordinate2_seconds": {"maximum": 2.0},
            "coulomb_seconds": {"maximum": 2.5},
        },
        "correctness": {"maximum_absolute_error": 1.0e-13},
    }
    dynamics = {
        "phase": "9h",
        "execution": "cartesian-mpi",
        "mpi_ranks": 4,
        "process_grid": [2, 2],
        "configuration": {
            "system": "H2+",
            "l_max": 2,
            "time_step": 0.25,
            "final_time": 1.0,
            "field": {"gauge": "length"},
        },
        "accepted_steps_this_segment": 4,
        "initial_norm": 1.0,
        "final_norm": 0.99,
        "wall_time_seconds": 2.0,
        "propagation_diagnostics": {
            "operator_calls_total": 40,
            "krylov_dimension_mean": 9.5,
            "krylov_dimension_maximum": 11,
            "rejected_substeps_total": 1,
        },
    }
    _write(
        tmp_path / "hamiltonian" / "results" / "hamiltonian.json",
        hamiltonian,
    )
    _write(
        tmp_path / "dynamics" / "runs" / "dynamics" / "summary.json",
        dynamics,
    )

    hamiltonian_row = SUMMARY._hamiltonian_rows(tmp_path)[0]
    dynamics_row = SUMMARY._dynamics_rows(tmp_path)[0]

    assert hamiltonian_row["median_seconds_per_action"] == 0.5
    assert hamiltonian_row["median_coordinate1_seconds_per_action"] == 0.1
    assert hamiltonian_row["median_coordinate2_seconds_per_action"] == 0.2
    assert hamiltonian_row["median_coulomb_seconds_per_action"] == 0.25
    assert hamiltonian_row["maximum_absolute_error"] == pytest.approx(1.0e-13)
    assert dynamics_row["median_seconds_per_step"] == 0.5
    assert dynamics_row["median_seconds_per_operator_call"] == 0.05
    assert dynamics_row["median_krylov_dimension_mean"] == 9.5
    assert dynamics_row["maximum_norm_loss"] == pytest.approx(0.01)
