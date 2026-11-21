import importlib.util
import json
from pathlib import Path


BENCHMARK = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "spherical_one_radial_off_axis_performance.py"
)


def _module():
    specification = importlib.util.spec_from_file_location(
        "spherical_one_radial_off_axis_performance",
        BENCHMARK,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_tiny_serial_benchmark_records_all_step17_diagnostics(tmp_path):
    benchmark = _module()
    destination = tmp_path / "result.json"

    payload = benchmark.main(
        [
            "--l-max",
            "1",
            "--r-max",
            "4",
            "--points-per-element",
            "4",
            "--threads",
            "1",
            "--action-warmups",
            "0",
            "--action-repeats",
            "1",
            "--observable-warmups",
            "0",
            "--observable-repeats",
            "1",
            "--propagation-steps",
            "1",
            "--integrators",
            "sia",
            "crank-nicolson",
            "--ground-state-tolerance",
            "1e-8",
            "--result-json",
            str(destination),
        ]
    )

    assert destination.is_file()
    assert json.loads(destination.read_text()) == payload
    assert payload["schema"] == benchmark.SCHEMA
    assert payload["thread_control"]["requested_threads"] == 1
    assert payload["geometry"]["theta_deg"] == 45.0
    assert len(payload["cases"]) == 1

    case = payload["cases"][0]
    problem = case["problem"]
    assert problem["l_max"] == 1
    assert problem["angular_channels"] == 4
    assert problem["state_coefficients"] == (
        problem["active_radial_points"] * problem["angular_channels"]
    )
    assert problem["active_potential_components"] > 0
    assert problem["potential_coupling_nonzeros"] > 0
    assert case["array_inventory"]["unique_dense_and_sparse_bytes"] > 0
    assert case["hamiltonian_action"]["median_seconds"] >= 0.0
    assert case["davidson"]["converged"]
    assert case["davidson"]["hamiltonian_actions"] > 0
    assert "field_free_energy_numerator" in case["observables"]
    assert "radial_probability" in case["observables"]

    sia = case["propagation"]["sia"][0]
    crank_nicolson = case["propagation"]["crank-nicolson"][0]
    assert sia["accepted_steps"] == 1
    assert len(sia["krylov_dimensions"]) == 1
    assert crank_nicolson["accepted_steps"] == 1
    assert crank_nicolson["linear_iterations"][0] > 0
