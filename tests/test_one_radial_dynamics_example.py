import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


EXAMPLE = (
    Path(__file__).parents[1]
    / "examples"
    / "spherical"
    / "one_radial"
    / "h2plus_laser_dynamics.py"
)


@pytest.fixture(scope="module")
def example_module():
    specification = importlib.util.spec_from_file_location(
        "h2plus_laser_dynamics", EXAMPLE
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_radial_nodes_retain_nuclear_boundary(example_module):
    nodes = example_module.radial_nodes(8.0, 3.0)

    np.testing.assert_allclose(nodes, [0.0, 1.0, 10.0 / 3.0, 17.0 / 3.0, 8.0])
    assert nodes[1] == 1.0
    assert np.max(np.diff(nodes)[1:]) <= 3.0


def test_example_runs_public_interface_and_writes_outputs(
    example_module,
    tmp_path,
    capsys,
):
    archive = tmp_path / "result.npz"
    summary = tmp_path / "result.json"

    metadata = example_module.main(
        [
            "--gauge",
            "velocity",
            "--integrator",
            "sia",
            "--absorber",
            "mask",
            "--field-strength",
            "0.001",
            "--frequency",
            "8.0",
            "--cycles",
            "1",
            "--time-step",
            "0.2",
            "--r-max",
            "4",
            "--element-width",
            "2",
            "--points-per-element",
            "4",
            "--l-max",
            "2",
            "--radial-probability-interval",
            "2",
            "--progress",
            "none",
            "--archive",
            str(archive),
            "--summary",
            str(summary),
        ]
    )

    output = capsys.readouterr().out
    assert "One-radial clamped-nuclei H2+ dynamics" in output
    assert "State source:               computed_projected_ground_state" in output
    assert "Gauge:                      velocity" in output
    assert "Absorber:                   mask" in output
    assert archive.is_file()
    assert summary.is_file()

    stored_summary = json.loads(summary.read_text())
    assert stored_summary == metadata
    assert stored_summary["warning"].startswith("smoke-test")
    assert stored_summary["pulse"]["gauge"] == "velocity"
    assert stored_summary["integrator"]["name"] == "sia"
    assert stored_summary["absorber"]["type"] == "RadialCosineMask"
    assert stored_summary["initial_state"]["source"] == (
        "computed_projected_ground_state"
    )
    assert 0.0 < stored_summary["results"]["final_norm"] <= (
        stored_summary["results"]["initial_norm"] + 1.0e-12
    )
    assert stored_summary["results"]["norm_change"] <= 1.0e-12

    with np.load(archive, allow_pickle=False) as stored:
        assert stored["final_state"].shape == (8, 3)
        assert stored["times"].shape == stored["norm"].shape
        assert stored["radial_probability"].shape[1] == 8
        assert stored["state_populations"].shape[1] == 1
        archived_metadata = json.loads(stored["metadata_json"].item())
    assert archived_metadata == metadata


def test_example_rejects_cap_with_unsupported_integrator(example_module):
    with pytest.raises(ValueError, match="cubic CAP"):
        example_module.main(
            [
                "--absorber",
                "cap",
                "--integrator",
                "crank-nicolson",
                "--progress",
                "none",
            ]
        )
