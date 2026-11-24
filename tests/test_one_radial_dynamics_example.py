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


def test_radial_nodes_retain_distinct_off_center_nuclear_radii(example_module):
    nodes = example_module.radial_nodes(
        8.0,
        3.0,
        nuclear_radii=[0.5, 1.5],
    )

    assert 0.5 in nodes
    assert 1.5 in nodes
    assert np.max(np.diff(nodes)) <= 3.0


def test_default_pulse_and_oriented_geometry_are_explicit(example_module):
    arguments = example_module.parse_arguments(
        ["--theta-deg", "45", "--phi-deg", "30"]
    )
    omega, wavelength_nm = example_module._frequency(arguments)
    system, specification = example_module._system(arguments, 1.0)

    assert wavelength_nm == 400.0
    assert omega == pytest.approx(0.113908381322799)
    assert arguments.cycles == 3.0
    assert arguments.field_strength == 0.05
    assert specification == "bond_orientation"
    np.testing.assert_allclose(
        system.bond_axis,
        [np.sqrt(3.0 / 8.0), np.sqrt(1.0 / 8.0), np.sqrt(0.5)],
    )


def test_explicit_positions_are_mutually_exclusive_with_angles(example_module):
    arguments = example_module.parse_arguments(
        [
            "--nuclear-position-1",
            "0",
            "0",
            "-1",
            "--nuclear-position-2",
            "0",
            "0",
            "1",
            "--theta-deg",
            "30",
        ]
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        example_module._system(arguments, 1.0)


def test_explicit_cartesian_positions_define_geometry(example_module):
    arguments = example_module.parse_arguments(
        [
            "--nuclear-position-1",
            "-0.5",
            "0.25",
            "-1.0",
            "--nuclear-position-2",
            "0.5",
            "0.25",
            "1.0",
        ]
    )
    system, specification = example_module._system(arguments, 1.0)

    assert specification == "cartesian_positions"
    np.testing.assert_allclose(
        system.nuclear_positions,
        [[-0.5, 0.25, -1.0], [0.5, 0.25, 1.0]],
    )
    np.testing.assert_allclose(system.nuclear_midpoint, [0.0, 0.25, 0.0])


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
            "--theta-deg",
            "45",
            "--phi-deg",
            "30",
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
    assert "Full-angular one-radial clamped-nuclei H2+ dynamics" in output
    assert "State source:               computed_full_angular_ground_state" in output
    assert "Gauge:                      velocity" in output
    assert "Absorber:                   mask" in output
    assert archive.is_file()
    assert summary.is_file()

    stored_summary = json.loads(summary.read_text())
    assert stored_summary == metadata
    assert stored_summary["warning"].startswith("development inputs")
    assert stored_summary["geometry"]["specification"] == "bond_orientation"
    assert stored_summary["grid"]["angular_representation"] == "full"
    assert stored_summary["grid"]["m_max"] == 2
    assert len(stored_summary["grid"]["channel_lm"]) == 9
    assert stored_summary["pulse"]["gauge"] == "velocity"
    assert stored_summary["integrator"]["name"] == "sia"
    assert stored_summary["absorber"]["type"] == "RadialCosineMask"
    assert stored_summary["initial_state"]["source"] == (
        "computed_full_angular_ground_state"
    )
    assert 0.0 < stored_summary["results"]["final_norm"] <= (
        stored_summary["results"]["initial_norm"] + 1.0e-12
    )
    assert stored_summary["results"]["norm_change"] <= 1.0e-12

    with np.load(archive, allow_pickle=False) as stored:
        assert stored["final_state"].shape == (8, 9)
        assert stored["times"].shape == stored["norm"].shape
        assert stored["radial_probability"].shape[1] == 8
        assert stored["state_populations"].shape[1] == 1
        assert stored["induced_dipole_numerator"].shape[1] == 3
        archived_metadata = json.loads(stored["metadata_json"].item())
    assert archived_metadata == metadata


def test_example_rejects_cap_with_sil(example_module):
    with pytest.raises(ValueError, match="cubic CAP"):
        example_module.main(
            [
                "--absorber",
                "cap",
                "--integrator",
                "sil",
                "--progress",
                "none",
            ]
        )
