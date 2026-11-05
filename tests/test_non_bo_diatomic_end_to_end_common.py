import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "spherical"
    / "two_radial"
    / "non_bo_diatomic_end_to_end_common.py"
)
SPEC = importlib.util.spec_from_file_location(
    "non_bo_diatomic_end_to_end_common",
    SCRIPT_PATH,
)
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)

DEFAULT_ELECTRONIC_NODES = COMMON.DEFAULT_ELECTRONIC_NODES
DEFAULT_INTERNUCLEAR_NODES = COMMON.DEFAULT_INTERNUCLEAR_NODES
absorbing_boundaries = COMMON.absorbing_boundaries
configuration_record = COMMON.configuration_record
dynamics_representation = COMMON.dynamics_representation
physical_setup = COMMON.physical_setup


def _arguments(**updates):
    values = {
        "system": "H2+",
        "l_max": 2,
        "M_tot": 0,
        "electronic_nodes": DEFAULT_ELECTRONIC_NODES,
        "electronic_points_per_element": 11,
        "internuclear_nodes": DEFAULT_INTERNUCLEAR_NODES,
        "internuclear_points_per_element": 15,
        "codata_set": "2018",
        "intensity_wcm2": 3.0e13,
        "wavelength_nm": 800.0,
        "carrier_cycles": 3.0,
        "field_gauge": "length",
        "time_step": 0.25,
        "final_time": 200.0,
        "cap_alpha": 0.8,
        "cap_eta_r": 1.0,
        "cap_eta_R": 1.0,
        "no_cap": False,
        "sample_interval": 1,
        "checkpoint_interval": 100,
        "no_checkpoints": False,
        "wavefunction_snapshot_interval": None,
        "wavefunction_snapshot_steps": None,
        "wavefunction_snapshot_times": None,
        "no_wavefunction_snapshots": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_phase9h_defaults_build_requested_physical_problem():
    arguments = _arguments()
    setup = physical_setup(arguments)
    representation = dynamics_representation(arguments, setup)
    boundaries = absorbing_boundaries(arguments, representation)
    record = configuration_record(arguments, setup)

    assert setup["electronic_grid"].nodes == DEFAULT_ELECTRONIC_NODES
    assert setup["electronic_grid"].n_points == (11,) * 5
    assert setup["internuclear_grid"].nodes == DEFAULT_INTERNUCLEAR_NODES
    assert setup["internuclear_grid"].n_points == (15,) * 2
    assert setup["requested_final_time"] == 200.0
    assert setup["final_time"] == 200.0
    assert setup["n_steps"] == 800
    assert representation.M_tot == 0
    assert representation.basis1.coordinate_parity is None
    assert representation.basis2.coordinate_parity == 1
    assert boundaries.has_cap
    assert not boundaries.has_mask
    assert record["ground_state_symmetry"] == {
        "inversion_parity": 1,
        "pair_exchange_parity": 1,
        "symmetry_adapted_basis": True,
    }
    assert record["integrator"] == "sia"
    assert record["time_step"] == 0.25
    assert record["checkpoint_interval"] == 100
    assert record["wavefunction_snapshot_interval"] is None
    assert record["wavefunction_snapshot_steps"] == []
    assert record["wavefunction_snapshot_times"] == []
    assert record["field"]["intensity_wcm2"] == 3.0e13
    assert record["field"]["wavelength_nm"] == 800.0
    assert record["field"]["carrier_cycles"] == 3.0


def test_phase9h_final_time_is_rounded_up_without_changing_time_step():
    arguments = _arguments(final_time=2.01)
    setup = physical_setup(arguments)

    assert setup["requested_final_time"] == 2.01
    assert setup["final_time"] == 2.25
    assert setup["n_steps"] == 9
    assert setup["pulse_duration"] > setup["final_time"]


def test_phase9h_records_explicit_wavefunction_snapshot_selection():
    arguments = _arguments(
        wavefunction_snapshot_interval=100,
        wavefunction_snapshot_steps=[0, 7, 25],
        wavefunction_snapshot_times=[0.0, 1.75, 6.25],
    )
    record = configuration_record(arguments, physical_setup(arguments))

    assert record["wavefunction_snapshot_interval"] == 100
    assert record["wavefunction_snapshot_steps"] == [0, 7, 25]
    assert record["wavefunction_snapshot_times"] == [0.0, 1.75, 6.25]
