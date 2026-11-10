"""Structural checks for Olivia qualification drivers."""

from pathlib import Path


MULTI_NODE_SCRIPT = (
    Path(__file__).parents[1]
    / "examples"
    / "spherical"
    / "two_radial"
    / "Olivia"
    / "qualification"
    / "qualify_grid_tdse_multi_node.sh"
)


def test_multi_node_mpi_probe_runs_inside_srun():
    source = MULTI_NODE_SCRIPT.read_text(encoding="utf-8")
    prelaunch, separator, placement_launch = source.partition("\n    srun \\")

    assert separator
    assert "from mpi4py import MPI" not in prelaunch
    assert "MPI.Get_library_version()" in placement_launch
    assert "communicator.rank == 0 and communicator.size == 2" in placement_launch
