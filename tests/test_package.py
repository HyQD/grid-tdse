import grid_tdse
from grid_tdse import ground_state
from grid_tdse import real_time
from grid_tdse import spherical
from grid_tdse.spherical import two_radial


def test_package_exports_spherical_two_radial_namespace():
    assert grid_tdse.__version__ == "0.0.0"
    assert grid_tdse.ground_state is ground_state
    assert grid_tdse.real_time is real_time
    assert grid_tdse.spherical is spherical
    assert spherical.two_radial is two_radial
    assert two_radial.TwoRadialHamiltonian.__module__ == (
        "grid_tdse.spherical.two_radial.hamiltonian"
    )
    assert ground_state.ImaginaryTimeGroundStateSolver.__module__ == (
        "grid_tdse.ground_state.imaginary_time"
    )
    assert real_time.RealTimePropagator.__module__ == (
        "grid_tdse.real_time.propagation"
    )
    assert real_time.RealTimeCheckpoint.__module__ == (
        "grid_tdse.real_time.checkpoint"
    )
    assert two_radial.two_radial_compatibility_metadata.__module__ == (
        "grid_tdse.spherical.two_radial.state_archive"
    )
    assert two_radial.TwoRadialDistributedLayout.__module__ == (
        "grid_tdse.spherical.two_radial.distributed_state"
    )
