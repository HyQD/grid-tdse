import grid_tdse
from grid_tdse import spherical
from grid_tdse.spherical import two_radial


def test_package_exports_spherical_two_radial_namespace():
    assert grid_tdse.__version__ == "0.0.0"
    assert grid_tdse.spherical is spherical
    assert spherical.two_radial is two_radial
    assert two_radial.Hamiltonian.__module__ == (
        "grid_tdse.spherical.two_radial.hamiltonian"
    )
