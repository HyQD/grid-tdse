import numpy as np
import pytest

from grid_tdse.spherical.one_radial import OneElectronDiatomicSystem


def test_h2_plus_geometry_and_derived_properties():
    system = OneElectronDiatomicSystem(
        1.0,
        1.0,
        0.7,
        name="H2+",
    )

    assert system.name == "H2+"
    assert system.electron_count == 1
    assert system.electron_charge == -1.0
    assert system.particle_mass == 1.0
    assert system.internuclear_distance == pytest.approx(1.4)
    assert system.net_charge == pytest.approx(1.0)
    assert system.nuclear_repulsion_energy == pytest.approx(1.0 / 1.4)
    np.testing.assert_array_equal(
        system.nuclear_positions,
        [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
    )
    np.testing.assert_array_equal(system.nuclear_charges, [1.0, 1.0])
    np.testing.assert_array_equal(system.nuclear_dipole, [0.0, 0.0, 0.0])


def test_heteronuclear_effective_charges_mass_and_nuclear_dipole():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.5,
        nuclear_charge2=np.float64(2.25),
        half_bond_length=1.2,
        particle_mass=0.8,
    )

    assert system.net_charge == pytest.approx(2.75)
    assert system.particle_mass == pytest.approx(0.8)
    assert system.nuclear_repulsion_energy == pytest.approx(1.5 * 2.25 / 2.4)
    np.testing.assert_allclose(
        system.nuclear_dipole,
        [0.0, 0.0, 1.2 * (2.25 - 1.5)],
    )


def test_returned_nuclear_arrays_are_read_only():
    system = OneElectronDiatomicSystem(1.0, 2.0, 0.5)

    assert not system.nuclear_positions.flags.writeable
    assert not system.nuclear_charges.flags.writeable
    assert not system.nuclear_dipole.flags.writeable


@pytest.mark.parametrize(
    "field",
    [
        "nuclear_charge1",
        "nuclear_charge2",
        "half_bond_length",
        "particle_mass",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_physical_parameters_must_be_positive_and_finite(field, value):
    parameters = {
        "nuclear_charge1": 1.0,
        "nuclear_charge2": 1.0,
        "half_bond_length": 1.0,
        "particle_mass": 1.0,
    }
    parameters[field] = value

    with pytest.raises(ValueError):
        OneElectronDiatomicSystem(**parameters)


@pytest.mark.parametrize("value", [True, "1.0", 1.0 + 0.0j])
def test_physical_parameters_must_be_real_scalars(value):
    with pytest.raises(TypeError, match="real scalar"):
        OneElectronDiatomicSystem(value, 1.0, 1.0)


@pytest.mark.parametrize(
    ("name", "error"),
    [(1, TypeError), ("", ValueError), ("   ", ValueError)],
)
def test_name_is_optional_but_must_be_a_nonempty_string(name, error):
    with pytest.raises(error):
        OneElectronDiatomicSystem(1.0, 1.0, 1.0, name=name)
