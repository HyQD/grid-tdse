import json

import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    SphericalOneRadialAbsorbingBoundaries,
)


@pytest.fixture
def hamiltonian():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    return OneElectronDiatomicHamiltonian(
        system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[6, 6, 6],
        l_max=3,
        m_max=1,
    )


def _random_state(hamiltonian, m=0, *, seed=4, batches=()):
    generator = np.random.default_rng(seed)
    shape = (*batches, *hamiltonian.fixed_m_shape(m))
    return generator.normal(size=shape) + 1j * generator.normal(size=shape)


def test_radial_cosine_mask_is_cached_and_has_documented_limits(hamiltonian):
    profile = RadialCosineMask(r0=3.0, n=4.0)
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, mask=profile
    )
    values = boundaries.mask_values()
    expected = profile(hamiltonian.r, 6.0)

    np.testing.assert_allclose(values, expected)
    np.testing.assert_array_equal(values[hamiltonian.r < 3.0], 1.0)
    assert values.flags.writeable is False
    assert boundaries.mask_values() is values
    np.testing.assert_array_equal(
        profile(np.array([0.0, 3.0, 6.0]), 6.0),
        [1.0, 1.0, 0.0],
    )
    assert boundaries.metadata["mask"]["outer_value"] == 0.0
    json.dumps(boundaries.metadata)


def test_cubic_cap_has_correct_onset_sign_and_strength(hamiltonian):
    weak = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, cap=CubicCAP(alpha=0.5, eta=0.25)
    )
    strong = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, cap=CubicCAP(alpha=0.5, eta=0.5)
    )
    values = weak.cap_values()

    np.testing.assert_allclose(
        values, CubicCAP(alpha=0.5, eta=0.25)(hamiltonian.r, 6.0)
    )
    np.testing.assert_array_equal(values[hamiltonian.r <= 3.0], 0.0)
    assert np.all(values.imag <= 0.0)
    assert np.any(values.imag < 0.0)
    np.testing.assert_allclose(strong.cap_values(), 2.0 * values)
    assert values.flags.writeable is False
    assert weak.metadata["cap"]["r_onset"] == pytest.approx(3.0)
    json.dumps(weak.metadata)


def test_actions_support_fixed_m_batches_without_modifying_input(hamiltonian):
    state = _random_state(hamiltonian, m=-1, batches=(2, 3))
    original = state.copy()
    mask = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, mask=RadialCosineMask(r0=3.0)
    )
    cap = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, cap=CubicCAP(alpha=0.5, eta=0.25)
    )

    masked = mask.apply_mask(state, time=0.25, m=-1)
    cap_action = cap.cap_action(state, m=-1)
    radial_shape = (1, 1, hamiltonian.r.size, 1)

    np.testing.assert_allclose(
        masked, state * mask.mask_values().reshape(radial_shape)
    )
    np.testing.assert_allclose(
        cap_action, state * cap.cap_values().reshape(radial_shape)
    )
    np.testing.assert_array_equal(state, original)


def test_absorber_free_actions_are_identity_and_zero(hamiltonian):
    state = _random_state(hamiltonian)
    original = state.copy()
    boundaries = SphericalOneRadialAbsorbingBoundaries(hamiltonian)

    masked = boundaries.apply_mask(state, time=0.5)
    cap_action = boundaries.cap_action(state)

    np.testing.assert_array_equal(masked, state)
    np.testing.assert_array_equal(cap_action, np.zeros_like(state))
    np.testing.assert_array_equal(state, original)
    assert masked is not state
    assert boundaries.metadata["mask"] is None
    assert boundaries.metadata["cap"] is None


def test_absorber_configuration_and_integrator_policy(hamiltonian):
    with pytest.raises(ValueError, match="cannot both"):
        SphericalOneRadialAbsorbingBoundaries(
            hamiltonian,
            mask=RadialCosineMask(r0=3.0),
            cap=CubicCAP(alpha=0.5),
        )
    with pytest.raises(TypeError, match="RadialCosineMask"):
        SphericalOneRadialAbsorbingBoundaries(
            hamiltonian, mask=lambda radial_grid, r_max: radial_grid / r_max
        )

    cap = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, cap=CubicCAP(alpha=0.5)
    )
    cap.validate_integrator("crank-nicolson")
    cap.validate_integrator("sia")
    cap.validate_integrator("cf4-sia")
    for integrator in ("sil", "cf4-sil"):
        with pytest.raises(ValueError, match="requires integrator"):
            cap.validate_integrator(integrator)

    mask = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, mask=RadialCosineMask(r0=3.0)
    )
    for integrator in ("sia", "sil", "crank-nicolson"):
        mask.validate_integrator(integrator)
