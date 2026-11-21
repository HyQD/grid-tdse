import json

import numpy as np
import pytest
from grid_lib.spherical_coordinates import (
    CubicCAP,
    RadialCosineMask,
    SphericalHarmonicRotation,
)

from grid_tdse.real_time import RealTimePropagator
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    SphericalOneRadialAbsorbingBoundaries,
    SphericalOneRadialAxialLengthGaugeInteraction,
    SphericalOneRadialAxialVelocityGaugeInteraction,
    SphericalOneRadialFullAngularAxialLengthGaugeInteraction,
    SphericalOneRadialFullAngularAxialVelocityGaugeInteraction,
    SphericalOneRadialFullAngularDynamicsObservables,
    SphericalOneRadialFullAngularTimeDependentHamiltonian,
    diagonalize_full_angular,
    transform_axial_gauge_state,
    transform_full_angular_axial_gauge_state,
)


class _VectorPotential:
    gauge = "velocity"

    def __call__(self, time):
        return 0.2 + 0.3 * time


def _rotation_y(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )


def _system(angle=0.0, *, charge1=1.0, charge2=1.0, shift=None):
    positions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    positions = positions @ _rotation_y(angle).T
    if shift is not None:
        positions += np.asarray(shift)
    return OneElectronDiatomicSystem.from_nuclear_positions(
        charge1,
        charge2,
        positions[0],
        positions[1],
        name="test",
    )


def _hamiltonian(
    angle=0.0,
    *,
    charge1=1.0,
    charge2=1.0,
    shift=None,
    l_max=2,
):
    return OneElectronDiatomicHamiltonian(
        _system(
            angle,
            charge1=charge1,
            charge2=charge2,
            shift=shift,
        ),
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[5, 5, 5],
        l_max=l_max,
        m_max=l_max,
    )


def _random_state(hamiltonian, *, seed=41, batches=()):
    generator = np.random.default_rng(seed)
    shape = (*batches, *hamiltonian.orbital_shape)
    values = generator.normal(size=shape)
    return values + 1j * generator.normal(size=shape)


def _propagator(integrator, *, final_time=0.02, time_step=0.01):
    return RealTimePropagator(
        final_time=final_time,
        time_step=time_step,
        integrator=integrator,
        max_krylov_dimension=30,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        max_krylov_refinements=6,
        progress_interval=None,
    )


@pytest.mark.parametrize("m", [-1, 0, 1])
def test_full_interactions_reproduce_fixed_m_blocks(m):
    hamiltonian = _hamiltonian(l_max=2)
    fixed_state = _random_state(hamiltonian, seed=43 + m)[
        :, : hamiltonian.fixed_m_shape(m)[1]
    ]
    indices = [
        index
        for index, (_, magnetic) in enumerate(hamiltonian.orbital_lm)
        if magnetic == m
    ]
    full_state = np.zeros(hamiltonian.orbital_shape, dtype=complex)
    full_state[:, indices] = fixed_state

    fixed_length = SphericalOneRadialAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 0.4,
    )
    full_length = SphericalOneRadialFullAngularAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 0.4,
    )
    fixed_velocity = SphericalOneRadialAxialVelocityGaugeInteraction(
        hamiltonian,
        fields=_VectorPotential(),
    )
    full_velocity = (
        SphericalOneRadialFullAngularAxialVelocityGaugeInteraction(
            hamiltonian,
            fields=_VectorPotential(),
        )
    )

    for full, fixed in (
        (full_length(full_state, 0.3), fixed_length(fixed_state, 0.3, m)),
        (
            full_velocity(full_state, 0.3),
            fixed_velocity(fixed_state, 0.3, m),
        ),
    ):
        np.testing.assert_allclose(full[:, indices], fixed)
        other = np.delete(full, indices, axis=1)
        np.testing.assert_array_equal(other, 0.0)


def test_zero_field_and_heteronuclear_nuclear_scalar():
    hamiltonian = _hamiltonian(
        np.pi / 4.0,
        charge1=1.0,
        charge2=2.0,
    )
    state = _random_state(hamiltonian)
    origin = np.array([0.1, -0.2, 0.3])
    electronic = SphericalOneRadialFullAngularAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 0.0,
        origin=origin,
    )
    complete = SphericalOneRadialFullAngularAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 0.4,
        origin=origin,
        include_nuclear_scalar=True,
    )
    omitted = SphericalOneRadialFullAngularAxialLengthGaugeInteraction(
        hamiltonian,
        fields=lambda time: 0.4,
        origin=origin,
    )

    np.testing.assert_array_equal(electronic(state, 0.2), 0.0)
    nuclear_z = hamiltonian.system.nuclear_dipole_about(origin)[2]
    np.testing.assert_allclose(
        complete(state, 0.2) - omitted(state, 0.2),
        -0.4 * nuclear_z * state,
    )
    assert complete.archive_metadata()["field_direction"] == "laboratory_z"
    json.dumps(complete.archive_metadata())


def test_full_gauge_transform_is_independent_in_every_m_block():
    hamiltonian = _hamiltonian(l_max=3)
    state = _random_state(hamiltonian)
    transformed = transform_full_angular_axial_gauge_state(
        hamiltonian,
        state,
        0.17,
        source_gauge="velocity",
        target_gauge="length",
    )

    expected = np.zeros_like(transformed)
    for m in range(-hamiltonian.m_max, hamiltonian.m_max + 1):
        indices = [
            index
            for index, (_, magnetic) in enumerate(hamiltonian.orbital_lm)
            if magnetic == m
        ]
        expected[:, indices] = transform_axial_gauge_state(
            hamiltonian,
            state[:, indices],
            0.17,
            m=m,
            source_gauge="velocity",
            target_gauge="length",
        )

    np.testing.assert_allclose(transformed, expected)
    unchanged = transform_full_angular_axial_gauge_state(
        hamiltonian,
        state,
        0.0,
        source_gauge="velocity",
        target_gauge="length",
    )
    np.testing.assert_array_equal(unchanged, state)
    assert unchanged is not state


def test_full_absorbers_broadcast_without_modifying_input():
    hamiltonian = _hamiltonian(np.pi / 4.0)
    state = _random_state(hamiltonian, batches=(2, 3))
    original = state.copy()
    mask = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian,
        mask=RadialCosineMask(r0=3.0),
    )
    cap = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian,
        cap=CubicCAP(alpha=0.5, eta=0.25),
    )
    radial_shape = (1, 1, hamiltonian.r.size, 1)

    np.testing.assert_allclose(
        mask.apply_mask_full(state, 0.2),
        state * mask.mask_values().reshape(radial_shape),
    )
    np.testing.assert_allclose(
        cap.cap_action_full(state),
        state * cap.cap_values().reshape(radial_shape),
    )
    np.testing.assert_array_equal(state, original)


def test_full_field_free_eigenstate_retains_phase_and_norm():
    hamiltonian = _hamiltonian(np.pi / 4.0)
    states = diagonalize_full_angular(
        hamiltonian,
        n_states=1,
        solver="direct",
    )
    dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
        hamiltonian
    )
    final_time = 0.02
    result = dynamics.propagate(
        _propagator("sia", final_time=final_time),
        states.orbital,
    )
    expected = (
        np.exp(-1j * states.electronic_energy * final_time)
        * states.orbital
    )

    np.testing.assert_allclose(result.state, expected, atol=3.0e-11)
    assert result.norm == pytest.approx(result.initial_norm, abs=3.0e-12)


def test_full_cap_loses_norm_and_sil_is_rejected():
    hamiltonian = _hamiltonian(np.pi / 4.0)
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian,
        cap=CubicCAP(alpha=0.5, eta=0.5),
    )
    dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
        hamiltonian,
        absorbing_boundaries=boundaries,
    )
    state = np.zeros(hamiltonian.orbital_shape, dtype=complex)
    state[-2:, 0] = 1.0
    state /= np.sqrt(hamiltonian.inner_product(state, state).real)

    result = dynamics.propagate(_propagator("sia"), state)
    assert result.norm < result.initial_norm
    with pytest.raises(ValueError, match="requires integrator"):
        dynamics.propagate(_propagator("sil"), state)


def test_vector_observables_and_radial_probability_are_raw():
    hamiltonian = _hamiltonian(
        np.pi / 4.0,
        charge1=1.0,
        charge2=2.0,
    )
    origin = np.array([0.1, -0.2, 0.3])
    dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
        hamiltonian,
        origin=origin,
    )
    state = _random_state(hamiltonian)
    state /= np.sqrt(hamiltonian.inner_product(state, state).real)
    observables = SphericalOneRadialFullAngularDynamicsObservables(
        dynamics,
        state,
        radial_probability_interval=2,
    )

    norm = observables.norm(state)
    expected_nuclear = hamiltonian.system.nuclear_dipole_about(origin) * norm
    np.testing.assert_allclose(
        observables.nuclear_dipole_numerator(state),
        expected_nuclear,
    )
    np.testing.assert_allclose(
        observables.total_dipole_numerator(state),
        observables.electronic_dipole_numerator(state) + expected_nuclear,
    )
    np.testing.assert_allclose(
        observables.induced_dipole_numerator(state),
        0.0,
        atol=2.0e-13,
    )
    probability = observables.radial_probability(state)
    assert np.sum(probability * hamiltonian.weights) == pytest.approx(norm)

    observables.observe_radial_probability(
        state,
        accepted_step=0,
        time=0.0,
        diagnostics=None,
        final=False,
        checkpoint=False,
    )
    observables.observe_radial_probability(
        state,
        accepted_step=3,
        time=0.3,
        diagnostics=None,
        final=True,
        checkpoint=False,
    )
    series = observables.radial_probability_series
    np.testing.assert_array_equal(series.accepted_steps, [0, 3])
    assert series.samples[0].is_initial
    assert series.samples[-1].is_final
    assert observables.metadata["component_order"] == ["x", "y", "z"]
    json.dumps(observables.metadata)


def test_homonuclear_nuclear_dipole_is_zero_at_midpoint():
    for angle in (0.0, np.pi / 6.0, np.pi / 2.0):
        hamiltonian = _hamiltonian(angle)
        dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
            hamiltonian
        )
        state = _random_state(hamiltonian, seed=55)
        observables = SphericalOneRadialFullAngularDynamicsObservables(
            dynamics,
            state,
        )
        np.testing.assert_allclose(
            observables.nuclear_dipole_numerator(state),
            0.0,
            atol=1.0e-14,
        )


def test_velocity_gauge_mechanical_momentum_changes_only_z():
    hamiltonian = _hamiltonian(np.pi / 4.0)
    field = _VectorPotential()
    dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
        hamiltonian,
        fields=field,
        field_gauge="velocity",
    )
    state = _random_state(hamiltonian)
    observables = SphericalOneRadialFullAngularDynamicsObservables(
        dynamics,
        state,
    )
    time = 0.4
    difference = (
        observables.mechanical_momentum_numerator(state, time)
        - observables.canonical_momentum_numerator(state, time)
    )
    expected = np.array(
        [
            0.0,
            0.0,
            -hamiltonian.system.electron_charge
            * field(time)
            * observables.norm(state),
        ]
    )
    np.testing.assert_allclose(difference, expected)


def test_dipole_and_momentum_vectors_rotate_with_state_and_geometry():
    angle = np.pi / 4.0
    axial = _hamiltonian(0.0)
    tilted = _hamiltonian(angle)
    axial_state = _random_state(axial, seed=63)
    rotation_matrix = _rotation_y(angle)
    rotation = SphericalHarmonicRotation(rotation_matrix, l_max=2)
    tilted_state = rotation(axial_state)
    axial_observables = SphericalOneRadialFullAngularDynamicsObservables(
        SphericalOneRadialFullAngularTimeDependentHamiltonian(axial),
        axial_state,
    )
    tilted_observables = SphericalOneRadialFullAngularDynamicsObservables(
        SphericalOneRadialFullAngularTimeDependentHamiltonian(tilted),
        tilted_state,
    )

    np.testing.assert_allclose(
        tilted_observables.electronic_dipole_numerator(tilted_state),
        rotation_matrix
        @ axial_observables.electronic_dipole_numerator(axial_state),
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        tilted_observables.canonical_momentum_numerator(tilted_state),
        rotation_matrix
        @ axial_observables.canonical_momentum_numerator(axial_state),
        atol=3.0e-12,
        rtol=3.0e-12,
    )
