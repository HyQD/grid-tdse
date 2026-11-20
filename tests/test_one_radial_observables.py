import json

import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.real_time import RealTimePropagator
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    SphericalOneRadialAbsorbingBoundaries,
    SphericalOneRadialDynamicsObservables,
    SphericalOneRadialFixedMTimeDependentHamiltonian,
    diagonalize_fixed_m,
)


class _VectorPotential:
    gauge = "velocity"

    def __call__(self, time):
        return 0.2 + 0.3 * time


@pytest.fixture(scope="module")
def stationary_problem():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[6, 6, 6],
        l_max=3,
        m_max=1,
    )
    eigenstates = diagonalize_fixed_m(
        hamiltonian, m=0, n_states=2
    )
    return hamiltonian, eigenstates


def _sia(final_time=0.05, time_step=0.01):
    return RealTimePropagator(
        final_time=final_time,
        time_step=time_step,
        integrator="sia",
        max_krylov_dimension=30,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        progress_interval=None,
    )


def test_raw_observables_scale_with_norm_and_exclude_cap(
    stationary_problem,
):
    hamiltonian, states = stationary_problem
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, cap=CubicCAP(alpha=0.5, eta=0.25)
    )
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=boundaries
    )
    observables = SphericalOneRadialDynamicsObservables(
        dynamics,
        states.orbital,
        reference_states={
            "ground": states.orbitals[0],
            "first_excited": states.orbitals[1],
        },
    )
    state = 2.0 * states.orbital

    assert observables.norm(state) == pytest.approx(4.0)
    assert observables.energy_numerator(state) == pytest.approx(
        4.0 * states.electronic_energy
    )
    expected = hamiltonian.inner_product(
        state, dynamics.field_free_action(state)
    ).real
    assert observables.energy_numerator(state) == pytest.approx(expected)
    assert observables.induced_dipole_numerator(state) == pytest.approx(
        0.0, abs=2.0e-14
    )
    np.testing.assert_allclose(
        observables.state_populations(state), [4.0, 0.0], atol=2.0e-13
    )
    assert observables.reference_labels == ("ground", "first_excited")


def test_canonical_and_mechanical_momentum_relation(stationary_problem):
    hamiltonian, states = stationary_problem
    field = _VectorPotential()
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=field,
        field_gauge="velocity",
    )
    state = states.orbitals[0] + 0.1j * states.orbitals[1]
    observables = SphericalOneRadialDynamicsObservables(
        dynamics, state, radial_probability_interval=None
    )
    time = 0.4
    norm = observables.norm(state)
    canonical = observables.canonical_momentum_numerator(state, time)
    mechanical = observables.mechanical_momentum_numerator(state, time)

    assert mechanical - canonical == pytest.approx(
        -hamiltonian.system.electron_charge * field(time) * norm
    )
    assert "mechanical_momentum_numerator" in observables.samplers
    assert observables.state_observer is None
    with pytest.raises(RuntimeError, match="requires velocity gauge"):
        length = SphericalOneRadialDynamicsObservables(
            SphericalOneRadialFixedMTimeDependentHamiltonian(hamiltonian),
            states.orbital,
        )
        length.mechanical_momentum_numerator(states.orbital, time)


def test_field_free_ground_state_has_zero_canonical_momentum(
    stationary_problem,
):
    hamiltonian, states = stationary_problem
    observables = SphericalOneRadialDynamicsObservables(
        SphericalOneRadialFixedMTimeDependentHamiltonian(hamiltonian),
        states.orbital,
    )

    assert observables.canonical_momentum_numerator(
        states.orbital
    ) == pytest.approx(0.0, abs=2.0e-13)


def test_scalar_and_radial_sampling_have_independent_cadences(
    stationary_problem,
):
    hamiltonian, states = stationary_problem
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(hamiltonian)
    observables = SphericalOneRadialDynamicsObservables(
        dynamics,
        states.orbital,
        radial_probability_interval=2,
    )

    result = dynamics.propagate(
        _sia(final_time=0.05, time_step=0.01),
        states.orbital,
        **observables.propagation_kwargs,
    )
    series = observables.radial_probability_series

    assert result.samples["norm"].shape == (6,)
    np.testing.assert_array_equal(series.accepted_steps, [0, 2, 4, 5])
    np.testing.assert_allclose(series.times, [0.0, 0.02, 0.04, 0.05])
    assert series.probabilities.shape == (4, hamiltonian.r.size)
    for sample in series.samples:
        integral = np.sum(sample.probability * hamiltonian.weights)
        assert integral == pytest.approx(sample.norm, abs=2.0e-13)
    assert series.samples[0].is_initial
    assert series.samples[-1].is_final


def test_radial_samples_are_recorded_after_mask_application(
    stationary_problem,
):
    hamiltonian, _ = stationary_problem
    state = np.zeros(hamiltonian.fixed_m_shape(0), dtype=complex)
    state[-2:, :] = 1.0
    state /= np.sqrt(hamiltonian.inner_product(state, state).real)
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian, mask=RadialCosineMask(r0=3.0)
    )
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=boundaries
    )
    observables = SphericalOneRadialDynamicsObservables(
        dynamics, state, radial_probability_interval=2
    )

    result = dynamics.propagate(
        _sia(final_time=0.03, time_step=0.01),
        state,
        **observables.propagation_kwargs,
    )
    series = observables.radial_probability_series

    np.testing.assert_array_equal(series.accepted_steps, [0, 2, 3])
    assert series.norms[-1] == pytest.approx(result.norm)
    assert series.norms[-1] < series.norms[0]


def test_observable_metadata_and_validation(stationary_problem):
    hamiltonian, states = stationary_problem
    dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(hamiltonian)
    observables = SphericalOneRadialDynamicsObservables(
        dynamics, states.orbital, radial_probability_interval=None
    )

    assert observables.radial_probability_series.samples == ()
    assert observables.metadata["radial_probability_interval"] is None
    assert observables.metadata["cap_excluded_from_energy"] is True
    json.dumps(observables.metadata)

    with pytest.raises(ValueError, match="positive"):
        SphericalOneRadialDynamicsObservables(
            dynamics, states.orbital, radial_probability_interval=0
        )
    with pytest.raises(ValueError, match="exactly one"):
        SphericalOneRadialDynamicsObservables(
            dynamics, np.stack([states.orbital, states.orbital])
        )
