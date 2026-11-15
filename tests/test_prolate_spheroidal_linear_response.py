import numpy as np
import pytest

from grid_tdse.prolate_spheroidal import (
    AreaNormalizedSineSquareImpulse,
    AxialDeltaKick,
    OneElectronDiatomicSystem,
    ProlateSpheroidalAbsorbingBoundaries,
    ProlateSpheroidalCubicCAP,
    ProlateSpheroidalDynamicsObservables,
    ProlateSpheroidalFixedMTimeDependentHamiltonian,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
    lorentzian_broadened_positive_frequency_spectrum,
    paired_kick_linear_response,
    windowed_positive_frequency_spectrum,
)
from grid_tdse.real_time import RealTimePropagator, load_real_time_checkpoint


@pytest.fixture(scope="module")
def stationary_problem():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 4.0],
        xi_points_per_element=6,
        eta_degree=7,
        m_max=0,
    )
    states = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=4,
        solver="direct",
    )
    return hamiltonian, states


def _sia(final_time, time_step, *, initial_time=0.0):
    return RealTimePropagator(
        initial_time=initial_time,
        final_time=final_time,
        time_step=time_step,
        integrator="sia",
        max_krylov_dimension=30,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        max_krylov_refinements=6,
        progress_interval=None,
    )


def _distance(hamiltonian, left, right):
    difference = left - right
    return np.sqrt(
        hamiltonian.inner_product(difference, difference).real
    )


def test_analytic_delta_kick_is_exactly_local_and_metric_unitary(
    stationary_problem,
):
    hamiltonian, states = stationary_problem
    initial = states.orbital
    kick = AxialDeltaKick(1.0e-3)

    positive = kick.apply(hamiltonian, initial)
    negative = AxialDeltaKick(-1.0e-3).apply(hamiltonian, initial)
    expected = initial * np.exp(-1.0e-3j * hamiltonian.fixed_m_z())

    np.testing.assert_allclose(positive, expected)
    assert hamiltonian.inner_product(
        positive, positive
    ).real == pytest.approx(1.0, abs=2.0e-15)
    np.testing.assert_allclose(
        hamiltonian.invert(positive), negative, atol=2.0e-14
    )
    assert kick.metadata["integrated_field"] == 1.0e-3


def test_area_normalized_sine_square_impulse_has_requested_area():
    pulse = AreaNormalizedSineSquareImpulse(
        integrated_field=0.125,
        duration=0.2,
        initial_time=0.3,
    )
    times = np.linspace(0.3, 0.5, 20001)
    fields = np.asarray([pulse(time) for time in times])

    assert np.trapezoid(fields, times) == pytest.approx(
        0.125, abs=2.0e-12
    )
    assert pulse(0.2) == 0.0
    assert pulse(0.6) == 0.0
    assert pulse.gauge == "length"


def test_finite_impulse_converges_toward_analytic_kick(stationary_problem):
    hamiltonian, states = stationary_problem
    initial = states.orbital
    integrated_field = 0.01
    errors = []

    for duration in (0.08, 0.04):
        propagator = _sia(duration, duration / 16.0)
        field_free = ProlateSpheroidalFixedMTimeDependentHamiltonian(
            hamiltonian
        )
        finite = ProlateSpheroidalFixedMTimeDependentHamiltonian(
            hamiltonian,
            fields=AreaNormalizedSineSquareImpulse(
                integrated_field=integrated_field,
                duration=duration,
            ),
        )
        exact_initial = AxialDeltaKick(integrated_field).apply(
            hamiltonian, initial
        )
        exact = field_free.propagate(propagator, exact_initial).state
        resolved = finite.propagate(propagator, initial).state
        errors.append(_distance(hamiltonian, resolved, exact))

    assert errors[1] < 0.55 * errors[0]


def test_dynamics_observables_sample_dipoles_populations_and_cap(
    stationary_problem,
):
    hamiltonian, states = stationary_problem
    boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.55, eta=0.1),
    )
    observables = ProlateSpheroidalDynamicsObservables(
        hamiltonian,
        absorbing_boundaries=boundaries,
        reference_states={
            "ground": states.orbitals[0],
            "first_excited": states.orbitals[1],
        },
    )
    state = AxialDeltaKick(0.01).apply(hamiltonian, states.orbital)
    numerator = observables.electronic_dipole_numerator(state)

    assert observables.electronic_dipole(state) == pytest.approx(
        numerator / observables.norm(state)
    )
    assert observables.total_dipole(state) == pytest.approx(
        observables.electronic_dipole(state)
    )
    populations = observables.state_populations(state)
    assert populations.shape == (2,)
    assert populations[0] > 0.99
    assert observables.absorber_region_probability(state) >= 0.0
    assert observables.cap_norm_loss_rate(state) >= 0.0
    assert observables.reference_labels == ("ground", "first_excited")
    assert set(observables.samplers) == {
        "electronic_dipole_numerator",
        "electronic_dipole",
        "total_dipole_numerator",
        "total_dipole",
        "state_populations",
        "absorber_region_probability",
        "cap_norm_loss_rate",
    }


def test_observable_samples_and_metadata_survive_checkpoint(
    stationary_problem, tmp_path
):
    hamiltonian, states = stationary_problem
    observables = ProlateSpheroidalDynamicsObservables(
        hamiltonian,
        reference_states={"ground": states.orbital},
    )
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(hamiltonian)
    checkpoint_path = tmp_path / "delta-kick.chkpt"
    kick = AxialDeltaKick(1.0e-3)

    result = action.propagate(
        _sia(0.02, 0.01),
        kick.apply(hamiltonian, states.orbital),
        samplers=observables.samplers,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        compatibility_metadata=action.archive_metadata(),
        dynamics_metadata={"delta_kick": kick.metadata},
        checkpoint_metadata={"observables": observables.metadata},
    )
    stored = load_real_time_checkpoint(checkpoint_path)

    segment = stored.segments[-1]
    assert set(segment.samples) == set(result.samples)
    for name, values in result.samples.items():
        np.testing.assert_allclose(
            segment.samples[name], values
        )
    assert segment.dynamics["delta_kick"] == kick.metadata
    assert stored.metadata["observables"] == observables.metadata


def test_paired_response_and_windowed_spectrum_recover_known_frequencies():
    duration = 100.0
    time_step = 0.1
    times = np.linspace(0.0, duration, 1001)
    fundamental = 2.0 * np.pi / duration
    expected = np.asarray([7.0, 13.0]) * fundamental
    response = np.sin(expected[0] * times) + 0.4 * np.sin(
        expected[1] * times
    )
    kick_strength = 1.0e-3
    positive = kick_strength * response
    negative = -kick_strength * response

    reconstructed = paired_kick_linear_response(
        positive, negative, kick_strength
    )
    spectrum = windowed_positive_frequency_spectrum(
        times, reconstructed, window="hann"
    )
    observed = []
    for frequency in expected:
        index = int(
            np.argmin(abs(spectrum.angular_frequencies - frequency))
        )
        assert spectrum.power[index] > spectrum.power[index - 1]
        assert spectrum.power[index] > spectrum.power[index + 1]
        observed.append(spectrum.angular_frequencies[index])

    np.testing.assert_allclose(reconstructed, response)
    np.testing.assert_allclose(observed, expected, atol=1.0e-14)
    assert spectrum.angular_frequencies[0] == pytest.approx(fundamental)
    assert spectrum.metadata["duration"] == pytest.approx(duration)


def test_lorentzian_broadening_recovers_line_and_records_fwhm():
    times = np.linspace(0.0, 400.0, 4001)
    frequency = 0.5
    fwhm = 0.05
    signal = np.sin(frequency * times)

    spectrum = lorentzian_broadened_positive_frequency_spectrum(
        times, signal, fwhm=fwhm
    )
    strength = -spectrum.transform.imag
    peak = spectrum.angular_frequencies[np.argmax(strength)]

    assert peak == pytest.approx(
        frequency, abs=2.0 * np.pi / (times[-1] - times[0])
    )
    assert spectrum.window == "lorentzian_fwhm_0.05"


def test_linear_response_validation_errors(stationary_problem):
    hamiltonian, states = stationary_problem

    with pytest.raises(ValueError, match="nonzero"):
        paired_kick_linear_response([0.0, 1.0], [0.0, -1.0], 0.0)
    with pytest.raises(ValueError, match="uniformly"):
        windowed_positive_frequency_spectrum(
            [0.0, 0.1, 0.25], [0.0, 1.0, 0.0]
        )
    with pytest.raises(ValueError, match="window"):
        windowed_positive_frequency_spectrum(
            [0.0, 0.1, 0.2], [0.0, 1.0, 0.0], window="blackman"
        )
    with pytest.raises(ValueError, match="different representation"):
        other = ProlateSpheroidalOneElectronHamiltonian(
            hamiltonian.system,
            xi_boundaries=[1.0, 2.0, 4.0],
            xi_points_per_element=6,
            eta_degree=7,
            m_max=0,
        )
        boundaries = ProlateSpheroidalAbsorbingBoundaries(
            other.representation,
            cap=ProlateSpheroidalCubicCAP(alpha=0.5),
        )
        ProlateSpheroidalDynamicsObservables(
            hamiltonian, absorbing_boundaries=boundaries
        )
    assert AxialDeltaKick(0.0).apply(
        hamiltonian, states.orbital
    ).shape == states.orbital.shape
