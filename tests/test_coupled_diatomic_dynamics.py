from dataclasses import asdict

import numpy as np
import pytest

from grid_tdse.real_time import (
    load_real_time_checkpoint,
    load_real_time_snapshot,
)
from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    CoupledAngularBasisSpec,
    CoupledAngularTransform,
    CoupledNonBornOppenheimerDiatomicIonTDSE,
    DiatomicIonDynamicsSamplingPolicy,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    TwoRadialSystemSpec,
    load_coupled_two_radial_state,
    radial_potential,
    save_coupled_two_radial_state,
)
from grid_tdse.spherical.two_radial.diatomic import _diatomic_parameters


GRID = FEMDVRGridSpec([0.0, 1.0, 2.0], [3, 3])


class _Pulse:
    gauge = "length"

    def __init__(self, amplitude=0.02, duration=0.006):
        self.amplitude = amplitude
        self.duration = duration

    def __call__(self, time):
        if time < 0.0 or time > self.duration:
            return 0.0
        return self.amplitude * np.sin(np.pi * time / self.duration) ** 2


class _VelocityPulse(_Pulse):
    gauge = "velocity"


class _NonzeroInitialVelocityPulse(_VelocityPulse):
    def __call__(self, time):
        del time
        return self.amplitude


def _representation_pair(system_name):
    parameters = _diatomic_parameters(system_name, "2022")
    exchange = 1 if parameters.has_identical_pair else None
    specification = TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            GRID,
            AngularBasisSpec(1),
            -0.5 / parameters.electronic_reduced_mass,
        ),
        coordinate2=CoordinateSpec(
            GRID,
            AngularBasisSpec(1, coordinate_parity=exchange),
            -0.5 / parameters.internuclear_reduced_mass,
            one_body_terms=(
                radial_potential(
                    lambda radius: parameters.reference_charge
                    * parameters.partner_charge
                    / radius
                ),
            ),
        ),
        multipoles=CoulombMultipoleSpec(
            2,
            2,
            lambda_parity=(1 if parameters.has_identical_pair else None),
        ),
        coulomb_terms=(
            CoulombTerm(
                parameters.reference_center_scale,
                parameters.electron_charge * parameters.reference_charge,
            ),
            CoulombTerm(
                parameters.partner_center_scale,
                parameters.electron_charge * parameters.partner_charge,
            ),
        ),
        M_tot=0,
        radial_coulomb_method="analytic",
    )
    basis = CoupledAngularBasisSpec.for_diatomic_ion(
        1,
        1,
        2,
        has_identical_pair=parameters.has_identical_pair,
        M_tot=0,
        pair_exchange_parity=exchange,
    )
    coupled = specification.build_coupled(basis)
    uncoupled = specification.build(radial_operator_mode="element-local")
    transform = CoupledAngularTransform(basis)
    expected = tuple(
        (
            *uncoupled.basis1.states[channel1],
            *uncoupled.basis2.states[channel2],
        )
        for channel1, channel2 in uncoupled.angular_pairs.pairs
    )
    assert expected == transform.uncoupled_states
    return parameters, exchange, coupled, uncoupled, transform


def _initial_state(system_name, seed=772):
    _, _, representation, _, transform = _representation_pair(system_name)
    state = representation.random_state(
        np.random.default_rng(seed),
        normalize=True,
    )
    return state, transform.to_uncoupled(state), transform


def _calculation_pair(
    system_name,
    *,
    integrator="sia",
    final_time=0.006,
    time_step=0.002,
    cap=None,
    krylov_rtol=2.0e-12,
    field_gauge="length",
):
    coupled_state, uncoupled_state, transform = _initial_state(system_name)
    parameters = _diatomic_parameters(system_name, "2022")
    exchange = 1 if parameters.has_identical_pair else None
    common = dict(
        electronic_grid=GRID,
        internuclear_grid=GRID,
        pair_exchange_parity=exchange,
        fields=(
            (_VelocityPulse if field_gauge == "velocity" else _Pulse)(
                duration=final_time
            ),
        ),
        field_gauge=field_gauge,
        M_tot=0,
        lambda_max=2,
        codata_set="2022",
        radial_coulomb_method="analytic",
        integrator=integrator,
        integrator_options={
            "max_krylov_dimension": 20,
            "rtol": krylov_rtol,
            "atol": 1.0e-14,
            "max_refinements": 8,
        },
        final_time=final_time,
        time_step=time_step,
        cap_r=cap,
        cap_R=cap,
        observables=None,
        dynamics_sampling_policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
        progress_interval=None,
        radial_operator_mode="element-local",
    )
    coupled = NonBornOppenheimerDiatomicIonTDSE(
        system_name,
        wf_representation="coupled",
        initial_state=coupled_state,
        l1_max=1,
        l2_max=1,
        L_max=2,
        **common,
    )
    uncoupled = NonBornOppenheimerDiatomicIonTDSE(
        system_name,
        initial_state=uncoupled_state,
        l_max=1,
        m_max=1,
        mu_max=2,
        **common,
    )
    return coupled, uncoupled, transform


def _phase_aligned_error(reference, candidate, inner_product):
    overlap = inner_product(reference, candidate)
    if overlap:
        candidate = candidate * np.exp(-1j * np.angle(overlap))
    difference = reference - candidate
    return np.sqrt(max(inner_product(difference, difference).real, 0.0))


def _assert_snapshot_agreement(coupled, uncoupled, *, atol=2.0e-10):
    for name, coupled_value in asdict(coupled.numerators).items():
        uncoupled_value = asdict(uncoupled.numerators)[name]
        if coupled_value is None:
            assert uncoupled_value is None
        else:
            np.testing.assert_allclose(coupled_value, uncoupled_value, atol=atol)
    np.testing.assert_allclose(
        coupled.angular_populations.L,
        uncoupled.angular_populations.L,
        atol=atol,
    )
    np.testing.assert_allclose(
        coupled.angular_populations.l1,
        uncoupled.angular_populations.l1,
        atol=atol,
    )
    np.testing.assert_allclose(
        coupled.angular_populations.l2,
        uncoupled.angular_populations.l2,
        atol=atol,
    )


@pytest.mark.parametrize("system_name", ["H2+", "HD+", "D2+"])
def test_sia_pulse_matches_complete_uncoupled_reference(system_name):
    coupled, uncoupled, transform = _calculation_pair(system_name)
    coupled_result = coupled.run(progress=False)
    uncoupled_result = uncoupled.run(progress=False)

    transformed = transform.to_uncoupled(coupled_result.state)
    assert _phase_aligned_error(
        uncoupled_result.state,
        transformed,
        uncoupled.representation.inner_product,
    ) < 2.0e-10
    assert isinstance(coupled, CoupledNonBornOppenheimerDiatomicIonTDSE)
    assert coupled.wf_representation == "coupled"
    for coupled_sample, uncoupled_sample in zip(
        coupled.dynamics_time_series.samples,
        uncoupled.dynamics_time_series.samples,
        strict=True,
    ):
        _assert_snapshot_agreement(
            coupled_sample.snapshot,
            uncoupled_sample.snapshot,
        )


@pytest.mark.parametrize("system_name", ["H2+", "HD+", "D2+"])
def test_velocity_gauge_sia_matches_complete_uncoupled_reference(system_name):
    coupled, uncoupled, transform = _calculation_pair(
        system_name,
        field_gauge="velocity",
    )
    coupled_result = coupled.run(progress=False)
    uncoupled_result = uncoupled.run(progress=False)

    transformed = transform.to_uncoupled(coupled_result.state)
    assert _phase_aligned_error(
        uncoupled_result.state,
        transformed,
        uncoupled.representation.inner_product,
    ) < 3.0e-10
    for coupled_sample, uncoupled_sample in zip(
        coupled.dynamics_time_series.samples,
        uncoupled.dynamics_time_series.samples,
        strict=True,
    ):
        assert coupled_sample.snapshot.gauge == "velocity"
        assert coupled_sample.snapshot.field_symbol == "A"
        assert coupled_sample.snapshot.field_amplitude == pytest.approx(
            uncoupled_sample.snapshot.field_amplitude,
            abs=1.0e-15,
        )
        assert coupled_sample.snapshot.energy_numerator == pytest.approx(
            uncoupled_sample.snapshot.energy_numerator,
            abs=3.0e-10,
        )
        _assert_snapshot_agreement(
            coupled_sample.snapshot,
            uncoupled_sample.snapshot,
            atol=3.0e-10,
        )


def test_cf4_sia_and_cap_match_uncoupled_reference():
    def cap(radius, outer):
        return -0.05j * (radius / outer) ** 4

    coupled, uncoupled, transform = _calculation_pair(
        "HD+",
        integrator="cf4-sia",
        cap=cap,
    )
    coupled_result = coupled.run(progress=False)
    uncoupled_result = uncoupled.run(progress=False)

    transformed = transform.to_uncoupled(coupled_result.state)
    assert _phase_aligned_error(
        uncoupled_result.state,
        transformed,
        uncoupled.representation.inner_product,
    ) < 3.0e-10
    final_coupled = coupled.dynamics_time_series.samples[-1]
    final_uncoupled = uncoupled.dynamics_time_series.samples[-1]
    assert final_coupled.cap_norm_loss > 0.0
    assert final_coupled.cap_norm_loss == pytest.approx(
        final_uncoupled.cap_norm_loss,
        abs=2.0e-11,
    )
    _assert_snapshot_agreement(
        final_coupled.snapshot,
        final_uncoupled.snapshot,
        atol=3.0e-10,
    )


def test_sia_time_step_and_krylov_tolerance_convergence():
    calculations = {}
    for name, time_step in (
        ("coarse", 0.003),
        ("fine", 0.0015),
        ("reference", 0.00075),
    ):
        calculation, _, _ = _calculation_pair(
            "HD+",
            time_step=time_step,
        )
        calculations[name] = (calculation, calculation.run(progress=False))
    reference, reference_result = calculations["reference"]
    coarse_error = _phase_aligned_error(
        reference_result.state,
        calculations["coarse"][1].state,
        reference.representation.inner_product,
    )
    fine_error = _phase_aligned_error(
        reference_result.state,
        calculations["fine"][1].state,
        reference.representation.inner_product,
    )
    assert fine_error < 0.25 * coarse_error

    tolerance_runs = {}
    for name, tolerance in (
        ("loose", 1.0e-7),
        ("tight", 1.0e-11),
        ("reference", 1.0e-13),
    ):
        calculation, _, _ = _calculation_pair(
            "HD+",
            time_step=0.003,
            krylov_rtol=tolerance,
        )
        tolerance_runs[name] = (calculation, calculation.run(progress=False))
    reference, reference_result = tolerance_runs["reference"]
    loose_error = _phase_aligned_error(
        reference_result.state,
        tolerance_runs["loose"][1].state,
        reference.representation.inner_product,
    )
    tight_error = _phase_aligned_error(
        reference_result.state,
        tolerance_runs["tight"][1].state,
        reference.representation.inner_product,
    )
    assert tight_error < loose_error


def test_symmetry_restricted_archive_embedding_is_norm_preserving(tmp_path):
    parameters, exchange, _, _, _ = _representation_pair("H2+")
    restricted_specification = TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            GRID,
            AngularBasisSpec(1),
            -0.5 / parameters.electronic_reduced_mass,
        ),
        coordinate2=CoordinateSpec(
            GRID,
            AngularBasisSpec(1),
            -0.5 / parameters.internuclear_reduced_mass,
            one_body_terms=(
                radial_potential(lambda radius: 1.0 / radius),
            ),
        ),
        multipoles=CoulombMultipoleSpec(2, 2, lambda_parity=1),
        coulomb_terms=(
            CoulombTerm(
                parameters.reference_center_scale,
                parameters.electron_charge * parameters.reference_charge,
            ),
            CoulombTerm(
                parameters.partner_center_scale,
                parameters.electron_charge * parameters.partner_charge,
            ),
        ),
        M_tot=0,
        radial_coulomb_method="analytic",
    )
    basis = CoupledAngularBasisSpec.for_diatomic_ion(
        1,
        1,
        0,
        has_identical_pair=True,
        M_tot=0,
        inversion_parity=1,
        pair_exchange_parity=exchange,
    )
    restricted = restricted_specification.build_coupled(basis)
    state = restricted.random_state(np.random.default_rng(10), normalize=True)
    path = tmp_path / "restricted.npz"
    save_coupled_two_radial_state(
        path,
        state,
        restricted,
        physical_state={
            "system_family": "non_bo_diatomic_ion",
            "system": "H2+",
            "parameters": asdict(parameters),
            "symmetry": {
                "inversion_parity": 1,
                "pair_exchange_parity": 1,
                "M_tot": 0,
            },
        },
        calculation={"converged": True},
    )
    archive = load_coupled_two_radial_state(path)
    calculation = NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        wf_representation="coupled",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=archive,
        pair_exchange_parity=1,
        fields=(_Pulse(),),
        l1_max=1,
        l2_max=1,
        L_max=2,
        lambda_max=2,
        radial_coulomb_method="analytic",
        integrator="sia",
        final_time=0.002,
        time_step=0.002,
        observables=None,
        progress_interval=None,
    )

    assert calculation.representation.norm(calculation.initial_state) == pytest.approx(
        restricted.norm(state),
        abs=2.0e-14,
    )
    for channel, angular_state in enumerate(basis.states):
        target = calculation.representation.angular_basis.index(*angular_state)
        np.testing.assert_array_equal(
            calculation.initial_state[target],
            state[channel],
        )

    initialized = NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        wf_representation="coupled",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        pair_exchange_parity=1,
        initial_state_policy="require",
        ground_state_path=path,
        ground_state_inversion_parity=1,
        ground_state_L_max=0,
        l1_max=1,
        l2_max=1,
        L_max=2,
        M_tot=0,
        lambda_max=2,
        radial_coulomb_method="analytic",
        integrator="sia",
        final_time=0.002,
        time_step=0.002,
        observables=None,
        progress_interval=None,
    )
    assert initialized.initialization.source == "archive"
    np.testing.assert_array_equal(
        initialized.initial_state,
        calculation.initial_state,
    )


def test_checkpoint_snapshot_metadata_restart_and_representation_rejection(tmp_path):
    coupled_state, uncoupled_state, _ = _initial_state("HD+")
    common = dict(
        electronic_grid=GRID,
        internuclear_grid=GRID,
        pair_exchange_parity=None,
        fields=(_Pulse(duration=0.003),),
        field_gauge="length",
        M_tot=0,
        lambda_max=2,
        radial_coulomb_method="analytic",
        integrator="sia",
        integrator_options={"rtol": 2.0e-12, "atol": 1.0e-14},
        final_time=0.003,
        time_step=0.001,
        observables=None,
        dynamics_sampling_policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
        progress_interval=None,
        radial_operator_mode="element-local",
    )
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    checkpoint_path = tmp_path / "rolling.npz"
    uninterrupted = NonBornOppenheimerDiatomicIonTDSE(
        "HD+",
        wf_representation="coupled",
        initial_state=coupled_state,
        l1_max=1,
        l2_max=1,
        L_max=2,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        wavefunction_snapshot_directory=snapshots,
        wavefunction_snapshot_interval=1,
        **common,
    )
    uninterrupted_result = uninterrupted.run(progress=False)
    intermediate = load_real_time_checkpoint(
        uninterrupted.wavefunction_snapshots[1].path
    )
    assert intermediate.current_step == 1
    assert not intermediate.latest_segment.complete

    standalone = load_real_time_snapshot(uninterrupted.wavefunction_snapshots[0].path)
    representation = standalone.compatibility["representation"]
    assert representation["wf_representation"] == "coupled"
    assert representation["angular_basis"]["states"]
    assert representation["conventions"]["spherical_harmonics"] == (
        "Condon-Shortley"
    )
    assert standalone.dynamics["field_gauge"] == "length"
    assert standalone.dynamics["fields"]

    resumed = NonBornOppenheimerDiatomicIonTDSE(
        "HD+",
        wf_representation="coupled",
        initial_state=intermediate.state_copy(),
        l1_max=1,
        l2_max=1,
        L_max=2,
        initial_time=0.001,
        **common,
    )
    resumed_result = resumed.resume(intermediate, progress=False)
    np.testing.assert_allclose(
        resumed_result.state,
        uninterrupted_result.state,
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    assert [
        sample.accepted_step for sample in resumed.dynamics_time_series.samples
    ] == [1, 2, 3]
    reference_samples = uninterrupted.dynamics_time_series.samples[1:]
    for actual, reference in zip(
        resumed.dynamics_time_series.samples,
        reference_samples,
        strict=True,
    ):
        assert actual.time == pytest.approx(reference.time, abs=1.0e-15)
        _assert_snapshot_agreement(
            actual.snapshot,
            reference.snapshot,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            actual.snapshot.radial_densities.P_r,
            reference.snapshot.radial_densities.P_r,
            atol=2.0e-13,
            rtol=2.0e-13,
        )
        np.testing.assert_allclose(
            actual.snapshot.radial_densities.P_R,
            reference.snapshot.radial_densities.P_R,
            atol=2.0e-13,
            rtol=2.0e-13,
        )

    uncoupled = NonBornOppenheimerDiatomicIonTDSE(
        "HD+",
        initial_state=uncoupled_state,
        l_max=1,
        m_max=1,
        mu_max=2,
        initial_time=0.001,
        **common,
    )
    with pytest.raises(ValueError, match="compatibility metadata"):
        uncoupled.resume(intermediate, progress=False)


def test_coupled_interface_rejects_unsupported_options():
    state, _, _ = _initial_state("H2+")
    common = dict(
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=state,
        pair_exchange_parity=1,
        l1_max=1,
        l2_max=1,
        L_max=2,
        lambda_max=2,
        final_time=0.001,
        time_step=0.001,
    )
    with pytest.raises(ValueError, match="uncoupled-only kernel"):
        NonBornOppenheimerDiatomicIonTDSE(
            "H2+",
            wf_representation="coupled",
            coulomb_kernel="multipole-grouped",
            **common,
        )
    with pytest.raises(ValueError, match=r"A\(initial_time\)=0"):
        NonBornOppenheimerDiatomicIonTDSE(
            "H2+",
            wf_representation="coupled",
            field_gauge="velocity",
            fields=(_NonzeroInitialVelocityPulse(),),
            **common,
        )
