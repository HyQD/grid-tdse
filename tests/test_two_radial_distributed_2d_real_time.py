from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from grid_tdse.real_time import RealTimePropagator
from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    DiatomicIonDynamicsObservableEvaluator,
    DiatomicIonDynamicsSampler,
    DiatomicIonDynamicsSamplingPolicy,
    FEMDVRGridSpec,
    TwoRadialAxialDipoleInteraction,
    TwoRadialAbsorbingBoundaries,
    TwoRadialDistributed2DLayout,
    TwoRadialDistributed2DHamiltonian,
    TwoRadialDistributed2DLengthGaugeInteraction,
    TwoRadialDistributed2DRealTimeSolver,
    TwoRadialDistributed2DVelocityGaugeInteraction,
    TwoRadialSystemSpec,
    radial_potential,
)
from grid_tdse.spherical.two_radial.diatomic import _diatomic_parameters


class _Pulse:
    gauge = "length"

    def __call__(self, time):
        return 0.03 * np.sin(np.pi * time / 0.04) ** 2


class _VelocityPulse:
    gauge = "velocity"

    def __call__(self, time):
        return 0.02 * np.sin(np.pi * time / 0.04) ** 2


def _problem(*, M_tot=0):
    parameters = _diatomic_parameters("H2+", "2022")
    grid_r = FEMDVRGridSpec([0.0, 1.0, 3.0], [4, 4])
    grid_R = FEMDVRGridSpec([0.0, 0.8, 2.5], [4, 4])
    electron = AngularBasisSpec(1, 1)
    pair = AngularBasisSpec(1, 1, coordinate_parity=1)
    system = TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=grid_r,
            basis=electron,
            kinetic_prefactor=-0.5 / parameters.electronic_reduced_mass,
        ),
        coordinate2=CoordinateSpec(
            grid=grid_R,
            basis=pair,
            kinetic_prefactor=-0.5 / parameters.internuclear_reduced_mass,
            one_body_terms=(
                radial_potential(
                    lambda radius: parameters.reference_charge
                    * parameters.partner_charge
                    / radius
                ),
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
        M_tot=M_tot,
        radial_coulomb_method="analytic",
    ).build()
    rng = np.random.default_rng(319)
    state = rng.normal(size=system.psi_shape) + 1j * rng.normal(size=system.psi_shape)
    state /= np.sqrt(system.inner_product(state, state).real)
    return system, parameters, state


class _CompletedRequest:
    def Wait(self):
        return None


class _ReceiveRequest:
    def __init__(self, group, key, buffer):
        self.group = group
        self.key = key
        self.buffer = buffer

    def Wait(self):
        with self.group.condition:
            self.group.condition.wait_for(lambda: bool(self.group.messages[self.key]))
            values = self.group.messages[self.key].popleft()
        np.copyto(self.buffer, values)


class _ThreadCommunicatorGroup:
    def __init__(self, size):
        self.size = size
        self.condition = threading.Condition()
        self.messages = defaultdict(deque)
        self.reduction_values = defaultdict(dict)
        self.reduction_results = {}

    def communicator(self, rank):
        return _ThreadCommunicator(self, rank)


class _ThreadCommunicator:
    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.size = group.size
        self.reduction_round = 0

    def Isend(self, buffer, dest, tag):
        key = (self.rank, dest, tag)
        with self.group.condition:
            self.group.messages[key].append(np.asarray(buffer).copy())
            self.group.condition.notify_all()
        return _CompletedRequest()

    def Irecv(self, buffer, source, tag):
        return _ReceiveRequest(
            self.group,
            (source, self.rank, tag),
            buffer,
        )

    def allreduce(self, value):
        reduction_round = self.reduction_round
        self.reduction_round += 1
        with self.group.condition:
            values = self.group.reduction_values[reduction_round]
            values[self.rank] = np.asarray(value).copy()
            if len(values) == self.size:
                self.group.reduction_results[reduction_round] = sum(values.values())
                self.group.condition.notify_all()
            else:
                self.group.condition.wait_for(
                    lambda: reduction_round in self.group.reduction_results
                )
            return self.group.reduction_results[reduction_round]

    def Allreduce(self, send_buffer, receive_buffer):
        np.copyto(receive_buffer, self.allreduce(send_buffer))


def _propagator(final_time, integrator="sil"):
    return RealTimePropagator(
        final_time=final_time,
        time_step=0.01,
        integrator=integrator,
        max_krylov_dimension=20,
        krylov_rtol=1.0e-11,
        krylov_atol=1.0e-13,
        max_krylov_refinements=6,
        progress_interval=None,
    )


def _serial_run(system, parameters, state, pulse, final_time, integrator="sil"):
    gauge = pulse.gauge
    hamiltonian = system.hamiltonian(radial_operator_mode="element-local")
    dipole = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
        system,
        parameters,
        gauge=gauge,
        fields=(pulse,),
        radial_operator_mode="element-local",
    )
    evaluator = DiatomicIonDynamicsObservableEvaluator(
        system,
        parameters,
        hamiltonian,
        gauge=gauge,
        fields=(pulse,),
        radial_operator_mode="element-local",
    )
    sampler = DiatomicIonDynamicsSampler(
        evaluator,
        policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
    )

    def total_action(values, time):
        return hamiltonian(values) + dipole(values, time)

    def observe(values, step, time, diagnostics, final, checkpoint):
        sampler.sample(
            values,
            step,
            time,
            diagnostics=diagnostics,
            final=final,
            checkpoint=checkpoint,
        )

    result = _propagator(final_time, integrator).propagate(
        state,
        total_action,
        system.inner_product,
        state_observer=observe,
    )
    return result, sampler.time_series, dipole


def _distributed_run(
    system,
    parameters,
    state,
    pulse,
    process_grid,
    final_time,
    integrator="sil",
    krylov_reduction_mode="scalar",
):
    n_ranks = process_grid[0] * process_grid[1]
    layout = TwoRadialDistributed2DLayout(
        system,
        n_ranks,
        process_grid=process_grid,
    )
    local = layout.distribute(state)
    group = _ThreadCommunicatorGroup(n_ranks)
    solvers = [
        TwoRadialDistributed2DRealTimeSolver(
            layout,
            parameters,
            group.communicator(rank),
            fields=(pulse,),
            field_gauge=pulse.gauge,
            final_time=final_time,
            time_step=0.01,
            integrator=integrator,
            krylov_reduction_mode=krylov_reduction_mode,
            max_krylov_dimension=20,
            krylov_rtol=1.0e-11,
            krylov_atol=1.0e-13,
            max_krylov_refinements=6,
            sampling_policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
            progress_interval=None,
            profile=True,
        )
        for rank in range(n_ranks)
    ]
    with ThreadPoolExecutor(max_workers=n_ranks) as executor:
        futures = [
            executor.submit(solver.run, initial)
            for solver, initial in zip(solvers, local, strict=True)
        ]
        results = tuple(future.result(timeout=60.0) for future in futures)
    return layout, solvers, results


@pytest.mark.parametrize("dipole_kernel", ["reference", "row-wise"])
@pytest.mark.parametrize("process_grid", [(1, 2), (2, 1), (2, 2)])
def test_cartesian_length_gauge_action_matches_serial(process_grid, dipole_kernel):
    system, parameters, state = _problem()
    pulse = _Pulse()
    layout = TwoRadialDistributed2DLayout(
        system,
        process_grid[0] * process_grid[1],
        process_grid=process_grid,
    )
    distributed = TwoRadialDistributed2DLengthGaugeInteraction(
        layout,
        parameters,
        fields=(pulse,),
        dipole_kernel=dipole_kernel,
    )
    serial = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
        system,
        parameters,
        gauge="length",
        fields=(pulse,),
        radial_operator_mode="element-local",
        dipole_kernel=dipole_kernel,
    )

    actual = layout.assemble(
        tuple(distributed(part, 0.013) for part in layout.distribute(state))
    )

    np.testing.assert_allclose(
        actual,
        serial(state, 0.013),
        atol=2.0e-14,
        rtol=2.0e-14,
    )


@pytest.mark.parametrize("M_tot", [None, 0])
@pytest.mark.parametrize("dipole_kernel", ["reference", "row-wise"])
@pytest.mark.parametrize("process_grid", [(1, 2), (2, 1), (2, 2)])
def test_cartesian_velocity_gauge_action_matches_serial(
    process_grid, M_tot, dipole_kernel
):
    system, parameters, state = _problem(M_tot=M_tot)
    pulse = _VelocityPulse()
    n_ranks = process_grid[0] * process_grid[1]
    layout = TwoRadialDistributed2DLayout(
        system,
        n_ranks,
        process_grid=process_grid,
    )
    local = layout.distribute(state)
    group = _ThreadCommunicatorGroup(n_ranks)
    hamiltonians = [
        TwoRadialDistributed2DHamiltonian(
            layout,
            group.communicator(rank),
        )
        for rank in range(n_ranks)
    ]
    distributed = [
        TwoRadialDistributed2DVelocityGaugeInteraction(
            layout,
            parameters,
            hamiltonians[rank],
            fields=(pulse,),
            dipole_kernel=dipole_kernel,
        )
        for rank in range(n_ranks)
    ]
    serial = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
        system,
        parameters,
        gauge="velocity",
        fields=(pulse,),
        radial_operator_mode="element-local",
        dipole_kernel=dipole_kernel,
    )

    with ThreadPoolExecutor(max_workers=n_ranks) as executor:
        actions = tuple(
            executor.map(
                lambda pair: pair[0](pair[1], 0.013),
                zip(distributed, local, strict=True),
            )
        )

    np.testing.assert_allclose(
        layout.assemble(actions),
        serial(state, 0.013),
        atol=2.0e-13,
        rtol=2.0e-13,
    )


@pytest.mark.parametrize("process_grid", [(1, 2), (2, 1), (2, 2)])
def test_cartesian_velocity_gauge_one_step_matches_serial(process_grid):
    system, parameters, state = _problem()
    pulse = _VelocityPulse()
    serial, serial_series, _ = _serial_run(
        system,
        parameters,
        state,
        pulse,
        0.01,
    )
    layout, _, distributed = _distributed_run(
        system,
        parameters,
        state,
        pulse,
        process_grid,
        0.01,
    )

    actual = layout.assemble(tuple(result.state for result in distributed))
    np.testing.assert_allclose(actual, serial.state, atol=4.0e-11, rtol=4.0e-11)
    expected_energy = serial_series.samples[-1].snapshot.energy_numerator
    for result in distributed:
        snapshot = result.time_series.samples[-1].snapshot
        assert snapshot.gauge == "velocity"
        assert snapshot.field_symbol == "A"
        assert snapshot.energy_numerator == pytest.approx(
            expected_energy,
            abs=8.0e-11,
            rel=8.0e-11,
        )


@pytest.mark.parametrize("process_grid", [(1, 2), (2, 1), (2, 2)])
def test_cartesian_one_step_matches_serial(process_grid):
    system, parameters, state = _problem()
    pulse = _Pulse()
    serial, _, _ = _serial_run(system, parameters, state, pulse, 0.01)
    layout, _, distributed = _distributed_run(
        system,
        parameters,
        state,
        pulse,
        process_grid,
        0.01,
    )

    actual = layout.assemble(tuple(result.state for result in distributed))
    np.testing.assert_allclose(actual, serial.state, atol=3.0e-11, rtol=3.0e-11)
    for result in distributed:
        np.testing.assert_allclose(
            result.norm_history,
            serial.norm_history,
            atol=3.0e-12,
            rtol=3.0e-12,
        )
        assert result.hamiltonian_profile.hamiltonian_calls > 0
        assert result.reduction_profile.collective_calls > 0


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_cartesian_exponential_integrators_match_serial(integrator):
    system, parameters, state = _problem()
    pulse = _Pulse()
    serial, _, _ = _serial_run(
        system,
        parameters,
        state,
        pulse,
        0.01,
        integrator,
    )
    layout, _, distributed = _distributed_run(
        system,
        parameters,
        state,
        pulse,
        (1, 2),
        0.01,
        integrator,
    )

    actual = layout.assemble(tuple(result.state for result in distributed))
    np.testing.assert_allclose(actual, serial.state, atol=3.0e-11, rtol=3.0e-11)


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_batched_krylov_reductions_match_scalar_and_use_fewer_collectives(
    integrator,
):
    system, parameters, state = _problem()
    pulse = _Pulse()
    layout, _, scalar = _distributed_run(
        system,
        parameters,
        state,
        pulse,
        (1, 2),
        0.01,
        integrator,
        "scalar",
    )
    _, _, batched = _distributed_run(
        system,
        parameters,
        state,
        pulse,
        (1, 2),
        0.01,
        integrator,
        "batched",
    )

    scalar_state = layout.assemble(tuple(result.state for result in scalar))
    batched_state = layout.assemble(tuple(result.state for result in batched))
    np.testing.assert_allclose(
        batched_state,
        scalar_state,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    for scalar_result, batched_result in zip(scalar, batched, strict=True):
        assert (
            batched_result.reduction_profile.collective_calls
            < scalar_result.reduction_profile.collective_calls
        )
        assert (
            batched_result.reduction_profile.local_inner_product_calls
            == scalar_result.reduction_profile.local_inner_product_calls
        )


def test_cartesian_real_time_rejects_unknown_krylov_reduction_mode():
    system, parameters, _ = _problem()
    layout = TwoRadialDistributed2DLayout(system, 1)

    with pytest.raises(ValueError, match="krylov_reduction_mode"):
        TwoRadialDistributed2DRealTimeSolver(
            layout,
            parameters,
            final_time=0.01,
            time_step=0.01,
            krylov_reduction_mode="unknown",
        )


def test_cartesian_short_time_series_and_final_state_match_serial():
    system, parameters, state = _problem()
    pulse = _Pulse()
    serial, serial_series, _ = _serial_run(
        system,
        parameters,
        state,
        pulse,
        0.03,
    )
    layout, _, distributed = _distributed_run(
        system,
        parameters,
        state,
        pulse,
        (2, 2),
        0.03,
    )

    actual = layout.assemble(tuple(result.state for result in distributed))
    np.testing.assert_allclose(actual, serial.state, atol=5.0e-11, rtol=5.0e-11)
    reference_records = serial_series.records()
    for result in distributed:
        assert result.time_series.accepted_steps.tolist() == [0, 1, 2, 3]
        records = result.time_series.records()
        assert len(records) == len(reference_records)
        for actual_record, reference_record in zip(
            records,
            reference_records,
            strict=True,
        ):
            assert actual_record.keys() == reference_record.keys()
            for name, reference_value in reference_record.items():
                actual_value = actual_record[name]
                if name.startswith("propagation."):
                    continue
                if isinstance(reference_value, (int, float)) and not isinstance(
                    reference_value,
                    bool,
                ):
                    assert actual_value == pytest.approx(
                        reference_value,
                        abs=8.0e-11,
                        rel=8.0e-11,
                    ), name
                else:
                    assert actual_value == reference_value
        assert all(
            sample.propagation.wall_time >= 0.0 for sample in result.time_series.samples
        )
        assert any(
            sample.propagation.krylov_operator_calls > 0
            for sample in result.time_series.samples[1:]
        )
        for actual_sample, reference_sample in zip(
            result.time_series.samples,
            serial_series.samples,
            strict=True,
        ):
            np.testing.assert_allclose(
                actual_sample.snapshot.radial_densities.P_r,
                reference_sample.snapshot.radial_densities.P_r,
                atol=8.0e-11,
                rtol=8.0e-11,
            )
            np.testing.assert_allclose(
                actual_sample.snapshot.radial_densities.P_R,
                reference_sample.snapshot.radial_densities.P_R,
                atol=8.0e-11,
                rtol=8.0e-11,
            )


def test_cartesian_cap_sia_matches_serial_and_samples_physical_energy():
    system, parameters, state = _problem()
    pulse = _Pulse()
    cap1 = -0.02j * (system.grid1.r / system.grid1.r[-1]) ** 3
    cap2 = -0.03j * (system.grid2.r / system.grid2.r[-1]) ** 3
    boundaries = TwoRadialAbsorbingBoundaries(
        system,
        cap1=cap1,
        cap2=cap2,
    )
    physical_hamiltonian = system.hamiltonian(radial_operator_mode="element-local")
    dipole = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
        system,
        parameters,
        gauge="length",
        fields=(pulse,),
        radial_operator_mode="element-local",
    )
    evaluator = DiatomicIonDynamicsObservableEvaluator(
        system,
        parameters,
        physical_hamiltonian,
        gauge="length",
        fields=(pulse,),
        radial_operator_mode="element-local",
    )
    sampler = DiatomicIonDynamicsSampler(
        evaluator,
        policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
    )

    def serial_action(values, time):
        return (
            physical_hamiltonian(values)
            + dipole(values, time)
            + boundaries.cap_action(values)
        )

    def observe(values, step, time, diagnostics, final, checkpoint):
        sampler.sample(
            values,
            step,
            time,
            diagnostics=diagnostics,
            final=final,
            checkpoint=checkpoint,
        )

    serial = _propagator(0.02, "sia").propagate(
        state,
        serial_action,
        system.inner_product,
        state_observer=observe,
    )

    process_grid = (1, 2)
    layout = TwoRadialDistributed2DLayout(
        system,
        2,
        process_grid=process_grid,
    )
    local = layout.distribute(state)
    group = _ThreadCommunicatorGroup(2)
    solvers = [
        TwoRadialDistributed2DRealTimeSolver(
            layout,
            parameters,
            group.communicator(rank),
            fields=(pulse,),
            field_gauge="length",
            final_time=0.02,
            time_step=0.01,
            integrator="sia",
            max_krylov_dimension=20,
            krylov_rtol=1.0e-11,
            krylov_atol=1.0e-13,
            max_krylov_refinements=6,
            sampling_policy=DiatomicIonDynamicsSamplingPolicy(interval=1),
            absorbing_boundaries=boundaries,
            progress_interval=None,
        )
        for rank in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(solver.run, initial)
            for solver, initial in zip(solvers, local, strict=True)
        ]
        distributed = tuple(future.result(timeout=60.0) for future in futures)

    actual = layout.assemble(tuple(result.state for result in distributed))
    np.testing.assert_allclose(actual, serial.state, atol=6.0e-11, rtol=6.0e-11)
    expected_energy = sampler.time_series.samples[-1].snapshot.energy_numerator
    for result in distributed:
        assert result.norm < result.propagation.initial_norm
        assert result.time_series.samples[
            -1
        ].snapshot.energy_numerator == pytest.approx(
            expected_energy,
            abs=1.0e-10,
            rel=1.0e-10,
        )


def test_cartesian_sil_rejects_cap():
    system, parameters, _ = _problem()
    boundaries = TwoRadialAbsorbingBoundaries(
        system,
        cap1=-0.01j * np.ones(system.grid1.n_points),
    )
    layout = TwoRadialDistributed2DLayout(system, 1)

    with pytest.raises(ValueError, match="cannot be used with a CAP"):
        TwoRadialDistributed2DRealTimeSolver(
            layout,
            parameters,
            final_time=0.01,
            time_step=0.01,
            integrator="sil",
            absorbing_boundaries=boundaries,
        )
