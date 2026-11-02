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
    TwoRadialDistributed2DLayout,
    TwoRadialDistributed2DLengthGaugeInteraction,
    TwoRadialDistributed2DRealTimeSolver,
    TwoRadialSystemSpec,
    radial_potential,
)
from grid_tdse.spherical.two_radial.diatomic import _diatomic_parameters


class _Pulse:
    gauge = "length"

    def __call__(self, time):
        return 0.03 * np.sin(np.pi * time / 0.04) ** 2


def _problem():
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
        M_tot=0,
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
    hamiltonian = system.hamiltonian(radial_operator_mode="element-local")
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
        hamiltonian,
        gauge="length",
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
            final_time=final_time,
            time_step=0.01,
            integrator=integrator,
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


@pytest.mark.parametrize("process_grid", [(1, 2), (2, 1), (2, 2)])
def test_cartesian_length_gauge_action_matches_serial(process_grid):
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
    )
    serial = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
        system,
        parameters,
        gauge="length",
        fields=(pulse,),
        radial_operator_mode="element-local",
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
