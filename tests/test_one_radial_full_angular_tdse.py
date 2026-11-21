import json

import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.real_time import RealTimePropagator, load_real_time_checkpoint
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    OneElectronDiatomicTDSE,
    SphericalOneRadialAbsorbingBoundaries,
    SphericalOneRadialFixedMTimeDependentHamiltonian,
    SphericalOneRadialFullAngularDynamicsObservables,
    SphericalOneRadialFullAngularKineticCrankNicolsonPreconditioner,
    SphericalOneRadialFullAngularTimeDependentHamiltonian,
    diagonalize_full_angular,
)


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


def _system(angle=np.pi / 4.0):
    positions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    positions = positions @ _rotation_y(angle).T
    return OneElectronDiatomicSystem.from_nuclear_positions(
        1.0,
        1.0,
        positions[0],
        positions[1],
        name="tilted H2+",
    )


def _hamiltonian(system=None, *, l_max=2, m_max=None):
    return OneElectronDiatomicHamiltonian(
        _system() if system is None else system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[5, 5, 5],
        l_max=l_max,
        m_max=m_max,
    )


@pytest.fixture(scope="module")
def full_problem():
    system = _system()
    hamiltonian = _hamiltonian(system)
    states = diagonalize_full_angular(
        hamiltonian,
        n_states=2,
        solver="direct",
    )
    return system, hamiltonian, states


def _pulse(time):
    return 0.01 * np.sin(np.pi * time / 0.08) ** 2


def _integrator_options(integrator, *, preconditioner=True):
    if integrator == "crank-nicolson":
        return {
            "solver": "gmres",
            "rtol": 1.0e-11,
            "atol": 1.0e-13,
            "preconditioner": preconditioner,
        }
    return {
        "max_krylov_dimension": 30,
        "rtol": 1.0e-12,
        "atol": 1.0e-14,
        "max_refinements": 6,
    }


def _calculation(system, state, integrator="sia", **overrides):
    arguments = dict(
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[5, 5, 5],
        l_max=2,
        angular_representation="full",
        initial_state=state,
        fields=_pulse,
        field_gauge="length",
        integrator=integrator,
        integrator_options=_integrator_options(integrator),
        final_time=0.04,
        time_step=0.01,
        radial_probability_interval=2,
        progress_interval=None,
    )
    arguments.update(overrides)
    return OneElectronDiatomicTDSE(system, **arguments)


def _distance(hamiltonian, left, right):
    difference = left - right
    return np.sqrt(hamiltonian.inner_product(difference, difference).real)


def test_full_preconditioner_matches_channelwise_dense_solves(full_problem):
    _, hamiltonian, _ = full_problem
    time_step = 0.03
    prepared = SphericalOneRadialFullAngularKineticCrankNicolsonPreconditioner(
        hamiltonian,
        time_step=time_step,
    )
    generator = np.random.default_rng(811)
    states = generator.normal(size=(2, *hamiltonian.orbital_shape))
    states = states + 1j * generator.normal(size=states.shape)
    actual = prepared(states)
    expected = np.empty_like(actual)
    identity = np.eye(hamiltonian.r.size)

    for angular_index, (l_value, _) in enumerate(hamiltonian.orbital_lm):
        kinetic = (
            -0.5
            / hamiltonian.system.particle_mass
            * hamiltonian.second_derivative
        )
        kinetic = kinetic + np.diag(
            l_value
            * (l_value + 1)
            / (
                2.0
                * hamiltonian.system.particle_mass
                * hamiltonian.r**2
            )
        )
        left = identity + 0.5j * time_step * kinetic
        for batch in range(states.shape[0]):
            expected[batch, :, angular_index] = np.linalg.solve(
                left,
                states[batch, :, angular_index],
            )

    np.testing.assert_allclose(actual, expected)
    assert prepared.applications == 1
    assert prepared.metadata["angular_representation"] == "full"
    json.dumps(prepared.metadata)


def test_full_cn_preconditioner_reduces_iterations_and_preserves_state(
    full_problem,
    tmp_path,
):
    _, hamiltonian, states = full_problem
    boundaries = SphericalOneRadialAbsorbingBoundaries(
        hamiltonian,
        mask=RadialCosineMask(r0=4.5, n=8),
    )
    dynamics = SphericalOneRadialFullAngularTimeDependentHamiltonian(
        hamiltonian,
        fields=_pulse,
        absorbing_boundaries=boundaries,
    )
    propagator = RealTimePropagator(
        final_time=0.08,
        time_step=0.04,
        integrator="crank-nicolson",
        linear_solver="gmres",
        linear_rtol=1.0e-10,
        linear_atol=1.0e-13,
        progress_interval=None,
    )
    baseline = dynamics.propagate(propagator, states.orbital)
    preconditioner = (
        SphericalOneRadialFullAngularKineticCrankNicolsonPreconditioner(
            hamiltonian,
            time_step=propagator.time_step,
        )
    )
    prepared = dynamics.propagate(
        propagator,
        states.orbital,
        preconditioner=preconditioner,
        checkpoint_path=tmp_path / "full-cn.chkpt",
    )

    assert np.all(prepared.linear_iterations > 0)
    assert np.all(np.isfinite(prepared.linear_residual_norms))
    assert np.sum(prepared.linear_iterations) < np.sum(
        baseline.linear_iterations
    )
    assert _distance(hamiltonian, prepared.state, baseline.state) < 3.0e-9
    assert preconditioner.applications > 0
    checkpoint = load_real_time_checkpoint(tmp_path / "full-cn.chkpt")
    assert (
        checkpoint.latest_segment.dynamics["preconditioner"]
        == preconditioner.metadata
    )
    with pytest.raises(ValueError, match="requires integrator"):
        cap = SphericalOneRadialFullAngularTimeDependentHamiltonian(
            hamiltonian,
            absorbing_boundaries=SphericalOneRadialAbsorbingBoundaries(
                hamiltonian,
                cap=CubicCAP(alpha=0.5),
            ),
        )
        cap.propagate(
            RealTimePropagator(
                final_time=0.01,
                time_step=0.01,
                integrator="crank-nicolson",
                progress_interval=None,
            ),
            states.orbital,
        )


def test_high_level_representation_selection_and_vector_samples(full_problem):
    system, _, states = full_problem
    calculation = _calculation(system, states.orbital)
    result = calculation.run(progress=False)

    assert calculation.angular_representation == "full"
    assert calculation.m is None
    assert isinstance(
        calculation.dynamics,
        SphericalOneRadialFullAngularTimeDependentHamiltonian,
    )
    assert isinstance(
        calculation.observables,
        SphericalOneRadialFullAngularDynamicsObservables,
    )
    assert result.samples["total_dipole_numerator"].shape == (5, 3)
    assert result.samples["canonical_momentum_numerator"].shape == (5, 3)
    assert calculation.compatibility_metadata["angular_representation"] == (
        "full"
    )
    assert calculation.compatibility_metadata["channel_lm"] == [
        list(channel) for channel in calculation.hamiltonian.orbital_lm
    ]
    assert calculation.dynamics_metadata["origin"] == [0.0, 0.0, 0.0]
    json.dumps(calculation.compatibility_metadata)
    json.dumps(calculation.dynamics_metadata)

    axial_system = OneElectronDiatomicSystem(1.0, 1.0, 1.0)
    fixed_hamiltonian = OneElectronDiatomicHamiltonian(
        axial_system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[5, 5, 5],
        l_max=2,
        m_max=0,
    )
    fixed_state = np.zeros(fixed_hamiltonian.fixed_m_shape(0))
    fixed_state[0, 0] = 1.0
    fixed = OneElectronDiatomicTDSE(
        axial_system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[5, 5, 5],
        l_max=2,
        initial_state=fixed_state,
        final_time=0.01,
        time_step=0.01,
        progress_interval=None,
    )
    assert fixed.angular_representation == "fixed_m"
    assert isinstance(
        fixed.dynamics,
        SphericalOneRadialFixedMTimeDependentHamiltonian,
    )


def test_full_checkpoint_exact_resume_matches_uninterrupted(
    full_problem,
    tmp_path,
):
    system, _, states = full_problem
    path = tmp_path / "full-interrupted.chkpt"
    interrupted = _calculation(
        system,
        states.orbital,
        fields=(),
        checkpoint_path=path,
        checkpoint_interval=1,
    )
    arguments = interrupted._propagation_arguments()

    def stop_after_first_checkpoint(
        values,
        accepted_step,
        time,
        diagnostics,
        final,
        checkpoint,
    ):
        del values, time, diagnostics, final
        if accepted_step == 1 and checkpoint:
            raise RuntimeError("simulated interruption")

    arguments["state_observer"] = stop_after_first_checkpoint
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.dynamics.propagate(
            interrupted.propagator,
            interrupted.initial_state,
            **arguments,
        )

    stored = load_real_time_checkpoint(path)
    assert stored.current_time == pytest.approx(0.01)
    resumed = _calculation(
        system,
        states.orbital,
        fields=(),
        initial_time=0.01,
        checkpoint_path=path,
    )
    resumed_result = resumed.resume(path, progress=False)
    reference = _calculation(
        system,
        states.orbital,
        fields=(),
    ).run(progress=False)

    np.testing.assert_allclose(
        resumed_result.state,
        reference.state,
        atol=3.0e-12,
    )


@pytest.mark.parametrize(
    "change",
    ["geometry", "origin", "l_max", "m_max"],
)
def test_full_checkpoint_rejects_incompatible_configuration(
    full_problem,
    tmp_path,
    change,
):
    system, _, states = full_problem
    path = tmp_path / f"compatibility-{change}.chkpt"
    first = _calculation(
        system,
        states.orbital,
        fields=(),
        final_time=0.01,
        checkpoint_path=path,
        checkpoint_interval=1,
    )
    first.run(progress=False)

    changed_system = system
    changed_state = states.orbital
    options = {"initial_time": 0.01, "fields": ()}
    if change == "geometry":
        changed_system = _system(np.pi / 6.0)
    elif change == "origin":
        options["origin"] = (0.0, 0.0, 0.2)
    elif change == "l_max":
        options["l_max"] = 3
        changed_state = np.zeros(
            (_hamiltonian(system, l_max=3).orbital_shape),
            dtype=complex,
        )
        changed_state[0, 0] = 1.0
    else:
        options["m_max"] = 1
        changed_state = np.zeros(
            (_hamiltonian(system, m_max=1).orbital_shape),
            dtype=complex,
        )
        changed_state[0, 0] = 1.0

    continuation = _calculation(
        changed_system,
        changed_state,
        **options,
    )
    with pytest.raises(ValueError, match="compatibility metadata"):
        continuation.continue_from(path, progress=False)


def test_full_interface_validation(full_problem):
    system, _, states = full_problem
    with pytest.raises(ValueError, match="m is not selectable"):
        _calculation(system, states.orbital, m=1)
    with pytest.raises(ValueError, match="different z origins"):
        _calculation(
            system,
            states.orbital,
            origin=(0.0, 0.0, 0.2),
            origin_z=0.3,
        )
    with pytest.raises(ValueError, match="requires angular_representation"):
        OneElectronDiatomicTDSE(
            system,
            nodes=[0.0, 1.0, 3.0, 6.0],
            n_points=[5, 5, 5],
            l_max=2,
            initial_state=states.orbital,
            final_time=0.01,
            time_step=0.01,
        )
