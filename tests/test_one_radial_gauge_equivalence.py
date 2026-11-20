import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser

from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    OneElectronDiatomicTDSE,
    diagonalize_fixed_m,
    transform_axial_gauge_state,
)


PULSE_PARAMETERS = {
    "E0": 0.05,
    "omega": 4.0,
    "ncycles": 1,
    "phase": 0.0,
}
NODES = [0.0, 1.0, 2.0, 4.0, 8.0]


def _phase_aligned_error(hamiltonian, reference, candidate):
    overlap = hamiltonian.inner_product(reference, candidate)
    phase = overlap / abs(overlap)
    difference = candidate / phase - reference
    return float(
        np.sqrt(hamiltonian.inner_product(difference, difference).real)
    )


def _relative_l2(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    scale = max(np.linalg.norm(left), np.linalg.norm(right))
    return 0.0 if scale == 0.0 else float(np.linalg.norm(left - right) / scale)


def _run_pair(*, l_max, points_per_element=6, time_step=0.05):
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        particle_mass=1.0,
        name="H2+ gauge regression",
    )
    n_points = [points_per_element] * (len(NODES) - 1)
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=NODES,
        n_points=n_points,
        l_max=l_max,
        m_max=0,
    )
    ground_state = diagonalize_fixed_m(hamiltonian, m=0).orbital
    results = {}
    pulses = {}
    calculations = {}
    for gauge in ("length", "velocity"):
        pulse = SineSquareLaser(gauge=gauge, **PULSE_PARAMETERS)
        calculation = OneElectronDiatomicTDSE(
            system,
            nodes=NODES,
            n_points=n_points,
            l_max=l_max,
            m=0,
            initial_state=ground_state,
            fields=pulse,
            field_gauge=gauge,
            integrator="sia",
            integrator_options={
                "max_krylov_dimension": 30,
                "rtol": 1.0e-11,
                "atol": 1.0e-13,
                "max_refinements": 6,
            },
            final_time=pulse.tprime,
            time_step=time_step,
            reference_states={"ground": ground_state},
            radial_probability_interval=None,
            samplers={
                "state_snapshot": lambda state, time: np.array(
                    state, copy=True
                )
            },
            progress_interval=None,
        )
        pulses[gauge] = pulse
        calculations[gauge] = calculation
        results[gauge] = calculation.run(progress=False)

    length = results["length"]
    velocity = results["velocity"]
    np.testing.assert_allclose(length.times, velocity.times, atol=1.0e-15)

    transformed_errors = []
    transformed_norm_defects = []
    for time, length_state, velocity_state in zip(
        length.times,
        length.samples["state_snapshot"],
        velocity.samples["state_snapshot"],
    ):
        transformed = transform_axial_gauge_state(
            hamiltonian,
            velocity_state,
            pulses["velocity"](time),
            source_gauge="velocity",
            target_gauge="length",
        )
        transformed_errors.append(
            _phase_aligned_error(hamiltonian, length_state, transformed)
        )
        transformed_norm = hamiltonian.inner_product(
            transformed, transformed
        ).real
        transformed_norm_defects.append(abs(transformed_norm - velocity.norm))

    return {
        "hamiltonian": hamiltonian,
        "length": length,
        "velocity": velocity,
        "endpoint_state_error": _phase_aligned_error(
            hamiltonian, length.state, velocity.state
        ),
        "maximum_transformed_state_error": max(transformed_errors),
        "maximum_transformed_norm_defect": max(transformed_norm_defects),
        "dipole_relative_l2": _relative_l2(
            length.samples["total_dipole_numerator"],
            velocity.samples["total_dipole_numerator"],
        ),
        "endpoint_energy_error": abs(
            length.samples["field_free_energy_numerator"][-1]
            - velocity.samples["field_free_energy_numerator"][-1]
        ),
        "endpoint_population_error": abs(
            length.samples["state_populations"][-1, 0]
            - velocity.samples["state_populations"][-1, 0]
        ),
        "endpoint_dipole_error": abs(
            length.samples["total_dipole_numerator"][-1]
            - velocity.samples["total_dipole_numerator"][-1]
        ),
    }


@pytest.fixture(scope="module")
def gauge_cases():
    return {
        "l2": _run_pair(l_max=2),
        "l4": _run_pair(l_max=4),
        "l6": _run_pair(l_max=6),
        "l8": _run_pair(l_max=8),
        "coarse_time": _run_pair(l_max=6, time_step=0.2),
        "coarse_radial": _run_pair(l_max=6, points_per_element=4),
        "fine_radial": _run_pair(l_max=6, points_per_element=8),
    }


def test_projected_axial_gauge_transform_has_expected_electron_phase():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
    )
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=[0.0, 1.0, 3.0],
        n_points=[4, 4],
        l_max=2,
        m_max=0,
    )
    state = np.zeros(hamiltonian.fixed_m_shape(0))
    state[:, 0] = 1.0
    amplitude = 1.0e-7

    transformed = transform_axial_gauge_state(
        hamiltonian,
        state,
        amplitude,
        source_gauge="velocity",
        target_gauge="length",
    )

    np.testing.assert_allclose(transformed[:, 0].real, 1.0, atol=2.0e-13)
    np.testing.assert_allclose(
        transformed[:, 1].imag,
        amplitude * hamiltonian.r / np.sqrt(3.0),
        atol=2.0e-13,
    )
    np.testing.assert_array_equal(
        transform_axial_gauge_state(
            hamiltonian,
            state,
            0.0,
            source_gauge="velocity",
            target_gauge="length",
        ),
        state,
    )


def test_gauge_discrepancy_decreases_with_angular_refinement(gauge_cases):
    cases = [gauge_cases[name] for name in ("l2", "l4", "l6", "l8")]

    for metric in (
        "endpoint_state_error",
        "maximum_transformed_state_error",
        "maximum_transformed_norm_defect",
        "dipole_relative_l2",
    ):
        values = np.asarray([case[metric] for case in cases])
        assert np.all(np.diff(values) < 0.0), (metric, values)
        assert values[-1] < 0.1 * values[0], (metric, values)


def test_endpoint_observables_agree_after_refinement(gauge_cases):
    case = gauge_cases["l8"]
    length = case["length"]
    velocity = case["velocity"]

    assert abs(length.norm - length.initial_norm) < 2.0e-10
    assert abs(velocity.norm - velocity.initial_norm) < 2.0e-10
    assert abs(length.norm - velocity.norm) < 2.0e-10
    assert case["endpoint_state_error"] < 1.0e-5
    assert case["endpoint_energy_error"] < 2.0e-6
    assert case["endpoint_population_error"] < 2.0e-7
    assert case["endpoint_dipole_error"] < 1.0e-5


def test_time_and_radial_refinement_resolve_angular_gauge_trend(gauge_cases):
    coarse_time = gauge_cases["coarse_time"]
    reference = gauge_cases["l6"]
    coarse_radial = gauge_cases["coarse_radial"]
    fine_radial = gauge_cases["fine_radial"]

    assert reference["endpoint_state_error"] < (
        0.3 * coarse_time["endpoint_state_error"]
    )
    assert reference["dipole_relative_l2"] < (
        0.35 * coarse_time["dipole_relative_l2"]
    )
    assert reference["endpoint_state_error"] < (
        0.9 * coarse_radial["endpoint_state_error"]
    )
    assert reference["dipole_relative_l2"] < (
        0.9 * coarse_radial["dipole_relative_l2"]
    )
    assert abs(
        fine_radial["endpoint_state_error"]
        - reference["endpoint_state_error"]
    ) < 0.1 * coarse_radial["endpoint_state_error"]
    assert abs(
        fine_radial["dipole_relative_l2"]
        - reference["dipole_relative_l2"]
    ) < 0.1 * coarse_radial["dipole_relative_l2"]
