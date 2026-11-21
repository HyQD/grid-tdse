from functools import lru_cache

import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser

from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    OneElectronDiatomicTDSE,
    diagonalize_full_angular,
    transform_full_angular_axial_gauge_state,
)


PULSE_PARAMETERS = {
    "E0": 0.05,
    "omega": 4.0,
    "ncycles": 1,
    "phase": 0.0,
}
ANGLES_DEGREES = (0, 30, 45, 60, 90, 120, 135, 150, 180)
NODES = (0.0, 1.0, 2.0, 4.0, 8.0)


def _system(polar_angle, azimuth=0.0):
    axis = np.array(
        [
            np.sin(polar_angle) * np.cos(azimuth),
            np.sin(polar_angle) * np.sin(azimuth),
            np.cos(polar_angle),
        ]
    )
    return OneElectronDiatomicSystem.from_nuclear_positions(
        1.0,
        1.0,
        -axis,
        axis,
        name="oriented H2+ gauge qualification",
    )


@lru_cache(maxsize=None)
def _independent_ground_state(
    angle_degrees,
    azimuth_degrees,
    l_max,
    n_points,
):
    system = _system(
        np.deg2rad(angle_degrees),
        np.deg2rad(azimuth_degrees),
    )
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=NODES,
        n_points=n_points,
        l_max=l_max,
        m_max=l_max,
        nuclear_L_max=2 * l_max,
        nuclear_M_max=2 * l_max,
    )
    result = diagonalize_full_angular(
        hamiltonian,
        residual_tolerance=1.0e-11,
        max_iterations=400,
        maximum_subspace_dimension=48,
        retained_ritz_vectors=8,
        initial_orbitals=None,
        print_progress=False,
    )
    if not result.converged:
        raise RuntimeError("independent full-angular Davidson solve failed")
    return result


def _phase_aligned_error(hamiltonian, reference, candidate):
    overlap = hamiltonian.inner_product(reference, candidate)
    phase = overlap / abs(overlap)
    difference = candidate / phase - reference
    return float(
        np.sqrt(hamiltonian.inner_product(difference, difference).real)
    )


def _run_pair(
    angle_degrees,
    *,
    l_max,
    time_step=0.05,
    points_per_element=6,
    azimuth_degrees=0,
):
    angle = np.deg2rad(angle_degrees)
    azimuth = np.deg2rad(azimuth_degrees)
    system = _system(angle, azimuth)
    n_points = (points_per_element,) * (len(NODES) - 1)
    ground_state = _independent_ground_state(
        angle_degrees,
        azimuth_degrees,
        l_max,
        n_points,
    )
    initial_state = ground_state.orbital
    results = {}
    calculations = {}
    pulses = {}
    for gauge in ("length", "velocity"):
        pulse = SineSquareLaser(gauge=gauge, **PULSE_PARAMETERS)
        calculation = OneElectronDiatomicTDSE(
            system,
            nodes=NODES,
            n_points=n_points,
            l_max=l_max,
            m_max=l_max,
            nuclear_L_max=2 * l_max,
            nuclear_M_max=2 * l_max,
            angular_representation="full",
            initial_state=initial_state,
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
            reference_states={"ground": initial_state},
            radial_probability_interval=None,
            samplers={
                "state_snapshot": lambda state, time: np.array(
                    state,
                    copy=True,
                )
            },
            progress_interval=None,
        )
        pulses[gauge] = pulse
        calculations[gauge] = calculation
        results[gauge] = calculation.run(progress=False)

    length = results["length"]
    velocity = results["velocity"]
    hamiltonian = calculations["length"].hamiltonian
    np.testing.assert_allclose(length.times, velocity.times, atol=1.0e-15)

    transformed_errors = []
    transformed_norm_defects = []
    for index, (time, length_state, velocity_state) in enumerate(
        zip(
            length.times,
            length.samples["state_snapshot"],
            velocity.samples["state_snapshot"],
        )
    ):
        transformed = transform_full_angular_axial_gauge_state(
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
            transformed,
            transformed,
        ).real
        transformed_norm_defects.append(
            abs(transformed_norm - velocity.norm_history[index])
        )

    dipole_difference = (
        length.samples["total_dipole_numerator"]
        - velocity.samples["total_dipole_numerator"]
    )
    momentum_difference = (
        length.samples["canonical_momentum_numerator"]
        - velocity.samples["mechanical_momentum_numerator"]
    )

    return {
        "angle_degrees": angle_degrees,
        "azimuth_degrees": azimuth_degrees,
        "l_max": l_max,
        "ground_energy": ground_state.electronic_energy,
        "ground_residual": ground_state.residual_norm,
        "length": length,
        "velocity": velocity,
        "vector_potential": pulses["velocity"],
        "endpoint_state_error": _phase_aligned_error(
            hamiltonian,
            length.state,
            velocity.state,
        ),
        "maximum_transformed_state_error": max(transformed_errors),
        "maximum_transformed_norm_defect": max(transformed_norm_defects),
        "dipole_component_rms_error": np.sqrt(
            np.mean(np.abs(dipole_difference) ** 2, axis=0)
        ),
        "momentum_component_rms_error": np.sqrt(
            np.mean(np.abs(momentum_difference) ** 2, axis=0)
        ),
        "endpoint_dipole_error": float(
            np.linalg.norm(
                length.samples["total_dipole_numerator"][-1]
                - velocity.samples["total_dipole_numerator"][-1]
            )
        ),
        "endpoint_momentum_error": float(
            np.linalg.norm(
                length.samples["canonical_momentum_numerator"][-1]
                - velocity.samples["canonical_momentum_numerator"][-1]
            )
        ),
        "endpoint_energy_error": abs(
            length.samples["field_free_energy_numerator"][-1]
            - velocity.samples["field_free_energy_numerator"][-1]
        ),
        "endpoint_population_error": abs(
            length.samples["state_populations"][-1, 0]
            - velocity.samples["state_populations"][-1, 0]
        ),
    }


@pytest.fixture(scope="module")
def gauge_cases():
    cases = {
        (angle, 4, 0.05): _run_pair(angle, l_max=4)
        for angle in ANGLES_DEGREES
    }
    for l_max in (2, 4, 6, 8):
        cases[(45, l_max, 0.025)] = _run_pair(
            45,
            l_max=l_max,
            time_step=0.025,
        )
    cases[(45, 6, 0.05)] = _run_pair(45, l_max=6)
    for points_per_element in (4, 8):
        cases[(45, 6, 0.025, points_per_element)] = _run_pair(
            45,
            l_max=6,
            time_step=0.025,
            points_per_element=points_per_element,
        )
    cases[("azimuth", 45, 60, 4, 0.05)] = _run_pair(
        45,
        l_max=4,
        azimuth_degrees=60,
    )
    return cases


def test_all_orientations_use_complete_full_angular_representation(gauge_cases):
    for angle in ANGLES_DEGREES:
        case = gauge_cases[(angle, 4, 0.05)]
        assert case["angle_degrees"] == angle
        assert case["l_max"] == 4
        assert case["ground_residual"] < 1.0e-10
        assert case["vector_potential"](0.0) == pytest.approx(
            0.0,
            abs=1.0e-15,
        )
        assert case["vector_potential"](
            case["vector_potential"].tprime
        ) == pytest.approx(0.0, abs=1.0e-15)
        assert case["endpoint_state_error"] < 1.0e-4
        assert case["maximum_transformed_state_error"] < 1.2e-4
        assert case["maximum_transformed_norm_defect"] < 1.0e-6


def test_gauge_discrepancies_decrease_with_angular_refinement(gauge_cases):
    cases = [gauge_cases[(45, l_max, 0.025)] for l_max in (2, 4, 6, 8)]

    for metric in (
        "endpoint_state_error",
        "maximum_transformed_state_error",
        "maximum_transformed_norm_defect",
        "endpoint_dipole_error",
        "endpoint_momentum_error",
    ):
        values = np.asarray([case[metric] for case in cases])
        assert np.all(np.diff(values) < 0.0), (metric, values)
        assert values[-1] < 0.1 * values[0], (metric, values)

    for metric in (
        "dipole_component_rms_error",
        "momentum_component_rms_error",
    ):
        values = np.asarray([case[metric] for case in cases])
        np.testing.assert_array_less(np.diff(values[:, (0, 2)], axis=0), 0.0)
        assert np.max(np.abs(values[:, 1])) < 5.0e-18
        assert np.all(values[-1, (0, 2)] < 0.1 * values[0, (0, 2)])


def test_refined_endpoint_observables_are_gauge_equivalent(gauge_cases):
    case = gauge_cases[(45, 8, 0.025)]
    length = case["length"]
    velocity = case["velocity"]

    assert abs(length.norm - length.initial_norm) < 3.0e-10
    assert abs(velocity.norm - velocity.initial_norm) < 3.0e-10
    assert abs(length.norm - velocity.norm) < 3.0e-10
    assert case["endpoint_state_error"] < 1.0e-5
    assert case["endpoint_energy_error"] < 5.0e-8
    assert case["endpoint_population_error"] < 5.0e-8
    assert case["endpoint_dipole_error"] < 1.0e-5
    assert case["endpoint_momentum_error"] < 1.0e-6


def test_time_and_radial_refinement_resolve_angular_trend(gauge_cases):
    coarse_time = gauge_cases[(45, 6, 0.05)]
    reference = gauge_cases[(45, 6, 0.025)]
    coarse_radial = gauge_cases[(45, 6, 0.025, 4)]
    fine_radial = gauge_cases[(45, 6, 0.025, 8)]
    angularly_refined = gauge_cases[(45, 8, 0.025)]

    assert reference["maximum_transformed_state_error"] < (
        coarse_time["maximum_transformed_state_error"]
    )
    assert reference["endpoint_state_error"] == pytest.approx(
        coarse_time["endpoint_state_error"],
        rel=0.02,
    )
    for metric in (
        "endpoint_state_error",
        "maximum_transformed_state_error",
        "endpoint_dipole_error",
    ):
        coarse_difference = abs(coarse_radial[metric] - reference[metric])
        fine_difference = abs(fine_radial[metric] - reference[metric])
        assert fine_difference < coarse_difference, metric
    assert angularly_refined["endpoint_state_error"] < (
        0.4 * coarse_radial["endpoint_state_error"]
    )


def test_homonuclear_paired_orientations_have_matching_gauge_errors(
    gauge_cases,
):
    for left_angle, right_angle in ((0, 180), (30, 150), (45, 135), (60, 120)):
        left = gauge_cases[(left_angle, 4, 0.05)]
        right = gauge_cases[(right_angle, 4, 0.05)]
        for metric in (
            "endpoint_state_error",
            "maximum_transformed_state_error",
            "endpoint_dipole_error",
            "endpoint_momentum_error",
        ):
            assert left[metric] == pytest.approx(
                right[metric],
                rel=5.0e-5,
                abs=2.0e-12,
            )


def test_xz_plane_reflection_forbids_y_response(gauge_cases):
    for angle in ANGLES_DEGREES:
        case = gauge_cases[(angle, 4, 0.05)]
        for gauge in ("length", "velocity"):
            result = case[gauge]
            dipole_y = result.samples["total_dipole_numerator"][:, 1]
            momentum_y = result.samples["canonical_momentum_numerator"][:, 1]
            assert np.max(np.abs(dipole_y)) < 2.0e-15
            assert np.max(np.abs(momentum_y)) < 2.0e-15


def test_parallel_intermediate_and_perpendicular_responses_differ(gauge_cases):
    parallel = gauge_cases[(0, 4, 0.05)]["length"]
    intermediate = gauge_cases[(45, 4, 0.05)]["length"]
    perpendicular = gauge_cases[(90, 4, 0.05)]["length"]

    parallel_dipole = parallel.samples["induced_dipole_numerator"]
    intermediate_dipole = intermediate.samples["induced_dipole_numerator"]
    perpendicular_dipole = perpendicular.samples["induced_dipole_numerator"]
    assert np.max(np.abs(parallel_dipole[:, 0])) < 2.0e-15
    assert np.max(np.abs(perpendicular_dipole[:, 0])) < 2.0e-15
    assert np.max(np.abs(intermediate_dipole[:, 0])) > 5.0e-4
    assert np.max(
        np.abs(parallel_dipole[:, 2] - perpendicular_dipole[:, 2])
    ) > 5.0e-4

    parallel_population = parallel.samples["state_populations"][:, 0]
    perpendicular_population = perpendicular.samples["state_populations"][:, 0]
    assert np.max(
        np.abs(parallel_population - perpendicular_population)
    ) > 5.0e-5


def test_homonuclear_mirrored_axes_rotate_vector_responses(gauge_cases):
    reflection = np.diag([-1.0, -1.0, 1.0])
    for left_angle, right_angle in (
        (0, 180),
        (30, 150),
        (45, 135),
        (60, 120),
    ):
        left = gauge_cases[(left_angle, 4, 0.05)]["length"]
        right = gauge_cases[(right_angle, 4, 0.05)]["length"]
        np.testing.assert_allclose(
            right.samples["field_free_energy_numerator"],
            left.samples["field_free_energy_numerator"],
            atol=3.0e-12,
            rtol=3.0e-12,
        )
        np.testing.assert_allclose(
            right.samples["state_populations"],
            left.samples["state_populations"],
            atol=3.0e-12,
            rtol=3.0e-12,
        )
        for name in (
            "induced_dipole_numerator",
            "canonical_momentum_numerator",
        ):
            np.testing.assert_allclose(
                right.samples[name],
                left.samples[name] @ reflection.T,
                atol=3.0e-12,
                rtol=3.0e-12,
            )


def test_azimuth_rotation_preserves_scalars_and_rotates_vectors(gauge_cases):
    reference = gauge_cases[(45, 4, 0.05)]["length"]
    rotated_case = gauge_cases[("azimuth", 45, 60, 4, 0.05)]
    rotated = rotated_case["length"]
    angle = np.deg2rad(rotated_case["azimuth_degrees"])
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert rotated_case["ground_energy"] == pytest.approx(
        gauge_cases[(45, 4, 0.05)]["ground_energy"],
        abs=2.0e-12,
    )
    for name in (
        "norm",
        "field_free_energy_numerator",
        "state_populations",
    ):
        np.testing.assert_allclose(
            rotated.samples[name],
            reference.samples[name],
            atol=3.0e-12,
            rtol=3.0e-12,
        )
    for name in (
        "induced_dipole_numerator",
        "canonical_momentum_numerator",
    ):
        np.testing.assert_allclose(
            rotated.samples[name],
            reference.samples[name] @ rotation.T,
            atol=3.0e-12,
            rtol=3.0e-12,
        )
