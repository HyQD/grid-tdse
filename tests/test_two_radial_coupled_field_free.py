import numpy as np
import pytest

from grid_tdse.ground_state import DiagonalDavidsonPreconditioner
from grid_tdse.real_time import RealTimePropagator
from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    CoupledAngularBasisSpec,
    CoupledAngularTransform,
    FEMDVRGridSpec,
    TwoRadialAbsorbingBoundaries,
    TwoRadialSystemSpec,
    radial_potential,
)


def _specification(l1_max, l2_max, M_tot):
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=FEMDVRGridSpec([0.0, 0.8, 2.4], [4, 5]),
            basis=AngularBasisSpec(l1_max),
            kinetic_prefactor=-0.7,
            one_body_terms=(radial_potential(lambda r: -1.2 / r),),
        ),
        coordinate2=CoordinateSpec(
            grid=FEMDVRGridSpec([0.0, 0.6, 2.0], [5, 4]),
            basis=AngularBasisSpec(l2_max),
            kinetic_prefactor=-0.004,
            one_body_terms=(radial_potential(lambda r: 0.8 / r),),
        ),
        multipoles=CoulombMultipoleSpec(2, 2),
        coulomb_terms=(
            CoulombTerm(-2.0 / 3.0, -1.0),
            CoulombTerm(1.0 / 3.0, -1.0),
        ),
        M_tot=M_tot,
        radial_coulomb_method="analytic",
    )


def _problem(l1_max=2, l2_max=2, L_max=4, M_tot=0):
    specification = _specification(l1_max, l2_max, M_tot)
    basis = CoupledAngularBasisSpec(
        l1_max,
        l2_max,
        L_max,
        M_tot=M_tot,
    )
    return specification, basis, specification.build_coupled(basis)


@pytest.mark.parametrize(
    ("l1_max", "l2_max", "L_max", "M_tot"),
    [(1, 1, 2, 0), (2, 1, 3, 1), (2, 2, 1, 0)],
)
@pytest.mark.parametrize("radial_operator_mode", ["global", "element-local"])
def test_total_and_component_actions_match_transformed_uncoupled_oracle(
    l1_max,
    l2_max,
    L_max,
    M_tot,
    radial_operator_mode,
):
    specification, basis, coupled_system = _problem(
        l1_max,
        l2_max,
        L_max,
        M_tot,
    )
    transform = CoupledAngularTransform(basis)
    coupled = coupled_system.hamiltonian(
        radial_operator_mode=radial_operator_mode
    )
    uncoupled_system = specification.build(
        radial_operator_mode=radial_operator_mode
    )
    uncoupled = uncoupled_system.hamiltonian(
        radial_operator_mode=radial_operator_mode
    )
    state = coupled_system.random_state(
        np.random.default_rng(307 + l1_max + 10 * l2_max + M_tot)
    )
    uncoupled_state = transform.to_uncoupled(state)

    comparisons = (
        (
            coupled.apply_one_body_coordinate1(state),
            uncoupled.apply_one_body_coordinate1(uncoupled_state),
        ),
        (
            coupled.apply_one_body_coordinate2(state),
            uncoupled.apply_one_body_coordinate2(uncoupled_state),
        ),
        (
            coupled.apply_coulomb(state),
            uncoupled.apply_coulomb(uncoupled_state),
        ),
        (coupled.physical_action(state), uncoupled(uncoupled_state)),
    )
    for actual, uncoupled_action in comparisons:
        np.testing.assert_allclose(
            actual,
            transform.to_coupled(uncoupled_action),
            atol=2.0e-11,
            rtol=2.0e-12,
        )


def test_global_and_element_local_complete_actions_agree():
    _, _, system = _problem(2, 2, 4, 0)
    global_action = system.hamiltonian(radial_operator_mode="global")
    local_action = system.hamiltonian(radial_operator_mode="element-local")
    state = system.random_state(np.random.default_rng(991))

    np.testing.assert_allclose(
        local_action.physical_action(state),
        global_action.physical_action(state),
        atol=2.0e-11,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        local_action.coefficient_diagonal(),
        global_action.coefficient_diagonal(),
        atol=2.0e-13,
        rtol=2.0e-13,
    )


@pytest.mark.parametrize("M_tot", [-1, 0, 1])
def test_physical_field_free_action_is_weighted_hermitian(M_tot):
    _, _, system = _problem(2, 2, 4, M_tot)
    hamiltonian = system.hamiltonian()
    rng = np.random.default_rng(703 + M_tot)
    left = system.random_state(rng)
    right = system.random_state(rng)

    left_right = system.inner_product(
        left,
        hamiltonian.physical_action(right),
    )
    right_left = system.inner_product(
        hamiltonian.physical_action(left),
        right,
    )
    assert left_right == pytest.approx(
        right_left,
        abs=3.0e-11,
        rel=2.0e-12,
    )


def test_cap_is_nonhermitian_but_is_excluded_from_physical_energy():
    _, _, system = _problem(1, 1, 2, 0)
    cap1 = -0.15j * (system.grid1.r / system.grid1.r[-1]) ** 3
    cap2 = -0.08j * (system.grid2.r / system.grid2.r[-1]) ** 3
    boundaries = TwoRadialAbsorbingBoundaries(
        system,
        cap1=cap1,
        cap2=cap2,
    )
    hamiltonian = system.hamiltonian(absorbing_boundaries=boundaries)
    state = system.random_state(np.random.default_rng(51), normalize=True)

    physical = hamiltonian.expectation(state, normalize=True)
    propagation = hamiltonian.expectation(
        state,
        normalize=True,
        include_cap=True,
    )

    assert hamiltonian.has_cap
    assert abs(physical.imag) < 2.0e-12
    assert propagation.imag < -1.0e-6
    assert hamiltonian.energy(state) == pytest.approx(
        physical.real,
        abs=2.0e-12,
    )
    np.testing.assert_allclose(
        hamiltonian(state),
        hamiltonian.physical_action(state) + boundaries.cap_action(state),
    )


def test_components_sum_to_physical_and_propagation_actions():
    _, _, system = _problem(1, 1, 2, 0)
    boundaries = TwoRadialAbsorbingBoundaries(
        system,
        cap2=-0.04j * np.ones(system.grid2.n_points),
    )
    hamiltonian = system.hamiltonian(absorbing_boundaries=boundaries)
    state = system.random_state(np.random.default_rng(72))
    components = hamiltonian.components(state)

    physical = components.coordinate1 + components.coordinate2
    physical += components.coulomb
    np.testing.assert_allclose(physical, hamiltonian.physical_action(state))
    np.testing.assert_allclose(
        physical + components.cap,
        hamiltonian(state),
    )


def test_normalized_energy_is_invariant_to_state_scaling():
    _, _, system = _problem(1, 1, 2, 0)
    hamiltonian = system.hamiltonian()
    state = system.random_state(np.random.default_rng(31))

    assert hamiltonian.energy(3.0j * state) == pytest.approx(
        hamiltonian.energy(state),
        abs=3.0e-12,
    )
    assert hamiltonian.expectation(state) == pytest.approx(
        system.inner_product(state, hamiltonian.physical_action(state))
    )
    with pytest.raises(ValueError, match="positive norm"):
        hamiltonian.energy(np.zeros(system.psi_shape))


def test_coefficient_diagonal_matches_explicit_operator_matrix():
    _, _, system = _problem(1, 1, 2, 0)
    boundaries = TwoRadialAbsorbingBoundaries(
        system,
        cap1=-0.03j * np.ones(system.grid1.n_points),
    )
    hamiltonian = system.hamiltonian(
        radial_operator_mode="global",
        absorbing_boundaries=boundaries,
    )
    physical_matrix = _operator_matrix(
        hamiltonian.physical_action,
        system.psi_shape,
    )
    propagation_matrix = _operator_matrix(hamiltonian, system.psi_shape)

    np.testing.assert_allclose(
        hamiltonian.coefficient_diagonal(),
        np.diag(physical_matrix).reshape(system.psi_shape),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        hamiltonian.coefficient_diagonal(include_cap=True),
        np.diag(propagation_matrix).reshape(system.psi_shape),
        atol=2.0e-13,
    )


def test_davidson_diagonal_drives_generic_preconditioner():
    _, _, system = _problem(1, 1, 2, 0)
    hamiltonian = system.hamiltonian()
    diagonal = hamiltonian.davidson_diagonal(include_coulomb=True)
    preconditioner = DiagonalDavidsonPreconditioner(
        diagonal,
        relative_floor=0.0,
        absolute_floor=1.0e-12,
    )
    residual = system.random_state(np.random.default_rng(411))
    theta = -1.4

    np.testing.assert_allclose(
        preconditioner(residual, theta),
        -residual / (diagonal - theta),
        atol=2.0e-14,
    )


def test_complete_action_satisfies_sia_callable_contract():
    _, _, system = _problem(0, 0, 0, 0)
    hamiltonian = system.hamiltonian()
    initial = system.random_state(np.random.default_rng(44), normalize=True)
    propagator = RealTimePropagator(
        final_time=0.002,
        time_step=0.001,
        integrator="sia",
        max_krylov_dimension=20,
        krylov_rtol=1.0e-11,
        progress_interval=None,
    )

    result = propagator(
        initial,
        lambda state, time: hamiltonian(state),
        system.inner_product,
    )

    assert result.state.shape == system.psi_shape
    assert result.norm == pytest.approx(result.initial_norm, abs=2.0e-10)


def test_fixed_L_operator_and_spectrum_are_M_independent():
    matrices = []
    eigenvalues = []
    for M_tot in (0, 1):
        _, basis, system = _problem(1, 1, 2, M_tot)
        hamiltonian = system.hamiltonian(radial_operator_mode="global")
        full_matrix = _operator_matrix(
            hamiltonian.physical_action,
            system.psi_shape,
        )
        channels = [
            channel
            for channel, (_, _, L, _) in enumerate(basis.states)
            if L == 1
        ]
        radial_size = np.prod(system.radial_shape)
        indices = np.concatenate(
            [
                np.arange(channel * radial_size, (channel + 1) * radial_size)
                for channel in channels
            ]
        )
        block = full_matrix[np.ix_(indices, indices)]
        matrices.append(block)
        weights = np.tile(
            np.outer(system.grid1.weights, system.grid2.weights).ravel(),
            len(channels),
        )
        sqrt_weights = np.sqrt(weights)
        weighted = sqrt_weights[:, None] * block / sqrt_weights[None, :]
        eigenvalues.append(np.linalg.eigvalsh(weighted))

    np.testing.assert_allclose(matrices[0], matrices[1], atol=2.0e-14)
    np.testing.assert_allclose(eigenvalues[0], eigenvalues[1], atol=2.0e-12)


def test_complete_hamiltonian_validates_absorber_ownership_and_options():
    _, _, system = _problem(1, 1, 2, 0)
    _, _, other = _problem(1, 1, 2, 0)
    foreign = TwoRadialAbsorbingBoundaries(
        other,
        cap1=-0.1j * np.ones(other.grid1.n_points),
    )

    with pytest.raises(ValueError, match="different representation"):
        system.hamiltonian(absorbing_boundaries=foreign)

    hamiltonian = system.hamiltonian()
    with pytest.raises(TypeError, match="include_coulomb"):
        hamiltonian.coefficient_diagonal(include_coulomb=1)
    with pytest.raises(ValueError, match="reality_tolerance"):
        hamiltonian.energy(system.zeros(), reality_tolerance=-1.0)


def _operator_matrix(action, shape):
    dimension = int(np.prod(shape))
    result = np.empty((dimension, dimension), dtype=complex)
    for column in range(dimension):
        state = np.zeros(dimension, dtype=complex)
        state[column] = 1.0
        result[:, column] = action(state.reshape(shape)).ravel()
    return result
