import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    CoupledAngularBasisSpec,
    CoupledAngularTransform,
    CoupledTwoRadialOneBodyHamiltonian,
    FEMDVRGridSpec,
    RadialSincDVRGridSpec,
    TwoRadialSystemSpec,
    angular_radial_potential,
    radial_potential,
)


def _specification(l1_max, l2_max, M_tot, *, grid_family="femdvr"):
    if grid_family == "femdvr":
        grid1 = FEMDVRGridSpec([0.0, 0.7, 2.0, 4.5], [4, 5, 6])
        grid2 = FEMDVRGridSpec([0.0, 0.4, 1.6, 3.5], [5, 4, 6])
    else:
        grid1 = RadialSincDVRGridSpec(5.0, 7)
        grid2 = RadialSincDVRGridSpec(4.0, 6)
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=grid1,
            basis=AngularBasisSpec(l1_max),
            kinetic_prefactor=-0.7,
            one_body_terms=(radial_potential(lambda r: -1.3 / r),),
        ),
        coordinate2=CoordinateSpec(
            grid=grid2,
            basis=AngularBasisSpec(l2_max),
            kinetic_prefactor=-0.004,
            one_body_terms=(radial_potential(lambda r: 0.8 / r),),
        ),
        multipoles=CoulombMultipoleSpec(0, 0),
        coulomb_terms=(CoulombTerm(0.5, -1.0),),
        M_tot=M_tot,
    )


def _coupled_problem(l1_max, l2_max, M_tot, *, grid_family="femdvr"):
    specification = _specification(
        l1_max,
        l2_max,
        M_tot,
        grid_family=grid_family,
    )
    basis = CoupledAngularBasisSpec(
        l1_max,
        l2_max,
        l1_max + l2_max,
        M_tot=M_tot,
    )
    return specification, basis, specification.build_coupled(basis)


def _uncoupled_term_actions(
    specification,
    system,
    state,
    *,
    radial_operator_mode,
):
    pairs = system.angular_pairs.pairs
    l1 = np.asarray(
        [system.basis1.states[channel1][0] for channel1, _ in pairs]
    )
    l2 = np.asarray(
        [system.basis2.states[channel2][0] for _, channel2 in pairs]
    )
    if radial_operator_mode == "global":
        kinetic1 = np.matmul(
            np.asarray(
                [
                    specification.coordinate1.kinetic_prefactor
                    * system.grid1.second_derivative(int(value))
                    for value in l1
                ]
            ),
            state,
        )
        kinetic2 = np.matmul(
            state,
            np.asarray(
                [
                    specification.coordinate2.kinetic_prefactor
                    * system.grid2.second_derivative(int(value))
                    for value in l2
                ]
            ).transpose(0, 2, 1),
        )
    else:
        kinetic1 = specification.coordinate1.kinetic_prefactor * (
            system.grid1.apply_local_second_derivative(state, axis=-2)
        )
        kinetic2 = specification.coordinate2.kinetic_prefactor * (
            system.grid2.apply_local_second_derivative(state, axis=-1)
        )

    centrifugal1 = (
        -specification.coordinate1.kinetic_prefactor
        * l1[:, None, None]
        * (l1[:, None, None] + 1)
        / system.grid1.r[None, :, None] ** 2
        * state
    )
    centrifugal2 = (
        -specification.coordinate2.kinetic_prefactor
        * l2[:, None, None]
        * (l2[:, None, None] + 1)
        / system.grid2.r[None, None, :] ** 2
        * state
    )
    potential1_values = sum(
        (
            term.diagonal(0, 0, system.grid1)
            for term in specification.coordinate1.one_body_terms
        ),
        start=np.zeros(system.grid1.n_points),
    )
    potential2_values = sum(
        (
            term.diagonal(0, 0, system.grid2)
            for term in specification.coordinate2.one_body_terms
        ),
        start=np.zeros(system.grid2.n_points),
    )
    return {
        "apply_kinetic_coordinate1": kinetic1,
        "apply_kinetic_coordinate2": kinetic2,
        "apply_centrifugal_coordinate1": centrifugal1,
        "apply_centrifugal_coordinate2": centrifugal2,
        "apply_radial_potential_coordinate1": (
            potential1_values[None, :, None] * state
        ),
        "apply_radial_potential_coordinate2": (
            potential2_values[None, None, :] * state
        ),
    }


@pytest.mark.parametrize(
    ("l1_max", "l2_max", "M_tot"),
    [(1, 1, 0), (2, 1, 1), (1, 2, -1)],
)
def test_global_and_element_local_actions_match(l1_max, l2_max, M_tot):
    _, _, system = _coupled_problem(l1_max, l2_max, M_tot)
    global_action = system.one_body_hamiltonian(radial_operator_mode="global")
    local_action = system.one_body_hamiltonian(
        radial_operator_mode="element-local"
    )
    rng = np.random.default_rng(1400 + l1_max + 10 * l2_max)
    state = rng.normal(size=system.psi_shape) + 1j * rng.normal(
        size=system.psi_shape
    )

    for method in (
        "apply_kinetic_coordinate1",
        "apply_kinetic_coordinate2",
        "apply_centrifugal_coordinate1",
        "apply_centrifugal_coordinate2",
        "apply_radial_potential_coordinate1",
        "apply_radial_potential_coordinate2",
        "apply_coordinate1",
        "apply_coordinate2",
    ):
        np.testing.assert_allclose(
            getattr(local_action, method)(state),
            getattr(global_action, method)(state),
            atol=8.0e-12,
            rtol=8.0e-13,
        )
    np.testing.assert_allclose(
        local_action(state),
        global_action(state),
        atol=1.2e-11,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("l1_max", "l2_max", "L_max", "M_tot"),
    [(1, 1, 2, 0), (2, 1, 3, 0), (2, 2, 4, 1), (2, 2, 1, 0)],
)
@pytest.mark.parametrize("radial_operator_mode", ["global", "element-local"])
def test_coupled_actions_match_transformed_uncoupled_oracle(
    l1_max,
    l2_max,
    L_max,
    M_tot,
    radial_operator_mode,
):
    specification = _specification(
        l1_max,
        l2_max,
        M_tot,
    )
    basis = CoupledAngularBasisSpec(
        l1_max,
        l2_max,
        L_max,
        M_tot=M_tot,
    )
    coupled_system = specification.build_coupled(basis)
    transform = CoupledAngularTransform(basis)
    coupled_action = coupled_system.one_body_hamiltonian(
        radial_operator_mode=radial_operator_mode
    )
    uncoupled_system = specification.build(
        radial_operator_mode=radial_operator_mode
    )
    uncoupled_action = uncoupled_system.hamiltonian(
        radial_operator_mode=radial_operator_mode
    )
    rng = np.random.default_rng(229 + l1_max + 10 * l2_max + M_tot)
    coupled_state = rng.normal(size=coupled_system.psi_shape) + 1j * rng.normal(
        size=coupled_system.psi_shape
    )
    uncoupled_state = transform.to_uncoupled(coupled_state)

    assert uncoupled_state.shape == uncoupled_system.psi_shape
    term_actions = _uncoupled_term_actions(
        specification,
        uncoupled_system,
        uncoupled_state,
        radial_operator_mode=radial_operator_mode,
    )
    for method, uncoupled_term in term_actions.items():
        np.testing.assert_allclose(
            getattr(coupled_action, method)(coupled_state),
            transform.to_coupled(uncoupled_term),
            atol=1.0e-11,
            rtol=1.0e-12,
        )
    for coupled_method, uncoupled_method in (
        ("apply_coordinate1", "apply_one_body_coordinate1"),
        ("apply_coordinate2", "apply_one_body_coordinate2"),
    ):
        expected = transform.to_coupled(
            getattr(uncoupled_action, uncoupled_method)(uncoupled_state)
        )
        np.testing.assert_allclose(
            getattr(coupled_action, coupled_method)(coupled_state),
            expected,
            atol=1.0e-11,
            rtol=1.0e-12,
        )


def test_coordinate_actions_are_sums_of_exposed_physical_terms():
    _, _, system = _coupled_problem(2, 2, 0)
    action = system.one_body_hamiltonian()
    state = system.random_state(np.random.default_rng(44))

    expected1 = action.apply_kinetic_coordinate1(state)
    expected1 += action.apply_centrifugal_coordinate1(state)
    expected1 += action.apply_radial_potential_coordinate1(state)
    expected2 = action.apply_kinetic_coordinate2(state)
    expected2 += action.apply_centrifugal_coordinate2(state)
    expected2 += action.apply_radial_potential_coordinate2(state)

    np.testing.assert_allclose(action.apply_coordinate1(state), expected1)
    np.testing.assert_allclose(action.apply_coordinate2(state), expected2)
    np.testing.assert_allclose(action(state), expected1 + expected2)


def test_centrifugal_terms_depend_on_l1_l2_but_not_L():
    _, _, system = _coupled_problem(2, 2, 0)
    action = system.one_body_hamiltonian()
    grouped = {}
    for channel, (l1, l2, _, _) in enumerate(system.angular_basis.states):
        grouped.setdefault((l1, l2), []).append(channel)

    checked = False
    for channels in grouped.values():
        if len(channels) < 2:
            continue
        checked = True
        for channel in channels[1:]:
            np.testing.assert_array_equal(
                action.centrifugal1[channel],
                action.centrifugal1[channels[0]],
            )
            np.testing.assert_array_equal(
                action.centrifugal2[channel],
                action.centrifugal2[channels[0]],
            )
    assert checked


def test_element_local_object_retains_no_dense_kinetic_matrices():
    _, _, system = _coupled_problem(2, 1, 0)
    local = system.one_body_hamiltonian()
    global_oracle = system.one_body_hamiltonian(radial_operator_mode="global")

    assert not local.has_global_radial_operators
    assert local._kinetic1 is None
    assert local._kinetic2_transpose is None
    assert global_oracle.has_global_radial_operators


def test_sinc_representation_supports_only_the_global_oracle():
    _, _, system = _coupled_problem(1, 1, 0, grid_family="sinc")
    with pytest.raises(ValueError, match="require FEMDVR"):
        system.one_body_hamiltonian()

    action = system.one_body_hamiltonian(radial_operator_mode="global")
    state = system.random_state(np.random.default_rng(19))
    assert np.isfinite(action(state)).all()


def test_angular_dependent_one_body_potential_is_rejected():
    specification = _specification(1, 1, 0)
    coordinate1 = CoordinateSpec(
        grid=specification.coordinate1.grid,
        basis=specification.coordinate1.basis,
        kinetic_prefactor=specification.coordinate1.kinetic_prefactor,
        one_body_terms=(
            angular_radial_potential(
                lambda l, m, r: (l + m + 1) / r,
            ),
        ),
    )
    incompatible = TwoRadialSystemSpec(
        coordinate1=coordinate1,
        coordinate2=specification.coordinate2,
        multipoles=specification.multipoles,
        coulomb_terms=specification.coulomb_terms,
        M_tot=0,
    ).build_coupled(CoupledAngularBasisSpec(1, 1, 2))

    with pytest.raises(TypeError, match="angular-independent RadialPotential"):
        incompatible.one_body_hamiltonian()


def test_one_body_action_validates_inputs_and_mode():
    _, _, system = _coupled_problem(1, 1, 0)
    with pytest.raises(TypeError, match="CoupledTwoRadialSystem"):
        CoupledTwoRadialOneBodyHamiltonian(object())
    with pytest.raises(ValueError, match="radial_operator_mode"):
        system.one_body_hamiltonian(radial_operator_mode="unknown")

    action = system.one_body_hamiltonian()
    with pytest.raises(ValueError, match="expected"):
        action(np.zeros((1, 2, 3)))
