import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialOneBodyPreconditioner,
    load_two_radial_state,
)


def _grids():
    electronic = FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4])
    internuclear = FEMDVRGridSpec([0.0, 1.5, 4.0], [4, 4])
    return electronic, internuclear


def _calculation(
    system="H2+",
    *,
    inversion_parity=1,
    pair_exchange_parity=1,
    symmetry_adapted_basis=True,
    initial_state="product",
    max_iterations=1,
):
    electronic, internuclear = _grids()
    return NonBornOppenheimerDiatomicIonGroundState(
        system,
        electronic_grid=electronic,
        internuclear_grid=internuclear,
        inversion_parity=inversion_parity,
        pair_exchange_parity=pair_exchange_parity,
        symmetry_adapted_basis=symmetry_adapted_basis,
        l_max=2,
        m_max=2,
        lambda_max=4,
        mu_max=4,
        codata_set="2022",
        initial_state=initial_state,
        time_step=0.05,
        max_iterations=max_iterations,
        energy_tolerance=1.0e-8,
        state_tolerance=1.0e-7,
        residual_tolerance=1.0e-7,
        linear_solver="bicgstab",
        linear_rtol=0.0,
        linear_atol=1.0e-11,
        use_preconditioner=False,
    )


@pytest.mark.parametrize(
    ("inversion", "exchange", "expected_l_e", "expected_l_R", "expected_l2"),
    [
        (1, 1, 0, 0, 0.0),
        (-1, 1, 1, 0, 2.0),
        (-1, -1, 0, 1, 2.0),
    ],
)
def test_product_guess_uses_lowest_allowed_pure_total_l_channel(
    inversion, exchange, expected_l_e, expected_l_R, expected_l2
):
    calculation = _calculation(
        inversion_parity=inversion,
        pair_exchange_parity=exchange,
    )
    state = calculation.initial_state
    populated = []
    for i, angular_e in enumerate(calculation.representation.basis1.states):
        for j, angular_R in enumerate(calculation.representation.basis2.states):
            if np.any(state[i, j] != 0.0):
                populated.append((angular_e, angular_R))

    assert populated == [((expected_l_e, 0), (expected_l_R, 0))]
    assert calculation.representation.inner_product(state, state) == pytest.approx(1)
    angular = calculation.angular_momentum_expectations(state)
    assert angular.total_l_squared == pytest.approx(expected_l2, abs=1.0e-12)
    assert angular.total_l_z == pytest.approx(0.0, abs=1.0e-12)


def test_symmetric_inversion_antisymmetric_pair_guess_is_coupled_total_l_zero():
    calculation = _calculation(inversion_parity=1, pair_exchange_parity=-1)
    state = calculation.initial_state
    basis_e = calculation.representation.basis1
    basis_R = calculation.representation.basis2

    blocks = {m: state[basis_e.index(1, m), basis_R.index(1, -m)] for m in (-1, 0, 1)}
    np.testing.assert_allclose(blocks[-1], blocks[1], atol=1.0e-14)
    np.testing.assert_allclose(blocks[0], -blocks[1], atol=1.0e-14)
    assert (
        sum(
            np.any(state[i, j] != 0.0)
            for i in range(basis_e.n_channels)
            for j in range(basis_R.n_channels)
        )
        == 3
    )

    angular = calculation.angular_momentum_expectations(state)
    assert angular.total_l_squared == pytest.approx(0.0, abs=1.0e-12)
    assert angular.total_l_z == pytest.approx(0.0, abs=1.0e-12)
    assert calculation.inversion_expectation(state) == pytest.approx(1.0)
    assert calculation.pair_exchange_expectation(state) == pytest.approx(-1.0)

    result = calculation.solve()
    assert result.angular_momentum.total_l_squared == pytest.approx(0.0, abs=1.0e-10)
    assert result.pair_exchange_expectation == pytest.approx(-1.0, abs=1.0e-12)


def test_homonuclear_system_uses_reduced_coordinate_bases_and_even_multipoles():
    calculation = _calculation(inversion_parity=-1, pair_exchange_parity=1)

    assert calculation.parameters.name == "H2+"
    assert calculation.parameters.particles == ("proton", "proton", "electron")
    assert calculation.parameters.has_identical_pair
    assert calculation.basis_is_reduced
    assert calculation.representation.basis1.coordinate_parity == -1
    assert calculation.representation.basis2.coordinate_parity == 1
    assert calculation.representation.multipoles.lambdas == (0, 2, 4)
    assert calculation.parameters.reference_center_scale == pytest.approx(-0.5)
    assert calculation.parameters.partner_center_scale == pytest.approx(0.5)
    assert calculation.parameters.electronic_reduced_mass < 1.0


def test_hd_rejects_pair_exchange_parity_and_uses_full_coordinate_bases():
    with pytest.raises(ValueError, match="not defined for HD\\+"):
        _calculation(
            "HD+",
            inversion_parity=1,
            pair_exchange_parity=1,
        )

    calculation = _calculation(
        "HD+",
        inversion_parity=1,
        pair_exchange_parity=None,
    )
    assert not calculation.parameters.has_identical_pair
    assert not calculation.basis_is_reduced
    assert calculation.representation.basis1.coordinate_parity is None
    assert calculation.representation.basis2.coordinate_parity is None
    assert calculation.representation.multipoles.lambdas == (0, 1, 2, 3, 4)
    assert calculation.pair_exchange_parity is None
    with pytest.raises(ValueError, match="not defined"):
        calculation.pair_exchange_expectation(calculation.initial_state)


def test_homonuclear_system_requires_an_explicit_pair_exchange_parity():
    with pytest.raises(ValueError, match="must be \\+1 or -1"):
        _calculation(pair_exchange_parity=None)


def test_angular_and_coulomb_cutoffs_are_validated():
    electronic, internuclear = _grids()
    common = dict(
        system="H2+",
        electronic_grid=electronic,
        internuclear_grid=internuclear,
        inversion_parity=1,
        pair_exchange_parity=1,
        l_max=2,
        m_max=2,
    )
    with pytest.raises(ValueError, match="m_max must equal l_max"):
        NonBornOppenheimerDiatomicIonGroundState(**(common | {"m_max": 1}))
    with pytest.raises(ValueError, match="lambda_max must not exceed"):
        NonBornOppenheimerDiatomicIonGroundState(
            **(common | {"lambda_max": 5, "mu_max": 4})
        )


@pytest.mark.parametrize(
    ("system", "particles"),
    [
        ("H2+", ("proton", "proton", "electron")),
        ("HD+", ("deuteron", "proton", "electron")),
        ("D2+", ("deuteron", "deuteron", "electron")),
    ],
)
def test_supported_systems_take_particle_data_from_matter_field_params(
    system, particles
):
    pair_parity = None if system == "HD+" else 1
    calculation = _calculation(
        system,
        inversion_parity=1,
        pair_exchange_parity=pair_parity,
    )
    assert calculation.parameters.particles == particles
    assert calculation.parameters.reference_charge == 1.0
    assert calculation.parameters.partner_charge == 1.0
    assert calculation.parameters.electron_charge == -1.0


@pytest.mark.parametrize("system", ["H2+", "HD+", "D2+"])
def test_velocity_couplings_are_derived_from_reduced_charges_and_masses(system):
    pair_parity = None if system == "HD+" else 1
    parameters = _calculation(
        system,
        inversion_parity=1,
        pair_exchange_parity=pair_parity,
    ).parameters

    assert parameters.internuclear_velocity_coupling == pytest.approx(
        parameters.internuclear_reduced_charge / parameters.internuclear_reduced_mass
    )
    assert parameters.electronic_velocity_coupling == pytest.approx(
        parameters.electronic_reduced_charge / parameters.electronic_reduced_mass
    )

    expected_internuclear = (
        parameters.partner_charge / parameters.partner_mass
        - parameters.reference_charge / parameters.reference_mass
    )
    pair_mass = parameters.reference_mass + parameters.partner_mass
    pair_charge = parameters.reference_charge + parameters.partner_charge
    expected_electronic = (
        parameters.electron_charge / parameters.electron_mass - pair_charge / pair_mass
    )
    assert parameters.internuclear_velocity_coupling == pytest.approx(
        expected_internuclear
    )
    assert parameters.electronic_velocity_coupling == pytest.approx(expected_electronic)


@pytest.mark.parametrize("system", ["H2+", "D2+"])
def test_homonuclear_internuclear_velocity_coupling_vanishes(system):
    parameters = _calculation(system).parameters

    assert parameters.internuclear_velocity_coupling == pytest.approx(0.0, abs=0.0)


def test_hd_has_a_nonzero_internuclear_velocity_coupling():
    parameters = _calculation(
        "HD+",
        pair_exchange_parity=None,
    ).parameters

    assert parameters.internuclear_velocity_coupling != pytest.approx(0.0)


def test_custom_array_is_projected_and_normalized_in_the_full_basis():
    template = _calculation(
        inversion_parity=1,
        pair_exchange_parity=1,
        symmetry_adapted_basis=False,
    )
    custom = np.ones(template.representation.psi_shape)
    calculation = _calculation(
        inversion_parity=1,
        pair_exchange_parity=1,
        symmetry_adapted_basis=False,
        initial_state=custom,
    )
    state = calculation.initial_state

    for i, (l_e, _) in enumerate(calculation.representation.basis1.states):
        for j, (l_R, _) in enumerate(calculation.representation.basis2.states):
            allowed = l_e % 2 == 0 and l_R % 2 == 0
            assert np.any(state[i, j]) == allowed
    assert calculation.representation.inner_product(state, state) == pytest.approx(1)


def test_one_body_preconditioner_matches_direct_block_inverse():
    calculation = _calculation()
    representation = calculation.representation
    time_step = 0.07
    preconditioner = TwoRadialOneBodyPreconditioner(representation, time_step)
    rng = np.random.default_rng(8)
    state = np.zeros(representation.psi_shape)
    state[0, 0] = rng.normal(
        size=(representation.grid1.n_points, representation.grid2.n_points)
    )

    actual = preconditioner(state)[0, 0]
    h1 = representation.h1[0]
    h2 = representation.h2[0]
    dimension = h1.shape[0] * h2.shape[0]
    matrix = np.eye(dimension) + 0.5 * time_step * (
        np.kron(h1, np.eye(h2.shape[0])) + np.kron(np.eye(h1.shape[0]), h2)
    )
    expected = np.linalg.solve(matrix, state[0, 0].ravel()).reshape(actual.shape)
    np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_high_level_solve_returns_physical_metadata_and_preserves_symmetry():
    calculation = _calculation(max_iterations=1)
    result = calculation.solve()

    assert result.parameters.name == "H2+"
    assert result.iterations == 1
    assert result.state.shape == calculation.representation.psi_shape
    assert calculation.representation.inner_product(result.state, result.state) == (
        pytest.approx(1.0)
    )
    assert result.inversion_expectation == pytest.approx(1.0, abs=1.0e-12)
    assert result.pair_exchange_expectation == pytest.approx(1.0, abs=1.0e-12)
    assert result.angular_momentum.total_l_squared == pytest.approx(0.0, abs=1e-10)


def test_diatomic_ground_state_result_saves_restart_metadata(tmp_path):
    result = _calculation().solve()
    path = tmp_path / "user-selected-diatomic-state"

    assert result.save(path) == path
    stored = load_two_radial_state(path)

    np.testing.assert_array_equal(stored.coefficients, result.state)
    assert stored.physical_state["system_family"] == "non_bo_diatomic_ion"
    assert stored.physical_state["system"] == "H2+"
    assert stored.physical_state["parameters"]["particles"] == [
        "proton",
        "proton",
        "electron",
    ]
    assert stored.physical_state["symmetry"]["pair_exchange_parity"] == 1
    assert stored.calculation["energy"] == pytest.approx(result.energy)
    assert stored.calculation["residual_norm"] == pytest.approx(
        result.residual_norm
    )
