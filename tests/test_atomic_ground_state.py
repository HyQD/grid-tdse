import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    RadialSincDVRGridSpec,
    TwoElectronAtomGroundState,
    load_two_radial_state,
)


def _grid():
    return FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4])


def _calculation(
    atom="He",
    *,
    inversion_parity=1,
    electron_exchange_parity=1,
    initial_state="product",
    max_iterations=1,
    M_tot=None,
):
    return TwoElectronAtomGroundState(
        atom,
        grid=_grid(),
        inversion_parity=inversion_parity,
        electron_exchange_parity=electron_exchange_parity,
        l_max=2,
        m_max=2,
        lambda_max=4,
        mu_max=4,
        M_tot=M_tot,
        initial_state=initial_state,
        time_step=0.01,
        max_iterations=max_iterations,
        energy_tolerance=1.0e-8,
        state_tolerance=1.0e-7,
        residual_tolerance=1.0e-7,
        linear_solver="bicgstab",
        linear_rtol=0.0,
        linear_atol=1.0e-11,
        progress_interval=None,
        use_preconditioner=False,
    )


@pytest.mark.parametrize(
    ("atom", "nuclear_charge", "ionic_charge"),
    [("He", 2.0, 0.0), ("Be2+", 4.0, 2.0)],
)
def test_supported_atoms_have_infinite_mass_two_electron_parameters(
    atom, nuclear_charge, ionic_charge
):
    calculation = _calculation(atom)

    assert calculation.parameters.name == atom
    assert calculation.parameters.nuclear_charge == nuclear_charge
    assert calculation.parameters.ionic_charge == ionic_charge
    assert calculation.parameters.electron_count == 2
    assert calculation.parameters.electron_mass == 1.0
    assert calculation.parameters.electron_charge == -1.0
    assert calculation.parameters.infinite_nuclear_mass


def test_atomic_coordinates_share_grid_basis_and_one_body_hamiltonian():
    calculation = _calculation()
    representation = calculation.representation

    assert (
        calculation.specification.coordinate1 is calculation.specification.coordinate2
    )
    np.testing.assert_allclose(representation.grid1.r, representation.grid2.r)
    np.testing.assert_allclose(
        representation.grid1.weights, representation.grid2.weights
    )
    assert representation.basis1.states == representation.basis2.states
    np.testing.assert_allclose(representation.h1, representation.h2)
    assert representation.basis1.coordinate_parity is None
    assert representation.multipoles.lambdas == (0, 1, 2, 3, 4)
    assert representation.multipoles.mu_max == 4


@pytest.mark.parametrize(
    ("inversion", "exchange", "expected_l_squared", "multiplicity"),
    [
        (1, 1, 0.0, 1),
        (1, -1, 0.0, 3),
        (-1, 1, 2.0, 1),
        (-1, -1, 2.0, 3),
    ],
)
def test_product_guess_has_requested_symmetry_and_total_angular_momentum(
    inversion, exchange, expected_l_squared, multiplicity
):
    calculation = _calculation(
        inversion_parity=inversion,
        electron_exchange_parity=exchange,
    )
    state = calculation.initial_state

    assert calculation.representation.inner_product(state, state) == pytest.approx(1)
    assert calculation.inversion_expectation(state) == pytest.approx(inversion)
    assert calculation.electron_exchange_expectation(state) == pytest.approx(exchange)
    angular = calculation.angular_momentum_expectations(state)
    assert angular.total_l_squared == pytest.approx(expected_l_squared, abs=1.0e-12)
    assert angular.total_l_z == pytest.approx(0.0, abs=1.0e-12)
    assert calculation.spin_multiplicity == multiplicity


def test_even_triplet_guess_is_an_antisymmetric_one_s_two_s_product():
    calculation = _calculation(electron_exchange_parity=-1)
    state = calculation.initial_state
    s_channel = calculation.representation.basis1.index(0, 0)

    assert np.any(state[s_channel, s_channel] != 0.0)
    assert (
        sum(
            np.any(state[i, j] != 0.0)
            for i in range(calculation.representation.basis1.n_channels)
            for j in range(calculation.representation.basis2.n_channels)
        )
        == 1
    )
    np.testing.assert_allclose(
        state[s_channel, s_channel],
        -state[s_channel, s_channel].T,
        atol=1.0e-14,
    )


@pytest.mark.parametrize("exchange", [1, -1])
def test_odd_product_guess_combines_one_s_one_p_in_exchange_sector(exchange):
    calculation = _calculation(
        inversion_parity=-1,
        electron_exchange_parity=exchange,
    )
    state = calculation.initial_state
    basis = calculation.representation.basis1
    s_channel = basis.index(0, 0)
    p_channel = basis.index(1, 0)

    populated = {
        (basis.states[i], basis.states[j])
        for i in range(basis.n_channels)
        for j in range(basis.n_channels)
        if np.any(state[i, j] != 0.0)
    }
    assert populated == {((0, 0), (1, 0)), ((1, 0), (0, 0))}
    np.testing.assert_allclose(
        state[p_channel, s_channel],
        exchange * state[s_channel, p_channel].T,
        atol=1.0e-14,
    )


def test_custom_array_is_projected_and_normalized():
    template = _calculation()
    rng = np.random.default_rng(12)
    custom = rng.normal(size=template.representation.psi_shape)
    calculation = _calculation(
        inversion_parity=-1,
        electron_exchange_parity=-1,
        initial_state=custom,
    )
    state = calculation.initial_state

    assert calculation.representation.inner_product(state, state) == pytest.approx(1)
    assert calculation.inversion_expectation(state) == pytest.approx(-1)
    assert calculation.electron_exchange_expectation(state) == pytest.approx(-1)


def test_zero_custom_array_is_rejected_after_symmetry_projection():
    template = _calculation()
    with pytest.raises(ValueError, match="zero norm after projection"):
        _calculation(initial_state=np.zeros(template.representation.psi_shape))


def test_hamiltonian_preserves_inversion_and_electron_exchange_symmetry():
    calculation = _calculation(
        inversion_parity=-1,
        electron_exchange_parity=-1,
    )
    rng = np.random.default_rng(5)
    state = calculation.project_symmetry(
        rng.normal(size=calculation.representation.psi_shape)
    )
    action = calculation.hamiltonian(state)

    assert calculation.inversion_expectation(action) == pytest.approx(-1, abs=1.0e-12)
    assert calculation.electron_exchange_expectation(action) == pytest.approx(
        -1, abs=1.0e-12
    )


def test_one_itp_step_returns_metadata_and_preserves_symmetry():
    calculation = _calculation()
    result = calculation.solve()

    assert result.parameters.name == "He"
    assert result.iterations == 1
    assert result.state.shape == calculation.representation.psi_shape
    assert calculation.representation.inner_product(result.state, result.state) == (
        pytest.approx(1.0)
    )
    assert result.inversion_expectation == pytest.approx(1.0, abs=1.0e-12)
    assert result.electron_exchange_expectation == pytest.approx(1.0, abs=1.0e-12)
    assert result.angular_momentum.total_l_squared == pytest.approx(0.0, abs=1.0e-10)
    assert result.spin_multiplicity == 1


def test_atomic_ground_state_result_saves_restart_metadata(tmp_path):
    result = _calculation().solve()
    path = tmp_path / "user-selected-atomic-state"

    assert result.save(path) == path
    stored = load_two_radial_state(path)

    np.testing.assert_array_equal(stored.coefficients, result.state)
    assert stored.physical_state["system_family"] == "two_electron_atom"
    assert stored.physical_state["system"] == "He"
    assert stored.physical_state["parameters"]["nuclear_charge"] == 2.0
    assert stored.physical_state["symmetry"]["spin_multiplicity"] == 1
    assert stored.calculation["energy"] == pytest.approx(result.energy)
    assert stored.calculation["radial_operator_mode"] == "global"
    assert stored.calculation["residual_norm"] == pytest.approx(
        result.residual_norm
    )


def test_compact_M_tot_atomic_calculation_matches_full_representation():
    full = _calculation(M_tot=None)
    compact = _calculation(M_tot=0)

    assert full.representation.psi_shape[0:2] == (9, 9)
    assert compact.representation.psi_shape[0] == 19
    np.testing.assert_allclose(
        compact.representation.unpack_state(compact.initial_state),
        full.initial_state,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        compact.hamiltonian(compact.initial_state),
        compact.representation.pack_state(full.hamiltonian(full.initial_state)),
        atol=3.0e-13,
        rtol=3.0e-13,
    )

    result = compact.solve()
    assert result.state.shape == compact.representation.psi_shape
    assert result.angular_momentum.total_l_z == pytest.approx(0.0, abs=1.0e-14)
    assert result.inversion_expectation == pytest.approx(1.0, abs=1.0e-12)
    assert result.electron_exchange_expectation == pytest.approx(1.0, abs=1.0e-12)


def test_product_initial_state_rejects_nonzero_M_tot():
    with pytest.raises(ValueError, match="product initial state has M_tot=0"):
        _calculation(M_tot=1)


def test_custom_initial_state_can_select_a_nonzero_M_tot_sector():
    full = _calculation()
    rng = np.random.default_rng(29)
    custom = rng.normal(size=full.representation.psi_shape)

    calculation = _calculation(M_tot=1, initial_state=custom)

    assert calculation.representation.M_tot == 1
    assert calculation.angular_momentum_expectations(
        calculation.initial_state
    ).total_l_z == pytest.approx(1.0, abs=1.0e-13)


def test_atomic_interface_validates_system_grid_symmetry_and_cutoffs():
    common = dict(
        atom="He",
        grid=_grid(),
        l_max=2,
        m_max=2,
        lambda_max=4,
        mu_max=4,
        progress_interval=None,
    )

    with pytest.raises(ValueError, match="unsupported two-electron atom"):
        TwoElectronAtomGroundState(**(common | {"atom": "Li+"}))
    with pytest.raises(TypeError, match="FEMDVRGridSpec"):
        TwoElectronAtomGroundState(
            **(
                common
                | {
                    "grid": RadialSincDVRGridSpec(r_max=5.0, n_points=8),
                }
            )
        )
    with pytest.raises(ValueError, match="inversion_parity must be"):
        TwoElectronAtomGroundState(**(common | {"inversion_parity": 0}))
    with pytest.raises(ValueError, match="electron_exchange_parity must be"):
        TwoElectronAtomGroundState(**(common | {"electron_exchange_parity": 0}))
    with pytest.raises(ValueError, match="m_max must equal l_max"):
        TwoElectronAtomGroundState(**(common | {"m_max": 1}))
    with pytest.raises(ValueError, match="lambda_max must not exceed"):
        TwoElectronAtomGroundState(**(common | {"lambda_max": 5}))
    with pytest.raises(ValueError, match="mu_max must not exceed lambda_max"):
        TwoElectronAtomGroundState(**(common | {"mu_max": 5}))
