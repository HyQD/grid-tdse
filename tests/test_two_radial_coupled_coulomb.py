import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    CoupledAngularBasisSpec,
    CoupledAngularTransform,
    CoupledCoulombCouplings,
    RadialSincDVRGridSpec,
    TwoRadialSystemSpec,
    build_coupled_coulomb_couplings,
    coupled_coulomb_coefficient_direct,
    coupled_coulomb_coefficient_reduced,
)


def _specification(
    l1_max,
    l2_max,
    M_tot,
    multipoles,
    terms,
):
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=RadialSincDVRGridSpec(5.0, 7),
            basis=AngularBasisSpec(l1_max),
            kinetic_prefactor=-0.7,
        ),
        coordinate2=CoordinateSpec(
            grid=RadialSincDVRGridSpec(4.0, 6),
            basis=AngularBasisSpec(l2_max),
            kinetic_prefactor=-0.004,
        ),
        multipoles=multipoles,
        coulomb_terms=terms,
        M_tot=M_tot,
        radial_coulomb_method="analytic",
    )


@pytest.mark.parametrize("M_tot", [-1, 0, 1])
def test_direct_and_reduced_coefficients_agree(M_tot):
    basis = CoupledAngularBasisSpec(2, 2, 4, M_tot=M_tot)

    for output_state in basis.states:
        for input_state in basis.states:
            for lambda_ in range(3):
                direct = coupled_coulomb_coefficient_direct(
                    output_state,
                    input_state,
                    lambda_,
                )
                reduced = coupled_coulomb_coefficient_reduced(
                    output_state,
                    input_state,
                    lambda_,
                )
                assert reduced == pytest.approx(
                    direct,
                    abs=2.0e-14,
                    rel=2.0e-13,
                )


def test_scalar_coefficient_preserves_L_M_and_is_independent_of_M():
    input_base = (1, 2)
    output_base = (2, 1)
    lambda_ = 1
    values = []
    for M in (-1, 0, 1):
        values.append(
            coupled_coulomb_coefficient_reduced(
                (*output_base, 1, M),
                (*input_base, 1, M),
                lambda_,
            )
        )

    np.testing.assert_allclose(values, values[0], atol=2.0e-15)
    assert (
        coupled_coulomb_coefficient_reduced(
            (2, 1, 2, 0),
            (1, 2, 1, 0),
            lambda_,
        )
        == 0.0
    )
    assert (
        coupled_coulomb_coefficient_direct(
            (2, 1, 1, 0),
            (1, 2, 1, 1),
            lambda_,
        )
        == 0.0
    )


@pytest.mark.parametrize("M_tot", [0, 1])
def test_direct_and_reduced_sparse_tables_agree_and_are_hermitian(M_tot):
    basis = CoupledAngularBasisSpec(2, 2, 4, M_tot=M_tot)
    multipoles = CoulombMultipoleSpec(3, 3)
    direct = build_coupled_coulomb_couplings(
        basis,
        multipoles,
        coefficient_method="direct",
    )
    reduced = build_coupled_coulomb_couplings(basis, multipoles)

    np.testing.assert_array_equal(reduced.output, direct.output)
    np.testing.assert_array_equal(reduced.offsets, direct.offsets)
    np.testing.assert_array_equal(reduced.input, direct.input)
    np.testing.assert_array_equal(reduced.radial_index, direct.radial_index)
    np.testing.assert_allclose(
        reduced.coefficient,
        direct.coefficient,
        atol=2.0e-14,
        rtol=2.0e-13,
    )

    matrices = _dense_multipole_matrices(reduced, len(multipoles.lambdas))
    for matrix in matrices:
        np.testing.assert_allclose(matrix, matrix.T, atol=2.0e-14)


@pytest.mark.parametrize(
    ("name", "terms", "lambda_parity", "M_tot"),
    [
        (
            "H2+",
            (CoulombTerm(-0.5, -1.0), CoulombTerm(0.5, -1.0)),
            1,
            0,
        ),
        (
            "D2+",
            (CoulombTerm(-0.5, -1.0), CoulombTerm(0.5, -1.0)),
            1,
            1,
        ),
        (
            "HD+",
            (CoulombTerm(-2.0 / 3.0, -1.0), CoulombTerm(1.0 / 3.0, -1.0)),
            None,
            0,
        ),
    ],
)
def test_complete_coupled_action_matches_transformed_uncoupled_action(
    name,
    terms,
    lambda_parity,
    M_tot,
):
    del name
    multipoles = CoulombMultipoleSpec(3, 3, lambda_parity=lambda_parity)
    specification = _specification(2, 2, M_tot, multipoles, terms)
    basis = CoupledAngularBasisSpec(2, 2, 4, M_tot=M_tot)
    transform = CoupledAngularTransform(basis)
    coupled_system = specification.build_coupled(basis)
    uncoupled_system = specification.build()
    coupled_action = coupled_system.coulomb_hamiltonian()
    uncoupled_action = uncoupled_system.hamiltonian()
    rng = np.random.default_rng(917 + M_tot)
    state = coupled_system.random_state(rng)

    expected = transform.to_coupled(
        uncoupled_action.apply_coulomb(transform.to_uncoupled(state))
    )
    actual = coupled_action(state)

    np.testing.assert_allclose(actual, expected, atol=3.0e-13, rtol=2.0e-13)


def test_truncated_L_coulomb_subspace_matches_projected_uncoupled_action():
    multipoles = CoulombMultipoleSpec(2, 2)
    terms = (
        CoulombTerm(-2.0 / 3.0, -1.0),
        CoulombTerm(1.0 / 3.0, -1.0),
    )
    specification = _specification(2, 2, 0, multipoles, terms)
    basis = CoupledAngularBasisSpec(2, 2, 1)
    transform = CoupledAngularTransform(basis)
    coupled_system = specification.build_coupled(basis)
    uncoupled_system = specification.build()
    state = coupled_system.random_state(np.random.default_rng(808))

    expected = transform.to_coupled(
        uncoupled_system.hamiltonian().apply_coulomb(
            transform.to_uncoupled(state)
        )
    )
    actual = coupled_system.coulomb_hamiltonian()(state)

    assert not transform.is_complete
    np.testing.assert_allclose(actual, expected, atol=3.0e-13, rtol=2.0e-13)


@pytest.mark.parametrize("name", ["H2+", "D2+"])
def test_homonuclear_radial_multipoles_remove_odd_lambda_exactly(name):
    del name
    multipoles = CoulombMultipoleSpec(3, 3)
    specification = _specification(
        1,
        1,
        0,
        multipoles,
        (CoulombTerm(-0.5, -1.0), CoulombTerm(0.5, -1.0)),
    )
    radial = specification.build_coupled(
        CoupledAngularBasisSpec(1, 1, 2)
    ).radial_multipoles

    for lambda_ in (1, 3):
        np.testing.assert_array_equal(
            radial.values[multipoles.radial_index(lambda_)],
            0.0,
        )


def test_heteronuclear_radial_multipoles_retain_odd_lambda():
    multipoles = CoulombMultipoleSpec(3, 3)
    specification = _specification(
        1,
        1,
        0,
        multipoles,
        (
            CoulombTerm(-2.0 / 3.0, -1.0),
            CoulombTerm(1.0 / 3.0, -1.0),
        ),
    )
    radial = specification.build_coupled(
        CoupledAngularBasisSpec(1, 1, 2)
    ).radial_multipoles

    assert np.any(radial.values[multipoles.radial_index(1)] != 0.0)


def test_sparse_transitions_preserve_scalar_and_selected_symmetries():
    basis = CoupledAngularBasisSpec(
        3,
        2,
        5,
        M_tot=0,
        inversion_parity=-1,
        pair_exchange_parity=1,
    )
    multipoles = CoulombMultipoleSpec(4, 4, lambda_parity=1)
    couplings = build_coupled_coulomb_couplings(basis, multipoles)

    for group, output in enumerate(couplings.output):
        output_state = basis.states[output]
        for pair in range(couplings.offsets[group], couplings.offsets[group + 1]):
            input_state = basis.states[couplings.input[pair]]
            lambda_ = multipoles.lambdas[couplings.radial_index[pair]]
            assert output_state[2:] == input_state[2:]
            assert (-1) ** (output_state[0] + output_state[1]) == (
                (-1) ** (input_state[0] + input_state[1])
            )
            assert (-1) ** output_state[1] == (-1) ** input_state[1]
            assert lambda_ % 2 == 0


def test_action_accumulates_into_output_and_validates_it():
    multipoles = CoulombMultipoleSpec(2, 2)
    specification = _specification(
        1,
        1,
        0,
        multipoles,
        (CoulombTerm(-0.5, -1.0), CoulombTerm(0.5, -1.0)),
    )
    system = specification.build_coupled(CoupledAngularBasisSpec(1, 1, 2))
    action = system.coulomb_hamiltonian()
    state = system.random_state(np.random.default_rng(501))
    initial = np.full(system.psi_shape, 0.125 + 0.25j)

    expected = initial + action(state)
    actual = action.apply(state, out=initial.copy())
    np.testing.assert_allclose(actual, expected, atol=2.0e-14)

    with pytest.raises(ValueError, match="out has shape"):
        action.apply(state, out=np.zeros((1, 1, 1)))
    with pytest.raises(TypeError, match="cannot accumulate"):
        action.apply(state, out=np.zeros(system.psi_shape, dtype=float))
    with pytest.raises(ValueError, match="C-contiguous"):
        action.apply(
            state,
            out=np.zeros(system.psi_shape, dtype=complex)[..., ::-1],
        )


def test_builder_validation_and_read_only_storage():
    basis = CoupledAngularBasisSpec(1, 1, 2)
    with pytest.raises(ValueError, match="mu_max"):
        build_coupled_coulomb_couplings(
            basis,
            CoulombMultipoleSpec(2, 1),
        )
    with pytest.raises(ValueError, match="coefficient_method"):
        build_coupled_coulomb_couplings(
            basis,
            CoulombMultipoleSpec(1, 1),
            coefficient_method="unknown",
        )
    with pytest.raises(ValueError, match="more than 1"):
        build_coupled_coulomb_couplings(
            basis,
            CoulombMultipoleSpec(1, 1),
            max_pairs=1,
        )

    couplings = build_coupled_coulomb_couplings(
        basis,
        CoulombMultipoleSpec(1, 1),
    )
    assert not couplings.coefficient.flags.writeable
    assert not couplings.offsets.flags.writeable


def test_coupling_container_rejects_inconsistent_arrays():
    with pytest.raises(ValueError, match="output-group"):
        CoupledCoulombCouplings(
            output=np.array([0]),
            offsets=np.array([0]),
            input=np.array([], dtype=int),
            radial_index=np.array([], dtype=int),
            coefficient=np.array([]),
            n_channels=1,
        )


def _dense_multipole_matrices(couplings, n_multipoles):
    matrices = np.zeros(
        (n_multipoles, couplings.n_channels, couplings.n_channels)
    )
    for group, output in enumerate(couplings.output):
        for pair in range(couplings.offsets[group], couplings.offsets[group + 1]):
            matrices[
                couplings.radial_index[pair],
                output,
                couplings.input[pair],
            ] = couplings.coefficient[pair]
    return matrices
