import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    AxialCoordinateDipole,
    AxialDipoleCouplings,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    CoupledAngularBasisSpec,
    CoupledAngularTransform,
    FEMDVRGridSpec,
    TwoRadialSystemSpec,
    build_coupled_dipole_couplings,
    coupled_dipole_coefficient_direct,
    coupled_dipole_coefficient_reduced,
)
from grid_tdse.spherical.two_radial.diatomic import _diatomic_parameters


@pytest.mark.parametrize("coordinate", [1, 2])
@pytest.mark.parametrize("q", [-1, 0, 1])
def test_direct_and_reduced_dipole_coefficients_agree(coordinate, q):
    input_basis = CoupledAngularBasisSpec(2, 2, 4, M_tot=0)
    output_basis = CoupledAngularBasisSpec(2, 2, 4, M_tot=q)

    for output_state in output_basis.states:
        for input_state in input_basis.states:
            direct = coupled_dipole_coefficient_direct(
                output_state,
                input_state,
                coordinate,
                q=q,
            )
            reduced = coupled_dipole_coefficient_reduced(
                output_state,
                input_state,
                coordinate,
                q=q,
            )
            assert reduced == pytest.approx(
                direct,
                abs=2.0e-14,
                rel=2.0e-13,
            )


@pytest.mark.parametrize("coordinate", [1, 2])
@pytest.mark.parametrize("q", [-1, 0, 1])
def test_direct_and_reduced_sparse_dipole_tables_agree(coordinate, q):
    input_basis = CoupledAngularBasisSpec(2, 2, 4, M_tot=0)
    output_basis = CoupledAngularBasisSpec(2, 2, 4, M_tot=q)
    keywords = {
        "input_basis": input_basis,
        "output_basis": output_basis,
        "coordinate": coordinate,
        "coupling": -0.73,
        "q": q,
    }

    direct = build_coupled_dipole_couplings(
        **keywords,
        coefficient_method="direct",
    )
    reduced = build_coupled_dipole_couplings(**keywords)

    np.testing.assert_array_equal(reduced.output, direct.output)
    np.testing.assert_array_equal(reduced.offsets, direct.offsets)
    np.testing.assert_array_equal(reduced.input, direct.input)
    np.testing.assert_allclose(
        reduced.coefficient,
        direct.coefficient,
        atol=2.0e-14,
        rtol=2.0e-13,
    )


@pytest.mark.parametrize("coordinate", [1, 2])
def test_axial_dipole_table_is_hermitian_and_obeys_selection_rules(
    coordinate,
):
    basis = CoupledAngularBasisSpec(3, 3, 6, M_tot=0)
    table = build_coupled_dipole_couplings(
        basis,
        coordinate,
        coupling=1.0,
    )
    matrix = _dense_matrix(table)

    np.testing.assert_allclose(matrix, matrix.T, atol=2.0e-14)
    for output, input_ in _table_pairs(table):
        l1_out, l2_out, L_out, M_out = basis.states[output]
        l1_in, l2_in, L_in, M_in = basis.states[input_]
        assert M_out == M_in
        assert abs(L_out - L_in) <= 1 <= L_out + L_in
        assert (l1_out + l2_out + l1_in + l2_in) % 2 == 1
        if coordinate == 1:
            assert l2_out == l2_in
            assert abs(l1_out - l1_in) == 1
        else:
            assert l1_out == l1_in
            assert abs(l2_out - l2_in) == 1


@pytest.mark.parametrize("system_name", ["H2+", "HD+", "D2+"])
@pytest.mark.parametrize("coordinate", [1, 2])
def test_coupled_dipole_action_matches_transformed_uncoupled_action(
    system_name,
    coordinate,
):
    parameters = _diatomic_parameters(system_name, "2022")
    exchange_parity = 1 if parameters.has_identical_pair else None
    coupled_basis = CoupledAngularBasisSpec.for_diatomic_ion(
        2,
        2,
        4,
        has_identical_pair=parameters.has_identical_pair,
        M_tot=0,
        inversion_parity=None,
        pair_exchange_parity=exchange_parity,
    )
    transform = CoupledAngularTransform(coupled_basis)
    uncoupled = _uncoupled_system(exchange_parity)
    assert _uncoupled_states(uncoupled) == transform.uncoupled_states

    couplings = AxialDipoleCouplings.for_diatomic_ion(
        parameters,
        "length",
    )
    coupling = (
        couplings.coordinate1 if coordinate == 1 else couplings.coordinate2
    )
    table = build_coupled_dipole_couplings(
        coupled_basis,
        coordinate,
        coupling,
    )
    rng = np.random.default_rng(4100 + coordinate)
    radial_shape = (uncoupled.grid1.n_points, uncoupled.grid2.n_points)
    state = rng.normal(
        size=(coupled_basis.n_channels, *radial_shape)
    ) + 1j * rng.normal(
        size=(coupled_basis.n_channels, *radial_shape)
    )

    angular_action = table(state)
    if coordinate == 1:
        actual = -uncoupled.grid1.r[None, :, None] * angular_action
    else:
        actual = -uncoupled.grid2.r[None, None, :] * angular_action

    if coupling == 0.0:
        assert table.n_pairs == 0
        expected = np.zeros_like(actual)
    else:
        uncoupled_action = AxialCoordinateDipole(
            uncoupled,
            coordinate,
            gauge="length",
            coupling=coupling,
        )(transform.to_uncoupled(state))
        expected = transform.to_coupled(uncoupled_action)

    np.testing.assert_allclose(actual, expected, atol=3.0e-13, rtol=2.0e-13)


def test_dipole_builder_validates_projection_and_scalar_inputs():
    input_basis = CoupledAngularBasisSpec(1, 1, 2, M_tot=0)
    wrong_output = CoupledAngularBasisSpec(1, 1, 2, M_tot=0)

    with pytest.raises(ValueError, match="output M_tot"):
        build_coupled_dipole_couplings(
            input_basis,
            1,
            1.0,
            q=1,
            output_basis=wrong_output,
        )
    with pytest.raises(ValueError, match="coordinate"):
        build_coupled_dipole_couplings(input_basis, 3, 1.0)
    with pytest.raises(TypeError, match="coordinate must be an integer"):
        build_coupled_dipole_couplings(input_basis, True, 1.0)
    with pytest.raises(TypeError, match="coordinate must be an integer"):
        build_coupled_dipole_couplings(input_basis, 1.0, 1.0)
    with pytest.raises(ValueError, match="q must"):
        build_coupled_dipole_couplings(input_basis, 1, 1.0, q=2)
    with pytest.raises(TypeError, match="finite real scalar"):
        build_coupled_dipole_couplings(input_basis, 1, 1.0j)


def _uncoupled_system(exchange_parity):
    grid1 = FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4])
    grid2 = FEMDVRGridSpec([0.0, 1.5, 4.0], [4, 4])
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid1,
            AngularBasisSpec(2),
            -0.5,
        ),
        coordinate2=CoordinateSpec(
            grid2,
            AngularBasisSpec(2, coordinate_parity=exchange_parity),
            -0.004,
        ),
        multipoles=CoulombMultipoleSpec(0),
        coulomb_terms=(CoulombTerm(0.5, -1.0),),
        M_tot=0,
        radial_coulomb_method="analytic",
    ).build(radial_operator_mode="element-local")


def _uncoupled_states(system):
    return tuple(
        (*system.basis1.states[channel1], *system.basis2.states[channel2])
        for channel1, channel2 in system.angular_pairs.pairs
    )


def _table_pairs(table):
    for group, output in enumerate(table.output):
        for pair in range(table.offsets[group], table.offsets[group + 1]):
            yield int(output), int(table.input[pair])


def _dense_matrix(table):
    result = np.zeros(
        (table.n_output_channels, table.n_input_channels),
        dtype=table.coefficient.dtype,
    )
    for group, output in enumerate(table.output):
        for pair in range(table.offsets[group], table.offsets[group + 1]):
            result[output, table.input[pair]] = table.coefficient[pair]
    return result
