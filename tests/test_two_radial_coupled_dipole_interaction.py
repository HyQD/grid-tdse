import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    AxialDipoleCouplings,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    CoupledAngularBasisSpec,
    CoupledAngularTransform,
    CoupledAxialCoordinateDipole,
    CoupledTwoRadialAxialDipoleInteraction,
    FEMDVRGridSpec,
    TwoRadialAxialDipoleInteraction,
    TwoRadialSystemSpec,
)
from grid_tdse.spherical.two_radial.diatomic import _diatomic_parameters


@pytest.mark.parametrize("coordinate", [1, 2])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_coordinate_action_matches_explicit_sparse_matrix(coordinate, dtype):
    coupled, _, _ = _system_pair("HD+")
    state = _random_state(coupled, dtype=dtype)
    operator = CoupledAxialCoordinateDipole(
        coupled,
        coordinate,
        coupling=-0.73,
        coefficient_method="direct",
    )
    matrix = _dense_matrix(operator.couplings)
    angular_action = np.einsum("oi,irs->ors", matrix, state, optimize=True)
    radial = operator.radial
    expected = -angular_action * (
        radial[None, :, None]
        if coordinate == 1
        else radial[None, None, :]
    )

    np.testing.assert_allclose(operator(state), expected, atol=3.0e-14)


def test_coordinate_action_accumulates_into_output():
    coupled, _, _ = _system_pair("HD+")
    state = _random_state(coupled)
    operator = CoupledAxialCoordinateDipole(
        coupled,
        1,
        coupling=-0.81,
    )
    initial = np.full(coupled.psi_shape, 0.25 + 0.1j)
    expected = initial + operator(state)

    returned = operator(state, out=initial)

    assert returned is initial
    np.testing.assert_allclose(returned, expected, atol=3.0e-14)


@pytest.mark.parametrize(
    "couplings",
    [
        AxialDipoleCouplings(-0.91, 0.0),
        AxialDipoleCouplings(0.0, 0.14),
        AxialDipoleCouplings(-0.91, 0.14),
    ],
    ids=["electronic", "internuclear", "combined"],
)
def test_interaction_components_match_explicit_actions(couplings):
    coupled, _, _ = _system_pair("HD+")
    state = _random_state(coupled)
    fields = (lambda time: 0.2 + time, lambda time: -0.5 * time)
    interaction = CoupledTwoRadialAxialDipoleInteraction(
        coupled,
        gauge="length",
        fields=fields,
        couplings=couplings,
    )
    expected_spatial = np.zeros(coupled.psi_shape, dtype=complex)
    for coordinate, coupling in (
        (1, couplings.coordinate1),
        (2, couplings.coordinate2),
    ):
        if coupling != 0.0:
            expected_spatial += CoupledAxialCoordinateDipole(
                coupled,
                coordinate,
                coupling=coupling,
            )(state)

    time = 0.3
    amplitude = sum(field(time) for field in fields)
    assert interaction.field_amplitude(time) == pytest.approx(amplitude)
    np.testing.assert_allclose(
        interaction.spatial_action(state),
        expected_spatial,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        interaction(state, time),
        amplitude * expected_spatial,
        atol=3.0e-14,
    )


def test_zero_field_returns_zeros_without_modifying_input():
    coupled, _, _ = _system_pair("HD+")
    state = _random_state(coupled)
    original = state.copy()
    interaction = CoupledTwoRadialAxialDipoleInteraction(
        coupled,
        gauge="length",
        fields=lambda time: 0.0,
        couplings=AxialDipoleCouplings(-0.91, 0.14),
    )

    result = interaction(state, 0.4)

    assert result.shape == coupled.psi_shape
    assert np.iscomplexobj(result)
    assert np.all(result == 0.0)
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize(
    "couplings",
    [
        AxialDipoleCouplings(-0.91, 0.0),
        AxialDipoleCouplings(0.0, 0.14),
        AxialDipoleCouplings(-0.91, 0.14),
    ],
    ids=["electronic", "internuclear", "combined"],
)
def test_components_match_transformed_uncoupled_actions(couplings):
    coupled, uncoupled, transform = _system_pair("HD+")
    state = _random_state(coupled)
    fields = (lambda time: 0.23 + time, lambda time: -0.11 * time)
    coupled_interaction = CoupledTwoRadialAxialDipoleInteraction(
        coupled,
        gauge="length",
        fields=fields,
        couplings=couplings,
    )
    uncoupled_interaction = TwoRadialAxialDipoleInteraction(
        uncoupled,
        gauge="length",
        fields=fields,
        couplings=couplings,
        radial_operator_mode="element-local",
        dipole_kernel="row-wise",
    )
    time = 0.41

    actual = transform.to_uncoupled(coupled_interaction(state, time))
    expected = uncoupled_interaction(transform.to_uncoupled(state), time)

    np.testing.assert_allclose(actual, expected, atol=5.0e-13, rtol=3.0e-13)


@pytest.mark.parametrize("system_name", ["H2+", "HD+", "D2+"])
def test_diatomic_interaction_matches_transformed_uncoupled_action(system_name):
    coupled, uncoupled, transform = _system_pair(system_name)
    parameters = _diatomic_parameters(system_name, "2022")
    state = _random_state(coupled)
    uncoupled_state = transform.to_uncoupled(state)
    fields = (lambda time: 0.31 - 0.2 * time, lambda time: 0.07 * time)
    coupled_interaction = (
        CoupledTwoRadialAxialDipoleInteraction.for_diatomic_ion(
            coupled,
            parameters,
            gauge="length",
            fields=fields,
        )
    )
    uncoupled_interaction = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
        uncoupled,
        parameters,
        gauge="length",
        fields=fields,
        radial_operator_mode="element-local",
        dipole_kernel="row-wise",
    )
    time = 0.37

    expected = uncoupled_interaction(uncoupled_state, time)
    actual = transform.to_uncoupled(coupled_interaction(state, time))

    np.testing.assert_allclose(actual, expected, atol=5.0e-13, rtol=3.0e-13)
    coordinates = tuple(
        operator.coordinate for operator in coupled_interaction.coordinate_operators
    )
    if parameters.has_identical_pair:
        assert coordinates == (1,)
    else:
        assert coordinates == (1, 2)


def test_rejects_unsupported_gauge_and_field_gauge_mismatch():
    coupled, _, _ = _system_pair("HD+")
    couplings = AxialDipoleCouplings(-0.91, 0.14)

    with pytest.raises(ValueError, match="only length gauge"):
        CoupledTwoRadialAxialDipoleInteraction(
            coupled,
            gauge="velocity",
            fields=lambda time: 0.0,
            couplings=couplings,
        )

    velocity_field = _FieldProfile("velocity", 0.2)
    with pytest.raises(ValueError, match="uses gauge 'velocity'"):
        CoupledTwoRadialAxialDipoleInteraction(
            coupled,
            gauge="length",
            fields=velocity_field,
            couplings=couplings,
        )


def test_rejects_invalid_fields_and_states():
    coupled, _, _ = _system_pair("HD+")
    interaction = CoupledTwoRadialAxialDipoleInteraction(
        coupled,
        gauge="length",
        fields=lambda time: np.ones(2),
        couplings=AxialDipoleCouplings(-0.91, 0.14),
    )

    with pytest.raises(ValueError, match="finite real scalar"):
        interaction(_random_state(coupled), 0.0)
    with pytest.raises(ValueError, match="expected"):
        interaction.spatial_action(np.zeros((2, 3)))


def test_field_coupled_basis_must_retain_required_parities():
    specification = _system_spec(exchange_parity=None)
    inversion_restricted = specification.build_coupled(
        CoupledAngularBasisSpec(
            2,
            2,
            4,
            M_tot=0,
            inversion_parity=1,
        )
    )
    with pytest.raises(ValueError, match="both inversion parities"):
        CoupledAxialCoordinateDipole(
            inversion_restricted,
            1,
            coupling=-1.0,
        )

    exchange_restricted = _system_spec(exchange_parity=1).build_coupled(
        CoupledAngularBasisSpec(
            2,
            2,
            4,
            M_tot=0,
            pair_exchange_parity=1,
        )
    )
    with pytest.raises(ValueError, match="both pair-exchange parities"):
        CoupledAxialCoordinateDipole(
            exchange_restricted,
            2,
            coupling=0.2,
        )


def test_rejects_zero_couplings_and_wrong_coupling_type():
    coupled, _, _ = _system_pair("HD+")

    with pytest.raises(ValueError, match="at least one"):
        CoupledTwoRadialAxialDipoleInteraction(
            coupled,
            gauge="length",
            fields=lambda time: 1.0,
            couplings=AxialDipoleCouplings(0.0, 0.0),
        )
    with pytest.raises(TypeError, match="AxialDipoleCouplings"):
        CoupledTwoRadialAxialDipoleInteraction(
            coupled,
            gauge="length",
            fields=lambda time: 1.0,
            couplings=(-1.0, 0.1),
        )


class _FieldProfile:
    def __init__(self, gauge, value):
        self.gauge = gauge
        self.value = value

    def __call__(self, time):
        del time
        return self.value


def _system_pair(system_name):
    parameters = _diatomic_parameters(system_name, "2022")
    exchange_parity = 1 if parameters.has_identical_pair else None
    specification = _system_spec(exchange_parity=exchange_parity)
    coupled_basis = CoupledAngularBasisSpec.for_diatomic_ion(
        2,
        2,
        4,
        has_identical_pair=parameters.has_identical_pair,
        M_tot=0,
        pair_exchange_parity=exchange_parity,
    )
    coupled = specification.build_coupled(coupled_basis)
    uncoupled = specification.build(radial_operator_mode="element-local")
    transform = CoupledAngularTransform(coupled_basis)
    assert _uncoupled_states(uncoupled) == transform.uncoupled_states
    return coupled, uncoupled, transform


def _system_spec(*, exchange_parity):
    return TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            FEMDVRGridSpec([0.0, 2.0, 5.0], [4, 4]),
            AngularBasisSpec(2),
            -0.5,
        ),
        coordinate2=CoordinateSpec(
            FEMDVRGridSpec([0.0, 1.5, 4.0], [4, 4]),
            AngularBasisSpec(2, coordinate_parity=exchange_parity),
            -0.004,
        ),
        multipoles=CoulombMultipoleSpec(0),
        coulomb_terms=(CoulombTerm(0.5, -1.0),),
        M_tot=0,
        radial_coulomb_method="analytic",
    )


def _random_state(system, *, dtype=np.complex128):
    rng = np.random.default_rng(9231)
    values = rng.normal(size=system.psi_shape)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * rng.normal(size=system.psi_shape)
    return np.asarray(values, dtype=dtype)


def _dense_matrix(couplings):
    matrix = np.zeros(
        (couplings.n_output_channels, couplings.n_input_channels),
        dtype=couplings.coefficient.dtype,
    )
    for group, output in enumerate(couplings.output):
        start = couplings.offsets[group]
        stop = couplings.offsets[group + 1]
        matrix[output, couplings.input[start:stop]] = couplings.coefficient[
            start:stop
        ]
    return matrix


def _uncoupled_states(system):
    return tuple(
        (*system.basis1.states[channel1], *system.basis2.states[channel2])
        for channel1, channel2 in system.angular_pairs.pairs
    )
