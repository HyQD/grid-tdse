import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    CoupledAngularTransform,
    CoupledNonBornOppenheimerDiatomicIonGroundState,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    load_coupled_two_radial_state,
    load_two_radial_state,
    save_coupled_two_radial_state,
)


ELECTRONIC_GRID = FEMDVRGridSpec([0.0, 2.0, 5.0], [3, 3])
INTERNUCLEAR_GRID = FEMDVRGridSpec([0.0, 1.5, 4.0], [3, 3])


def _common(system):
    return {
        "system": system,
        "electronic_grid": ELECTRONIC_GRID,
        "internuclear_grid": INTERNUCLEAR_GRID,
        "inversion_parity": 1,
        "pair_exchange_parity": None if system == "HD+" else 1,
        "radial_coulomb_method": "poisson",
        "radial_operator_mode": "element-local",
        "max_iterations": 80,
        "residual_tolerance": 1.0e-8,
        "maximum_subspace_dimension": 12,
        "retained_ritz_vectors": 2,
        "progress_interval": None,
    }


def _coupled_calculation(system="H2+"):
    return NonBornOppenheimerDiatomicIonGroundState(
        wf_representation="coupled",
        l1_max=2,
        l2_max=2,
        L_max=0,
        M_tot=0,
        lambda_max=4,
        **_common(system),
    )


def _uncoupled_calculation(system="H2+"):
    return NonBornOppenheimerDiatomicIonGroundState(
        wf_representation="uncoupled",
        symmetry_adapted_basis=system != "HD+",
        l_max=2,
        m_max=2,
        lambda_max=4,
        mu_max=4,
        M_tot=0,
        solver="davidson",
        davidson_preconditioner="diagonal",
        sample_interval=None,
        **_common(system),
    )


def _uncoupled_transform_order(state, calculation, transform):
    representation = calculation.representation
    blocks = []
    for l1, m1, l2, m2 in transform.uncoupled_states:
        channel1 = representation.basis1.index(l1, m1)
        channel2 = representation.basis2.index(l2, m2)
        blocks.append(
            representation.angular_block(state, channel1, channel2)
        )
    return np.ascontiguousarray(blocks)


def _weighted_norm_squared(values, representation):
    return float(
        np.einsum(
            "cab,cab,a,b->",
            values.conj(),
            values,
            representation.grid1.weights,
            representation.grid2.weights,
            optimize=True,
        ).real
    )


@pytest.mark.parametrize(
    ("system", "expected_energy"),
    [
        ("H2+", -0.662012661915),
        ("HD+", -0.687526171998),
        ("D2+", -0.663133355959),
    ],
)
def test_coupled_davidson_ground_states_match_uncoupled_L_zero_sector(
    system,
    expected_energy,
):
    coupled_calculation = _coupled_calculation(system)
    uncoupled_calculation = _uncoupled_calculation(system)
    coupled = coupled_calculation.solve()
    uncoupled = uncoupled_calculation.solve()

    assert isinstance(
        coupled_calculation,
        CoupledNonBornOppenheimerDiatomicIonGroundState,
    )
    assert coupled.converged
    assert uncoupled.converged
    assert coupled.energy == pytest.approx(expected_energy, abs=2.0e-11)
    assert coupled.energy == pytest.approx(uncoupled.energy, abs=2.0e-12)
    assert coupled.norm == pytest.approx(1.0, abs=2.0e-12)
    assert coupled.residual_norm <= 1.0e-8
    assert coupled.observables.total_l_squared == pytest.approx(0.0, abs=1.0e-14)
    assert coupled.observables.total_l_z == pytest.approx(0.0, abs=1.0e-14)
    assert coupled.observables.inversion == pytest.approx(1.0, abs=1.0e-14)
    assert np.dot(
        coupled.radial_density_r,
        coupled.representation.grid1.weights,
    ) == pytest.approx(1.0, abs=2.0e-12)
    assert np.dot(
        coupled.radial_density_R,
        coupled.representation.grid2.weights,
    ) == pytest.approx(1.0, abs=2.0e-12)
    if system == "HD+":
        assert coupled.observables.pair_exchange is None
    else:
        assert coupled.observables.pair_exchange == pytest.approx(1.0)

    transform = CoupledAngularTransform(
        coupled.representation.angular_basis
    )
    uncoupled_state = _uncoupled_transform_order(
        uncoupled.state,
        uncoupled_calculation,
        transform,
    )
    projection = transform.project(uncoupled_state)
    assert projection.discarded_fraction < 2.0e-20
    discarded = uncoupled_state - transform.to_uncoupled(
        projection.coefficients
    )
    discarded_norm_squared = _weighted_norm_squared(
        discarded,
        coupled.representation,
    )
    assert discarded_norm_squared < 2.0e-20

    overlap = coupled.representation.inner_product(
        coupled.state,
        projection.coefficients,
    )
    aligned = projection.coefficients * np.exp(-1j * np.angle(overlap))
    difference = coupled.state - aligned
    assert coupled.representation.norm(difference) < 2.0e-8


def test_coupled_selector_and_options_are_explicit():
    calculation = _coupled_calculation()

    assert calculation.wf_representation == "coupled"
    assert calculation.representation.angular_basis.states == (
        (0, 0, 0, 0),
        (2, 2, 0, 0),
    )
    assert calculation.representation.multipoles.lambda_max == 4
    assert calculation.representation.multipoles.mu_max == 4
    assert calculation.preconditioner.name == "one-body-spectral"
    assert calculation.solver_options["preconditioner"] == (
        "one-body-spectral"
    )

    with pytest.raises(ValueError, match="wf_representation"):
        NonBornOppenheimerDiatomicIonGroundState(
            wf_representation="invalid",
            **_common("H2+"),
        )
    with pytest.raises(ValueError, match="requires solver='davidson'"):
        NonBornOppenheimerDiatomicIonGroundState(
            wf_representation="coupled",
            solver="lanczos",
            l1_max=2,
            l2_max=2,
            L_max=0,
            lambda_max=4,
            M_tot=0,
            **_common("H2+"),
        )


def test_coupled_ground_state_archive_round_trip_and_convergence_gate(tmp_path):
    result = _coupled_calculation().solve()
    path = tmp_path / "coupled-h2p.npz"

    returned = result.save(path)
    archived = load_coupled_two_radial_state(path)

    assert returned == path
    assert archived.solver_converged
    archived.require_solver_converged()
    np.testing.assert_array_equal(
        archived.state_for(result.representation),
        result.state,
    )
    assert archived.metadata["representation"]["wf_representation"] == "coupled"
    assert archived.calculation["solver"] == "davidson"
    assert archived.calculation["physical_basis_convergence_claimed"] is False
    assert archived.angular_basis == result.representation.angular_basis
    with pytest.raises(ValueError):
        load_two_radial_state(path)

    continued = NonBornOppenheimerDiatomicIonGroundState(
        wf_representation="coupled",
        l1_max=2,
        l2_max=2,
        L_max=0,
        M_tot=0,
        lambda_max=4,
        initial_state=path,
        **_common("H2+"),
    )
    overlap = continued.representation.inner_product(
        continued.initial_state,
        result.state,
    )
    assert abs(overlap) == pytest.approx(1.0, abs=2.0e-12)

    unsafe = tmp_path / "unconverged.npz"
    save_coupled_two_radial_state(
        unsafe,
        result.state,
        result.representation,
        physical_state=archived.physical_state,
        calculation={"converged": False},
    )
    unconverged = load_coupled_two_radial_state(unsafe)
    with pytest.raises(ValueError, match="unsafe override"):
        unconverged.require_solver_converged()
    unconverged.require_solver_converged(allow_unconverged=True)
