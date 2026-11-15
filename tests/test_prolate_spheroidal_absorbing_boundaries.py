import numpy as np
import pytest
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask

from grid_tdse.prolate_spheroidal import (
    CartesianRadiusCosineMask,
    OneElectronDiatomicSystem,
    ProlateSpheroidalAbsorbingBoundaries,
    ProlateSpheroidalCubicCAP,
    ProlateSpheroidalFixedMTimeDependentHamiltonian,
    ProlateSpheroidalOneElectronHamiltonian,
)
from grid_tdse.real_time import RealTimePropagator, load_real_time_checkpoint


@pytest.fixture
def hamiltonian():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    return ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 4.0],
        xi_points_per_element=6,
        eta_degree=7,
        m_max=1,
    )


def _normalized_outer_state(hamiltonian, m=0):
    state = np.zeros(hamiltonian.fixed_m_shape(m), dtype=complex)
    state[-2:, :] = 1.0
    norm = np.sqrt(hamiltonian.inner_product(state, state, m).real)
    return state / norm


def _sia(final_time=0.02, time_step=0.01):
    return RealTimePropagator(
        final_time=final_time,
        time_step=time_step,
        integrator="sia",
        max_krylov_dimension=30,
        krylov_rtol=1.0e-11,
        krylov_atol=1.0e-13,
        progress_interval=None,
    )


def test_zhang_mask_uses_cartesian_radius_and_grid_lib_profile(hamiltonian):
    representation = hamiltonian.representation
    profile = CartesianRadiusCosineMask(r0=2.0)
    boundaries = ProlateSpheroidalAbsorbingBoundaries(
        representation, mask=profile
    )
    radius = representation.grid.cartesian_radius()
    r_max = representation.grid.a * representation.xi_boundaries[-1]
    expected_full = RadialCosineMask(r0=2.0, n=8.0)(
        radius.reshape(-1), r_max
    ).reshape(representation.grid.shape)
    expected = representation.restrict_full_grid(expected_full, 0)

    np.testing.assert_allclose(boundaries.mask_values(0), expected)
    assert boundaries.mask_values(0).flags.writeable is False
    assert profile.n == 8.0
    metadata = boundaries.metadata["mask"]
    assert metadata["coordinate"] == "cartesian_radius_from_bond_midpoint"
    assert "Eq. (42)" in metadata["reference"]

    outer_row = expected_full[-2]
    assert np.ptp(outer_row) > 0.0


def test_q_based_cubic_cap_uses_grid_lib_profile(hamiltonian):
    representation = hamiltonian.representation
    profile = ProlateSpheroidalCubicCAP(alpha=0.5, eta=0.25)
    boundaries = ProlateSpheroidalAbsorbingBoundaries(
        representation, cap=profile
    )
    q = representation.grid.a * representation.grid.xi
    q_max = representation.grid.a * representation.xi_boundaries[-1]
    radial = CubicCAP(alpha=0.5, eta=0.25)(q, q_max)
    expected_full = np.broadcast_to(
        radial[:, None], representation.grid.shape
    )
    expected = representation.restrict_full_grid(expected_full, 0)

    np.testing.assert_allclose(boundaries.cap_values(), expected)
    np.testing.assert_allclose(
        boundaries.cap_values()[:, 0], boundaries.cap_values()[:, -1]
    )
    assert np.all(boundaries.cap_values().imag <= 0.0)
    assert boundaries.metadata["cap"]["coordinate"] == "q = a * xi"


def test_mask_and_cap_actions_preserve_shape_and_do_not_modify_state(
    hamiltonian,
):
    state = _normalized_outer_state(hamiltonian)
    original = state.copy()
    mask_boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        mask=CartesianRadiusCosineMask(r0=2.0),
    )
    cap_boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.5, eta=0.25),
    )

    masked = mask_boundaries.apply_mask(state, 0.0)
    cap_action = cap_boundaries.cap_action(state)

    assert masked.shape == state.shape
    assert cap_action.shape == state.shape
    assert hamiltonian.inner_product(masked, masked).real < 1.0
    assert np.linalg.norm(cap_action) > 0.0
    np.testing.assert_array_equal(state, original)


def test_mask_and_cap_are_mutually_exclusive_and_representation_bound(
    hamiltonian,
):
    with pytest.raises(ValueError, match="cannot both"):
        ProlateSpheroidalAbsorbingBoundaries(
            hamiltonian.representation,
            mask=CartesianRadiusCosineMask(r0=2.0),
            cap=ProlateSpheroidalCubicCAP(alpha=0.5),
        )

    other = ProlateSpheroidalOneElectronHamiltonian(
        hamiltonian.system,
        xi_boundaries=[1.0, 2.0, 4.0],
        xi_points_per_element=6,
        eta_degree=7,
        m_max=1,
    )
    foreign = ProlateSpheroidalAbsorbingBoundaries(
        other.representation,
        mask=CartesianRadiusCosineMask(r0=2.0),
    )
    with pytest.raises(ValueError, match="different representation"):
        ProlateSpheroidalFixedMTimeDependentHamiltonian(
            hamiltonian, absorbing_boundaries=foreign
        )


@pytest.mark.parametrize(
    "integrator", ["sil", "cf4-sil", "crank-nicolson"]
)
def test_cap_requires_arnoldi_integrator(hamiltonian, integrator):
    boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.5, eta=0.25),
    )
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=boundaries
    )
    propagator = RealTimePropagator(
        final_time=0.01,
        time_step=0.01,
        integrator=integrator,
        progress_interval=None,
    )

    with pytest.raises(ValueError, match="requires integrator"):
        action.propagate(propagator, _normalized_outer_state(hamiltonian))


def test_mask_and_cap_propagation_remove_outer_probability(hamiltonian):
    initial = _normalized_outer_state(hamiltonian)
    mask_boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        mask=CartesianRadiusCosineMask(r0=2.0),
    )
    cap_boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.5, eta=0.25),
    )
    mask_action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=mask_boundaries
    )
    cap_action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=cap_boundaries
    )

    masked = mask_action.propagate(_sia(), initial)
    capped = cap_action.propagate(_sia(), initial)

    assert masked.norm < masked.initial_norm
    assert capped.norm < capped.initial_norm


def test_absorber_onset_strength_and_application_frequency(hamiltonian):
    state = _normalized_outer_state(hamiltonian)
    early = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.4, eta=0.25),
    )
    late = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.7, eta=0.25),
    )
    strong = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.4, eta=0.5),
    )

    assert np.linalg.norm(early.cap_action(state)) > np.linalg.norm(
        late.cap_action(state)
    )
    np.testing.assert_allclose(
        strong.cap_action(state), 2.0 * early.cap_action(state)
    )

    mask = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        mask=CartesianRadiusCosineMask(r0=2.0),
    )
    five_applications = state.copy()
    ten_applications = state.copy()
    for _ in range(5):
        five_applications = mask.apply_mask(five_applications, 0.0)
    for _ in range(10):
        ten_applications = mask.apply_mask(ten_applications, 0.0)
    five_norm = hamiltonian.inner_product(
        five_applications, five_applications
    ).real
    ten_norm = hamiltonian.inner_product(
        ten_applications, ten_applications
    ).real
    assert ten_norm < five_norm


def test_cap_time_step_refinement_and_rolling_checkpoint(
    hamiltonian, tmp_path
):
    initial = _normalized_outer_state(hamiltonian)
    boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.5, eta=0.1),
    )
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian, absorbing_boundaries=boundaries
    )

    coarse = action.propagate(_sia(0.04, 0.02), initial)
    checkpoint = tmp_path / "prolate.chkpt"
    fine = action.propagate(
        _sia(0.04, 0.01),
        initial,
        checkpoint_path=checkpoint,
        checkpoint_interval=1,
        compatibility_metadata=action.archive_metadata(),
    )

    assert fine.norm == pytest.approx(coarse.norm, abs=2.0e-7)
    saved = load_real_time_checkpoint(checkpoint)
    np.testing.assert_allclose(saved.state, fine.state)
    assert saved.current_step == fine.n_steps
    assert list(tmp_path.iterdir()) == [checkpoint]


def _outgoing_packet(hamiltonian):
    q = hamiltonian.grid.a * hamiltonian.grid.xi[:-1]
    eta = hamiltonian.grid.eta
    state = np.exp(-((q[:, None] - 2.3) / 0.22) ** 2)
    state = state * np.exp(4.0j * q[:, None])
    state = state * np.exp(-(eta[None, :] / 0.55) ** 2)
    norm = np.sqrt(hamiltonian.inner_product(state, state).real)
    return state / norm


def _inner_probability(hamiltonian, state, q_boundary=2.2):
    q = hamiltonian.grid.a * hamiltonian.grid.xi[:-1]
    selected = np.broadcast_to(q[:, None] < q_boundary, state.shape)
    weights = hamiltonian.representation.metric_weights()
    return float(np.sum(np.abs(state[selected]) ** 2 * weights[selected]))


def test_cap_suppresses_hard_wall_reflection_and_tracks_larger_box():
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    small = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 3.0, 4.0],
        xi_points_per_element=8,
        eta_degree=7,
        m_max=0,
    )
    large = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=[1.0, 2.0, 3.0, 4.0, 5.0],
        xi_points_per_element=8,
        eta_degree=7,
        m_max=0,
    )
    cap_boundaries = ProlateSpheroidalAbsorbingBoundaries(
        small.representation,
        cap=ProlateSpheroidalCubicCAP(alpha=0.62, eta=1.0),
    )
    small_free = ProlateSpheroidalFixedMTimeDependentHamiltonian(small)
    small_cap = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        small, absorbing_boundaries=cap_boundaries
    )
    large_free = ProlateSpheroidalFixedMTimeDependentHamiltonian(large)
    propagator = _sia(final_time=0.7, time_step=0.02)

    reflected = small_free.propagate(propagator, _outgoing_packet(small))
    absorbed = small_cap.propagate(propagator, _outgoing_packet(small))
    enlarged = large_free.propagate(propagator, _outgoing_packet(large))
    reflected_inner = _inner_probability(small, reflected.state)
    absorbed_inner = _inner_probability(small, absorbed.state)
    enlarged_inner = _inner_probability(large, enlarged.state)

    assert absorbed_inner < 0.8 * reflected_inner
    assert abs(absorbed_inner - enlarged_inner) < abs(
        reflected_inner - enlarged_inner
    )
    assert absorbed.norm < reflected.norm
