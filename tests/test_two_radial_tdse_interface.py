from dataclasses import asdict

import numpy as np
import pytest

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    NonBornOppenheimerDiatomicIonTDSE,
    TwoElectronAtomGroundState,
    TwoElectronAtomTDSE,
    load_two_radial_state,
    save_two_radial_state,
)


GRID = FEMDVRGridSpec([0.0, 2.0], [4])


def _atomic_initial_state():
    calculation = TwoElectronAtomGroundState(
        "He",
        grid=GRID,
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        max_iterations=1,
        use_preconditioner=False,
    )
    return calculation.initial_state


def test_atomic_interface_propagates_array_and_records_builtin_observables():
    initial = _atomic_initial_state()
    calculation = TwoElectronAtomTDSE(
        "He",
        grid=GRID,
        initial_state=initial,
        fields=(),
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        integrator="crank-nicolson",
        integrator_options={
            "solver": "gmres",
            "rtol": 0.0,
            "atol": 1.0e-11,
            "preconditioner": False,
        },
        time_step=0.01,
        final_time=0.01,
    )

    result = calculation.run(progress=False)

    assert result.n_steps == 1
    assert result.times.tolist() == pytest.approx([0.0, 0.01])
    assert result.samples["expec_z"].shape == (2,)
    assert result.norm_history.shape == (2,)
    assert calculation.integrator == "crank-nicolson"


@pytest.mark.parametrize(
    "preconditioner",
    ["spectral", "diagonal", "element-block-jacobi", "additive-schwarz"],
)
def test_atomic_interface_exposes_named_cn_preconditioners(preconditioner):
    initial = _atomic_initial_state()
    preconditioner_options = (
        {"overlap_elements": 0}
        if preconditioner == "additive-schwarz"
        else {}
    )
    calculation = TwoElectronAtomTDSE(
        "He",
        grid=GRID,
        initial_state=initial,
        l_max=1,
        M_tot=0,
        integrator="crank-nicolson",
        integrator_options={
            "preconditioner": preconditioner,
            "preconditioner_options": preconditioner_options,
        },
        final_time=0.001,
        time_step=0.001,
    )

    assert calculation.preconditioner.name == preconditioner
    assert calculation.preconditioner_name == preconditioner
    result = calculation.run(progress=False)
    assert result.n_steps == 1


def test_atomic_interface_accepts_axial_field_and_rejects_transverse_field():
    initial = _atomic_initial_state()

    class Field:
        gauge = "length"

        def __call__(self, time):
            return 0.01 * time

    calculation = TwoElectronAtomTDSE(
        "He",
        grid=GRID,
        initial_state=initial,
        fields=[(Field(), (0.0, 0.0, -2.0))],
        field_gauge="length",
        l_max=1,
        M_tot=0,
        final_time=0.01,
        time_step=0.01,
        integrator_options={"preconditioner": False},
    )
    assert calculation.dipole_interaction is not None

    with pytest.raises(NotImplementedError, match="z-polarized"):
        TwoElectronAtomTDSE(
            "He",
            grid=GRID,
            initial_state=initial,
            fields=[(Field(), (1.0, 0.0, 0.0))],
            l_max=1,
            M_tot=0,
            final_time=0.01,
            time_step=0.01,
            integrator_options={"preconditioner": False},
        )


def _archived_h2p_state(tmp_path, *, codata_set="2022"):
    ground_state = NonBornOppenheimerDiatomicIonGroundState(
        "H2+",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        inversion_parity=1,
        pair_exchange_parity=1,
        symmetry_adapted_basis=True,
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        codata_set=codata_set,
        max_iterations=1,
        use_preconditioner=False,
    )
    path = tmp_path / f"h2p-{codata_set}.npz"
    save_two_radial_state(
        path,
        ground_state.initial_state,
        ground_state.representation,
        physical_state={
            "system_family": "non_bo_diatomic_ion",
            "system": "H2+",
            "parameters": asdict(ground_state.parameters),
            "symmetry": {
                "inversion_parity": 1,
                "pair_exchange_parity": 1,
            },
        },
    )
    return load_two_radial_state(path)


def test_diatomic_interface_embeds_field_free_symmetry_archive(tmp_path):
    archived = _archived_h2p_state(tmp_path)
    calculation = NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=archived,
        pair_exchange_parity=1,
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        codata_set="2022",
        final_time=0.01,
        time_step=0.01,
        integrator_options={"preconditioner": False},
    )

    assert archived.basis1.coordinate_parity == 1
    assert calculation.representation.basis1.coordinate_parity is None
    assert calculation.representation.basis2.coordinate_parity == 1
    assert calculation.initial_state.shape == calculation.representation.psi_shape
    assert calculation.representation.inner_product(
        calculation.initial_state, calculation.initial_state
    ).real == pytest.approx(1.0)


def test_diatomic_tdse_exposes_gauge_covariant_velocity_energy_sampler(
    tmp_path,
):
    archived = _archived_h2p_state(tmp_path)

    class VectorPotential:
        gauge = "velocity"

        def __call__(self, time):
            return 0.2 * time

    calculation = NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=archived,
        pair_exchange_parity=1,
        fields=(VectorPotential(),),
        field_gauge="velocity",
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        final_time=0.01,
        time_step=0.01,
        integrator_options={"preconditioner": False},
    )
    time = 0.5
    state = calculation.initial_state
    snapshot = calculation.dynamics_observable_evaluator(state, time)
    amplitude = 0.2 * time
    physical_action = calculation.field_free_hamiltonian(state)
    physical_action += calculation.dipole_interaction(state, time)
    expected = calculation.representation.inner_product(
        state, physical_action
    ).real
    parameters = calculation.parameters
    expected += 0.5 * amplitude**2 * (
        parameters.electronic_reduced_charge**2
        / parameters.electronic_reduced_mass
        + parameters.internuclear_reduced_charge**2
        / parameters.internuclear_reduced_mass
    )

    assert snapshot.field_symbol == "A"
    assert snapshot.field_amplitude == pytest.approx(amplitude)
    assert snapshot.conditional_energy == pytest.approx(expected, abs=2.0e-12)


def test_diatomic_tdse_exposes_element_local_radial_action(tmp_path):
    archived = _archived_h2p_state(tmp_path)
    common = dict(
        system="H2+",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=archived,
        pair_exchange_parity=1,
        l_max=1,
        m_max=1,
        lambda_max=2,
        mu_max=2,
        M_tot=0,
        final_time=0.01,
        time_step=0.01,
        integrator_options={"preconditioner": False},
    )
    global_calculation = NonBornOppenheimerDiatomicIonTDSE(
        **common, radial_operator_mode="global"
    )
    local_calculation = NonBornOppenheimerDiatomicIonTDSE(
        **common, radial_operator_mode="element-local"
    )

    assert local_calculation.radial_operator_mode == "element-local"
    assert (
        local_calculation.dynamics_metadata["radial_operator_mode"]
        == "element-local"
    )
    np.testing.assert_allclose(
        local_calculation.field_free_hamiltonian(local_calculation.initial_state),
        global_calculation.field_free_hamiltonian(global_calculation.initial_state),
        atol=3.0e-12,
        rtol=3.0e-13,
    )


def test_diatomic_archive_requires_matching_codata_set(tmp_path):
    archived = _archived_h2p_state(tmp_path, codata_set="2022")
    with pytest.raises(ValueError, match="CODATA set"):
        NonBornOppenheimerDiatomicIonTDSE(
            "H2+",
            electronic_grid=GRID,
            internuclear_grid=GRID,
            initial_state=archived,
            pair_exchange_parity=1,
            l_max=1,
            M_tot=0,
            codata_set="2018",
            final_time=0.01,
            time_step=0.01,
            integrator_options={"preconditioner": False},
        )


def test_atomic_absorber_options_reject_conflicts_and_apply_cap():
    initial = _atomic_initial_state()
    with pytest.raises(ValueError, match="cannot both"):
        TwoElectronAtomTDSE(
            "He",
            grid=GRID,
            initial_state=initial,
            l_max=1,
            final_time=0.01,
            time_step=0.01,
            mask=object(),
            cap=object(),
        )
    cap = -0.2j * np.ones(2)
    calculation = TwoElectronAtomTDSE(
        "He",
        grid=GRID,
        initial_state=initial,
        l_max=1,
        final_time=0.01,
        time_step=0.01,
        cap=cap,
        integrator_options={"preconditioner": False},
    )
    state = calculation.initial_state
    expected_cap_action = 2.0 * cap[None, :, None] * state

    np.testing.assert_allclose(
        calculation.hamiltonian(state, 0.0)
        - calculation.field_free_hamiltonian(state),
        expected_cap_action,
    )


def test_atomic_mask_is_applied_after_each_real_time_step():
    initial = _atomic_initial_state()
    mask = np.array([1.0, 0.5])
    calculation = TwoElectronAtomTDSE(
        "He",
        grid=GRID,
        initial_state=initial,
        l_max=1,
        final_time=0.01,
        time_step=0.01,
        mask=mask,
        integrator_options={"preconditioner": False},
    )

    result = calculation.run(progress=False)

    assert result.norm < result.initial_norm
    assert calculation.dynamics_metadata["absorbing_boundaries"]["mask1"]


def test_diatomic_allows_mask_and_cap_on_different_coordinates(tmp_path):
    archived = _archived_h2p_state(tmp_path)
    calculation = NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=archived,
        pair_exchange_parity=1,
        l_max=1,
        M_tot=0,
        final_time=0.01,
        time_step=0.01,
        mask_r=np.array([1.0, 0.8]),
        cap_R=-0.1j * np.ones(2),
        integrator_options={"preconditioner": False},
    )

    assert calculation.absorbing_boundaries.mask1 is not None
    assert calculation.absorbing_boundaries.cap2 is not None


def test_atomic_interface_exposes_sil_with_krylov_diagnostics():
    initial = _atomic_initial_state()
    calculation = TwoElectronAtomTDSE(
        "He",
        grid=GRID,
        initial_state=initial,
        l_max=1,
        M_tot=0,
        integrator="sil",
        integrator_options={
            "max_krylov_dimension": 8,
            "rtol": 1.0e-9,
            "atol": 1.0e-12,
            "max_refinements": 2,
            "reorthogonalize": True,
            "breakdown_tolerance": 1.0e-14,
        },
        final_time=0.001,
        time_step=0.001,
    )

    result = calculation.run(progress=False)

    assert calculation.integrator == "sil"
    assert calculation.preconditioner is None
    assert result.krylov_operator_calls[0] > 0
    assert result.krylov_dimensions[0] > 0
    assert result.linear_iterations[0] == 0


def test_diatomic_interface_exposes_sia_with_a_cap(tmp_path):
    archived = _archived_h2p_state(tmp_path)
    calculation = NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        electronic_grid=GRID,
        internuclear_grid=GRID,
        initial_state=archived,
        pair_exchange_parity=1,
        l_max=1,
        M_tot=0,
        cap_R=-0.1j * np.ones(2),
        radial_operator_mode="element-local",
        integrator="sia",
        integrator_options={
            "max_krylov_dimension": 12,
            "rtol": 1.0e-9,
            "atol": 1.0e-12,
            "max_refinements": 2,
            "reorthogonalize": True,
            "breakdown_tolerance": 1.0e-14,
        },
        final_time=0.001,
        time_step=0.001,
    )

    result = calculation.run(progress=False)

    assert calculation.integrator == "sia"
    assert calculation.radial_operator_mode == "element-local"
    assert calculation.preconditioner is None
    assert result.krylov_operator_calls[0] > 0
    assert result.norm < result.initial_norm


def test_sil_rejects_a_cap():
    initial = _atomic_initial_state()

    with pytest.raises(ValueError, match="sil.*Hermitian.*CAP"):
        TwoElectronAtomTDSE(
            "He",
            grid=GRID,
            initial_state=initial,
            l_max=1,
            M_tot=0,
            cap=-0.1j * np.ones(2),
            integrator="sil",
            final_time=0.01,
            time_step=0.01,
        )


@pytest.mark.parametrize("integrator", ["sil", "sia"])
def test_krylov_interface_rejects_cn_only_options(integrator):
    initial = _atomic_initial_state()

    with pytest.raises(ValueError, match="unknown.*integrator options"):
        TwoElectronAtomTDSE(
            "He",
            grid=GRID,
            initial_state=initial,
            l_max=1,
            M_tot=0,
            integrator=integrator,
            integrator_options={"preconditioner": False},
            final_time=0.01,
            time_step=0.01,
        )
