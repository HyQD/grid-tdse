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


def test_absorber_options_reject_conflicts_and_reserve_future_support():
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
    with pytest.raises(NotImplementedError, match="not implemented"):
        TwoElectronAtomTDSE(
            "He",
            grid=GRID,
            initial_state=initial,
            l_max=1,
            final_time=0.01,
            time_step=0.01,
            mask=object(),
        )
