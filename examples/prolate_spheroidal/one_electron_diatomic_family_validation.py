"""Validate matched field-free and short dynamics for one-electron diatomics.

The three target ions use the same ``R=2`` geometry, prolate grid, axial
length-gauge pulse, time step, and propagator. The calculation checks
field-free residuals, homonuclear versus heteronuclear inversion parity,
unitary short dynamics, and the origin transformation of the total dipole
and wavefunction phase for these positively charged systems.

All quantities are in atomic units.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from electromagnetic_fields.dipole_fields import SineSquareLaser

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalFixedMTimeDependentHamiltonian,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
)
from grid_tdse.real_time import RealTimePropagator


@dataclass(frozen=True)
class ValidationCase:
    name: str
    nuclear_charge1: float
    nuclear_charge2: float
    reference_electronic_energy: float | None = None


CASES = (
    ValidationCase("H2+", 1.0, 1.0, -1.102634214494946),
    ValidationCase("LiH3+", 3.0, 1.0, -5.002042016849),
    ValidationCase("BH5+", 5.0, 1.0),
)

XI_BOUNDARIES = (1.0, 4.0, 14.0, 24.0)
XI_POINTS_PER_ELEMENT = 14
ETA_DEGREE = 16
HALF_BOND_LENGTH = 1.0
ORIGIN_Z = 0.25
TIME_STEP = 0.05


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "one_electron_diatomic_family_validation.json"
        ),
    )
    return parser.parse_args()


def metric_norm(hamiltonian, state):
    value = hamiltonian.inner_product(state, state)
    return float(np.sqrt(max(value.real, 0.0)))


def midpoint_integral(field, final_time, time_step):
    number_of_steps = int(np.ceil(final_time / time_step))
    actual_step = final_time / number_of_steps
    midpoints = (np.arange(number_of_steps) + 0.5) * actual_step
    return float(actual_step * np.sum(field(midpoints)))


def propagate(hamiltonian, ground_state, pulse, origin_z):
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=pulse,
        include_nuclear_scalar=True,
        origin_z=origin_z,
    )
    interaction = action.dipole_interaction
    propagator = RealTimePropagator(
        final_time=pulse.tprime,
        time_step=TIME_STEP,
        integrator="sia",
        max_krylov_dimension=30,
        krylov_rtol=1.0e-12,
        krylov_atol=1.0e-14,
        max_krylov_refinements=6,
        progress_interval=None,
    )
    result = action.propagate(
        propagator,
        ground_state,
        samplers={
            "electronic_dipole": lambda state, time: (
                interaction.electronic_dipole_expectation(state)
            ),
            "total_dipole": lambda state, time: (
                interaction.total_dipole_expectation(state)
            ),
        },
    )
    return action, result


def validate_case(case, pulse):
    system = OneElectronDiatomicSystem(
        nuclear_charge1=case.nuclear_charge1,
        nuclear_charge2=case.nuclear_charge2,
        half_bond_length=HALF_BOND_LENGTH,
        name=case.name,
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=XI_BOUNDARIES,
        xi_points_per_element=XI_POINTS_PER_ELEMENT,
        eta_degree=ETA_DEGREE,
        m_max=0,
    )
    eigenstate = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=1,
        solver="direct",
    )
    midpoint_action, midpoint = propagate(
        hamiltonian, eigenstate.orbital, pulse, 0.0
    )
    shifted_action, shifted = propagate(
        hamiltonian, eigenstate.orbital, pulse, ORIGIN_Z
    )

    field_area = midpoint_integral(pulse, pulse.tprime, TIME_STEP)
    predicted_phase = np.exp(
        -1j * system.net_charge * ORIGIN_Z * field_area
    )
    phase_error_state = shifted.state - predicted_phase * midpoint.state
    phase_aligned_state_error = metric_norm(
        hamiltonian, phase_error_state
    )
    corrected_total_dipole = (
        shifted.samples["total_dipole"]
        + system.net_charge * ORIGIN_Z
    )
    corrected_electronic_dipole = (
        shifted.samples["electronic_dipole"] - ORIGIN_Z
    )
    total_dipole_error = float(
        np.max(
            np.abs(
                corrected_total_dipole
                - midpoint.samples["total_dipole"]
            )
        )
    )
    electronic_dipole_error = float(
        np.max(
            np.abs(
                corrected_electronic_dipole
                - midpoint.samples["electronic_dipole"]
            )
        )
    )
    survival = abs(
        hamiltonian.inner_product(eigenstate.orbital, midpoint.state)
    ) ** 2
    parity = float(hamiltonian.inversion_expectation(eigenstate.orbital))
    energy_error = (
        None
        if case.reference_electronic_energy is None
        else abs(
            eigenstate.electronic_energy
            - case.reference_electronic_energy
        )
    )
    nuclear_dipole_midpoint = midpoint_action.dipole_interaction.nuclear_dipole_z
    nuclear_dipole_shifted = shifted_action.dipole_interaction.nuclear_dipole_z

    if eigenstate.residual_norm > 1.0e-10:
        raise RuntimeError(f"{case.name} field-free residual is too large")
    if abs(midpoint.norm - 1.0) > 1.0e-10:
        raise RuntimeError(f"{case.name} midpoint propagation lost norm")
    if abs(shifted.norm - 1.0) > 1.0e-10:
        raise RuntimeError(f"{case.name} shifted propagation lost norm")
    if phase_aligned_state_error > 1.0e-9:
        raise RuntimeError(f"{case.name} origin phase check failed")
    if total_dipole_error > 1.0e-10 or electronic_dipole_error > 1.0e-10:
        raise RuntimeError(f"{case.name} origin dipole check failed")
    if case.nuclear_charge1 == case.nuclear_charge2:
        if abs(parity - 1.0) > 1.0e-12:
            raise RuntimeError(f"{case.name} ground-state parity check failed")
    elif abs(parity) > 0.99:
        raise RuntimeError(f"{case.name} did not exhibit parity mixing")

    return {
        "name": case.name,
        "nuclear_charges": [
            case.nuclear_charge1,
            case.nuclear_charge2,
        ],
        "net_charge": system.net_charge,
        "electronic_energy": eigenstate.electronic_energy,
        "clamped_nuclei_total_energy": eigenstate.total_energy,
        "reference_electronic_energy": case.reference_electronic_energy,
        "reference_energy_error": energy_error,
        "residual_norm": eigenstate.residual_norm,
        "inversion_expectation": parity,
        "initial_electronic_dipole": float(
            midpoint.samples["electronic_dipole"][0]
        ),
        "initial_total_dipole": float(
            midpoint.samples["total_dipole"][0]
        ),
        "nuclear_dipole_midpoint": nuclear_dipole_midpoint,
        "nuclear_dipole_shifted": nuclear_dipole_shifted,
        "final_survival_probability": float(survival),
        "midpoint_norm_error": abs(midpoint.norm - 1.0),
        "shifted_norm_error": abs(shifted.norm - 1.0),
        "origin_phase_aligned_state_error": phase_aligned_state_error,
        "origin_corrected_electronic_dipole_error": (
            electronic_dipole_error
        ),
        "origin_corrected_total_dipole_error": total_dipole_error,
    }


def main():
    options = arguments()
    pulse = SineSquareLaser(
        E0=0.01,
        omega=4.0,
        ncycles=1,
        gauge="length",
        phase=np.pi / 2.0,
    )
    results = [validate_case(case, pulse) for case in CASES]
    report = {
        "grid": {
            "xi_boundaries": list(XI_BOUNDARIES),
            "xi_points_per_element": XI_POINTS_PER_ELEMENT,
            "eta_degree": ETA_DEGREE,
            "half_bond_length": HALF_BOND_LENGTH,
            "m": 0,
        },
        "pulse": {
            "type": "SineSquareLaser",
            "E0": pulse.E0,
            "omega": pulse.omega,
            "ncycles": pulse.ncycles,
            "phase": pulse.phase,
            "duration": pulse.tprime,
        },
        "propagation": {
            "integrator": "sia",
            "time_step": TIME_STEP,
            "shifted_origin_z": ORIGIN_Z,
            "midpoint_field_area": midpoint_integral(
                pulse, pulse.tprime, TIME_STEP
            ),
        },
        "systems": results,
        "status": "PASS",
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")

    print("\n** One-electron diatomic family validation **")
    print(
        f"{'system':7s} {'E_el':>17s} {'<I>':>11s} "
        f"{'d_tot(0)':>13s} {'survival':>12s} {'origin err':>12s}"
    )
    for result in results:
        print(
            f"{result['name']:7s} "
            f"{result['electronic_energy']:17.12f} "
            f"{result['inversion_expectation']:11.7f} "
            f"{result['initial_total_dipole']:13.7f} "
            f"{result['final_survival_probability']:12.9f} "
            f"{result['origin_phase_aligned_state_error']:12.3e}"
        )
    print("\nFamily validation: PASS")
    print(f"Report: {options.output}")


if __name__ == "__main__":
    main()
