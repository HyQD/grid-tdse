"""Qualify axial length- and velocity-gauge H2+ propagation.

The calculation uses a three-cycle pulse on several compact prolate grids
and at two time steps.  It compares field-free energies, norm conservation,
phase-aligned final states, dipole histories, and dipole-form HHG spectra.
This is a gauge-equivalence qualification, not a production HHG convergence
study.  All quantities are in atomic units.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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
class Case:
    name: str
    xi_boundaries: tuple[float, ...]
    xi_points_per_element: int
    eta_degree: int
    time_step: float


CASES = (
    Case("reference", (1.0, 2.0, 4.0, 8.0, 12.0, 16.0), 9, 13, 0.05),
    Case("coarse_time_step", (1.0, 2.0, 4.0, 8.0, 12.0, 16.0), 9, 13, 0.1),
    Case("compact_box", (1.0, 2.0, 4.0, 8.0, 12.0), 9, 13, 0.05),
    Case("lower_resolution", (1.0, 2.0, 4.0, 8.0, 12.0, 16.0), 7, 9, 0.05),
)
PULSE = {
    "E0": 0.05,
    "omega": 0.8,
    "ncycles": 3,
    "phase": 0.0,
}
REFERENCE_ELECTRONIC_ENERGY = -1.102634214494946


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("H2plus_length_velocity_gauge.json"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("H2plus_length_velocity_gauge.npz"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("H2plus_length_velocity_gauge.pdf"),
    )
    return parser.parse_args()


def metric_norm(hamiltonian, state):
    value = hamiltonian.inner_product(state, state)
    return float(np.sqrt(max(value.real, 0.0)))


def relative_l2(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    denominator = max(np.linalg.norm(left), np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(left - right) / denominator)


def hhg_spectrum(times, dipoles):
    """Return harmonic order and a normalized dipole-form HHG spectrum."""

    times = np.asarray(times)
    dipoles = np.asarray(dipoles)
    spacing = float(times[1] - times[0])
    window = np.hanning(times.size)
    transform = np.fft.rfft((dipoles - dipoles[0]) * window)
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(times.size, spacing)
    intensity = frequencies**4 * np.abs(transform) ** 2
    selected = (frequencies >= 0.5 * PULSE["omega"]) & (
        frequencies <= 8.0 * PULSE["omega"]
    )
    scale = float(np.max(intensity[selected]))
    normalized = intensity / scale if scale > 0.0 else intensity
    return frequencies / PULSE["omega"], normalized, selected


def propagate(hamiltonian, initial_state, gauge, time_step):
    pulse = SineSquareLaser(gauge=gauge, **PULSE)
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        fields=pulse,
        field_gauge=gauge,
    )
    propagator = RealTimePropagator(
        final_time=pulse.tprime,
        time_step=time_step,
        integrator="sia",
        max_krylov_dimension=30,
        krylov_rtol=1.0e-11,
        krylov_atol=1.0e-13,
        max_krylov_refinements=6,
        progress_interval=None,
    )
    result = action.propagate(
        propagator,
        initial_state,
        samplers={
            "dipole": lambda state, time: -hamiltonian.inner_product(
                state,
                hamiltonian.apply_z_fixed_m(state),
            ).real
        },
    )
    harmonic_order, spectrum, selected = hhg_spectrum(
        result.times, result.samples["dipole"]
    )
    return result, harmonic_order, spectrum, selected


def run_case(case):
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        name="H2+",
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=case.xi_boundaries,
        xi_points_per_element=case.xi_points_per_element,
        eta_degree=case.eta_degree,
        m_max=0,
    )
    eigenstate = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=1,
        solver="direct",
    )
    length, harmonic_order, length_spectrum, selected = propagate(
        hamiltonian, eigenstate.orbital, "length", case.time_step
    )
    velocity, velocity_order, velocity_spectrum, velocity_selected = (
        propagate(
            hamiltonian,
            eigenstate.orbital,
            "velocity",
            case.time_step,
        )
    )
    np.testing.assert_allclose(velocity_order, harmonic_order)
    np.testing.assert_array_equal(velocity_selected, selected)

    overlap = hamiltonian.inner_product(length.state, velocity.state)
    if abs(overlap) == 0.0:
        raise RuntimeError(
            "length- and velocity-gauge final states are orthogonal"
        )
    fidelity = abs(overlap) ** 2 / (length.norm * velocity.norm)
    phase = overlap / abs(overlap)
    final_error = metric_norm(
        hamiltonian, velocity.state / phase - length.state
    )
    dipole_error = relative_l2(
        length.samples["dipole"], velocity.samples["dipole"]
    )
    spectrum_error = float(
        np.linalg.norm(
            length_spectrum[selected] - velocity_spectrum[selected]
        )
        / np.sqrt(np.count_nonzero(selected))
    )

    report = {
        "case": asdict(case),
        "active_state_shape": list(hamiltonian.fixed_m_shape()),
        "electronic_energy": eigenstate.electronic_energy,
        "electronic_energy_error": abs(
            eigenstate.electronic_energy - REFERENCE_ELECTRONIC_ENERGY
        ),
        "residual_norm": eigenstate.residual_norm,
        "length_norm_error": abs(length.norm - length.initial_norm),
        "velocity_norm_error": abs(velocity.norm - velocity.initial_norm),
        "final_state_fidelity": float(fidelity),
        "phase_aligned_final_state_error": final_error,
        "dipole_relative_l2_error": dipole_error,
        "hhg_spectrum_rms_error": spectrum_error,
    }
    arrays = {
        f"{case.name}_times": length.times,
        f"{case.name}_length_dipole": length.samples["dipole"],
        f"{case.name}_velocity_dipole": velocity.samples["dipole"],
        f"{case.name}_harmonic_order": harmonic_order,
        f"{case.name}_length_hhg": length_spectrum,
        f"{case.name}_velocity_hhg": velocity_spectrum,
    }
    return report, arrays


def refinement_metrics(archive, candidate):
    reference_order = archive["reference_harmonic_order"]
    selected = (reference_order >= 0.5) & (reference_order <= 8.0)
    metrics = {}
    for gauge in ("length", "velocity"):
        metrics[f"{gauge}_dipole_relative_l2"] = relative_l2(
            archive[f"reference_{gauge}_dipole"],
            archive[f"{candidate}_{gauge}_dipole"],
        )
        metrics[f"{gauge}_hhg_spectrum_rms"] = float(
            np.linalg.norm(
                archive[f"reference_{gauge}_hhg"][selected]
                - archive[f"{candidate}_{gauge}_hhg"][selected]
            )
            / np.sqrt(np.count_nonzero(selected))
        )
    return metrics


def validate(reports, refinements):
    by_name = {report["case"]["name"]: report for report in reports}
    reference = by_name["reference"]
    coarse = by_name["coarse_time_step"]
    checks = {
        "field_free_energy": reference["electronic_energy_error"] < 1.0e-9,
        "length_norm": reference["length_norm_error"] < 1.0e-9,
        "velocity_norm": reference["velocity_norm_error"] < 1.0e-9,
        "final_state_overlap": (
            1.0 - reference["final_state_fidelity"] < 1.0e-8
        ),
        "phase_aligned_final_state": (
            reference["phase_aligned_final_state_error"] < 1.0e-5
        ),
        "dipole_history": reference["dipole_relative_l2_error"] < 1.0e-3,
        "hhg_spectrum": reference["hhg_spectrum_rms_error"] < 1.0e-3,
        "time_step_convergence": (
            reference["dipole_relative_l2_error"]
            < 0.35 * coarse["dipole_relative_l2_error"]
        ),
        "time_step_final_state_convergence": (
            reference["phase_aligned_final_state_error"]
            < 0.1 * coarse["phase_aligned_final_state_error"]
        ),
        "time_step_spectrum_convergence": (
            reference["hhg_spectrum_rms_error"]
            < 0.35 * coarse["hhg_spectrum_rms_error"]
        ),
    }
    for name in ("compact_box", "lower_resolution"):
        checks[f"{name}_energy"] = (
            by_name[name]["electronic_energy_error"] < 1.0e-6
        )
        checks[f"{name}_gauge_spectrum"] = (
            by_name[name]["hhg_spectrum_rms_error"] < 1.0e-3
        )
        checks[f"{name}_gauge_final_state"] = (
            by_name[name]["phase_aligned_final_state_error"] < 2.0e-5
        )
        checks[f"{name}_spatial_dipole"] = max(
            refinements[name]["length_dipole_relative_l2"],
            refinements[name]["velocity_dipole_relative_l2"],
        ) < 1.0e-3
        checks[f"{name}_spatial_spectrum"] = max(
            refinements[name]["length_hhg_spectrum_rms"],
            refinements[name]["velocity_hhg_spectrum_rms"],
        ) < 1.0e-3
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("failed validation checks: " + ", ".join(failed))
    return checks


def plot_reference(archive, path):
    times = archive["reference_times"]
    harmonic_order = archive["reference_harmonic_order"]
    selected = (harmonic_order >= 0.5) & (harmonic_order <= 8.0)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))
    axes[0].plot(times, archive["reference_length_dipole"], label="length")
    axes[0].plot(
        times,
        archive["reference_velocity_dipole"],
        "--",
        label="velocity",
    )
    axes[0].set_ylabel("Dipole $d_z(t)$ (a.u.)")
    axes[0].legend()
    axes[1].semilogy(
        harmonic_order[selected],
        archive["reference_length_hhg"][selected],
        label="length",
    )
    axes[1].semilogy(
        harmonic_order[selected],
        archive["reference_velocity_hhg"][selected],
        "--",
        label="velocity",
    )
    axes[1].set_xlabel("Harmonic order")
    axes[1].set_ylabel(r"Normalized $\omega^4|d_z(\omega)|^2$")
    axes[1].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def main():
    options = arguments()
    reports = []
    archive = {}
    for case in CASES:
        report, arrays = run_case(case)
        reports.append(report)
        archive.update(arrays)
    refinements = {
        name: refinement_metrics(archive, name)
        for name in ("compact_box", "lower_resolution")
    }
    checks = validate(reports, refinements)
    document = {
        "system": "clamped-nuclei H2+",
        "internuclear_distance": 2.0,
        "m": 0,
        "pulse": PULSE,
        "propagator": "SIA",
        "hhg_spectrum": {
            "form": "omega**4 * abs(FFT(window * (dipole - dipole[0])))**2",
            "window": "Hann",
            "normalization": "unit maximum from harmonic order 0.5 to 8",
        },
        "cases": reports,
        "spatial_refinement_against_reference": refinements,
        "checks": checks,
        "status": "PASS",
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(document, indent=2) + "\n")
    options.archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(options.archive, **archive)
    plot_reference(archive, options.plot)

    print("\n** H2+ axial length/velocity-gauge qualification **")
    print(
        f"{'case':20s} {'dE':>10s} {'1-F':>10s} "
        f"{'dipole L2':>12s} {'HHG RMS':>12s}"
    )
    for report in reports:
        print(
            f"{report['case']['name']:20s} "
            f"{report['electronic_energy_error']:10.3e} "
            f"{1.0 - report['final_state_fidelity']:10.3e} "
            f"{report['dipole_relative_l2_error']:12.3e} "
            f"{report['hhg_spectrum_rms_error']:12.3e}"
        )
    print("\nLength/velocity qualification: PASS")
    print(f"Report:  {options.output}")
    print(f"Archive: {options.archive}")
    print(f"Plot:    {options.plot}")


if __name__ == "__main__":
    main()
