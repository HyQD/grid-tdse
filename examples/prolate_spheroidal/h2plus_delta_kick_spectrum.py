"""Compute the axial delta-kick response of clamped-nuclei H2+.

The default calculation follows Step 11 of the prolate-spheroidal
implementation plan. It uses the Tao-comparable R=2 grid, paired analytic
delta kicks, SIA propagation with a cubic CAP, and a Hann-windowed transform
of the unnormalized induced dipole. All quantities are in atomic units.

The default 1000-a.u. calculation is a validation run rather than a quick
smoke test. Use ``--final-time 20 --time-step 0.1 --kick-strengths 1e-3`` to
exercise the complete workflow rapidly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grid_tdse.prolate_spheroidal import (
    AxialDeltaKick,
    OneElectronDiatomicSystem,
    ProlateSpheroidalAbsorbingBoundaries,
    ProlateSpheroidalCubicCAP,
    ProlateSpheroidalDynamicsObservables,
    ProlateSpheroidalFixedMTimeDependentHamiltonian,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
    paired_kick_linear_response,
    windowed_positive_frequency_spectrum,
)
from grid_tdse.real_time import RealTimePropagator


EXPECTED_ALLOWED = {
    "2p sigma_u": 0.435099822292,
    "3p sigma_u": 0.847221049407,
    "additional Sigma_u root": 0.965321290216,
    "4f sigma_u": 0.975990344344,
}
EXPECTED_FORBIDDEN = {
    "gerade root": 0.924953169367,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-time", type=float, default=1000.0)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument(
        "--kick-strengths",
        type=float,
        nargs="+",
        default=[1.0e-3, 5.0e-4],
    )
    parser.add_argument("--cap-alpha", type=float, default=0.8)
    parser.add_argument("--cap-eta", type=float, default=1.0e-3)
    parser.add_argument("--xi-maximum", type=float, default=80.0)
    parser.add_argument("--outer-xi-elements", type=int, default=10)
    parser.add_argument("--late-time-start", type=float, default=200.0)
    parser.add_argument("--max-krylov-dimension", type=int, default=50)
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-10)
    parser.add_argument("--krylov-atol", type=float, default=1.0e-13)
    parser.add_argument("--max-krylov-refinements", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("H2plus_delta_kick_R2.npz"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("H2plus_delta_kick_R2.pdf"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def build_problem(arguments):
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        particle_mass=1.0,
        name="H2+",
    )
    xi_boundaries = np.concatenate(
        (
            [1.0],
            np.linspace(
                2.0,
                arguments.xi_maximum,
                arguments.outer_xi_elements + 1,
            ),
        )
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=xi_boundaries,
        xi_points_per_element=15,
        eta_degree=9,
        m_max=0,
    )
    ground_state = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=1,
        solver="davidson",
        residual_tolerance=1.0e-10,
        max_iterations=400,
        print_progress=False,
    )
    if not ground_state.converged:
        raise RuntimeError(
            "Davidson ground state did not converge: "
            f"residual={ground_state.residual_norm:.3e}, "
            f"reason={ground_state.termination_reason}"
        )
    return hamiltonian, ground_state


def propagator(arguments):
    return RealTimePropagator(
        final_time=arguments.final_time,
        time_step=arguments.time_step,
        integrator="sia",
        max_krylov_dimension=arguments.max_krylov_dimension,
        krylov_rtol=arguments.krylov_rtol,
        krylov_atol=arguments.krylov_atol,
        max_krylov_refinements=arguments.max_krylov_refinements,
        progress_interval=arguments.progress_interval,
    )


def checkpoint_path(directory, label):
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{label}.chkpt"


def run_state(
    arguments,
    action,
    observables,
    initial_state,
    *,
    label,
    dynamics_metadata,
):
    destination = checkpoint_path(arguments.checkpoint_directory, label)
    return action.propagate(
        propagator(arguments),
        initial_state,
        samplers=observables.samplers,
        checkpoint_path=destination,
        checkpoint_interval=1000 if destination is not None else None,
        checkpoint_overwrite=bool(
            destination is not None and destination.exists()
        ),
        compatibility_metadata=action.archive_metadata(),
        dynamics_metadata=dynamics_metadata,
        checkpoint_metadata={
            "observables": observables.metadata,
            "purpose": "H2+ axial delta-kick linear-response validation",
        },
    )


def run_state_specification(specification):
    """Run one independently restartable trajectory in a worker process."""

    (
        arguments,
        action,
        observables,
        initial_state,
        label,
        dynamics_metadata,
    ) = specification
    return label, run_state(
        arguments,
        action,
        observables,
        initial_state,
        label=label,
        dynamics_metadata=dynamics_metadata,
    )


def run_specifications(specifications, workers):
    """Run independent signed-kick trajectories serially or in parallel."""

    if workers == 1:
        completed = map(run_state_specification, specifications)
        return dict(completed)
    with ProcessPoolExecutor(
        max_workers=min(workers, len(specifications)),
        mp_context=get_context("fork"),
    ) as executor:
        return dict(executor.map(run_state_specification, specifications))


def strength_label(strength):
    return f"K{strength:.3e}".replace("+", "p").replace("-", "m")


def late_spectrum(times, response, start):
    selected = np.searchsorted(times, start, side="left")
    if selected >= times.size - 2:
        return None
    return windowed_positive_frequency_spectrum(
        times[selected:], response[selected:], window="hann"
    )


def nearest_peak(spectrum, target):
    frequencies = spectrum.angular_frequencies
    if not frequencies.size:
        return np.nan
    if frequencies.size == 1:
        return float(frequencies[0])
    spacing = frequencies[1] - frequencies[0]
    power = spectrum.power
    local_maxima = np.flatnonzero(
        (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    ) + 1
    selected = abs(frequencies[local_maxima] - target) <= 3.0 * spacing
    indices = local_maxima[selected]
    if not indices.size:
        return np.nan
    index = indices[np.argmin(abs(frequencies[indices] - target))]
    return float(frequencies[index])


def main():
    arguments = parse_arguments()
    if arguments.final_time <= 0.0 or arguments.time_step <= 0.0:
        raise ValueError("final time and time step must be positive")
    if arguments.late_time_start < 0.0:
        raise ValueError("late-time start must be nonnegative")
    if arguments.xi_maximum <= 2.0:
        raise ValueError("xi maximum must be larger than two")
    if arguments.outer_xi_elements < 1:
        raise ValueError("outer xi elements must be positive")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    if arguments.progress_interval < 1:
        raise ValueError("progress interval must be positive")
    strengths = tuple(abs(value) for value in arguments.kick_strengths)
    if not strengths or any(value == 0.0 for value in strengths):
        raise ValueError("kick strengths must be nonzero")

    hamiltonian, ground_state = build_problem(arguments)
    boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(
            alpha=arguments.cap_alpha,
            eta=arguments.cap_eta,
        ),
    )
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        m=0,
        absorbing_boundaries=boundaries,
    )
    observables = ProlateSpheroidalDynamicsObservables(
        hamiltonian,
        m=0,
        absorbing_boundaries=boundaries,
        reference_states={"ground": ground_state.orbital},
    )

    specifications = [
        (
            arguments,
            action,
            observables,
            ground_state.orbital,
            "no-kick",
            {"delta_kick": None},
        )
    ]
    for strength in strengths:
        label = strength_label(strength)
        for sign, prefix in ((1.0, "positive"), (-1.0, "negative")):
            kick = AxialDeltaKick(sign * strength)
            specifications.append(
                (
                    arguments,
                    action,
                    observables,
                    kick.apply(hamiltonian, ground_state.orbital),
                    f"{prefix}-{label}",
                    {"delta_kick": kick.metadata},
                )
            )
    results = run_specifications(specifications, arguments.workers)
    no_kick = results["no-kick"]
    archives = {
        "times": no_kick.times,
        "no_kick_norm": no_kick.norm_history,
    }
    for name, values in no_kick.samples.items():
        archives[f"no_kick_{name}"] = values
    spectra = {}
    responses = {}
    response_metadata = {}

    for strength in strengths:
        label = strength_label(strength)
        kick = AxialDeltaKick(strength)
        positive = results[f"positive-{label}"]
        negative_kick = AxialDeltaKick(-strength)
        negative = results[f"negative-{label}"]
        response = paired_kick_linear_response(
            positive.samples["electronic_dipole_numerator"],
            negative.samples["electronic_dipole_numerator"],
            strength,
        )
        spectrum = windowed_positive_frequency_spectrum(
            positive.times, response, window="hann"
        )
        late = late_spectrum(
            positive.times, response, arguments.late_time_start
        )
        spectra[strength] = spectrum
        responses[strength] = response
        odd_dipole = 0.5 * (
            positive.samples["electronic_dipole_numerator"]
            - negative.samples["electronic_dipole_numerator"]
        )
        even_remainder = 0.5 * (
            positive.samples["electronic_dipole_numerator"]
            + negative.samples["electronic_dipole_numerator"]
        ) - no_kick.samples["electronic_dipole_numerator"]
        antisymmetry_error = np.linalg.norm(even_remainder) / max(
            np.linalg.norm(odd_dipole), np.finfo(float).tiny
        )
        response_metadata[label] = {
            "positive_kick": kick.metadata,
            "negative_kick": negative_kick.metadata,
            "complete_spectrum": spectrum.metadata,
            "late_spectrum": None if late is None else late.metadata,
            "kick_sign_antisymmetry_relative_l2": float(
                antisymmetry_error
            ),
        }
        archives[f"response_{label}"] = response
        archives[f"omega_{label}"] = spectrum.angular_frequencies
        archives[f"transform_{label}"] = spectrum.transform
        archives[f"positive_norm_{label}"] = positive.norm_history
        archives[f"negative_norm_{label}"] = negative.norm_history
        for name, values in positive.samples.items():
            archives[f"positive_{name}_{label}"] = values
        for name, values in negative.samples.items():
            archives[f"negative_{name}_{label}"] = values
        if late is not None:
            archives[f"late_omega_{label}"] = late.angular_frequencies
            archives[f"late_transform_{label}"] = late.transform

    reference_strength = strengths[0]
    reference_response = responses[reference_strength]
    reference_scale = max(
        np.linalg.norm(reference_response), np.finfo(float).tiny
    )
    linearity_errors = {
        strength_label(strength): float(
            np.linalg.norm(response - reference_response) / reference_scale
        )
        for strength, response in responses.items()
    }

    metadata = {
        "system": hamiltonian.archive_metadata(),
        "ground_electronic_energy": ground_state.electronic_energy,
        "ground_residual_norm": ground_state.residual_norm,
        "absorbing_boundaries": boundaries.metadata,
        "observables": observables.metadata,
        "integrator": propagator(arguments).integrator_metadata,
        "final_time": arguments.final_time,
        "requested_time_step": arguments.time_step,
        "parallel_trajectory_workers": arguments.workers,
        "responses": response_metadata,
        "linearity_reference_strength": reference_strength,
        "linearity_relative_l2_to_reference": linearity_errors,
        "expected_allowed_excitations": EXPECTED_ALLOWED,
        "expected_forbidden_excitations": EXPECTED_FORBIDDEN,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        **archives,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for strength, spectrum in spectra.items():
        power = spectrum.power
        power = power / max(float(power.max()), np.finfo(float).tiny)
        selected = spectrum.angular_frequencies <= 1.2
        axis.semilogy(
            spectrum.angular_frequencies[selected],
            np.maximum(power[selected], 1.0e-16),
            label=f"K={strength:g}",
        )
    for label, energy in EXPECTED_ALLOWED.items():
        axis.axvline(energy, color="0.5", linewidth=0.7, linestyle="--")
        axis.text(
            energy,
            2.0e-15,
            label,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
        )
    for energy in EXPECTED_FORBIDDEN.values():
        axis.axvline(energy, color="tab:red", linewidth=0.8, linestyle=":")
    axis.set(
        xlabel="Excitation energy (Eh)",
        ylabel="Normalized dipole-response power",
        xlim=(0.0, 1.2),
        ylim=(1.0e-16, 2.0),
    )
    axis.legend()
    figure.tight_layout()
    arguments.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.plot)
    plt.close(figure)

    print("\n** H2+ axial delta-kick response **")
    print(f"Active state shape:       {hamiltonian.fixed_m_shape(0)}")
    print(f"Ground energy:            {ground_state.electronic_energy:.15f} Eh")
    print(f"No-kick final norm:       {no_kick.norm:.15f}")
    for strength, spectrum in spectra.items():
        label_key = strength_label(strength)
        print(f"\nK = {strength:g}")
        print(
            "  kick-sign antisymmetry L2: "
            f"{response_metadata[label_key]['kick_sign_antisymmetry_relative_l2']:.3e}"
        )
        print(
            "  linearity L2 vs reference: "
            f"{linearity_errors[label_key]:.3e}"
        )
        for label, energy in EXPECTED_ALLOWED.items():
            frequency = nearest_peak(spectrum, energy)
            print(
                f"  {label:27s}: expected {energy:.12f}, "
                f"nearby peak {frequency:.12f}"
            )
    print(f"\nArchive: {arguments.output}")
    print(f"Plot:    {arguments.plot}")


if __name__ == "__main__":
    main()
