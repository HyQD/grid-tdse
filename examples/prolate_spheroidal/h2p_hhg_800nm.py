"""Propagate clamped-nuclei H2+ in a three-cycle 800-nm laser pulse.

The molecule is fixed along the laboratory z axis at R=2 bohr.  The default
length-gauge pulse has a sine-square envelope, field strength 0.05 a.u.,
carrier phase zero, and three optical cycles.  Its angular frequency is
computed from 800 nm using CODATA 2018.  A prolate cubic CAP absorbs outgoing
electronic probability near the outer xi boundary.

The defaults define an initial HHG calculation, not a production-converged
result.  Converge the box, CAP, xi and eta resolutions, time step, Krylov
tolerances, and spectrum window before using the spectrum quantitatively.
All quantities are in atomic units unless stated otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields import SineSquareLaser
from matter_field_params import wavelength_nm_to_omega_au

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalAbsorbingBoundaries,
    ProlateSpheroidalCubicCAP,
    ProlateSpheroidalDynamicsObservables,
    ProlateSpheroidalFixedMTimeDependentHamiltonian,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
)
from grid_tdse.real_time import RealTimePropagator


DEFAULT_WAVELENGTH_NM = 800.0
FREQUENCY_CODATA_SET = "2018"
FIELD_STRENGTH = 0.05
N_CYCLES = 3.0
PHASE = 0.0
PULSE_START = 0.0


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gauge",
        choices=("length", "velocity"),
        default="length",
    )
    parser.add_argument(
        "--field-strength",
        type=float,
        default=FIELD_STRENGTH,
        help="peak electric-field parameter in atomic units",
    )
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=DEFAULT_WAVELENGTH_NM,
        help="carrier wavelength in nanometres",
    )
    parser.add_argument("--time-step", type=float, default=1/32)
    parser.add_argument("--xi-maximum", type=float, default=60.0)
    parser.add_argument("--outer-element-width", type=float, default=4.0)
    parser.add_argument("--xi-points-per-element", type=int, default=11)
    parser.add_argument("--eta-degree", type=int, default=15)
    parser.add_argument("--cap-alpha", type=float, default=0.8)
    parser.add_argument("--cap-eta", type=float, default=1.0e-3)
    parser.add_argument("--max-harmonic", type=float, default=150.0)
    parser.add_argument("--max-krylov-dimension", type=int, default=40)
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-10)
    parser.add_argument("--krylov-atol", type=float, default=1.0e-13)
    parser.add_argument("--max-krylov-refinements", type=int, default=8)
    parser.add_argument(
        "--progress",
        choices=("bar", "log", "none"),
        default="bar",
        help="interactive bar, periodic log records, or no progress output",
    )
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="rolling checkpoint; default is derived from the wavelength",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--overwrite-checkpoint", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="NPZ archive; default is derived from the wavelength",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="PDF plot; default is derived from the wavelength",
    )
    return parser.parse_args()


def xi_boundaries(xi_maximum, outer_element_width):
    """Return focus-resolving inner elements and near-uniform outer ones."""

    if not np.isfinite(xi_maximum) or xi_maximum <= 4.0:
        raise ValueError("xi_maximum must be finite and larger than 4")
    if not np.isfinite(outer_element_width) or outer_element_width <= 0.0:
        raise ValueError("outer_element_width must be finite and positive")
    number_of_outer_elements = int(
        np.ceil((xi_maximum - 4.0) / outer_element_width)
    )
    outer = np.linspace(4.0, xi_maximum, number_of_outer_elements + 1)
    return np.concatenate(([1.0, 2.0], outer))


def build_problem(arguments):
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        particle_mass=1.0,
        name="H2+",
    )
    boundaries = xi_boundaries(
        arguments.xi_maximum,
        arguments.outer_element_width,
    )
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=boundaries,
        xi_points_per_element=arguments.xi_points_per_element,
        eta_degree=arguments.eta_degree,
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
    return boundaries, hamiltonian, ground_state


def hhg_spectrum(times, dipole, omega, max_harmonic):
    """Return a Hann-windowed normalized dipole-form HHG spectrum."""

    times = np.asarray(times)
    dipole = np.asarray(dipole)
    window = np.hanning(times.size)
    transform = np.fft.rfft((dipole - dipole[0]) * window)
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(
        times.size,
        float(times[1] - times[0]),
    )
    intensity = frequencies**4 * np.abs(transform) ** 2
    harmonic_order = frequencies / omega
    selected = (harmonic_order > 0.0) & (harmonic_order <= max_harmonic)
    if not np.any(selected):
        raise ValueError(
            "the sampled time series has no positive-frequency bins below "
            f"max_harmonic={max_harmonic}"
        )
    scale = float(np.max(intensity[selected]))
    normalized = intensity / scale if scale > 0.0 else intensity
    return harmonic_order, intensity, normalized


def classical_hhg_cutoff(electronic_energy, field_strength, omega):
    """Return the three-step ``Ip + 3.17 Up`` cutoff estimate."""

    ionization_potential = -float(electronic_energy)
    if ionization_potential <= 0.0:
        raise ValueError("the bound-state electronic energy must be negative")
    ponderomotive_energy = field_strength**2 / (4.0 * omega**2)
    cutoff_energy = ionization_potential + 3.17 * ponderomotive_energy
    return {
        "ionization_potential": ionization_potential,
        "ponderomotive_energy": ponderomotive_energy,
        "cutoff_energy": cutoff_energy,
        "cutoff_harmonic_order": cutoff_energy / omega,
    }


def plot_results(
    destination,
    times,
    electric_field,
    dipole,
    harmonic_order,
    normalized_spectrum,
    max_harmonic,
    cutoff_harmonic_order,
):
    selected = (harmonic_order > 0.0) & (harmonic_order <= max_harmonic)
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.4))
    axes[0].plot(times, electric_field)
    axes[0].set_ylabel(r"$E_z(t)$ (a.u.)")
    axes[1].plot(times, dipole)
    axes[1].set_ylabel(r"$d_z(t)$ (a.u.)")
    axes[2].semilogy(
        harmonic_order[selected],
        np.maximum(normalized_spectrum[selected], np.finfo(float).tiny),
    )
    axes[2].axvline(
        cutoff_harmonic_order,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=(
            r"$I_p + 3.17U_p$ cutoff "
            f"(H{cutoff_harmonic_order:.1f})"
        ),
    )
    axes[2].set_xlim(0.0, max_harmonic)
    axes[2].set_xlabel("Harmonic order")
    axes[2].set_ylabel(r"Normalized $\omega^4|d_z(\omega)|^2$")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    plt.close(figure)


def main():
    arguments = parse_arguments()
    omega = wavelength_nm_to_omega_au(
        arguments.wavelength_nm,
        codata_set=FREQUENCY_CODATA_SET,
    )
    wavelength_label = f"{arguments.wavelength_nm:g}".replace(".", "p")
    if (
        arguments.field_strength == FIELD_STRENGTH
        and arguments.wavelength_nm == DEFAULT_WAVELENGTH_NM
    ):
        output_stem = "H2plus_hhg_800nm"
    else:
        field_label = f"{arguments.field_strength:g}".replace(".", "p")
        output_stem = f"H2plus_hhg_E{field_label}_{wavelength_label}nm"
    if arguments.checkpoint_path is None:
        arguments.checkpoint_path = Path(f"{output_stem}.chkpt")
    if arguments.output is None:
        arguments.output = Path(f"{output_stem}.npz")
    if arguments.plot is None:
        arguments.plot = Path(f"{output_stem}.pdf")
    pulse = SineSquareLaser(
        E0=arguments.field_strength,
        omega=omega,
        ncycles=N_CYCLES,
        gauge=arguments.gauge,
        phase=PHASE,
        t0=PULSE_START,
    )
    electric_field = SineSquareLaser(
        E0=arguments.field_strength,
        omega=omega,
        ncycles=N_CYCLES,
        gauge="length",
        phase=PHASE,
        t0=PULSE_START,
    )
    vector_potential = SineSquareLaser(
        E0=arguments.field_strength,
        omega=omega,
        ncycles=N_CYCLES,
        gauge="velocity",
        phase=PHASE,
        t0=PULSE_START,
    )
    boundaries, hamiltonian, ground_state = build_problem(arguments)
    absorbing_boundaries = ProlateSpheroidalAbsorbingBoundaries(
        hamiltonian.representation,
        cap=ProlateSpheroidalCubicCAP(
            alpha=arguments.cap_alpha,
            eta=arguments.cap_eta,
        ),
    )
    action = ProlateSpheroidalFixedMTimeDependentHamiltonian(
        hamiltonian,
        m=0,
        fields=pulse,
        field_gauge=arguments.gauge,
        absorbing_boundaries=absorbing_boundaries,
    )
    observables = ProlateSpheroidalDynamicsObservables(
        hamiltonian,
        m=0,
        absorbing_boundaries=absorbing_boundaries,
        reference_states={"ground": ground_state.orbital},
    )
    propagator = RealTimePropagator(
        initial_time=PULSE_START,
        final_time=pulse.tprime,
        time_step=arguments.time_step,
        integrator="sia",
        max_krylov_dimension=arguments.max_krylov_dimension,
        krylov_rtol=arguments.krylov_rtol,
        krylov_atol=arguments.krylov_atol,
        max_krylov_refinements=arguments.max_krylov_refinements,
        progress_interval=(
            arguments.progress_interval
            if arguments.progress == "log"
            else None
        ),
        progress_bar=arguments.progress == "bar",
    )
    dynamics_metadata = {
        "purpose": "initial clamped-nuclei H2+ HHG calculation",
        "field": {
            "type": "SineSquareLaser",
            "gauge": arguments.gauge,
            "field_strength": arguments.field_strength,
            "wavelength_nm": arguments.wavelength_nm,
            "frequency_codata_set": FREQUENCY_CODATA_SET,
            "angular_frequency": omega,
            "n_cycles": N_CYCLES,
            "phase": PHASE,
            "start_time": PULSE_START,
            "duration": pulse.tprime,
        },
    }
    arguments.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result = action.propagate(
        propagator,
        ground_state.orbital,
        samplers=observables.samplers,
        checkpoint_path=arguments.checkpoint_path,
        checkpoint_interval=arguments.checkpoint_interval,
        checkpoint_overwrite=arguments.overwrite_checkpoint,
        compatibility_metadata=action.archive_metadata(),
        dynamics_metadata=dynamics_metadata,
        checkpoint_metadata={"observables": observables.metadata},
    )

    dipole = result.samples["electronic_dipole_numerator"]
    harmonic_order, raw_spectrum, normalized_spectrum = hhg_spectrum(
        result.times,
        dipole,
        omega,
        arguments.max_harmonic,
    )
    cutoff = classical_hhg_cutoff(
        ground_state.electronic_energy,
        arguments.field_strength,
        omega,
    )
    field_values = electric_field(result.times)
    metadata = {
        **dynamics_metadata,
        "system": hamiltonian.archive_metadata(),
        "grid": {
            "xi_boundaries": boundaries.tolist(),
            "xi_points_per_element": arguments.xi_points_per_element,
            "eta_degree": arguments.eta_degree,
            "active_state_shape": list(hamiltonian.fixed_m_shape()),
        },
        "absorber": absorbing_boundaries.metadata,
        "propagator": propagator.integrator_metadata,
        "time_step": result.time_step,
        "number_of_steps": result.n_steps,
        "ground_state": {
            "electronic_energy": ground_state.electronic_energy,
            "total_energy": ground_state.total_energy,
            "residual_norm": ground_state.residual_norm,
        },
        "spectrum": {
            "form": "omega**4 * abs(FFT(Hann * (dipole - dipole[0])))**2",
            "dipole": "unnormalized electronic dipole numerator",
            "classical_cutoff_model": "Ip + 3.17 Up",
            **cutoff,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        times=result.times,
        electric_field=field_values,
        vector_potential=vector_potential(result.times),
        norm=result.norm_history,
        electronic_dipole_numerator=dipole,
        electronic_dipole=result.samples["electronic_dipole"],
        total_dipole_numerator=result.samples["total_dipole_numerator"],
        total_dipole=result.samples["total_dipole"],
        ground_state_population=result.samples["state_populations"][:, 0],
        absorber_region_probability=(
            result.samples["absorber_region_probability"]
        ),
        cap_norm_loss_rate=result.samples["cap_norm_loss_rate"],
        harmonic_order=harmonic_order,
        hhg_spectrum=raw_spectrum,
        normalized_hhg_spectrum=normalized_spectrum,
        ionization_potential=np.array(cutoff["ionization_potential"]),
        ponderomotive_energy=np.array(cutoff["ponderomotive_energy"]),
        classical_cutoff_energy=np.array(cutoff["cutoff_energy"]),
        classical_cutoff_harmonic_order=np.array(
            cutoff["cutoff_harmonic_order"]
        ),
        final_state=result.state,
    )
    plot_results(
        arguments.plot,
        result.times,
        field_values,
        dipole,
        harmonic_order,
        normalized_spectrum,
        arguments.max_harmonic,
        cutoff["cutoff_harmonic_order"],
    )

    print(
        "\n** Clamped-nuclei H2+ HHG at "
        f"{arguments.wavelength_nm:g} nm **"
    )
    print(f"Gauge:                    {arguments.gauge}")
    print(f"Field strength:           {arguments.field_strength:.8f} a.u.")
    print(f"Wavelength:               {arguments.wavelength_nm:.6f} nm")
    print(f"Angular frequency:        {omega:.12f} a.u.")
    print(f"Optical cycles:           {N_CYCLES:.1f}")
    print(f"Pulse duration:           {pulse.tprime:.12f} a.u.")
    print(f"Active state shape:       {hamiltonian.fixed_m_shape()}")
    print(f"Ground-state energy:      {ground_state.total_energy:.12f} E_h")
    print(f"Ground-state residual:    {ground_state.residual_norm:.3e}")
    print(
        "Ionization potential:      "
        f"{cutoff['ionization_potential']:.12f} E_h"
    )
    print(
        "Ponderomotive energy:      "
        f"{cutoff['ponderomotive_energy']:.12f} E_h"
    )
    print(
        "Classical HHG cutoff:      "
        f"{cutoff['cutoff_energy']:.12f} E_h "
        f"(H{cutoff['cutoff_harmonic_order']:.3f})"
    )
    print(f"Time step:                {result.time_step:.12f} a.u.")
    print(f"Number of steps:          {result.n_steps}")
    print(f"Initial norm:             {result.initial_norm:.12f}")
    print(f"Final norm:               {result.norm:.12f}")
    print(f"Output:                   {arguments.output}")
    print(f"Plot:                     {arguments.plot}")
    print(f"Checkpoint:               {arguments.checkpoint_path}")


if __name__ == "__main__":
    main()
