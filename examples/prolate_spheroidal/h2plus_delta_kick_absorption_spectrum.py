"""Lorentz-broadened axial absorption spectrum of clamped-nuclei H2+.

The reference linear response is assembled from the stationary eigenstates
on the Tao-comparable prolate grid. When a Step-11 propagation archive is
provided, the spectrum also includes the one-sided positive-kick response
and the paired positive/negative-kick response. All three use the same time
grid, exponential damping, Lorentzian FWHM, and eigenstate-reference
normalization. The plotted quantity is the ``zz`` contribution to the
dipole-strength function in Eq. (24) of Repisky et al., J. Chem. Theory
Comput. 11, 980 (2015):

``S_zz(omega) = 4*pi*omega/(3*c) * Im(alpha_zz(omega))``.

Only the axial response is evaluated. Consequently, this is neither the full
Cartesian trace nor a calculation with perpendicular polarization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants

from grid_tdse.prolate_spheroidal import (
    OneElectronDiatomicSystem,
    ProlateSpheroidalOneElectronHamiltonian,
    diagonalize_fixed_m,
    lorentzian_broadened_positive_frequency_spectrum,
)


HARTREE_TO_EV = physical_constants["Hartree energy in eV"][0]
SPEED_OF_LIGHT_AU = physical_constants[
    "inverse fine-structure constant"
][0]


def strength_label(strength):
    return f"K{abs(strength):.3e}".replace("+", "p").replace("-", "m")


def load_propagated_responses(path, kick_strength):
    """Load one-sided and paired responses from a Step-11 archive."""

    if not path.is_file():
        raise FileNotFoundError(f"propagation archive not found: {path}")
    label = strength_label(kick_strength)
    positive_key = f"positive_electronic_dipole_numerator_{label}"
    paired_key = f"response_{label}"
    with np.load(path, allow_pickle=False) as archive:
        missing = {
            key
            for key in ("times", positive_key, paired_key)
            if key not in archive
        }
        if missing:
            raise KeyError(
                "propagation archive is missing: " + ", ".join(sorted(missing))
            )
        times = np.asarray(archive["times"], dtype=float).copy()
        positive_dipole = np.asarray(
            archive[positive_key], dtype=float
        ).copy()
        paired_response = np.asarray(archive[paired_key], dtype=float).copy()
    if times.shape != positive_dipole.shape or times.shape != paired_response.shape:
        raise ValueError("propagation response arrays must share one shape")
    one_sided_response = (
        positive_dipole - positive_dipole[0]
    ) / kick_strength
    return times, one_sided_response, paired_response, label


def eq24_zz_strength(spectrum):
    """Return the nonnegative-sign Eq. (24) ``zz`` contribution."""

    omega = spectrum.angular_frequencies
    absorptive_imaginary_alpha_zz = -spectrum.transform.imag
    strength = (
        4.0
        * np.pi
        * omega
        / (3.0 * SPEED_OF_LIGHT_AU)
        * absorptive_imaginary_alpha_zz
    )
    return absorptive_imaginary_alpha_zz, strength


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number-of-states", type=int, default=80)
    parser.add_argument("--final-time", type=float, default=4000.0)
    parser.add_argument("--sample-step", type=float, default=0.1)
    parser.add_argument("--lorentzian-fwhm", type=float, default=0.005)
    parser.add_argument("--maximum-energy", type=float, default=1.2)
    parser.add_argument("--kick-strength", type=float, default=1.0e-3)
    parser.add_argument(
        "--propagation-archive",
        type=Path,
        default=None,
        help=(
            "optional h2plus_delta_kick_spectrum.py archive used to "
            "overlay one-sided and paired-kick spectra"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("H2plus_delta_kick_absorption_spectrum.npz"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("H2plus_delta_kick_absorption_spectrum.pdf"),
    )
    return parser.parse_args()


def main():
    options = arguments()
    if options.number_of_states < 2:
        raise ValueError("number of states must be at least two")
    if options.final_time <= 0.0 or options.sample_step <= 0.0:
        raise ValueError("final time and sample step must be positive")
    if options.lorentzian_fwhm <= 0.0:
        raise ValueError("Lorentzian FWHM must be positive")
    if options.kick_strength == 0.0:
        raise ValueError("kick strength must be nonzero")

    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        particle_mass=1.0,
        name="H2+",
    )
    xi_boundaries = np.concatenate(([1.0], np.linspace(2.0, 80.0, 11)))
    hamiltonian = ProlateSpheroidalOneElectronHamiltonian(
        system,
        xi_boundaries=xi_boundaries,
        xi_points_per_element=15,
        eta_degree=9,
        m_max=0,
    )
    states = diagonalize_fixed_m(
        hamiltonian,
        m=0,
        n_states=options.number_of_states,
        solver="direct",
    )
    ground = states.orbitals[0]
    z_ground = hamiltonian.apply_z_fixed_m(ground)
    transition_dipoles = np.asarray(
        [
            hamiltonian.inner_product(orbital, z_ground)
            for orbital in states.orbitals[1:]
        ]
    )
    excitation_energies = (
        states.electronic_energies[1:] - states.electronic_energies[0]
    )
    inversion = np.asarray(
        [hamiltonian.inversion_expectation(orbital) for orbital in states.orbitals]
    )
    line_weights = np.abs(transition_dipoles) ** 2

    propagated = None
    if options.propagation_archive is None:
        number_of_intervals = int(
            np.ceil(options.final_time / options.sample_step)
        )
        times = np.linspace(0.0, options.final_time, number_of_intervals + 1)
    else:
        propagated = load_propagated_responses(
            options.propagation_archive, options.kick_strength
        )
        times = propagated[0]
    eigenstate_response = 2.0 * np.sum(
        line_weights[:, None]
        * np.sin(excitation_energies[:, None] * times[None, :]),
        axis=0,
    )
    induced_dipole = options.kick_strength * eigenstate_response
    polarizability_response = induced_dipole / options.kick_strength
    eigenstate_spectrum = lorentzian_broadened_positive_frequency_spectrum(
        times,
        polarizability_response,
        fwhm=options.lorentzian_fwhm,
    )
    omega = eigenstate_spectrum.angular_frequencies
    # numpy.rfft uses exp(-i*omega*t), as does PositiveFrequencySpectrum.
    # The absorptive part is therefore -Im(alpha_zz) for the present response
    # sign. This is the nonnegative Im(alpha_zz) entering Eq. (24) when the
    # opposite, conventional causal-transform sign is used.
    absorptive_imaginary_alpha_zz, eq24_zz_dipole_strength = (
        eq24_zz_strength(eigenstate_spectrum)
    )
    directional_zz_dipole_strength = 3.0 * eq24_zz_dipole_strength

    comparison_spectra = {}
    comparison_strengths = {}
    if propagated is not None:
        _, one_sided_response, paired_response, propagation_label = propagated
        comparison_responses = {
            "one_sided": one_sided_response,
            "paired": paired_response,
        }
        for name, values in comparison_responses.items():
            comparison_spectra[name] = (
                lorentzian_broadened_positive_frequency_spectrum(
                    times,
                    values,
                    fwhm=options.lorentzian_fwhm,
                )
            )
            _, comparison_strengths[name] = eq24_zz_strength(
                comparison_spectra[name]
            )

    selected = omega <= options.maximum_energy
    peak_strength = float(eq24_zz_dipole_strength[selected].max())
    if peak_strength <= 0.0:
        raise RuntimeError("the computed zz absorption strength is not positive")
    relative_eq24_zz_intensity = eq24_zz_dipole_strength / peak_strength
    relative_comparison_strengths = {
        name: values / peak_strength
        for name, values in comparison_strengths.items()
    }
    metadata = {
        "system": hamiltonian.archive_metadata(),
        "number_of_states": options.number_of_states,
        "ground_electronic_energy": states.electronic_energy,
        "kick_strength": options.kick_strength,
        "final_time": times[-1],
        "sample_step": times[1] - times[0],
        "lorentzian_fwhm_hartree": options.lorentzian_fwhm,
        "lorentzian_fwhm_ev": options.lorentzian_fwhm * HARTREE_TO_EV,
        "spectrum": eigenstate_spectrum.metadata,
        "response_construction": (
            "2*sum_n(abs(<n|z|0>)**2*sin((E_n-E_0)*t))"
        ),
        "propagation_comparison": None
        if propagated is None
        else {
            "archive": str(options.propagation_archive),
            "strength_label": propagation_label,
            "one_sided_response": "(d(+K,t)-d(+K,0))/K",
            "paired_response": "(d(+K,t)-d(-K,t))/(2*K)",
            "common_normalization": "eigenstate reference maximum",
        },
        "repisky_eq24": {
            "reference": (
                "Repisky et al., J. Chem. Theory Comput. 11, 980-991 "
                "(2015), doi:10.1021/ct501078d"
            ),
            "paper_equation": "S(omega)=4*pi*omega/(3*c)*Im Tr(alpha)",
            "computed_quantity": (
                "zz contribution only: replace Tr(alpha) by alpha_zz"
            ),
            "perpendicular_components_computed": False,
            "speed_of_light_atomic_units": SPEED_OF_LIGHT_AU,
            "stored_transform_convention": (
                "integral alpha_zz(t)*exp(-1j*omega*t) dt"
            ),
            "absorptive_imaginary_part": (
                "-imag(stored alpha_zz transform)"
            ),
            "eq24_zz_formula_for_stored_transform": (
                "-4*pi*omega/(3*c)*imag(alpha_zz_minus)"
            ),
            "directional_parallel_formula": (
                "-4*pi*omega/c*imag(alpha_zz_minus)"
            ),
            "relative_intensity_normalization": (
                "eq24_zz_dipole_strength divided by its maximum over the "
                "plotted energy interval"
            ),
            "normalization_maximum_energy_hartree": options.maximum_energy,
            "normalization_factor": peak_strength,
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    archive_values = {
        "times": times,
        "induced_dipole": induced_dipole,
        "polarizability_response": polarizability_response,
        "angular_frequency": omega,
        "energy_ev": omega * HARTREE_TO_EV,
        "alpha_zz_minus_convention": eigenstate_spectrum.transform,
        "absorptive_imaginary_alpha_zz": absorptive_imaginary_alpha_zz,
        "eq24_zz_dipole_strength": eq24_zz_dipole_strength,
        "relative_eq24_zz_intensity": relative_eq24_zz_intensity,
        "directional_zz_dipole_strength": directional_zz_dipole_strength,
        "excitation_energies": excitation_energies,
        "transition_dipoles": transition_dipoles,
        "inversion_expectations": inversion,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if propagated is not None:
        archive_values.update(
            {
                "one_sided_response": one_sided_response,
                "paired_response": paired_response,
                "one_sided_alpha_zz_minus_convention": (
                    comparison_spectra["one_sided"].transform
                ),
                "paired_alpha_zz_minus_convention": (
                    comparison_spectra["paired"].transform
                ),
                "one_sided_relative_eq24_zz_intensity": (
                    relative_comparison_strengths["one_sided"]
                ),
                "paired_relative_eq24_zz_intensity": (
                    relative_comparison_strengths["paired"]
                ),
            }
        )
    np.savez_compressed(options.output, **archive_values)

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    displayed_intensity = relative_eq24_zz_intensity[selected]
    axis.plot(
        omega[selected],
        displayed_intensity,
        color="black",
        linewidth=1.5,
        label="Eigenstate reference",
        zorder=1,
    )
    displayed_comparisons = {}
    comparison_styles = {
        "one_sided": {
            "color": "tab:orange",
            "label": r"One-sided $+K$ propagation",
            "linestyle": "--",
            "zorder": 2,
        },
        "paired": {
            "color": "tab:blue",
            "label": r"Paired $\pm K$ propagation",
            "linestyle": ":",
            "zorder": 3,
        },
    }
    for name, values in relative_comparison_strengths.items():
        displayed_comparisons[name] = values[selected]
        axis.plot(
            omega[selected],
            displayed_comparisons[name],
            linewidth=1.25,
            **comparison_styles[name],
        )
    axis.axvline(
        -states.electronic_energy,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        label="Ionization threshold",
    )
    for energy, weight, parity in zip(
        excitation_energies,
        line_weights,
        inversion[1:],
        strict=True,
    ):
        if energy > options.maximum_energy or weight < 1.0e-10:
            continue
        color = "tab:orange" if parity < 0.0 else "tab:red"
        axis.axvline(energy, color=color, alpha=0.35, linewidth=0.6)
    axis.set(
        xlabel=r"Excitation energy ($E_h$)",
        ylabel=r"Relative $zz$ absorption intensity",
        xlim=(0.0, options.maximum_energy),
        title=(
            r"Clamped-nuclei $\mathrm{H}_2^+$ axial absorption, "
            rf"$R=2a_0$, Lorentzian FWHM="
            rf"${options.lorentzian_fwhm:g}\,E_h$"
        ),
    )
    all_displayed = [displayed_intensity, *displayed_comparisons.values()]
    positive_intensity = np.concatenate(
        [values[values > 0.0] for values in all_displayed]
    )
    if not positive_intensity.size:
        raise RuntimeError("the plotted absorption intensity is not positive")
    axis.set_yscale("log")
    axis.set_ylim(0.5 * positive_intensity.min(), 1.2)
    axis.grid(alpha=0.2)
    axis.legend()
    axis.text(
        0.985,
        0.965,
        r"$\mathrm{Tr}\,\alpha\rightarrow\alpha_{zz}$; no $x/y$ response",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=8.5,
        color="0.25",
    )
    top = axis.secondary_xaxis(
        "top",
        functions=(
            lambda energy: energy * HARTREE_TO_EV,
            lambda energy_ev: energy_ev / HARTREE_TO_EV,
        ),
    )
    top.set_xlabel("Excitation energy (eV)")
    figure.tight_layout()
    options.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(options.plot)
    plt.close(figure)

    print("\n** H2+ Lorentz-broadened axial absorption spectrum **")
    print(f"Ground energy:       {states.electronic_energy:.15f} Eh")
    print(f"States included:     {options.number_of_states}")
    print(f"Record duration:     {times[-1]:g} a.u.")
    print(
        f"Lorentzian FWHM:     {options.lorentzian_fwhm:g} Eh = "
        f"{options.lorentzian_fwhm * HARTREE_TO_EV:.6f} eV"
    )
    print(f"Speed of light:       {SPEED_OF_LIGHT_AU:.12f} a.u.")
    print("Strength definition:  Repisky Eq. (24), zz contribution only")
    print("Plot normalization:   maximum zz strength in plotted range = 1")
    print("Plot y-axis:          logarithmic relative intensity")
    if propagated is not None:
        paired_scale = max(
            np.linalg.norm(paired_response), np.finfo(float).tiny
        )
        spectrum_scale = max(
            np.linalg.norm(displayed_intensity), np.finfo(float).tiny
        )
        print(f"Propagation archive: {options.propagation_archive}")
        print(f"Propagation kick:    {options.kick_strength:g} a.u.")
        print(
            "One-sided/paired response L2: "
            f"{np.linalg.norm(one_sided_response - paired_response) / paired_scale:.3e}"
        )
        for name, values in displayed_comparisons.items():
            error = np.linalg.norm(values - displayed_intensity) / spectrum_scale
            print(f"{name:>20s}/reference spectrum L2: {error:.3e}")
    print(f"Ionization threshold:{-states.electronic_energy:18.12f} Eh")
    print("\nDipole-active bound-state lines:")
    threshold = -states.electronic_energy
    for energy, dipole, parity in zip(
        excitation_energies,
        transition_dipoles,
        inversion[1:],
        strict=True,
    ):
        if energy >= threshold or abs(dipole) ** 2 < 1.0e-10:
            continue
        print(
            f"  {energy:16.12f} Eh  {energy * HARTREE_TO_EV:12.6f} eV  "
            f"|<n|z|0>|^2={abs(dipole) ** 2:.6e}  P={parity:+.0f}"
        )
    print(f"\nData: {options.output}")
    print(f"Plot: {options.plot}")


if __name__ == "__main__":
    main()
