"""Shared configuration for the serial and Cartesian-MPI Phase 9h examples."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path

import numpy as np
from electromagnetic_fields import SineSquareLaser
from grid_lib.spherical_coordinates import CubicCAP
from matter_field_params import (
    intensity_wcm2_to_field_au,
    wavelength_nm_to_omega_au,
)

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombMultipoleSpec,
    CoulombTerm,
    FEMDVRGridSpec,
    TwoRadialAbsorbingBoundaries,
    TwoRadialSystemSpec,
    radial_potential,
)
from grid_tdse.spherical.two_radial.diatomic import (
    _diatomic_parameters,
    _pair_exchange_parity,
)


DEFAULT_ELECTRONIC_NODES = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)
DEFAULT_INTERNUCLEAR_NODES = (0.0, 5.0, 10.0)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared physical, numerical, and output options."""

    parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), default="H2+")
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--M-tot", type=int, default=0)
    parser.add_argument(
        "--electronic-nodes",
        type=float,
        nargs="+",
        default=DEFAULT_ELECTRONIC_NODES,
    )
    parser.add_argument("--electronic-points-per-element", type=int, default=11)
    parser.add_argument(
        "--internuclear-nodes",
        type=float,
        nargs="+",
        default=DEFAULT_INTERNUCLEAR_NODES,
    )
    parser.add_argument("--internuclear-points-per-element", type=int, default=15)
    parser.add_argument("--codata-set", choices=("2018", "2022"), default="2018")
    parser.add_argument(
        "--ground-state-path",
        type=Path,
        help="Davidson archive; defaults inside the run directory",
    )
    parser.add_argument(
        "--initial-state-policy",
        choices=("require", "compute-if-missing", "recompute"),
        default="compute-if-missing",
    )
    parser.add_argument("--intensity-wcm2", type=float, default=3.0e13)
    parser.add_argument("--wavelength-nm", type=float, default=800.0)
    parser.add_argument("--carrier-cycles", type=float, default=3.0)
    parser.add_argument(
        "--field-gauge",
        choices=("length", "velocity"),
        default="length",
    )
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument(
        "--final-time",
        type=float,
        required=True,
        help=(
            "requested propagation endpoint in a.u.; the actual endpoint is "
            "rounded up to the next complete timestep"
        ),
    )
    parser.add_argument("--cap-alpha", type=float, default=0.8)
    parser.add_argument("--cap-eta-r", type=float, default=1.0)
    parser.add_argument("--cap-eta-R", type=float, default=1.0)
    parser.add_argument("--no-cap", action="store_true")
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument(
        "--wavefunction-directory",
        type=Path,
        help="snapshot directory; defaults to RUN_DIRECTORY/wavefunctions",
    )
    parser.add_argument(
        "--wavefunction-snapshot-interval",
        type=int,
        help="optional accepted-step interval in addition to initial/final states",
    )
    parser.add_argument("--no-wavefunction-snapshots", action="store_true")
    parser.add_argument(
        "--figure-path",
        type=Path,
        help="diagnostic figure path; defaults to RUN_DIRECTORY/dynamics.png",
    )
    parser.add_argument(
        "--show-figure",
        dest="show_figure",
        action="store_true",
        default=True,
        help="display the saved diagnostic figure interactively (default)",
    )
    parser.add_argument(
        "--no-show-figure",
        dest="show_figure",
        action="store_false",
        help="save the diagnostic figure without displaying it",
    )
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-11)
    parser.add_argument("--krylov-atol", type=float, default=1.0e-13)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--run-directory", type=Path, default=Path("h2p-end-to-end"))
    parser.add_argument("--overwrite", action="store_true")


def validate_arguments(parser: argparse.ArgumentParser, arguments) -> None:
    """Validate shared CLI options."""

    if arguments.l_max < 1:
        parser.error("--l-max must be at least one for field-driven dynamics")
    if arguments.M_tot != 0:
        parser.error("the current non-BO dynamics initialization requires --M-tot 0")
    for name in (
        "intensity_wcm2",
        "wavelength_nm",
        "carrier_cycles",
        "time_step",
        "cap_eta_r",
        "cap_eta_R",
        "krylov_rtol",
    ):
        value = getattr(arguments, name)
        if not isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < arguments.cap_alpha < 1.0:
        parser.error("--cap-alpha must lie strictly between zero and one")
    if arguments.krylov_atol < 0.0 or not isfinite(arguments.krylov_atol):
        parser.error("--krylov-atol must be finite and nonnegative")
    if not isfinite(arguments.final_time) or arguments.final_time <= 0.0:
        parser.error("--final-time must be positive")
    for name in (
        "sample_interval",
        "checkpoint_interval",
        "progress_interval",
    ):
        if getattr(arguments, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    interval = arguments.wavefunction_snapshot_interval
    if interval is not None and interval < 1:
        parser.error("--wavefunction-snapshot-interval must be positive")
    for option, nodes in (
        ("electronic", arguments.electronic_nodes),
        ("internuclear", arguments.internuclear_nodes),
    ):
        if len(nodes) < 2 or nodes[0] != 0.0:
            parser.error(f"--{option}-nodes must start at zero")
        if any(not isfinite(value) for value in nodes):
            parser.error(f"--{option}-nodes must contain finite values")
        if any(right <= left for left, right in zip(nodes, nodes[1:])):
            parser.error(f"--{option}-nodes must be strictly increasing")
    for name in (
        "electronic_points_per_element",
        "internuclear_points_per_element",
    ):
        if getattr(arguments, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be at least two")


def grid(nodes, points_per_element) -> FEMDVRGridSpec:
    return FEMDVRGridSpec(nodes, [points_per_element] * (len(nodes) - 1))


def prepare_run_directory(path: Path, *, overwrite: bool) -> Path:
    destination = path.resolve()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"path exists and is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"run directory is not empty: {destination}; pass --overwrite"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def resolved_paths(arguments, run_directory: Path) -> dict[str, Path | None]:
    label = arguments.system.replace("+", "p")
    ground_state = arguments.ground_state_path
    if ground_state is None:
        ground_state = run_directory / f"ground-state-{label}-l{arguments.l_max}.npz"
    wavefunctions = None
    if not arguments.no_wavefunction_snapshots:
        wavefunctions = arguments.wavefunction_directory
        if wavefunctions is None:
            wavefunctions = run_directory / "wavefunctions"
    checkpoint = None
    if not arguments.no_checkpoints:
        checkpoint = run_directory / "checkpoint.npz"
    return {
        "ground_state": Path(ground_state).resolve(),
        "wavefunctions": (
            None if wavefunctions is None else Path(wavefunctions).resolve()
        ),
        "checkpoint": checkpoint,
        "figure": (
            run_directory / "dynamics.png"
            if arguments.figure_path is None
            else arguments.figure_path.resolve()
        ),
    }


def physical_setup(arguments):
    """Build grids, parameters, pulse, and propagation endpoint."""

    electronic_grid = grid(
        arguments.electronic_nodes,
        arguments.electronic_points_per_element,
    )
    internuclear_grid = grid(
        arguments.internuclear_nodes,
        arguments.internuclear_points_per_element,
    )
    parameters = _diatomic_parameters(arguments.system, arguments.codata_set)
    pair_exchange_parity = _pair_exchange_parity(
        1 if parameters.has_identical_pair else None,
        parameters,
    )
    field_strength = intensity_wcm2_to_field_au(
        arguments.intensity_wcm2,
        codata_set=arguments.codata_set,
    )
    omega = wavelength_nm_to_omega_au(
        arguments.wavelength_nm,
        codata_set=arguments.codata_set,
    )
    pulse = SineSquareLaser(
        E0=field_strength,
        omega=omega,
        ncycles=arguments.carrier_cycles,
        gauge=arguments.field_gauge,
        phase=0.0,
        t0=0.0,
    )
    requested_final_time = float(arguments.final_time)
    step_ratio = requested_final_time / arguments.time_step
    nearest_step_count = round(step_ratio)
    if np.isclose(
        step_ratio,
        nearest_step_count,
        rtol=0.0,
        atol=16.0 * np.finfo(float).eps * max(1.0, abs(step_ratio)),
    ):
        n_steps = int(nearest_step_count)
    else:
        n_steps = int(np.ceil(step_ratio))
    final_time = n_steps * arguments.time_step
    return {
        "electronic_grid": electronic_grid,
        "internuclear_grid": internuclear_grid,
        "parameters": parameters,
        "pair_exchange_parity": pair_exchange_parity,
        "field_strength": float(field_strength),
        "omega": float(omega),
        "pulse": pulse,
        "pulse_duration": float(pulse.tprime),
        "requested_final_time": requested_final_time,
        "final_time": float(final_time),
        "n_steps": n_steps,
    }


def dynamics_representation(arguments, setup):
    """Build the full-parity dynamics representation used by MPI."""

    p = setup["parameters"]
    pair_parity = setup["pair_exchange_parity"]
    electron_basis = AngularBasisSpec(arguments.l_max, arguments.l_max)
    pair_basis = AngularBasisSpec(
        arguments.l_max,
        arguments.l_max,
        coordinate_parity=pair_parity if p.has_identical_pair else None,
    )
    specification = TwoRadialSystemSpec(
        coordinate1=CoordinateSpec(
            grid=setup["electronic_grid"],
            basis=electron_basis,
            kinetic_prefactor=-0.5 / p.electronic_reduced_mass,
        ),
        coordinate2=CoordinateSpec(
            grid=setup["internuclear_grid"],
            basis=pair_basis,
            kinetic_prefactor=-0.5 / p.internuclear_reduced_mass,
            one_body_terms=(
                radial_potential(
                    lambda radius: p.reference_charge
                    * p.partner_charge
                    / radius
                ),
            ),
        ),
        multipoles=CoulombMultipoleSpec(
            2 * arguments.l_max,
            2 * arguments.l_max,
            lambda_parity=1 if p.has_identical_pair else None,
        ),
        coulomb_terms=(
            CoulombTerm(
                p.reference_center_scale,
                p.electron_charge * p.reference_charge,
            ),
            CoulombTerm(
                p.partner_center_scale,
                p.electron_charge * p.partner_charge,
            ),
        ),
        M_tot=arguments.M_tot,
        radial_coulomb_method="poisson",
    )
    return specification.build()


def absorbing_boundaries(arguments, representation):
    if arguments.no_cap:
        return TwoRadialAbsorbingBoundaries(representation)
    return TwoRadialAbsorbingBoundaries(
        representation,
        cap1=CubicCAP(alpha=arguments.cap_alpha, eta=arguments.cap_eta_r),
        cap2=CubicCAP(alpha=arguments.cap_alpha, eta=arguments.cap_eta_R),
    )


def physical_state_metadata(setup) -> dict[str, object]:
    return {
        "system_family": "non_bo_diatomic_ion",
        "system": setup["parameters"].name,
        "parameters": asdict(setup["parameters"]),
    }


def configuration_record(arguments, setup) -> dict[str, object]:
    return {
        "system": arguments.system,
        "l_max": arguments.l_max,
        "m_max": arguments.l_max,
        "M_tot": arguments.M_tot,
        "ground_state_symmetry": {
            "inversion_parity": 1,
            "pair_exchange_parity": setup["pair_exchange_parity"],
            "symmetry_adapted_basis": True,
        },
        "codata_set": arguments.codata_set,
        "electronic_nodes": list(arguments.electronic_nodes),
        "electronic_points_per_element": arguments.electronic_points_per_element,
        "internuclear_nodes": list(arguments.internuclear_nodes),
        "internuclear_points_per_element": arguments.internuclear_points_per_element,
        "field": {
            "type": "SineSquareLaser",
            "gauge": arguments.field_gauge,
            "intensity_wcm2": arguments.intensity_wcm2,
            "field_strength_au": setup["field_strength"],
            "wavelength_nm": arguments.wavelength_nm,
            "omega_au": setup["omega"],
            "carrier_cycles": arguments.carrier_cycles,
            "pulse_duration_au": setup["pulse_duration"],
        },
        "absorber": None
        if arguments.no_cap
        else {
            "type": "CubicCAP",
            "alpha": arguments.cap_alpha,
            "eta_r": arguments.cap_eta_r,
            "eta_R": arguments.cap_eta_R,
        },
        "integrator": "sia",
        "time_step": arguments.time_step,
        "requested_final_time": setup["requested_final_time"],
        "final_time": setup["final_time"],
        "sample_interval": arguments.sample_interval,
        "checkpoint_interval": (
            None if arguments.no_checkpoints else arguments.checkpoint_interval
        ),
        "wavefunction_snapshot_interval": (
            None
            if arguments.no_wavefunction_snapshots
            else arguments.wavefunction_snapshot_interval
        ),
    }


def write_json(path: Path, values) -> Path:
    path.write_text(
        json.dumps(values, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def print_initial_observables(
    snapshot,
    *,
    codata_set: str,
    electronic_grid: FEMDVRGridSpec,
    internuclear_grid: FEMDVRGridSpec,
) -> None:
    """Print the common physical observables before propagation starts."""

    conditional = snapshot.conditional
    numerators = snapshot.numerators
    pair_exchange = conditional.pair_exchange

    def vector(values) -> str:
        return "(" + ", ".join(f"{value:.12f}" for value in values) + ")"

    def nodes(values) -> str:
        return "[" + ", ".join(f"{value:g}" for value in values) + "]"

    def points_per_element(values) -> str:
        values = tuple(int(value) for value in values)
        if len(set(values)) == 1:
            return str(values[0])
        return "[" + ", ".join(str(value) for value in values) + "]"

    print("\nInitial physical observables")
    print(f"CODATA set:               {codata_set}")
    print(f"Electronic grid nodes:    {nodes(electronic_grid.nodes)}")
    print(
        "Electronic points/element: "
        + points_per_element(electronic_grid.n_points)
    )
    print(f"Internuclear grid nodes:  {nodes(internuclear_grid.nodes)}")
    print(
        "Internuclear points/element: "
        + points_per_element(internuclear_grid.n_points)
    )
    print(f"Energy:                   {snapshot.conditional_energy:.12f} Eh")
    print(f"<L^2>:                    {conditional.total_l_squared:.12f}")
    print(f"<L_z>:                    {conditional.total_l_z:.12f}")
    print(f"<cos^2(theta_R)>:         {conditional.cos2_theta_R:.12f}")
    print(f"<cos(theta_R)>:           {conditional.cos_theta_R:.12f}")
    print(f"<R^2>:                    {numerators.R_squared:.12f} a0^2")
    print(f"<cos^4(theta_R)>:         {numerators.cos4_theta_R:.12f}")
    print(f"<cos^6(theta_R)>:         {numerators.cos6_theta_R:.12f}")
    print(f"<cos^8(theta_R)>:         {numerators.cos8_theta_R:.12f}")
    print(
        "<R cos^2(theta_R)>:     "
        f"{numerators.R_cos2_theta_R:.12f} a0"
    )
    print(f"<I>:                      {conditional.inversion:.12f}")
    print(
        "<P>:                      "
        + ("N/A" if pair_exchange is None else f"{pair_exchange:.12f}")
    )
    print(f"Electronic dipole:        {vector(conditional.dipole_r)} ea0")
    print(f"Protonic dipole:          {vector(conditional.dipole_R)} ea0")
    print(f"Total dipole:             {vector(conditional.dipole_total)} ea0")


def plot_dynamics(
    time_series,
    pulse,
    path: Path,
    *,
    show: bool = False,
    overwrite: bool = False,
) -> Path:
    """Save the common Phase 9h field and observable diagnostic figure."""

    import matplotlib.pyplot as plt

    samples = time_series.samples
    if not samples:
        raise ValueError("cannot plot an empty dynamics time series")
    times = time_series.times
    electric_field = np.asarray(pulse.electric_field(times), dtype=float)
    dipole = np.asarray(
        [sample.snapshot.conditional.dipole_total[2] for sample in samples],
        dtype=float,
    )
    induced_dipole = dipole - dipole[0]
    energy = np.asarray(
        [sample.snapshot.conditional_energy for sample in samples],
        dtype=float,
    )
    norms = np.asarray(
        [sample.snapshot.norm for sample in samples],
        dtype=float,
    )
    norm_loss = 1.0 - norms
    plotted_norm_loss = np.maximum(norm_loss, np.finfo(float).eps)

    figure, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(8.0, 10.0),
        constrained_layout=True,
    )
    axes[0].plot(times, electric_field, color="red")
    axes[0].set_ylabel(r"$E(t)$ (a.u.)")
    axes[1].plot(times, induced_dipole)
    axes[1].set_ylabel(r"$D_z(t)-D_z(0)$ (a.u.)")
    axes[2].plot(times, energy)
    axes[2].set_ylabel(r"$E_{\mathrm{phys}}(t)$ (Eh)")
    axes[3].semilogy(times, plotted_norm_loss)
    axes[3].set_ylabel(r"$1-\|\Psi(t)\|^2$")
    axes[3].set_xlabel("Time (a.u.)")
    for axis in axes:
        axis.grid(alpha=0.3)

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"diagnostic figure already exists: {destination}; pass --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return destination


def ground_state_options() -> dict[str, object]:
    return {
        "max_iterations": 500,
        "residual_tolerance": 1.0e-8,
        "maximum_subspace_dimension": 30,
        "retained_ritz_vectors": 4,
        "davidson_preconditioner": "one-body-spectral",
        "progress_interval": 10,
        "sample_interval": 10,
    }
