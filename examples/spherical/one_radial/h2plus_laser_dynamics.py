"""Run a short clamped-nuclei H2+ laser-dynamics calculation.

The molecule lies on the laboratory z axis with internuclear distance
``R=2`` bohr. The defaults are deliberately compact smoke-test inputs, not
convergence recommendations. Converge the radial grid, angular cutoff, time
step, pulse, and absorber before using an observable quantitatively.

All quantities are in atomic units unless stated otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from electromagnetic_fields.dipole_fields import SineSquareLaser
from grid_lib.spherical_coordinates import CubicCAP, RadialCosineMask
from matter_field_params import wavelength_nm_to_omega_au

from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    OneElectronDiatomicTDSE,
    diagonalize_fixed_m,
)


DEFAULT_FREQUENCY = 4.0
FREQUENCY_CODATA_SET = "2018"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gauge", choices=("length", "velocity"), default="length"
    )
    parser.add_argument(
        "--integrator",
        choices=("crank-nicolson", "sil", "sia"),
        default="sia",
    )
    parser.add_argument(
        "--absorber", choices=("none", "mask", "cap"), default="none"
    )
    parser.add_argument("--particle-mass", type=float, default=1.0)
    parser.add_argument("--field-strength", type=float, default=0.01)
    frequency = parser.add_mutually_exclusive_group()
    frequency.add_argument(
        "--frequency",
        type=float,
        default=None,
        help=f"carrier angular frequency; default is {DEFAULT_FREQUENCY:g}",
    )
    frequency.add_argument(
        "--wavelength-nm",
        type=float,
        default=None,
        help="carrier wavelength converted using CODATA 2018",
    )
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--element-width", type=float, default=2.0)
    parser.add_argument("--points-per-element", type=int, default=7)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument(
        "--nuclear-method",
        choices=("analytic", "poisson"),
        default="analytic",
    )
    parser.add_argument(
        "--radial-probability-interval",
        type=int,
        default=1,
        help="accepted-step interval; zero disables radial sampling",
    )
    parser.add_argument("--mask-alpha", type=float, default=0.8)
    parser.add_argument("--mask-n", type=float, default=4.0)
    parser.add_argument("--cap-alpha", type=float, default=0.8)
    parser.add_argument("--cap-eta", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-13)
    parser.add_argument("--max-krylov-dimension", type=int, default=30)
    parser.add_argument("--max-krylov-refinements", type=int, default=8)
    parser.add_argument("--max-linear-iterations", type=int, default=500)
    parser.add_argument("--disable-cn-preconditioner", action="store_true")
    parser.add_argument(
        "--progress", choices=("bar", "log", "none"), default="bar"
    )
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=None,
        help="compatible NPY state or NPZ containing final_state/state",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="rolling checkpoint"
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--overwrite-checkpoint", action="store_true")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("h2plus_laser_dynamics.npz"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("h2plus_laser_dynamics.json"),
    )
    return parser.parse_args(argv)


def radial_nodes(r_max, element_width, *, nuclear_radius=1.0):
    """Return radial elements with the nuclear radius as a boundary."""

    if not np.isfinite(r_max) or r_max <= nuclear_radius:
        raise ValueError("r_max must be finite and exceed the nuclear radius")
    if not np.isfinite(element_width) or element_width <= 0.0:
        raise ValueError("element_width must be finite and positive")
    n_outer = int(np.ceil((r_max - nuclear_radius) / element_width))
    outer = np.linspace(nuclear_radius, r_max, n_outer + 1)
    return np.concatenate(([0.0], outer))


def _positive(value, name):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _nonnegative_integer(value, name):
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _positive_integer(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _frequency(arguments):
    if arguments.wavelength_nm is not None:
        wavelength = _positive(arguments.wavelength_nm, "wavelength_nm")
        return (
            wavelength_nm_to_omega_au(
                wavelength, codata_set=FREQUENCY_CODATA_SET
            ),
            wavelength,
        )
    value = (
        DEFAULT_FREQUENCY
        if arguments.frequency is None
        else arguments.frequency
    )
    return _positive(value, "frequency"), None


def _initial_state(path, hamiltonian):
    if path is None:
        eigenstate = diagonalize_fixed_m(hamiltonian, m=0, n_states=1)
        metadata = {
            "source": "computed_projected_ground_state",
            "electronic_energy": eigenstate.electronic_energy,
            "total_energy": eigenstate.total_energy,
        }
        return np.array(eigenstate.orbital, copy=True), metadata

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"initial-state archive not found: {source}")
    loaded = np.load(source, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            for key in ("final_state", "state", "initial_state"):
                if key in loaded:
                    state = np.array(loaded[key], copy=True)
                    break
            else:
                raise ValueError(
                    "NPZ initial state must contain final_state, state, "
                    "or initial_state"
                )
        finally:
            loaded.close()
    else:
        state = np.array(loaded, copy=True)
    hamiltonian.validate_fixed_m_state(state, 0)
    return state, {"source": "loaded_archive", "path": str(source.resolve())}


def _absorber(arguments, r_max):
    if arguments.absorber == "none":
        return None, None, {"type": "none"}
    if arguments.absorber == "mask":
        alpha = _positive(arguments.mask_alpha, "mask_alpha")
        if alpha >= 1.0:
            raise ValueError("mask_alpha must be smaller than one")
        mask = RadialCosineMask(
            r0=alpha * r_max,
            n=_positive(arguments.mask_n, "mask_n"),
        )
        return mask, None, {
            "type": "RadialCosineMask",
            "alpha": alpha,
            "r0": alpha * r_max,
            "n": float(arguments.mask_n),
        }
    alpha = _positive(arguments.cap_alpha, "cap_alpha")
    if alpha >= 1.0:
        raise ValueError("cap_alpha must be smaller than one")
    cap = CubicCAP(
        alpha=alpha,
        eta=_positive(arguments.cap_eta, "cap_eta"),
    )
    return None, cap, {
        "type": "CubicCAP",
        "alpha": alpha,
        "eta": float(arguments.cap_eta),
    }


def _integrator_options(arguments):
    if arguments.integrator == "crank-nicolson":
        return {
            "rtol": _positive(arguments.rtol, "rtol"),
            "atol": _positive(arguments.atol, "atol"),
            "max_iterations": _positive_integer(
                arguments.max_linear_iterations, "max_linear_iterations"
            ),
            "preconditioner": not arguments.disable_cn_preconditioner,
        }
    return {
        "max_krylov_dimension": _positive_integer(
            arguments.max_krylov_dimension, "max_krylov_dimension"
        ),
        "rtol": _positive(arguments.rtol, "rtol"),
        "atol": _positive(arguments.atol, "atol"),
        "max_refinements": _positive_integer(
            arguments.max_krylov_refinements, "max_krylov_refinements"
        ),
    }


def _write_outputs(arguments, calculation, result, metadata):
    radial = calculation.radial_probability_series
    archive_values = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "times": result.times,
        "norm": result.norm_history,
        "final_state": result.state,
        "linear_iterations": result.linear_iterations,
        "linear_residual_norms": result.linear_residual_norms,
        "krylov_operator_calls": result.krylov_operator_calls,
        "krylov_dimensions": result.krylov_dimensions,
        "step_wall_times": result.step_wall_times,
        "radial_points": radial.radial_points,
        "radial_weights": radial.radial_weights,
        "radial_times": radial.times,
        "radial_norms": radial.norms,
        "radial_probability": radial.probabilities,
    }
    archive_values.update(result.samples)
    arguments.archive.parent.mkdir(parents=True, exist_ok=True)
    arguments.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.archive, **archive_values)
    arguments.summary.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    arguments = parse_arguments(argv)
    particle_mass = _positive(arguments.particle_mass, "particle_mass")
    field_strength = _positive(arguments.field_strength, "field_strength")
    cycles = _positive(arguments.cycles, "cycles")
    time_step = _positive(arguments.time_step, "time_step")
    r_max = _positive(arguments.r_max, "r_max")
    if arguments.points_per_element < 2:
        raise ValueError("points_per_element must be at least two")
    if arguments.l_max < 0:
        raise ValueError("l_max must be non-negative")
    radial_interval = _nonnegative_integer(
        arguments.radial_probability_interval,
        "radial_probability_interval",
    )
    if arguments.absorber == "cap" and arguments.integrator != "sia":
        raise ValueError("the cubic CAP currently requires --integrator sia")

    omega, wavelength_nm = _frequency(arguments)
    nodes = radial_nodes(r_max, arguments.element_width)
    n_points = [arguments.points_per_element] * (nodes.size - 1)
    system = OneElectronDiatomicSystem(
        nuclear_charge1=1.0,
        nuclear_charge2=1.0,
        half_bond_length=1.0,
        particle_mass=particle_mass,
        name="H2+",
    )
    stationary = OneElectronDiatomicHamiltonian(
        system,
        nodes=nodes,
        n_points=n_points,
        l_max=arguments.l_max,
        m_max=0,
        nuclear_L_max=2 * arguments.l_max,
        nuclear_method=arguments.nuclear_method,
    )
    initial_state, state_metadata = _initial_state(
        arguments.initial_state, stationary
    )
    pulse = SineSquareLaser(
        E0=field_strength,
        omega=omega,
        ncycles=cycles,
        gauge=arguments.gauge,
        phase=arguments.phase,
        t0=0.0,
    )
    mask, cap, absorber_metadata = _absorber(arguments, r_max)
    progress_interval = (
        arguments.progress_interval if arguments.progress == "log" else None
    )
    calculation = OneElectronDiatomicTDSE(
        system,
        nodes=nodes,
        n_points=n_points,
        l_max=arguments.l_max,
        m=0,
        nuclear_L_max=2 * arguments.l_max,
        nuclear_method=arguments.nuclear_method,
        initial_state=initial_state,
        fields=pulse,
        field_gauge=arguments.gauge,
        integrator=arguments.integrator,
        integrator_options=_integrator_options(arguments),
        initial_time=0.0,
        final_time=pulse.tprime,
        time_step=time_step,
        mask=mask,
        cap=cap,
        reference_states={"initial": initial_state},
        radial_probability_interval=(
            None if radial_interval == 0 else radial_interval
        ),
        checkpoint_path=arguments.checkpoint,
        checkpoint_interval=(
            None
            if arguments.checkpoint is None
            else arguments.checkpoint_interval
        ),
        checkpoint_overwrite=arguments.overwrite_checkpoint,
        checkpoint_metadata={"purpose": "one-radial H2+ development example"},
        progress_interval=progress_interval,
        progress_bar=arguments.progress == "bar",
    )
    result = calculation.run(progress=arguments.progress != "none")

    metadata = {
        "purpose": "one-radial clamped-nuclei H2+ development dynamics",
        "warning": "smoke-test inputs; not a convergence recommendation",
        "system": calculation.compatibility_metadata,
        "grid": {
            "nodes": nodes.tolist(),
            "points_per_element": int(arguments.points_per_element),
            "active_radial_points": int(calculation.hamiltonian.r.size),
            "l_max": int(arguments.l_max),
            "m": 0,
            "nuclear_L_max": 2 * int(arguments.l_max),
        },
        "pulse": {
            "type": "SineSquareLaser",
            "gauge": arguments.gauge,
            "field_strength": field_strength,
            "angular_frequency": omega,
            "wavelength_nm": wavelength_nm,
            "frequency_codata_set": (
                FREQUENCY_CODATA_SET if wavelength_nm is not None else None
            ),
            "cycles": cycles,
            "phase": float(arguments.phase),
            "duration": float(pulse.tprime),
        },
        "integrator": {
            "name": arguments.integrator,
            "options": calculation.dynamics_metadata["integrator_options"],
            "requested_time_step": result.requested_time_step,
            "effective_time_step": result.time_step,
            "accepted_steps": result.n_steps,
        },
        "absorber": absorber_metadata,
        "initial_state": state_metadata,
        "results": {
            "initial_norm": result.initial_norm,
            "final_norm": result.norm,
            "norm_change": result.norm - result.initial_norm,
            "final_initial_state_population": float(
                result.samples["state_populations"][-1, 0]
            ),
        },
        "outputs": {
            "archive": str(arguments.archive.resolve()),
            "summary": str(arguments.summary.resolve()),
            "checkpoint": (
                None
                if arguments.checkpoint is None
                else str(arguments.checkpoint.resolve())
            ),
        },
    }
    _write_outputs(arguments, calculation, result, metadata)

    print("\n** One-radial clamped-nuclei H2+ dynamics **")
    print(f"Grid nodes:                 {nodes.tolist()}")
    print(f"Points per element:         {arguments.points_per_element}")
    print(f"Active radial points:       {calculation.hamiltonian.r.size}")
    print(f"Angular basis:              l_max={arguments.l_max}, m=0")
    print(f"Particle mass:              {particle_mass:.12g}")
    print(f"State source:               {state_metadata['source']}")
    print(f"Pulse:                      SineSquareLaser, {cycles:g} cycle(s)")
    print(f"Gauge:                      {arguments.gauge}")
    print(f"Field strength:             {field_strength:.12g}")
    print(f"Angular frequency:          {omega:.12g}")
    print(f"Integrator:                 {arguments.integrator}")
    print(f"Absorber:                   {arguments.absorber}")
    print(f"Effective time step:        {result.time_step:.12g}")
    print(f"Initial norm:               {result.initial_norm:.12g}")
    print(f"Final norm:                 {result.norm:.12g}")
    print(f"Archive:                    {arguments.archive.resolve()}")
    print(f"Summary:                    {arguments.summary.resolve()}")
    print(
        "Checkpoint:                 "
        + (
            "disabled"
            if arguments.checkpoint is None
            else str(arguments.checkpoint.resolve())
        )
    )
    return metadata


if __name__ == "__main__":
    main()
