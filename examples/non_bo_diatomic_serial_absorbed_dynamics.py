"""Validate serial non-BO masks, CAP/SIA, densities, and restart.

The compact default problem is intended as the runnable Phase 9f gate.  It
computes or reuses a Davidson ground state, propagates it with separate
electronic and internuclear masks, repeats the calculation with CAPs and SIA,
and resumes the masked calculation from an intermediate checkpoint.  Raw
radial marginals are written as compressed NPZ archives.

For example::

    python examples/non_bo_diatomic_serial_absorbed_dynamics.py \
        --run-directory phase9f-smoke \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

import numpy as np

from grid_tdse.real_time import load_real_time_checkpoint
from grid_tdse.spherical.two_radial import (
    DiatomicIonDynamicsSamplingPolicy,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    phase_aligned_two_radial_state_difference,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), default="H2+")
    parser.add_argument("--l-max", type=int, default=1)
    parser.add_argument(
        "--electronic-nodes",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 2.0),
    )
    parser.add_argument("--electronic-points-per-element", type=int, default=3)
    parser.add_argument(
        "--internuclear-nodes",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 2.0),
    )
    parser.add_argument("--internuclear-points-per-element", type=int, default=3)
    parser.add_argument("--codata-set", choices=("2018", "2022"), default="2022")
    parser.add_argument("--ground-state-path", type=Path)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=0.04)
    parser.add_argument("--mask-strength", type=float, default=0.15)
    parser.add_argument("--cap-strength", type=float, default=0.10)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=Path("phase9f_serial_absorbed_dynamics"),
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    if arguments.l_max < 0:
        parser.error("--l-max must be nonnegative")
    for name in ("time_step", "final_time", "mask_strength", "cap_strength"):
        value = getattr(arguments, name)
        if not isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.mask_strength >= 1.0:
        parser.error("--mask-strength must be smaller than one")
    steps = arguments.final_time / arguments.time_step
    if not np.isclose(steps, round(steps), atol=1.0e-12) or round(steps) < 4:
        parser.error(
            "--final-time must contain at least four complete --time-step steps"
        )
    if round(steps) % 2:
        parser.error("the number of propagation steps must be even")
    if arguments.sample_interval < 1:
        parser.error("--sample-interval must be positive")
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
    return arguments


def _grid(nodes, points_per_element) -> FEMDVRGridSpec:
    return FEMDVRGridSpec(nodes, [points_per_element] * (len(nodes) - 1))


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"run directory is not empty: {path}; pass --overwrite to reuse it"
        )
    path.mkdir(parents=True, exist_ok=True)


def _radial_profile(points, outer_radius, strength, *, cap):
    scaled = np.asarray(points) / float(outer_radius)
    profile = strength * scaled**4
    return -1j * profile if cap else 1.0 - profile


def _density_errors(calculation):
    system = calculation.representation
    norm_error = 0.0
    moment_error = 0.0
    for sample in calculation.dynamics_time_series.samples:
        P_r = np.asarray(sample.snapshot.radial_densities.P_r)
        P_R = np.asarray(sample.snapshot.radial_densities.P_R)
        norm = sample.snapshot.norm
        norm_error = max(
            norm_error,
            abs(np.dot(system.grid1.weights, P_r) - norm),
            abs(np.dot(system.grid2.weights, P_R) - norm),
        )
        moment_error = max(
            moment_error,
            abs(
                np.dot(system.grid1.weights * system.grid1.r, P_r) / norm
                - sample.snapshot.conditional.r
            ),
            abs(
                np.dot(system.grid2.weights * system.grid2.r, P_R) / norm
                - sample.snapshot.conditional.R
            ),
        )
    return float(norm_error), float(moment_error)


arguments = _arguments()
run_directory = arguments.run_directory.resolve()
_prepare_directory(run_directory, overwrite=arguments.overwrite)
snapshot_directory = run_directory / "masked-snapshots"
snapshot_directory.mkdir(exist_ok=True)
electronic_grid = _grid(
    arguments.electronic_nodes,
    arguments.electronic_points_per_element,
)
internuclear_grid = _grid(
    arguments.internuclear_nodes,
    arguments.internuclear_points_per_element,
)
system_label = arguments.system.replace("+", "p")
ground_state_path = (
    run_directory / f"ground-state-{system_label}-l{arguments.l_max}.npz"
    if arguments.ground_state_path is None
    else arguments.ground_state_path
)
pair_exchange_parity = None if arguments.system == "HD+" else 1
checkpoint_path = run_directory / "masked.chkpt"
restart_checkpoint_path = run_directory / "masked-resumed.chkpt"
result_path = run_directory / "result.json"


def _calculation(
    *,
    initial_state,
    integrator,
    initial_time=0.0,
    mask_r=None,
    mask_R=None,
    cap_r=None,
    cap_R=None,
    checkpoint=None,
    snapshot_directory=None,
):
    automatic = initial_state is None
    return NonBornOppenheimerDiatomicIonTDSE(
        arguments.system,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        initial_state=initial_state,
        pair_exchange_parity=pair_exchange_parity,
        initial_state_policy="compute-if-missing" if automatic else "require",
        ground_state_path=ground_state_path if automatic else None,
        ground_state_symmetry_adapted_basis=(arguments.system != "HD+"),
        ground_state_options=(
            {
                "max_iterations": 500,
                "residual_tolerance": 1.0e-8,
                "maximum_subspace_dimension": 30,
                "retained_ritz_vectors": 4,
                "davidson_preconditioner": "one-body-spectral",
                "progress_interval": 10,
                "sample_interval": 10,
            }
            if automatic
            else None
        ),
        fields=(),
        field_gauge="length",
        l_max=arguments.l_max,
        m_max=arguments.l_max,
        lambda_max=2 * arguments.l_max,
        mu_max=2 * arguments.l_max,
        M_tot=0,
        codata_set=arguments.codata_set,
        radial_coulomb_method="poisson",
        radial_operator_mode="element-local",
        integrator=integrator,
        integrator_options={
            "max_krylov_dimension": 20,
            "rtol": 1.0e-11,
            "atol": 1.0e-13,
            "max_refinements": 8,
            "reorthogonalize": True,
            "breakdown_tolerance": 1.0e-14,
        },
        initial_time=initial_time,
        final_time=arguments.final_time,
        time_step=arguments.time_step,
        mask_r=mask_r,
        mask_R=mask_R,
        cap_r=cap_r,
        cap_R=cap_R,
        observables=None,
        dynamics_sampling_policy=DiatomicIonDynamicsSamplingPolicy(
            interval=arguments.sample_interval
        ),
        checkpoint_path=checkpoint,
        checkpoint_interval=2 if checkpoint is not None else None,
        checkpoint_overwrite=arguments.overwrite or initial_time > 0.0,
        checkpoint_metadata={"phase": "9f", "example": Path(__file__).name},
        wavefunction_snapshot_directory=snapshot_directory,
        wavefunction_snapshot_interval=(
            2 if snapshot_directory is not None else None
        ),
        wavefunction_snapshot_prefix="masked",
        wavefunction_snapshot_overwrite=arguments.overwrite,
        wavefunction_snapshot_metadata={
            "code_provenance": {"example": Path(__file__).name}
        },
        progress_interval=10,
    )


print("Preparing the shared Davidson initial state", flush=True)
preparation = _calculation(initial_state=None, integrator="sil")
initial_archive = preparation.initialization.archive
system = preparation.representation
mask_r = _radial_profile(
    system.grid1.r,
    arguments.electronic_nodes[-1],
    arguments.mask_strength,
    cap=False,
)
mask_R = _radial_profile(
    system.grid2.r,
    arguments.internuclear_nodes[-1],
    arguments.mask_strength,
    cap=False,
)
cap_r = _radial_profile(
    system.grid1.r,
    arguments.electronic_nodes[-1],
    arguments.cap_strength,
    cap=True,
)
cap_R = _radial_profile(
    system.grid2.r,
    arguments.internuclear_nodes[-1],
    arguments.cap_strength,
    cap=True,
)

print("Running masked SIL propagation", flush=True)
masked = _calculation(
    initial_state=initial_archive,
    integrator="sil",
    mask_r=mask_r,
    mask_R=mask_R,
    checkpoint=checkpoint_path,
    snapshot_directory=snapshot_directory,
)
masked_result = masked.run(progress=True)
masked_density_path = masked.dynamics_time_series.save_radial_densities(
    run_directory / "masked-radial-densities.npz",
    overwrite=arguments.overwrite,
)

print("Running CAP/SIA propagation", flush=True)
cap = _calculation(
    initial_state=initial_archive,
    integrator="sia",
    cap_r=cap_r,
    cap_R=cap_R,
)
cap_result = cap.run(progress=True)
cap_density_path = cap.dynamics_time_series.save_radial_densities(
    run_directory / "cap-radial-densities.npz",
    overwrite=arguments.overwrite,
)

interrupted = load_real_time_checkpoint(masked.wavefunction_snapshots[1].path)
print(f"Resuming masked propagation from step {interrupted.current_step}", flush=True)
resumed = _calculation(
    initial_state=interrupted.state_copy(),
    integrator="sil",
    initial_time=interrupted.current_time,
    mask_r=mask_r,
    mask_R=mask_R,
    checkpoint=restart_checkpoint_path,
)
resumed_result = resumed.resume(interrupted, progress=True)

restart_state_difference = phase_aligned_two_radial_state_difference(
    masked_result.state,
    resumed_result.state,
    system.inner_product,
)
mask_density_norm_error, mask_density_moment_error = _density_errors(masked)
cap_density_norm_error, cap_density_moment_error = _density_errors(cap)
cap_final_sample = cap.dynamics_time_series.samples[-1]
cap_physical_energy = system.inner_product(
    cap_result.state,
    cap.field_free_hamiltonian(cap_result.state),
).real
cap_energy_error = abs(
    cap_final_sample.snapshot.energy_numerator - cap_physical_energy
)
mask_monotone = bool(np.all(np.diff(masked_result.norm_history) < 0.0))
cap_monotone = bool(np.all(np.diff(cap_result.norm_history) < 0.0))
status = (
    "PASS"
    if mask_monotone
    and cap_monotone
    and masked_result.norm < masked_result.initial_norm
    and cap_result.norm < cap_result.initial_norm
    and mask_density_norm_error < 2.0e-12
    and cap_density_norm_error < 2.0e-12
    and mask_density_moment_error < 2.0e-12
    and cap_density_moment_error < 2.0e-12
    and cap_energy_error < 2.0e-12
    and restart_state_difference < 2.0e-12
    else "FAIL"
)
summary = {
    "phase": "9f",
    "status": status,
    "system": arguments.system,
    "l_max": arguments.l_max,
    "time_step": arguments.time_step,
    "final_time": arguments.final_time,
    "masked": {
        "initial_norm": masked_result.initial_norm,
        "final_norm": masked_result.norm,
        "strictly_monotone": mask_monotone,
        "radial_density_norm_error": mask_density_norm_error,
        "radial_density_moment_error": mask_density_moment_error,
        "radial_density_archive": str(masked_density_path),
    },
    "cap_sia": {
        "initial_norm": cap_result.initial_norm,
        "final_norm": cap_result.norm,
        "strictly_monotone": cap_monotone,
        "radial_density_norm_error": cap_density_norm_error,
        "radial_density_moment_error": cap_density_moment_error,
        "physical_energy_error": cap_energy_error,
        "radial_density_archive": str(cap_density_path),
    },
    "restart_state_difference": restart_state_difference,
    "checkpoint": str(checkpoint_path),
}
result_path.write_text(
    json.dumps(summary, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(f"Phase 9f gate:                   {status}")
print(f"Masked final norm:              {masked_result.norm:.12f}")
print(f"CAP/SIA final norm:             {cap_result.norm:.12f}")
print(f"Mask density norm error:        {mask_density_norm_error:.3e}")
print(f"CAP density norm error:         {cap_density_norm_error:.3e}")
print(f"CAP physical-energy error:      {cap_energy_error:.3e}")
print(f"Absorbed restart difference:    {restart_state_difference:.3e}")
print(f"Result JSON:                    {result_path}")
if status != "PASS":
    raise RuntimeError("Phase 9f absorbed-dynamics validation failed")
