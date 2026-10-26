"""Benchmark preconditioned and unpreconditioned CN steps for non-BO H2+.

The benchmark loads the archived H2+ ground state used by the high-level
examples.  For each requested gauge, it applies one Crank--Nicolson step to
the same state with and without the production one-body preconditioner.

The default step is centered near the largest absolute value of the selected
one-cycle sine-square pulse profile.  Length and velocity gauge generally
reach their largest temporal amplitudes at different times, which are
reported explicitly.  Use ``--step-start-time`` to force a common step.

Run from the repository root, for example::

    python benchmarks/non_bo_diatomic_cn_preconditioner.py

Write a machine-readable result without overwriting an existing file::

    python benchmarks/non_bo_diatomic_cn_preconditioner.py \
        --json-output cn_reference.json

The timed CN calls include all Hamiltonian and preconditioner applications.
Hamiltonian construction, preconditioner construction, and one warm-up solve
per configuration are excluded.  Preconditioner construction is timed and
reported separately.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from electromagnetic_fields import SineSquareLaser
from integrator_lib import CrankNicolson, LinearSolveError
from matter_field_params import wavelength_nm_to_omega_au
from numba import __version__ as numba_version
from numba import get_num_threads
import numpy as np
from scipy import __version__ as scipy_version

from grid_tdse.spherical.two_radial import (
    NonBornOppenheimerDiatomicIonTDSE,
    TwoRadialRealTimePreconditioner,
    load_two_radial_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "ground_states"
    / "H2p__sp_3e11-4e21_16-8_2__prod_1e-08.npz"
)
PACKAGE_DISTRIBUTIONS = (
    "HyQD-grid-tdse",
    "HyQD-grid-lib",
    "HyQD-integrator-lib",
    "HyQD-electromagnetic-fields",
    "HyQD-matter-field-params",
)
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass
class SolveAttempt:
    elapsed_seconds: float
    converged: bool
    iterations: int
    hamiltonian_calls: int
    residual_norm: float
    state: np.ndarray | None
    error: str | None

    def record(self) -> dict:
        """Return the JSON-compatible diagnostics without the state array."""

        return {
            "elapsed_seconds": self.elapsed_seconds,
            "converged": self.converged,
            "iterations": self.iterations,
            "hamiltonian_calls": self.hamiltonian_calls,
            "residual_norm": self.residual_norm,
            "error": self.error,
        }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare one-body-preconditioned and unpreconditioned CN steps "
            "for an archived non-BO H2+ ground state."
        )
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="grid-tdse H2+ state archive",
    )
    parser.add_argument(
        "--gauges",
        nargs="+",
        choices=("length", "velocity"),
        default=("length", "velocity"),
        help="field gauges to benchmark",
    )
    parser.add_argument(
        "--field-strength",
        type=float,
        default=0.001,
        help="peak electric-field parameter E0 in atomic units",
    )
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=400.0,
        help="laser wavelength in nm",
    )
    parser.add_argument(
        "--ncycles",
        type=float,
        default=1.0,
        help="number of optical cycles in the sine-square pulse",
    )
    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="carrier phase in radians",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.5,
        help="Crank--Nicolson step size in atomic units",
    )
    parser.add_argument(
        "--step-start-time",
        type=float,
        default=None,
        help=(
            "optional common step start; by default each gauge is sampled "
            "near the maximum absolute value of its pulse profile"
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("bicgstab", "gmres"),
        default="bicgstab",
        help="Krylov solver",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="relative Krylov tolerance",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-10,
        help="absolute Krylov tolerance",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="maximum Krylov iteration count",
    )
    parser.add_argument(
        "--gmres-restart",
        type=int,
        default=None,
        help="optional GMRES restart length; ignored by BiCGSTAB",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="untimed warm-up solves per gauge and preconditioner choice",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed solves per gauge and preconditioner choice",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="radial tile size used by the Poisson Coulomb action",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="optional exact path for machine-readable benchmark results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing JSON output file",
    )
    return parser.parse_args()


def validate_arguments(args):
    if not args.state_path.is_file():
        raise FileNotFoundError(f"state archive does not exist: {args.state_path}")
    for name in ("field_strength", "phase"):
        if not np.isfinite(getattr(args, name)):
            raise ValueError(f"{name.replace('_', '-')} must be finite")
    for name in ("wavelength_nm", "ncycles", "time_step"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive and finite")
    if args.step_start_time is not None and not np.isfinite(
        args.step_start_time
    ):
        raise ValueError("step-start-time must be finite")
    for name in ("rtol", "atol"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if args.rtol == args.atol == 0.0:
        raise ValueError("at least one linear tolerance must be positive")
    for name in ("max_iterations", "warmups", "repeats", "tile_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.gmres_restart is not None and args.gmres_restart < 1:
        raise ValueError("gmres-restart must be positive")
    if args.json_output is not None:
        if args.json_output.exists() and not args.overwrite:
            raise FileExistsError(
                f"JSON output already exists: {args.json_output}"
            )
        if not args.json_output.parent.is_dir():
            raise FileNotFoundError(
                f"JSON output directory does not exist: "
                f"{args.json_output.parent}"
            )


def pulse_and_step(gauge, args, codata_set):
    omega = wavelength_nm_to_omega_au(
        args.wavelength_nm,
        codata_set=codata_set,
    )
    pulse = SineSquareLaser(
        E0=args.field_strength,
        omega=omega,
        ncycles=args.ncycles,
        gauge=gauge,
        phase=args.phase,
    )
    if args.step_start_time is None:
        support_start, support_end = pulse.time_support
        if args.time_step > support_end - support_start:
            raise ValueError(
                "time-step must not exceed the pulse duration when the "
                "benchmark selects the step automatically"
            )
        sample_times = np.linspace(support_start, support_end, 20_001)
        peak_index = int(np.argmax(np.abs(pulse(sample_times))))
        midpoint = float(sample_times[peak_index])
        step_start = midpoint - 0.5 * args.time_step
        step_start = max(support_start, step_start)
        step_start = min(support_end - args.time_step, step_start)
    else:
        step_start = args.step_start_time
    midpoint = step_start + 0.5 * args.time_step
    return pulse, float(step_start), float(midpoint), float(pulse(midpoint))


def construct_calculation(archive, pulse, gauge, step_start, args):
    physical_state = archive.physical_state
    if physical_state.get("system_family") != "non_bo_diatomic_ion":
        raise ValueError("state archive is not a non-BO diatomic-ion state")
    if physical_state.get("system") != "H2+":
        raise ValueError("this preliminary benchmark requires an H2+ state")
    if archive.M_tot != 0:
        raise ValueError("this preliminary benchmark requires M_tot=0")
    if archive.basis1.l_max != archive.basis2.l_max:
        raise ValueError("the two archived angular bases must use the same l_max")
    if archive.basis1.m_max != archive.basis2.m_max:
        raise ValueError("the two archived angular bases must use the same m_max")

    symmetry = physical_state["symmetry"]
    codata_set = physical_state["parameters"]["codata_set"]
    return NonBornOppenheimerDiatomicIonTDSE(
        "H2+",
        electronic_grid=archive.grid1_specification,
        internuclear_grid=archive.grid2_specification,
        initial_state=archive,
        pair_exchange_parity=symmetry["pair_exchange_parity"],
        fields=((pulse, (0.0, 0.0, 1.0)),),
        field_gauge=gauge,
        l_max=archive.basis1.l_max,
        m_max=archive.basis1.m_max,
        lambda_max=archive.multipoles.lambda_max,
        mu_max=archive.multipoles.mu_max,
        M_tot=archive.M_tot,
        codata_set=codata_set,
        radial_coulomb_method="poisson",
        integrator="crank-nicolson",
        integrator_options={"preconditioner": False},
        initial_time=step_start,
        final_time=step_start + args.time_step,
        time_step=args.time_step,
        mask_r=None,
        mask_R=None,
        cap_r=None,
        cap_R=None,
        observables=(),
        samplers=None,
        checkpoint_path=None,
        checkpoint_interval=None,
        progress_interval=None,
        tile_size=args.tile_size,
    )


def solve_once(integrator, state, time, time_step, hamiltonian):
    hamiltonian_calls = 0

    def linear_rhs(values, evaluation_time):
        nonlocal hamiltonian_calls
        hamiltonian_calls += 1
        return -1j * hamiltonian(values, evaluation_time)

    start = perf_counter()
    try:
        result = integrator(state, time, time_step, linear_rhs)
    except LinearSolveError as error:
        elapsed = perf_counter() - start
        return SolveAttempt(
            elapsed_seconds=elapsed,
            converged=False,
            iterations=int(integrator.last_iterations),
            hamiltonian_calls=hamiltonian_calls,
            residual_norm=float(integrator.last_residual_norm),
            state=None,
            error=str(error),
        )
    elapsed = perf_counter() - start
    return SolveAttempt(
        elapsed_seconds=elapsed,
        converged=True,
        iterations=int(integrator.last_iterations),
        hamiltonian_calls=hamiltonian_calls,
        residual_norm=float(integrator.last_residual_norm),
        state=result,
        error=None,
    )


def benchmark_solver(
    state,
    hamiltonian,
    preconditioner,
    step_start,
    args,
):
    integrator = CrankNicolson(
        solver=args.solver,
        rtol=args.rtol,
        atol=args.atol,
        maxiter=args.max_iterations,
        restart=args.gmres_restart,
        preconditioner=preconditioner,
    )
    for _ in range(args.warmups):
        warmup = solve_once(
            integrator,
            state,
            step_start,
            args.time_step,
            hamiltonian,
        )
        if not warmup.converged:
            record = summarize_attempts((warmup,))
            record["status"] = "failed during warm-up"
            return record, None

    attempts = []
    final_state = None
    for _ in range(args.repeats):
        attempt = solve_once(
            integrator,
            state,
            step_start,
            args.time_step,
            hamiltonian,
        )
        attempts.append(attempt)
        if not attempt.converged:
            final_state = None
            break
        final_state = attempt.state
    record = summarize_attempts(tuple(attempts))
    record["status"] = "converged" if all(
        attempt.converged for attempt in attempts
    ) else "failed during timed solve"
    return record, final_state


def summarize_attempts(attempts):
    elapsed = np.asarray([attempt.elapsed_seconds for attempt in attempts])
    iterations = np.asarray([attempt.iterations for attempt in attempts])
    calls = np.asarray([attempt.hamiltonian_calls for attempt in attempts])
    residuals = np.asarray([attempt.residual_norm for attempt in attempts])
    return {
        "attempts": [attempt.record() for attempt in attempts],
        "minimum_seconds": float(elapsed.min()),
        "median_seconds": float(np.median(elapsed)),
        "minimum_iterations": int(iterations.min()),
        "maximum_iterations": int(iterations.max()),
        "median_iterations": float(np.median(iterations)),
        "minimum_hamiltonian_calls": int(calls.min()),
        "maximum_hamiltonian_calls": int(calls.max()),
        "median_hamiltonian_calls": float(np.median(calls)),
        "maximum_residual_norm": float(residuals.max()),
    }


def state_diagnostics(state, initial_norm, representation):
    if state is None:
        return None
    norm = float(representation.inner_product(state, state).real)
    return {
        "norm": norm,
        "absolute_norm_change": abs(norm - initial_norm),
    }


def state_comparison(left, right, representation):
    if left is None or right is None:
        return None
    difference = left - right
    difference_norm = float(
        np.sqrt(representation.inner_product(difference, difference).real)
    )
    reference_norm = float(
        np.sqrt(representation.inner_product(left, left).real)
    )
    return {
        "physical_difference_norm": difference_norm,
        "relative_physical_difference": difference_norm / reference_norm,
        "maximum_absolute_coefficient_difference": float(
            np.max(np.abs(difference))
        ),
    }


def package_versions():
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "numba": numba_version,
    }
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = None
    return result


def git_metadata():
    def run(*arguments):
        completed = subprocess.run(
            ("git", *arguments),
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def environment_metadata():
    try:
        affinity_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_cpus = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity_cpus,
        "numba_threads": get_num_threads(),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
        "packages": package_versions(),
        "git": git_metadata(),
        "command": [sys.executable, *sys.argv],
    }


def range_text(result, prefix):
    minimum = result[f"minimum_{prefix}"]
    maximum = result[f"maximum_{prefix}"]
    if minimum == maximum:
        return str(minimum)
    median = result[f"median_{prefix}"]
    return f"{minimum}-{maximum} ({median:g})"


def print_report(report):
    configuration = report["configuration"]
    environment = report["environment"]
    print(f"State archive:                  {configuration['state_path']}")
    print(f"Representation ID:              {configuration['representation_id']}")
    print(f"State data ID:                  {configuration['state_data_id']}")
    print(f"System:                         {configuration['system']}")
    print(f"CODATA set:                     {configuration['codata_set']}")
    print(f"State shape after embedding:    {configuration['state_shape']}")
    print(f"State size:                     {configuration['state_mebibytes']:.3f} MiB")
    print(f"Solver:                         {configuration['solver']}")
    print(f"Linear rtol:                    {configuration['rtol']:.3e}")
    print(f"Linear atol:                    {configuration['atol']:.3e}")
    print(f"Maximum iterations:             {configuration['max_iterations']}")
    print(f"Time step:                      {configuration['time_step']:.6f} a.u.")
    print(f"Warm-up solves:                 {configuration['warmups']}")
    print(f"Timed solves:                   {configuration['repeats']}")
    print(f"Host:                           {environment['hostname']}")
    print(f"CPU:                            {environment['cpu_model']}")
    print(f"CPU affinity:                   {environment['affinity_cpus']}")
    print(f"Numba threads:                  {environment['numba_threads']}")

    for gauge, gauge_result in report["gauges"].items():
        print()
        print(f"Gauge: {gauge}")
        print(f"Step start:                     {gauge_result['step_start']:.12f}")
        print(f"Step midpoint:                  {gauge_result['midpoint']:.12f}")
        print(
            f"Temporal field value:           "
            f"{gauge_result['temporal_field_value']:.12e}"
        )
        print(
            f"Preconditioner construction:    "
            f"{gauge_result['preconditioner_construction_seconds']:.2g} s"
        )
        print(
            f"{'Configuration':18s} "
            f"{'status':>24s} "
            f"{'minimum / s':>14s} "
            f"{'median / s':>14s} "
            f"{'iterations':>18s} "
            f"{'H calls':>18s} "
            f"{'max residual':>14s} "
            f"{'norm change':>14s}"
        )
        print("-" * 146)
        for name in ("one_body", "none"):
            result = gauge_result["configurations"][name]
            diagnostics = result["state_diagnostics"]
            norm_change = (
                "unavailable"
                if diagnostics is None
                else f"{diagnostics['absolute_norm_change']:.2e}"
            )
            print(
                f"{name:18s} "
                f"{result['status']:>24s} "
                f"{result['minimum_seconds']:14.2g} "
                f"{result['median_seconds']:14.2g} "
                f"{range_text(result, 'iterations'):>18s} "
                f"{range_text(result, 'hamiltonian_calls'):>18s} "
                f"{result['maximum_residual_norm']:14.2e} "
                f"{norm_change:>14s}"
            )
        comparison = gauge_result["state_comparison"]
        if comparison is None:
            print("State comparison:               unavailable")
        else:
            print(
                "Relative state difference:       "
                f"{comparison['relative_physical_difference']:.3e}"
            )


def main():
    args = parse_arguments()
    validate_arguments(args)
    archive = load_two_radial_state(args.state_path)
    physical_state = archive.physical_state
    codata_set = physical_state["parameters"]["codata_set"]

    report = {
        "format": "grid-tdse-non-bo-cn-preconditioner-benchmark",
        "format_version": 1,
        "environment": environment_metadata(),
        "configuration": {
            "state_path": str(args.state_path.resolve()),
            "representation_id": archive.representation_id,
            "state_data_id": archive.state_data_id,
            "system": physical_state["system"],
            "codata_set": codata_set,
            "stored_state_shape": list(archive.coefficients.shape),
            "l_max": archive.basis1.l_max,
            "m_max": archive.basis1.m_max,
            "lambda_max": archive.multipoles.lambda_max,
            "mu_max": archive.multipoles.mu_max,
            "M_tot": archive.M_tot,
            "solver": args.solver,
            "rtol": args.rtol,
            "atol": args.atol,
            "max_iterations": args.max_iterations,
            "gmres_restart": args.gmres_restart,
            "time_step": args.time_step,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "tile_size": args.tile_size,
            "field_strength": args.field_strength,
            "wavelength_nm": args.wavelength_nm,
            "ncycles": args.ncycles,
            "phase": args.phase,
            "requested_step_start_time": args.step_start_time,
        },
        "gauges": {},
    }

    for gauge in dict.fromkeys(args.gauges):
        pulse, step_start, midpoint, field_value = pulse_and_step(
            gauge,
            args,
            codata_set,
        )
        calculation = construct_calculation(
            archive,
            pulse,
            gauge,
            step_start,
            args,
        )
        state = calculation.initial_state.copy()
        representation = calculation.representation
        initial_norm = float(representation.inner_product(state, state).real)
        report["configuration"]["state_shape"] = list(state.shape)
        report["configuration"]["state_mebibytes"] = state.nbytes / 2**20
        report["configuration"]["initial_norm"] = initial_norm

        setup_start = perf_counter()
        preconditioner = TwoRadialRealTimePreconditioner(
            representation,
            args.time_step,
        )
        preconditioner_setup = perf_counter() - setup_start

        configurations = {}
        states = {}
        for name, selected_preconditioner in (
            ("one_body", preconditioner),
            ("none", None),
        ):
            result, final_state = benchmark_solver(
                state,
                calculation.hamiltonian,
                selected_preconditioner,
                step_start,
                args,
            )
            result["state_diagnostics"] = state_diagnostics(
                final_state,
                initial_norm,
                representation,
            )
            configurations[name] = result
            states[name] = final_state

        report["gauges"][gauge] = {
            "step_start": step_start,
            "midpoint": midpoint,
            "temporal_field_value": field_value,
            "preconditioner_construction_seconds": preconditioner_setup,
            "configurations": configurations,
            "state_comparison": state_comparison(
                states["one_body"],
                states["none"],
                representation,
            ),
        }

    print_report(report)
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print()
        print(f"JSON output:                    {args.json_output.resolve()}")


if __name__ == "__main__":
    main()
