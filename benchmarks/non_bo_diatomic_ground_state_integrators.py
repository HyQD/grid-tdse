"""Compare CN, SIL, and SIA non-BO diatomic ground-state iteration.

Run the lightweight comparison from the repository root with::

    python benchmarks/non_bo_diatomic_ground_state_integrators.py --grid small

Run one moderate H2+ comparison with::

    python benchmarks/non_bo_diatomic_ground_state_integrators.py \
        --systems H2+ --grid moderate

Each calculation starts from the same deterministic product state. CN uses
the existing one-body preconditioner; SIL and SIA are unpreconditioned
exponential actions. Run one integrator at a time when independent process
peak-memory measurements are required.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import numpy as np

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
)


INTEGRATORS = ("crank-nicolson", "sil", "sia")
SYSTEMS = ("H2+", "HD+", "D2+")
GRID_PRESETS = {
    "small": {
        "electronic_nodes": [0.0, 2.0, 5.0],
        "electronic_points": [4, 4],
        "internuclear_nodes": [0.0, 1.5, 4.0],
        "internuclear_points": [4, 4],
    },
    "moderate": {
        "electronic_nodes": [0.0, 4.0, 8.0, 16.0],
        "electronic_points": [11, 11, 11],
        "internuclear_nodes": [0.0, 2.0, 4.0, 6.0, 8.0],
        "internuclear_points": [11, 11, 11, 11],
    },
}


@dataclass
class BenchmarkRecord:
    """Serializable result for one system and integrator."""

    system: str
    integrator: str
    converged: bool
    iterations: int
    imaginary_time: float
    energy: float
    energy_change: float
    state_change: float
    residual_norm: float
    hamiltonian_calls: int
    linear_iterations: int
    krylov_operator_calls: int
    elapsed_seconds: float
    state_storage_mib: float
    process_peak_rss_mib: float
    inversion_expectation: float
    pair_exchange_expectation: float | None
    total_l_squared: float
    total_l_z: float
    energy_difference_from_reference: float | None = None
    state_difference_from_reference: float | None = None


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEMS,
        default=SYSTEMS,
        help="diatomic ions to compare",
    )
    parser.add_argument(
        "--integrators",
        nargs="+",
        choices=INTEGRATORS,
        default=INTEGRATORS,
        help="imaginary-time integrators to compare",
    )
    parser.add_argument(
        "--grid",
        choices=tuple(GRID_PRESETS),
        default="small",
        help="FEM-DVR benchmark grid preset",
    )
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--time-step", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--state-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--linear-atol", type=float, default=1.0e-11)
    parser.add_argument("--max-krylov-dimension", type=int, default=20)
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-10)
    parser.add_argument("--krylov-atol", type=float, default=0.0)
    parser.add_argument("--max-krylov-refinements", type=int, default=8)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
        help="ITP reporting interval; zero disables progress output",
    )
    parser.add_argument("--codata-set", default="2022")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="optional exact output path for machine-readable results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing JSON output",
    )
    return parser.parse_args()


def validate_arguments(args):
    if args.l_max < 0:
        raise ValueError("l-max must be non-negative")
    if args.max_iterations <= 0:
        raise ValueError("max-iterations must be positive")
    if args.max_krylov_dimension <= 0:
        raise ValueError("max-krylov-dimension must be positive")
    if args.max_krylov_refinements < 0:
        raise ValueError("max-krylov-refinements must be non-negative")
    if args.progress_interval < 0:
        raise ValueError("progress-interval must be non-negative")
    for name in (
        "time_step",
        "energy_tolerance",
        "state_tolerance",
        "residual_tolerance",
        "linear_atol",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.krylov_rtol < 0.0 or args.krylov_atol < 0.0:
        raise ValueError("Krylov tolerances must be non-negative")
    if args.krylov_rtol == args.krylov_atol == 0.0:
        raise ValueError("at least one Krylov tolerance must be positive")
    if args.json_output is not None and args.json_output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.json_output}")


def grids(name):
    values = GRID_PRESETS[name]
    return (
        FEMDVRGridSpec(
            values["electronic_nodes"], values["electronic_points"]
        ),
        FEMDVRGridSpec(
            values["internuclear_nodes"], values["internuclear_points"]
        ),
    )


def calculation(system, integrator, args):
    electronic_grid, internuclear_grid = grids(args.grid)
    homonuclear = system != "HD+"
    return NonBornOppenheimerDiatomicIonGroundState(
        system,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        inversion_parity=1,
        pair_exchange_parity=1 if homonuclear else None,
        symmetry_adapted_basis=homonuclear,
        l_max=args.l_max,
        m_max=args.l_max,
        lambda_max=2 * args.l_max,
        mu_max=2 * args.l_max,
        M_tot=0,
        radial_coulomb_method="poisson",
        radial_operator_mode="element-local",
        codata_set=args.codata_set,
        initial_state="product",
        time_step=args.time_step,
        max_iterations=args.max_iterations,
        energy_tolerance=args.energy_tolerance,
        state_tolerance=args.state_tolerance,
        residual_tolerance=args.residual_tolerance,
        integrator=integrator,
        linear_solver="bicgstab",
        linear_rtol=0.0,
        linear_atol=args.linear_atol,
        max_krylov_dimension=args.max_krylov_dimension,
        krylov_rtol=args.krylov_rtol,
        krylov_atol=args.krylov_atol,
        max_krylov_refinements=args.max_krylov_refinements,
        krylov_reorthogonalize=True,
        progress_interval=(args.progress_interval or None),
        use_preconditioner=integrator == "crank-nicolson",
    )


def max_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return float(value / divisor)


def run_case(system, integrator, args):
    configured = calculation(system, integrator, args)
    hamiltonian = configured.hamiltonian
    calls = 0

    def counted_hamiltonian(state):
        nonlocal calls
        calls += 1
        return hamiltonian(state)

    configured.hamiltonian = counted_hamiltonian
    start = perf_counter()
    result = configured.solve()
    elapsed = perf_counter() - start
    last = result.history[-1]
    record = BenchmarkRecord(
        system=system,
        integrator=integrator,
        converged=result.converged,
        iterations=result.iterations,
        imaginary_time=result.ground_state.imaginary_time,
        energy=result.energy,
        energy_change=last.energy_change,
        state_change=last.state_change,
        residual_norm=result.residual_norm,
        hamiltonian_calls=calls,
        linear_iterations=sum(item.linear_iterations for item in result.history),
        krylov_operator_calls=sum(
            item.krylov_operator_calls for item in result.history
        ),
        elapsed_seconds=elapsed,
        state_storage_mib=result.state.nbytes / 1024.0**2,
        process_peak_rss_mib=max_rss_mib(),
        inversion_expectation=result.inversion_expectation,
        pair_exchange_expectation=result.pair_exchange_expectation,
        total_l_squared=result.angular_momentum.total_l_squared,
        total_l_z=result.angular_momentum.total_l_z,
    )
    return record, result.state, configured.representation


def add_reference_differences(cases):
    by_system = {}
    for record, state, representation in cases:
        by_system.setdefault(record.system, []).append(
            (record, state, representation)
        )
    for system_cases in by_system.values():
        reference = next(
            (
                case
                for case in system_cases
                if case[0].integrator == "crank-nicolson"
            ),
            system_cases[0],
        )
        reference_record, reference_state, representation = reference
        for record, state, _ in system_cases:
            overlap = representation.inner_product(reference_state, state)
            aligned = (
                state
                if overlap == 0.0
                else state * overlap.conjugate() / abs(overlap)
            )
            difference = aligned - reference_state
            record.energy_difference_from_reference = abs(
                record.energy - reference_record.energy
            )
            record.state_difference_from_reference = float(
                np.sqrt(representation.inner_product(difference, difference).real)
            )


def print_records(cases):
    header = (
        "system method             conv  iter       energy          residual  "
        "H calls  seconds   dE(ref)   dPsi(ref)"
    )
    print(header)
    print("-" * len(header))
    for record, _, _ in cases:
        print(
            f"{record.system:6s} {record.integrator:18s} "
            f"{str(record.converged):5s} {record.iterations:5d} "
            f"{record.energy:16.10f} {record.residual_norm:10.3e} "
            f"{record.hamiltonian_calls:8d} {record.elapsed_seconds:8.3f} "
            f"{record.energy_difference_from_reference:9.2e} "
            f"{record.state_difference_from_reference:11.2e}"
        )


def write_json(path, args, cases):
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "systems": list(args.systems),
            "integrators": list(args.integrators),
            "grid": args.grid,
            "grid_specification": GRID_PRESETS[args.grid],
            "l_max": args.l_max,
            "time_step": args.time_step,
            "max_iterations": args.max_iterations,
            "energy_tolerance": args.energy_tolerance,
            "state_tolerance": args.state_tolerance,
            "residual_tolerance": args.residual_tolerance,
            "codata_set": args.codata_set,
        },
        "memory_note": (
            "process_peak_rss_mib is cumulative within this process; run one "
            "integrator per invocation for independent peak-memory values"
        ),
        "results": [asdict(record) for record, _, _ in cases],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    args = parse_arguments()
    validate_arguments(args)
    cases = [
        run_case(system, integrator, args)
        for system in args.systems
        for integrator in args.integrators
    ]
    add_reference_differences(cases)
    print_records(cases)
    if args.json_output is not None:
        write_json(args.json_output, args, cases)
        print(f"\nSaved benchmark data: {args.json_output}")


if __name__ == "__main__":
    main()
