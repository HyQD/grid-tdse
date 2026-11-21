"""Benchmark serial full-angular one-radial H2+ calculations.

The benchmark exercises a tilted clamped-nuclei H2+ geometry and measures
Hamiltonian construction and action, a matrix-free Davidson ground-state
solve, every default observable, and short SIA and Crank--Nicolson
propagations.  It writes one self-describing JSON record for a Cartesian
product of radial extents and angular cutoffs.

Run thread counts in separate processes.  For example::

    python benchmarks/spherical_one_radial_off_axis_performance.py \
        --l-max 2 4 6 8 --threads 1 --result-json serial-t1.json

The defaults are performance-development inputs, not scientific convergence
parameters.  The program is serial even when numerical libraries use more
than one thread.
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
from pathlib import Path
from time import perf_counter

from electromagnetic_fields import SineSquareLaser
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from grid_tdse.benchmarking import runtime_environment, write_json_exclusive
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    OneElectronDiatomicTDSE,
    diagonalize_full_angular,
)


SCHEMA = "grid-tdse-spherical-one-radial-off-axis-performance-v1"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l-max", type=int, nargs="+", default=(2, 4, 6, 8))
    parser.add_argument("--r-max", type=float, nargs="+", default=(8.0,))
    parser.add_argument("--element-width", type=float, default=2.0)
    parser.add_argument("--points-per-element", type=int, default=7)
    parser.add_argument("--bond-length", type=float, default=2.0)
    parser.add_argument("--theta-deg", type=float, default=45.0)
    parser.add_argument("--phi-deg", type=float, default=0.0)
    parser.add_argument("--particle-mass", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--action-warmups", type=int, default=2)
    parser.add_argument("--action-repeats", type=int, default=7)
    parser.add_argument("--observable-warmups", type=int, default=1)
    parser.add_argument("--observable-repeats", type=int, default=5)
    parser.add_argument("--propagation-repeats", type=int, default=1)
    parser.add_argument("--propagation-steps", type=int, default=2)
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument(
        "--integrators",
        nargs="+",
        choices=("sia", "sil", "crank-nicolson"),
        default=("sia", "crank-nicolson"),
    )
    parser.add_argument("--field-strength", type=float, default=0.01)
    parser.add_argument("--frequency", type=float, default=4.0)
    parser.add_argument("--ground-state-tolerance", type=float, default=1e-10)
    parser.add_argument("--ground-state-max-iterations", type=int, default=300)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)

    integer_minima = {
        "l_max": 0,
        "points_per_element": 2,
        "threads": 1,
        "action_warmups": 0,
        "action_repeats": 1,
        "observable_warmups": 0,
        "observable_repeats": 1,
        "propagation_repeats": 1,
        "propagation_steps": 1,
        "ground_state_max_iterations": 1,
    }
    for name, minimum in integer_minima.items():
        values = getattr(arguments, name)
        values = values if isinstance(values, (tuple, list)) else (values,)
        if any(value < minimum for value in values):
            parser.error(
                f"--{name.replace('_', '-')} values must be at least {minimum}"
            )
    for name in (
        "element_width",
        "bond_length",
        "particle_mass",
        "time_step",
        "field_strength",
        "frequency",
        "ground_state_tolerance",
    ):
        value = getattr(arguments, name)
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(not np.isfinite(value) or value <= 0.0 for value in arguments.r_max):
        parser.error("--r-max values must be positive")
    if not np.isfinite(arguments.theta_deg) or not np.isfinite(
        arguments.phi_deg
    ):
        parser.error("orientation angles must be finite")
    arguments.l_max = tuple(dict.fromkeys(arguments.l_max))
    arguments.r_max = tuple(dict.fromkeys(arguments.r_max))
    arguments.integrators = tuple(dict.fromkeys(arguments.integrators))
    return arguments


def _system(arguments):
    theta = np.deg2rad(arguments.theta_deg)
    phi = np.deg2rad(arguments.phi_deg)
    axis = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    displacement = 0.5 * arguments.bond_length * axis
    return OneElectronDiatomicSystem.from_nuclear_positions(
        1.0,
        1.0,
        -displacement,
        displacement,
        particle_mass=arguments.particle_mass,
        name="H2+",
    )


def _radial_grid(r_max, element_width, points_per_element, system):
    radii = np.linalg.norm(system.nuclear_positions, axis=1)
    if np.any(radii >= r_max):
        raise ValueError("r_max must exceed every nuclear radius")
    boundaries = np.unique(np.concatenate(([0.0], radii, [r_max])))
    pieces = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        count = max(1, int(np.ceil((right - left) / element_width)))
        segment = np.linspace(left, right, count + 1)
        pieces.append(segment if not pieces else segment[1:])
    nodes = np.concatenate(pieces)
    return nodes, [points_per_element] * (nodes.size - 1)


def _memory_snapshot():
    values = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            name, separator, remainder = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                values[name] = float(remainder.split()[0]) / 1024.0
    except OSError:
        pass
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resource_peak = maximum / (1024.0 if sys.platform.startswith("linux") else 2**20)
    return {
        "current_rss_mib": values.get("VmRSS"),
        "peak_rss_mib": values.get("VmHWM", resource_peak),
        "resource_peak_rss_mib": resource_peak,
    }


def _array_inventory(hamiltonian):
    arrays = {
        "radial_points": hamiltonian.r,
        "radial_weights": hamiltonian.weights,
        "first_derivative": hamiltonian.first_derivative,
        "second_derivative": hamiltonian.second_derivative,
        "centrifugal": hamiltonian._centrifugal,
        "potential_coefficients": hamiltonian.potential.coefficients,
    }
    coordinate = hamiltonian._coordinate_operators
    sparse = {
        "coordinate_cos_theta": coordinate._cos_theta,
        "coordinate_sin_theta_dtheta": coordinate._sin_theta_dtheta,
    }
    for index, (_, matrix) in enumerate(hamiltonian.potential._couplings):
        sparse[f"potential_coupling_{index}"] = matrix

    entries = {}
    roots = {}
    for name, array in arrays.items():
        value = np.asarray(array)
        root = value
        while isinstance(root.base, np.ndarray):
            root = root.base
        roots.setdefault(id(root), root.nbytes)
        entries[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": int(value.nbytes),
        }
    sparse_bytes = 0
    sparse_nnz = 0
    for matrix in sparse.values():
        sparse_nnz += int(matrix.nnz)
        for array in (matrix.data, matrix.indices, matrix.indptr):
            root = array
            while isinstance(root.base, np.ndarray):
                root = root.base
            roots.setdefault(id(root), root.nbytes)
            sparse_bytes += int(array.nbytes)
    return {
        "named_arrays": entries,
        "unique_dense_and_sparse_bytes": int(sum(roots.values())),
        "sparse_storage_bytes": sparse_bytes,
        "sparse_nonzeros": sparse_nnz,
    }


def _timed(function, *, warmups, repeats):
    for _ in range(warmups):
        function()
    samples = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = function()
        samples.append(perf_counter() - started)
    values = np.asarray(samples)
    checksum = float(np.sum(np.abs(np.asarray(result))))
    return {
        "samples_seconds": values.tolist(),
        "minimum_seconds": float(np.min(values)),
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(np.mean(values)),
        "maximum_seconds": float(np.max(values)),
        "checksum": checksum,
    }


def _integrator_options(name):
    if name == "crank-nicolson":
        return {
            "solver": "gmres",
            "rtol": 1e-10,
            "atol": 1e-13,
            "max_iterations": 500,
            "preconditioner": True,
        }
    return {
        "max_krylov_dimension": 30,
        "rtol": 1e-10,
        "atol": 1e-13,
        "max_refinements": 8,
    }


def _calculation(arguments, system, nodes, n_points, l_max, state, integrator):
    pulse = SineSquareLaser(
        E0=arguments.field_strength,
        omega=arguments.frequency,
        ncycles=1.0,
        gauge="length",
        phase=0.0,
        t0=0.0,
    )
    return OneElectronDiatomicTDSE(
        system,
        nodes=nodes,
        n_points=n_points,
        l_max=l_max,
        angular_representation="full",
        m_max=l_max,
        nuclear_L_max=2 * l_max,
        nuclear_M_max=2 * l_max,
        initial_state=state,
        fields=pulse,
        field_gauge="length",
        integrator=integrator,
        integrator_options=_integrator_options(integrator),
        final_time=arguments.propagation_steps * arguments.time_step,
        time_step=arguments.time_step,
        reference_states={"ground": state},
        radial_probability_interval=None,
        progress_interval=None,
        progress_bar=False,
    )


def _propagation_record(result, wall_time):
    return {
        "wall_time_seconds": wall_time,
        "accepted_steps": int(result.n_steps),
        "step_wall_seconds": np.asarray(result.step_wall_times).tolist(),
        "step_wall_total_seconds": float(np.sum(result.step_wall_times)),
        "krylov_dimensions": np.asarray(result.krylov_dimensions).tolist(),
        "krylov_operator_calls": np.asarray(
            result.krylov_operator_calls
        ).tolist(),
        "linear_iterations": np.asarray(result.linear_iterations).tolist(),
        "linear_residual_norms": np.asarray(
            result.linear_residual_norms
        ).tolist(),
        "initial_norm": float(result.initial_norm),
        "final_norm": float(result.norm),
    }


def benchmark_case(arguments, *, system, r_max, l_max):
    gc.collect()
    memory = {"baseline": _memory_snapshot()}
    nodes, n_points = _radial_grid(
        r_max,
        arguments.element_width,
        arguments.points_per_element,
        system,
    )
    started = perf_counter()
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=nodes,
        n_points=n_points,
        l_max=l_max,
        m_max=l_max,
        nuclear_L_max=2 * l_max,
        nuclear_M_max=2 * l_max,
    )
    construction_wall = perf_counter() - started
    memory["hamiltonian"] = _memory_snapshot()

    generator = np.random.default_rng(1729 + 101 * l_max + int(r_max))
    state = generator.normal(size=hamiltonian.orbital_shape)
    state = state + 1j * generator.normal(size=hamiltonian.orbital_shape)
    state /= np.sqrt(hamiltonian.inner_product(state, state).real)
    action = _timed(
        lambda: hamiltonian.apply(state),
        warmups=arguments.action_warmups,
        repeats=arguments.action_repeats,
    )

    started = perf_counter()
    eigenstates = diagonalize_full_angular(
        hamiltonian,
        residual_tolerance=arguments.ground_state_tolerance,
        max_iterations=arguments.ground_state_max_iterations,
        print_progress=False,
    )
    davidson_wall = perf_counter() - started
    memory["davidson"] = _memory_snapshot()
    ground_state = np.array(eigenstates.orbital, copy=True)
    davidson = {
        "wall_time_seconds": davidson_wall,
        "converged": eigenstates.converged,
        "termination_reason": eigenstates.termination_reason,
        "electronic_energy": eigenstates.electronic_energy,
        "residual_norm": eigenstates.residual_norm,
        "iterations": eigenstates.iterations,
        "restarts": eigenstates.restarts,
        "hamiltonian_actions": eigenstates.hamiltonian_actions,
        "preconditioner_applications": (
            eigenstates.preconditioner_applications
        ),
    }

    coefficients = np.asarray(hamiltonian.potential.coefficients)
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    active_components = int(
        np.count_nonzero(np.max(np.abs(coefficients), axis=1) > 1e-13 * scale)
    )
    case = {
        "problem": {
            "r_max": float(r_max),
            "nodes": nodes.tolist(),
            "points_per_element": arguments.points_per_element,
            "active_radial_points": int(hamiltonian.r.size),
            "l_max": int(l_max),
            "m_max": int(l_max),
            "angular_channels": len(hamiltonian.orbital_lm),
            "state_coefficients": int(np.prod(hamiltonian.orbital_shape)),
            "complex128_state_bytes": int(
                np.prod(hamiltonian.orbital_shape) * np.dtype(complex).itemsize
            ),
            "nominal_potential_components": len(
                hamiltonian.potential.multipole_lm
            ),
            "active_potential_components": active_components,
            "potential_coupling_nonzeros": int(
                sum(matrix.nnz for _, matrix in hamiltonian.potential._couplings)
            ),
        },
        "construction": {"wall_time_seconds": construction_wall},
        "array_inventory": _array_inventory(hamiltonian),
        "hamiltonian_action": action,
        "davidson": davidson,
        "memory": memory,
        "observables": None,
        "propagation": {},
    }
    if not eigenstates.converged:
        return case

    observables_record = None
    for integrator in arguments.integrators:
        attempts = []
        for repetition in range(arguments.propagation_repeats):
            calculation = _calculation(
                arguments,
                system,
                nodes,
                n_points,
                l_max,
                ground_state,
                integrator,
            )
            if observables_record is None:
                observables_record = {}
                for name, sampler in calculation.observables.samplers.items():
                    observables_record[name] = _timed(
                        lambda sampler=sampler: sampler(ground_state, 0.0),
                        warmups=arguments.observable_warmups,
                        repeats=arguments.observable_repeats,
                    )
                observables_record["radial_probability"] = _timed(
                    lambda: calculation.observables.radial_probability(
                        ground_state
                    ),
                    warmups=arguments.observable_warmups,
                    repeats=arguments.observable_repeats,
                )
                memory["calculation"] = _memory_snapshot()
            started = perf_counter()
            result = calculation.run(progress=False)
            wall_time = perf_counter() - started
            attempts.append(_propagation_record(result, wall_time))
            del result, calculation
            gc.collect()
        case["propagation"][integrator] = attempts
    case["observables"] = observables_record
    memory["complete"] = _memory_snapshot()
    return case


def run(arguments):
    system = _system(arguments)
    with threadpool_limits(limits=arguments.threads):
        environment = runtime_environment(include_build_configuration=False)
        effective_pools = threadpool_info()
        cases = [
            benchmark_case(
                arguments,
                system=system,
                r_max=r_max,
                l_max=l_max,
            )
            for r_max in arguments.r_max
            for l_max in arguments.l_max
        ]
    return {
        "schema": SCHEMA,
        "environment": environment,
        "thread_control": {
            "requested_threads": arguments.threads,
            "effective_thread_pools": effective_pools,
            "execution_model": "one_serial_python_process",
        },
        "geometry": {
            "system": "H2+",
            "nuclear_positions": system.nuclear_positions.tolist(),
            "bond_axis": system.bond_axis.tolist(),
            "internuclear_distance": system.internuclear_distance,
            "particle_mass": system.particle_mass,
            "theta_deg": arguments.theta_deg,
            "phi_deg": arguments.phi_deg,
        },
        "configuration": {
            "l_max": list(arguments.l_max),
            "r_max": list(arguments.r_max),
            "element_width": arguments.element_width,
            "points_per_element": arguments.points_per_element,
            "action_warmups": arguments.action_warmups,
            "action_repeats": arguments.action_repeats,
            "observable_warmups": arguments.observable_warmups,
            "observable_repeats": arguments.observable_repeats,
            "propagation_repeats": arguments.propagation_repeats,
            "propagation_steps": arguments.propagation_steps,
            "time_step": arguments.time_step,
            "integrators": list(arguments.integrators),
            "field_strength": arguments.field_strength,
            "frequency": arguments.frequency,
            "ground_state_tolerance": arguments.ground_state_tolerance,
            "ground_state_max_iterations": (
                arguments.ground_state_max_iterations
            ),
        },
        "cases": cases,
    }


def _print_summary(payload, destination):
    print(f"Result: {destination}")
    print(f"Threads: {payload['thread_control']['requested_threads']}")
    print(
        "lmax  nr  nlm  components  action(s)  Davidson(s)  "
        "SIA(s)  CN(s)"
    )
    for case in payload["cases"]:
        problem = case["problem"]
        propagation = case["propagation"]

        def first_wall(name):
            attempts = propagation.get(name, ())
            return float("nan") if not attempts else attempts[0]["wall_time_seconds"]

        print(
            f"{problem['l_max']:4d} {problem['active_radial_points']:3d} "
            f"{problem['angular_channels']:4d} "
            f"{problem['active_potential_components']:10d} "
            f"{case['hamiltonian_action']['median_seconds']:10.4e} "
            f"{case['davidson']['wall_time_seconds']:12.4e} "
            f"{first_wall('sia'):7.3f} "
            f"{first_wall('crank-nicolson'):7.3f}"
        )


def main(argv=None):
    arguments = parse_arguments(argv)
    payload = run(arguments)
    destination = write_json_exclusive(
        arguments.result_json,
        payload,
        overwrite=arguments.overwrite,
    )
    _print_summary(payload, destination)
    return payload


if __name__ == "__main__":
    main()
