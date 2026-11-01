"""Continue and archive a converged Cartesian-MPI non-BO ground-state chain.

This production-oriented example embeds a converged lower-``l_max`` archive
into the next angular basis, solves with distributed Krylov imaginary-time
propagation, and writes a new self-describing state archive.  It is intended
for the dependent Saga chains in ``examples/Saga/ground_state_chains``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributed2DGroundStateSolver,
    TwoRadialDistributed2DLayout,
    load_two_radial_state,
    save_distributed_two_radial_2d_state,
    scatter_two_radial_2d_state,
)


def _process_grid(value: str) -> tuple[int, int]:
    try:
        first, second = value.lower().split("x", maxsplit=1)
        result = int(first), int(second)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "process grid must have the form P_r x P_R, for example 2x4"
        ) from exc
    if result[0] <= 0 or result[1] <= 0:
        raise argparse.ArgumentTypeError("process-grid dimensions must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), required=True)
    parser.add_argument("--electronic-points-per-element", type=int, required=True)
    parser.add_argument("--l-max", type=int, required=True)
    parser.add_argument("--process-grid", type=_process_grid, required=True)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument(
        "--integrator",
        choices=("sil", "sia", "exponential-euler"),
        default="sil",
    )
    parser.add_argument("--time-step", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=15_000)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--state-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-krylov-dimension", type=int, default=20)
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-10)
    parser.add_argument("--max-krylov-refinements", type=int, default=8)
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    mpi_size: int,
) -> None:
    if arguments.electronic_points_per_element < 2:
        parser.error("--electronic-points-per-element must be at least 2")
    if arguments.l_max < 0 or arguments.l_max % 2:
        parser.error("--l-max must be a nonnegative even integer")
    if arguments.l_max == 0 and arguments.initial_state is not None:
        parser.error("the l_max=0 chain head must not specify --initial-state")
    if arguments.l_max > 0 and arguments.initial_state is None:
        parser.error("l_max>0 continuation requires --initial-state")
    if np.prod(arguments.process_grid) != mpi_size:
        parser.error("--process-grid dimensions must multiply to the MPI size")
    for name in (
        "time_step",
        "energy_tolerance",
        "state_tolerance",
        "residual_tolerance",
        "krylov_rtol",
    ):
        if getattr(arguments, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "max_iterations",
        "sample_interval",
        "max_krylov_dimension",
        "max_krylov_refinements",
    ):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def _calculation(arguments: argparse.Namespace):
    identical_pair = arguments.system != "HD+"
    l_max = arguments.l_max
    return NonBornOppenheimerDiatomicIonGroundState(
        arguments.system,
        electronic_grid=FEMDVRGridSpec(
            nodes=[0.0, 4.0, 8.0, 12.0, 16.0],
            n_points=[arguments.electronic_points_per_element] * 4,
        ),
        internuclear_grid=FEMDVRGridSpec(
            nodes=[0.0, 2.0, 4.0, 6.0, 8.0],
            n_points=[21] * 4,
        ),
        inversion_parity=+1,
        pair_exchange_parity=+1 if identical_pair else None,
        symmetry_adapted_basis=identical_pair,
        l_max=l_max,
        m_max=l_max,
        lambda_max=2 * l_max,
        mu_max=2 * l_max,
        M_tot=0,
        radial_coulomb_method="poisson",
        radial_operator_mode="element-local",
        codata_set="2022",
        initial_state="product",
        integrator=arguments.integrator,
        use_preconditioner=False,
        time_step=arguments.time_step,
        max_iterations=arguments.max_iterations,
        energy_tolerance=arguments.energy_tolerance,
        state_tolerance=arguments.state_tolerance,
        residual_tolerance=arguments.residual_tolerance,
        max_krylov_dimension=arguments.max_krylov_dimension,
        krylov_rtol=arguments.krylov_rtol,
        krylov_atol=0.0,
        max_krylov_refinements=arguments.max_krylov_refinements,
        krylov_reorthogonalize=True,
        progress_interval=arguments.sample_interval,
    )


def _validate_parent_metadata(archived, calculation, path: Path) -> None:
    physical = archived.physical_state
    expected_pair = +1 if calculation.parameters.has_identical_pair else None
    expected = {
        "system_family": "non_bo_diatomic_ion",
        "system": calculation.system_name,
    }
    for key, value in expected.items():
        if physical.get(key) != value:
            raise ValueError(
                f"initial archive {path} has physical_state[{key!r}]="
                f"{physical.get(key)!r}; expected {value!r}"
            )
    symmetry = physical.get("symmetry", {})
    expected_symmetry = {
        "inversion_parity": +1,
        "pair_exchange_parity": expected_pair,
        "symmetry_adapted_basis": calculation.parameters.has_identical_pair,
        "M_tot": 0,
    }
    for key, value in expected_symmetry.items():
        if symmetry.get(key) != value:
            raise ValueError(
                f"initial archive {path} has symmetry[{key!r}]="
                f"{symmetry.get(key)!r}; expected {value!r}"
            )


def _root_initial_state(arguments, calculation, communicator):
    state = None
    original_error: Exception | None = None
    error = None
    parent_metadata = None
    if communicator.rank == 0:
        try:
            if arguments.initial_state is None:
                state = calculation.initial_state.copy()
            else:
                archived = load_two_radial_state(arguments.initial_state)
                _validate_parent_metadata(
                    archived,
                    calculation,
                    arguments.initial_state,
                )
                state = archived.embed_into(calculation.representation)
                norm = calculation.representation.inner_product(state, state).real
                state /= np.sqrt(norm)
                parent_metadata = {
                    "path": str(arguments.initial_state),
                    "state_data_id": archived.state_data_id,
                    "representation_id": archived.representation_id,
                }
        except Exception as exc:
            original_error = exc
            error = (type(exc).__name__, str(exc))
    error = communicator.bcast(error, root=0)
    if error is not None:
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"rank 0 could not prepare the initial state: {error[1]}")
    parent_metadata = communicator.bcast(parent_metadata, root=0)
    return state, parent_metadata


def _validate_output_paths(arguments, communicator) -> None:
    error = None
    if communicator.rank == 0:
        for path, label in (
            (arguments.output_state, "output state"),
            (arguments.log_file, "log file"),
        ):
            if path.exists():
                error = f"{label} already exists: {path}"
                break
            if not path.parent.is_dir():
                error = f"{label} directory does not exist: {path.parent}"
                break
    error = communicator.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)


def _profile_maxima(result, communicator) -> dict[str, float]:
    hamiltonian = result.hamiltonian_profile
    reductions = result.reduction_profile
    names = (
        "hamiltonian_calls",
        "coordinate1_seconds",
        "coordinate2_seconds",
        "coulomb_seconds",
        "coordinate1_halo_seconds",
        "coordinate2_halo_seconds",
        "collective_calls",
        "collective_values",
        "collective_seconds",
    )
    values = np.asarray(
        [
            hamiltonian.hamiltonian_calls,
            hamiltonian.coordinate1_seconds,
            hamiltonian.coordinate2_seconds,
            hamiltonian.coulomb_seconds,
            hamiltonian.coordinate1_halo_seconds,
            hamiltonian.coordinate2_halo_seconds,
            reductions.collective_calls,
            reductions.collective_values,
            reductions.collective_seconds,
        ],
        dtype=float,
    )
    maximum = np.empty_like(values)
    communicator.Allreduce(values, maximum, op=MPI.MAX)
    return dict(zip(names, maximum, strict=True))


def _physical_metadata(calculation) -> dict[str, Any]:
    return {
        "system_family": "non_bo_diatomic_ion",
        "system": calculation.system_name,
        "state_index": 0,
        "parameters": asdict(calculation.parameters),
        "symmetry": {
            "inversion_parity": calculation.inversion_parity,
            "pair_exchange_parity": calculation.pair_exchange_parity,
            "symmetry_adapted_basis": calculation.symmetry_adapted_basis,
            "M_tot": calculation.M_tot,
        },
    }


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    communicator = MPI.COMM_WORLD
    _validate_arguments(parser, arguments, communicator.size)

    _validate_output_paths(arguments, communicator)

    setup_started = MPI.Wtime()
    calculation = _calculation(arguments)
    layout = TwoRadialDistributed2DLayout(
        calculation.representation,
        communicator.size,
        process_grid=arguments.process_grid,
    )
    root_state, parent_metadata = _root_initial_state(
        arguments, calculation, communicator
    )
    local_initial_state = scatter_two_radial_2d_state(
        layout,
        communicator,
        root_state if communicator.rank == 0 else None,
    )
    setup_time = communicator.allreduce(MPI.Wtime() - setup_started, op=MPI.MAX)

    solver = TwoRadialDistributed2DGroundStateSolver(
        layout,
        communicator,
        time_step=arguments.time_step,
        max_iterations=arguments.max_iterations,
        energy_tolerance=arguments.energy_tolerance,
        state_tolerance=arguments.state_tolerance,
        residual_tolerance=arguments.residual_tolerance,
        integrator=arguments.integrator,
        max_krylov_dimension=arguments.max_krylov_dimension,
        krylov_rtol=arguments.krylov_rtol,
        krylov_atol=0.0,
        max_krylov_refinements=arguments.max_krylov_refinements,
        krylov_reorthogonalize=True,
        progress_interval=arguments.sample_interval,
        observable_evaluator=calculation.observable_evaluator,
        sample_interval=arguments.sample_interval,
        log_path=arguments.log_file,
        profile=True,
    )

    communicator.Barrier()
    warmup_started = MPI.Wtime()
    solver.hamiltonian(local_initial_state)
    warmup_time = communicator.allreduce(MPI.Wtime() - warmup_started, op=MPI.MAX)
    solver.hamiltonian.reset_profile()
    solver.reductions.reset_profile()

    communicator.Barrier()
    solve_started = MPI.Wtime()
    result = solver.solve(local_initial_state)
    solve_time = communicator.allreduce(MPI.Wtime() - solve_started, op=MPI.MAX)
    profile = _profile_maxima(result, communicator)
    last = result.history[-1]

    if communicator.rank == 0:
        local_sizes = np.asarray(
            [np.prod(layout.owned_shape(rank)) for rank in range(layout.n_ranks)]
        )
        print(f"MPI ranks:                {communicator.size}")
        print(f"Process grid:             {layout.process_grid}")
        print(f"System:                   {arguments.system}")
        print(f"Integrator:               {arguments.integrator}")
        print(f"Time step:                {arguments.time_step}")
        print(f"Maximum Krylov dimension: {arguments.max_krylov_dimension}")
        print(f"Legacy symmetry label:    {'sp' if arguments.system != 'HD+' else 's'}")
        print(f"Electronic points/element: {arguments.electronic_points_per_element}")
        initial_label = arguments.initial_state or "generated product"
        print(f"Initial state:            {initial_label}")
        print(f"l_max:                    {arguments.l_max}")
        print(f"Global coefficients:      {np.prod(layout.global_shape)}")
        print(f"Local coefficients:       {local_sizes.min()}--{local_sizes.max()}")
        print(f"Converged:                {result.converged}")
        print(f"Termination reason:       {result.termination_reason}")
        print(f"Iterations:               {result.iterations}")
        print(f"Energy:                   {result.energy:.12f} Eh")
        print(f"Energy change:            {last.energy_change:.3e} Eh")
        print(f"State change:             {result.state_change:.3e}")
        print(f"Eigenstate residual:      {result.residual_norm:.3e}")
        print(f"Norm:                     {result.norm:.12f}")
        print(f"Representation setup:     {setup_time:.6f} s")
        print(f"Hamiltonian warmup:       {warmup_time:.6f} s")
        print(f"Distributed solve time:   {solve_time:.6f} s")
        print(f"Solve time per iteration: {solve_time / result.iterations:.6f} s")
        print(f"Hamiltonian calls:        {int(profile['hamiltonian_calls'])}")
        print(
            "Krylov operator calls:    "
            f"{int(np.sum(result.krylov_operator_calls))}"
        )
        print(
            "Krylov accepted attempts: "
            f"{int(np.sum(result.krylov_accepted_substeps))}"
        )
        print(
            "Krylov rejected attempts: "
            f"{int(np.sum(result.krylov_rejected_substeps))}"
        )
        print(
            "Krylov max dimension:     "
            f"{int(np.max(result.krylov_dimensions))}"
        )
        print(
            "Krylov max error estimate: "
            f"{np.max(result.krylov_error_estimates):.6e}"
        )
        print(f"Coordinate-1 work (max):  {profile['coordinate1_seconds']:.6f} s")
        print(f"Coordinate-2 work (max):  {profile['coordinate2_seconds']:.6f} s")
        print(f"Coulomb work (max):       {profile['coulomb_seconds']:.6f} s")
        print(f"Coordinate-1 halo (max):  {profile['coordinate1_halo_seconds']:.6f} s")
        print(f"Coordinate-2 halo (max):  {profile['coordinate2_halo_seconds']:.6f} s")
        print(
            "Collective reductions (max): "
            f"{profile['collective_seconds']:.6f} s "
            f"({int(profile['collective_calls'])} calls, "
            f"{int(profile['collective_values'])} values)"
        )

    if not result.converged:
        if communicator.rank == 0:
            print("State archive:            not written (solver did not converge)")
        raise SystemExit(2)

    calculation_metadata = {
        "method": "normalized imaginary-time propagation",
        "integrator": arguments.integrator,
        "radial_operator_mode": calculation.radial_operator_mode,
        "radial_coulomb_method": calculation.radial_coulomb_method,
        "electronic_points_per_element": arguments.electronic_points_per_element,
        "l_max": arguments.l_max,
        "energy": result.energy,
        "norm": result.norm,
        "converged": result.converged,
        "iterations": result.iterations,
        "imaginary_time": result.imaginary_time,
        "state_change": result.state_change,
        "residual_norm": result.residual_norm,
        "time_step": arguments.time_step,
        "max_krylov_dimension": arguments.max_krylov_dimension,
        "krylov_rtol": arguments.krylov_rtol,
        "max_krylov_refinements": arguments.max_krylov_refinements,
        "energy_tolerance": arguments.energy_tolerance,
        "state_tolerance": arguments.state_tolerance,
        "residual_tolerance": arguments.residual_tolerance,
        "parent_state": parent_metadata,
        "final_observables": (
            None if not result.samples else asdict(result.samples[-1].observables)
        ),
    }
    saved = save_distributed_two_radial_2d_state(
        arguments.output_state,
        result.state,
        communicator,
        physical_state=_physical_metadata(calculation),
        calculation=calculation_metadata,
    )
    if communicator.rank == 0:
        print(f"State archive:            {saved}")


if __name__ == "__main__":
    main()
