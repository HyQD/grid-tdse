"""Davidson ground state for a configurable non-BO diatomic-ion grid.

Unlike ``non_bo_diatomic_davidson_ground_state.py``, this example does not
contain reference energies or assume a particular radial representation. The
FEM-DVR nodes and the uniform number of points per element are command-line
inputs for each radial coordinate.

For example, the reference calculation's ``8e21-4e21_16-8`` grid can be
reproduced with::

    python examples/spherical/two_radial/non_bo_diatomic_davidson_custom_grid.py \
        --system H2+ --l-max 8 --codata-set 2018 \
        --electronic-nodes 0 2 4 6 8 10 12 14 16 \
        --electronic-points-per-element 21 \
        --internuclear-nodes 0 2 4 6 8 \
        --internuclear-points-per-element 21

The points-per-element values are uniform within each coordinate, so the
compact ``XeY`` grid notation is unambiguous. The two coordinates may use
different values of ``Y``.

When a compatible state for the same system, grid, CODATA set, and
``l_max - 2`` exists in the output directory, it is selected automatically.
Use ``--initial-state`` to override this continuation or start from the
product state when no preceding archive exists.
"""

import argparse
from math import isfinite
from pathlib import Path
from time import perf_counter

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
)


MAXIMUM_SUBSPACE_DIMENSION = 30
RETAINED_RITZ_VECTORS = 4
DAVIDSON_PRECONDITIONER = "one-body-spectral"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=("H2+", "HD+", "D2+"),
        required=True,
    )
    parser.add_argument("--l-max", type=int, required=True)
    parser.add_argument(
        "--electronic-nodes",
        type=float,
        nargs="+",
        required=True,
        metavar="R",
        help="electronic FEM-DVR element boundaries",
    )
    parser.add_argument(
        "--electronic-points-per-element",
        type=int,
        required=True,
        metavar="N",
    )
    parser.add_argument(
        "--internuclear-nodes",
        type=float,
        nargs="+",
        required=True,
        metavar="R",
        help="internuclear FEM-DVR element boundaries",
    )
    parser.add_argument(
        "--internuclear-points-per-element",
        type=int,
        required=True,
        metavar="N",
    )
    parser.add_argument(
        "--codata-set",
        choices=("2018", "2022"),
        default="2022",
    )
    parser.add_argument(
        "--residual-tolerance",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument(
        "--initial-state",
        type=Path,
        help="compatible lower-l_max archive; default: automatic or product",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "davidson_ground_states",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing state and logfile outputs",
    )
    arguments = parser.parse_args()

    if arguments.l_max < 0 or arguments.l_max % 2:
        parser.error("--l-max must be a nonnegative even integer")
    if arguments.electronic_points_per_element < 2:
        parser.error("--electronic-points-per-element must be at least 2")
    if arguments.internuclear_points_per_element < 2:
        parser.error("--internuclear-points-per-element must be at least 2")
    if (
        not isfinite(arguments.residual_tolerance)
        or arguments.residual_tolerance <= 0.0
    ):
        parser.error("--residual-tolerance must be positive")
    if arguments.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if arguments.sample_interval < 1:
        parser.error("--sample-interval must be positive")
    _validate_nodes(parser, "--electronic-nodes", arguments.electronic_nodes)
    _validate_nodes(
        parser,
        "--internuclear-nodes",
        arguments.internuclear_nodes,
    )
    return arguments


def _validate_nodes(
    parser: argparse.ArgumentParser,
    option: str,
    nodes: list[float],
) -> None:
    if len(nodes) < 2:
        parser.error(f"{option} requires at least two boundaries")
    if any(not isfinite(node) for node in nodes):
        parser.error(f"{option} must contain only finite values")
    if nodes[0] != 0.0:
        parser.error(f"{option} must start at zero")
    if any(right <= left for left, right in zip(nodes, nodes[1:])):
        parser.error(f"{option} must be strictly increasing")


def _grid_label(
    electronic_nodes: list[float],
    electronic_points: int,
    internuclear_nodes: list[float],
    internuclear_points: int,
) -> str:
    electronic_elements = len(electronic_nodes) - 1
    internuclear_elements = len(internuclear_nodes) - 1
    return (
        f"{electronic_elements}e{electronic_points}-"
        f"{internuclear_elements}e{internuclear_points}_"
        f"{electronic_nodes[-1]:g}-{internuclear_nodes[-1]:g}"
    )


def _state_path(
    output_directory: Path,
    system: str,
    grid_label: str,
    l_max: int,
    codata_set: str,
    residual_tolerance: float,
) -> Path:
    identical_pair = system != "HD+"
    system_label = system.replace("+", "p")
    symmetry_label = "sp" if identical_pair else "s"
    return output_directory / (
        f"{system_label}__{symmetry_label}_{grid_label}_{l_max}__"
        f"davidson_codata{codata_set}_{residual_tolerance:.0e}.npz"
    )


arguments = _arguments()
system = arguments.system
l_max = arguments.l_max
arguments.output_directory.mkdir(parents=True, exist_ok=True)
identical_pair = system != "HD+"
pair_exchange_parity = +1 if identical_pair else None
grid_label = _grid_label(
    arguments.electronic_nodes,
    arguments.electronic_points_per_element,
    arguments.internuclear_nodes,
    arguments.internuclear_points_per_element,
)
electronic_grid = FEMDVRGridSpec(
    nodes=arguments.electronic_nodes,
    n_points=[arguments.electronic_points_per_element]
    * (len(arguments.electronic_nodes) - 1),
)
internuclear_grid = FEMDVRGridSpec(
    nodes=arguments.internuclear_nodes,
    n_points=[arguments.internuclear_points_per_element]
    * (len(arguments.internuclear_nodes) - 1),
)
state_path = _state_path(
    arguments.output_directory,
    system,
    grid_label,
    l_max,
    arguments.codata_set,
    arguments.residual_tolerance,
)
log_path = state_path.with_name(state_path.stem + "__log.txt")
previous_state_path = (
    _state_path(
        arguments.output_directory,
        system,
        grid_label,
        l_max - 2,
        arguments.codata_set,
        arguments.residual_tolerance,
    )
    if l_max >= 2
    else None
)

if arguments.initial_state is not None:
    initial_state = arguments.initial_state
    initial_state_description = f"explicit archive {initial_state}"
elif previous_state_path is not None and previous_state_path.is_file():
    initial_state = previous_state_path
    initial_state_description = (
        f"automatic l_max={l_max - 2} archive {previous_state_path}"
    )
else:
    initial_state = "product"
    initial_state_description = "product (no preceding archive found)"

print(f"Initial-state selection:  {initial_state_description}", flush=True)
print(f"Output state:             {state_path}", flush=True)

calculation = NonBornOppenheimerDiatomicIonGroundState(
    system,
    electronic_grid=electronic_grid,
    internuclear_grid=internuclear_grid,
    inversion_parity=+1,
    pair_exchange_parity=pair_exchange_parity,
    symmetry_adapted_basis=identical_pair,
    l_max=l_max,
    m_max=l_max,
    lambda_max=2 * l_max,
    mu_max=2 * l_max,
    M_tot=0,
    radial_coulomb_method="poisson",
    radial_operator_mode="element-local",
    codata_set=arguments.codata_set,
    initial_state=initial_state,
    solver="davidson",
    max_iterations=arguments.max_iterations,
    residual_tolerance=arguments.residual_tolerance,
    maximum_subspace_dimension=MAXIMUM_SUBSPACE_DIMENSION,
    retained_ritz_vectors=RETAINED_RITZ_VECTORS,
    eigensolver_breakdown_tolerance=1.0e-14,
    eigensolver_reorthogonalization="always",
    eigensolver_reorthogonalization_threshold=0.5,
    eigensolver_hermiticity_tolerance=1.0e-10,
    eigensolver_reality_tolerance=1.0e-12,
    track_subspace_orthogonality=True,
    davidson_preconditioner=DAVIDSON_PRECONDITIONER,
    davidson_relative_floor=1.0e-8,
    davidson_absolute_floor=1.0e-12,
    davidson_level_shift=0.0,
    progress_interval=arguments.sample_interval,
    sample_interval=arguments.sample_interval,
    log_path=log_path,
    log_overwrite=arguments.overwrite,
    use_preconditioner=False,
)

solve_started = perf_counter()
result = calculation.solve()
solve_wall_time = perf_counter() - solve_started
last = result.history[-1]

print(f"System:                   {result.parameters.name}")
print(f"CODATA set:               {result.parameters.codata_set}")
print(f"Radial grid:              {grid_label}")
print(f"Electronic nodes:         {arguments.electronic_nodes}")
print(
    "Electronic points/elem: "
    f"{arguments.electronic_points_per_element}"
)
print(f"Internuclear nodes:       {arguments.internuclear_nodes}")
print(
    "Internuclear points/el: "
    f"{arguments.internuclear_points_per_element}"
)
print(f"l_max:                    {l_max}")
print(f"Initial state:            {initial_state_description}")
print(f"Ground-state solver:      {result.solver_name}")
print(f"Davidson preconditioner:  {DAVIDSON_PRECONDITIONER}")
print(f"Converged:                {result.converged}")
print(f"Termination reason:       {result.termination_reason}")
print(f"Iterations:               {result.iterations}")
print(f"Hamiltonian actions:      {result.ground_state.hamiltonian_actions}")
print(f"Subspace restarts:        {result.ground_state.restarts}")
print(f"Energy:                   {result.energy:.12f} Eh")
print(f"Energy change:            {last.energy_change:.3e} Eh")
print(f"State change:             {last.state_change:.3e}")
print(f"Eigenstate residual:      {result.residual_norm:.3e}")
print(f"Norm:                     {result.norm:.12f}")
print(f"Solve wall time:          {solve_wall_time:.6f} s")
print(f"<I>:                      {result.inversion_expectation:.12f}")
pair_exchange = result.pair_exchange_expectation
print(
    "<P>:                      "
    + ("N/A" if pair_exchange is None else f"{pair_exchange:.12f}")
)
print(f"<r>:                      {result.observables.r:.12f} a0")
print(f"<R>:                      {result.observables.R:.12f} a0")
print(f"<R^2>:                    {result.observables.R_squared:.12f} a0^2")
print(f"<cos^2(theta_R)>:         {result.observables.cos2_theta_R:.12f}")
print(f"<cos^4(theta_R)>:         {result.observables.cos4_theta_R:.12f}")
print(f"<cos^6(theta_R)>:         {result.observables.cos6_theta_R:.12f}")
print(f"<cos^8(theta_R)>:         {result.observables.cos8_theta_R:.12f}")
print(
    "<R cos^2(theta_R)>:     "
    f"{result.observables.R_cos2_theta_R:.12f} a0"
)

if not result.converged:
    raise RuntimeError("refusing to save an unconverged ground state")

saved_path = result.save(state_path, overwrite=arguments.overwrite)
print(f"Saved state:              {saved_path}")
print(f"Progress logfile:         {log_path}")
