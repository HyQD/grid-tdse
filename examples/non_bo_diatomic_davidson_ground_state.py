"""CODATA-2018 non-BO diatomic-ion ground state with Davidson.

The radial representation matches the legacy/co-worker reference grid
``8e21-4e21_16-8``: eight 21-point electronic FEM-DVR elements on
``[0, 16]`` and four 21-point internuclear elements on ``[0, 8]``.

For example::

    python examples/non_bo_diatomic_davidson_ground_state.py \
        --system H2+ --l-max 0

For angular-cutoff continuation, the example automatically uses the same
system's saved ``l_max - 2`` state when it is available. An explicit
``--initial-state`` overrides that lookup; otherwise the product state is the
fallback.
"""

import argparse
from pathlib import Path
from time import perf_counter

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
)


CODATA_SET = "2018"
RESIDUAL_TOLERANCE = 1.0e-8
MAX_ITERATIONS = 500
MAXIMUM_SUBSPACE_DIMENSION = 30
RETAINED_RITZ_VECTORS = 4
DAVIDSON_PRECONDITIONER = "one-body-spectral"
SAMPLE_INTERVAL = 10

# Co-worker reference energies on this grid, indexed by system and l_max.
# Their stated precision is eight decimal places, so comparisons below use
# exactly that precision rather than the full computed energy.
REFERENCE_ENERGIES = {
    "H2+": {
        0: -0.51565872,
        2: -0.57817060,
        4: -0.59068402,
        6: -0.59429873,
        8: -0.59565550,
    },
    "HD+": {
        0: -0.51026392,
        2: -0.57212175,
        4: -0.58835832,
        6: -0.59366972,
        8: -0.59572012,
    },
    "D2+": {
        0: -0.51775485,
        2: -0.58008322,
        4: -0.59245357,
        6: -0.59600755,
        8: -0.59733748,
    },
}

ELECTRONIC_GRID = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
    n_points=[21] * 8,
)
INTERNUCLEAR_GRID = FEMDVRGridSpec(
    nodes=[0.0, 2.0, 4.0, 6.0, 8.0],
    n_points=[21] * 4,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        choices=("H2+", "HD+", "D2+"),
        required=True,
    )
    parser.add_argument("--l-max", type=int, required=True)
    parser.add_argument(
        "--initial-state",
        type=Path,
        help="compatible lower-l_max archive; default: product state",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing state and logfile outputs",
    )
    arguments = parser.parse_args()
    if arguments.l_max < 0 or arguments.l_max % 2:
        parser.error("--l-max must be a nonnegative even integer")
    return arguments


def _state_path(system: str, l_max: int) -> Path:
    identical_pair = system != "HD+"
    system_label = system.replace("+", "p")
    symmetry_label = "sp" if identical_pair else "s"
    return OUTPUT_DIRECTORY / (
        f"{system_label}__{symmetry_label}_8e21-4e21_16-8_{l_max}__"
        f"davidson_codata2018_{RESIDUAL_TOLERANCE:.0e}.npz"
    )


arguments = _arguments()
system = arguments.system
l_max = arguments.l_max
identical_pair = system != "HD+"
pair_exchange_parity = +1 if identical_pair else None
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "davidson_ground_states"
state_path = _state_path(system, l_max)
log_path = state_path.with_name(state_path.stem + "__log.txt")

previous_state_path = (
    _state_path(system, l_max - 2) if l_max >= 2 else None
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
    electronic_grid=ELECTRONIC_GRID,
    internuclear_grid=INTERNUCLEAR_GRID,
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
    codata_set=CODATA_SET,
    initial_state=initial_state,
    solver="davidson",
    max_iterations=MAX_ITERATIONS,
    residual_tolerance=RESIDUAL_TOLERANCE,
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
    progress_interval=SAMPLE_INTERVAL,
    sample_interval=SAMPLE_INTERVAL,
    log_path=log_path,
    log_overwrite=arguments.overwrite,
    use_preconditioner=False,
)

solve_started = perf_counter()
result = calculation.solve()
solve_wall_time = perf_counter() - solve_started
last = result.history[-1]
reference = REFERENCE_ENERGIES[system].get(l_max)

print(f"System:                   {result.parameters.name}")
print(f"CODATA set:               {result.parameters.codata_set}")
print("Radial grid:              8e21-4e21_16-8")
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
if reference is not None:
    computed_at_reference_precision = f"{result.energy:.8f}"
    reference_at_reference_precision = f"{reference:.8f}"
    print(f"Energy at 8 decimals:     {computed_at_reference_precision} Eh")
    print(f"Reference energy:         {reference_at_reference_precision} Eh")
    print(
        "Matches reference:        "
        f"{computed_at_reference_precision == reference_at_reference_precision}"
    )
else:
    print("Reference energy:         N/A")
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

if not result.converged:
    raise RuntimeError("refusing to save an unconverged ground state")

saved_path = result.save(state_path, overwrite=arguments.overwrite)
print(f"Saved state:              {saved_path}")
print(f"Progress logfile:         {log_path}")
