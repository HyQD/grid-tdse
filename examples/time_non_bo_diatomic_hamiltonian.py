"""Time non-BO diatomic TDSE Hamiltonian components on a complex state.

The default FEM-DVR grids are the grids used by
``non_bo_diatomic_ground_state.py``. For example::

    python examples/time_non_bo_diatomic_hamiltonian.py H2+ 4

Custom grids can be supplied explicitly::

    python examples/time_non_bo_diatomic_hamiltonian.py HD+ 4 \
        --r-nodes 0 4 8 16 --r-points 11 11 11 \
        --R-nodes 0 2 4 6 8 --R-points 21 21 21 21

One point count is broadcast over all elements, so ``--r-points 11`` is
equivalent to ``--r-points 11 11 11`` for the default electronic nodes.

The field benchmarks use a unit temporal amplitude and therefore measure the
spatial length- and velocity-gauge actions. Setup is excluded from every
action timing. Preconditioner construction is reported separately. Complete
Crank--Nicolson steps use the production one-body preconditioner and report
end-to-end linear-solver timings and diagnostics.
"""

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import get_num_threads

from integrator_lib import CrankNicolson

from grid_tdse.spherical.two_radial import (
    AngularBasisSpec,
    AngularPairBasis,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    TwoRadialAxialDipoleInteraction,
    TwoRadialRealTimePreconditioner,
)


DEFAULT_r_NODES = [0.0, 4.0, 8.0, 16.0]
DEFAULT_r_POINTS = [11, 11, 11]
DEFAULT_R_NODES = [0.0, 2.0, 4.0, 6.0, 8.0]
DEFAULT_R_POINTS = [21, 21, 21, 21]


@dataclass(frozen=True)
class Timing:
    minimum: float
    median: float


@dataclass(frozen=True)
class CrankNicolsonTiming:
    timing: Timing
    minimum_iterations: int
    maximum_iterations: int
    median_iterations: float
    minimum_hamiltonian_calls: int
    maximum_hamiltonian_calls: int
    median_hamiltonian_calls: float
    maximum_residual_norm: float


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Time the principal non-BO diatomic TDSE Hamiltonian actions."
        )
    )
    parser.add_argument("molecule", choices=("H2+", "HD+", "D2+"))
    parser.add_argument("lmax", type=int)
    parser.add_argument(
        "--r-nodes",
        nargs="+",
        type=float,
        default=DEFAULT_r_NODES,
        metavar="R",
        help="electronic FEM-DVR element boundaries",
    )
    parser.add_argument(
        "--r-points",
        nargs="+",
        type=int,
        default=DEFAULT_r_POINTS,
        metavar="N",
        help="electronic FEM-DVR points per element",
    )
    parser.add_argument(
        "--R-nodes",
        nargs="+",
        type=float,
        default=DEFAULT_R_NODES,
        metavar="R",
        help="internuclear FEM-DVR element boundaries",
    )
    parser.add_argument(
        "--R-points",
        nargs="+",
        type=int,
        default=DEFAULT_R_POINTS,
        metavar="N",
        help="internuclear FEM-DVR points per element",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.25,
        help="time step used by the real-time preconditioner",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="radial tile size used by the optimized Coulomb action",
    )
    parser.add_argument(
        "--radial-operator-mode",
        choices=("global", "element-local"),
        default="element-local",
        help=(
            "radial one-body action; 'global' retains the dense reference path"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="number of timed calls after one warm-up call",
    )
    parser.add_argument(
        "--cn-repeats",
        type=int,
        default=3,
        help="number of complete CN steps timed after one warm-up step",
    )
    parser.add_argument(
        "--linear-solver",
        choices=("gmres", "bicgstab"),
        default="bicgstab",
        help="Krylov solver used for the complete CN-step benchmarks",
    )
    parser.add_argument(
        "--linear-rtol",
        type=float,
        default=0.0,
        help="relative Krylov tolerance used by the CN benchmarks",
    )
    parser.add_argument(
        "--linear-atol",
        type=float,
        default=1.0e-10,
        help="absolute Krylov tolerance used by the CN benchmarks",
    )
    parser.add_argument(
        "--linear-max-iterations",
        type=int,
        default=None,
        help="optional maximum Krylov iteration count",
    )
    parser.add_argument(
        "--gmres-restart",
        type=int,
        default=None,
        help="optional GMRES restart length; ignored by BiCGSTAB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the representative complex state",
    )
    return parser.parse_args()


def grid_specification(nodes, points, name):
    n_elements = len(nodes) - 1
    if len(points) == 1:
        points = points * n_elements
    if len(points) != n_elements:
        raise ValueError(
            f"{name} requires one point count per element or one count to "
            "broadcast over every element"
        )
    return FEMDVRGridSpec(nodes=nodes, n_points=points)


def benchmark(function, repeats):
    function()
    elapsed = np.empty(repeats)
    for repeat in range(repeats):
        start = perf_counter()
        function()
        elapsed[repeat] = perf_counter() - start
    return Timing(minimum=float(elapsed.min()), median=float(np.median(elapsed)))


def benchmark_crank_nicolson(
    hamiltonian,
    state,
    time_step,
    preconditioner,
    repeats,
    *,
    solver,
    rtol,
    atol,
    maxiter,
    restart,
):
    integrator = CrankNicolson(
        solver=solver,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        restart=restart,
        preconditioner=preconditioner,
    )
    hamiltonian_calls = 0

    def linear_rhs(values, time):
        nonlocal hamiltonian_calls
        hamiltonian_calls += 1
        return -1j * hamiltonian(values, time)

    # Exclude compilation and other first-call setup from the measurements.
    integrator(state, 0.0, time_step, linear_rhs)

    elapsed = np.empty(repeats)
    iterations = np.empty(repeats, dtype=int)
    calls = np.empty(repeats, dtype=int)
    residual_norms = np.empty(repeats)
    for repeat in range(repeats):
        hamiltonian_calls = 0
        start = perf_counter()
        integrator(state, 0.0, time_step, linear_rhs)
        elapsed[repeat] = perf_counter() - start
        iterations[repeat] = integrator.last_iterations
        calls[repeat] = hamiltonian_calls
        residual_norms[repeat] = integrator.last_residual_norm

    return CrankNicolsonTiming(
        timing=Timing(
            minimum=float(elapsed.min()),
            median=float(np.median(elapsed)),
        ),
        minimum_iterations=int(iterations.min()),
        maximum_iterations=int(iterations.max()),
        median_iterations=float(np.median(iterations)),
        minimum_hamiltonian_calls=int(calls.min()),
        maximum_hamiltonian_calls=int(calls.max()),
        median_hamiltonian_calls=float(np.median(calls)),
        maximum_residual_norm=float(residual_norms.max()),
    )


def integer_range(minimum, maximum, median):
    if minimum == maximum:
        return str(minimum)
    return f"{minimum}-{maximum} ({median:g})"


args = parse_arguments()
if args.lmax < 1:
    raise ValueError("lmax must be at least one for a nonzero dipole action")
if args.repeats < 1:
    raise ValueError("repeats must be positive")
if args.cn_repeats < 1:
    raise ValueError("cn-repeats must be positive")
if args.time_step <= 0.0 or not np.isfinite(args.time_step):
    raise ValueError("time-step must be positive and finite")
if args.tile_size < 1:
    raise ValueError("tile-size must be positive")
if args.linear_rtol < 0.0 or not np.isfinite(args.linear_rtol):
    raise ValueError("linear-rtol must be nonnegative and finite")
if args.linear_atol < 0.0 or not np.isfinite(args.linear_atol):
    raise ValueError("linear-atol must be nonnegative and finite")
if args.linear_rtol == args.linear_atol == 0.0:
    raise ValueError("at least one linear tolerance must be positive")
if args.linear_max_iterations is not None and args.linear_max_iterations < 1:
    raise ValueError("linear-max-iterations must be positive")
if args.gmres_restart is not None and args.gmres_restart < 1:
    raise ValueError("gmres-restart must be positive")

electronic_grid = grid_specification(args.r_nodes, args.r_points, "r grid")
internuclear_grid = grid_specification(args.R_nodes, args.R_points, "R grid")

has_identical_pair = args.molecule in {"H2+", "D2+"}
pair_exchange_parity = +1 if has_identical_pair else None
electron_basis = AngularBasisSpec(args.lmax, args.lmax)
pair_basis = AngularBasisSpec(
    args.lmax,
    args.lmax,
    coordinate_parity=pair_exchange_parity,
)
angular_pairs = AngularPairBasis(electron_basis, pair_basis, M_tot=0)

# Construct the expected compact state shape before the high-level interface
# builds the equivalent representation.
n_r = electronic_grid.build(args.lmax).n_points
n_R = internuclear_grid.build(args.lmax).n_points
state_shape = (angular_pairs.n_pairs, n_r, n_R)
rng = np.random.default_rng(args.seed)
initial_state = rng.standard_normal(state_shape) + 1j * rng.standard_normal(
    state_shape
)

calculation = NonBornOppenheimerDiatomicIonTDSE(
    args.molecule,
    electronic_grid=electronic_grid,
    internuclear_grid=internuclear_grid,
    initial_state=initial_state,
    pair_exchange_parity=pair_exchange_parity,
    fields=(),
    field_gauge="length",
    l_max=args.lmax,
    m_max=args.lmax,
    lambda_max=2 * args.lmax,
    mu_max=2 * args.lmax,
    M_tot=0,
    codata_set="2022",
    radial_coulomb_method="poisson",
    radial_operator_mode=args.radial_operator_mode,
    integrator="crank-nicolson",
    integrator_options={"preconditioner": False},
    initial_time=0.0,
    final_time=args.time_step,
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

representation = calculation.representation
state = calculation.initial_state.copy()
state /= np.sqrt(representation.inner_product(state, state).real)
hamiltonian = calculation.field_free_hamiltonian

def apply_one_body_r():
    return hamiltonian.apply_one_body_coordinate1(state)


def apply_one_body_R():
    return hamiltonian.apply_one_body_coordinate2(state)


def apply_coulomb():
    return hamiltonian.apply_coulomb(state)


def unit_field(time):
    del time
    return 1.0


length_interaction = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
    representation,
    calculation.parameters,
    gauge="length",
    fields=(unit_field,),
    radial_operator_mode=args.radial_operator_mode,
)
velocity_interaction = TwoRadialAxialDipoleInteraction.for_diatomic_ion(
    representation,
    calculation.parameters,
    gauge="velocity",
    fields=(unit_field,),
    radial_operator_mode=args.radial_operator_mode,
)


def apply_length_interaction():
    return length_interaction(state, 0.0)


def apply_velocity_interaction():
    return velocity_interaction(state, 0.0)


def apply_field_free_hamiltonian():
    return hamiltonian(state)


def length_gauge_hamiltonian(values, time):
    result = hamiltonian(values)
    result += length_interaction(values, time)
    return result


def velocity_gauge_hamiltonian(values, time):
    result = hamiltonian(values)
    result += velocity_interaction(values, time)
    return result


def apply_length_gauge_hamiltonian():
    return length_gauge_hamiltonian(state, 0.0)


def apply_velocity_gauge_hamiltonian():
    return velocity_gauge_hamiltonian(state, 0.0)


components = (
    apply_one_body_r()
    + apply_one_body_R()
    + apply_coulomb()
)
np.testing.assert_allclose(
    components,
    apply_field_free_hamiltonian(),
    rtol=2.0e-13,
    atol=2.0e-13,
)

actions = {
    "Electronic one-body h_r = T_r": apply_one_body_r,
    "Internuclear h_R = T_R + 1/R": apply_one_body_R,
    "Multipole Coulomb W": apply_coulomb,
    "Length-gauge field interaction": apply_length_interaction,
    "Velocity-gauge field interaction": apply_velocity_interaction,
    "Complete field-free H0": apply_field_free_hamiltonian,
    "Cumulative H(t), length gauge": apply_length_gauge_hamiltonian,
    "Cumulative H(t), velocity gauge": apply_velocity_gauge_hamiltonian,
}
timings = {
    name: benchmark(function, args.repeats)
    for name, function in actions.items()
}

preconditioner_setup = benchmark(
    lambda: TwoRadialRealTimePreconditioner(
        representation,
        args.time_step,
    ),
    args.repeats,
)
preconditioner = TwoRadialRealTimePreconditioner(
    representation,
    args.time_step,
)
timings["Real-time preconditioner application"] = benchmark(
    lambda: preconditioner(state),
    args.repeats,
)

cn_options = {
    "solver": args.linear_solver,
    "rtol": args.linear_rtol,
    "atol": args.linear_atol,
    "maxiter": args.linear_max_iterations,
    "restart": args.gmres_restart,
}
cn_timings = {
    "CN step, length gauge": benchmark_crank_nicolson(
        length_gauge_hamiltonian,
        state,
        args.time_step,
        preconditioner,
        args.cn_repeats,
        **cn_options,
    ),
    "CN step, velocity gauge": benchmark_crank_nicolson(
        velocity_gauge_hamiltonian,
        state,
        args.time_step,
        preconditioner,
        args.cn_repeats,
        **cn_options,
    ),
}

print(f"System:                         {calculation.system_name}")
print(f"l_max = m_max:                 {args.lmax}")
print("M_tot:                          0")
print(f"Pair-exchange parity:           {pair_exchange_parity}")
print(f"Electronic radial points:       {representation.grid1.n_points}")
print(f"Internuclear radial points:     {representation.grid2.n_points}")
print(f"Retained angular pairs:         {representation.angular_pairs.n_pairs}")
print(f"Complex state shape:            {state.shape}")
print(f"Complex state size:             {state.nbytes / 2**20:.3f} MiB")
print(f"Poisson Coulomb tile size:      {args.tile_size}")
print(f"Radial operator mode:           {args.radial_operator_mode}")
print(f"Numba threads:                  {get_num_threads()}")
print(f"Timed repetitions:              {args.repeats}")
print(f"Timed CN repetitions:           {args.cn_repeats}")
print(f"CN linear solver:               {args.linear_solver}")
print(f"CN linear rtol:                 {args.linear_rtol:.3e}")
print(f"CN linear atol:                 {args.linear_atol:.3e}")
print("CN preconditioner:              one-body")
print()
print(f"{'Action':38s} {'minimum / s':>14s} {'median / s':>14s}")
print("-" * 70)
for name, timing in timings.items():
    print(
        f"{name:38s} "
        f"{timing.minimum:14.2g} "
        f"{timing.median:14.2g}"
    )
print(
    f"{'Preconditioner construction (one time)':38s} "
    f"{preconditioner_setup.minimum:14.2g} "
    f"{preconditioner_setup.median:14.2g}"
)
print()
print("Complete Crank--Nicolson step (preconditioned)")
print(
    f"{'Gauge':26s} "
    f"{'minimum / s':>14s} "
    f"{'median / s':>14s} "
    f"{'iterations':>18s} "
    f"{'H calls':>18s} "
    f"{'max residual':>14s}"
)
print("-" * 110)
for name, result in cn_timings.items():
    iteration_summary = integer_range(
        result.minimum_iterations,
        result.maximum_iterations,
        result.median_iterations,
    )
    call_summary = integer_range(
        result.minimum_hamiltonian_calls,
        result.maximum_hamiltonian_calls,
        result.median_hamiltonian_calls,
    )
    print(
        f"{name:26s} "
        f"{result.timing.minimum:14.2g} "
        f"{result.timing.median:14.2g} "
        f"{iteration_summary:>18s} "
        f"{call_summary:>18s} "
        f"{result.maximum_residual_norm:14.2e}"
    )
