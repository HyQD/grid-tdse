"""Validate serial non-BO diatomic field-free and length-gauge dynamics.

The example implements the Phase 9c validation sequence:

1. load or compute a Davidson ground state;
2. verify field-free SIL stationarity up to a global phase;
3. propagate a finite z-polarized length-gauge pulse;
4. compare a time-step refinement against a finer SIL reference; and
5. compare SIL and SIA for the same Hermitian pulse calculation.

The default problem is deliberately small enough for a local smoke test. Use
the radial-grid and angular command-line options for production resolutions.
The selected pulse time series is written as JSON, including the common
Phase 9b observable schema and Krylov diagnostics.
"""

import argparse
from dataclasses import dataclass
import json
from math import isfinite, pi
from pathlib import Path
from time import perf_counter

import numpy as np

from grid_tdse.spherical.two_radial import (
    DiatomicIonDynamicsSamplingPolicy,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonTDSE,
    phase_aligned_two_radial_state_difference,
)


@dataclass(frozen=True)
class SineSquaredElectricField:
    """Finite sine-square electric field with a sinusoidal carrier."""

    amplitude: float
    duration: float
    carrier_cycles: float = 1.0
    phase: float = 0.0
    gauge: str = "length"

    def __call__(self, time: float) -> float:
        if time < 0.0 or time > self.duration:
            return 0.0
        envelope = np.sin(pi * time / self.duration) ** 2
        carrier = np.sin(
            2.0 * pi * self.carrier_cycles * time / self.duration + self.phase
        )
        return float(self.amplitude * envelope * carrier)


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
    parser.add_argument(
        "--ground-state-path",
        type=Path,
        help=(
            "Davidson archive; default: a grid-labelled file under "
            "examples/ground_states"
        ),
    )
    parser.add_argument(
        "--initial-state-policy",
        choices=("require", "compute-if-missing", "recompute"),
        default="compute-if-missing",
    )
    parser.add_argument("--ground-state-residual-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--ground-state-max-iterations", type=int, default=500)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=0.04)
    parser.add_argument("--field-strength", type=float, default=0.02)
    parser.add_argument("--carrier-cycles", type=float, default=1.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-11)
    parser.add_argument("--krylov-atol", type=float, default=1.0e-13)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("phase9c_serial_dynamics.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    if arguments.l_max < 1:
        parser.error("--l-max must be at least 1 for a length-gauge pulse")
    for name in (
        "time_step",
        "final_time",
        "ground_state_residual_tolerance",
        "carrier_cycles",
        "krylov_rtol",
    ):
        value = getattr(arguments, name)
        if not isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.krylov_atol < 0.0 or not isfinite(arguments.krylov_atol):
        parser.error("--krylov-atol must be nonnegative")
    if arguments.ground_state_max_iterations < 1:
        parser.error("--ground-state-max-iterations must be positive")
    if arguments.sample_interval < 1:
        parser.error("--sample-interval must be positive")
    if arguments.electronic_points_per_element < 3:
        parser.error("--electronic-points-per-element must be at least 3")
    if arguments.internuclear_points_per_element < 3:
        parser.error("--internuclear-points-per-element must be at least 3")
    _validate_nodes(parser, "--electronic-nodes", arguments.electronic_nodes)
    _validate_nodes(parser, "--internuclear-nodes", arguments.internuclear_nodes)
    return arguments


def _validate_nodes(parser, option: str, nodes) -> None:
    if len(nodes) < 2 or nodes[0] != 0.0:
        parser.error(f"{option} must contain at least two boundaries and start at zero")
    if any(not isfinite(value) for value in nodes):
        parser.error(f"{option} must contain finite values")
    if any(right <= left for left, right in zip(nodes, nodes[1:])):
        parser.error(f"{option} must be strictly increasing")


def _grid(nodes, points_per_element) -> FEMDVRGridSpec:
    return FEMDVRGridSpec(
        nodes,
        [points_per_element] * (len(nodes) - 1),
    )


def _default_ground_state_path(arguments) -> Path:
    system = arguments.system.replace("+", "p")
    electronic = (
        f"{len(arguments.electronic_nodes) - 1}e"
        f"{arguments.electronic_points_per_element}r"
        f"{arguments.electronic_nodes[-1]:g}"
    )
    internuclear = (
        f"{len(arguments.internuclear_nodes) - 1}e"
        f"{arguments.internuclear_points_per_element}R"
        f"{arguments.internuclear_nodes[-1]:g}"
    )
    return Path(__file__).resolve().parent / "ground_states" / (
        f"phase9c_{system}_{electronic}_{internuclear}_l{arguments.l_max}.npz"
    )


def _state_difference(reference, candidate, calculation) -> float:
    return phase_aligned_two_radial_state_difference(
        reference,
        candidate,
        calculation.representation.inner_product,
    )


arguments = _arguments()
electronic_grid = _grid(
    arguments.electronic_nodes,
    arguments.electronic_points_per_element,
)
internuclear_grid = _grid(
    arguments.internuclear_nodes,
    arguments.internuclear_points_per_element,
)
ground_state_path = (
    _default_ground_state_path(arguments)
    if arguments.ground_state_path is None
    else arguments.ground_state_path
)
if arguments.output_json.exists() and not arguments.overwrite:
    raise FileExistsError(
        f"output JSON exists: {arguments.output_json}; pass --overwrite to replace it"
    )

identical_pair = arguments.system != "HD+"
pair_exchange_parity = 1 if identical_pair else None
sampling_policy = DiatomicIonDynamicsSamplingPolicy(
    interval=arguments.sample_interval
)
integrator_options = {
    "max_krylov_dimension": 20,
    "rtol": arguments.krylov_rtol,
    "atol": arguments.krylov_atol,
    "max_refinements": 8,
    "reorthogonalize": True,
    "breakdown_tolerance": 1.0e-14,
}
ground_state_options = {
    "max_iterations": arguments.ground_state_max_iterations,
    "residual_tolerance": arguments.ground_state_residual_tolerance,
    "maximum_subspace_dimension": 30,
    "retained_ritz_vectors": 4,
    "davidson_preconditioner": "one-body-spectral",
    "progress_interval": 10,
    "sample_interval": 10,
}


def _calculation(*, initial_state, fields, integrator, time_step):
    automatic = initial_state is None
    return NonBornOppenheimerDiatomicIonTDSE(
        arguments.system,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        initial_state=initial_state,
        pair_exchange_parity=pair_exchange_parity,
        initial_state_policy=(
            arguments.initial_state_policy if automatic else "require"
        ),
        ground_state_path=ground_state_path if automatic else None,
        ground_state_symmetry_adapted_basis=identical_pair,
        ground_state_options=ground_state_options if automatic else None,
        fields=fields,
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
        integrator_options=integrator_options,
        time_step=time_step,
        final_time=arguments.final_time,
        observables=None,
        dynamics_sampling_policy=sampling_policy,
        progress_interval=10,
    )


def _run(label, calculation):
    print(f"\nRunning {label}", flush=True)
    started = perf_counter()
    result = calculation.run(progress=True)
    wall_time = perf_counter() - started
    return {
        "label": label,
        "calculation": calculation,
        "result": result,
        "wall_time": wall_time,
    }


field_free = _calculation(
    initial_state=None,
    fields=(),
    integrator="sil",
    time_step=arguments.time_step,
)
archive = field_free.initialization.archive
cases = {
    "field_free_sil": _run("field-free SIL", field_free),
}
pulse = SineSquaredElectricField(
    amplitude=arguments.field_strength,
    duration=arguments.final_time,
    carrier_cycles=arguments.carrier_cycles,
    phase=arguments.phase,
)
for name, integrator, time_step in (
    ("pulse_sil_coarse", "sil", arguments.time_step),
    ("pulse_sil_fine", "sil", arguments.time_step / 2.0),
    ("pulse_sil_reference", "sil", arguments.time_step / 4.0),
    ("pulse_sia_fine", "sia", arguments.time_step / 2.0),
):
    cases[name] = _run(
        name.replace("_", " "),
        _calculation(
            initial_state=archive,
            fields=(pulse,),
            integrator=integrator,
            time_step=time_step,
        ),
    )

field_free_result = cases["field_free_sil"]["result"]
field_free_state_difference = _state_difference(
    field_free.initial_state,
    field_free_result.state,
    field_free,
)
field_free_norm_drift = float(
    np.max(np.abs(field_free_result.norm_history - field_free_result.initial_norm))
)
field_free_energy_drift = max(
    abs(sample.conditional_energy_gain)
    for sample in field_free.dynamics_time_series.samples
)

reference = cases["pulse_sil_reference"]
coarse_error = _state_difference(
    reference["result"].state,
    cases["pulse_sil_coarse"]["result"].state,
    reference["calculation"],
)
fine_error = _state_difference(
    reference["result"].state,
    cases["pulse_sil_fine"]["result"].state,
    reference["calculation"],
)
sil_sia_difference = _state_difference(
    cases["pulse_sil_fine"]["result"].state,
    cases["pulse_sia_fine"]["result"].state,
    reference["calculation"],
)
maximum_norm_drift = max(
    float(np.max(np.abs(case["result"].norm_history - 1.0)))
    for case in cases.values()
)
convergence_ratio = fine_error / coarse_error if coarse_error else 0.0
passed = (
    field_free_state_difference < 1.0e-8
    and field_free_norm_drift < 1.0e-10
    and field_free_energy_drift < 1.0e-8
    and maximum_norm_drift < 1.0e-10
    and fine_error < coarse_error
    and sil_sia_difference < 1.0e-8
)

fine_series = cases["pulse_sil_fine"]["calculation"].dynamics_time_series
report = {
    "phase": "9c",
    "status": "PASS" if passed else "FAIL",
    "system": arguments.system,
    "l_max": arguments.l_max,
    "codata_set": arguments.codata_set,
    "ground_state_path": str(ground_state_path),
    "ground_state_source": field_free.initialization.source,
    "ground_state_residual": archive.calculation["residual_norm"],
    "field_free_state_difference": field_free_state_difference,
    "field_free_norm_drift": field_free_norm_drift,
    "field_free_conditional_energy_drift": field_free_energy_drift,
    "maximum_norm_drift": maximum_norm_drift,
    "pulse_coarse_error": coarse_error,
    "pulse_fine_error": fine_error,
    "pulse_convergence_ratio": convergence_ratio,
    "sil_sia_state_difference": sil_sia_difference,
    "wall_times": {
        name: case["wall_time"] for name, case in cases.items()
    },
    "pulse_sil_fine_time_series": fine_series.records(),
}
arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
arguments.output_json.write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print("\nPhase 9c serial validation")
print(f"Status:                         {report['status']}")
print(f"Ground-state source:            {field_free.initialization.source}")
print(f"Field-free state difference:    {field_free_state_difference:.3e}")
print(f"Field-free norm drift:          {field_free_norm_drift:.3e}")
print(f"Field-free energy drift:        {field_free_energy_drift:.3e} Eh")
print(f"Maximum norm drift:             {maximum_norm_drift:.3e}")
print(f"Coarse pulse state error:       {coarse_error:.3e}")
print(f"Fine pulse state error:         {fine_error:.3e}")
print(f"Fine/coarse convergence ratio:  {convergence_ratio:.3e}")
print(f"SIL/SIA state difference:       {sil_sia_difference:.3e}")
print(f"Result JSON:                    {arguments.output_json}")

if not passed:
    raise SystemExit(1)
