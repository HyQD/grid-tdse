from dataclasses import asdict
import json
import os
from pathlib import Path

from mpi4py import MPI
import numpy as np

from grid_tdse.spherical.two_radial import (
    DiatomicIonDynamicsObservableEvaluator,
    DiatomicIonDynamicsSampler,
    DiatomicIonDynamicsSamplingPolicy,
    DiatomicIonPropagationDiagnostics,
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDistributed2DLayout,
    TwoRadialDistributed2DReductions,
    TwoRadialDistributedLayout,
    TwoRadialDistributedReductions,
)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size != 4:
    raise RuntimeError(
        f"phase 9b gate requires 4 ranks, received {size}"
    )


def scaled(factor):
    def action(state):
        if isinstance(state, np.ndarray):
            return factor * state
        return type(state)(
            state.layout,
            state.rank,
            factor * state.coefficients,
        )

    return action


def leaves(value, prefix=""):
    if isinstance(value, dict):
        output = {}
        for name, item in value.items():
            key = f"{prefix}.{name}" if prefix else name
            output.update(leaves(item, key))
        return output

    if isinstance(value, (tuple, list)):
        output = {}
        for index, item in enumerate(value):
            output.update(leaves(item, f"{prefix}.{index}"))
        return output

    return {prefix: value}


def maximum_difference(actual, expected):
    actual = leaves(asdict(actual))
    expected = leaves(asdict(expected))

    if actual.keys() != expected.keys():
        raise AssertionError("snapshot schemas differ")

    maximum = 0.0
    for name, reference in expected.items():
        value = actual[name]

        if reference is None or isinstance(reference, str):
            if value != reference:
                raise AssertionError(
                    f"{name}: {value!r} != {reference!r}"
                )
            continue

        difference = abs(float(value) - float(reference))
        maximum = max(maximum, difference)

        if not np.isclose(
            value,
            reference,
            rtol=5.0e-12,
            atol=5.0e-12,
        ):
            raise AssertionError(
                f"{name}: difference={difference:.3e}"
            )

    return maximum


grid_r = FEMDVRGridSpec(
    [0.0, 1.0, 2.0, 3.5, 6.0],
    [4, 4, 4, 4],
)
grid_R = FEMDVRGridSpec(
    [0.0, 0.8, 1.8, 3.0, 5.0],
    [4, 4, 4, 4],
)

setup = NonBornOppenheimerDiatomicIonGroundState(
    "H2+",
    electronic_grid=grid_r,
    internuclear_grid=grid_R,
    inversion_parity=1,
    pair_exchange_parity=1,
    symmetry_adapted_basis=False,
    l_max=1,
    m_max=1,
    lambda_max=2,
    mu_max=2,
    M_tot=0,
    radial_coulomb_method="poisson",
    radial_operator_mode="element-local",
    codata_set="2022",
    initial_state="product",
    max_iterations=1,
    use_preconditioner=False,
    progress_interval=None,
)

representation = setup.representation

rng = np.random.default_rng(901)
state = rng.normal(size=representation.psi_shape)
state = state + 1j * rng.normal(
    size=representation.psi_shape
)
state *= 0.37 / np.sqrt(
    representation.inner_product(state, state).real
)

evaluator = DiatomicIonDynamicsObservableEvaluator(
    representation,
    setup.parameters,
    scaled(2.5),
    gauge="velocity",
    fields=(
        lambda time: 0.3,
        lambda time: -0.1,
    ),
    velocity_action=scaled(-0.75),
)

expected = evaluator(state, 0.625)
comparisons = []

for coordinate, label in (
    (1, "1d-r"),
    (2, "1d-R"),
):
    layout = TwoRadialDistributedLayout(
        representation,
        size,
        partitioned_coordinate=coordinate,
    )
    local_state = layout.distribute(state)[rank]
    reductions = TwoRadialDistributedReductions(
        layout,
        comm,
    )
    actual = evaluator.distributed(
        local_state,
        reductions,
        0.625,
    )
    comparisons.append(
        (
            label,
            maximum_difference(actual, expected),
        )
    )

layout = TwoRadialDistributed2DLayout(
    representation,
    size,
    process_grid=(2, 2),
)
local_state = layout.distribute(state)[rank]
reductions = TwoRadialDistributed2DReductions(
    layout,
    comm,
)
actual = evaluator.distributed_2d(
    local_state,
    reductions,
    0.625,
)
comparisons.append(
    (
        "2d-2x2",
        maximum_difference(actual, expected),
    )
)

sampler = DiatomicIonDynamicsSampler(
    evaluator,
    policy=DiatomicIonDynamicsSamplingPolicy(
        interval=2,
    ),
)

initial = sampler.sample_distributed_2d(
    local_state,
    reductions,
    0,
    0.0,
)
skipped = sampler.sample_distributed_2d(
    local_state,
    reductions,
    1,
    0.1,
)
checkpoint = sampler.sample_distributed_2d(
    local_state,
    reductions,
    1,
    0.1,
    checkpoint=True,
    diagnostics=DiatomicIonPropagationDiagnostics(
        wall_time=0.25,
        krylov_operator_calls=11,
        krylov_accepted_substeps=2,
        krylov_rejected_substeps=1,
        krylov_refinement_depth=1,
        krylov_dimension=7,
        krylov_error_estimate=1.0e-11,
    ),
)
final = sampler.sample_distributed_2d(
    local_state,
    reductions,
    3,
    0.3,
    final=True,
)

steps = sampler.time_series.accepted_steps.tolist()

if skipped is not None or steps != [0, 1, 3]:
    raise AssertionError(
        f"sampling policy failed: {steps}"
    )

if (
    not initial.is_initial
    or not checkpoint.is_checkpoint
    or not final.is_final
):
    raise AssertionError(
        "mandatory sample flags are incorrect"
    )

if checkpoint.propagation.krylov_dimension != 7:
    raise AssertionError(
        "Krylov diagnostics were not retained"
    )

comm.Barrier()

if rank == 0:
    payload = {
        "phase": "9b",
        "status": "PASS",
        "mpi_ranks": size,
        "norm": expected.norm,
        "field_symbol": expected.field_symbol,
        "field_amplitude": expected.field_amplitude,
        "conditional_energy": (
            expected.conditional_energy
        ),
        "sample_steps": steps,
        "comparisons": dict(comparisons),
    }

    result_dir = Path(
        os.environ[
            "GRID_TDSE_PHASE9B_RESULT_DIR"
        ]
    )
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination = result_dir / (
        f"phase9b-{os.environ['SLURM_JOB_ID']}.json"
    )
    destination.write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    print("Phase 9b gate: PASS")
    print(f"MPI ranks: {size}")
    print(f"Norm: {expected.norm:.12f}")
    print(
        f"Field: {expected.field_symbol}="
        f"{expected.field_amplitude:.12f}"
    )
    print(
        "Sample steps: "
        + ",".join(map(str, steps))
    )

    for label, difference in comparisons:
        print(
            f"{label} max difference: "
            f"{difference:.3e}"
        )

    print(f"Result JSON: {destination}")
