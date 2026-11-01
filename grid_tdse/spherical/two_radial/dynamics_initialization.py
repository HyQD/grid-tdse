"""Ground-state initialization for non-BO diatomic-ion dynamics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from numbers import Integral
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping

import numpy as np

from .angular import Parity
from .coulomb import RadialCoulombMethod
from .diatomic import (
    DiatomicIonGroundStateResult,
    NonBornOppenheimerDiatomicIonGroundState,
    _diatomic_parameters,
    _pair_exchange_parity,
    _parity,
)
from .distributed_2d_state import (
    DistributedTwoRadial2DState,
    TwoRadialDistributed2DLayout,
    scatter_two_radial_2d_state,
)
from .grids import FEMDVRGridSpec
from .state_archive import TwoRadialState, load_two_radial_state
from .system import TwoRadialSystem


DiatomicDynamicsInitialStatePolicy = Literal[
    "require",
    "compute-if-missing",
    "recompute",
]


@dataclass(frozen=True)
class DiatomicDynamicsInitializationResult:
    """Prepared global dynamics state and its setup provenance."""

    state: np.ndarray
    source: Literal["archive", "computed"]
    archive_path: Path
    archive: TwoRadialState
    setup_wall_time: float
    ground_state_wall_time: float | None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible initialization metadata."""

        return {
            "source": self.source,
            "archive_path": str(self.archive_path),
            "state_data_id": self.archive.state_data_id,
            "representation_id": self.archive.representation_id,
            "setup_wall_time": self.setup_wall_time,
            "ground_state_wall_time": self.ground_state_wall_time,
        }


@dataclass(frozen=True)
class DistributedDiatomicDynamicsInitializationResult:
    """One rank's Cartesian initial state and shared setup provenance."""

    state: DistributedTwoRadial2DState
    source: Literal["archive", "computed"]
    archive_path: Path
    state_data_id: str
    representation_id: str
    setup_wall_time: float
    ground_state_wall_time: float | None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible initialization metadata."""

        return {
            "source": self.source,
            "archive_path": str(self.archive_path),
            "state_data_id": self.state_data_id,
            "representation_id": self.representation_id,
            "setup_wall_time": self.setup_wall_time,
            "ground_state_wall_time": self.ground_state_wall_time,
        }


def prepare_non_bo_diatomic_dynamics_initial_state(
    representation: TwoRadialSystem,
    *,
    system: str,
    electronic_grid: FEMDVRGridSpec,
    internuclear_grid: FEMDVRGridSpec,
    pair_exchange_parity: Parity | None,
    archive_path: str | PathLike[str],
    policy: DiatomicDynamicsInitialStatePolicy = "require",
    inversion_parity: Parity = 1,
    symmetry_adapted_basis: bool | None = None,
    codata_set: str = "2022",
    radial_coulomb_method: RadialCoulombMethod = "poisson",
    ground_state_options: Mapping[str, Any] | None = None,
    symmetry_tolerance: float = 1.0e-8,
) -> DiatomicDynamicsInitializationResult:
    """Load or compute a compatible Davidson state for a dynamics basis.

    ``representation`` must be the full electronic-parity, fixed-``M_tot=0``
    dynamics representation.  A loaded or newly computed symmetry-adapted
    ground state is validated and embedded into that larger angular basis.
    """

    started = perf_counter()
    policy = _initial_state_policy(policy)
    path = Path(archive_path)
    if not isinstance(representation, TwoRadialSystem):
        raise TypeError("representation must be a TwoRadialSystem")
    if not isinstance(electronic_grid, FEMDVRGridSpec):
        raise TypeError("electronic_grid must be a FEMDVRGridSpec")
    if not isinstance(internuclear_grid, FEMDVRGridSpec):
        raise TypeError("internuclear_grid must be a FEMDVRGridSpec")

    parameters = _diatomic_parameters(system, codata_set)
    pair_parity = _pair_exchange_parity(pair_exchange_parity, parameters)
    inversion_parity = _parity(inversion_parity, "inversion_parity")
    if symmetry_adapted_basis is None:
        symmetry_adapted_basis = parameters.has_identical_pair
    if not isinstance(symmetry_adapted_basis, (bool, np.bool_)):
        raise TypeError("symmetry_adapted_basis must be boolean or None")
    symmetry_adapted_basis = bool(symmetry_adapted_basis)
    tolerance = _positive_real(symmetry_tolerance, "symmetry_tolerance")
    options = {} if ground_state_options is None else dict(ground_state_options)

    _validate_dynamics_representation(
        representation,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        pair_exchange_parity=pair_parity,
    )
    _validate_ground_state_options(options)

    compute = policy == "recompute" or (
        policy == "compute-if-missing" and not path.is_file()
    )
    if policy == "require" and not path.is_file():
        raise FileNotFoundError(f"required ground-state archive does not exist: {path}")

    ground_state_wall_time = None
    if compute:
        if path.exists() and not path.is_file():
            raise ValueError(f"ground-state archive path is not a file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        solve_started = perf_counter()
        result = _compute_ground_state(
            system=parameters.name,
            electronic_grid=electronic_grid,
            internuclear_grid=internuclear_grid,
            inversion_parity=inversion_parity,
            pair_exchange_parity=pair_parity,
            symmetry_adapted_basis=symmetry_adapted_basis,
            representation=representation,
            codata_set=codata_set,
            radial_coulomb_method=radial_coulomb_method,
            options=options,
        )
        ground_state_wall_time = perf_counter() - solve_started
        _validate_computed_ground_state(result, tolerance=tolerance)
        result.save(path, overwrite=policy == "recompute")
        source: Literal["archive", "computed"] = "computed"
    else:
        source = "archive"

    archive = load_two_radial_state(path)
    _validate_ground_state_archive(
        archive,
        parameters=parameters,
        inversion_parity=inversion_parity,
        pair_exchange_parity=pair_parity,
        symmetry_adapted_basis=symmetry_adapted_basis,
        tolerance=tolerance,
        residual_tolerance=float(options.get("residual_tolerance", 1.0e-8)),
    )
    state = archive.embed_into(representation)
    state = _normalized_state(representation, state)
    return DiatomicDynamicsInitializationResult(
        state=state,
        source=source,
        archive_path=path,
        archive=archive,
        setup_wall_time=perf_counter() - started,
        ground_state_wall_time=ground_state_wall_time,
    )


def prepare_distributed_non_bo_diatomic_dynamics_initial_state(
    layout: TwoRadialDistributed2DLayout,
    communicator,
    *,
    root: int = 0,
    **initialization_options,
) -> DistributedDiatomicDynamicsInitializationResult:
    """Prepare on ``root``, synchronize, and scatter a Cartesian state."""

    if not isinstance(layout, TwoRadialDistributed2DLayout):
        raise TypeError("layout must be a TwoRadialDistributed2DLayout")
    rank, size = _communicator_rank_size(communicator)
    root = _validated_root(root, size)
    if size != layout.n_ranks:
        raise ValueError(
            f"communicator has {size} ranks; layout requires {layout.n_ranks}"
        )

    global_result = None
    original_error = None
    error = None
    if rank == root:
        try:
            global_result = prepare_non_bo_diatomic_dynamics_initial_state(
                layout.representation,
                **initialization_options,
            )
        except Exception as exc:  # keep all ranks on the same collective path
            original_error = exc
            error = (type(exc).__name__, str(exc))
    error = communicator.bcast(error, root=root)
    if error is not None:
        if original_error is not None:
            raise original_error
        raise RuntimeError(
            f"rank {root} could not prepare the dynamics initial state: {error[1]}"
        )

    metadata = communicator.bcast(
        None if global_result is None else global_result.metadata,
        root=root,
    )
    _barrier(communicator)
    local_state = scatter_two_radial_2d_state(
        layout,
        communicator,
        None if global_result is None else global_result.state,
        root=root,
    )
    return DistributedDiatomicDynamicsInitializationResult(
        state=local_state,
        source=metadata["source"],
        archive_path=Path(metadata["archive_path"]),
        state_data_id=metadata["state_data_id"],
        representation_id=metadata["representation_id"],
        setup_wall_time=float(metadata["setup_wall_time"]),
        ground_state_wall_time=(
            None
            if metadata["ground_state_wall_time"] is None
            else float(metadata["ground_state_wall_time"])
        ),
    )


def _compute_ground_state(
    *,
    system,
    electronic_grid,
    internuclear_grid,
    inversion_parity,
    pair_exchange_parity,
    symmetry_adapted_basis,
    representation,
    codata_set,
    radial_coulomb_method,
    options,
):
    calculation = NonBornOppenheimerDiatomicIonGroundState(
        system,
        electronic_grid=electronic_grid,
        internuclear_grid=internuclear_grid,
        inversion_parity=inversion_parity,
        pair_exchange_parity=pair_exchange_parity,
        symmetry_adapted_basis=symmetry_adapted_basis,
        l_max=representation.basis1.l_max,
        m_max=representation.basis1.m_max,
        lambda_max=representation.multipoles.lambda_max,
        mu_max=representation.multipoles.mu_max,
        M_tot=0,
        radial_coulomb_method=radial_coulomb_method,
        codata_set=codata_set,
        solver="davidson",
        initial_state="product",
        **options,
    )
    return calculation.solve()


def _validate_dynamics_representation(
    representation,
    *,
    electronic_grid,
    internuclear_grid,
    pair_exchange_parity,
):
    if representation.M_tot != 0:
        raise ValueError("dynamics initialization requires compact M_tot=0 storage")
    if representation.grid1.specification != electronic_grid:
        raise ValueError("dynamics electronic grid does not match electronic_grid")
    if representation.grid2.specification != internuclear_grid:
        raise ValueError("dynamics internuclear grid does not match internuclear_grid")
    if representation.basis1.coordinate_parity is not None:
        raise ValueError(
            "dynamics electronic basis must retain both even and odd channels"
        )
    if representation.basis2.coordinate_parity != pair_exchange_parity:
        raise ValueError(
            "dynamics internuclear basis does not preserve pair_exchange_parity"
        )
    if representation.basis1.l_max != representation.basis2.l_max:
        raise ValueError("dynamics coordinates must use the same l_max")
    if representation.basis1.m_max != representation.basis2.m_max:
        raise ValueError("dynamics coordinates must use the same m_max")


def _validate_ground_state_options(options):
    reserved = {
        "system",
        "electronic_grid",
        "internuclear_grid",
        "inversion_parity",
        "pair_exchange_parity",
        "symmetry_adapted_basis",
        "l_max",
        "m_max",
        "lambda_max",
        "mu_max",
        "M_tot",
        "radial_coulomb_method",
        "codata_set",
        "initial_state",
        "solver",
    }
    conflicts = reserved.intersection(options)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"ground_state_options cannot override: {names}")


def _validate_computed_ground_state(result, *, tolerance):
    if not result.converged:
        raise RuntimeError(
            "Davidson ground-state initialization did not converge: "
            f"{result.termination_reason}"
        )
    if abs(result.inversion_expectation - result.inversion_parity) > tolerance:
        raise RuntimeError("computed ground state failed inversion-symmetry validation")
    if result.pair_exchange_parity is not None and (
        result.pair_exchange_expectation is None
        or abs(
            result.pair_exchange_expectation - result.pair_exchange_parity
        ) > tolerance
    ):
        raise RuntimeError("computed ground state failed pair-symmetry validation")


def _validate_ground_state_archive(
    archive,
    *,
    parameters,
    inversion_parity,
    pair_exchange_parity,
    symmetry_adapted_basis,
    tolerance,
    residual_tolerance,
):
    physical = archive.physical_state
    if physical.get("system_family") != "non_bo_diatomic_ion":
        raise ValueError("ground-state archive is not a non-BO diatomic-ion state")
    if physical.get("system") != parameters.name:
        raise ValueError(
            f"ground-state archive contains {physical.get('system')!r}; "
            f"expected {parameters.name!r}"
        )
    stored_parameters = physical.get("parameters")
    expected_parameters = _json_value(asdict(parameters))
    if stored_parameters != expected_parameters:
        stored_codata = (
            stored_parameters.get("codata_set")
            if isinstance(stored_parameters, Mapping)
            else None
        )
        if stored_codata != parameters.codata_set:
            raise ValueError(
                "ground-state archive CODATA set does not match dynamics: "
                f"{stored_codata!r} != {parameters.codata_set!r}"
            )
        raise ValueError(
            "ground-state archive physical parameters do not match dynamics"
        )

    symmetry = physical.get("symmetry", {})
    expected_symmetry = {
        "inversion_parity": inversion_parity,
        "pair_exchange_parity": pair_exchange_parity,
        "symmetry_adapted_basis": symmetry_adapted_basis,
        "M_tot": 0,
    }
    for name, expected in expected_symmetry.items():
        if symmetry.get(name) != expected:
            raise ValueError(
                f"ground-state archive symmetry {name!r} is "
                f"{symmetry.get(name)!r}; expected {expected!r}"
            )

    calculation = archive.calculation
    if calculation.get("converged") is not True:
        raise ValueError("ground-state archive is not marked as converged")
    residual = calculation.get("residual_norm")
    if not isinstance(residual, (int, float)) or not np.isfinite(residual):
        raise ValueError("ground-state archive has no finite residual norm")
    if residual > residual_tolerance:
        raise ValueError(
            "ground-state archive residual exceeds the requested tolerance: "
            f"{residual:.3e} > {residual_tolerance:.3e}"
        )
    inversion = calculation.get("inversion_expectation")
    if not isinstance(inversion, (int, float)) or (
        abs(inversion - inversion_parity) > tolerance
    ):
        raise ValueError("ground-state archive failed inversion-symmetry validation")
    if pair_exchange_parity is not None:
        pair = calculation.get("pair_exchange_expectation")
        if not isinstance(pair, (int, float)) or (
            abs(pair - pair_exchange_parity) > tolerance
        ):
            raise ValueError("ground-state archive failed pair-symmetry validation")


def _normalized_state(representation, state):
    values = np.asarray(state)
    norm_squared = representation.inner_product(values, values).real
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("dynamics initial state must have finite positive norm")
    return np.ascontiguousarray(
        values / np.sqrt(norm_squared),
        dtype=np.result_type(values.dtype, complex),
    )


def _initial_state_policy(value):
    choices = {"require", "compute-if-missing", "recompute"}
    if value not in choices:
        names = ", ".join(sorted(choices))
        raise ValueError(f"initial_state_policy must be one of: {names}")
    return value


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _communicator_rank_size(communicator):
    rank = (
        communicator.Get_rank()
        if hasattr(communicator, "Get_rank")
        else communicator.rank
    )
    size = (
        communicator.Get_size()
        if hasattr(communicator, "Get_size")
        else communicator.size
    )
    return int(rank), int(size)


def _validated_root(root, size):
    if isinstance(root, (bool, np.bool_)) or not isinstance(root, Integral):
        raise TypeError("root must be an integer")
    root = int(root)
    if root < 0 or root >= size:
        raise ValueError(f"root must be between 0 and {size - 1}")
    return root


def _barrier(communicator):
    barrier = getattr(communicator, "Barrier", None)
    if barrier is None:
        barrier = getattr(communicator, "barrier", None)
    if not callable(barrier):
        raise TypeError("communicator must provide Barrier() or barrier()")
    barrier()


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
