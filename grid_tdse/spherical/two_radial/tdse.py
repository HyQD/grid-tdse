"""High-level real-time TDSE interfaces for two-radial systems."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np

from grid_tdse.real_time import (
    RealTimeCheckpoint,
    RealTimePropagationResult,
    RealTimePropagator,
    RealTimeSnapshotRecord,
    RealTimeSnapshotWriter,
    load_real_time_checkpoint,
    real_time_state_data_id,
)

from .absorbing_boundaries import TwoRadialAbsorbingBoundaries
from .angular import AngularBasisSpec, CoulombMultipoleSpec, Parity
from .atomic import _atomic_parameters
from .coulomb import CoulombTerm, RadialCoulombMethod
from .diatomic import (
    _diatomic_parameters,
    _pair_exchange_parity,
)
from .dipole import (
    DipoleGauge,
    DipoleKernelMode,
    TwoRadialAxialDipoleInteraction,
)
from .dynamics_initialization import (
    DiatomicDynamicsInitialStatePolicy,
    prepare_non_bo_diatomic_dynamics_initial_state,
)
from .dynamics_sampling import (
    DiatomicIonDynamicsObservableEvaluator,
    DiatomicIonDynamicsSampler,
    DiatomicIonDynamicsSamplingPolicy,
    DiatomicIonDynamicsTimeSeries,
)
from .grids import FEMDVRGridSpec
from .preconditioner import build_two_radial_real_time_preconditioner
from .state_archive import TwoRadialState, two_radial_compatibility_metadata
from .system import (
    CoordinateSpec,
    CoupledTwoRadialSystem,
    CoulombKernelMode,
    RadialKernelMode,
    RadialOperatorMode,
    TwoRadialSystemSpec,
    radial_potential,
)


IntegratorName = Literal[
    "crank-nicolson",
    "sil",
    "sia",
    "cf4-sil",
    "cf4-sia",
]
InitialState = np.ndarray | TwoRadialState
Sampler = Callable[[np.ndarray, float], np.ndarray | complex | float]


@dataclass(frozen=True)
class _SignedAxialField:
    """Apply the sign of an axial polarization without changing its gauge."""

    profile: Callable[[float], float]
    sign: float
    gauge: DipoleGauge

    def __call__(self, time: float):
        return self.sign * self.profile(time)


class _TwoRadialTDSE:
    """Shared orchestration for the public atomic and diatomic interfaces."""

    _INTEGRATOR_OPTIONS = {
        "crank-nicolson": {
            "solver",
            "rtol",
            "atol",
            "max_iterations",
            "restart",
            "preconditioner",
            "preconditioner_options",
            "reality_tolerance",
        },
        "sil": {
            "max_krylov_dimension",
            "rtol",
            "atol",
            "max_refinements",
            "reorthogonalize",
            "breakdown_tolerance",
            "reality_tolerance",
        },
        "sia": {
            "max_krylov_dimension",
            "rtol",
            "atol",
            "max_refinements",
            "reorthogonalize",
            "breakdown_tolerance",
            "reality_tolerance",
        },
        "cf4-sil": {
            "max_krylov_dimension",
            "rtol",
            "atol",
            "max_refinements",
            "reorthogonalize",
            "breakdown_tolerance",
            "reality_tolerance",
        },
        "cf4-sia": {
            "max_krylov_dimension",
            "rtol",
            "atol",
            "max_refinements",
            "reorthogonalize",
            "breakdown_tolerance",
            "reality_tolerance",
        },
    }

    def _configure_real_time(
        self,
        *,
        initial_state: InitialState,
        fields,
        field_gauge: DipoleGauge,
        integrator: IntegratorName,
        integrator_options: Mapping[str, Any] | None,
        time_step: float,
        final_time: float,
        initial_time: float,
        observables: Iterable[str] | None,
        samplers: Mapping[str, Sampler] | None,
        checkpoint_path: str | PathLike[str] | None,
        checkpoint_interval: int | None,
        checkpoint_overwrite: bool,
        checkpoint_metadata: Mapping[str, Any] | None,
        wavefunction_snapshot_directory: str | PathLike[str] | None,
        wavefunction_snapshot_interval: int | None,
        wavefunction_snapshot_steps: Iterable[int] | None,
        wavefunction_snapshot_times: Iterable[float] | None,
        wavefunction_snapshot_initial: bool,
        wavefunction_snapshot_final: bool,
        wavefunction_snapshot_prefix: str,
        wavefunction_snapshot_overwrite: bool,
        wavefunction_snapshot_metadata: Mapping[str, Any] | None,
        progress_interval: int | None,
        radial_operator_mode: RadialOperatorMode,
        tile_size: int,
        coulomb_kernel: CoulombKernelMode,
        radial_kernel: RadialKernelMode,
        dipole_kernel: DipoleKernelMode,
    ) -> None:
        if integrator not in self._INTEGRATOR_OPTIONS:
            choices = ", ".join(sorted(self._INTEGRATOR_OPTIONS))
            raise ValueError(f"integrator must be one of: {choices}")
        self.integrator = integrator
        options = {} if integrator_options is None else dict(integrator_options)
        unknown = set(options).difference(self._INTEGRATOR_OPTIONS[integrator])
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown {integrator} integrator options: {names}")
        self.integrator_options = options.copy()

        self.field_gauge = _field_gauge(field_gauge)
        self.fields, field_metadata = _axial_fields(fields, self.field_gauge)
        _validate_field_angular_basis(self.representation, self.fields)
        _validate_velocity_gauge_initial_vector_potential(
            self.fields,
            gauge=self.field_gauge,
            initial_time=initial_time,
        )
        self.initial_state = self._prepare_initial_state(initial_state)
        self.field_free_hamiltonian = self._build_field_free_hamiltonian(
            tile_size=tile_size,
            radial_operator_mode=radial_operator_mode,
            coulomb_kernel=coulomb_kernel,
            radial_kernel=radial_kernel,
        )
        self.radial_operator_mode = radial_operator_mode
        self.dipole_kernel = dipole_kernel
        self.dipole_interaction = (
            self._build_dipole_interaction(self.fields) if self.fields else None
        )
        if (
            integrator in {"sil", "cf4-sil"}
            and self.absorbing_boundaries.has_cap
        ):
            raise ValueError(
                f"integrator {integrator!r} requires a Hermitian Hamiltonian "
                "and cannot be used with a CAP; use 'sia', 'cf4-sia', or "
                "'crank-nicolson'"
            )

        self.propagator = RealTimePropagator(
            initial_time=initial_time,
            final_time=final_time,
            time_step=time_step,
            integrator=integrator,
            linear_solver=options.get("solver", "bicgstab"),
            linear_rtol=options.get("rtol", 0.0),
            linear_atol=options.get("atol", 1.0e-10),
            linear_max_iterations=options.get("max_iterations"),
            gmres_restart=options.get("restart"),
            max_krylov_dimension=options.get("max_krylov_dimension", 20),
            krylov_rtol=(
                options.get("rtol", 1.0e-10)
                if integrator != "crank-nicolson"
                else 1.0e-10
            ),
            krylov_atol=(
                options.get("atol", 0.0)
                if integrator != "crank-nicolson"
                else 0.0
            ),
            max_krylov_refinements=options.get("max_refinements", 8),
            krylov_reorthogonalize=options.get("reorthogonalize", True),
            krylov_breakdown_tolerance=options.get(
                "breakdown_tolerance", 1.0e-14
            ),
            reality_tolerance=options.get("reality_tolerance", 1.0e-10),
            progress_interval=progress_interval,
        )
        if integrator == "crank-nicolson":
            self.preconditioner = build_two_radial_real_time_preconditioner(
                self.representation,
                self.propagator.time_step,
                options.get("preconditioner", True),
                options=options.get("preconditioner_options"),
            )
            self.preconditioner_name = (
                None
                if self.preconditioner is None
                else self.preconditioner.name
            )
        else:
            self.preconditioner = None
            self.preconditioner_name = None

        self.progress_interval = progress_interval
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_overwrite = bool(checkpoint_overwrite)
        self.checkpoint_metadata = (
            None if checkpoint_metadata is None else dict(checkpoint_metadata)
        )
        if (
            wavefunction_snapshot_directory is None
            and (
                wavefunction_snapshot_interval is not None
                or wavefunction_snapshot_steps is not None
                or wavefunction_snapshot_times is not None
            )
        ):
            raise ValueError(
                "wavefunction snapshot selection requires "
                "wavefunction_snapshot_directory"
            )
        self.wavefunction_snapshot_directory = (
            None
            if wavefunction_snapshot_directory is None
            else Path(wavefunction_snapshot_directory)
        )
        self.wavefunction_snapshot_interval = wavefunction_snapshot_interval
        self.wavefunction_snapshot_steps = wavefunction_snapshot_steps
        self.wavefunction_snapshot_times = wavefunction_snapshot_times
        self.wavefunction_snapshot_initial = wavefunction_snapshot_initial
        self.wavefunction_snapshot_final = wavefunction_snapshot_final
        self.wavefunction_snapshot_prefix = wavefunction_snapshot_prefix
        self.wavefunction_snapshot_overwrite = wavefunction_snapshot_overwrite
        self.wavefunction_snapshot_metadata = (
            None
            if wavefunction_snapshot_metadata is None
            else dict(wavefunction_snapshot_metadata)
        )
        self.wavefunction_snapshots: tuple[RealTimeSnapshotRecord, ...] = ()
        self.wavefunction_snapshot_manifest: Path | None = None
        self._snapshot_writer: RealTimeSnapshotWriter | None = None
        self._active_checkpoint_path: Path | None = None
        self._active_restart_checkpoint: RealTimeCheckpoint | None = None
        self._active_execution_mode = "new"
        self._active_checkpoint_metadata = self.checkpoint_metadata
        self._active_parent_state_data_id = real_time_state_data_id(
            self.initial_state
        )
        self.compatibility_metadata = self._build_compatibility_metadata()
        self.dynamics_metadata = {
            "interaction": "electric_dipole" if self.fields else "field_free",
            "polarization_axis": "z" if self.fields else None,
            "field_gauge": self.field_gauge,
            "fields": field_metadata,
            "integrator": self.integrator,
            "integrator_options": _json_value(self.integrator_options),
            "absorbing_boundaries": self.absorbing_boundaries.metadata,
            "radial_operator_mode": self.radial_operator_mode,
            "coulomb_kernel": coulomb_kernel,
            "radial_kernel": radial_kernel,
            "dipole_kernel": dipole_kernel,
        }
        self.samplers = self._prepare_samplers(observables, samplers)

    def run(self, *, progress: bool = True) -> RealTimePropagationResult:
        """Propagate a copy of the configured initial state."""

        return self._run_real_time(None, mode="new", progress=progress)

    def resume(
        self,
        checkpoint: RealTimeCheckpoint | str | PathLike[str],
        *,
        progress: bool = True,
    ) -> RealTimePropagationResult:
        """Exactly resume an interrupted propagation segment."""

        return self._run_real_time(checkpoint, mode="resume", progress=progress)

    def continue_from(
        self,
        checkpoint: RealTimeCheckpoint | str | PathLike[str],
        *,
        progress: bool = True,
    ) -> RealTimePropagationResult:
        """Start a compatible propagation stage from a checkpoint."""

        return self._run_real_time(checkpoint, mode="continue", progress=progress)

    def _run_real_time(self, checkpoint, *, mode, progress):
        """Run a new, exactly resumed, or compatible continuation stage."""

        if not isinstance(progress, (bool, np.bool_)):
            raise TypeError("progress must be boolean")
        if mode not in {"new", "resume", "continue"}:
            raise ValueError("invalid real-time execution mode")
        stored = None
        if mode != "new":
            stored = (
                checkpoint
                if isinstance(checkpoint, RealTimeCheckpoint)
                else load_real_time_checkpoint(checkpoint)
            )
            self._active_parent_state_data_id = stored.state_data_id
            self._active_checkpoint_path = (
                Path(self.checkpoint_path)
                if self.checkpoint_path is not None
                else stored.source_path
            )
        else:
            self._active_parent_state_data_id = real_time_state_data_id(
                self.initial_state
            )
            self._active_checkpoint_path = (
                None if self.checkpoint_path is None else Path(self.checkpoint_path)
            )
        self._active_restart_checkpoint = stored
        self._active_execution_mode = mode
        self._active_checkpoint_metadata = (
            {} if stored is None else dict(stored.metadata)
        )
        if self.checkpoint_metadata is not None:
            self._active_checkpoint_metadata.update(self.checkpoint_metadata)
        configured_interval = self.propagator.progress_interval
        self.propagator.progress_interval = (
            self.progress_interval if progress else None
        )
        try:
            propagation_arguments = dict(
                linear_combination_factory=(
                    self._cf4_linear_combination_factory
                    if self.integrator in {"cf4-sil", "cf4-sia"}
                    else None
                ),
                preconditioner=self.preconditioner,
                samplers=self.samplers,
                state_observer=self._build_state_observer(),
                post_step=(
                    self.absorbing_boundaries.apply_mask
                    if self.absorbing_boundaries.has_mask
                    else None
                ),
                checkpoint_path=self.checkpoint_path,
                checkpoint_interval=self.checkpoint_interval,
                compatibility_metadata=self.compatibility_metadata,
                dynamics_metadata=self.dynamics_metadata,
                checkpoint_metadata=self._active_checkpoint_metadata,
                checkpoint_overwrite=self.checkpoint_overwrite,
            )
            if mode == "new":
                result = self.propagator.propagate(
                    self.initial_state,
                    self.hamiltonian,
                    self.representation.inner_product,
                    **propagation_arguments,
                )
            elif mode == "resume":
                result = self.propagator.resume(
                    stored,
                    self.hamiltonian,
                    self.representation.inner_product,
                    **propagation_arguments,
                )
            else:
                result = self.propagator.continue_from(
                    stored,
                    self.hamiltonian,
                    self.representation.inner_product,
                    **propagation_arguments,
                )
            self._finish_state_observer()
            return result
        finally:
            self.propagator.progress_interval = configured_interval

    __call__ = run

    def hamiltonian(self, state: np.ndarray, time: float) -> np.ndarray:
        """Apply the field-free Hamiltonian and all configured axial fields."""

        result = self.static_hamiltonian(state)
        if self.dipole_interaction is not None:
            result += self.dipole_interaction(state, time)
        return result

    def static_hamiltonian(self, state: np.ndarray) -> np.ndarray:
        """Apply all time-independent propagation-Hamiltonian terms."""

        result = self.field_free_hamiltonian(state)
        if self.absorbing_boundaries.has_cap:
            result += self.absorbing_boundaries.cap_action(state)
        return result

    def _cf4_linear_combination_factory(self, linear_rhs, nodes, weights):
        """Build one structured CF4 generator without duplicate static work."""

        del linear_rhs
        static_weight = float(sum(weights))
        field_weight = 0.0
        if self.dipole_interaction is not None:
            field_weight = sum(
                weight * self.dipole_interaction.field_amplitude(node)
                for node, weight in zip(nodes, weights, strict=True)
            )

        def action(state):
            result = static_weight * self.static_hamiltonian(state)
            if self.dipole_interaction is not None and field_weight != 0.0:
                result += field_weight * self.dipole_interaction.spatial_action(
                    state
                )
            return -1j * result

        return action

    def _prepare_initial_state(self, initial_state: InitialState) -> np.ndarray:
        if isinstance(initial_state, TwoRadialState):
            self._validate_archived_physical_state(initial_state)
            values = initial_state.embed_into(self.representation)
        else:
            values = self.representation.pack_state(initial_state)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("initial_state must contain numeric values")
        if not np.isfinite(values).all():
            raise ValueError("initial_state must contain only finite values")
        norm = self.representation.inner_product(values, values).real
        if norm <= 0.0:
            raise ValueError("initial_state must have positive norm")
        return np.asarray(values, dtype=np.result_type(values.dtype, complex))

    def _validate_archived_physical_state(self, state: TwoRadialState) -> None:
        stored = state.physical_state
        expected = self._physical_state_metadata()
        for key in ("system_family", "system"):
            if stored.get(key) != expected[key]:
                raise ValueError(
                    f"initial-state {key} {stored.get(key)!r} does not match "
                    f"the requested value {expected[key]!r}"
                )
        if _json_value(stored.get("parameters")) != expected["parameters"]:
            stored_codata = stored.get("parameters", {}).get("codata_set")
            expected_codata = expected["parameters"].get("codata_set")
            if stored_codata != expected_codata:
                raise ValueError(
                    "initial-state CODATA set does not match the requested "
                    f"real-time setup: {stored_codata!r} != {expected_codata!r}"
                )
            raise ValueError(
                "initial-state physical parameters do not match the requested "
                "real-time setup"
            )

    def _prepare_samplers(
        self,
        observables: Iterable[str] | None,
        samplers: Mapping[str, Sampler] | None,
    ) -> dict[str, Sampler]:
        if observables is None:
            names = ()
        elif isinstance(observables, str):
            names = (observables,)
        else:
            names = tuple(observables)
        unknown = set(names).difference({"expec_z"})
        if unknown:
            values = ", ".join(sorted(unknown))
            raise ValueError(f"unknown built-in observables: {values}")
        result = {} if samplers is None else dict(samplers)
        if "expec_z" in names:
            if "expec_z" in result:
                raise ValueError("custom sampler name 'expec_z' is reserved")
            position = self._build_position_operator()

            def expec_z(state, time):
                action = position(state, time)
                value = self.representation.inner_product(state, action)
                tolerance = 1.0e-10 * max(1.0, abs(value.real))
                if abs(value.imag) > tolerance:
                    raise RuntimeError(
                        "expec_z acquired a significant imaginary part"
                    )
                return float(value.real)

            result["expec_z"] = expec_z
        if not all(
            isinstance(name, str) and callable(value)
            for name, value in result.items()
        ):
            raise TypeError("samplers must map string names to callables")
        return result

    def _build_position_operator(self):
        return self._build_dipole_interaction((lambda time: 1.0,), gauge="length")

    def _build_state_observer(self):
        if self.wavefunction_snapshot_directory is None:
            self._snapshot_writer = None
            return None
        writer = RealTimeSnapshotWriter(
            self.wavefunction_snapshot_directory,
            interval=self.wavefunction_snapshot_interval,
            steps=self.wavefunction_snapshot_steps,
            times=self.wavefunction_snapshot_times,
            include_initial=self.wavefunction_snapshot_initial,
            include_final=self.wavefunction_snapshot_final,
            prefix=self.wavefunction_snapshot_prefix,
            overwrite=self.wavefunction_snapshot_overwrite,
            compatibility_metadata=self.compatibility_metadata,
            dynamics_metadata=self.dynamics_metadata,
            integrator_metadata=self.propagator.integrator_metadata,
            metadata=self.wavefunction_snapshot_metadata,
            parent_state_data_id=self._active_parent_state_data_id,
            accepted_step_offset=self._accepted_step_offset(),
            inner_product=self.representation.inner_product,
            checkpoint_path=self._active_checkpoint_path,
        )
        self._snapshot_writer = writer
        return writer

    def _finish_state_observer(self) -> None:
        if self._snapshot_writer is None:
            self.wavefunction_snapshots = ()
            self.wavefunction_snapshot_manifest = None
            return
        self.wavefunction_snapshot_manifest = self._snapshot_writer.finish()
        self.wavefunction_snapshots = self._snapshot_writer.records

    def _accepted_step_offset(self) -> int:
        checkpoint = self._active_restart_checkpoint
        if checkpoint is None:
            return 0
        if self._active_execution_mode == "resume":
            return checkpoint.total_steps - checkpoint.current_step
        return checkpoint.total_steps

    def _build_dipole_interaction(self, fields, gauge=None):
        raise NotImplementedError

    def _build_field_free_hamiltonian(self, **options):
        return self.representation.hamiltonian(**options)

    def _build_compatibility_metadata(self):
        return two_radial_compatibility_metadata(
            self.representation,
            physical_state=self._physical_state_metadata(),
        )

    def _physical_state_metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class TwoElectronAtomTDSE(_TwoRadialTDSE):
    """Propagate an infinite-nuclear-mass two-electron atom in real time."""

    def __init__(
        self,
        atom: str,
        *,
        grid: FEMDVRGridSpec,
        initial_state: InitialState,
        fields=(),
        field_gauge: DipoleGauge = "length",
        l_max: int = 2,
        m_max: int | None = None,
        lambda_max: int | None = None,
        mu_max: int | None = None,
        M_tot: int | None = 0,
        radial_coulomb_method: RadialCoulombMethod = "poisson",
        integrator: IntegratorName = "crank-nicolson",
        integrator_options: Mapping[str, Any] | None = None,
        initial_time: float = 0.0,
        final_time: float,
        time_step: float,
        mask=None,
        cap=None,
        observables: Iterable[str] | None = ("expec_z",),
        samplers: Mapping[str, Sampler] | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        checkpoint_overwrite: bool = False,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        wavefunction_snapshot_directory: str | PathLike[str] | None = None,
        wavefunction_snapshot_interval: int | None = None,
        wavefunction_snapshot_steps: Iterable[int] | None = None,
        wavefunction_snapshot_times: Iterable[float] | None = None,
        wavefunction_snapshot_initial: bool = True,
        wavefunction_snapshot_final: bool = True,
        wavefunction_snapshot_prefix: str = "wavefunction",
        wavefunction_snapshot_overwrite: bool = False,
        wavefunction_snapshot_metadata: Mapping[str, Any] | None = None,
        progress_interval: int | None = 10,
        radial_operator_mode: RadialOperatorMode = "global",
        tile_size: int = 2048,
        coulomb_kernel: CoulombKernelMode = "reference",
        radial_kernel: RadialKernelMode = "reference",
        dipole_kernel: DipoleKernelMode = "reference",
    ) -> None:
        _validate_absorber_pair(mask, cap, "atomic radial coordinate")
        if not isinstance(grid, FEMDVRGridSpec):
            raise TypeError("grid must be a FEMDVRGridSpec")
        self.parameters = _atomic_parameters(atom)
        self.grid_spec = grid
        self.l_max, self.m_max, self.lambda_max, self.mu_max = _angular_limits(
            l_max, m_max, lambda_max, mu_max
        )
        self.M_tot = M_tot
        self.radial_coulomb_method = radial_coulomb_method
        basis = AngularBasisSpec(self.l_max, self.m_max)
        coordinate = CoordinateSpec(
            grid=grid,
            basis=basis,
            kinetic_prefactor=-0.5 / self.parameters.electron_mass,
            one_body_terms=(
                radial_potential(
                    lambda radius: self.parameters.electron_charge
                    * self.parameters.nuclear_charge
                    / radius
                ),
            ),
        )
        self.specification = TwoRadialSystemSpec(
            coordinate1=coordinate,
            coordinate2=coordinate,
            multipoles=CoulombMultipoleSpec(self.lambda_max, self.mu_max),
            coulomb_terms=(
                CoulombTerm(
                    radial_scale=1.0,
                    charge_product=self.parameters.electron_charge**2,
                ),
            ),
            M_tot=M_tot,
            radial_coulomb_method=radial_coulomb_method,
        )
        options = {} if integrator_options is None else dict(integrator_options)
        needs_global_operators = radial_operator_mode == "global" or (
            integrator == "crank-nicolson"
            and options.get("preconditioner", True) is not False
        )
        self.representation = self.specification.build(
            radial_operator_mode=(
                "global" if needs_global_operators else "element-local"
            )
        )
        self.absorbing_boundaries = TwoRadialAbsorbingBoundaries(
            self.representation,
            mask1=mask,
            mask2=mask,
            cap1=cap,
            cap2=cap,
        )
        self._configure_real_time(
            initial_state=initial_state,
            fields=fields,
            field_gauge=field_gauge,
            integrator=integrator,
            integrator_options=integrator_options,
            time_step=time_step,
            final_time=final_time,
            initial_time=initial_time,
            observables=observables,
            samplers=samplers,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            checkpoint_overwrite=checkpoint_overwrite,
            checkpoint_metadata=checkpoint_metadata,
            wavefunction_snapshot_directory=wavefunction_snapshot_directory,
            wavefunction_snapshot_interval=wavefunction_snapshot_interval,
            wavefunction_snapshot_steps=wavefunction_snapshot_steps,
            wavefunction_snapshot_times=wavefunction_snapshot_times,
            wavefunction_snapshot_initial=wavefunction_snapshot_initial,
            wavefunction_snapshot_final=wavefunction_snapshot_final,
            wavefunction_snapshot_prefix=wavefunction_snapshot_prefix,
            wavefunction_snapshot_overwrite=wavefunction_snapshot_overwrite,
            wavefunction_snapshot_metadata=wavefunction_snapshot_metadata,
            progress_interval=progress_interval,
            radial_operator_mode=radial_operator_mode,
            tile_size=tile_size,
            coulomb_kernel=coulomb_kernel,
            radial_kernel=radial_kernel,
            dipole_kernel=dipole_kernel,
        )

    @property
    def atom_name(self) -> str:
        return self.parameters.name

    def _build_dipole_interaction(self, fields, gauge=None):
        return TwoRadialAxialDipoleInteraction.for_two_electron_atom(
            self.representation,
            self.parameters,
            gauge=self.field_gauge if gauge is None else gauge,
            fields=fields,
            radial_operator_mode=self.radial_operator_mode,
            dipole_kernel=self.dipole_kernel,
        )

    def _physical_state_metadata(self):
        return {
            "system_family": "two_electron_atom",
            "system": self.parameters.name,
            "parameters": _json_value(asdict(self.parameters)),
        }


class NonBornOppenheimerDiatomicIonTDSE(_TwoRadialTDSE):
    """Propagate a non-Born--Oppenheimer three-body diatomic ion."""

    def __new__(
        cls,
        *args,
        wf_representation: Literal["uncoupled", "coupled"] = "uncoupled",
        **kwargs,
    ):
        if cls is NonBornOppenheimerDiatomicIonTDSE:
            if wf_representation == "coupled":
                from .coupled_tdse import (
                    CoupledNonBornOppenheimerDiatomicIonTDSE,
                )

                return object.__new__(
                    CoupledNonBornOppenheimerDiatomicIonTDSE
                )
            if wf_representation != "uncoupled":
                raise ValueError(
                    "wf_representation must be 'uncoupled' or 'coupled'"
                )
        return super().__new__(cls)

    def __init__(
        self,
        system: str,
        *,
        wf_representation: Literal["uncoupled"] = "uncoupled",
        electronic_grid: FEMDVRGridSpec,
        internuclear_grid: FEMDVRGridSpec,
        initial_state: InitialState | None = None,
        pair_exchange_parity: Parity | None,
        initial_state_policy: DiatomicDynamicsInitialStatePolicy = "require",
        ground_state_path: str | PathLike[str] | None = None,
        ground_state_inversion_parity: Parity = 1,
        ground_state_symmetry_adapted_basis: bool | None = None,
        ground_state_options: Mapping[str, Any] | None = None,
        fields=(),
        field_gauge: DipoleGauge = "length",
        l_max: int = 2,
        m_max: int | None = None,
        lambda_max: int | None = None,
        mu_max: int | None = None,
        M_tot: int | None = 0,
        codata_set: str = "2022",
        radial_coulomb_method: RadialCoulombMethod = "poisson",
        integrator: IntegratorName = "crank-nicolson",
        integrator_options: Mapping[str, Any] | None = None,
        initial_time: float = 0.0,
        final_time: float,
        time_step: float,
        mask_r=None,
        mask_R=None,
        cap_r=None,
        cap_R=None,
        observables: Iterable[str] | None = ("expec_z",),
        samplers: Mapping[str, Sampler] | None = None,
        dynamics_sampling_policy: DiatomicIonDynamicsSamplingPolicy | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        checkpoint_overwrite: bool = False,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        wavefunction_snapshot_directory: str | PathLike[str] | None = None,
        wavefunction_snapshot_interval: int | None = None,
        wavefunction_snapshot_steps: Iterable[int] | None = None,
        wavefunction_snapshot_times: Iterable[float] | None = None,
        wavefunction_snapshot_initial: bool = True,
        wavefunction_snapshot_final: bool = True,
        wavefunction_snapshot_prefix: str = "wavefunction",
        wavefunction_snapshot_overwrite: bool = False,
        wavefunction_snapshot_metadata: Mapping[str, Any] | None = None,
        progress_interval: int | None = 10,
        radial_operator_mode: RadialOperatorMode = "global",
        tile_size: int = 2048,
        coulomb_kernel: CoulombKernelMode = "reference",
        radial_kernel: RadialKernelMode = "reference",
        dipole_kernel: DipoleKernelMode = "reference",
    ) -> None:
        if wf_representation != "uncoupled":
            raise ValueError(
                "the uncoupled constructor requires "
                "wf_representation='uncoupled'"
            )
        self.wf_representation = "uncoupled"
        _validate_absorber_pair(mask_r, cap_r, "electronic coordinate r")
        _validate_absorber_pair(mask_R, cap_R, "internuclear coordinate R")
        if not isinstance(electronic_grid, FEMDVRGridSpec):
            raise TypeError("electronic_grid must be a FEMDVRGridSpec")
        if not isinstance(internuclear_grid, FEMDVRGridSpec):
            raise TypeError("internuclear_grid must be a FEMDVRGridSpec")
        self.parameters = _diatomic_parameters(system, codata_set)
        self.pair_exchange_parity = _pair_exchange_parity(
            pair_exchange_parity, self.parameters
        )
        self.electronic_grid_spec = electronic_grid
        self.internuclear_grid_spec = internuclear_grid
        self.l_max, self.m_max, self.lambda_max, self.mu_max = _angular_limits(
            l_max, m_max, lambda_max, mu_max
        )
        self.M_tot = M_tot
        self.radial_coulomb_method = radial_coulomb_method
        if dynamics_sampling_policy is None:
            dynamics_sampling_policy = DiatomicIonDynamicsSamplingPolicy()
        if not isinstance(
            dynamics_sampling_policy, DiatomicIonDynamicsSamplingPolicy
        ):
            raise TypeError(
                "dynamics_sampling_policy must be a "
                "DiatomicIonDynamicsSamplingPolicy or None"
            )
        self.dynamics_sampling_policy = dynamics_sampling_policy

        # A z-polarized field changes electronic l by one, so both electronic
        # parities are retained. Pair exchange remains a symmetry for an
        # identical positive pair and fixes the parity of the R coordinate.
        electron_basis = AngularBasisSpec(self.l_max, self.m_max)
        pair_basis = AngularBasisSpec(
            self.l_max,
            self.m_max,
            coordinate_parity=(
                self.pair_exchange_parity
                if self.parameters.has_identical_pair
                else None
            ),
        )
        p = self.parameters
        self.specification = TwoRadialSystemSpec(
            coordinate1=CoordinateSpec(
                grid=electronic_grid,
                basis=electron_basis,
                kinetic_prefactor=-0.5 / p.electronic_reduced_mass,
            ),
            coordinate2=CoordinateSpec(
                grid=internuclear_grid,
                basis=pair_basis,
                kinetic_prefactor=-0.5 / p.internuclear_reduced_mass,
                one_body_terms=(
                    radial_potential(
                        lambda radius: p.reference_charge
                        * p.partner_charge
                        / radius
                    ),
                ),
            ),
            multipoles=CoulombMultipoleSpec(
                self.lambda_max,
                self.mu_max,
                lambda_parity=(1 if p.has_identical_pair else None),
            ),
            coulomb_terms=(
                CoulombTerm(
                    p.reference_center_scale,
                    p.electron_charge * p.reference_charge,
                ),
                CoulombTerm(
                    p.partner_center_scale,
                    p.electron_charge * p.partner_charge,
                ),
            ),
            M_tot=M_tot,
            radial_coulomb_method=radial_coulomb_method,
        )
        options = {} if integrator_options is None else dict(integrator_options)
        needs_global_operators = radial_operator_mode == "global" or (
            integrator == "crank-nicolson"
            and options.get("preconditioner", True) is not False
        )
        self.representation = self.specification.build(
            radial_operator_mode=(
                "global" if needs_global_operators else "element-local"
            )
        )
        self.initialization = None
        if initial_state is None:
            if ground_state_path is None:
                raise ValueError(
                    "ground_state_path is required when initial_state is None"
                )
            self.initialization = (
                prepare_non_bo_diatomic_dynamics_initial_state(
                    self.representation,
                    system=self.parameters.name,
                    electronic_grid=electronic_grid,
                    internuclear_grid=internuclear_grid,
                    pair_exchange_parity=self.pair_exchange_parity,
                    archive_path=ground_state_path,
                    policy=initial_state_policy,
                    inversion_parity=ground_state_inversion_parity,
                    symmetry_adapted_basis=(
                        ground_state_symmetry_adapted_basis
                    ),
                    codata_set=codata_set,
                    radial_coulomb_method=radial_coulomb_method,
                    ground_state_options=ground_state_options,
                )
            )
            initial_state = self.initialization.state
        else:
            if initial_state_policy != "require":
                raise ValueError(
                    "initial_state_policy must be 'require' when an explicit "
                    "initial_state is provided"
                )
            if ground_state_path is not None:
                raise ValueError(
                    "ground_state_path cannot be combined with an explicit "
                    "initial_state"
                )
            if ground_state_options is not None:
                raise ValueError(
                    "ground_state_options cannot be combined with an explicit "
                    "initial_state"
                )
        self.absorbing_boundaries = TwoRadialAbsorbingBoundaries(
            self.representation,
            mask1=mask_r,
            mask2=mask_R,
            cap1=cap_r,
            cap2=cap_R,
        )
        self._configure_real_time(
            initial_state=initial_state,
            fields=fields,
            field_gauge=field_gauge,
            integrator=integrator,
            integrator_options=integrator_options,
            time_step=time_step,
            final_time=final_time,
            initial_time=initial_time,
            observables=observables,
            samplers=samplers,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            checkpoint_overwrite=checkpoint_overwrite,
            checkpoint_metadata=checkpoint_metadata,
            wavefunction_snapshot_directory=wavefunction_snapshot_directory,
            wavefunction_snapshot_interval=wavefunction_snapshot_interval,
            wavefunction_snapshot_steps=wavefunction_snapshot_steps,
            wavefunction_snapshot_times=wavefunction_snapshot_times,
            wavefunction_snapshot_initial=wavefunction_snapshot_initial,
            wavefunction_snapshot_final=wavefunction_snapshot_final,
            wavefunction_snapshot_prefix=wavefunction_snapshot_prefix,
            wavefunction_snapshot_overwrite=wavefunction_snapshot_overwrite,
            wavefunction_snapshot_metadata=wavefunction_snapshot_metadata,
            progress_interval=progress_interval,
            radial_operator_mode=radial_operator_mode,
            tile_size=tile_size,
            coulomb_kernel=coulomb_kernel,
            radial_kernel=radial_kernel,
            dipole_kernel=dipole_kernel,
        )
        self.dynamics_metadata["initial_state"] = (
            {"source": "provided"}
            if self.initialization is None
            else self.initialization.metadata
        )
        self.dynamics_metadata["sampling_policy"] = asdict(
            self.dynamics_sampling_policy
        )
        self.dynamics_observable_evaluator = (
            DiatomicIonDynamicsObservableEvaluator(
                self.representation,
                self.parameters,
                self.field_free_hamiltonian,
                gauge=self.field_gauge,
                fields=self.fields,
                radial_operator_mode=self.radial_operator_mode,
            )
        )
        self.dynamics_time_series = DiatomicIonDynamicsTimeSeries(())
        self._dynamics_sampler: DiatomicIonDynamicsSampler | None = None

    @property
    def system_name(self) -> str:
        return self.parameters.name

    def _build_dipole_interaction(self, fields, gauge=None):
        return TwoRadialAxialDipoleInteraction.for_diatomic_ion(
            self.representation,
            self.parameters,
            gauge=self.field_gauge if gauge is None else gauge,
            fields=fields,
            radial_operator_mode=self.radial_operator_mode,
            dipole_kernel=self.dipole_kernel,
        )

    def _build_state_observer(self):
        snapshot_observer = super()._build_state_observer()
        accepted_step_offset = self._accepted_step_offset()
        context = self._active_checkpoint_metadata.setdefault(
            "dynamics_sampling_context",
            {},
        )
        references = context.get("reference_values")
        if references is None:
            state = (
                self.initial_state
                if self._active_restart_checkpoint is None
                else self._active_restart_checkpoint.state
            )
            snapshot = self.dynamics_observable_evaluator(
                state,
                self.propagator.initial_time,
            )
            references = {
                "norm": snapshot.norm,
                "energy_numerator": snapshot.energy_numerator,
                "conditional_energy": snapshot.conditional_energy,
            }
            context["reference_values"] = references
        sampler = DiatomicIonDynamicsSampler(
            self.dynamics_observable_evaluator,
            policy=self.dynamics_sampling_policy,
            accepted_step_offset=accepted_step_offset,
            reference_norm=(None if references is None else references["norm"]),
            reference_energy_numerator=(
                None if references is None else references["energy_numerator"]
            ),
            reference_conditional_energy=(
                None if references is None else references["conditional_energy"]
            ),
        )
        self._dynamics_sampler = sampler

        def observe(state, accepted_step, time, diagnostics, final, checkpoint):
            sampler.sample(
                state,
                accepted_step,
                time,
                diagnostics=diagnostics,
                final=final,
                checkpoint=checkpoint,
            )
            if "reference_values" not in context:
                context["reference_values"] = sampler.reference_values
            if snapshot_observer is not None:
                snapshot_observer(
                    state,
                    accepted_step,
                    time,
                    diagnostics,
                    final,
                    checkpoint,
                )

        return observe

    def _finish_state_observer(self) -> None:
        if self._dynamics_sampler is None:
            raise RuntimeError("dynamics sampler was not initialized")
        self.dynamics_time_series = self._dynamics_sampler.time_series
        super()._finish_state_observer()

    def _physical_state_metadata(self):
        return {
            "system_family": "non_bo_diatomic_ion",
            "system": self.parameters.name,
            "parameters": _json_value(asdict(self.parameters)),
        }


def _axial_fields(fields, gauge: DipoleGauge):
    if fields is None:
        entries = ()
    elif callable(fields):
        entries = (fields,)
    elif (
        isinstance(fields, Sequence)
        and len(fields) == 2
        and callable(fields[0])
    ):
        entries = (fields,)
    elif isinstance(fields, (str, bytes)) or not isinstance(fields, Iterable):
        raise TypeError("fields must be a callable or an iterable")
    else:
        entries = tuple(fields)

    profiles = []
    metadata = []
    for entry in entries:
        if callable(entry):
            profile = entry
            polarization = (0.0, 0.0, 1.0)
        elif isinstance(entry, Sequence) and len(entry) == 2 and callable(entry[0]):
            profile, polarization = entry
        else:
            raise TypeError(
                "each field must be callable or a (callable, polarization) pair"
            )
        vector = np.asarray(polarization, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("field polarization must be a finite three-vector")
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("field polarization must be nonzero")
        vector = vector / norm
        if not np.allclose(vector[:2], 0.0, rtol=0.0, atol=1.0e-14):
            raise NotImplementedError(
                "the two-radial interface currently supports only z-polarized fields"
            )
        field_gauge = getattr(profile, "gauge", gauge)
        if field_gauge != gauge:
            raise ValueError(
                f"field {type(profile).__name__} uses gauge {field_gauge!r}, "
                f"but field_gauge is {gauge!r}"
            )
        signed = (
            profile
            if vector[2] > 0
            else _SignedAxialField(profile, -1.0, gauge)
        )
        profiles.append(signed)
        metadata.append(
            {
                "profile": _profile_metadata(profile),
                "polarization": vector.tolist(),
            }
        )
    return tuple(profiles), metadata


def _profile_metadata(profile) -> dict[str, Any]:
    values = {
        name: value
        for name, value in vars(profile).items()
        if not name.startswith("_")
    } if hasattr(profile, "__dict__") else {}
    serializable = {}
    for name, value in values.items():
        try:
            serializable[name] = _json_value(value)
        except TypeError:
            serializable[name] = repr(value)
    return {
        "module": type(profile).__module__,
        "type": type(profile).__name__,
        "parameters": serializable,
    }


def _angular_limits(l_max, m_max, lambda_max, mu_max):
    l_max = _nonnegative_integer(l_max, "l_max")
    m_max = l_max if m_max is None else _nonnegative_integer(m_max, "m_max")
    if m_max != l_max:
        raise ValueError("m_max must equal l_max")
    lambda_max = (
        2 * l_max
        if lambda_max is None
        else _nonnegative_integer(lambda_max, "lambda_max")
    )
    if lambda_max > 2 * l_max:
        raise ValueError("lambda_max must not exceed 2*l_max")
    mu_max = (
        lambda_max
        if mu_max is None
        else _nonnegative_integer(mu_max, "mu_max")
    )
    if mu_max > lambda_max:
        raise ValueError("mu_max must not exceed lambda_max")
    if mu_max > 2 * m_max:
        raise ValueError("mu_max must not exceed 2*m_max")
    return l_max, m_max, lambda_max, mu_max


def _validate_absorber_pair(mask, cap, coordinate: str) -> None:
    if mask is not None and cap is not None:
        raise ValueError(f"mask and cap cannot both be set for the {coordinate}")


def _field_gauge(value: str) -> DipoleGauge:
    if value not in ("length", "velocity"):
        raise ValueError("field_gauge must be 'length' or 'velocity'")
    return value


def _validate_velocity_gauge_initial_vector_potential(
    fields,
    *,
    gauge: DipoleGauge,
    initial_time: float,
) -> None:
    if gauge != "velocity" or not fields:
        return
    values = []
    for field in fields:
        value = np.asarray(field(initial_time))
        if value.shape != () or not np.issubdtype(value.dtype, np.number):
            raise TypeError(
                "velocity-gauge fields must return scalar numeric values"
            )
        value = complex(value.item())
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError("velocity-gauge fields must return finite values")
        values.append(value)
    vector_potential = sum(values, 0.0j)
    if abs(vector_potential) > 1.0e-14:
        raise ValueError(
            "velocity-gauge dynamics currently require A(initial_time)=0; "
            "a nonzero vector potential requires an explicit gauge "
            "transformation of the field-free ground state"
        )


def _validate_field_angular_basis(representation, fields) -> None:
    if not fields:
        return
    if isinstance(representation, CoupledTwoRadialSystem):
        basis = representation.angular_basis
        if basis.inversion_parity is not None:
            raise ValueError(
                "a z-polarized field requires both inversion parities in "
                "the coupled angular basis"
            )
        parities = {(l1 + l2) % 2 for l1, l2, _, _ in basis.states}
        if parities != {0, 1}:
            raise ValueError(
                "a z-polarized field requires both even and odd coupled "
                "angular channels"
            )
        return
    electronic_parities = {
        l_value % 2 for l_value, _ in representation.basis1.states
    }
    if electronic_parities != {0, 1}:
        raise ValueError(
            "a z-polarized field requires both even and odd electronic "
            "angular-momentum channels; increase l_max and do not impose an "
            "electronic coordinate parity"
        )


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
