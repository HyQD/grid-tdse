"""High-level real-time TDSE interface for one-radial diatomics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np

from grid_tdse.real_time import (
    RealTimeCheckpoint,
    RealTimePropagationResult,
    RealTimePropagator,
    load_real_time_checkpoint,
    real_time_state_data_id,
)
from grid_tdse.systems import OneElectronDiatomicSystem

from .absorbing_boundaries import SphericalOneRadialAbsorbingBoundaries
from .dynamics import (
    SphericalOneRadialFixedMTimeDependentHamiltonian,
    SphericalOneRadialFullAngularTimeDependentHamiltonian,
)
from .hamiltonian import (
    NuclearPotentialMethod,
    OneElectronDiatomicHamiltonian,
)
from .observables import (
    SphericalOneRadialDynamicsObservables,
    SphericalOneRadialFullAngularDynamicsObservables,
    SphericalOneRadialRadialProbabilitySeries,
)
from .preconditioner import (
    build_spherical_one_radial_real_time_preconditioner,
)


IntegratorName = Literal["crank-nicolson", "sil", "sia"]
FieldGauge = Literal["length", "velocity"]
AngularRepresentation = Literal["fixed_m", "full"]
Sampler = Callable[[np.ndarray, float], np.ndarray | complex | float]


class OneElectronDiatomicTDSE:
    """Prepare and propagate a clamped one-electron diatomic state.

    The calculation owns the physical system, spherical FEM-DVR
    representation, time-dependent Hamiltonian, absorber, integrator,
    observables, and checkpoint policy. The supplied initial state is copied
    but never normalized.
    """

    _INTEGRATOR_OPTIONS = {
        "crank-nicolson": {
            "solver",
            "rtol",
            "atol",
            "max_iterations",
            "restart",
            "preconditioner",
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
    }

    def __init__(
        self,
        system: OneElectronDiatomicSystem,
        *,
        nodes,
        n_points,
        l_max: int,
        initial_state,
        angular_representation: AngularRepresentation = "fixed_m",
        m: int = 0,
        m_max: int | None = None,
        nuclear_L_max: int | None = None,
        nuclear_M_max: int | None = None,
        nuclear_method: NuclearPotentialMethod = "analytic",
        fields=(),
        field_gauge: FieldGauge = "length",
        include_nuclear_scalar: bool = False,
        origin=(0.0, 0.0, 0.0),
        origin_z: float | None = None,
        integrator: IntegratorName = "sia",
        integrator_options: Mapping[str, Any] | None = None,
        initial_time: float = 0.0,
        final_time: float,
        time_step: float,
        mask=None,
        cap=None,
        reference_states: Mapping[str, np.ndarray] | None = None,
        radial_probability_interval: int | None = 1,
        samplers: Mapping[str, Sampler] | None = None,
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        checkpoint_overwrite: bool = False,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        progress_interval: int | None = 10,
        progress_bar: bool = False,
    ) -> None:
        if not isinstance(system, OneElectronDiatomicSystem):
            raise TypeError("system must be a OneElectronDiatomicSystem")
        self.system = system
        self.angular_representation = _angular_representation(
            angular_representation
        )
        requested_m = _integer(m, "m")
        if self.angular_representation == "full" and requested_m != 0:
            raise ValueError("m is not selectable for the full representation")
        if (
            self.angular_representation == "fixed_m"
            and not system.is_z_axis_aligned
        ):
            raise ValueError(
                "an off-axis geometry requires angular_representation='full'"
            )
        self.m = requested_m if self.angular_representation == "fixed_m" else None
        if m_max is None:
            m_max = (
                abs(requested_m)
                if self.angular_representation == "fixed_m"
                else l_max
            )
        self.origin = _resolved_origin(origin, origin_z)
        if self.angular_representation == "fixed_m" and np.any(
            self.origin[:2] != 0.0
        ):
            raise ValueError(
                "fixed_m dynamics require an origin on the Cartesian z axis"
            )
        self.hamiltonian = OneElectronDiatomicHamiltonian(
            system,
            nodes=nodes,
            n_points=n_points,
            l_max=l_max,
            m_max=m_max,
            nuclear_L_max=nuclear_L_max,
            nuclear_M_max=nuclear_M_max,
            nuclear_method=nuclear_method,
        )
        if self.angular_representation == "fixed_m":
            self.m = self.hamiltonian.validate_m(self.m)
        self.initial_state = self._initial_state(initial_state)
        self.fields = _field_profiles(fields, field_gauge)
        self.field_gauge = _field_gauge(field_gauge)
        self.absorbing_boundaries = SphericalOneRadialAbsorbingBoundaries(
            self.hamiltonian,
            mask=mask,
            cap=cap,
        )
        if self.angular_representation == "fixed_m":
            self.dynamics = SphericalOneRadialFixedMTimeDependentHamiltonian(
                self.hamiltonian,
                m=self.m,
                fields=self.fields,
                field_gauge=self.field_gauge,
                include_nuclear_scalar=include_nuclear_scalar,
                origin_z=float(self.origin[2]),
                absorbing_boundaries=self.absorbing_boundaries,
            )
        else:
            self.dynamics = (
                SphericalOneRadialFullAngularTimeDependentHamiltonian(
                    self.hamiltonian,
                    fields=self.fields,
                    field_gauge=self.field_gauge,
                    include_nuclear_scalar=include_nuclear_scalar,
                    origin=self.origin,
                    absorbing_boundaries=self.absorbing_boundaries,
                )
            )
        self.integrator = _integrator(integrator)
        self.integrator_options = self._integrator_options(
            self.integrator, integrator_options
        )
        self.propagator = self._build_propagator(
            initial_time=initial_time,
            final_time=final_time,
            time_step=time_step,
            progress_interval=progress_interval,
            progress_bar=progress_bar,
        )
        self._validate_initial_vector_potential()
        self.absorbing_boundaries.validate_integrator(self.integrator)
        self.preconditioner = self._build_preconditioner()
        observables_type = (
            SphericalOneRadialDynamicsObservables
            if self.angular_representation == "fixed_m"
            else SphericalOneRadialFullAngularDynamicsObservables
        )
        self.observables = observables_type(
            self.dynamics,
            self.initial_state,
            reference_states=reference_states,
            radial_probability_interval=radial_probability_interval,
            reality_tolerance=self.integrator_options.get(
                "reality_tolerance", 1.0e-10
            ),
        )
        self.samplers = self._samplers(samplers)

        self.checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path)
        )
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_overwrite = _boolean(
            checkpoint_overwrite, "checkpoint_overwrite"
        )
        self.checkpoint_metadata = (
            None if checkpoint_metadata is None else dict(checkpoint_metadata)
        )
        self.progress_interval = progress_interval
        self.progress_bar = _boolean(progress_bar, "progress_bar")
        self.compatibility_metadata = self._compatibility_metadata()
        self.dynamics_metadata = self._dynamics_metadata()

    @property
    def radial_probability_series(
        self,
    ) -> SphericalOneRadialRadialProbabilitySeries:
        """Return the radial distributions collected by the latest run."""

        return self.observables.radial_probability_series

    def run(self, *, progress: bool = True) -> RealTimePropagationResult:
        """Propagate a copy of the configured initial state."""

        return self._execute(None, mode="new", progress=progress)

    propagate = run
    __call__ = run

    def resume(
        self,
        checkpoint: RealTimeCheckpoint | str | PathLike[str],
        *,
        progress: bool = True,
    ) -> RealTimePropagationResult:
        """Exactly resume an interrupted propagation segment."""

        return self._execute(checkpoint, mode="resume", progress=progress)

    def continue_from(
        self,
        checkpoint: RealTimeCheckpoint | str | PathLike[str],
        *,
        progress: bool = True,
    ) -> RealTimePropagationResult:
        """Start a compatible propagation stage from a checkpoint."""

        return self._execute(checkpoint, mode="continue", progress=progress)

    def _execute(self, checkpoint, *, mode: str, progress: bool):
        progress = _boolean(progress, "progress")
        if mode not in {"new", "resume", "continue"}:
            raise ValueError("invalid execution mode")
        stored = None
        if checkpoint is not None:
            stored = (
                checkpoint
                if isinstance(checkpoint, RealTimeCheckpoint)
                else load_real_time_checkpoint(checkpoint)
            )
        self.observables.reset_radial_probability_samples()
        configured_interval = self.propagator.progress_interval
        configured_bar = self.propagator.progress_bar
        if not progress:
            self.propagator.progress_interval = None
            self.propagator.progress_bar = False
        try:
            arguments = self._propagation_arguments()
            if mode == "new":
                arguments.pop("post_step")
                return self.dynamics.propagate(
                    self.propagator,
                    self.initial_state,
                    **arguments,
                )
            method = (
                self.propagator.resume
                if mode == "resume"
                else self.propagator.continue_from
            )
            return method(
                stored,
                self.dynamics,
                self.dynamics.inner_product,
                **arguments,
            )
        finally:
            self.propagator.progress_interval = configured_interval
            self.propagator.progress_bar = configured_bar

    def _propagation_arguments(self) -> dict[str, Any]:
        return {
            "inner_products": (
                None
                if self.integrator == "crank-nicolson"
                else self.dynamics.inner_products
            ),
            "preconditioner": self.preconditioner,
            "samplers": self.samplers,
            "state_observer": self.observables.state_observer,
            "post_step": (
                self._apply_mask
                if self.absorbing_boundaries.has_mask
                else None
            ),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_interval": self.checkpoint_interval,
            "compatibility_metadata": self.compatibility_metadata,
            "dynamics_metadata": self.dynamics_metadata,
            "checkpoint_metadata": self.checkpoint_metadata,
            "checkpoint_overwrite": self.checkpoint_overwrite,
        }

    def _apply_mask(self, state, time):
        if self.angular_representation == "fixed_m":
            return self.absorbing_boundaries.apply_mask(
                state,
                time,
                self.m,
            )
        return self.absorbing_boundaries.apply_mask_full(state, time)

    def _initial_state(self, initial_state) -> np.ndarray:
        values = (
            self.hamiltonian.validate_fixed_m_state(initial_state, self.m)
            if self.angular_representation == "fixed_m"
            else self.hamiltonian.validate_state(initial_state)
        )
        if values.ndim != 2:
            raise ValueError("initial_state must contain exactly one state")
        norm = complex(self.hamiltonian.inner_product(values, values))
        tolerance = 1.0e-12 * max(1.0, abs(norm.real))
        if abs(norm.imag) > tolerance or norm.real <= 0.0:
            raise ValueError("initial_state must have positive finite norm")
        return np.array(values, dtype=np.result_type(values, complex), copy=True)

    def _integrator_options(self, integrator, options):
        supplied = {} if options is None else dict(options)
        unknown = set(supplied).difference(
            self._INTEGRATOR_OPTIONS[integrator]
        )
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"unknown {integrator} integrator options: {names}"
            )
        return supplied

    def _build_propagator(self, **time_options) -> RealTimePropagator:
        options = self.integrator_options
        crank_nicolson = self.integrator == "crank-nicolson"
        return RealTimePropagator(
            **time_options,
            integrator=self.integrator,
            linear_solver=options.get("solver", "bicgstab"),
            linear_rtol=options.get("rtol", 0.0),
            linear_atol=options.get("atol", 1.0e-10),
            linear_max_iterations=options.get("max_iterations"),
            gmres_restart=options.get("restart"),
            max_krylov_dimension=options.get("max_krylov_dimension", 20),
            krylov_rtol=(
                1.0e-10 if crank_nicolson else options.get("rtol", 1.0e-10)
            ),
            krylov_atol=(
                0.0 if crank_nicolson else options.get("atol", 0.0)
            ),
            max_krylov_refinements=options.get("max_refinements", 8),
            krylov_reorthogonalize=options.get("reorthogonalize", True),
            krylov_breakdown_tolerance=options.get(
                "breakdown_tolerance", 1.0e-14
            ),
            reality_tolerance=options.get("reality_tolerance", 1.0e-10),
        )

    def _build_preconditioner(self):
        if self.integrator != "crank-nicolson":
            return None
        return build_spherical_one_radial_real_time_preconditioner(
            self.hamiltonian,
            m=self.m,
            time_step=self.propagator.time_step,
            preconditioner=self.integrator_options.get(
                "preconditioner", True
            ),
            angular_representation=self.angular_representation,
        )

    def _validate_initial_vector_potential(self) -> None:
        if self.field_gauge != "velocity" or not self.fields:
            return
        value = self.dynamics.dipole_interaction.field_amplitude(
            self.propagator.initial_time
        )
        if abs(value) > 1.0e-14:
            raise ValueError(
                "velocity-gauge dynamics require A(initial_time)=0; a "
                "nonzero vector potential requires an explicit gauge "
                "transformation of the initial state"
            )

    def _samplers(self, custom):
        result = dict(self.observables.samplers)
        if custom is None:
            return result
        if not isinstance(custom, Mapping):
            raise TypeError("samplers must be a mapping or None")
        overlap = set(custom).intersection(result)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"custom sampler names are reserved: {names}")
        if not all(
            isinstance(name, str) and name and callable(sampler)
            for name, sampler in custom.items()
        ):
            raise TypeError("samplers must map nonempty strings to callables")
        result.update(custom)
        return result

    def _dynamics_metadata(self) -> dict[str, Any]:
        preconditioner = (
            None
            if self.preconditioner is None
            else self.preconditioner.metadata
        )
        return {
            "angular_representation": self.angular_representation,
            "origin": self.origin.tolist(),
            "interaction": "electric_dipole" if self.fields else "field_free",
            "polarization_axis": "z" if self.fields else None,
            "field_gauge": self.field_gauge,
            "fields": [_profile_metadata(field) for field in self.fields],
            "integrator": self.integrator,
            "integrator_options": _json_value(self.integrator_options),
            "preconditioner": preconditioner,
            "absorbing_boundaries": self.absorbing_boundaries.metadata,
            "observables": self.observables.metadata,
            "initial_state": {
                "source": "provided",
                "state_data_id": real_time_state_data_id(self.initial_state),
                "norm": self.observables.initial_norm,
            },
            "time_dependent_hamiltonian": self.dynamics.archive_metadata(),
        }

    def _compatibility_metadata(self) -> dict[str, Any]:
        metadata = self.hamiltonian.archive_metadata()
        metadata.update(
            {
                "angular_representation": self.angular_representation,
                "origin": self.origin.tolist(),
                "state_shape": list(self.initial_state.shape),
                "channel_lm": (
                    None
                    if self.angular_representation == "fixed_m"
                    else [
                        list(channel)
                        for channel in self.hamiltonian.orbital_lm
                    ]
                ),
                "m": self.m,
            }
        )
        return metadata


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _integrator(value: str) -> IntegratorName:
    if value not in {"crank-nicolson", "sil", "sia"}:
        raise ValueError("integrator must be 'crank-nicolson', 'sil', or 'sia'")
    return value


def _angular_representation(value: str) -> AngularRepresentation:
    if value not in {"fixed_m", "full"}:
        raise ValueError(
            "angular_representation must be 'fixed_m' or 'full'"
        )
    return value


def _resolved_origin(origin, origin_z) -> np.ndarray:
    raw = np.asarray(origin)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise TypeError("origin must contain real numbers")
    result = np.array(raw, dtype=float, copy=True)
    if result.shape != (3,):
        raise ValueError("origin must have shape (3,)")
    if not np.all(np.isfinite(result)):
        raise ValueError("origin must contain finite values")
    if origin_z is not None:
        if isinstance(origin_z, (bool, np.bool_)) or not isinstance(
            origin_z,
            Real,
        ):
            raise TypeError("origin_z must be a real scalar or None")
        alias = float(origin_z)
        if not np.isfinite(alias):
            raise ValueError("origin_z must be finite")
        if result[2] != 0.0 and not np.isclose(
            result[2],
            alias,
            rtol=0.0,
            atol=16.0 * np.finfo(float).eps,
        ):
            raise ValueError("origin and origin_z specify different z origins")
        result[2] = alias
    result.setflags(write=False)
    return result


def _field_gauge(value: str) -> FieldGauge:
    if value not in {"length", "velocity"}:
        raise ValueError("field_gauge must be 'length' or 'velocity'")
    return value


def _field_profiles(fields, gauge: str) -> tuple[Callable, ...]:
    gauge = _field_gauge(gauge)
    if fields is None:
        profiles = ()
    elif callable(fields):
        profiles = (fields,)
    elif isinstance(fields, (str, bytes)) or not isinstance(fields, Iterable):
        raise TypeError("fields must be a callable or an iterable")
    else:
        profiles = tuple(fields)
    if not all(callable(profile) for profile in profiles):
        raise TypeError("every field must be callable")
    for profile in profiles:
        profile_gauge = getattr(profile, "gauge", gauge)
        if profile_gauge != gauge:
            raise ValueError(
                f"field {type(profile).__name__} uses gauge "
                f"{profile_gauge!r}, but field_gauge is {gauge!r}"
            )
    return profiles


def _profile_metadata(profile) -> dict[str, Any]:
    parameters = {}
    if hasattr(profile, "__dict__"):
        for name, value in vars(profile).items():
            if name.startswith("_"):
                continue
            try:
                parameters[name] = _json_value(value)
            except TypeError:
                parameters[name] = repr(value)
    return {
        "module": type(profile).__module__,
        "type": type(profile).__name__,
        "parameters": parameters,
    }


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _boolean(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)
