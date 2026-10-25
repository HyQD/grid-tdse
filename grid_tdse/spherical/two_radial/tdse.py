"""High-level real-time TDSE interfaces for two-radial systems."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from numbers import Integral
from os import PathLike
from typing import Any, Literal

import numpy as np

from grid_tdse.real_time import RealTimePropagationResult, RealTimePropagator

from .absorbing_boundaries import TwoRadialAbsorbingBoundaries
from .angular import AngularBasisSpec, CoulombMultipoleSpec, Parity
from .atomic import _atomic_parameters
from .coulomb import CoulombTerm, RadialCoulombMethod
from .diatomic import (
    _diatomic_parameters,
    _pair_exchange_parity,
)
from .dipole import DipoleGauge, TwoRadialAxialDipoleInteraction
from .grids import FEMDVRGridSpec
from .preconditioner import TwoRadialRealTimePreconditioner
from .state_archive import TwoRadialState, two_radial_compatibility_metadata
from .system import (
    CoordinateSpec,
    TwoRadialSystemSpec,
    radial_potential,
)


IntegratorName = Literal["crank-nicolson"]
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

    _INTEGRATORS = {"crank-nicolson"}
    _INTEGRATOR_OPTIONS = {
        "solver",
        "rtol",
        "atol",
        "max_iterations",
        "restart",
        "preconditioner",
        "reality_tolerance",
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
        progress_interval: int | None,
        tile_size: int,
    ) -> None:
        if integrator not in self._INTEGRATORS:
            choices = ", ".join(sorted(self._INTEGRATORS))
            raise ValueError(f"integrator must be one of: {choices}")
        self.integrator = integrator
        options = {} if integrator_options is None else dict(integrator_options)
        unknown = set(options).difference(self._INTEGRATOR_OPTIONS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown {integrator} integrator options: {names}")
        self.integrator_options = options.copy()

        self.initial_state = self._prepare_initial_state(initial_state)
        self.field_gauge = _field_gauge(field_gauge)
        self.fields, field_metadata = _axial_fields(fields, self.field_gauge)
        self.field_free_hamiltonian = self.representation.hamiltonian(
            tile_size=tile_size
        )
        self.dipole_interaction = (
            self._build_dipole_interaction(self.fields) if self.fields else None
        )

        self.propagator = RealTimePropagator(
            initial_time=initial_time,
            final_time=final_time,
            time_step=time_step,
            linear_solver=options.get("solver", "bicgstab"),
            linear_rtol=options.get("rtol", 0.0),
            linear_atol=options.get("atol", 1.0e-10),
            linear_max_iterations=options.get("max_iterations"),
            gmres_restart=options.get("restart"),
            reality_tolerance=options.get("reality_tolerance", 1.0e-10),
            progress_interval=progress_interval,
        )
        use_preconditioner = options.get("preconditioner", True)
        if not isinstance(use_preconditioner, (bool, np.bool_)):
            raise TypeError("integrator option 'preconditioner' must be boolean")
        self.preconditioner = (
            TwoRadialRealTimePreconditioner(
                self.representation, self.propagator.time_step
            )
            if use_preconditioner
            else None
        )

        self.progress_interval = progress_interval
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_overwrite = bool(checkpoint_overwrite)
        self.checkpoint_metadata = (
            None if checkpoint_metadata is None else dict(checkpoint_metadata)
        )
        self.compatibility_metadata = two_radial_compatibility_metadata(
            self.representation,
            physical_state=self._physical_state_metadata(),
        )
        self.dynamics_metadata = {
            "interaction": "electric_dipole" if self.fields else "field_free",
            "polarization_axis": "z" if self.fields else None,
            "field_gauge": self.field_gauge,
            "fields": field_metadata,
            "integrator": self.integrator,
            "integrator_options": _json_value(self.integrator_options),
            "absorbing_boundaries": self.absorbing_boundaries.metadata,
        }
        self.samplers = self._prepare_samplers(observables, samplers)

    def run(self, *, progress: bool = True) -> RealTimePropagationResult:
        """Propagate a copy of the configured initial state."""

        if not isinstance(progress, (bool, np.bool_)):
            raise TypeError("progress must be boolean")
        configured_interval = self.propagator.progress_interval
        self.propagator.progress_interval = (
            self.progress_interval if progress else None
        )
        try:
            return self.propagator.propagate(
                self.initial_state,
                self.hamiltonian,
                self.representation.inner_product,
                preconditioner=self.preconditioner,
                samplers=self.samplers,
                post_step=(
                    self.absorbing_boundaries.apply_mask
                    if self.absorbing_boundaries.has_mask
                    else None
                ),
                checkpoint_path=self.checkpoint_path,
                checkpoint_interval=self.checkpoint_interval,
                compatibility_metadata=self.compatibility_metadata,
                dynamics_metadata=self.dynamics_metadata,
                checkpoint_metadata=self.checkpoint_metadata,
                checkpoint_overwrite=self.checkpoint_overwrite,
            )
        finally:
            self.propagator.progress_interval = configured_interval

    __call__ = run

    def hamiltonian(self, state: np.ndarray, time: float) -> np.ndarray:
        """Apply the field-free Hamiltonian and all configured axial fields."""

        result = self.field_free_hamiltonian(state)
        if self.dipole_interaction is not None:
            result += self.dipole_interaction(state, time)
        if self.absorbing_boundaries.has_cap:
            result += self.absorbing_boundaries.cap_action(state)
        return result

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

    def _build_dipole_interaction(self, fields, gauge=None):
        raise NotImplementedError

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
        progress_interval: int | None = 10,
        tile_size: int = 2048,
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
        self.representation = self.specification.build()
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
            progress_interval=progress_interval,
            tile_size=tile_size,
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
        )

    def _physical_state_metadata(self):
        return {
            "system_family": "two_electron_atom",
            "system": self.parameters.name,
            "parameters": _json_value(asdict(self.parameters)),
        }


class NonBornOppenheimerDiatomicIonTDSE(_TwoRadialTDSE):
    """Propagate a non-Born--Oppenheimer three-body diatomic ion."""

    def __init__(
        self,
        system: str,
        *,
        electronic_grid: FEMDVRGridSpec,
        internuclear_grid: FEMDVRGridSpec,
        initial_state: InitialState,
        pair_exchange_parity: Parity | None,
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
        checkpoint_path: str | PathLike[str] | None = None,
        checkpoint_interval: int | None = None,
        checkpoint_overwrite: bool = False,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        progress_interval: int | None = 10,
        tile_size: int = 2048,
    ) -> None:
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
        self.representation = self.specification.build()
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
            progress_interval=progress_interval,
            tile_size=tile_size,
        )

    @property
    def system_name(self) -> str:
        return self.parameters.name

    def _build_dipole_interaction(self, fields, gauge=None):
        return TwoRadialAxialDipoleInteraction.for_diatomic_ion(
            self.representation,
            self.parameters,
            gauge=self.field_gauge if gauge is None else gauge,
            fields=fields,
        )

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


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
