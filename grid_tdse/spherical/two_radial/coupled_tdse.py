"""High-level serial coupled-angular dynamics for non-BO diatomic ions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .absorbing_boundaries import TwoRadialAbsorbingBoundaries
from .angular import AngularBasisSpec, CoulombMultipoleSpec, Parity
from .coulomb import CoulombTerm, RadialCoulombMethod
from .coupled_angular import CoupledAngularBasisSpec
from .coupled_dipole import CoupledTwoRadialAxialDipoleInteraction
from .coupled_state_archive import (
    CoupledTwoRadialState,
    coupled_two_radial_compatibility_metadata,
)
from .diatomic import _diatomic_parameters, _pair_exchange_parity
from .dipole import DipoleGauge, DipoleKernelMode
from .dynamics_initialization import (
    DiatomicDynamicsInitialStatePolicy,
    prepare_coupled_non_bo_diatomic_dynamics_initial_state,
)
from .dynamics_sampling import (
    DiatomicIonDynamicsObservableEvaluator,
    DiatomicIonDynamicsSamplingPolicy,
    DiatomicIonDynamicsTimeSeries,
)
from .grids import FEMDVRGridSpec
from .system import (
    CoordinateSpec,
    CoulombKernelMode,
    RadialKernelMode,
    RadialOperatorMode,
    TwoRadialSystemSpec,
    radial_potential,
)
from .tdse import (
    IntegratorName,
    NonBornOppenheimerDiatomicIonTDSE,
    Sampler,
    _json_value,
    _nonnegative_integer,
    _validate_absorber_pair,
)


CoupledInitialState = np.ndarray | CoupledTwoRadialState


class CoupledNonBornOppenheimerDiatomicIonTDSE(
    NonBornOppenheimerDiatomicIonTDSE
):
    """Propagate a serial bipolar-harmonic diatomic-ion wavefunction."""

    def __init__(
        self,
        system: str,
        *,
        wf_representation: Literal["coupled"] = "coupled",
        electronic_grid: FEMDVRGridSpec,
        internuclear_grid: FEMDVRGridSpec,
        initial_state: CoupledInitialState | None = None,
        pair_exchange_parity: Parity | None,
        inversion_parity: Parity | None = None,
        initial_state_policy: DiatomicDynamicsInitialStatePolicy = "require",
        ground_state_path: str | PathLike[str] | None = None,
        ground_state_inversion_parity: Parity = 1,
        ground_state_L_max: int = 0,
        ground_state_options: Mapping[str, Any] | None = None,
        allow_unconverged_ground_state: bool = False,
        fields=(),
        field_gauge: DipoleGauge = "length",
        l1_max: int = 2,
        l2_max: int = 2,
        L_max: int | None = None,
        M_tot: int = 0,
        lambda_max: int | None = None,
        codata_set: str = "2022",
        radial_coulomb_method: RadialCoulombMethod = "poisson",
        integrator: IntegratorName = "sia",
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
        radial_operator_mode: RadialOperatorMode = "element-local",
        tile_size: int = 2048,
        coulomb_kernel: CoulombKernelMode = "reference",
        radial_kernel: RadialKernelMode = "reference",
        dipole_kernel: DipoleKernelMode = "reference",
        coupling_tolerance: float = 1.0e-12,
        max_coupling_pairs: int = 20_000_000,
    ) -> None:
        if wf_representation != "coupled":
            raise ValueError(
                "the coupled constructor requires wf_representation='coupled'"
            )
        if integrator not in {"sia", "cf4-sia"}:
            raise ValueError(
                "coupled dynamics currently support integrator='sia' or "
                "'cf4-sia'"
            )
        unsupported_kernels = {
            "coulomb_kernel": coulomb_kernel,
            "radial_kernel": radial_kernel,
            "dipole_kernel": dipole_kernel,
        }
        nondefault = {
            name: value
            for name, value in unsupported_kernels.items()
            if value != "reference"
        }
        if nondefault:
            names = ", ".join(sorted(nondefault))
            raise ValueError(
                f"the coupled representation does not accept uncoupled-only "
                f"kernel options: {names}"
            )
        _validate_absorber_pair(mask_r, cap_r, "electronic coordinate r")
        _validate_absorber_pair(mask_R, cap_R, "internuclear coordinate R")
        if not isinstance(electronic_grid, FEMDVRGridSpec):
            raise TypeError("electronic_grid must be a FEMDVRGridSpec")
        if not isinstance(internuclear_grid, FEMDVRGridSpec):
            raise TypeError("internuclear_grid must be a FEMDVRGridSpec")

        self.wf_representation = "coupled"
        self.parameters = _diatomic_parameters(system, codata_set)
        self.pair_exchange_parity = _pair_exchange_parity(
            pair_exchange_parity,
            self.parameters,
        )
        self.inversion_parity = inversion_parity
        self.electronic_grid_spec = electronic_grid
        self.internuclear_grid_spec = internuclear_grid
        self.l1_max = _nonnegative_integer(l1_max, "l1_max")
        self.l2_max = _nonnegative_integer(l2_max, "l2_max")
        maximum_L = self.l1_max + self.l2_max
        self.L_max = (
            maximum_L
            if L_max is None
            else _nonnegative_integer(L_max, "L_max")
        )
        if self.L_max > maximum_L:
            raise ValueError("L_max must not exceed l1_max + l2_max")
        if isinstance(M_tot, (bool, np.bool_)) or not isinstance(M_tot, Integral):
            raise TypeError("M_tot must be an integer")
        self.M_tot = int(M_tot)
        complete_lambda = 2 * min(self.l1_max, self.l2_max)
        self.lambda_max = (
            complete_lambda
            if lambda_max is None
            else _nonnegative_integer(lambda_max, "lambda_max")
        )
        if self.lambda_max > complete_lambda:
            raise ValueError(
                "lambda_max must not exceed 2*min(l1_max, l2_max)"
            )
        self.radial_coulomb_method = radial_coulomb_method
        self.coupling_tolerance = float(coupling_tolerance)
        if (
            not np.isfinite(self.coupling_tolerance)
            or self.coupling_tolerance < 0.0
        ):
            raise ValueError("coupling_tolerance must be finite and non-negative")
        self.max_coupling_pairs = _nonnegative_integer(
            max_coupling_pairs,
            "max_coupling_pairs",
        )
        if self.max_coupling_pairs == 0:
            raise ValueError("max_coupling_pairs must be positive")
        if dynamics_sampling_policy is None:
            dynamics_sampling_policy = DiatomicIonDynamicsSamplingPolicy()
        if not isinstance(
            dynamics_sampling_policy,
            DiatomicIonDynamicsSamplingPolicy,
        ):
            raise TypeError(
                "dynamics_sampling_policy must be a "
                "DiatomicIonDynamicsSamplingPolicy or None"
            )
        self.dynamics_sampling_policy = dynamics_sampling_policy

        p = self.parameters
        angular_basis = CoupledAngularBasisSpec.for_diatomic_ion(
            self.l1_max,
            self.l2_max,
            self.L_max,
            has_identical_pair=p.has_identical_pair,
            M_tot=self.M_tot,
            inversion_parity=inversion_parity,
            pair_exchange_parity=self.pair_exchange_parity,
        )
        self.specification = TwoRadialSystemSpec(
            coordinate1=CoordinateSpec(
                grid=electronic_grid,
                basis=AngularBasisSpec(self.l1_max),
                kinetic_prefactor=-0.5 / p.electronic_reduced_mass,
            ),
            coordinate2=CoordinateSpec(
                grid=internuclear_grid,
                basis=AngularBasisSpec(self.l2_max),
                kinetic_prefactor=-0.5 / p.internuclear_reduced_mass,
                one_body_terms=(
                    radial_potential(
                        lambda radius: (
                            p.reference_charge * p.partner_charge / radius
                        )
                    ),
                ),
            ),
            multipoles=CoulombMultipoleSpec(
                self.lambda_max,
                self.lambda_max,
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
            M_tot=self.M_tot,
            radial_coulomb_method=radial_coulomb_method,
        )
        self.representation = self.specification.build_coupled(angular_basis)

        self.initialization = None
        if initial_state is None:
            if ground_state_path is None:
                raise ValueError(
                    "ground_state_path is required when initial_state is None"
                )
            self.initialization = (
                prepare_coupled_non_bo_diatomic_dynamics_initial_state(
                    self.representation,
                    system=p.name,
                    electronic_grid=electronic_grid,
                    internuclear_grid=internuclear_grid,
                    pair_exchange_parity=self.pair_exchange_parity,
                    archive_path=ground_state_path,
                    policy=initial_state_policy,
                    inversion_parity=ground_state_inversion_parity,
                    ground_state_L_max=ground_state_L_max,
                    codata_set=codata_set,
                    radial_coulomb_method=radial_coulomb_method,
                    ground_state_options=ground_state_options,
                    allow_unconverged=allow_unconverged_ground_state,
                )
            )
            initial_state = self.initialization.state
        else:
            if initial_state_policy != "require":
                raise ValueError(
                    "initial_state_policy must be 'require' when an explicit "
                    "initial_state is provided"
                )
            if ground_state_path is not None or ground_state_options is not None:
                raise ValueError(
                    "ground-state archive options cannot be combined with an "
                    "explicit initial_state"
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
        self.dynamics_metadata.update(
            {
                "wf_representation": "coupled",
                "angular_basis": self.representation.angular_basis.to_metadata(),
                "initial_state": (
                    {"source": "provided"}
                    if self.initialization is None
                    else self.initialization.metadata
                ),
                "sampling_policy": asdict(self.dynamics_sampling_policy),
            }
        )
        self.dynamics_observable_evaluator = (
            DiatomicIonDynamicsObservableEvaluator(
                self.representation,
                self.parameters,
                self.field_free_hamiltonian,
                gauge=self.field_gauge,
                fields=self.fields,
                velocity_action=(
                    self._velocity_action()
                    if self.field_gauge == "velocity"
                    else None
                ),
                radial_operator_mode=self.radial_operator_mode,
            )
        )
        self.dynamics_time_series = DiatomicIonDynamicsTimeSeries(())
        self._dynamics_sampler = None

    def _prepare_initial_state(self, initial_state):
        if isinstance(initial_state, CoupledTwoRadialState):
            self._validate_archived_physical_state(initial_state)
            values = initial_state.embed_into(self.representation)
        else:
            values = self.representation.validate_state(initial_state)
        if not np.isfinite(values).all():
            raise ValueError("initial_state must contain only finite values")
        norm_squared = self.representation.inner_product(values, values).real
        if norm_squared <= 0.0:
            raise ValueError("initial_state must have positive norm")
        return np.ascontiguousarray(
            values,
            dtype=np.result_type(values.dtype, complex),
        )

    def _build_field_free_hamiltonian(self, **options):
        return self.representation.hamiltonian(
            radial_operator_mode=options["radial_operator_mode"],
            coupling_tolerance=self.coupling_tolerance,
            max_coupling_pairs=self.max_coupling_pairs,
        )

    def _build_dipole_interaction(self, fields, gauge=None):
        return CoupledTwoRadialAxialDipoleInteraction.for_diatomic_ion(
            self.representation,
            self.parameters,
            gauge=self.field_gauge if gauge is None else gauge,
            fields=fields,
            radial_operator_mode=self.radial_operator_mode,
            coupling_tolerance=self.coupling_tolerance,
            max_coupling_pairs=self.max_coupling_pairs,
        )

    def _velocity_action(self):
        if self.dipole_interaction is not None:
            return self.dipole_interaction.spatial_action
        return self._build_dipole_interaction(
            (),
            gauge="velocity",
        ).spatial_action

    def _build_compatibility_metadata(self):
        return coupled_two_radial_compatibility_metadata(
            self.representation,
            physical_state=self._physical_state_metadata(),
        )

    def _physical_state_metadata(self):
        return {
            "system_family": "non_bo_diatomic_ion",
            "system": self.parameters.name,
            "parameters": _json_value(asdict(self.parameters)),
        }
