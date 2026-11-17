"""Prolate-spheroidal one-electron TDSE representations."""

from grid_tdse.systems import OneElectronDiatomicSystem

from .absorbing_boundaries import (
    CartesianRadiusCosineMask,
    ProlateSpheroidalAbsorbingBoundaries,
    ProlateSpheroidalCubicCAP,
)
from .dipole import (
    ProlateSpheroidalAxialLengthGaugeInteraction,
    ProlateSpheroidalAxialVelocityGaugeInteraction,
)
from .dynamics import ProlateSpheroidalFixedMTimeDependentHamiltonian
from .ground_state import (
    ProlateSpheroidalOneElectronEigenstates,
    diagonalize_fixed_m,
)
from .hamiltonian import ProlateSpheroidalOneElectronHamiltonian
from .linear_response import (
    AreaNormalizedSineSquareImpulse,
    AxialDeltaKick,
    PositiveFrequencySpectrum,
    ProlateSpheroidalDynamicsObservables,
    lorentzian_broadened_positive_frequency_spectrum,
    paired_kick_linear_response,
    windowed_positive_frequency_spectrum,
)
from .representation import ProlateSpheroidalRepresentation

__all__ = [
    "AreaNormalizedSineSquareImpulse",
    "AxialDeltaKick",
    "CartesianRadiusCosineMask",
    "OneElectronDiatomicSystem",
    "PositiveFrequencySpectrum",
    "ProlateSpheroidalAxialLengthGaugeInteraction",
    "ProlateSpheroidalAxialVelocityGaugeInteraction",
    "ProlateSpheroidalAbsorbingBoundaries",
    "ProlateSpheroidalCubicCAP",
    "ProlateSpheroidalDynamicsObservables",
    "ProlateSpheroidalFixedMTimeDependentHamiltonian",
    "ProlateSpheroidalOneElectronEigenstates",
    "ProlateSpheroidalOneElectronHamiltonian",
    "ProlateSpheroidalRepresentation",
    "diagonalize_fixed_m",
    "lorentzian_broadened_positive_frequency_spectrum",
    "paired_kick_linear_response",
    "windowed_positive_frequency_spectrum",
]
