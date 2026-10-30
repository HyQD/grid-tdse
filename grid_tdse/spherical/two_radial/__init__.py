"""Two-radial spherical-coordinate Hamiltonian foundation."""

from .angular import AngularBasisSpec, AngularPairBasis, CoulombMultipoleSpec
from .absorbing_boundaries import TwoRadialAbsorbingBoundaries
from .atomic import (
    TwoElectronAtomGroundState,
    TwoElectronAtomGroundStateResult,
    TwoElectronAtomParameters,
)
from .coulomb import (
    CoulombTerm,
    RadialCoulombMethod,
    RadialCoulombMultipoles,
    build_radial_coulomb_multipoles,
)
from .couplings import CoulombCouplings, build_coulomb_couplings
from .diatomic import (
    DiatomicIonGroundStateResult,
    DiatomicIonParameters,
    NonBornOppenheimerDiatomicIonGroundState,
)
from .dipole import (
    AxialCoordinateDipole,
    AxialDipoleCouplings,
    DipoleGauge,
    TwoRadialAxialDipoleInteraction,
)
from .distributed_2d_state import (
    DistributedTwoRadial2DState,
    TwoRadial2DRankLayout,
    TwoRadialDistributed2DLayout,
    gather_two_radial_2d_state,
    load_distributed_two_radial_2d_state,
    phase_align_two_radial_state,
    phase_aligned_two_radial_state_difference,
    redistribute_two_radial_2d_state,
    save_distributed_two_radial_2d_state,
    scatter_two_radial_2d_state,
)
from .distributed_2d_hamiltonian import (
    Distributed2DHamiltonianComponents,
    Distributed2DHamiltonianProfile,
    TwoRadialDistributed2DHamiltonian,
)
from .distributed_state import (
    DistributedTwoRadialState,
    TwoRadialDistributedLayout,
    TwoRadialRankLayout,
    gather_two_radial_state,
    load_distributed_two_radial_state,
    save_distributed_two_radial_state,
    scatter_two_radial_state,
)
from .distributed_hamiltonian import (
    DistributedHamiltonianComponents,
    DistributedHamiltonianProfile,
    DistributedReductionProfile,
    TwoRadialDistributedHamiltonian,
    TwoRadialDistributedReductions,
)
from .distributed_ground_state import (
    DistributedGroundStateResult,
    TwoRadialDistributedGroundStateSolver,
)
from .grids import FEMDVRGridSpec, RadialGrid, RadialSincDVRGridSpec
from .hamiltonian import TwoRadialHamiltonian
from .observables import AngularMomentumExpectations
from .preconditioner import (
    RealTimePreconditionerName,
    TwoRadialAdditiveSchwarzRealTimePreconditioner,
    TwoRadialDiagonalRealTimePreconditioner,
    TwoRadialElementBlockJacobiRealTimePreconditioner,
    TwoRadialOneBodyPreconditioner,
    TwoRadialRealTimePreconditioner,
    build_two_radial_real_time_preconditioner,
)
from .state_archive import (
    TwoRadialState,
    load_two_radial_state,
    save_two_radial_state,
    two_radial_compatibility_metadata,
)
from .system import (
    AngularRadialPotential,
    CoordinateSpec,
    RadialOperatorMode,
    RadialPotential,
    TwoRadialSystem,
    TwoRadialSystemSpec,
    angular_radial_potential,
    radial_potential,
)
from .tdse import (
    NonBornOppenheimerDiatomicIonTDSE,
    TwoElectronAtomTDSE,
)

__all__ = [
    "AngularBasisSpec",
    "AngularPairBasis",
    "AngularMomentumExpectations",
    "AngularRadialPotential",
    "AxialCoordinateDipole",
    "AxialDipoleCouplings",
    "CoordinateSpec",
    "CoulombCouplings",
    "CoulombMultipoleSpec",
    "CoulombTerm",
    "DiatomicIonGroundStateResult",
    "DiatomicIonParameters",
    "DipoleGauge",
    "Distributed2DHamiltonianComponents",
    "Distributed2DHamiltonianProfile",
    "DistributedGroundStateResult",
    "DistributedHamiltonianComponents",
    "DistributedHamiltonianProfile",
    "DistributedReductionProfile",
    "DistributedTwoRadial2DState",
    "DistributedTwoRadialState",
    "FEMDVRGridSpec",
    "NonBornOppenheimerDiatomicIonGroundState",
    "NonBornOppenheimerDiatomicIonTDSE",
    "RadialCoulombMultipoles",
    "RadialCoulombMethod",
    "RadialGrid",
    "RadialOperatorMode",
    "RadialPotential",
    "RadialSincDVRGridSpec",
    "RealTimePreconditionerName",
    "TwoRadialAdditiveSchwarzRealTimePreconditioner",
    "TwoRadialDiagonalRealTimePreconditioner",
    "TwoRadialElementBlockJacobiRealTimePreconditioner",
    "TwoRadialHamiltonian",
    "TwoRadialAxialDipoleInteraction",
    "TwoRadialAbsorbingBoundaries",
    "TwoRadial2DRankLayout",
    "TwoRadialDistributed2DLayout",
    "TwoRadialDistributed2DHamiltonian",
    "TwoRadialDistributedGroundStateSolver",
    "TwoRadialDistributedLayout",
    "TwoRadialDistributedHamiltonian",
    "TwoRadialDistributedReductions",
    "TwoRadialOneBodyPreconditioner",
    "TwoRadialRealTimePreconditioner",
    "TwoRadialRankLayout",
    "TwoRadialState",
    "TwoRadialSystem",
    "TwoRadialSystemSpec",
    "TwoElectronAtomGroundState",
    "TwoElectronAtomGroundStateResult",
    "TwoElectronAtomParameters",
    "TwoElectronAtomTDSE",
    "angular_radial_potential",
    "build_coulomb_couplings",
    "build_radial_coulomb_multipoles",
    "build_two_radial_real_time_preconditioner",
    "gather_two_radial_state",
    "gather_two_radial_2d_state",
    "load_distributed_two_radial_2d_state",
    "load_distributed_two_radial_state",
    "load_two_radial_state",
    "phase_align_two_radial_state",
    "phase_aligned_two_radial_state_difference",
    "radial_potential",
    "redistribute_two_radial_2d_state",
    "save_distributed_two_radial_2d_state",
    "save_two_radial_state",
    "save_distributed_two_radial_state",
    "scatter_two_radial_2d_state",
    "scatter_two_radial_state",
    "two_radial_compatibility_metadata",
]
