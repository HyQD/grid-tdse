"""Becke-weighted atom-centered spherical representations."""

from .assembly import (
    BeckeAssemblyDiagnostics,
    BeckeAssemblyParameters,
    BeckeDenseMatrices,
    MatrixAssemblyDiagnostics,
    assemble_becke_matrices,
)
from .basis import (
    BeckeBasisEvaluation,
    BeckeBasisFunction,
    BeckeSphericalBasis,
    fixed_m_channels,
    full_lm_channels,
)
from .geometry import CenterGeometry, classify_centers
from .ground_state import (
    GeneralizedEigenDiagnostics,
    GeneralizedEigenResult,
    solve_becke_eigenproblem,
    solve_generalized_hermitian,
)
from .h2plus_validation import (
    H2PlusBeckeResult,
    H2PlusGroundStateValidation,
    H2PlusMatrices,
    H2PlusMidpointReference,
    validate_h2plus_ground_state,
)
from .kinetic_validation import (
    KineticElementComparison,
    KineticIntegrationSettings,
    TwoCenterKineticMatrix,
    TwoCenterKineticValidation,
    validate_two_center_kinetic,
)
from .metric import CanonicalOrthogonalization, canonical_orthogonalization
from .partition import BeckePartition, BeckePartitionEvaluation
from .quadrature import BeckeMulticenterQuadrature, BeckeQuadratureShell
from .specification import FEMDVRQuadratureSpec
from .validation import (
    MatrixRelativeErrors,
    MidpointL0Validation,
    MidpointMatrixSet,
    OverlapRankDecision,
    TwoCenterOverlapStudy,
    TwoCenterOverlapValidation,
    run_two_center_overlap_study,
    validate_midpoint_l0,
    validate_two_center_overlap,
)

__all__ = [
    "BeckeAssemblyDiagnostics",
    "BeckeAssemblyParameters",
    "BeckeBasisEvaluation",
    "BeckeBasisFunction",
    "BeckeDenseMatrices",
    "BeckeMulticenterQuadrature",
    "BeckePartition",
    "BeckePartitionEvaluation",
    "BeckeQuadratureShell",
    "BeckeSphericalBasis",
    "CanonicalOrthogonalization",
    "CenterGeometry",
    "FEMDVRQuadratureSpec",
    "GeneralizedEigenDiagnostics",
    "GeneralizedEigenResult",
    "H2PlusBeckeResult",
    "H2PlusGroundStateValidation",
    "H2PlusMatrices",
    "H2PlusMidpointReference",
    "KineticElementComparison",
    "KineticIntegrationSettings",
    "MatrixAssemblyDiagnostics",
    "MatrixRelativeErrors",
    "MidpointL0Validation",
    "MidpointMatrixSet",
    "OverlapRankDecision",
    "TwoCenterKineticMatrix",
    "TwoCenterKineticValidation",
    "TwoCenterOverlapStudy",
    "TwoCenterOverlapValidation",
    "assemble_becke_matrices",
    "canonical_orthogonalization",
    "classify_centers",
    "fixed_m_channels",
    "full_lm_channels",
    "run_two_center_overlap_study",
    "solve_becke_eigenproblem",
    "solve_generalized_hermitian",
    "validate_h2plus_ground_state",
    "validate_midpoint_l0",
    "validate_two_center_kinetic",
    "validate_two_center_overlap",
]
