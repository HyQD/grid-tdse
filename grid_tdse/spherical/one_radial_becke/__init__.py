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
from .metric import CanonicalOrthogonalization, canonical_orthogonalization
from .partition import BeckePartition, BeckePartitionEvaluation
from .quadrature import BeckeMulticenterQuadrature, BeckeQuadratureShell
from .specification import FEMDVRQuadratureSpec
from .validation import (
    MatrixRelativeErrors,
    MidpointL0Validation,
    MidpointMatrixSet,
    validate_midpoint_l0,
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
    "MatrixAssemblyDiagnostics",
    "MatrixRelativeErrors",
    "MidpointL0Validation",
    "MidpointMatrixSet",
    "assemble_becke_matrices",
    "canonical_orthogonalization",
    "classify_centers",
    "fixed_m_channels",
    "full_lm_channels",
    "solve_becke_eigenproblem",
    "solve_generalized_hermitian",
    "validate_midpoint_l0",
]
