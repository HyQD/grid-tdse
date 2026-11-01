"""Ground-state solvers independent of a spatial representation."""

from .davidson import (
    DavidsonGroundStateSolver,
    DavidsonPreconditioner,
    DiagonalDavidsonPreconditioner,
    IdentityDavidsonPreconditioner,
)
from .eigenproblem import (
    EigenTerminationReason,
    GroundStateEigenIteration,
    GroundStateEigenResult,
)
from .imaginary_time import (
    GroundStateIteration,
    GroundStateResult,
    ImaginaryTimeGroundStateSolver,
)
from .lanczos import LanczosGroundStateSolver
from .subspace import (
    BatchedInnerProducts,
    InnerProduct,
    OrthonormalizationResult,
    RayleighRitzResult,
    basis_orthogonality_error,
    basis_overlaps,
    extend_projected_hermitian_matrix,
    normalize,
    orthonormalize,
    phase_align,
    phase_aligned_distance,
    physical_norm,
    projected_hermitian_matrix,
    rayleigh_ritz,
)

__all__ = [
    "BatchedInnerProducts",
    "DavidsonGroundStateSolver",
    "DavidsonPreconditioner",
    "DiagonalDavidsonPreconditioner",
    "EigenTerminationReason",
    "GroundStateIteration",
    "GroundStateEigenIteration",
    "GroundStateEigenResult",
    "GroundStateResult",
    "InnerProduct",
    "ImaginaryTimeGroundStateSolver",
    "IdentityDavidsonPreconditioner",
    "LanczosGroundStateSolver",
    "OrthonormalizationResult",
    "RayleighRitzResult",
    "basis_orthogonality_error",
    "basis_overlaps",
    "extend_projected_hermitian_matrix",
    "normalize",
    "orthonormalize",
    "phase_align",
    "phase_aligned_distance",
    "physical_norm",
    "projected_hermitian_matrix",
    "rayleigh_ritz",
]
