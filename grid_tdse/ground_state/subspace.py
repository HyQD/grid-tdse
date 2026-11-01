"""Weighted subspace primitives for Hermitian ground-state eigensolvers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


InnerProduct = Callable[[np.ndarray, np.ndarray], complex]
BatchedInnerProducts = Callable[
    [Sequence[np.ndarray], np.ndarray], np.ndarray
]


@dataclass(frozen=True)
class OrthonormalizationResult:
    """Result of projecting and normalizing one subspace direction.

    ``vector`` is normalized when ``accepted`` is true.  For a rejected
    direction it contains the final, unnormalized projected vector so callers
    can inspect the breakdown without silently replacing it.
    """

    vector: np.ndarray
    norm: float
    initial_norm: float
    overlaps: np.ndarray
    passes: int
    accepted: bool


@dataclass(frozen=True)
class RayleighRitzResult:
    """Lowest Ritz pairs extracted from a weighted trial subspace."""

    projected_matrix: np.ndarray
    eigenvalues: np.ndarray
    coefficients: np.ndarray
    vectors: tuple[np.ndarray, ...]
    hamiltonian_images: tuple[np.ndarray, ...]
    residuals: tuple[np.ndarray, ...]
    residual_norms: np.ndarray


def physical_norm(
    vector: np.ndarray,
    inner_product: InnerProduct,
    *,
    reality_tolerance: float = 1.0e-12,
) -> float:
    """Return the norm induced by ``inner_product``.

    A significant imaginary part or a negative squared norm indicates that
    the callback does not provide a valid physical inner product for the
    supplied vector.
    """

    values = _numeric_array(vector, "vector")
    product = _inner_product_value(inner_product, values, values)
    norm_squared = _real_value(
        product,
        "squared norm",
        reality_tolerance=reality_tolerance,
    )
    if norm_squared < 0.0:
        raise ValueError("inner product produced a negative squared norm")
    return float(np.sqrt(norm_squared))


def normalize(
    vector: np.ndarray,
    inner_product: InnerProduct,
    *,
    breakdown_tolerance: float = 0.0,
    reality_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return a physically normalized copy of ``vector``."""

    tolerance = _nonnegative_real(
        breakdown_tolerance,
        "breakdown_tolerance",
    )
    values = _numeric_array(vector, "vector").copy()
    norm = physical_norm(
        values,
        inner_product,
        reality_tolerance=reality_tolerance,
    )
    if norm <= tolerance:
        raise ValueError(
            "vector norm does not exceed the breakdown tolerance"
        )
    return values / norm


def basis_overlaps(
    basis: Sequence[np.ndarray],
    vector: np.ndarray,
    inner_product: InnerProduct,
    *,
    inner_products: BatchedInnerProducts | None = None,
) -> np.ndarray:
    """Return ``<basis[i], vector>`` with one optional batched callback."""

    if not callable(inner_product):
        raise TypeError("inner_product must be callable")
    if inner_products is not None and not callable(inner_products):
        raise TypeError("inner_products must be callable or None")

    right = _numeric_array(vector, "vector")
    vectors = _basis_arrays(basis, right.shape)
    if not vectors:
        return np.empty(0, dtype=complex)

    if inner_products is None:
        return np.asarray(
            [
                _inner_product_value(inner_product, left, right)
                for left in vectors
            ]
        )

    overlaps = np.asarray(inner_products(vectors, right))
    if overlaps.shape != (len(vectors),):
        raise ValueError(
            "inner_products must return one value per left vector"
        )
    if not np.issubdtype(overlaps.dtype, np.number):
        raise TypeError("inner_products must return numeric values")
    if not np.isfinite(overlaps).all():
        raise ValueError("inner_products returned non-finite values")
    return overlaps.copy()


def orthonormalize(
    vector: np.ndarray,
    basis: Sequence[np.ndarray],
    inner_product: InnerProduct,
    *,
    inner_products: BatchedInnerProducts | None = None,
    reorthogonalization: Literal["if-needed", "always"] = "if-needed",
    reorthogonalization_threshold: float = 0.5,
    breakdown_tolerance: float = 1.0e-14,
    reality_tolerance: float = 1.0e-12,
) -> OrthonormalizationResult:
    """Project one direction against a physical-orthonormal basis.

    The first projection uses one call to ``inner_products`` when that callback
    is supplied.  A second pass is made unconditionally in ``"always"`` mode,
    or in ``"if-needed"`` mode when the first-pass norm has fallen below
    ``reorthogonalization_threshold`` times the input norm (the DGKS
    criterion).

    The returned direction is accepted only if its final norm exceeds
    ``breakdown_tolerance`` times its input norm.
    """

    if reorthogonalization not in {"if-needed", "always"}:
        raise ValueError(
            "reorthogonalization must be 'if-needed' or 'always'"
        )
    threshold = _unit_interval_real(
        reorthogonalization_threshold,
        "reorthogonalization_threshold",
    )
    tolerance = _nonnegative_real(
        breakdown_tolerance,
        "breakdown_tolerance",
    )

    direction = _numeric_array(vector, "vector").copy()
    vectors = _basis_arrays(basis, direction.shape)
    initial_norm = physical_norm(
        direction,
        inner_product,
        reality_tolerance=reality_tolerance,
    )
    if initial_norm == 0.0:
        return OrthonormalizationResult(
            vector=direction,
            norm=0.0,
            initial_norm=0.0,
            overlaps=np.zeros(len(vectors), dtype=complex),
            passes=0,
            accepted=False,
        )

    total_overlaps = np.zeros(len(vectors), dtype=complex)
    passes = 0
    if vectors:
        overlaps = basis_overlaps(
            vectors,
            direction,
            inner_product,
            inner_products=inner_products,
        )
        direction = _subtract_projection(direction, vectors, overlaps)
        total_overlaps += overlaps
        passes = 1

        first_norm = physical_norm(
            direction,
            inner_product,
            reality_tolerance=reality_tolerance,
        )
        needs_second_pass = (
            reorthogonalization == "always"
            or first_norm < threshold * initial_norm
        )
        if needs_second_pass:
            overlaps = basis_overlaps(
                vectors,
                direction,
                inner_product,
                inner_products=inner_products,
            )
            direction = _subtract_projection(direction, vectors, overlaps)
            total_overlaps += overlaps
            passes = 2

    norm = physical_norm(
        direction,
        inner_product,
        reality_tolerance=reality_tolerance,
    )
    accepted = norm > tolerance * initial_norm
    if accepted:
        direction = direction / norm
    return OrthonormalizationResult(
        vector=direction,
        norm=norm,
        initial_norm=initial_norm,
        overlaps=total_overlaps,
        passes=passes,
        accepted=accepted,
    )


def basis_orthogonality_error(
    basis: Sequence[np.ndarray],
    inner_product: InnerProduct,
    *,
    inner_products: BatchedInnerProducts | None = None,
) -> float:
    """Return ``max(abs(V** W V - I))`` for a sequence of basis vectors."""

    vectors = _nonempty_basis_arrays(basis)
    gram = np.column_stack(
        [
            basis_overlaps(
                vectors,
                right,
                inner_product,
                inner_products=inner_products,
            )
            for right in vectors
        ]
    )
    identity = np.eye(len(vectors), dtype=complex)
    return float(np.max(np.abs(gram - identity)))


def projected_hermitian_matrix(
    basis: Sequence[np.ndarray],
    hamiltonian_images: Sequence[np.ndarray],
    inner_product: InnerProduct,
    *,
    inner_products: BatchedInnerProducts | None = None,
    hermiticity_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Construct and symmetrize ``V** W H V``.

    Significant non-Hermiticity is rejected before the small roundoff-level
    asymmetry is removed.
    """

    tolerance = _nonnegative_real(
        hermiticity_tolerance,
        "hermiticity_tolerance",
    )
    vectors = _nonempty_basis_arrays(basis)
    images = _basis_arrays(
        hamiltonian_images,
        vectors[0].shape,
        name="hamiltonian_images",
    )
    if len(images) != len(vectors):
        raise ValueError(
            "basis and hamiltonian_images must have the same length"
        )

    projected = np.column_stack(
        [
            basis_overlaps(
                vectors,
                image,
                inner_product,
                inner_products=inner_products,
            )
            for image in images
        ]
    )
    defect = projected - projected.conj().T
    scale = max(1.0, float(np.max(np.abs(projected))))
    error = float(np.max(np.abs(defect)))
    if error > tolerance * scale:
        raise ValueError(
            "projected Hamiltonian is not Hermitian within tolerance: "
            f"{error:.3e}"
        )
    return 0.5 * (projected + projected.conj().T)


def extend_projected_hermitian_matrix(
    previous: np.ndarray,
    basis: Sequence[np.ndarray],
    new_hamiltonian_image: np.ndarray,
    inner_product: InnerProduct,
    *,
    inner_products: BatchedInnerProducts | None = None,
    hermiticity_tolerance: float = 1.0e-10,
    reality_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Append one row and column to a projected Hermitian matrix.

    ``basis`` includes the newly added vector and ``previous`` represents all
    earlier vectors.  Self-adjointness supplies the new row from the conjugate
    of the batched new column, so only one overlap batch is required.
    """

    tolerance = _nonnegative_real(
        hermiticity_tolerance,
        "hermiticity_tolerance",
    )
    vectors = _nonempty_basis_arrays(basis)
    dimension = len(vectors)
    old = _validated_hermitian_matrix(
        previous,
        dimension=dimension - 1,
        hermiticity_tolerance=tolerance,
    )
    image = _numeric_array(
        new_hamiltonian_image,
        "new_hamiltonian_image",
    )
    if image.shape != vectors[0].shape:
        raise ValueError(
            "new_hamiltonian_image must have the basis-vector shape"
        )
    column = basis_overlaps(
        vectors,
        image,
        inner_product,
        inner_products=inner_products,
    )
    diagonal = _real_value(
        complex(column[-1]),
        "projected Hamiltonian diagonal",
        reality_tolerance=reality_tolerance,
    )
    result_dtype = np.result_type(old.dtype, column.dtype, float)
    result = np.empty((dimension, dimension), dtype=result_dtype)
    result[:-1, :-1] = old
    result[:, -1] = column
    result[-1, :-1] = column[:-1].conj()
    result[-1, -1] = diagonal
    return result


def rayleigh_ritz(
    basis: Sequence[np.ndarray],
    hamiltonian_images: Sequence[np.ndarray],
    inner_product: InnerProduct,
    *,
    inner_products: BatchedInnerProducts | None = None,
    number_of_roots: int | None = None,
    projected_matrix: np.ndarray | None = None,
    hermiticity_tolerance: float = 1.0e-10,
    reality_tolerance: float = 1.0e-12,
) -> RayleighRitzResult:
    """Extract the lowest Ritz pairs from cached basis and Hamiltonian images."""

    vectors = _nonempty_basis_arrays(basis)
    images = _basis_arrays(
        hamiltonian_images,
        vectors[0].shape,
        name="hamiltonian_images",
    )
    if len(images) != len(vectors):
        raise ValueError(
            "basis and hamiltonian_images must have the same length"
        )
    roots = _root_count(number_of_roots, len(vectors))
    if projected_matrix is None:
        projected = projected_hermitian_matrix(
            vectors,
            images,
            inner_product,
            inner_products=inner_products,
            hermiticity_tolerance=hermiticity_tolerance,
        )
    else:
        projected = _validated_hermitian_matrix(
            projected_matrix,
            dimension=len(vectors),
            hermiticity_tolerance=hermiticity_tolerance,
        )
    eigenvalues, coefficients = np.linalg.eigh(projected)
    eigenvalues = np.asarray(eigenvalues[:roots], dtype=float)
    coefficients = coefficients[:, :roots]

    ritz_vectors = tuple(
        _linear_combination(vectors, coefficients[:, root])
        for root in range(roots)
    )
    ritz_images = tuple(
        _linear_combination(images, coefficients[:, root])
        for root in range(roots)
    )
    residuals = tuple(
        image - eigenvalue * vector
        for eigenvalue, vector, image in zip(
            eigenvalues,
            ritz_vectors,
            ritz_images,
            strict=True,
        )
    )
    residual_norms = np.asarray(
        [
            physical_norm(
                residual,
                inner_product,
                reality_tolerance=reality_tolerance,
            )
            for residual in residuals
        ],
        dtype=float,
    )
    return RayleighRitzResult(
        projected_matrix=projected,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        vectors=ritz_vectors,
        hamiltonian_images=ritz_images,
        residuals=residuals,
        residual_norms=residual_norms,
    )


def phase_align(
    reference: np.ndarray,
    candidate: np.ndarray,
    inner_product: InnerProduct,
) -> np.ndarray:
    """Return a copy of ``candidate`` aligned to ``reference`` globally."""

    left = _numeric_array(reference, "reference")
    right = _numeric_array(candidate, "candidate")
    if right.shape != left.shape:
        raise ValueError("reference and candidate must have the same shape")
    overlap = _inner_product_value(inner_product, left, right)
    if abs(overlap) == 0.0:
        return right.copy()
    return right * overlap.conjugate() / abs(overlap)


def phase_aligned_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
    inner_product: InnerProduct,
    *,
    reality_tolerance: float = 1.0e-12,
) -> float:
    """Return the physical norm of a global-phase-aligned state difference."""

    aligned = phase_align(reference, candidate, inner_product)
    return physical_norm(
        aligned - np.asarray(reference),
        inner_product,
        reality_tolerance=reality_tolerance,
    )


def _subtract_projection(
    vector: np.ndarray,
    basis: Sequence[np.ndarray],
    overlaps: np.ndarray,
) -> np.ndarray:
    result_dtype = np.result_type(
        vector.dtype,
        overlaps.dtype,
        *(basis_vector.dtype for basis_vector in basis),
    )
    result = np.asarray(vector, dtype=result_dtype).copy()
    for basis_vector, overlap in zip(basis, overlaps, strict=True):
        result -= overlap * basis_vector
    return result


def _linear_combination(
    vectors: Sequence[np.ndarray],
    coefficients: np.ndarray,
) -> np.ndarray:
    result_dtype = np.result_type(
        coefficients.dtype,
        *(vector.dtype for vector in vectors),
    )
    result = np.zeros(vectors[0].shape, dtype=result_dtype)
    for vector, coefficient in zip(vectors, coefficients, strict=True):
        result += coefficient * vector
    return result


def _numeric_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _basis_arrays(
    basis: Sequence[np.ndarray],
    shape: tuple[int, ...],
    *,
    name: str = "basis",
) -> tuple[np.ndarray, ...]:
    if isinstance(basis, np.ndarray):
        raise TypeError(f"{name} must be a sequence of arrays")
    try:
        values = tuple(basis)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of arrays") from error
    arrays = tuple(
        _numeric_array(vector, f"{name}[{index}]")
        for index, vector in enumerate(values)
    )
    for array in arrays:
        if array.shape != shape:
            raise ValueError(f"all {name} vectors must have shape {shape}")
    return arrays


def _nonempty_basis_arrays(
    basis: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    if isinstance(basis, np.ndarray):
        raise TypeError("basis must be a sequence of arrays")
    try:
        values = tuple(basis)
    except TypeError as error:
        raise TypeError("basis must be a sequence of arrays") from error
    if not values:
        raise ValueError("basis must contain at least one vector")
    first = _numeric_array(values[0], "basis[0]")
    return _basis_arrays(values, first.shape)


def _inner_product_value(
    inner_product: InnerProduct,
    left: np.ndarray,
    right: np.ndarray,
) -> complex | float:
    if not callable(inner_product):
        raise TypeError("inner_product must be callable")
    value = np.asarray(inner_product(left, right))
    if value.shape != ():
        raise ValueError("inner_product must return a scalar")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError("inner_product must return a numeric scalar")
    result = value.item()
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("inner_product returned a non-finite value")
    return result


def _real_value(
    value: complex,
    name: str,
    *,
    reality_tolerance: float,
) -> float:
    tolerance = _nonnegative_real(
        reality_tolerance,
        "reality_tolerance",
    )
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > tolerance * scale:
        raise ValueError(
            f"{name} has a significant imaginary part: {value.imag}"
        )
    return float(value.real)


def _root_count(number_of_roots: int | None, dimension: int) -> int:
    if number_of_roots is None:
        return dimension
    if isinstance(number_of_roots, (bool, np.bool_)) or not isinstance(
        number_of_roots,
        (int, np.integer),
    ):
        raise TypeError("number_of_roots must be a positive integer or None")
    roots = int(number_of_roots)
    if roots <= 0:
        raise ValueError("number_of_roots must be positive")
    if roots > dimension:
        raise ValueError("number_of_roots cannot exceed the subspace dimension")
    return roots


def _validated_hermitian_matrix(
    matrix: np.ndarray,
    *,
    dimension: int,
    hermiticity_tolerance: float,
) -> np.ndarray:
    values = np.asarray(matrix)
    if values.shape != (dimension, dimension):
        raise ValueError(
            f"projected matrix must have shape {(dimension, dimension)}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("projected matrix must contain numeric values")
    if not np.isfinite(values).all():
        raise ValueError("projected matrix must contain only finite values")
    defect = values - values.conj().T
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    error = float(np.max(np.abs(defect), initial=0.0))
    if error > hermiticity_tolerance * scale:
        raise ValueError(
            "projected matrix is not Hermitian within tolerance: "
            f"{error:.3e}"
        )
    return (0.5 * (values + values.conj().T)).copy()


def _nonnegative_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _unit_interval_real(value, name: str) -> float:
    result = _nonnegative_real(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return result


__all__ = [
    "BatchedInnerProducts",
    "InnerProduct",
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
