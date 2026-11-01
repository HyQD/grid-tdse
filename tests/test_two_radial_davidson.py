import numpy as np
import pytest

from grid_tdse.ground_state import DavidsonGroundStateSolver
from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    NonBornOppenheimerDiatomicIonGroundState,
    TwoRadialDiagonalDavidsonPreconditioner,
    TwoRadialOneBodyDavidsonPreconditioner,
)


def _calculation(l_max=2):
    electronic = FEMDVRGridSpec([0.0, 2.0, 5.0], [3, 3])
    internuclear = FEMDVRGridSpec([0.0, 1.5, 4.0], [3, 3])
    return NonBornOppenheimerDiatomicIonGroundState(
        "H2+",
        electronic_grid=electronic,
        internuclear_grid=internuclear,
        inversion_parity=1,
        pair_exchange_parity=1,
        l_max=l_max,
        m_max=l_max,
        lambda_max=2 * l_max,
        mu_max=2 * l_max,
        M_tot=0,
        max_iterations=1,
        progress_interval=None,
        sample_interval=None,
        use_preconditioner=False,
    )


def test_two_radial_diagonal_davidson_matches_one_body_diagonal():
    system = _calculation().representation
    preconditioner = TwoRadialDiagonalDavidsonPreconditioner(
        system,
        relative_floor=0.0,
        absolute_floor=1.0e-12,
    )
    rng = np.random.default_rng(41)
    residual = rng.normal(size=system.psi_shape) + 1j * rng.normal(
        size=system.psi_shape
    )
    theta = -1.7

    actual = preconditioner(residual, theta)
    expected = np.empty_like(actual)
    for pair, (i, j) in enumerate(system.angular_pairs.pairs):
        denominator = (
            np.diag(system.h1[i])[:, None]
            + np.diag(system.h2[j])[None, :]
            - theta
        )
        expected[pair] = -residual[pair] / denominator

    np.testing.assert_allclose(actual, expected, atol=2.0e-13, rtol=2.0e-13)
    np.testing.assert_allclose(preconditioner.diagonal - theta, np.asarray([
        np.diag(system.h1[i])[:, None]
        + np.diag(system.h2[j])[None, :]
        - theta
        for i, j in system.angular_pairs.pairs
    ]))
    assert preconditioner.last_regularized_entries == 0


def test_two_radial_spectral_davidson_matches_direct_one_body_inverse():
    system = _calculation().representation
    preconditioner = TwoRadialOneBodyDavidsonPreconditioner(
        system,
        relative_floor=0.0,
        absolute_floor=1.0e-12,
    )
    rng = np.random.default_rng(42)
    residual = np.zeros(system.psi_shape)
    residual[0] = rng.normal(size=residual.shape[1:])
    theta = -2.0

    actual = preconditioner(residual, theta)[0]
    i, j = system.angular_pairs.pairs[0]
    h1 = system.h1[i]
    h2 = system.h2[j]
    dimension = h1.shape[0] * h2.shape[0]
    matrix = (
        np.kron(h1, np.eye(h2.shape[0]))
        + np.kron(np.eye(h1.shape[0]), h2)
        - theta * np.eye(dimension)
    )
    expected = -np.linalg.solve(matrix, residual[0].ravel()).reshape(
        actual.shape
    )

    np.testing.assert_allclose(actual, expected, atol=3.0e-12, rtol=3.0e-12)
    assert preconditioner.last_regularized_entries == 0


@pytest.mark.parametrize(
    "preconditioner_class",
    [
        TwoRadialDiagonalDavidsonPreconditioner,
        TwoRadialOneBodyDavidsonPreconditioner,
    ],
)
def test_two_radial_davidson_preconditioners_validate_state_and_ritz_value(
    preconditioner_class,
):
    system = _calculation().representation
    preconditioner = preconditioner_class(system)

    with pytest.raises(ValueError, match="shape"):
        preconditioner(np.ones(3), -0.5)
    with pytest.raises(ValueError, match="finite"):
        preconditioner(np.ones(system.psi_shape), np.inf)


def test_spectral_davidson_preconditioner_converges_small_physical_problem():
    calculation = _calculation(l_max=2)
    system = calculation.representation
    solver = DavidsonGroundStateSolver(
        max_iterations=80,
        maximum_subspace_dimension=10,
        retained_ritz_vectors=2,
        residual_tolerance=1.0e-8,
        track_orthogonality=False,
        progress_interval=None,
    )

    result = solver.solve(
        calculation.initial_state,
        calculation.hamiltonian,
        system.inner_product,
        preconditioner=TwoRadialOneBodyDavidsonPreconditioner(
            system,
            relative_floor=1.0e-6,
            level_shift=0.1,
        ),
    )

    assert result.converged
    assert result.residual_norm <= solver.residual_tolerance
    assert result.preconditioner_applications > 0
    assert result.restarts > 0


def test_physical_preconditioner_reduces_lmax4_hamiltonian_actions():
    calculation = _calculation(l_max=4)
    system = calculation.representation
    options = {
        "max_iterations": 80,
        "maximum_subspace_dimension": 16,
        "retained_ritz_vectors": 4,
        "residual_tolerance": 1.0e-8,
        "track_orthogonality": False,
        "progress_interval": None,
    }

    identity = DavidsonGroundStateSolver(**options).solve(
        calculation.initial_state,
        calculation.hamiltonian,
        system.inner_product,
    )
    spectral = DavidsonGroundStateSolver(**options).solve(
        calculation.initial_state,
        calculation.hamiltonian,
        system.inner_product,
        preconditioner=TwoRadialOneBodyDavidsonPreconditioner(
            system,
            relative_floor=1.0e-6,
            level_shift=0.1,
        ),
    )

    assert identity.converged
    assert spectral.converged
    assert spectral.eigenvalue == pytest.approx(identity.eigenvalue, abs=1.0e-11)
    assert spectral.hamiltonian_actions < identity.hamiltonian_actions
