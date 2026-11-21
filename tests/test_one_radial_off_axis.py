import numpy as np
import pytest

from grid_lib.spherical_coordinates import SphericalHarmonicRotation
from grid_tdse.spherical.one_radial import (
    OneElectronDiatomicHamiltonian,
    OneElectronDiatomicSystem,
    diagonalize_full_angular,
)


def _rotation_y(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )


def _system(angle=0.0, *, charge1=1.0, charge2=1.0):
    rotation = _rotation_y(angle)
    positions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    positions = positions @ rotation.T
    return OneElectronDiatomicSystem.from_nuclear_positions(
        charge1,
        charge2,
        positions[0],
        positions[1],
        name="test",
    )


def _hamiltonian(
    angle=0.0,
    *,
    points=5,
    l_max=2,
    m_max=None,
    nuclear_L_max=None,
    nuclear_M_max=None,
    nuclear_method="analytic",
    system=None,
):
    return OneElectronDiatomicHamiltonian(
        _system(angle) if system is None else system,
        nodes=[0.0, 1.0, 3.0, 6.0],
        n_points=[points, points, points],
        l_max=l_max,
        m_max=m_max,
        nuclear_L_max=nuclear_L_max,
        nuclear_M_max=nuclear_M_max,
        nuclear_method=nuclear_method,
    )


def _norm(hamiltonian, state):
    return np.sqrt(hamiltonian.inner_product(state, state).real)


def test_off_axis_defaults_complete_multipole_space():
    tilted = _hamiltonian(np.pi / 4.0, l_max=3)
    axial = _hamiltonian(0.0, l_max=3)

    assert tilted.m_max == 3
    assert tilted.nuclear_L_max == 6
    assert tilted.nuclear_M_max == 6
    assert not tilted.potential.is_axisymmetric
    assert axial.nuclear_M_max == 0
    assert axial.potential.is_axisymmetric


def test_all_distinct_nonzero_nuclear_radii_must_be_element_boundaries():
    system = OneElectronDiatomicSystem.from_nuclear_positions(
        1.0,
        2.0,
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    )
    hamiltonian = OneElectronDiatomicHamiltonian(
        system,
        nodes=[0.0, 1.0, 2.0, 4.0],
        n_points=[4, 4, 4],
        l_max=1,
    )

    assert hamiltonian.orbital_shape[1] == 4
    with pytest.raises(ValueError, match=r"missing radii: \[2.0\]"):
        OneElectronDiatomicHamiltonian(
            system,
            nodes=[0.0, 1.0, 4.0],
            n_points=[4, 4],
            l_max=1,
        )


def test_fixed_m_hamiltonian_rejects_off_axis_geometry():
    hamiltonian = _hamiltonian(np.pi / 4.0, l_max=2)
    state = np.zeros(hamiltonian.fixed_m_shape(0))

    with pytest.raises(ValueError, match="nuclei on the Cartesian z axis"):
        hamiltonian.apply_fixed_m(state, 0)
    with pytest.raises(ValueError, match="nuclei on the Cartesian z axis"):
        hamiltonian.fixed_m_matrix(0)


def test_full_matrix_action_metric_and_transform_agree():
    hamiltonian = _hamiltonian(np.pi / 4.0, l_max=2)
    generator = np.random.default_rng(121)
    state = generator.normal(size=(2, *hamiltonian.orbital_shape))
    state = state + 1j * generator.normal(size=state.shape)
    strong = hamiltonian.matrix(representation="strong")
    symmetric = hamiltonian.matrix(representation="symmetric")
    weights = hamiltonian.metric_weights()

    expected = np.asarray(
        [
            (strong @ orbital.reshape(-1)).reshape(
                hamiltonian.orbital_shape
            )
            for orbital in state
        ]
    )
    np.testing.assert_allclose(
        hamiltonian.apply(state),
        expected,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        strong.conj().T * weights[None, :],
        weights[:, None] * strong,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        symmetric,
        symmetric.conj().T,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    euclidean = hamiltonian.to_symmetric(state)
    np.testing.assert_allclose(hamiltonian.from_symmetric(euclidean), state)
    np.testing.assert_allclose(
        np.sum(np.abs(euclidean) ** 2),
        sum(
            hamiltonian.inner_product(orbital, orbital).real
            for orbital in state
        ),
    )
    np.testing.assert_allclose(
        hamiltonian.diagonal().reshape(-1),
        np.diag(strong),
    )


@pytest.mark.parametrize(
    ("coordinate", "momentum"),
    [
        ("apply_x", "apply_px"),
        ("apply_y", "apply_py"),
        ("apply_z", "apply_pz"),
    ],
)
def test_cartesian_operators_are_metric_hermitian(coordinate, momentum):
    hamiltonian = _hamiltonian(np.pi / 4.0, points=6, l_max=3)
    generator = np.random.default_rng(141)
    left = generator.normal(size=hamiltonian.orbital_shape)
    left = left + 1j * generator.normal(size=left.shape)
    right = generator.normal(size=hamiltonian.orbital_shape)
    right = right + 1j * generator.normal(size=right.shape)

    for action_name in (coordinate, momentum):
        action = getattr(hamiltonian, action_name)
        lhs = hamiltonian.inner_product(left, action(right))
        rhs = hamiltonian.inner_product(action(left), right)
        assert lhs == pytest.approx(rhs, abs=8.0e-13, rel=8.0e-13)


def test_multipoles_beyond_projected_cutoffs_do_not_change_full_matrix():
    complete = _hamiltonian(
        np.pi / 4.0,
        l_max=2,
        nuclear_L_max=4,
        nuclear_M_max=4,
    )
    extended = _hamiltonian(
        np.pi / 4.0,
        l_max=2,
        nuclear_L_max=6,
        nuclear_M_max=6,
    )

    np.testing.assert_allclose(
        complete.matrix(),
        extended.matrix(),
        atol=4.0e-14,
        rtol=4.0e-14,
    )


def test_direct_and_davidson_full_angular_states_agree():
    hamiltonian = _hamiltonian(np.pi / 4.0, points=5, l_max=2)
    direct = diagonalize_full_angular(
        hamiltonian,
        n_states=2,
        solver="direct",
    )
    davidson = diagonalize_full_angular(
        hamiltonian,
        n_states=2,
        residual_tolerance=2.0e-10,
        max_iterations=120,
        maximum_subspace_dimension=32,
    )

    assert direct.converged
    assert davidson.converged
    np.testing.assert_allclose(
        davidson.electronic_energies,
        direct.electronic_energies,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    assert np.max(davidson.residual_norms) < 2.0e-10
    overlaps = np.abs(
        [
            hamiltonian.inner_product(left, right)
            for left, right in zip(direct.orbitals, davidson.orbitals)
        ]
    )
    np.testing.assert_allclose(overlaps, 1.0, atol=2.0e-9)


@pytest.mark.parametrize("angle", [np.pi / 6.0, np.pi / 4.0, np.pi / 2.0])
def test_rotated_h2plus_state_is_exact_finite_basis_solution(angle):
    l_max = 2
    axial = _hamiltonian(0.0, points=5, l_max=l_max)
    tilted = _hamiltonian(angle, points=5, l_max=l_max)
    axial_states = diagonalize_full_angular(
        axial,
        n_states=4,
        solver="direct",
    )
    tilted_states = diagonalize_full_angular(
        tilted,
        n_states=4,
        solver="direct",
    )
    rotation = SphericalHarmonicRotation(
        _rotation_y(angle),
        l_max=l_max,
    )
    rotated = rotation(axial_states.orbital)
    residual = tilted.apply(rotated) - (
        axial_states.electronic_energy * rotated
    )

    np.testing.assert_allclose(
        tilted_states.electronic_energies,
        axial_states.electronic_energies,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    assert _norm(tilted, residual) < 3.0e-11
    assert abs(tilted.inner_product(tilted_states.orbital, rotated)) == (
        pytest.approx(1.0, abs=2.0e-11)
    )


def test_full_hamiltonian_and_vector_operators_are_rotation_covariant():
    angle = np.pi / 4.0
    l_max = 2
    rotation_matrix = _rotation_y(angle)
    rotation = SphericalHarmonicRotation(
        rotation_matrix,
        l_max=l_max,
    )
    axial = _hamiltonian(0.0, points=5, l_max=l_max)
    tilted = _hamiltonian(angle, points=5, l_max=l_max)
    generator = np.random.default_rng(191)
    state = generator.normal(size=axial.orbital_shape)
    state = state + 1j * generator.normal(size=state.shape)
    rotated = rotation(state)

    np.testing.assert_allclose(
        tilted.apply(rotated),
        rotation(axial.apply(state)),
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    for action_names in (
        ("apply_x", "apply_y", "apply_z"),
        ("apply_px", "apply_py", "apply_pz"),
    ):
        original_vector = np.asarray(
            [
                axial.inner_product(state, getattr(axial, name)(state))
                for name in action_names
            ]
        )
        rotated_vector = np.asarray(
            [
                tilted.inner_product(
                    rotated,
                    getattr(tilted, name)(rotated),
                )
                for name in action_names
            ]
        )
        np.testing.assert_allclose(
            rotated_vector,
            rotation_matrix @ original_vector,
            atol=2.0e-12,
            rtol=2.0e-12,
        )


def test_truncated_nuclear_magnetic_space_breaks_rotation_covariance():
    angle = np.pi / 4.0
    l_max = 2
    rotation = SphericalHarmonicRotation(
        _rotation_y(angle),
        l_max=l_max,
    )
    axial = _hamiltonian(0.0, points=5, l_max=l_max)
    truncated = _hamiltonian(
        angle,
        points=5,
        l_max=l_max,
        nuclear_M_max=0,
    )
    generator = np.random.default_rng(211)
    state = generator.normal(size=axial.orbital_shape)
    state = state + 1j * generator.normal(size=state.shape)
    rotated = rotation(state)
    covariance_defect = (
        truncated.apply(rotated) - rotation(axial.apply(state))
    )

    assert _norm(truncated, covariance_defect) > 1.0e-2


def test_analytic_and_poisson_tilted_energies_converge_radially():
    energy_errors = []
    action_errors = []
    coefficient_errors = []
    for points in (6, 8):
        analytic_hamiltonian = _hamiltonian(
            np.pi / 4.0,
            points=points,
            l_max=1,
            nuclear_method="analytic",
        )
        poisson_hamiltonian = _hamiltonian(
            np.pi / 4.0,
            points=points,
            l_max=1,
            nuclear_method="poisson",
        )
        analytic = diagonalize_full_angular(
            analytic_hamiltonian,
            solver="direct",
        )
        poisson = diagonalize_full_angular(
            poisson_hamiltonian,
            solver="direct",
        )
        energy_errors.append(
            abs(analytic.electronic_energy - poisson.electronic_energy)
        )
        generator = np.random.default_rng(231)
        state = generator.normal(size=analytic_hamiltonian.orbital_shape)
        state = state + 1j * generator.normal(size=state.shape)
        reference = analytic_hamiltonian.apply(state)
        difference = reference - poisson_hamiltonian.apply(state)
        action_errors.append(
            _norm(analytic_hamiltonian, difference)
            / _norm(analytic_hamiltonian, reference)
        )
        coefficient_errors.append(
            np.max(
                np.abs(
                    analytic_hamiltonian.potential.coefficients
                    - poisson_hamiltonian.potential.coefficients
                )
            )
        )

    assert energy_errors[1] < energy_errors[0]
    assert action_errors[1] < action_errors[0]
    assert coefficient_errors[1] < coefficient_errors[0]
    assert energy_errors[1] < 1.0e-6
    assert action_errors[1] < 2.0e-7


def test_off_axis_metadata_records_geometry_and_cutoffs():
    hamiltonian = _hamiltonian(np.pi / 4.0, l_max=2)
    metadata = hamiltonian.archive_metadata()

    np.testing.assert_allclose(
        metadata["nuclear_positions"],
        hamiltonian.system.nuclear_positions,
    )
    np.testing.assert_allclose(metadata["nuclear_midpoint"], 0.0)
    np.testing.assert_allclose(
        metadata["bond_axis"],
        hamiltonian.system.bond_axis,
    )
    assert metadata["nuclear_L_max"] == 4
    assert metadata["nuclear_M_max"] == 4
