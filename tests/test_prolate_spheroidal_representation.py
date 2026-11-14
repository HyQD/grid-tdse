import numpy as np
import pytest

from grid_tdse.prolate_spheroidal import ProlateSpheroidalRepresentation


@pytest.fixture
def representation():
    return ProlateSpheroidalRepresentation(
        internuclear_distance=2.0,
        xi_boundaries=[1.0, 4.0, 14.0, 24.0],
        xi_points_per_element=14,
        eta_degree=9,
        m_max=1,
    )


def test_haxton_comparable_grid_shape_and_metadata(representation):
    assert representation.grid.shape == (40, 9)
    assert representation.fixed_m_shape(0) == (39, 9)
    assert representation.fixed_m_shape(1) == (39, 9)
    assert representation.fixed_m_shape(-1) == (39, 9)

    metadata = representation.archive_metadata()
    assert metadata["xi_boundaries"] == [1.0, 4.0, 14.0, 24.0]
    assert metadata["xi_points_per_element"] == [14, 14, 14]
    assert metadata["eta_degree"] == 9
    assert metadata["full_grid_shape"] == [40, 9]


@pytest.mark.parametrize("m", [0, 1, -1])
def test_nodal_euclidean_round_trip_and_norm(representation, m):
    generator = np.random.default_rng(104 + m)
    state = generator.normal(size=representation.fixed_m_shape(m))
    state = state + 1j * generator.normal(size=state.shape)

    transformed = representation.to_euclidean(state, m)
    restored = representation.from_euclidean(transformed, m)

    assert np.allclose(restored, state, rtol=2.0e-14, atol=2.0e-14)
    physical_norm = representation.inner_product(state, state, m).real
    euclidean_norm = np.vdot(transformed, transformed).real
    assert euclidean_norm == pytest.approx(physical_norm, rel=2.0e-14)


def test_active_embedding_and_restriction_round_trip(representation):
    state = np.arange(np.prod(representation.fixed_m_shape(0))).reshape(
        representation.fixed_m_shape(0)
    )
    full = representation.embed_active_state(state, 0, fill_value=-1.0)

    assert full.shape == representation.grid.shape
    assert np.all(full[-1] == -1.0)
    assert np.array_equal(representation.restrict_full_grid(full, 0), state)
