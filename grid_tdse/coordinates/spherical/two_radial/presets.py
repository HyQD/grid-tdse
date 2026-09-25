from __future__ import annotations

from dataclasses import dataclass

from .system import (
    AngularBasisSpec,
    CoordinateSpec,
    CoulombCenter,
    RadialGridSpec,
    TwoRadialSystemSpec,
    coulomb_w_lm,
    radial_potential,
)


@dataclass(frozen=True)
class ThreeBodyParameters:
    particles: tuple[str, str, str]
    mu_nuclear: float
    mu_electronic: float
    xi_nuclear: float
    xi_electronic: float
    reference_mass: float
    particle1_mass: float
    particle2_mass: float
    reference_charge: float
    particle1_charge: float
    particle2_charge: float


def h2plus(
    r_max: float = 10.0,
    n_r: int = 50,
    R_max: float = 5.0,
    n_R: int = 50,
    l_max_e: int = 2,
    l_max_p: int = 2,
    m_max_e: int | None = None,
    m_max_p: int | None = None,
    codata_set: str = "2022",
) -> TwoRadialSystemSpec:
    return symmetric_diatomic_ion(
        "H2+",
        r_max=r_max,
        n_r=n_r,
        internuclear_R_max=R_max,
        n_R=n_R,
        l_max_e=l_max_e,
        l_max_n=l_max_p,
        m_max_e=m_max_e,
        m_max_n=m_max_p,
        codata_set=codata_set,
    )


def d2plus(
    r_max: float = 10.0,
    n_r: int = 50,
    R_max: float = 5.0,
    n_R: int = 50,
    l_max_e: int = 2,
    l_max_d: int = 2,
    m_max_e: int | None = None,
    m_max_d: int | None = None,
    codata_set: str = "2022",
) -> TwoRadialSystemSpec:
    return symmetric_diatomic_ion(
        "D2+",
        r_max=r_max,
        n_r=n_r,
        internuclear_R_max=R_max,
        n_R=n_R,
        l_max_e=l_max_e,
        l_max_n=l_max_d,
        m_max_e=m_max_e,
        m_max_n=m_max_d,
        codata_set=codata_set,
    )


def hdplus(
    r_max: float = 10.0,
    n_r: int = 50,
    R_max: float = 5.0,
    n_R: int = 50,
    l_max_e: int = 2,
    l_max_n: int = 2,
    m_max_e: int | None = None,
    m_max_n: int | None = None,
    codata_set: str = "2022",
) -> TwoRadialSystemSpec:
    return diatomic_ion(
        "HD+",
        r_max=r_max,
        n_r=n_r,
        internuclear_R_max=R_max,
        n_R=n_R,
        l_max_e=l_max_e,
        l_max_n=l_max_n,
        m_max_e=m_max_e,
        m_max_n=m_max_n,
        codata_set=codata_set,
    )


def symmetric_diatomic_ion(
    system: str,
    r_max: float,
    n_r: int,
    internuclear_R_max: float,
    n_R: int,
    l_max_e: int,
    l_max_n: int,
    m_max_e: int | None = None,
    m_max_n: int | None = None,
    codata_set: str = "2022",
) -> TwoRadialSystemSpec:
    """Preset using the half-internuclear radial coordinate.

    This mirrors the current H2+ examples: the second coordinate is
    ``R = internuclear_distance / 2``.
    """

    params = three_body_parameters(system, codata_set=codata_set)
    if params.reference_mass != params.particle1_mass:
        raise ValueError(
            "symmetric_diatomic_ion requires equal nuclear masses; use "
            "diatomic_ion for heteronuclear systems"
        )

    basis_e = AngularBasisSpec(l_max_e, m_max_e)
    basis_n = AngularBasisSpec(l_max_n, m_max_n)
    L_max = basis_e.l_max + basis_n.l_max
    M_max = L_max

    coord_e = CoordinateSpec(
        grid=RadialGridSpec(r_max, n_r),
        basis=basis_e,
        kinetic_prefactor=-1.0 / (2.0 * params.mu_electronic),
    )
    coord_n = CoordinateSpec(
        grid=RadialGridSpec(internuclear_R_max / 2.0, n_R),
        basis=basis_n,
        kinetic_prefactor=-1.0 / (8.0 * params.mu_nuclear),
        one_body_terms=(
            radial_potential(
                lambda R: params.reference_charge * params.particle1_charge
                / (2.0 * R)
            ),
        ),
    )

    grid_e = coord_e.grid.build()
    grid_n = coord_n.grid.build()
    W_LM = coulomb_w_lm(
        grid_e.r,
        grid_n.r,
        L_max,
        M_max,
        centers=(
            CoulombCenter(
                params.particle2_charge * params.reference_charge,
                radial_scale=1.0,
                angular_parity=-1,
            ),
            CoulombCenter(
                params.particle2_charge * params.particle1_charge,
                radial_scale=1.0,
                angular_parity=1,
            ),
        ),
        weights_R=grid_n.weights,
    )

    return TwoRadialSystemSpec(
        coord1=coord_e,
        coord2=coord_n,
        W_LM=W_LM,
        L_max=L_max,
        M_max=M_max,
        interaction_sign=1.0,
    )


def diatomic_ion(
    system: str,
    r_max: float,
    n_r: int,
    internuclear_R_max: float,
    n_R: int,
    l_max_e: int,
    l_max_n: int,
    m_max_e: int | None = None,
    m_max_n: int | None = None,
    codata_set: str = "2022",
) -> TwoRadialSystemSpec:
    """Preset using the actual internuclear radial coordinate."""

    params = three_body_parameters(system, codata_set=codata_set)

    basis_e = AngularBasisSpec(l_max_e, m_max_e)
    basis_n = AngularBasisSpec(l_max_n, m_max_n)
    L_max = basis_e.l_max + basis_n.l_max
    M_max = L_max

    coord_e = CoordinateSpec(
        grid=RadialGridSpec(r_max, n_r),
        basis=basis_e,
        kinetic_prefactor=-1.0 / (2.0 * params.mu_electronic),
    )
    coord_n = CoordinateSpec(
        grid=RadialGridSpec(internuclear_R_max, n_R),
        basis=basis_n,
        kinetic_prefactor=-1.0 / (2.0 * params.mu_nuclear),
        one_body_terms=(
            radial_potential(
                lambda R: params.reference_charge * params.particle1_charge / R
            ),
        ),
    )

    grid_e = coord_e.grid.build()
    grid_n = coord_n.grid.build()
    nuclear_mass = params.reference_mass + params.particle1_mass
    reference_scale = params.particle1_mass / nuclear_mass
    particle1_scale = params.reference_mass / nuclear_mass

    W_LM = coulomb_w_lm(
        grid_e.r,
        grid_n.r,
        L_max,
        M_max,
        centers=(
            CoulombCenter(
                params.particle2_charge * params.reference_charge,
                radial_scale=reference_scale,
                angular_parity=-1,
            ),
            CoulombCenter(
                params.particle2_charge * params.particle1_charge,
                radial_scale=particle1_scale,
                angular_parity=1,
            ),
        ),
        weights_R=grid_n.weights,
    )

    return TwoRadialSystemSpec(
        coord1=coord_e,
        coord2=coord_n,
        W_LM=W_LM,
        L_max=L_max,
        M_max=M_max,
        interaction_sign=1.0,
    )


def three_body_parameters(
    system: str, codata_set: str = "2022"
) -> ThreeBodyParameters:
    try:
        from matter_field_params.particle_params_manager import (
            get_reduced_masses_and_charges_3body,
            predefined_3body_systems,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "two_radial presets require matter_field_params to be importable"
        ) from exc

    particles = predefined_3body_systems(system)
    (
        mu_nuclear,
        mu_electronic,
        xi_nuclear,
        xi_electronic,
        reference_mass,
        particle1_mass,
        particle2_mass,
        reference_charge,
        particle1_charge,
        particle2_charge,
    ) = get_reduced_masses_and_charges_3body(
        particles=particles, codata_set=codata_set, frame="nuclear_center"
    )

    return ThreeBodyParameters(
        particles=particles,
        mu_nuclear=mu_nuclear,
        mu_electronic=mu_electronic,
        xi_nuclear=xi_nuclear,
        xi_electronic=xi_electronic,
        reference_mass=reference_mass,
        particle1_mass=particle1_mass,
        particle2_mass=particle2_mass,
        reference_charge=reference_charge,
        particle1_charge=particle1_charge,
        particle2_charge=particle2_charge,
    )
