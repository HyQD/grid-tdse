"""Quick midpoint-versus-two-center Becke comparison for H2+.

The presets are deliberately small enough for an interactive development
loop.  ``--level all`` shows a paired refinement sequence; it is not intended
to replace the more expensive convergence study in
``one_radial_becke_convergence_study.py``.
"""

import argparse
from dataclasses import dataclass
from time import perf_counter

from grid_tdse.spherical.one_radial_becke import (
    KineticIntegrationSettings,
    validate_h2plus_ground_state,
)


@dataclass(frozen=True)
class ComparisonPreset:
    """Parameters refined together for a short exploratory calculation."""

    radial_nodes: tuple[float, ...]
    radial_n_points: int
    l_max: int
    integration: KineticIntegrationSettings
    midpoint_radial_spacing: float
    midpoint_radial_n_points: int
    midpoint_l_max: int


PRESETS = {
    "smoke": ComparisonPreset(
        radial_nodes=(0.0, 2.0, 4.0),
        radial_n_points=3,
        l_max=0,
        integration=KineticIntegrationSettings(2.0, 3, 5),
        midpoint_radial_spacing=2.0,
        midpoint_radial_n_points=4,
        midpoint_l_max=2,
    ),
    "quick": ComparisonPreset(
        radial_nodes=(0.0, 2.0, 4.0),
        radial_n_points=4,
        l_max=1,
        integration=KineticIntegrationSettings(1.0, 4, 9),
        midpoint_radial_spacing=1.0,
        midpoint_radial_n_points=6,
        midpoint_l_max=4,
    ),
    "refined": ComparisonPreset(
        radial_nodes=(0.0, 1.0, 2.0, 3.0, 4.0),
        radial_n_points=5,
        l_max=2,
        integration=KineticIntegrationSettings(0.5, 6, 17),
        midpoint_radial_spacing=0.5,
        midpoint_radial_n_points=8,
        midpoint_l_max=6,
    ),
}


def run_preset(name: str, preset: ComparisonPreset):
    """Run one cached Becke assembly and its midpoint comparison."""

    started = perf_counter()
    result = validate_h2plus_ground_state(
        radial_nodes=preset.radial_nodes,
        radial_n_points=preset.radial_n_points,
        l_max=preset.l_max,
        overlap_cutoffs=(1.0e-10,),
        coarse_settings=preset.integration,
        refined_settings=preset.integration,
        reference_settings=preset.integration,
        midpoint_radial_spacing=preset.midpoint_radial_spacing,
        midpoint_radial_n_points=preset.midpoint_radial_n_points,
        midpoint_l_max=preset.midpoint_l_max,
    )
    elapsed = perf_counter() - started
    becke = result.reference
    return {
        "name": name,
        "elapsed": elapsed,
        "midpoint_energy": result.midpoint.electronic_energy,
        "becke_energy": becke.electronic_energies[0],
        "difference": result.becke_to_midpoint_energy_error,
        "midpoint_size": result.midpoint.basis_size,
        "becke_size": becke.matrices.basis_size,
        "quadrature_size": becke.matrices.quadrature_size,
        "condition_number": becke.matrices.overlap_condition_number,
        "retained_rank": becke.retained_ranks[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        choices=(*PRESETS, "all"),
        default="quick",
        help="paired refinement level (default: quick)",
    )
    arguments = parser.parse_args()
    names = tuple(PRESETS) if arguments.level == "all" else (arguments.level,)
    rows = [run_preset(name, PRESETS[name]) for name in names]

    print("\nH2+ at R=2 bohr; electronic energies in Eh")
    print(
        f"{'level':<8} {'midpoint E':>16} {'Becke E':>16} "
        f"{'|difference|':>13} {'N(mid/Becke)':>14} {'rank':>7} "
        f"{'cond(S)':>11} {'points':>8} {'time/s':>8}"
    )
    for row in rows:
        dimensions = f"{row['midpoint_size']}/{row['becke_size']}"
        rank = f"{row['retained_rank']}/{row['becke_size']}"
        print(
            f"{row['name']:<8} {row['midpoint_energy']:16.10f} "
            f"{row['becke_energy']:16.10f} {row['difference']:13.3e} "
            f"{dimensions:>14} {rank:>7} "
            f"{row['condition_number']:11.3e} "
            f"{row['quadrature_size']:8d} {row['elapsed']:8.3f}"
        )

    print(
        "\nThe two columns use different finite-domain shapes. Treat their "
        "difference as a development diagnostic, not a strict error bound."
    )


if __name__ == "__main__":
    main()
