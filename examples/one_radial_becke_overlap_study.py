"""Run the Phase 6.3 two-center Becke overlap conditioning gate."""

from grid_tdse.spherical.one_radial_becke import (
    run_two_center_overlap_study,
)


def _print_sweep(title, parameter, results, value):
    print(f"\n{title}")
    print(
        f"{parameter:>22s} {'N':>5s} {'s_min':>12s} {'s_max':>12s} "
        f"{'cond(S)':>12s} {'rank 1e-8/1e-10/1e-12':>24s}"
    )
    for result in results:
        ranks = "/".join(
            str(decision.retained_rank)
            for decision in result.rank_decisions
        )
        print(
            f"{value(result):>22s} {result.basis_size:5d} "
            f"{result.minimum_eigenvalue:12.4e} "
            f"{result.maximum_eigenvalue:12.4e} "
            f"{result.condition_number:12.4e} {ranks:>24s}"
        )


def main():
    """Print all one-parameter-at-a-time overlap sweeps."""

    study = run_two_center_overlap_study()
    baseline = study.baseline
    print("\n** one_radial_becke two-center overlap study **")
    print(
        "Baseline: "
        f"R={baseline.internuclear_distance:g}, "
        f"nodes={baseline.radial_nodes}, "
        f"orders={baseline.radial_n_points}, "
        f"l_max={baseline.l_max}, m={baseline.m}, "
        f"Lebedev={baseline.lebedev_order}, "
        f"Becke iterations={baseline.orbital_switch_iterations}"
    )
    print(
        "Each atomic integration grid reaches r_max + R, so both cells "
        "cover the complete union of the finite basis supports."
    )

    _print_sweep(
        "Internuclear distance",
        "R",
        study.internuclear_distance,
        lambda result: f"{result.internuclear_distance:g}",
    )
    _print_sweep(
        "Radial extent",
        "r_max",
        study.radial_extent,
        lambda result: f"{result.radial_nodes[-1]:g}",
    )
    _print_sweep(
        "Radial polynomial order",
        "points / element",
        study.radial_order,
        lambda result: str(result.radial_n_points[0]),
    )
    _print_sweep(
        "Finite-element layout",
        "radial nodes",
        study.element_layout,
        lambda result: str(result.radial_nodes),
    )
    _print_sweep(
        "Angular cutoff",
        "l_max",
        study.angular_cutoff,
        lambda result: str(result.l_max),
    )
    _print_sweep(
        "Lebedev order at baseline l_max",
        "Lebedev order",
        study.lebedev_order,
        lambda result: str(result.lebedev_order),
    )
    _print_sweep(
        "Lebedev refinement at l_max=11",
        "Lebedev order",
        study.high_l_lebedev_order,
        lambda result: str(result.lebedev_order),
    )
    _print_sweep(
        "Becke smoothing order",
        "iterations",
        study.becke_iterations,
        lambda result: str(result.orbital_switch_iterations),
    )

    high_l = study.high_l_lebedev_order[-1]
    print("\nFeasibility assessment")
    if high_l.rank_decisions[0].discarded_rank:
        print(
            "The two-center representation is usable at modest angular "
            "cutoff, but it develops cutoff-relevant near-linear dependence: "
            f"at l_max={high_l.l_max}, the 1e-8 cutoff discards "
            f"{high_l.rank_decisions[0].discarded_rank} of "
            f"{high_l.basis_size} directions."
        )
    else:
        print(
            "No overlap direction in this study falls below the 1e-8 "
            "relative cutoff."
        )
    print(
        "The next Hamiltonian phases must therefore continue to report "
        "overlap ranks and test energy stability at 1e-8, 1e-10, and 1e-12."
    )


if __name__ == "__main__":
    main()
