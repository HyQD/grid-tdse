"""Estimate storage for one non-BO diatomic wavefunction.

For example::

    python examples/non_bo_diatomic_wavefunction_size.py --system H2+ --l-max 4

Use ``--M-tot none`` to estimate selected-basis Cartesian storage. The report
always includes fully unrestricted real and complex reference arrays as well.
"""

import argparse

from grid_tdse.spherical.two_radial import (
    FEMDVRGridSpec,
    get_size_NBO_Psi,
)


def _parse_M_tot(value: str) -> int | None:
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "M_tot must be an integer or 'none'"
        ) from exc


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--system", choices=("H2+", "HD+", "D2+"), default="H2+")
parser.add_argument("--l-max", type=int, default=2)
parser.add_argument(
    "--m-max",
    type=int,
    help="magnetic cutoff; default: l_max",
)
parser.add_argument(
    "--M-tot",
    type=_parse_M_tot,
    default=0,
    help="fixed total magnetic sector, or 'none' for Cartesian storage",
)
parser.add_argument(
    "--inversion-parity",
    type=int,
    choices=(-1, 1),
    default=1,
)
parser.add_argument(
    "--pair-exchange-parity",
    choices=("auto", "-1", "1"),
    default="auto",
    help="default: +1 for H2+/D2+ and unavailable for HD+",
)
parser.add_argument(
    "--no-symmetry-adaptation",
    action="store_false",
    dest="symmetry_adapted_basis",
    help="retain both coordinate parities in the field-free basis",
)
arguments = parser.parse_args()

if arguments.l_max < 0:
    parser.error("--l-max must be nonnegative")
if arguments.m_max is not None and arguments.m_max < 0:
    parser.error("--m-max must be nonnegative")

# The electron coordinate is measured from the nuclear center of mass.
electronic_grid = FEMDVRGridSpec(
    nodes=[0.0, 4.0, 8.0, 12.0, 16.0],
    n_points=[21, 21, 21, 21],
)

# A (nodes, points-per-element) tuple is accepted as a shorter alternative to
# constructing a second FEMDVRGridSpec explicitly.
internuclear_grid = ([0.0, 2.0, 4.0, 6.0, 8.0], 21)

pair_exchange_parity = (
    "auto"
    if arguments.pair_exchange_parity == "auto"
    else int(arguments.pair_exchange_parity)
)
sizes = get_size_NBO_Psi(
    arguments.system,
    electronic_grid,
    internuclear_grid,
    l_max=arguments.l_max,
    m_max=arguments.m_max,
    M_tot=arguments.M_tot,
    inversion_parity=arguments.inversion_parity,
    pair_exchange_parity=pair_exchange_parity,
    symmetry_adapted_basis=arguments.symmetry_adapted_basis,
)

print(sizes)
