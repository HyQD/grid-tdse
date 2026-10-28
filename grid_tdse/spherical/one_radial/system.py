"""Physical systems for one-radial-coordinate spherical calculations."""

from dataclasses import dataclass
from numbers import Real

import numpy as np


def _positive_real(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class OneElectronDiatomicSystem:
    r"""Clamped two-centre system containing one dynamical electron.

    The nuclei are fixed at the bond-midpoint coordinates

    .. math::

        \mathbf R_1=(0,0,-a),\qquad \mathbf R_2=(0,0,a),

    and the electronic Hamiltonian to be constructed from this description is

    .. math::

        H=-\frac{1}{2\mu}\nabla^2
        -\frac{Z_1}{|\mathbf r-\mathbf R_1|}
        -\frac{Z_2}{|\mathbf r-\mathbf R_2|}.

    The nuclear-repulsion energy is provided separately and is not part of
    the electronic Hamiltonian.

    Parameters
    ----------
    nuclear_charge1, nuclear_charge2 : float
        Positive nuclear charges ``Z_1`` and ``Z_2``. Noninteger effective
        charges are allowed.
    half_bond_length : float
        Positive half bond length ``a``. The internuclear distance is ``2*a``.
    particle_mass : float, optional
        Positive electronic or reduced mass ``mu`` in atomic units. Defaults
        to one.
    name : str, optional
        User-facing system label, for example ``"H2+"``.
    """

    nuclear_charge1: float
    nuclear_charge2: float
    half_bond_length: float
    particle_mass: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nuclear_charge1",
            _positive_real(self.nuclear_charge1, "nuclear_charge1"),
        )
        object.__setattr__(
            self,
            "nuclear_charge2",
            _positive_real(self.nuclear_charge2, "nuclear_charge2"),
        )
        object.__setattr__(
            self,
            "half_bond_length",
            _positive_real(self.half_bond_length, "half_bond_length"),
        )
        object.__setattr__(
            self,
            "particle_mass",
            _positive_real(self.particle_mass, "particle_mass"),
        )
        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError("name must be a string or None")
            if not self.name.strip():
                raise ValueError("name must not be empty")

    @property
    def electron_count(self) -> int:
        """Return the fixed number of dynamical electrons."""

        return 1

    @property
    def electron_charge(self) -> float:
        """Return the electron charge in atomic units."""

        return -1.0

    @property
    def internuclear_distance(self) -> float:
        """Return the distance between the clamped nuclei."""

        return 2.0 * self.half_bond_length

    @property
    def net_charge(self) -> float:
        """Return the total charge of the one-electron system."""

        return self.nuclear_charge1 + self.nuclear_charge2 - 1.0

    @property
    def nuclear_repulsion_energy(self) -> float:
        """Return ``Z_1*Z_2/(2*a)`` in atomic units."""

        return (
            self.nuclear_charge1
            * self.nuclear_charge2
            / self.internuclear_distance
        )

    @property
    def nuclear_positions(self) -> np.ndarray:
        """Return Cartesian nuclear positions with shape ``(2, 3)``."""

        a = self.half_bond_length
        positions = np.array([[0.0, 0.0, -a], [0.0, 0.0, a]])
        positions.setflags(write=False)
        return positions

    @property
    def nuclear_charges(self) -> np.ndarray:
        """Return the two positive charges in position order."""

        charges = np.array([self.nuclear_charge1, self.nuclear_charge2])
        charges.setflags(write=False)
        return charges

    @property
    def nuclear_dipole(self) -> np.ndarray:
        """Return the static nuclear dipole about the bond midpoint."""

        dipole = np.array(
            [
                0.0,
                0.0,
                self.half_bond_length
                * (self.nuclear_charge2 - self.nuclear_charge1),
            ]
        )
        dipole.setflags(write=False)
        return dipole
