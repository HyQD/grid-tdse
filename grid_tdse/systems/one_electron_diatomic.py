"""Clamped two-centre, one-electron physical systems."""

from dataclasses import dataclass, field
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


def _position_vector(value, name: str) -> tuple[float, float, float]:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise TypeError(f"{name} must contain real numbers")
    result = np.array(raw, dtype=float, copy=True)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return tuple(float(component) for component in result)


def _optional_name(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("name must be a string or None")
    if not value.strip():
        raise ValueError("name must not be empty")
    return value


@dataclass(frozen=True, init=False)
class OneElectronDiatomicSystem:
    r"""Clamped two-centre system containing one dynamical electron.

    The legacy constructor fixes the nuclei at the bond-midpoint coordinates

    .. math::

        \mathbf R_1=(0,0,-a),\qquad \mathbf R_2=(0,0,a),

    and the electronic Hamiltonian to be constructed from this description is

    .. math::

        H=-\frac{1}{2\mu}\nabla^2
        -\frac{Z_1}{|\mathbf r-\mathbf R_1|}
        -\frac{Z_2}{|\mathbf r-\mathbf R_2|}.

    :meth:`from_nuclear_positions` instead accepts two explicit Cartesian
    positions. The nuclear-repulsion energy is provided separately and is
    not part of the electronic Hamiltonian. This class contains no
    coordinate-grid or wavefunction-representation policy.

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
    _nuclear_position1: tuple[float, float, float] = field(
        repr=False,
    )
    _nuclear_position2: tuple[float, float, float] = field(
        repr=False,
    )

    def __init__(
        self,
        nuclear_charge1,
        nuclear_charge2,
        half_bond_length,
        particle_mass=1.0,
        name=None,
    ) -> None:
        half_bond_length = _positive_real(
            half_bond_length,
            "half_bond_length",
        )
        self._initialize(
            nuclear_charge1=nuclear_charge1,
            nuclear_charge2=nuclear_charge2,
            half_bond_length=half_bond_length,
            particle_mass=particle_mass,
            name=name,
            nuclear_position1=(0.0, 0.0, -half_bond_length),
            nuclear_position2=(0.0, 0.0, half_bond_length),
        )

    @classmethod
    def from_nuclear_positions(
        cls,
        nuclear_charge1,
        nuclear_charge2,
        nuclear_position1,
        nuclear_position2,
        particle_mass=1.0,
        name=None,
    ):
        """Construct a system from two Cartesian nuclear positions.

        Positions are expressed in bohr in the same fixed Cartesian frame as
        the spatial representation and external fields. The position order
        matches the order of ``nuclear_charge1`` and ``nuclear_charge2``.
        """

        position1 = _position_vector(
            nuclear_position1,
            "nuclear_position1",
        )
        position2 = _position_vector(
            nuclear_position2,
            "nuclear_position2",
        )
        distance = float(
            np.linalg.norm(np.subtract(position2, position1))
        )
        if distance == 0.0:
            raise ValueError("nuclear positions must be distinct")
        result = object.__new__(cls)
        result._initialize(
            nuclear_charge1=nuclear_charge1,
            nuclear_charge2=nuclear_charge2,
            half_bond_length=0.5 * distance,
            particle_mass=particle_mass,
            name=name,
            nuclear_position1=position1,
            nuclear_position2=position2,
        )
        return result

    def _initialize(
        self,
        *,
        nuclear_charge1,
        nuclear_charge2,
        half_bond_length,
        particle_mass,
        name,
        nuclear_position1,
        nuclear_position2,
    ) -> None:
        object.__setattr__(
            self,
            "nuclear_charge1",
            _positive_real(nuclear_charge1, "nuclear_charge1"),
        )
        object.__setattr__(
            self,
            "nuclear_charge2",
            _positive_real(nuclear_charge2, "nuclear_charge2"),
        )
        object.__setattr__(
            self,
            "half_bond_length",
            _positive_real(half_bond_length, "half_bond_length"),
        )
        object.__setattr__(
            self,
            "particle_mass",
            _positive_real(particle_mass, "particle_mass"),
        )
        object.__setattr__(self, "name", _optional_name(name))
        object.__setattr__(
            self,
            "_nuclear_position1",
            _position_vector(nuclear_position1, "nuclear_position1"),
        )
        object.__setattr__(
            self,
            "_nuclear_position2",
            _position_vector(nuclear_position2, "nuclear_position2"),
        )

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
        """Return ``Z_1*Z_2/R`` in atomic units."""

        return (
            self.nuclear_charge1
            * self.nuclear_charge2
            / self.internuclear_distance
        )

    @property
    def nuclear_positions(self) -> np.ndarray:
        """Return Cartesian nuclear positions with shape ``(2, 3)``."""

        positions = np.array(
            [self._nuclear_position1, self._nuclear_position2],
        )
        positions.setflags(write=False)
        return positions

    @property
    def nuclear_position1(self) -> np.ndarray:
        """Return the first Cartesian nuclear position."""

        position = np.asarray(self._nuclear_position1).copy()
        position.setflags(write=False)
        return position

    @property
    def nuclear_position2(self) -> np.ndarray:
        """Return the second Cartesian nuclear position."""

        position = np.asarray(self._nuclear_position2).copy()
        position.setflags(write=False)
        return position

    @property
    def nuclear_midpoint(self) -> np.ndarray:
        """Return the geometric midpoint of the nuclear positions."""

        midpoint = 0.5 * (
            self.nuclear_position1 + self.nuclear_position2
        )
        midpoint.setflags(write=False)
        return midpoint

    @property
    def bond_axis(self) -> np.ndarray:
        """Return the unit vector from nucleus one to nucleus two."""

        axis = (
            self.nuclear_position2 - self.nuclear_position1
        ) / self.internuclear_distance
        axis.setflags(write=False)
        return axis

    @property
    def is_midpoint_centered(self) -> bool:
        """Return whether the nuclear midpoint is the Cartesian origin."""

        scale = max(1.0, self.half_bond_length)
        tolerance = 128.0 * np.finfo(float).eps * scale
        return bool(np.linalg.norm(self.nuclear_midpoint) <= tolerance)

    @property
    def is_z_axis_aligned(self) -> bool:
        """Return whether both nuclei lie on the Cartesian ``z`` axis."""

        scale = max(
            1.0,
            float(np.max(np.abs(self.nuclear_positions))),
        )
        tolerance = 128.0 * np.finfo(float).eps * scale
        return bool(
            np.max(np.abs(self.nuclear_positions[:, :2])) <= tolerance
        )

    @property
    def is_canonical_axial_geometry(self) -> bool:
        """Return whether positions are ``(0,0,-a)`` and ``(0,0,a)``."""

        a = self.half_bond_length
        expected = np.array([[0.0, 0.0, -a], [0.0, 0.0, a]])
        scale = max(1.0, a)
        tolerance = 128.0 * np.finfo(float).eps * scale
        return bool(
            np.allclose(
                self.nuclear_positions,
                expected,
                rtol=0.0,
                atol=tolerance,
            )
        )

    @property
    def nuclear_charges(self) -> np.ndarray:
        """Return the two positive charges in position order."""

        charges = np.array([self.nuclear_charge1, self.nuclear_charge2])
        charges.setflags(write=False)
        return charges

    @property
    def nuclear_dipole(self) -> np.ndarray:
        """Return the static nuclear dipole about the bond midpoint."""

        return self.nuclear_dipole_about(np.zeros(3))

    def nuclear_dipole_about(self, origin) -> np.ndarray:
        """Return the static nuclear dipole about a Cartesian origin.

        Shifting the origin from zero to ``origin`` changes the nuclear
        contribution by ``-(Z1 + Z2) * origin``. The returned array is a
        read-only copy.
        """

        origin = np.asarray(origin)
        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")
        if not np.issubdtype(origin.dtype, np.number) or np.iscomplexobj(
            origin
        ):
            raise TypeError("origin must contain real numbers")
        if not np.all(np.isfinite(origin)):
            raise ValueError("origin must contain finite values")
        dipole = np.sum(
            self.nuclear_charges[:, None]
            * (self.nuclear_positions - origin.astype(float)),
            axis=0,
        )
        dipole.setflags(write=False)
        return dipole
