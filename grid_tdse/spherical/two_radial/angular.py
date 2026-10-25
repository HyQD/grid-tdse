"""Explicit angular state tables for two-radial spherical calculations."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal


Parity = Literal[-1, 1]


@dataclass(frozen=True)
class AngularBasisSpec:
    """Uncoupled ``(l, m)`` basis for one radial coordinate.

    ``coordinate_parity=1`` retains even ``l`` and ``-1`` retains odd ``l``.
    This is a basis reduction, not a total-angular-momentum quantum number.
    """

    l_max: int
    m_max: int | None = None
    coordinate_parity: Parity | None = None
    states: tuple[tuple[int, int], ...] = field(init=False)
    _indices: dict[tuple[int, int], int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        m_max = self.l_max if self.m_max is None else self.m_max
        if self.l_max < 0:
            raise ValueError("l_max must be non-negative")
        if not 0 <= m_max <= self.l_max:
            raise ValueError("m_max must satisfy 0 <= m_max <= l_max")
        if self.coordinate_parity not in (None, -1, 1):
            raise ValueError("coordinate_parity must be None, -1, or 1")

        states = tuple(
            (l, m)
            for m in range(-m_max, m_max + 1)
            for l in range(abs(m), self.l_max + 1)
            if self.coordinate_parity is None
            or (-1 if l % 2 else 1) == self.coordinate_parity
        )
        if not states:
            raise ValueError("angular parity selection leaves no basis states")
        object.__setattr__(self, "m_max", m_max)
        object.__setattr__(self, "states", states)
        indices = {state: i for i, state in enumerate(states)}
        object.__setattr__(self, "_indices", indices)

    @property
    def n_channels(self) -> int:
        return len(self.states)

    def index(self, l: int, m: int) -> int:
        try:
            return self._indices[(l, m)]
        except KeyError as exc:
            raise ValueError(f"angular state {(l, m)} is not in this basis") from exc


@dataclass(frozen=True)
class AngularPairBasis:
    """Ordered product channels, optionally restricted to a fixed ``M_tot``.

    ``M_tot=None`` retains the complete Cartesian product and therefore the
    legacy four-dimensional state representation.  An integer ``M_tot``
    retains only channel pairs satisfying ``m1 + m2 == M_tot`` and stores the
    angular pair on one compact state axis.
    """

    basis1: AngularBasisSpec
    basis2: AngularBasisSpec
    M_tot: int | None = None
    pairs: tuple[tuple[int, int], ...] = field(init=False)
    _indices: dict[tuple[int, int], int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.M_tot is not None:
            if isinstance(self.M_tot, bool) or not isinstance(self.M_tot, Integral):
                raise TypeError("M_tot must be an integer or None")
            object.__setattr__(self, "M_tot", int(self.M_tot))

        pairs = tuple(
            (i, j)
            for i, (_, m1) in enumerate(self.basis1.states)
            for j, (_, m2) in enumerate(self.basis2.states)
            if self.M_tot is None or m1 + m2 == self.M_tot
        )
        if not pairs:
            raise ValueError("M_tot leaves no angular-channel pairs")
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "_indices", {pair: p for p, pair in enumerate(pairs)})

    @property
    def is_compact(self) -> bool:
        return self.M_tot is not None

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def index(self, channel1: int, channel2: int) -> int:
        try:
            return self._indices[(channel1, channel2)]
        except KeyError as exc:
            raise ValueError(
                f"angular channel pair {(channel1, channel2)} is not retained"
            ) from exc

    def get(self, channel1: int, channel2: int) -> int | None:
        """Return the compact index, or ``None`` when the pair is absent."""

        return self._indices.get((channel1, channel2))


@dataclass(frozen=True)
class CoulombMultipoleSpec:
    """Truncation of the Coulomb expansion in ``(lambda, mu)``."""

    lambda_max: int
    mu_max: int | None = None
    lambda_parity: Parity | None = None
    components: tuple[tuple[int, int], ...] = field(init=False)
    lambdas: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        mu_max = self.lambda_max if self.mu_max is None else self.mu_max
        if self.lambda_max < 0:
            raise ValueError("lambda_max must be non-negative")
        if not 0 <= mu_max <= self.lambda_max:
            raise ValueError("mu_max must satisfy 0 <= mu_max <= lambda_max")
        if self.lambda_parity not in (None, -1, 1):
            raise ValueError("lambda_parity must be None, -1, or 1")

        lambdas = tuple(
            value
            for value in range(self.lambda_max + 1)
            if self.lambda_parity is None
            or (-1 if value % 2 else 1) == self.lambda_parity
        )
        if not lambdas:
            raise ValueError("lambda parity selection leaves no Coulomb multipoles")
        components = tuple(
            (lam, mu)
            for mu in range(-mu_max, mu_max + 1)
            for lam in lambdas
            if abs(mu) <= lam
        )
        object.__setattr__(self, "mu_max", mu_max)
        object.__setattr__(self, "lambdas", lambdas)
        object.__setattr__(self, "components", components)

    def radial_index(self, lambda_: int) -> int:
        try:
            return self.lambdas.index(lambda_)
        except ValueError as exc:
            raise ValueError(f"lambda={lambda_} is not retained") from exc
