"""
parameters.py
=============

Physical parameters of the one-dimensional diatomic harmonic chain.

Unit cell (lattice constant ``a``)::

        K2        K1        K2        K1        K2
    ~~~~~~~ (m1) ~~~~~~~ (m2) ~~~~~~~ (m1) ~~~~~~~ (m2) ~~~~~~~
             u_{1,n}       u_{2,n}      u_{1,n+1}     u_{2,n+1}
            |<------------ a ------------>|

* ``m1, m2`` : masses of the two atoms of the basis (sublattices s = 1, 2).
* ``K1``     : spring constant of the *intra-cell* bond  (1,n) -- (2,n).
* ``K2``     : spring constant of the *inter-cell* bond  (2,n) -- (1,n+1).
* ``a``      : lattice constant (length of the unit cell, two atoms per cell).

The textbook diatomic chain is the particular case ``K1 == K2 == K``.
Allowing ``K1 != K2`` costs nothing in the code and makes it possible to
discuss the general case of a 1D crystal with a two-atom basis (the gap can be
opened either by a mass contrast or by a stiffness contrast).

Units
-----
The whole package works in *reduced units* with

    hbar = k_B = 1 .

Therefore an angular frequency, an energy and a temperature are all measured
in the same unit (e.g. ``T = 0.5`` means ``k_B T = 0.5 hbar * omega_unit``).
Masses, spring constants and lengths are free numbers; the natural frequency
unit is ``sqrt(K/m)`` of whatever reference values you choose. The default
values (m1 = 1, m2 = 2, K = 1, a = 1) give frequencies of order one.

Data types
----------
* ``ChainParameters`` is an immutable ``dataclass`` (``frozen=True``): physical
  parameters must not change silently once a calculation has started, and a
  frozen dataclass is also hashable, which lets us use it as a cache key.
* All derived quantities are plain Python ``float``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class ChainParameters:
    """Immutable container for the parameters of the diatomic chain.

    Attributes
    ----------
    m1, m2 : float
        Masses of the two atoms of the basis (must be > 0).
    K1, K2 : float
        Spring constants of the intra-cell and inter-cell bonds (must be > 0).
    a : float
        Lattice constant (length of the two-atom unit cell, must be > 0).
    """

    m1: float = 1.0
    m2: float = 2.0
    K1: float = 1.0
    K2: float = 1.0
    a: float = 1.0

    # ------------------------------------------------------------------ #
    # Validation: a dataclass calls __post_init__ right after __init__.  #
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        for name in ("m1", "m2", "K1", "K2", "a"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value!r}")

    # ------------------------------------------------------------------ #
    # Convenience constructors                                           #
    # ------------------------------------------------------------------ #
    @classmethod
    def textbook(cls, mass_ratio: float = 2.0, K: float = 1.0,
                 m1: float = 1.0, a: float = 1.0) -> "ChainParameters":
        """Classic diatomic chain: equal springs, ``m2 = mass_ratio * m1``."""
        return cls(m1=m1, m2=mass_ratio * m1, K1=K, K2=K, a=a)

    def with_(self, **changes) -> "ChainParameters":
        """Return a copy with some fields changed (the original is frozen)."""
        return replace(self, **changes)

    # ------------------------------------------------------------------ #
    # Derived quantities (properties -> always consistent with fields)   #
    # ------------------------------------------------------------------ #
    @property
    def total_mass(self) -> float:
        """Mass of the unit cell, M = m1 + m2."""
        return self.m1 + self.m2

    @property
    def reduced_mass(self) -> float:
        """Reduced mass mu = m1 m2 / (m1 + m2)."""
        return self.m1 * self.m2 / (self.m1 + self.m2)

    @property
    def K_sum(self) -> float:
        """sigma = K1 + K2 (appears on the diagonal of the dynamical matrix)."""
        return self.K1 + self.K2

    @property
    def K_series(self) -> float:
        """Effective stiffness of the two springs in series, K1 K2 / (K1 + K2).

        It is the elastic constant of one unit cell seen as a single spring,
        i.e. the long-wavelength (continuum) limit of the chain.
        """
        return self.K1 * self.K2 / (self.K1 + self.K2)

    @property
    def sound_velocity(self) -> float:
        """Sound velocity v_s = lim_{k->0} omega_ac(k)/k.

        A cell of mass M behaves, at long wavelength, like a point mass joined
        to its neighbours by the series spring K_series:

            v_s = a * sqrt( K_series / M ).

        For K1 = K2 = K this reduces to the textbook a*sqrt(K / (2 (m1+m2))).
        """
        return self.a * math.sqrt(self.K_series / self.total_mass)

    @property
    def omega_max(self) -> float:
        """Top of the optical branch, reached at k = 0:  sqrt(sigma / mu)."""
        return math.sqrt(self.K_sum / self.reduced_mass)

    @property
    def zone_boundary_frequencies(self) -> tuple[float, float]:
        """(omega_-(pi/a), omega_+(pi/a)): edges of the gap.

        At the zone boundary the dynamical matrix has |D12| = |K1 - K2|/sqrt(m1 m2)
        and its two eigenvalues bound the forbidden band.
        """
        A = self.K_sum / self.m1
        C = self.K_sum / self.m2
        B2 = (self.K1 - self.K2) ** 2 / (self.m1 * self.m2)
        mean, half = 0.5 * (A + C), math.sqrt(0.25 * (A - C) ** 2 + B2)
        return math.sqrt(mean - half), math.sqrt(mean + half)

    @property
    def gap(self) -> float:
        """Width of the forbidden frequency band, omega_+(pi/a) - omega_-(pi/a)."""
        lo, hi = self.zone_boundary_frequencies
        return hi - lo

    @property
    def mean_square_frequency(self) -> float:
        """<omega^2> averaged over all 2N modes = Tr D(k)/2 = sigma / (2 mu).

        Trace of the dynamical matrix does not depend on k, so the second moment
        of the phonon DOS is exact and simple. It fixes the *high-temperature*
        expansion C/k_B = 1 - <(hbar omega)^2>/(12 (k_B T)^2) + ..., and hence the
        Einstein frequency that matches that expansion.
        """
        return self.K_sum / (2.0 * self.reduced_mass)

    def describe(self) -> str:
        """Human-readable summary (used by scripts and README examples)."""
        lo, hi = self.zone_boundary_frequencies
        return (
            f"m1={self.m1:g}, m2={self.m2:g}, K1={self.K1:g}, K2={self.K2:g}, a={self.a:g}\n"
            f"  sound velocity v_s      = {self.sound_velocity:.6f}\n"
            f"  acoustic band top       = {lo:.6f}\n"
            f"  optical band bottom     = {hi:.6f}\n"
            f"  optical band top        = {self.omega_max:.6f}\n"
            f"  gap                     = {self.gap:.6f}\n"
            f"  sqrt(<omega^2>)         = {math.sqrt(self.mean_square_frequency):.6f}"
        )
