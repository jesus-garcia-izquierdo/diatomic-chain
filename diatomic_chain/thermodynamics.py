"""
thermodynamics.py
=================

Equilibrium thermodynamics of the phonon gas of the diatomic chain in the
canonical ensemble (system in contact with a heat bath at temperature T).

In normal coordinates the harmonic crystal is an *ideal* (non-interacting)
system of 2N distinguishable quantum oscillators, so the partition function
factorises,

    Z(beta) = prod_{k, b} 1 / (2 sinh(beta hbar omega_b(k) / 2)),

and every macroscopic value is a sum of single-mode contributions. Per unit
cell, in the thermodynamic limit,

    X(T) = (a/2pi) int_BZ dk sum_b x(omega_b(k), T) = int_0^inf g(omega) x(omega, T) d omega,

with x one of the single-oscillator functions below (hbar = k_B = 1,
y = omega / T):

    energy           e = (omega/2) coth(y/2)                 (zero point included)
    heat capacity    c = (y/2)^2 / sinh^2(y/2)               ("Einstein function")
    free energy      f = T ln[2 sinh(y/2)]
    entropy          s = (y/2) coth(y/2) - ln[2 sinh(y/2)]
    occupation       n = 1 / (e^y - 1)                       (Bose-Einstein, mu = 0)

Numerical strategy
------------------
We integrate over **k**, not over omega: the integrand is smooth in k while
g(omega) has inverse-square-root van Hove singularities. Gauss-Legendre
quadrature on [0, pi/a] (the integrand is even in k) converges exponentially
fast. At low T only small k contribute to the acoustic branch, so the interval
is split at k_c ~ 40 T / v_s to put enough nodes where the integrand lives.

Data types
----------
* Temperatures: ``numpy.ndarray`` (float64). Scalars are promoted.
* Results: ``ThermoResult`` frozen dataclass of arrays with the same shape as
  ``T``; grouping them prevents mixing curves computed with different inputs.
* Gauss-Legendre nodes/weights are cached with ``functools.lru_cache``
  (key = number of nodes, an ``int``) because ``leggauss`` is called many times.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .parameters import ChainParameters
from .dispersion import omega as dispersion_omega


# ---------------------------------------------------------------------- #
# Single-oscillator functions (vectorised, numerically stable)           #
# ---------------------------------------------------------------------- #
def _y(w, T):
    """y = hbar omega / k_B T with broadcasting; T must be > 0."""
    return np.asarray(w, dtype=float) / np.asarray(T, dtype=float)


def mode_heat_capacity(w, T) -> np.ndarray:
    """c(y) = y^2 e^{-y} / (1 - e^{-y})^2, with c(0) = 1 (classical limit).

    Written with ``expm1`` so it neither overflows for y >> 1 nor loses
    precision for y << 1.
    """
    y = _y(w, T)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        c = y ** 2 * np.exp(-y) / np.expm1(-y) ** 2
    return np.where(y < 1e-8, 1.0, np.nan_to_num(c, nan=0.0))


def mode_energy(w, T) -> np.ndarray:
    """e = (omega/2) coth(y/2) = omega/2 + omega n_BE(y)."""
    w = np.asarray(w, dtype=float)
    y = _y(w, T)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        n = 1.0 / np.expm1(y)
        e = 0.5 * w + w * n
    # y -> 0: e -> T (equipartition); use it to avoid 0 * inf.
    return np.where(y < 1e-8, np.broadcast_to(T, e.shape), e)


def mode_free_energy(w, T) -> np.ndarray:
    """f = omega/2 + T ln(1 - e^{-y}) = T ln(2 sinh(y/2))."""
    w = np.asarray(w, dtype=float)
    T = np.asarray(T, dtype=float)
    y = _y(w, T)
    with np.errstate(divide="ignore"):
        return 0.5 * w + T * np.log(-np.expm1(-y))


def mode_entropy(w, T) -> np.ndarray:
    """s / k_B = y n_BE - ln(1 - e^{-y}) = (e - f)/T."""
    y = _y(w, T)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        s = y / np.expm1(y) - np.log(-np.expm1(-y))
    return np.nan_to_num(s, nan=0.0, posinf=np.inf)


def bose_einstein(w, T) -> np.ndarray:
    """Mean phonon occupation number n(omega, T) = 1/(e^{hbar omega / k_B T} - 1)."""
    y = _y(w, T)
    with np.errstate(over="ignore", divide="ignore"):
        return 1.0 / np.expm1(y)


# Classical (high-T, hbar -> 0) counterparts, for comparison.
def classical_mode_energy(w, T):
    return np.broadcast_to(np.asarray(T, dtype=float), np.shape(_y(w, T))).copy()


def classical_mode_heat_capacity(w, T):
    return np.ones_like(_y(w, T))


# ---------------------------------------------------------------------- #
# Quadrature over the Brillouin zone                                      #
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=16)
def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights on [-1, 1] (cached)."""
    return np.polynomial.legendre.leggauss(n)


def _k_nodes(params: ChainParameters, T: float, n_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes/weights on [0, pi/a], refined near k = 0 when T is small.

    Returns (k, weight) such that  (a/pi) sum weight * f(k) ~ (a/pi) int_0^{pi/a} f dk.
    """
    k_max = np.pi / params.a
    x, wq = _gauss_legendre(n_k)
    # Split point: beyond k_c the acoustic contribution is ~ e^{-40}.
    k_c = 40.0 * T / params.sound_velocity if T > 0 else k_max
    if 0.0 < k_c < 0.5 * k_max:
        segments = [(0.0, k_c), (k_c, k_max)]
    else:
        segments = [(0.0, k_max)]
    ks, ws = [], []
    for lo, hi in segments:
        ks.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * wq)
    return np.concatenate(ks), np.concatenate(ws)


def bz_average(params: ChainParameters, func, T, n_k: int = 1024,
               branch: int | None = None) -> np.ndarray:
    """Per-cell sum over modes:  (a/pi) int_0^{pi/a} dk  sum_b func(omega_b(k), T).

    ``func(w, T)`` must accept an array of frequencies and a scalar T.
    ``branch`` = None sums both branches; 0 (acoustic) or 1 (optical) keeps one.
    Loops over temperatures (each with its own adapted k-grid).
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    out = np.empty_like(T)
    for i, Ti in enumerate(T):
        k, wk = _k_nodes(params, Ti, n_k)
        w = dispersion_omega(params, k)                     # (n, 2)
        if branch is not None:
            w = w[..., branch:branch + 1]                   # keep shape (n, 1)
        vals = func(w, Ti).sum(axis=-1)                     # sum over branches
        out[i] = (params.a / np.pi) * np.sum(wk * vals)
    return out


# ---------------------------------------------------------------------- #
# Main API                                                                #
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThermoResult:
    """Thermodynamic functions per unit cell (hbar = k_B = 1)."""

    T: np.ndarray
    energy: np.ndarray           # U / N_cells
    heat_capacity: np.ndarray    # C_V / (N_cells k_B)
    free_energy: np.ndarray      # F / N_cells
    entropy: np.ndarray          # S / (N_cells k_B)


def phonon_thermodynamics(params: ChainParameters, T, quantum: bool = True,
                          n_k: int = 1024) -> ThermoResult:
    """U, C_V, F and S per cell in the thermodynamic limit.

    Parameters
    ----------
    quantum : bool
        True  -> quantum harmonic oscillators (Bose-Einstein phonons).
        False -> classical oscillators (equipartition: C = 2 k_B per cell).
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    if np.any(T <= 0):
        raise ValueError("Temperatures must be > 0 (use zero_point_energy for T = 0).")
    if quantum:
        U = bz_average(params, mode_energy, T, n_k)
        C = bz_average(params, mode_heat_capacity, T, n_k)
        F = bz_average(params, mode_free_energy, T, n_k)
    else:
        U = 2.0 * T
        C = np.full_like(T, 2.0)
        # Classical free energy of an oscillator: T ln(hbar omega / k_B T).
        F = bz_average(params, lambda w, t: t * np.log(np.where(w > 0, w, 1.0) / t), T, n_k)
    S = (U - F) / T
    return ThermoResult(T=T, energy=U, heat_capacity=C, free_energy=F, entropy=S)


def heat_capacity(params: ChainParameters, T, n_k: int = 1024) -> np.ndarray:
    """Quantum C_V per cell (in units of k_B), thermodynamic limit."""
    return bz_average(params, mode_heat_capacity, T, n_k)


def branch_heat_capacity(params: ChainParameters, T, n_k: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Acoustic and optical contributions to C_V per cell: (C_ac, C_op).

    Each saturates at 1 k_B per cell; their sum is ``heat_capacity``. The
    optical part is exponentially "frozen" for T << omega_o (bottom of the
    optical band), which is precisely the regime where Einstein's model applies.
    """
    return (bz_average(params, mode_heat_capacity, T, n_k, branch=0),
            bz_average(params, mode_heat_capacity, T, n_k, branch=1))


def zero_point_energy(params: ChainParameters, n_k: int = 1024) -> float:
    """Zero-point energy per cell, E_0/N = (a/pi) int_0^{pi/a} dk sum_b hbar omega_b(k)/2."""
    x, wq = _gauss_legendre(n_k)
    k_max = np.pi / params.a
    k = 0.5 * k_max * (x + 1.0)
    w = dispersion_omega(params, k).sum(axis=-1)
    return float((params.a / np.pi) * np.sum(0.5 * k_max * wq * 0.5 * w))


def finite_chain_heat_capacity(frequencies, n_cells: int, T) -> np.ndarray:
    """Exact C_V per cell of a *finite* chain from its list of 2N frequencies.

    Zero modes (rigid translation, omega = 0) are excluded: a free translation
    is not an oscillator (its classical contribution, 1/2 k_B, is O(1/N)).
    """
    w = np.asarray(frequencies, dtype=float)
    w = w[w > 1e-10]
    T = np.atleast_1d(np.asarray(T, dtype=float))
    return np.array([mode_heat_capacity(w, t).sum() for t in T]) / n_cells


# ---------------------------------------------------------------------- #
# Asymptotic laws                                                         #
# ---------------------------------------------------------------------- #
def low_temperature_heat_capacity(params: ChainParameters, T) -> np.ndarray:
    """C/(N k_B) -> (pi/3) a k_B T / (hbar v_s)   for T -> 0  (1D 'Debye T law').

    Only the acoustic branch near k = 0 is excited; its DOS is flat,
    g(0) = a/(pi v_s), and int_0^inf y^2 e^y/(e^y-1)^2 dy = pi^2/3.
    """
    T = np.asarray(T, dtype=float)
    return (np.pi / 3.0) * params.a * T / params.sound_velocity


def high_temperature_heat_capacity(params: ChainParameters, T) -> np.ndarray:
    """C/(N k_B) = 2 [1 - <omega^2>/(12 T^2) + O(T^-4)] (approach to Dulong-Petit).

    Uses c(y) = 1 - y^2/12 + ... and the exact second moment of the DOS.
    """
    T = np.asarray(T, dtype=float)
    return 2.0 * (1.0 - params.mean_square_frequency / (12.0 * T ** 2))


def energy_fluctuation(params: ChainParameters, T, n_cells: int) -> np.ndarray:
    """Relative energy fluctuation Delta E / <E> of a chain of N cells.

    Fluctuation-response relation of the canonical ensemble:
    (Delta E)^2 = k_B T^2 C_V, extensive like C_V, so
    Delta E / <E> = sqrt(T^2 N c) / (N u) ~ N^{-1/2} -> 0 in the thermodynamic
    limit (canonical and microcanonical ensembles become equivalent).
    """
    th = phonon_thermodynamics(params, T)
    return np.sqrt(th.T ** 2 * n_cells * th.heat_capacity) / (n_cells * th.energy)
