"""
observables.py
==============

Thermal fluctuations of the atomic displacements: "other observables" beyond
the heat capacity.

Normal-mode expansion (N cells, Born-von Karman)::

    u_{s,n} = N^{-1/2} sum_{k,b} [e_s(k,b) / sqrt(m_s)] e^{i k n a} Q_{k,b},

where each normal coordinate Q_{k,b} is a unit-mass oscillator of frequency
omega_b(k). In the canonical ensemble

    <|Q_{k,b}|^2> = (hbar / 2 omega) coth(hbar omega / 2 k_B T)   (quantum)
                  = k_B T / omega^2                              (classical).

Observables implemented
-----------------------
1. ``displacement_correlation``: G_{ss'}(r) = <(u_{s',n+r} - u_{s,n})^2>,
   the mean-square *relative* displacement of two atoms r cells apart.
   It is finite in the thermodynamic limit and grows like r (T > 0) or like
   ln r (T = 0): a 1D crystal has no true long-range positional order
   (Landau-Peierls). Classically, for nearest-neighbour springs the bond
   stretches are independent, so G_{11}(r) = r k_B T (1/K1 + 1/K2) exactly.
2. ``mean_square_displacement``: <u_s^2> of a *finite* periodic chain
   (the k = 0 translation excluded). It diverges with N (~N at T > 0,
   ~ln N at T = 0), i.e. the Debye-Waller factor exp(-q^2 <u^2>) of a 1D
   crystal vanishes in the thermodynamic limit: no sharp Bragg peaks.
3. ``bond_fluctuations``: rms stretch of the K1 and K2 bonds (r = 0 and r = 1
   special cases of item 1), the quantity entering a Lindemann-type criterion.

Data types
----------
Inputs are floats / ints; outputs ``numpy.ndarray`` (float64). The polarisation
vectors are obtained by vectorised ``numpy.linalg.eigh`` on an array of 2x2
matrices (complex128).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .parameters import ChainParameters
from .dispersion import normal_modes, allowed_wavevectors


@lru_cache(maxsize=8)
def _gl(order: int):
    return np.polynomial.legendre.leggauss(order)


def _composite_nodes(k_max: float, n_panels: int, order: int = 48):
    """Composite Gauss-Legendre rule on (-k_max, k_max).

    A single high-order rule would need O(r) nodes for the oscillating factor
    e^{ikra}, and ``leggauss`` costs O(n^3); panels of fixed order keep the cost
    linear in r while resolving each oscillation with ~order/2 nodes.
    """
    x, w = _gl(order)
    edges = np.linspace(-k_max, k_max, n_panels + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    k = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    wk = (half[:, None] * w[None, :]).ravel()
    return k, wk


def _mode_amplitude(w: np.ndarray, T: float, quantum: bool) -> np.ndarray:
    """<|Q|^2> of a unit-mass oscillator; returns +inf for w = 0."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if quantum:
            if T <= 0.0:
                return 0.5 / w                                   # zero point only
            return 0.5 / w / np.tanh(0.5 * w / T)
        return T / w ** 2


def _projection_weight(params: ChainParameters, k: np.ndarray, e: np.ndarray,
                       r: int, s: int, s_prime: int) -> np.ndarray:
    """|e_{s'} e^{ikra}/sqrt(m_{s'}) - e_s/sqrt(m_s)|^2 for each k and branch."""
    m = np.array([params.m1, params.m2])
    phase = np.exp(1j * k * r * params.a)[:, None]               # (n_k, 1)
    amp = e[:, s_prime, :] * phase / np.sqrt(m[s_prime]) - e[:, s, :] / np.sqrt(m[s])
    return np.abs(amp) ** 2                                      # (n_k, 2)


def displacement_correlation(params: ChainParameters, r, T: float,
                             s: int = 0, s_prime: int = 0,
                             quantum: bool = True, n_panels: int | None = None) -> np.ndarray:
    """G_{ss'}(r) = <(u_{s',n+r} - u_{s,n})^2> in the thermodynamic limit.

    Parameters
    ----------
    r : int or array of int
        Separation in unit cells.
    s, s_prime : {0, 1}
        Sublattices (0 -> m1, 1 -> m2).
    quantum : bool
        Quantum (coth) or classical (k_B T / omega^2) mode amplitudes.
    n_panels : int, optional
        Number of 48-point Gauss-Legendre panels on [-pi/a, pi/a]; by default
        it grows with r to resolve the oscillating factor e^{ikra}.
    """
    r_arr = np.atleast_1d(np.asarray(r, dtype=int))
    out = np.empty(r_arr.shape, dtype=float)
    k_max = np.pi / params.a
    for i, ri in enumerate(r_arr):
        npan = n_panels if n_panels is not None else 32 + abs(int(ri))
        k, wq = _composite_nodes(k_max, 2 * (npan // 2))         # even -> k=0 is a panel edge
        w, e = normal_modes(params, k)                           # (nk,2), (nk,2,2)
        amp = _mode_amplitude(w, T, quantum)
        weight = _projection_weight(params, k, e, int(ri), s, s_prime)
        integrand = np.nan_to_num(weight * amp, nan=0.0, posinf=0.0).sum(axis=-1)
        out[i] = (params.a / (2.0 * np.pi)) * np.sum(wq * integrand)
    return out


def bond_fluctuations(params: ChainParameters, T: float,
                      quantum: bool = True) -> tuple[float, float]:
    """Mean-square stretch of the intra-cell (K1) and inter-cell (K2) bonds.

    intra: <(u_{2,n} - u_{1,n})^2>   (r = 0, s = 0 -> s' = 1)
    inter: <(u_{1,n+1} - u_{2,n})^2> (r = 1, s = 1 -> s' = 0)
    Classically they are exactly k_B T / K1 and k_B T / K2 (equipartition
    applied to each spring, because the bond stretches of an infinite chain
    are independent Gaussian variables).
    """
    intra = displacement_correlation(params, 0, T, s=0, s_prime=1, quantum=quantum)[0]
    inter = displacement_correlation(params, 1, T, s=1, s_prime=0, quantum=quantum)[0]
    return float(intra), float(inter)


def mean_square_displacement(params: ChainParameters, n_cells: int, T: float,
                             quantum: bool = True) -> tuple[float, float]:
    """<u_1^2>, <u_2^2> of a finite periodic chain (translation mode removed).

    Sum over the N allowed wave numbers; at k = 0 only the optical branch is
    kept (the acoustic k = 0 mode is the rigid translation of the whole ring,
    whose amplitude is fixed by fixing the centre of mass).
    """
    k = allowed_wavevectors(params, n_cells)
    w, e = normal_modes(params, k)
    amp = _mode_amplitude(w, T, quantum)
    amp[np.isclose(k, 0.0), 0] = 0.0                             # drop translation
    m = np.array([params.m1, params.m2])
    res = []
    for s in (0, 1):
        res.append(float(np.sum(np.abs(e[:, s, :]) ** 2 * amp) / (n_cells * m[s])))
    return res[0], res[1]
