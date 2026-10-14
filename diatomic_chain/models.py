"""
models.py
=========

Reference models for the lattice heat capacity and their comparison with the
exact (numerical) result of the diatomic chain.

* **Einstein**: every mode vibrates at the same frequency omega_E.
  DOS = n_modes * delta(omega - omega_E).
      C/(N k_B) = n_modes * c(omega_E / T),   c(y) = (y/2)^2 / sinh^2(y/2).

* **Debye (1D)**: linear dispersion omega = v|k| up to a cutoff omega_D fixed
  by counting modes. In 1D the DOS is flat: g = n_modes / omega_D.
      C/(N k_B) = n_modes * D_1(omega_D / T),
      D_1(x_D) = (1/x_D) int_0^{x_D} y^2 e^y / (e^y - 1)^2 dy .

* **Debye + Einstein (hybrid)**: Debye for the acoustic branch (1 mode/cell,
  cutoff pi v_s / a) and Einstein for the optical branch (1 mode/cell).
  This is the natural model for a crystal with a basis.

Choice of the characteristic frequencies (all in units hbar = k_B = 1, so a
frequency is also a characteristic temperature Theta = hbar omega / k_B):

* ``debye_frequency`` from the *sound velocity* (exact at low T),
* ``einstein_frequency_second_moment`` from <omega^2> (exact at high T),
* ``optical_mean_frequency`` = average of omega_+ over the BZ,
* least-squares fits of a single Theta to the exact C(T) curve.

Data types
----------
Scalars are ``float``; curves are ``numpy.ndarray``. Fits return a small
``FitResult`` dataclass (best value + rms error) rather than a bare tuple, so
the meaning of each number is explicit at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .parameters import ChainParameters
from .dispersion import omega_optical
from .thermodynamics import mode_heat_capacity


# ---------------------------------------------------------------------- #
# Debye function in one dimension                                         #
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _gl(n: int):
    return np.polynomial.legendre.leggauss(n)


def debye_function_1d(x_D) -> np.ndarray:
    """D_1(x_D) = (1/x_D) int_0^{x_D} c(y) dy, with D_1(0) = 1 and D_1 -> pi^2/(3 x_D).

    The integrand is exponentially small beyond y ~ 60, so the integral is cut
    there (relative error < 1e-20), which keeps a fixed 200-node Gauss-Legendre
    rule accurate for any x_D.
    """
    x_D = np.atleast_1d(np.asarray(x_D, dtype=float))
    x, w = _gl(200)
    out = np.empty_like(x_D)
    for i, xd in enumerate(x_D):
        if xd < 1e-10:
            out[i] = 1.0
            continue
        upper = min(xd, 60.0)
        y = 0.5 * upper * (x + 1.0)
        integral = 0.5 * upper * np.sum(w * mode_heat_capacity(y, 1.0))
        out[i] = integral / xd
    return out


# ---------------------------------------------------------------------- #
# Model heat capacities (per unit cell, units of k_B)                     #
# ---------------------------------------------------------------------- #
def einstein_heat_capacity(T, omega_E: float, n_modes: float = 2.0) -> np.ndarray:
    """Einstein model: n_modes oscillators of frequency omega_E per cell."""
    T = np.asarray(T, dtype=float)
    return n_modes * mode_heat_capacity(omega_E, T)


def debye_heat_capacity(T, omega_D: float, n_modes: float = 2.0) -> np.ndarray:
    """1D Debye model with cutoff omega_D and n_modes modes per cell."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    return n_modes * debye_function_1d(omega_D / T)


def hybrid_heat_capacity(params: ChainParameters, T,
                         omega_E: float | None = None) -> np.ndarray:
    """Debye for the acoustic branch + Einstein for the optical branch.

    Acoustic cutoff: one mode per cell with the exact sound velocity,
    omega_D^ac = pi v_s / a. Optical frequency: BZ average of omega_+ unless
    ``omega_E`` is given.
    """
    if omega_E is None:
        omega_E = optical_mean_frequency(params)
    w_D_ac = np.pi * params.sound_velocity / params.a
    return debye_heat_capacity(T, w_D_ac, 1.0) + einstein_heat_capacity(T, omega_E, 1.0)


# ---------------------------------------------------------------------- #
# Characteristic frequencies                                              #
# ---------------------------------------------------------------------- #
def debye_frequency(params: ChainParameters, n_modes: float = 2.0) -> float:
    """Debye cutoff for a flat DOS a/(pi v_s) holding n_modes modes per cell:
    omega_D = n_modes * pi v_s / a."""
    return n_modes * np.pi * params.sound_velocity / params.a


def einstein_frequency_second_moment(params: ChainParameters) -> float:
    """omega_E = sqrt(<omega^2>): reproduces the exact 1/T^2 high-T correction."""
    return float(np.sqrt(params.mean_square_frequency))


def optical_mean_frequency(params: ChainParameters, n_k: int = 512) -> float:
    """<omega_+>_BZ = (a/pi) int_0^{pi/a} omega_+(k) dk (Gauss-Legendre)."""
    x, w = _gl(n_k)
    k_max = np.pi / params.a
    k = 0.5 * k_max * (x + 1.0)
    return float((params.a / np.pi) * 0.5 * k_max * np.sum(w * omega_optical(params, k)))


# ---------------------------------------------------------------------- #
# Fits and effective Debye temperature                                    #
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class FitResult:
    """Best-fit characteristic frequency (= temperature) of a one-parameter model."""

    theta: float          # best omega_D or omega_E (hbar = k_B = 1)
    rms_error: float      # rms of C_model - C_exact over the fitted points
    max_rel_error: float  # max |C_model - C_exact| / C_exact


def _fit(model, T, C_exact, bracket) -> FitResult:
    T = np.asarray(T, dtype=float)
    C_exact = np.asarray(C_exact, dtype=float)

    def cost(log_theta: float) -> float:
        return float(np.mean((model(T, np.exp(log_theta)) - C_exact) ** 2))

    res = minimize_scalar(cost, bounds=np.log(bracket), method="bounded",
                          options={"xatol": 1e-10})
    theta = float(np.exp(res.x))
    diff = model(T, theta) - C_exact
    return FitResult(theta=theta, rms_error=float(np.sqrt(np.mean(diff ** 2))),
                     max_rel_error=float(np.max(np.abs(diff) / C_exact)))


def fit_debye(T, C_exact, bracket=(1e-3, 1e3)) -> FitResult:
    """Least-squares Debye cutoff (2 modes per cell) to an exact C(T) curve."""
    return _fit(lambda t, th: debye_heat_capacity(t, th, 2.0), T, C_exact, bracket)


def fit_einstein(T, C_exact, bracket=(1e-3, 1e3)) -> FitResult:
    """Least-squares Einstein frequency (2 modes per cell) to an exact C(T) curve."""
    return _fit(lambda t, th: einstein_heat_capacity(t, th, 2.0), T, C_exact, bracket)


def effective_debye_temperature(T, C_exact, n_modes: float = 2.0) -> np.ndarray:
    """Theta_D(T): the Debye cutoff that reproduces C_exact *at each* T.

    Solves n_modes * D_1(Theta / T) = C_exact(T) for Theta. A constant
    Theta_D(T) would mean the Debye model is exact; its variation measures how
    far the real DOS is from Debye's flat DOS. Points where C is too close to
    the classical value (inversion ill-conditioned) return NaN.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    C_exact = np.atleast_1d(np.asarray(C_exact, dtype=float))
    out = np.full_like(T, np.nan)
    for i, (t, c) in enumerate(zip(T, C_exact)):
        target = c / n_modes
        if not (1e-12 < target < 1.0 - 1e-9):
            continue

        def f(log_x: float) -> float:
            return debye_function_1d(np.exp(log_x))[0] - target

        try:
            log_x = brentq(f, np.log(1e-6), np.log(1e6), xtol=1e-12)
            out[i] = t * np.exp(log_x)
        except ValueError:
            pass
    return out
