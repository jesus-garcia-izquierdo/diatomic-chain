"""
dos.py
======

Phonon density of states (DOS) g(omega) of the diatomic chain.

Convention: g(omega) d omega is the number of normal modes *per unit cell*
with frequency in (omega, omega + d omega). With two atoms per cell in 1D
there are two modes per cell, so

    int_0^inf g(omega) d omega = 2 .

Three independent routes are implemented:

1. **Analytic** (thermodynamic limit). Inverting the secular equation
   det(D(k) - omega^2) = 0 for s = sin^2(ka/2) gives

       s(omega) = m1 m2 omega^2 (sigma/mu - omega^2) / (4 K1 K2),   sigma = K1 + K2,

   and g(omega) = (a/pi) |dk/d omega| becomes

       g(omega) = (1/pi) |ds/d omega| / sqrt(s (1 - s)),     0 < s < 1,

   which diverges as an inverse square root at every band edge
   (1D van Hove singularities, where the group velocity vanishes).
   The cumulative count N(omega) = int_0^omega g is also available in closed
   form, which lets us compute *exact* bin averages for histograms.
2. **Brillouin-zone sampling**: evaluate omega_{+-}(k) on a fine uniform k-grid
   and histogram the frequencies.
3. **Real-space diagonalisation** of a finite chain (``real_space.py``) and
   histogram of its 2N eigenfrequencies.

Data types
----------
Frequencies and DOS values are ``numpy.ndarray`` of ``float64``. Histograms
return a ``DOSHistogram`` frozen dataclass (bin edges, centres and values) so
the three arrays cannot get out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import ChainParameters
from .dispersion import omega as dispersion_omega


@dataclass(frozen=True)
class DOSHistogram:
    """Histogram estimate of the DOS (normalised to 2 modes per cell)."""

    edges: np.ndarray     # shape (n_bins + 1,)
    centers: np.ndarray   # shape (n_bins,)
    values: np.ndarray    # shape (n_bins,)  -> g averaged over each bin

    @property
    def width(self) -> np.ndarray:
        return np.diff(self.edges)

    def total(self) -> float:
        """Total number of modes per cell (should be 2)."""
        return float((self.values * self.width).sum())


# ---------------------------------------------------------------------- #
# Analytic DOS                                                            #
# ---------------------------------------------------------------------- #
def _s_of_omega(params: ChainParameters, w: np.ndarray) -> np.ndarray:
    """s = sin^2(ka/2) as a function of omega (from the secular equation).

    Expanding the product, sigma^2 cancels exactly and

        s = m1 m2 omega^2 (sigma/mu - omega^2) / (4 K1 K2),

    a form free of the catastrophic cancellation that the unexpanded
    expression suffers for omega -> 0.
    """
    p = params
    w2 = w ** 2
    return p.m1 * p.m2 * w2 * (p.K_sum / p.reduced_mass - w2) / (4.0 * p.K1 * p.K2)


def analytic_dos(params: ChainParameters, w) -> np.ndarray:
    """Exact DOS per unit cell in the thermodynamic limit.

    Returns 0 outside the allowed bands (inside the gap and above omega_max).
    At the band edges the value is +inf (integrable singularity); callers that
    integrate g should use ``integrated_dos`` or quadrature in k instead.
    """
    p = params
    w = np.asarray(w, dtype=float)
    s = _s_of_omega(p, w)
    ds_dw = p.m1 * p.m2 * w * (p.K_sum / p.m1 + p.K_sum / p.m2 - 2.0 * w ** 2) \
        / (2.0 * p.K1 * p.K2)
    inside = (s > 0.0) & (s < 1.0) & (w > 0.0)
    g = np.zeros_like(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        g[inside] = np.abs(ds_dw[inside]) / (np.pi * np.sqrt(s[inside] * (1.0 - s[inside])))
    # omega -> 0+ limit (s -> 0 but g -> a/(pi v_s), finite): handled by the
    # formula for w > 0; at exactly w = 0 return the limit.
    g[w == 0.0] = params.a / (np.pi * params.sound_velocity)
    return g


def integrated_dos(params: ChainParameters, w) -> np.ndarray:
    """Cumulative number of modes per cell, N(omega) = int_0^omega g.

    * acoustic band  (0 <= omega <= omega_a):  N = (2/pi) arcsin sqrt(s)
    * gap            (omega_a < omega < omega_o):  N = 1
    * optical band   (omega_o <= omega <= omega_max):  N = 2 - (2/pi) arcsin sqrt(s)
      (the optical branch *decreases* with |k|)
    * above the top: N = 2
    """
    p = params
    w = np.asarray(w, dtype=float)
    w_a, w_o = p.zone_boundary_frequencies
    s = np.clip(_s_of_omega(p, w), 0.0, 1.0)
    frac = (2.0 / np.pi) * np.arcsin(np.sqrt(s))
    N = np.where(w <= w_a, frac, 1.0)
    N = np.where((w >= w_o) & (w <= p.omega_max), 2.0 - frac, N)
    N = np.where(w > p.omega_max, 2.0, N)
    N = np.where(w <= 0.0, 0.0, N)
    return N


def bin_averaged_dos(params: ChainParameters, edges) -> np.ndarray:
    """Exact average of g(omega) over each bin: [N(w_{i+1}) - N(w_i)] / dw."""
    edges = np.asarray(edges, dtype=float)
    return np.diff(integrated_dos(params, edges)) / np.diff(edges)


# ---------------------------------------------------------------------- #
# Histogram estimators                                                    #
# ---------------------------------------------------------------------- #
def histogram_dos(frequencies, n_cells: int, edges) -> DOSHistogram:
    """Histogram DOS from a list of 2N eigenfrequencies of N cells."""
    frequencies = np.ravel(np.asarray(frequencies, dtype=float))
    edges = np.asarray(edges, dtype=float)
    counts, _ = np.histogram(frequencies, bins=edges)
    values = counts / (n_cells * np.diff(edges))      # modes per cell per d omega
    centers = 0.5 * (edges[1:] + edges[:-1])
    return DOSHistogram(edges=edges, centers=centers, values=values)


def sampled_dos(params: ChainParameters, n_k: int, edges) -> DOSHistogram:
    """Histogram DOS from a uniform midpoint grid of n_k wave numbers in the BZ.

    The midpoint grid k_j = -pi/a + (j + 1/2) 2 pi/(n_k a) avoids k = 0 exactly
    and treats +k and -k symmetrically.
    """
    j = np.arange(n_k)
    k = -np.pi / params.a + (j + 0.5) * 2.0 * np.pi / (n_k * params.a)
    w = dispersion_omega(params, k)
    return histogram_dos(w, n_k, edges)


def frequency_grid(params: ChainParameters, n_bins: int = 200,
                   margin: float = 1.02) -> np.ndarray:
    """Equally spaced bin edges from 0 to slightly above omega_max."""
    return np.linspace(0.0, margin * params.omega_max, n_bins + 1)


# ---------------------------------------------------------------------- #
# Smooth (broadened) DOS                                                  #
# ---------------------------------------------------------------------- #
def broadened_dos(params: ChainParameters, w, width: float,
                  kind: str = "lorentzian", n_k: int = 20000) -> np.ndarray:
    """DOS convolved with a Lorentzian or Gaussian of half-width ``width``.

    Useful to compare with spectra obtained from finite-time simulations
    (e.g. Langevin dynamics, where the friction gamma broadens every mode
    into a Lorentzian of FWHM ~ gamma).

    Implemented as a sum over a dense k-grid: g_eta(w) = (1/N) sum_{k,b} L(w - w_b(k)).
    """
    w = np.asarray(w, dtype=float)
    j = np.arange(n_k)
    k = -np.pi / params.a + (j + 0.5) * 2.0 * np.pi / (n_k * params.a)
    modes = dispersion_omega(params, k).ravel()          # (2 n_k,)
    x = w[:, None] - modes[None, :]
    if kind == "lorentzian":
        kernel = (width / np.pi) / (x ** 2 + width ** 2)
    elif kind == "gaussian":
        kernel = np.exp(-0.5 * (x / width) ** 2) / (np.sqrt(2.0 * np.pi) * width)
    else:
        raise ValueError("kind must be 'lorentzian' or 'gaussian'")
    return kernel.sum(axis=1) / n_k


# ---------------------------------------------------------------------- #
# Model DOS (Debye and Einstein)                                          #
# ---------------------------------------------------------------------- #
def debye_dos(w, omega_D: float, n_modes: float = 2.0) -> np.ndarray:
    """1D Debye DOS: constant up to the cutoff, n_modes / omega_D.

    In one dimension a linear dispersion omega = v|k| gives a *flat* DOS
    g = a/(pi v); the cutoff omega_D is fixed by counting modes.
    """
    w = np.asarray(w, dtype=float)
    return np.where((w >= 0.0) & (w <= omega_D), n_modes / omega_D, 0.0)
