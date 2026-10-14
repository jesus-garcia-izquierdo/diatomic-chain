"""
langevin.py
===========

Classical stochastic dynamics of the diatomic chain: an *independent*
numerical experiment that does not use normal modes at all.

Why
---
Lattice dynamics (``dispersion.py``) and canonical statistics
(``thermodynamics.py``) are exact for the harmonic model. A trajectory-based
simulation tests the same physics from a completely different angle:

* the velocity power spectrum of the atoms reproduces the phonon DOS
  (Wiener-Khinchin + equipartition: every mode carries k_B T, so the
  mass-weighted spectrum is proportional to g(omega));
* the variance of the total energy gives C_V through the fluctuation-response
  relation (Delta E)^2 = k_B T^2 C_V  ->  Dulong-Petit in the classical limit;
* the bond-stretch and relative-displacement variances reproduce the
  equipartition results of ``observables.py``.

A purely harmonic chain is *integrable* (each normal mode keeps its energy
forever), hence not ergodic: a single microcanonical (NVE) trajectory would
never sample the canonical distribution. The Langevin bath supplies the
"residual interaction" that drives the chain to equilibrium:

    dq = p/m dt,
    dp = F(q) dt - gamma p dt + sqrt(2 m gamma k_B T) dW .

Integrator: BAOAB splitting (Leimkuhler & Matthews, 2013)::

    B: p += F dt/2     A: q += p/m dt/2     O: p = c p + sqrt((1-c^2) m T) xi
    A: q += p/m dt/2   B: p += F dt/2       with c = exp(-gamma dt).

With gamma = 0 it reduces exactly to velocity Verlet (symplectic).
BAOAB samples configurational averages with very small time-step error.

Data types
----------
* State arrays ``q, p`` are ``numpy.ndarray`` float64 of length 2N (atom
  index i = 2n + s); masses and spring constants per bond are precomputed
  arrays of the same length so every step is fully vectorised.
* ``numpy.random.Generator`` (PCG64) is used for reproducible noise: pass a
  seed and you get bit-identical trajectories.
* Results are returned in a ``LangevinResult`` frozen dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from .parameters import ChainParameters


@dataclass(frozen=True)
class LangevinResult:
    """Time series and derived estimators of a Langevin run (per unit cell where noted)."""

    params: ChainParameters
    n_cells: int
    T: float
    gamma: float
    dt: float
    sample_interval: float
    energy: np.ndarray            # total energy at each sample, shape (n_samples,)
    kinetic_T: np.ndarray         # (n_samples, 2) instantaneous kinetic temperature per sublattice
    bond_sq: np.ndarray           # (n_samples, 2) mean square stretch of K1 and K2 bonds
    spectrum_omega: np.ndarray    # angular frequencies of the velocity spectrum
    spectrum: np.ndarray          # mass-weighted velocity spectrum, normalised to 2 per cell
    relative_disp_r: np.ndarray   # separations r (cells)
    relative_disp: np.ndarray     # <(u_{1,n+r} - u_{1,n})^2> averaged over n and time

    # ---- derived estimators -------------------------------------------- #
    def heat_capacity_per_cell(self) -> tuple[float, float]:
        """C_V/(N k_B) from energy fluctuations, with a block-averaging error bar."""
        return _block_estimate(self.energy, lambda x: np.var(x) / self.T ** 2 / self.n_cells)

    def mean_kinetic_temperature(self) -> np.ndarray:
        """Time-averaged kinetic temperature of each sublattice (equipartition check)."""
        return self.kinetic_T.mean(axis=0)


def _block_estimate(series: np.ndarray, estimator, n_blocks: int = 20) -> tuple[float, float]:
    """Value of ``estimator`` on the full series and standard error from blocks."""
    full = float(estimator(series))
    blocks = np.array_split(series, n_blocks)
    vals = np.array([estimator(b) for b in blocks])
    return full, float(vals.std(ddof=1) / np.sqrt(n_blocks))


class LangevinChain:
    """Periodic diatomic chain coupled to a Langevin heat bath."""

    def __init__(self, params: ChainParameters, n_cells: int, seed: int | None = 12345):
        self.params = params
        self.n_cells = n_cells
        self.n_atoms = 2 * n_cells
        # Per-atom masses: m1, m2, m1, m2, ...
        self.mass = np.tile([params.m1, params.m2], n_cells).astype(float)
        # Spring between atom i and atom i+1 (ring): K1 if i even else K2.
        self.k_bond = np.tile([params.K1, params.K2], n_cells).astype(float)
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros(self.n_atoms)
        self.p = np.zeros(self.n_atoms)

    # ---- mechanics ------------------------------------------------------ #
    def stretches(self, q: np.ndarray | None = None) -> np.ndarray:
        """Bond stretches e_i = q_{i+1} - q_i (periodic)."""
        q = self.q if q is None else q
        return np.roll(q, -1) - q

    def forces(self, q: np.ndarray) -> np.ndarray:
        """F_i = k_i e_i - k_{i-1} e_{i-1} (bond i pulls atom i forward)."""
        t = self.k_bond * (np.roll(q, -1) - q)          # tension of each bond
        return t - np.roll(t, 1)

    def energy(self) -> float:
        kin = 0.5 * np.sum(self.p ** 2 / self.mass)
        pot = 0.5 * np.sum(self.k_bond * self.stretches() ** 2)
        return float(kin + pot)

    def thermalise_momenta(self, T: float) -> None:
        """Draw momenta from the Maxwell-Boltzmann distribution at temperature T."""
        self.p = self.rng.normal(0.0, np.sqrt(self.mass * T))

    # ---- integrator ----------------------------------------------------- #
    def run(self, T: float, gamma: float, dt: float, n_steps: int,
            stride: int = 1, record: bool = True):
        """Propagate with BAOAB; yield (q, p) every ``stride`` steps if record."""
        c = np.exp(-gamma * dt)
        noise = np.sqrt((1.0 - c ** 2) * self.mass * T)
        inv_m = 1.0 / self.mass
        F = self.forces(self.q)
        q, p = self.q, self.p
        for step in range(1, n_steps + 1):
            p += 0.5 * dt * F                                    # B
            q += 0.5 * dt * p * inv_m                            # A
            if gamma > 0.0:
                p *= c                                           # O (friction ...
                p += noise * self.rng.standard_normal(self.n_atoms)  # ... + noise)
            q += 0.5 * dt * p * inv_m                            # A
            F = self.forces(q)
            p += 0.5 * dt * F                                    # B
            if record and step % stride == 0:
                yield q, p
        self.q, self.p = q, p


def simulate(params: ChainParameters, n_cells: int = 64, T: float = 1.0,
             gamma: float = 0.02, dt: float = 0.05, n_equil: int = 20_000,
             n_samples: int = 2 ** 15, stride: int = 10, seed: int = 2026,
             r_max: int | None = None) -> LangevinResult:
    """Run a full Langevin experiment and compute all estimators.

    Parameters
    ----------
    gamma : float
        Friction rate. Small gamma -> sharp spectral lines (Lorentzian FWHM ~ gamma)
        but slow thermalisation (relaxation time ~ 1/gamma).
    dt : float
        Time step; must satisfy dt * omega_max << 1 (default 0.05 with omega_max ~ 1.7).
    stride : int
        Steps between stored samples; the Nyquist frequency pi/(stride dt) must
        exceed omega_max.
    """
    chain = LangevinChain(params, n_cells, seed=seed)
    chain.thermalise_momenta(T)
    # Equilibration (no recording): several relaxation times 1/gamma.
    for _ in chain.run(T, gamma, dt, n_equil, record=False):
        pass

    r_max = n_cells // 2 if r_max is None else r_max
    rs = np.arange(1, r_max + 1)
    energy = np.empty(n_samples)
    kinT = np.empty((n_samples, 2))
    bond_sq = np.empty((n_samples, 2))
    # float32 halves the memory of the largest array (n_samples x 2N);
    # single precision is ample for a power spectrum.
    velocities = np.empty((n_samples, chain.n_atoms), dtype=np.float32)
    rel = np.zeros(rs.size)

    for i, (q, p) in enumerate(chain.run(T, gamma, dt, n_samples * stride, stride)):
        v = p / chain.mass
        velocities[i] = v
        energy[i] = chain.energy()
        # Equipartition: <m v^2> = k_B T for each atom.
        kinT[i, 0] = np.mean(chain.mass[0::2] * v[0::2] ** 2)
        kinT[i, 1] = np.mean(chain.mass[1::2] * v[1::2] ** 2)
        e = chain.stretches(q)
        bond_sq[i, 0] = np.mean(e[0::2] ** 2)                     # K1 bonds
        bond_sq[i, 1] = np.mean(e[1::2] ** 2)                     # K2 bonds
        # Relative displacement of sublattice-1 atoms r cells apart (subsampled).
        if i % 16 == 0:
            q1 = q[0::2]
            for j, r in enumerate(rs):
                rel[j] += np.mean((np.roll(q1, -r) - q1) ** 2)
    rel /= len(range(0, n_samples, 16))

    # Mass-weighted velocity power spectrum (Welch average over segments).
    fs = 1.0 / (stride * dt)                                      # sampling frequency
    f, P = welch(velocities, fs=fs, nperseg=min(4096, n_samples // 4), axis=0,
                 detrend="constant", scaling="density")
    S = (P * chain.mass[None, :]).sum(axis=1)                     # sum_i m_i |v_i(f)|^2
    w = 2.0 * np.pi * f
    S_w = S / np.trapezoid(S, w) * 2.0                            # normalise to 2 modes/cell

    return LangevinResult(params=params, n_cells=n_cells, T=T, gamma=gamma, dt=dt,
                          sample_interval=stride * dt, energy=energy, kinetic_T=kinT,
                          bond_sq=bond_sq, spectrum_omega=w, spectrum=S_w,
                          relative_disp_r=rs, relative_disp=rel)


# ---------------------------------------------------------------------- #
# Exact finite-ring predictions for the classical chain                   #
# ---------------------------------------------------------------------- #
def exact_ring_relative_displacement(params: ChainParameters, n_cells: int,
                                     T: float, r) -> np.ndarray:
    """<(u_{1,n+r} - u_{1,n})^2> for a classical *ring* of N cells.

    The bond stretches are independent Gaussians (variance T/K_i) conditioned
    on summing to zero around the ring, i.e. a discrete Brownian bridge:

        G(r) = r (1 - r/N) k_B T (1/K1 + 1/K2).

    For N -> infinity this is the linear (Landau-Peierls) growth r T (1/K1 + 1/K2).
    """
    r = np.asarray(r, dtype=float)
    return r * (1.0 - r / n_cells) * T * (1.0 / params.K1 + 1.0 / params.K2)


def exact_ring_heat_capacity(n_cells: int) -> float:
    """Classical C_V/(N k_B) of the ring including the free centre of mass.

    2N kinetic quadratic terms + (2N - 1) independent potential terms
    (the stretches sum to zero) -> C = (4N - 1)/2 k_B -> per cell 2 - 1/(2N).
    """
    return 2.0 - 0.5 / n_cells


def exact_ring_bond_variance(params: ChainParameters, n_cells: int, T: float) -> tuple[float, float]:
    """<e^2> of the K1 and K2 bonds of a classical ring (Gaussian conditioning).

    Var(e_i) = sigma_i^2 - sigma_i^4 / sigma_tot^2, sigma_i^2 = T/K_i,
    sigma_tot^2 = N T (1/K1 + 1/K2): the O(1/N) correction comes from the ring
    closure constraint sum_i e_i = 0.
    """
    s_tot = n_cells * T * (1.0 / params.K1 + 1.0 / params.K2)
    s1, s2 = T / params.K1, T / params.K2
    return s1 - s1 ** 2 / s_tot, s2 - s2 ** 2 / s_tot
