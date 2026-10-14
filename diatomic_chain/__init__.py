"""
diatomic_chain
==============

Lattice dynamics and phonon thermodynamics of the one-dimensional diatomic
harmonic chain: acoustic and optical branches, phonon density of states,
numerical heat capacity versus the Debye and Einstein models, and thermal
displacement fluctuations, cross-checked with Langevin dynamics.

Author : Jesús García Izquierdo
Repo   : https://github.com/jesus-garcia-izquierdo/diatomic-chain
License: MIT

Units: hbar = k_B = 1 (frequencies, energies and temperatures share a unit).

Module map
----------
parameters      ChainParameters (m1, m2, K1, K2, a) and derived constants
dispersion      dynamical matrix D(k), branches omega_{+-}(k), eigenvectors, v_g
real_space      finite chain: det(K - omega^2 M) = 0 without Bloch's theorem
dos             analytic / sampled / finite-chain phonon DOS
thermodynamics  canonical U, C_V, F, S per cell (quantum and classical)
models          Debye (1D), Einstein and hybrid models, fits, Theta_D(T)
observables     displacement correlations, <u^2>, bond fluctuations
langevin        BAOAB Langevin dynamics: spectrum, C_V from fluctuations
plotting        matplotlib style and figure helpers
"""

from .parameters import ChainParameters
from . import dispersion, real_space, dos, thermodynamics, models, observables, langevin

__all__ = [
    "ChainParameters",
    "dispersion",
    "real_space",
    "dos",
    "thermodynamics",
    "models",
    "observables",
    "langevin",
]

__version__ = "1.0.0"
__author__ = "Jesús García Izquierdo"
