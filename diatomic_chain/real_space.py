"""
real_space.py
=============

Finite chain in real space: build the mass matrix M and the stiffness
(force-constant) matrix K of a chain of N unit cells (2N atoms) and solve the
small-oscillation problem

    det(K - omega^2 M) = 0

*without* using Bloch's theorem. This is an independent check of the
reciprocal-space results (``dispersion.py``): with periodic boundary
conditions both must give exactly the same 2N frequencies.

Atom ordering
-------------
Atom ``i = 2 n + s`` (s = 0 -> mass m1, s = 1 -> mass m2) of cell ``n``.
Bonds::

    (2n, 2n+1)   spring K1   (intra-cell)
    (2n+1, 2n+2) spring K2   (inter-cell)

Data types
----------
* ``M`` and ``K`` are dense ``numpy.ndarray`` of ``float64`` with shape
  (2N, 2N). For the sizes used here (N <= a few thousand) dense LAPACK
  (``scipy.linalg.eigh``) is fast and exact; a sparse solver would only pay
  off for N >~ 10^4 and would return just a few eigenvalues.
* ``FiniteChainModes`` is a small frozen dataclass bundling frequencies,
  eigenvectors and the boundary condition, so the results travel together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .parameters import ChainParameters

BOUNDARY_CONDITIONS = ("periodic", "free", "fixed")


@dataclass(frozen=True)
class FiniteChainModes:
    """Normal modes of a finite chain.

    Attributes
    ----------
    omega : ndarray, shape (2N,)
        Eigenfrequencies in ascending order.
    vectors : ndarray, shape (2N, 2N)
        Column ``j`` is the displacement pattern of mode ``j``; columns are
        M-orthonormal: ``vectors.T @ M @ vectors = identity``.
    n_cells : int
    boundary : str
    """

    omega: np.ndarray
    vectors: np.ndarray
    n_cells: int
    boundary: str


def mass_matrix(params: ChainParameters, n_cells: int) -> np.ndarray:
    """Diagonal mass matrix M = diag(m1, m2, m1, m2, ...)."""
    masses = np.tile([params.m1, params.m2], n_cells).astype(float)
    return np.diag(masses)


def stiffness_matrix(params: ChainParameters, n_cells: int,
                     boundary: str = "periodic") -> np.ndarray:
    """Force-constant matrix K_{ij} = d^2 V / du_i du_j of the harmonic chain.

    Each spring of constant k between atoms i and j adds the 2x2 block
    [[k, -k], [-k, k]] to rows/columns (i, j). This guarantees the acoustic
    sum rule  sum_j K_ij = 0  (rigid translations cost no energy), except
    when the ends are clamped (``boundary="fixed"``), which breaks
    translational invariance on purpose.

    Parameters
    ----------
    boundary : {"periodic", "free", "fixed"}
        * periodic : Born-von Karman ring (last atom bonded to the first).
        * free     : open chain, free ends.
        * fixed    : open chain whose two end atoms are tied to rigid walls
                     (with a K2 spring), so u_0 = u_{2N+1} = 0 effectively.
    """
    if boundary not in BOUNDARY_CONDITIONS:
        raise ValueError(f"boundary must be one of {BOUNDARY_CONDITIONS}")
    n_atoms = 2 * n_cells
    K = np.zeros((n_atoms, n_atoms))

    def add_spring(i: int, j: int, k_spring: float) -> None:
        K[i, i] += k_spring
        K[j, j] += k_spring
        K[i, j] -= k_spring
        K[j, i] -= k_spring

    for i in range(n_atoms - 1):
        add_spring(i, i + 1, params.K1 if i % 2 == 0 else params.K2)

    if boundary == "periodic":
        # Close the ring (for N = 1 both springs join the same pair of atoms,
        # which is still correct: the matrix elements simply add up).
        add_spring(n_atoms - 1, 0, params.K2)
    elif boundary == "fixed":
        K[0, 0] += params.K2          # wall -- K2 -- atom 0
        K[-1, -1] += params.K2        # atom 2N-1 -- K2 -- wall
    return K


def solve_finite_chain(params: ChainParameters, n_cells: int,
                       boundary: str = "periodic") -> FiniteChainModes:
    """Solve the generalised eigenproblem K v = omega^2 M v.

    ``scipy.linalg.eigh(K, M)`` exploits that K is symmetric and M symmetric
    positive definite; it returns M-orthonormal eigenvectors.
    """
    M = mass_matrix(params, n_cells)
    K = stiffness_matrix(params, n_cells, boundary)
    w2, vecs = eigh(K, M)
    # Round-off makes the zero (translation) mode slightly negative: clip it.
    w = np.sqrt(np.clip(w2, 0.0, None))
    return FiniteChainModes(omega=w, vectors=vecs, n_cells=n_cells, boundary=boundary)


def mode_participation_ratio(modes: FiniteChainModes, params: ChainParameters) -> np.ndarray:
    """Inverse participation ratio IPR_j = sum_i |v_ij|^4 / (sum_i |v_ij|^2)^2.

    A mode spread over the whole chain has IPR ~ 1/(2N); a mode localised on
    a few atoms (e.g. an edge mode of an open chain) has IPR ~ O(1). Useful to
    detect boundary-localised states with ``boundary="free"``.
    """
    sqrt_m = np.sqrt(np.diag(mass_matrix(params, modes.n_cells)))
    x = modes.vectors * sqrt_m[:, None]          # mass-weighted amplitudes
    p = np.abs(x) ** 2
    return (p ** 2).sum(axis=0) / (p.sum(axis=0) ** 2)
