"""
dispersion.py
=============

Lattice dynamics in reciprocal space: dynamical matrix, phonon branches
omega_-(k) (acoustic) and omega_+(k) (optical), polarisation vectors and
group velocities.

Physics in two lines
--------------------
With the Bloch ansatz  u_{s,n}(t) = e_s / sqrt(m_s) * exp[i (k n a - omega t)]
the 2N coupled Newton equations collapse, for every wave number k of the
first Brillouin zone, into a 2x2 Hermitian eigenvalue problem

    D(k) e = omega^2 e ,

    D(k) = [[ (K1+K2)/m1                , -(K1 + K2 e^{-ika})/sqrt(m1 m2) ],
            [ -(K1 + K2 e^{+ika})/sqrt(m1 m2),  (K1+K2)/m2               ]] .

Its two eigenvalues are the two phonon branches.

Data types
----------
* ``k`` is accepted as a Python float or any array-like; internally it is a
  ``numpy.ndarray`` of ``float64`` so every function is vectorised (no Python
  loops over k: numpy broadcasting is ~100x faster).
* ``D(k)`` is returned as a complex array of shape ``k.shape + (2, 2)``
  (``complex128``) because the off-diagonal elements carry a Bloch phase.
* Polarisation vectors are complex arrays of shape ``k.shape + (2, 2)``
  where the *last* index labels the branch (0 = acoustic, 1 = optical), exactly
  like the output convention of ``numpy.linalg.eigh``.
"""

from __future__ import annotations

import numpy as np

from .parameters import ChainParameters

# Branch labels (plain ints, used as indices into arrays of shape (..., 2)).
ACOUSTIC: int = 0
OPTICAL: int = 1


# ---------------------------------------------------------------------- #
# Dynamical matrix                                                        #
# ---------------------------------------------------------------------- #
def dynamical_matrix(params: ChainParameters, k) -> np.ndarray:
    """Mass-weighted dynamical matrix D(k), shape ``k.shape + (2, 2)``.

    Parameters
    ----------
    params : ChainParameters
    k : float or array-like
        Wave number(s). Any real value is allowed; D is 2*pi/a periodic.
    """
    k = np.asarray(k, dtype=float)
    p = params
    phase = np.exp(1j * k * p.a)                    # Bloch phase e^{ika}
    sqrt_mm = np.sqrt(p.m1 * p.m2)

    D = np.empty(k.shape + (2, 2), dtype=complex)
    D[..., 0, 0] = p.K_sum / p.m1
    D[..., 1, 1] = p.K_sum / p.m2
    D[..., 0, 1] = -(p.K1 + p.K2 * np.conj(phase)) / sqrt_mm
    D[..., 1, 0] = -(p.K1 + p.K2 * phase) / sqrt_mm   # Hermitian partner
    return D


# ---------------------------------------------------------------------- #
# Closed-form branches                                                    #
# ---------------------------------------------------------------------- #
def omega_squared(params: ChainParameters, k) -> np.ndarray:
    """Closed-form eigenvalues omega^2_{-}(k), omega^2_{+}(k).

    Using |K1 + K2 e^{ika}|^2 = (K1+K2)^2 - 4 K1 K2 sin^2(ka/2):

        omega^2_{+/-} = sigma/(2 mu) * [ 1 +/- sqrt(1 - 16 K1 K2 mu^2 sin^2(ka/2)
                                                      / (sigma^2 m1 m2)) ]

    Returns an array of shape ``k.shape + (2,)`` with the acoustic branch in
    position 0 and the optical branch in position 1.
    """
    k = np.asarray(k, dtype=float)
    p = params
    s2 = np.sin(0.5 * k * p.a) ** 2                 # sin^2(ka/2)
    mean = 0.5 * p.K_sum * (1.0 / p.m1 + 1.0 / p.m2)  # (A + C)/2 = sigma/(2 mu)
    det = 4.0 * p.K1 * p.K2 * s2 / (p.m1 * p.m2)    # det D(k) = omega_-^2 omega_+^2
    disc = mean ** 2 - det
    root = np.sqrt(np.clip(disc, 0.0, None))        # clip: round-off safety
    w2_plus = mean + root
    # Acoustic branch via Vieta's formula, omega_-^2 = det / omega_+^2, instead
    # of (mean - root): the subtraction cancels catastrophically for small k
    # (omega_-^2 ~ v_s^2 k^2 << mean) and would return 0 for ka < 1e-8.
    w2_minus = det / w2_plus
    return np.stack([w2_minus, w2_plus], axis=-1)


def omega(params: ChainParameters, k) -> np.ndarray:
    """Phonon frequencies omega_{-}(k), omega_{+}(k); shape ``k.shape + (2,)``."""
    return np.sqrt(omega_squared(params, k))


def omega_acoustic(params: ChainParameters, k) -> np.ndarray:
    """Acoustic branch omega_-(k) (vanishes linearly at k = 0)."""
    return omega(params, k)[..., ACOUSTIC]


def omega_optical(params: ChainParameters, k) -> np.ndarray:
    """Optical branch omega_+(k) (finite at k = 0)."""
    return omega(params, k)[..., OPTICAL]


# ---------------------------------------------------------------------- #
# Numerical diagonalisation (eigenvalues + polarisation vectors)          #
# ---------------------------------------------------------------------- #
def normal_modes(params: ChainParameters, k) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise D(k) numerically.

    Returns
    -------
    w : ndarray, shape ``k.shape + (2,)``
        Frequencies sorted in ascending order (acoustic, optical).
    e : ndarray (complex), shape ``k.shape + (2, 2)``
        Normalised polarisation vectors; ``e[..., s, b]`` is the component on
        sublattice ``s`` of branch ``b``. The *physical* displacement amplitude
        of atom s is ``e[..., s, b] / sqrt(m_s)``.
    """
    D = dynamical_matrix(params, k)
    _, e = np.linalg.eigh(D)             # Hermitian solver: sorted eigenpairs
    # Eigenvalues are taken from the closed form (same order, acoustic first):
    # LAPACK's absolute error ~1e-16 |D| would spoil omega_-^2 ~ k^2 at tiny k,
    # whereas the eigenvectors are well conditioned (the branches never cross).
    return omega(params, k), e


def amplitude_ratio(params: ChainParameters, k) -> np.ndarray:
    """Physical amplitude ratio u_2 / u_1 for each branch, shape ``k.shape + (2,)``.

    From the first row of (D - omega^2) e = 0 and u_s = e_s / sqrt(m_s):

        u2/u1 = (K1+K2 - m1 omega^2) / (K1 + K2 e^{-ika}).

    It is a complex number (Bloch phase); at k = 0 it is real:
    +1 for the acoustic branch (rigid translation of the cell) and
    -m1/m2 for the optical branch (centre of mass at rest, atoms in antiphase).
    """
    k = np.asarray(k, dtype=float)
    p = params
    w2 = omega_squared(params, k)                       # (..., 2)
    numerator = p.K_sum - p.m1 * w2
    denominator = (p.K1 + p.K2 * np.exp(-1j * k * p.a))[..., None]
    return numerator / denominator


# ---------------------------------------------------------------------- #
# Group velocity                                                          #
# ---------------------------------------------------------------------- #
def group_velocity(params: ChainParameters, k) -> np.ndarray:
    """Group velocity v_g = d omega / d k for both branches (analytic).

    Differentiating omega^2 = mean -/+ sqrt(disc) with
    d disc/dk = -2 K1 K2 a sin(ka) / (m1 m2):

        d omega/dk = (d omega^2/dk) / (2 omega),
        d omega^2_{-/+}/dk = +/- K1 K2 a sin(ka) / (m1 m2 sqrt(disc)).

    At k = 0 exactly the acoustic value 0/0 is replaced by its limit v_s
    (the limit from k > 0; v_g is an odd function of k).
    """
    k = np.asarray(k, dtype=float)
    p = params
    s2 = np.sin(0.5 * k * p.a) ** 2
    mean = 0.5 * p.K_sum * (1.0 / p.m1 + 1.0 / p.m2)
    disc = np.clip(mean ** 2 - 4.0 * p.K1 * p.K2 * s2 / (p.m1 * p.m2), 1e-300, None)
    dw2 = p.K1 * p.K2 * p.a * np.sin(k * p.a) / (p.m1 * p.m2 * np.sqrt(disc))
    w = omega(params, k)

    with np.errstate(divide="ignore", invalid="ignore"):
        v_ac = np.where(w[..., ACOUSTIC] > 0, dw2 / (2.0 * w[..., ACOUSTIC]),
                        p.sound_velocity)
        v_op = np.where(w[..., OPTICAL] > 0, -dw2 / (2.0 * w[..., OPTICAL]), 0.0)
    return np.stack([v_ac, v_op], axis=-1)


# ---------------------------------------------------------------------- #
# Brillouin zone helpers                                                  #
# ---------------------------------------------------------------------- #
def brillouin_zone(params: ChainParameters, n_points: int = 801) -> np.ndarray:
    """Uniform grid on the first Brillouin zone [-pi/a, pi/a] (for plotting)."""
    return np.linspace(-np.pi / params.a, np.pi / params.a, n_points)


def allowed_wavevectors(params: ChainParameters, n_cells: int) -> np.ndarray:
    """Wave numbers allowed by Born-von Karman (periodic) boundary conditions.

    k_j = 2 pi j / (N a),  j = -N/2+1, ..., N/2  ->  exactly N values in the 1st BZ.
    The integer grid is built first (exact) and then scaled, which avoids
    accumulating floating-point error.
    """
    if n_cells % 2 == 0:                        # even N: j = -N/2+1, ..., N/2
        j = np.arange(-n_cells // 2 + 1, n_cells // 2 + 1)
    else:                                       # odd N: j = -(N-1)/2, ..., (N-1)/2
        j = np.arange(-(n_cells - 1) // 2, (n_cells - 1) // 2 + 1)
    return 2.0 * np.pi * j / (n_cells * params.a)
