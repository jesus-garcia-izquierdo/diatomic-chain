"""
Physics regression tests.

Every test checks the code against an *independent* exact result (a closed
formula, a sum rule or a different numerical route), never against numbers
copied from a previous run.

Run with ``pytest`` or, if pytest is not installed, with
``python tests/run_tests.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diatomic_chain import ChainParameters                       # noqa: E402
from diatomic_chain import dispersion as dp                      # noqa: E402
from diatomic_chain import real_space as rs                      # noqa: E402
from diatomic_chain import dos as ds                             # noqa: E402
from diatomic_chain import thermodynamics as th                  # noqa: E402
from diatomic_chain import models as md                          # noqa: E402
from diatomic_chain import observables as ob                     # noqa: E402
from diatomic_chain import langevin as lg                        # noqa: E402

REF = ChainParameters.textbook(mass_ratio=2.0)
GENERAL = ChainParameters(m1=1.0, m2=3.0, K1=1.0, K2=2.5, a=1.3)
CASES = [REF, GENERAL, ChainParameters.textbook(mass_ratio=1.0)]


# ---------------------------------------------------------------- dispersion
def test_closed_form_matches_numerical_diagonalisation():
    k = np.linspace(-np.pi, np.pi, 101)
    for p in CASES:
        kk = k / p.a
        w2_num = np.linalg.eigvalsh(dp.dynamical_matrix(p, kk))
        assert np.allclose(dp.omega_squared(p, kk), w2_num, atol=1e-12)


def test_real_space_ring_equals_bloch_spectrum():
    for p in CASES:
        for n in (1, 2, 7, 30):
            fr = rs.solve_finite_chain(p, n).omega
            fb = np.sort(dp.omega(p, dp.allowed_wavevectors(p, n)).ravel())
            assert np.allclose(fr, fb, atol=1e-6), (p, n)


def test_acoustic_sum_rule_and_zero_mode():
    for p in CASES:
        K = rs.stiffness_matrix(p, 10, "periodic")
        assert np.allclose(K.sum(axis=1), 0.0)            # rigid translation costs nothing
        assert rs.solve_finite_chain(p, 10).omega[0] < 1e-6


def test_band_edges_and_sound_velocity():
    for p in CASES:
        lo, hi = p.zone_boundary_frequencies
        w_pi = dp.omega(p, np.pi / p.a)
        assert np.isclose(w_pi[0], lo) and np.isclose(w_pi[1], hi)
        assert np.isclose(dp.omega(p, 0.0)[1], p.omega_max)
        k = 1e-6 / p.a
        assert np.isclose(dp.omega_acoustic(p, k) / k, p.sound_velocity, rtol=1e-8)
    # Textbook formula for equal springs
    p = REF
    assert np.isclose(p.sound_velocity, p.a * np.sqrt(p.K1 / (2 * (p.m1 + p.m2))))


def test_small_k_acoustic_branch_is_accurate():
    """Vieta form avoids catastrophic cancellation: omega_- = v_s k down to tiny k."""
    k = np.array([1e-12, 1e-9, 1e-6])
    assert np.allclose(dp.omega_acoustic(REF, k), REF.sound_velocity * k, rtol=1e-6)


def test_group_velocity_matches_finite_differences():
    k = np.linspace(0.05, 3.0, 40) / GENERAL.a
    h = 1e-6
    num = (dp.omega(GENERAL, k + h) - dp.omega(GENERAL, k - h)) / (2 * h)
    assert np.allclose(dp.group_velocity(GENERAL, k), num, atol=1e-7)


def test_monatomic_limit_is_folded_sine_dispersion():
    """m1 = m2, K1 = K2: two branches = 2 sqrt(K/m) |sin(q a'/2)| with a' = a/2."""
    p = ChainParameters(m1=1.0, m2=1.0, K1=1.0, K2=1.0, a=1.0)
    k = np.linspace(0, np.pi, 50)
    w = dp.omega(p, k)
    ap = 0.5
    assert np.allclose(w[:, 0], 2 * np.abs(np.sin(k * ap / 2)), atol=1e-12)
    assert np.allclose(w[:, 1], 2 * np.abs(np.sin((k - 2 * np.pi) * ap / 2)), atol=1e-12)


def test_optical_mode_at_k0_keeps_centre_of_mass_fixed():
    r = dp.amplitude_ratio(REF, 0.0)
    assert np.isclose(r[0], 1.0) and np.isclose(r[1], -REF.m1 / REF.m2)


def test_trace_gives_second_moment():
    k = np.linspace(-np.pi, np.pi, 2001)[:-1]
    w2 = dp.omega_squared(GENERAL, k / GENERAL.a)
    assert np.isclose(w2.mean(axis=0).sum() / 2, GENERAL.mean_square_frequency)


# ---------------------------------------------------------------- DOS
def test_dos_normalisation_and_low_frequency_limit():
    for p in CASES:
        assert np.isclose(ds.integrated_dos(p, [10 * p.omega_max])[0], 2.0)
        lo, hi = p.zone_boundary_frequencies
        if hi - lo > 1e-9:
            assert np.isclose(ds.integrated_dos(p, [0.5 * (lo + hi)])[0], 1.0)
        assert np.isclose(ds.analytic_dos(p, [1e-7])[0], p.a / (np.pi * p.sound_velocity), rtol=1e-6)


def test_histogram_routes_agree_with_exact_bin_averages():
    p = GENERAL
    edges = ds.frequency_grid(p, 40)
    exact = ds.bin_averaged_dos(p, edges)
    sampled = ds.sampled_dos(p, 400_000, edges)
    assert np.isclose(sampled.total(), 2.0)
    assert np.max(np.abs(sampled.values - exact)) < 1e-3
    finite = ds.histogram_dos(rs.solve_finite_chain(p, 300).omega, 300, edges)
    assert np.isclose(finite.total(), 2.0)


def test_analytic_dos_is_derivative_of_integrated_dos():
    p = REF
    w = np.array([0.3, 0.8, 1.5, 1.65])
    h = 1e-6
    num = (ds.integrated_dos(p, w + h) - ds.integrated_dos(p, w - h)) / (2 * h)
    assert np.allclose(ds.analytic_dos(p, w), num, rtol=1e-5)


# ---------------------------------------------------------------- thermodynamics
def test_heat_capacity_limits():
    for p in CASES:
        # Low T: 1D Debye law, C = (pi/3) a T / v_s
        T = np.array([1e-4, 1e-3])
        assert np.allclose(th.heat_capacity(p, T), th.low_temperature_heat_capacity(p, T), rtol=1e-4)
        # High T: 2 (1 - <w^2>/(12 T^2))
        T = np.array([20.0, 50.0])
        assert np.allclose(th.heat_capacity(p, T), th.high_temperature_heat_capacity(p, T), rtol=1e-6)


def test_bz_quadrature_equals_dos_integral_and_finite_chain():
    from scipy.integrate import quad
    p = REF
    lo, hi = p.zone_boundary_frequencies
    for T in (0.1, 0.5, 1.5):
        f = lambda w: ds.analytic_dos(p, [w])[0] * th.mode_heat_capacity(w, T)   # noqa: E731
        c_dos = quad(f, 0, lo, limit=400)[0] + quad(f, hi, p.omega_max, limit=400)[0]
        c_bz = th.heat_capacity(p, [T])[0]
        c_fin = th.finite_chain_heat_capacity(rs.solve_finite_chain(p, 400).omega, 400, [T])[0]
        assert abs(c_dos - c_bz) < 1e-9
        assert abs(c_fin - c_bz) < 1e-4


def test_thermodynamic_consistency():
    """S = -dF/dT and C = dU/dT (finite differences)."""
    p = GENERAL
    T0, h = 0.4, 1e-5
    r = th.phonon_thermodynamics(p, [T0 - h, T0, T0 + h])
    assert np.isclose(-(r.free_energy[2] - r.free_energy[0]) / (2 * h), r.entropy[1], rtol=1e-6)
    assert np.isclose((r.energy[2] - r.energy[0]) / (2 * h), r.heat_capacity[1], rtol=1e-6)
    assert np.allclose(r.energy, r.free_energy + r.T * r.entropy)


def test_branch_decomposition_and_zero_point():
    C_ac, C_op = th.branch_heat_capacity(REF, [0.3, 1.0])
    assert np.allclose(C_ac + C_op, th.heat_capacity(REF, [0.3, 1.0]))
    # Zero-point energy equals U(T -> 0)
    assert np.isclose(th.phonon_thermodynamics(REF, [1e-4]).energy[0], th.zero_point_energy(REF), rtol=1e-6)


def test_classical_limit_dulong_petit():
    r = th.phonon_thermodynamics(REF, [0.3, 3.0], quantum=False)
    assert np.allclose(r.heat_capacity, 2.0) and np.allclose(r.energy, 2 * r.T)


# ---------------------------------------------------------------- models
def test_debye_function_limits():
    assert np.isclose(md.debye_function_1d([1e-6])[0], 1.0)
    assert np.isclose(md.debye_function_1d([1e4])[0], np.pi ** 2 / 3 / 1e4, rtol=1e-10)


def test_debye_model_is_exact_at_low_temperature():
    T = np.array([1e-4, 5e-4])
    C = th.heat_capacity(REF, T)
    assert np.allclose(md.debye_heat_capacity(T, md.debye_frequency(REF)), C, rtol=1e-4)
    assert np.allclose(md.effective_debye_temperature(T, C), md.debye_frequency(REF), rtol=1e-3)


def test_high_temperature_debye_temperature():
    """Theta_D(T -> inf) = sqrt(3 <omega^2>) (Debye second moment: omega_D^2 / 3)."""
    T = np.array([30.0])
    C = th.heat_capacity(REF, T)
    thD = md.effective_debye_temperature(T, C)[0]
    assert np.isclose(thD, np.sqrt(3 * REF.mean_square_frequency), rtol=1e-3)


def test_einstein_second_moment_matches_high_T_expansion():
    T = np.array([40.0])
    wE = md.einstein_frequency_second_moment(REF)
    assert np.isclose(md.einstein_heat_capacity(T, wE)[0], th.heat_capacity(REF, T)[0], rtol=1e-8)


# ---------------------------------------------------------------- observables
def test_classical_relative_displacement_is_exact_random_walk():
    for p in (REF, GENERAL):
        r = np.array([1, 3, 10, 50])
        G = ob.displacement_correlation(p, r, 0.7, quantum=False)
        assert np.allclose(G, r * 0.7 * (1 / p.K1 + 1 / p.K2), rtol=1e-8)
        b1, b2 = ob.bond_fluctuations(p, 0.7, quantum=False)
        assert np.isclose(b1, 0.7 / p.K1) and np.isclose(b2, 0.7 / p.K2)


def test_zero_temperature_logarithmic_growth():
    G = ob.displacement_correlation(REF, [1000, 10000], 0.0)
    slope = (G[1] - G[0]) / np.log(10)
    assert np.isclose(slope, REF.a / (np.pi * REF.total_mass * REF.sound_velocity), rtol=1e-3)


def test_classical_mean_square_displacement_grows_linearly_with_N():
    u2 = ob.mean_square_displacement(REF, 512, 1.0, quantum=False)[0]
    assert np.isclose(u2 / 512, (1 / REF.K1 + 1 / REF.K2) / 12, rtol=1e-4)


# ---------------------------------------------------------------- Langevin
def test_langevin_equipartition_short_run():
    res = lg.simulate(REF, n_cells=16, T=0.5, gamma=1.0, dt=0.05, n_equil=2000,
                      n_samples=4000, stride=5, seed=7)
    assert np.allclose(res.mean_kinetic_temperature(), 0.5, rtol=0.05)
    assert np.allclose(res.bond_sq.mean(axis=0), lg.exact_ring_bond_variance(REF, 16, 0.5), rtol=0.06)


def test_velocity_verlet_conserves_energy():
    chain = lg.LangevinChain(REF, 20, seed=3)
    chain.thermalise_momenta(1.0)
    E0 = chain.energy()
    for _ in chain.run(1.0, 0.0, 0.02, 5000, record=False):
        pass
    assert abs(chain.energy() - E0) / E0 < 1e-3
