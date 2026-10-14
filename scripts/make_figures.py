#!/usr/bin/env python3
"""
make_figures.py
===============

Reproduce every figure of the project (``figures/*.png``) and the table of
key numbers (``results/results.json``) quoted in the documentation.

Usage (from the repository root)::

    python scripts/make_figures.py            # everything (~1-2 min)
    python scripts/make_figures.py --quick    # shorter Langevin runs

Program structure
-----------------
main()
 |- fig_dispersion()          omega_{+-}(k) for several mass ratios + finite chain
 |- fig_eigenvectors()        amplitude ratio u2/u1 and real-space mode patterns
 |- fig_gap_map()             band edges vs m2/m1 and vs K2/K1 (general case)
 |- fig_dos()                 analytic / finite-chain DOS vs Debye and Einstein
 |- fig_heat_capacity()       C(T): exact vs models, linear and log-log
 |- fig_branches()            acoustic and optical contributions to C(T)
 |- fig_model_errors()        relative model errors for several mass ratios
 |- fig_debye_temperature()   effective Theta_D(T)
 |- fig_thermo_functions()    U, F, S per cell (quantum vs classical)
 |- fig_displacements()       G(r), <u^2>(N), bond fluctuations
 |- fig_langevin()            BAOAB simulation vs exact results
 '- results.json              every number quoted in the documents

Each figure function receives the reference ``ChainParameters`` and a
``dict`` (``results``) into which it writes the numbers it computes; the dict
is dumped to JSON at the end so the documents never quote a value that was not
produced by the code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running the script without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diatomic_chain import ChainParameters                         # noqa: E402
from diatomic_chain import dispersion as dp                        # noqa: E402
from diatomic_chain import real_space as rs                        # noqa: E402
from diatomic_chain import dos as ds                               # noqa: E402
from diatomic_chain import thermodynamics as th                    # noqa: E402
from diatomic_chain import models as md                            # noqa: E402
from diatomic_chain import observables as ob                       # noqa: E402
from diatomic_chain import langevin as lg                          # noqa: E402
from diatomic_chain.plotting import (use_style, panel_label, zone_ticks, save,  # noqa: E402
                                     ROLE_COLORS as RC, SERIES, INK_2, INK_3, GRID)

FIG = ROOT / "figures"
RES = ROOT / "results"


# ====================================================================== #
# 1. Dispersion relations                                                 #
# ====================================================================== #
def fig_dispersion(p: ChainParameters, results: dict) -> None:
    ratios = [1.0, 2.0, 5.0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    k = dp.brillouin_zone(p, 801)
    n_cells = 16
    for ax, r, tag in zip(axes, ratios, "abc"):
        q = ChainParameters.textbook(mass_ratio=r)
        w = dp.omega(q, k)
        lo, hi = q.zone_boundary_frequencies
        if hi - lo > 1e-9:
            ax.axhspan(lo, hi, color=GRID, alpha=0.8, lw=0)
            ax.text(0.0, 0.5 * (lo + hi), "gap", ha="center", va="center",
                    color=INK_2, fontsize=9)
        ax.plot(k, w[:, 0], color=RC["acoustic"], label=r"acoustic $\omega_-(k)$")
        ax.plot(k, w[:, 1], color=RC["optical"], label=r"optical $\omega_+(k)$")
        # Long-wavelength (Debye) line and Einstein level
        ax.plot(k, q.sound_velocity * np.abs(k), ls="--", lw=1.2, color=RC["debye"],
                label=r"Debye $v_s|k|$")
        wE = md.optical_mean_frequency(q)
        ax.axhline(wE, ls=":", lw=1.4, color=RC["einstein"],
                   label=r"$\langle\omega_+\rangle_{BZ}$ (Einstein)")
        # Finite chain (real-space diagonalisation, N = 16 cells, BvK)
        kk = dp.allowed_wavevectors(q, n_cells)
        fr = rs.solve_finite_chain(q, n_cells).omega
        wk = np.sort(dp.omega(q, kk).ravel())
        assert np.allclose(fr, wk, atol=1e-7), "real-space and Bloch spectra differ"
        ax.plot(kk, dp.omega(q, kk)[:, 0], "o", ms=4, mfc="white", mec=RC["acoustic"], mew=1.3)
        ax.plot(kk, dp.omega(q, kk)[:, 1], "o", ms=4, mfc="white", mec=RC["optical"], mew=1.3)
        zone_ticks(ax, q.a)
        ax.set_ylim(0, 2.1)
        ax.set_xlabel("wave number $k$")
        ax.set_title(f"$m_2/m_1 = {r:g}$" + ("  (zone folded)" if r == 1 else ""))
        panel_label(ax, f"({tag})")
    axes[0].set_ylabel(r"frequency $\omega$  [$\sqrt{K/m_1}$]")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[1].text(0.03, 0.97, "open circles: $N=16$ ring, real-space\ndiagonalisation of $\\det(K-\\omega^2M)=0$",
                 transform=axes[1].transAxes, fontsize=8, color=INK_2, va="top")
    save(fig, FIG / "dispersion.png")

    results["reference_parameters"] = dict(m1=p.m1, m2=p.m2, K1=p.K1, K2=p.K2, a=p.a)
    lo, hi = p.zone_boundary_frequencies
    results["reference_spectrum"] = dict(
        sound_velocity=p.sound_velocity, acoustic_top=lo, optical_bottom=hi,
        optical_top=p.omega_max, gap=p.gap, sqrt_mean_omega2=np.sqrt(p.mean_square_frequency),
        optical_mean=md.optical_mean_frequency(p), debye_frequency=md.debye_frequency(p),
        debye_frequency_acoustic=md.debye_frequency(p, 1.0))
    # Maximum discrepancy real-space vs Bloch for a larger, general chain
    g = ChainParameters(m1=1.0, m2=3.0, K1=1.0, K2=2.5)
    err = np.abs(rs.solve_finite_chain(g, 60).omega
                 - np.sort(dp.omega(g, dp.allowed_wavevectors(g, 60)).ravel())).max()
    results["realspace_vs_bloch_max_abs_diff"] = float(err)


# ====================================================================== #
# 2. Eigenvectors                                                         #
# ====================================================================== #
def fig_eigenvectors(p: ChainParameters, results: dict) -> None:
    fig = plt.figure(figsize=(12, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], wspace=0.28, hspace=0.55)
    ax0 = fig.add_subplot(gs[:, 0])
    k = np.linspace(1e-4, np.pi / p.a, 400)
    ratio = dp.amplitude_ratio(p, k)
    # Multiplying by e^{-ika/2} measures each atom's phase from its own position
    # (x_2 = x_1 + a/2). For K1 = K2 the ratio becomes real:
    #     A2/A1 = (2K - m1 omega^2) / (2K cos(ka/2)),
    # positive = in phase, negative = antiphase.
    signed = np.real(ratio * np.exp(-0.5j * k * p.a)[:, None])
    ax0.plot(k, signed[:, 0], color=RC["acoustic"], label="acoustic")
    ax0.plot(k, signed[:, 1], color=RC["optical"], label="optical")
    ax0.axhline(0, color=INK_3, lw=0.8)
    ax0.axhline(-p.m1 / p.m2, ls=":", lw=1.2, color=INK_2)
    ax0.text(0.03, -p.m1 / p.m2 - 0.32, r"$-m_1/m_2$: centre of mass at rest",
             color=INK_2, fontsize=8.5)
    ax0.set_ylim(-2.5, 4.0)
    ax0.set_xticks([0, np.pi / 2 / p.a, np.pi / p.a])
    ax0.set_xticklabels(["0", r"$\pi/2a$", r"$\pi/a$"])
    ax0.set_xlabel("wave number $k$")
    ax0.set_ylabel(r"amplitude ratio $u_2/u_1$ (Bloch phase removed)")
    ax0.set_title("Relative motion of the two atoms")
    ax0.legend(loc="center right")
    panel_label(ax0, "(a)")

    # Real-space snapshots for N = 8 cells (BvK): k = 2 pi/(8a) and k = pi/a
    n_cells = 8
    x_atoms = np.arange(2 * n_cells) * 0.5 * p.a
    cases = [(2 * np.pi / (n_cells * p.a), 0, "acoustic, $k=2\\pi/8a$"),
             (np.pi / p.a, 0, "acoustic, $k=\\pi/a$"),
             (0.0, 1, "optical, $k=0$"),
             (np.pi / p.a, 1, "optical, $k=\\pi/a$")]
    for idx, (kk, b, title) in enumerate(cases):
        ax = fig.add_subplot(gs[idx % 2, 1 + idx // 2])
        _, e = dp.normal_modes(p, np.array([kk]))
        n = np.arange(n_cells)
        amp = np.empty(2 * n_cells)
        u1 = e[0, 0, b] / np.sqrt(p.m1) * np.exp(1j * kk * n * p.a)
        u2 = e[0, 1, b] / np.sqrt(p.m2) * np.exp(1j * kk * n * p.a)
        # choose the global phase that makes the snapshot maximal (real part)
        z = np.concatenate([u1, u2])
        phase = np.exp(-1j * np.angle(z[np.argmax(np.abs(z))]))
        amp[0::2] = np.real(u1 * phase)
        amp[1::2] = np.real(u2 * phase)
        amp /= np.abs(amp).max()
        cols = [RC["sublattice1"] if i % 2 == 0 else RC["sublattice2"] for i in range(2 * n_cells)]
        ax.axhline(0, color=INK_3, lw=0.8)
        ax.vlines(x_atoms, 0, amp, colors=cols, lw=2)
        ax.scatter(x_atoms, amp, c=cols, s=[18 if i % 2 == 0 else 34 for i in range(2 * n_cells)], zorder=3)
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        if idx == 0:
            ax.text(0.0, 1.32, "displacement $u_{s,n}$ (small dot $m_1$, large dot $m_2$)",
                    transform=ax.transAxes, fontsize=8.5, color=INK_2)
    save(fig, FIG / "eigenvectors.png")
    results["optical_k0_amplitude_ratio"] = float(np.real(dp.amplitude_ratio(p, 0.0)[1]))


# ====================================================================== #
# 3. Band edges vs parameters (general case K1 != K2)                    #
# ====================================================================== #
def fig_gap_map(p: ChainParameters, results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    mr = np.geomspace(0.1, 10.0, 300)
    edges = np.array([ChainParameters(m1=1.0, m2=r, K1=1.0, K2=1.0).zone_boundary_frequencies
                      + (ChainParameters(m1=1.0, m2=r).omega_max,) for r in mr])
    ax = axes[0]
    ax.fill_between(mr, 0, edges[:, 0], color=RC["acoustic"], alpha=0.25, lw=0, label="acoustic band")
    ax.fill_between(mr, edges[:, 1], edges[:, 2], color=RC["optical"], alpha=0.25, lw=0, label="optical band")
    ax.plot(mr, edges[:, 0], color=RC["acoustic"], lw=1.5)
    ax.plot(mr, edges[:, 1], color=RC["optical"], lw=1.5)
    ax.plot(mr, edges[:, 2], color=RC["optical"], lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"mass ratio $m_2/m_1$  ($K_1=K_2$)")
    ax.set_ylabel(r"frequency $\omega$  [$\sqrt{K_1/m_1}$]")
    ax.set_title("Gap opened by a mass contrast")
    ax.legend(loc="upper right")
    panel_label(ax, "(a)")

    kr = np.geomspace(0.1, 10.0, 300)
    edges = np.array([ChainParameters(m1=1.0, m2=1.0, K1=1.0, K2=r).zone_boundary_frequencies
                      + (ChainParameters(m1=1.0, m2=1.0, K1=1.0, K2=r).omega_max,) for r in kr])
    ax = axes[1]
    ax.fill_between(kr, 0, edges[:, 0], color=RC["acoustic"], alpha=0.25, lw=0)
    ax.fill_between(kr, edges[:, 1], edges[:, 2], color=RC["optical"], alpha=0.25, lw=0)
    ax.plot(kr, edges[:, 0], color=RC["acoustic"], lw=1.5)
    ax.plot(kr, edges[:, 1], color=RC["optical"], lw=1.5)
    ax.plot(kr, edges[:, 2], color=RC["optical"], lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"stiffness ratio $K_2/K_1$  ($m_1=m_2$)")
    ax.set_title("Gap opened by a stiffness contrast")
    panel_label(ax, "(b)")
    save(fig, FIG / "gap_map.png")


# ====================================================================== #
# 4. Density of states                                                    #
# ====================================================================== #
def fig_dos(p: ChainParameters, results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    w = np.linspace(1e-6, 1.05 * p.omega_max, 6000)
    g = ds.analytic_dos(p, w)
    n_cells = 1200
    fr = rs.solve_finite_chain(p, n_cells).omega         # real space, 2400 x 2400
    edges = ds.frequency_grid(p, 70)
    hist = ds.histogram_dos(fr, n_cells, edges)
    ax.bar(hist.centers, hist.values, width=hist.width, color=GRID, edgecolor="white",
           lw=0.4, label=f"finite ring, $N={n_cells}$ (real space)")
    ax.plot(w, np.minimum(g, 50), color=RC["exact"], lw=1.6, label="exact $g(\\omega)$")
    wD = md.debye_frequency(p)
    ax.plot([0, wD, wD], [2 / wD, 2 / wD, 0], color=RC["debye"], ls="--", lw=1.6,
            label=r"Debye, $\omega_D=2\pi v_s/a$")
    wD_ac = md.debye_frequency(p, 1.0)
    ax.plot([0, wD_ac, wD_ac], [1 / wD_ac, 1 / wD_ac, 0], color=RC["hybrid"], lw=1.4,
            label=r"Debye ($\pi v_s/a$) + Einstein ($\langle\omega_+\rangle$)")
    wE = md.optical_mean_frequency(p)
    ax.annotate("", xy=(wE, 4.0), xytext=(wE, 0), arrowprops=dict(arrowstyle="-|>", color=RC["hybrid"], lw=1.4))
    wE2 = md.einstein_frequency_second_moment(p)
    ax.annotate("", xy=(wE2, 4.0), xytext=(wE2, 0), arrowprops=dict(arrowstyle="-|>", color=RC["einstein"], lw=1.6))
    ax.text(1.86, 3.3, "arrows = delta peaks\nviolet: Einstein\npink: hybrid (optical)",
            color=INK_2, fontsize=8.5, va="top")
    ax.set_ylim(0, 4.5)
    ax.set_xlim(0, 1.05 * max(p.omega_max, wD))
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"$g(\omega)$  [modes per cell per unit $\omega$]")
    ax.set_title("Phonon density of states ($m_2/m_1=2$)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    panel_label(ax, "(a)")

    ax = axes[1]
    wl = np.linspace(0, 1.05 * max(p.omega_max, wD), 2000)
    ax.plot(wl, ds.integrated_dos(p, wl), color=RC["exact"], label="exact $N(\\omega)$")
    ax.plot(wl, np.minimum(2 * wl / wD, 2), color=RC["debye"], ls="--", label="Debye")
    ax.plot(wl, np.minimum(wl / wD_ac, 1) + (wl >= wE), color=RC["hybrid"], label="Debye + Einstein")
    ax.plot(wl, 2.0 * (wl >= wE2), color=RC["einstein"], ls=":", label="Einstein")
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"$N(\omega)=\int_0^\omega g$  [modes per cell]")
    ax.set_title("Integrated DOS (mode counting)")
    ax.legend(loc="lower right")
    panel_label(ax, "(b)")
    save(fig, FIG / "dos.png")

    # Numbers: normalisation and agreement of the three DOS routes
    sampled = ds.sampled_dos(p, 200_000, edges)
    exact_bins = ds.bin_averaged_dos(p, edges)
    results["dos"] = dict(
        normalisation_sampled=sampled.total(),
        normalisation_finite=hist.total(),
        max_abs_diff_sampled_vs_exact_bins=float(np.abs(sampled.values - exact_bins).max()),
        g0_exact=float(p.a / (np.pi * p.sound_velocity)),
        debye_flat_value=2 / wD,
    )


# ====================================================================== #
# 5. Heat capacity: exact vs models                                       #
# ====================================================================== #
def fig_heat_capacity(p: ChainParameters, results: dict) -> None:
    T = np.geomspace(1e-3, 10.0, 400)
    C = th.heat_capacity(p, T)
    wD = md.debye_frequency(p)
    wE2 = md.einstein_frequency_second_moment(p)
    CD = md.debye_heat_capacity(T, wD)
    CE = md.einstein_heat_capacity(T, wE2)
    CH = md.hybrid_heat_capacity(p, T)
    # Least-squares fits over a window covering the crossover
    Tfit = np.geomspace(0.02, 3.0, 200)
    Cfit = th.heat_capacity(p, Tfit)
    fD = md.fit_debye(Tfit, Cfit)
    fE = md.fit_einstein(Tfit, Cfit)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    lin = T <= 2.5
    ax.axhline(2.0, color=RC["classical"], lw=1, ls="-")
    ax.text(2.45, 2.03, "Dulong–Petit", ha="right", va="bottom", color=INK_2, fontsize=8.5)
    ax.plot(T[lin], C[lin], color=RC["exact"], lw=2.4, label="exact (numerical BZ integral)")
    ax.plot(T[lin], CD[lin], color=RC["debye"], ls="--", label=fr"Debye, $\Theta_D=2\pi v_s/a={wD:.2f}$")
    ax.plot(T[lin], CE[lin], color=RC["einstein"], ls=":", lw=2.2,
            label=fr"Einstein, $\Theta_E=\sqrt{{\langle\omega^2\rangle}}={wE2:.2f}$")
    ax.plot(T[lin], CH[lin], color=RC["hybrid"], label="Debye (acoustic) + Einstein (optical)")
    ax.set_xlabel(r"temperature $k_BT/\hbar\sqrt{K/m_1}$")
    ax.set_ylabel(r"$C_V/(N_{cells}\,k_B)$")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.2)
    ax.set_title("Heat capacity per unit cell")
    ax.legend(loc="lower right", fontsize=8)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.loglog(T, C, color=RC["exact"], lw=2.4, label="exact")
    ax.loglog(T, CD, color=RC["debye"], ls="--", label="Debye")
    ax.loglog(T, np.clip(CE, 1e-12, None), color=RC["einstein"], ls=":", lw=2.2, label="Einstein")
    ax.loglog(T, CH, color=RC["hybrid"], label="Debye + Einstein")
    low = T < 0.08
    ax.loglog(T[low], th.low_temperature_heat_capacity(p, T[low]), color=INK_2, lw=1, ls="-.",
              label=r"$\frac{\pi}{3}\,\frac{a\,k_BT}{\hbar v_s}$ (1D law)")
    ax.set_ylim(1e-3, 4)
    ax.set_xlim(1e-3, 10)
    ax.set_xlabel(r"temperature $k_BT/\hbar\sqrt{K/m_1}$")
    ax.set_ylabel(r"$C_V/(N_{cells}\,k_B)$")
    ax.set_title("Low-temperature behaviour (log–log)")
    ax.legend(loc="lower right", fontsize=8)
    panel_label(ax, "(b)")
    save(fig, FIG / "heat_capacity.png")

    def rel(model):
        return float(np.max(np.abs(model - C) / C))

    tsel = [0.01, 0.1, 0.3, 0.5, 1.0, 2.0]
    Csel = th.heat_capacity(p, tsel)
    results["heat_capacity"] = dict(
        table=[dict(T=t, exact=float(c),
                    debye=float(md.debye_heat_capacity([t], wD)[0]),
                    einstein=float(md.einstein_heat_capacity([t], wE2)[0]),
                    hybrid=float(md.hybrid_heat_capacity(p, [t])[0]))
               for t, c in zip(tsel, Csel)],
        low_T_slope_exact=float(th.heat_capacity(p, [1e-4])[0] / 1e-4),
        low_T_slope_theory=float(np.pi * p.a / (3 * p.sound_velocity)),
        max_rel_error_debye=rel(CD), max_rel_error_hybrid=rel(CH),
        fit_debye=dict(theta=fD.theta, rms=fD.rms_error, max_rel=fD.max_rel_error),
        fit_einstein=dict(theta=fE.theta, rms=fE.rms_error, max_rel=fE.max_rel_error),
        high_T_check=dict(T=10.0, exact=float(C[-1]),
                          expansion=float(th.high_temperature_heat_capacity(p, 10.0))),
    )


def fig_branches(p: ChainParameters, results: dict) -> None:
    T = np.linspace(0.005, 2.5, 300)
    Cac, Cop = th.branch_heat_capacity(p, T)
    wD_ac = md.debye_frequency(p, 1.0)
    wE = md.optical_mean_frequency(p)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(T, Cac + Cop, color=RC["exact"], lw=2.4, label="total (exact)")
    ax.plot(T, Cac, color=RC["acoustic"], label="acoustic branch (exact)")
    ax.plot(T, Cop, color=RC["optical"], label="optical branch (exact)")
    ax.plot(T, md.debye_heat_capacity(T, wD_ac, 1.0), color=RC["acoustic"], ls="--", lw=1.4,
            label=r"Debye, 1 mode/cell, $\omega_D^{ac}=\pi v_s/a$")
    ax.plot(T, md.einstein_heat_capacity(T, wE, 1.0), color=RC["optical"], ls="--", lw=1.4,
            label=r"Einstein, $\omega_E=\langle\omega_+\rangle_{BZ}$")
    ax.set_xlabel(r"temperature $k_BT/\hbar\sqrt{K/m_1}$")
    ax.set_ylabel(r"$C_V/(N_{cells}\,k_B)$")
    ax.set_title("Branch-resolved heat capacity ($m_2/m_1=2$)")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.1)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, FIG / "heat_capacity_branches.png")
    a, o = th.branch_heat_capacity(p, [0.1, 0.25, 0.5])
    results["branch_fraction_optical"] = {f"T={t}": float(oo / (aa + oo))
                                          for t, aa, oo in zip([0.1, 0.25, 0.5], a, o)}


def fig_model_errors(p: ChainParameters, results: dict) -> None:
    ratios = [1.0, 2.0, 5.0, 20.0]
    T = np.geomspace(5e-3, 5.0, 300)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    out = {}
    for col, r in zip(SERIES, ratios):
        q = ChainParameters.textbook(mass_ratio=r)
        C = th.heat_capacity(q, T)
        eD = (md.debye_heat_capacity(T, md.debye_frequency(q)) - C) / C
        eH = (md.hybrid_heat_capacity(q, T) - C) / C
        x = T / md.debye_frequency(q)
        axes[0].semilogx(x, 100 * eD, color=col, label=f"$m_2/m_1={r:g}$")
        axes[1].semilogx(x, 100 * eH, color=col, label=f"$m_2/m_1={r:g}$")
        out[f"mass_ratio={r:g}"] = dict(max_abs_rel_err_debye=float(np.abs(eD).max()),
                                        max_abs_rel_err_hybrid=float(np.abs(eH).max()))
    for ax, title, tag in zip(axes, ["Debye (2 modes/cell, $\\Theta_D=2\\pi v_s/a$)",
                                     "Debye (acoustic) + Einstein (optical)"], "ab"):
        ax.axhline(0, color=INK_3, lw=0.8)
        ax.set_xlabel(r"reduced temperature $T/\Theta_D$")
        ax.set_title(title)
        panel_label(ax, f"({tag})")
    axes[0].set_ylabel(r"relative error $(C_{model}-C)/C$  [%]")
    axes[0].legend(loc="lower left")
    save(fig, FIG / "model_errors.png")
    results["model_errors_vs_mass_ratio"] = out


def fig_debye_temperature(p: ChainParameters, results: dict) -> None:
    ratios = [1.0, 2.0, 5.0, 20.0]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    out = {}
    for col, r in zip(SERIES, ratios):
        q = ChainParameters.textbook(mass_ratio=r)
        th0 = md.debye_frequency(q)
        T = np.geomspace(1e-3, 3.0, 220) * th0
        C = th.heat_capacity(q, T)
        thD = md.effective_debye_temperature(T, C)
        ax.semilogx(T / th0, thD / th0, color=col, label=f"$m_2/m_1={r:g}$")
        out[f"mass_ratio={r:g}"] = dict(theta_D0=th0, min_ratio=float(np.nanmin(thD / th0)),
                                        high_T_ratio=float(thD[np.isfinite(thD)][-1] / th0))
    ax.axhline(1.0, color=INK_3, lw=0.8)
    ax.set_xlabel(r"$T/\Theta_D(0)$,   $\Theta_D(0)=2\pi\hbar v_s/(a k_B)$")
    ax.set_ylabel(r"effective $\Theta_D(T)/\Theta_D(0)$")
    ax.set_title("Effective Debye temperature")
    ax.legend(loc="upper left")
    save(fig, FIG / "debye_temperature.png")
    results["effective_debye_temperature"] = out


# ====================================================================== #
# 6. Other thermodynamic functions                                        #
# ====================================================================== #
def fig_thermo_functions(p: ChainParameters, results: dict) -> None:
    T = np.linspace(0.01, 2.5, 300)
    q = th.phonon_thermodynamics(p, T, quantum=True)
    c = th.phonon_thermodynamics(p, T, quantum=False)
    E0 = th.zero_point_energy(p)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    ax.plot(T, q.energy, color=RC["exact"], label="quantum $U/N$")
    ax.plot(T, c.energy, color=RC["classical"], ls="--", label=r"classical $2k_BT$")
    ax.axhline(E0, color=INK_2, lw=0.8)
    ax.text(1.25, E0 - 0.45, fr"zero point $E_0/N={E0:.3f}$", color=INK_2, fontsize=8.5)
    ax.set_title("Internal energy per cell")
    ax.set_xlabel(r"$k_BT$")
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "(a)")
    ax = axes[1]
    ax.plot(T, q.entropy, color=RC["exact"], label="quantum")
    ax.plot(T, c.entropy, color=RC["classical"], ls="--", label="classical (diverges as $T\\to0$)")
    ax.set_ylim(-3, 6)
    ax.axhline(0, color=INK_3, lw=0.8)
    ax.set_title("Entropy per cell  $S/(N k_B)$")
    ax.set_xlabel(r"$k_BT$")
    ax.legend(loc="lower right", fontsize=8)
    panel_label(ax, "(b)")
    ax = axes[2]
    ax.plot(T, q.free_energy, color=RC["exact"], label="quantum")
    ax.plot(T, c.free_energy, color=RC["classical"], ls="--", label="classical")
    ax.set_title("Helmholtz free energy per cell")
    ax.set_xlabel(r"$k_BT$")
    ax.legend(loc="lower left", fontsize=8)
    panel_label(ax, "(c)")
    save(fig, FIG / "thermo_functions.png")
    results["zero_point_energy_per_cell"] = E0


# ====================================================================== #
# 7. Displacement fluctuations                                            #
# ====================================================================== #
def fig_displacements(p: ChainParameters, results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    # (a) G(r) relative displacement
    ax = axes[0]
    r = np.unique(np.geomspace(1, 2000, 40).astype(int))
    for col, T, ls in [(SERIES[0], 0.0, "-"), (SERIES[1], 0.05, "-"), (SERIES[2], 0.2, "-")]:
        G = ob.displacement_correlation(p, r, T, quantum=True)
        ax.loglog(r, G, color=col, ls=ls, label=f"quantum, $k_BT={T:g}$")
    for col, T in [(SERIES[1], 0.05), (SERIES[2], 0.2)]:
        ax.loglog(r, r * T * (1 / p.K1 + 1 / p.K2), color=col, ls=":", lw=1.4)
    ax.loglog([], [], color=INK_2, ls=":", lw=1.4, label=r"classical $r\,k_BT(1/K_1+1/K_2)$")
    ax.set_xlabel("separation $r$ (unit cells)")
    ax.set_ylabel(r"$G(r)=\langle(u_{1,n+r}-u_{1,n})^2\rangle$")
    ax.set_title("Relative displacement $G(r)$")
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "(a)")
    G0 = ob.displacement_correlation(p, [100, 1000], 0.0)
    slope_per_ln = (G0[1] - G0[0]) / np.log(10)

    # (b) <u^2> vs N
    ax = axes[1]
    Ns = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    for col, T in [(SERIES[0], 0.0), (SERIES[1], 0.05), (SERIES[2], 0.2)]:
        u2 = np.array([ob.mean_square_displacement(p, n, T)[0] for n in Ns])
        ax.loglog(Ns, u2, "o-", color=col, ms=4, lw=1.5, label=f"$k_BT={T:g}$")
    ax.set_xlabel("number of cells $N$")
    ax.set_ylabel(r"$\langle u_1^2\rangle$ (translation removed)")
    ax.set_title(r"$\langle u^2\rangle$ vs chain length")
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "(b)")
    u2_T0 = [ob.mean_square_displacement(p, n, 0.0)[0] for n in (256, 2048)]

    # (c) bond fluctuations vs T
    ax = axes[2]
    T = np.linspace(0.002, 1.5, 80)
    bq = np.array([ob.bond_fluctuations(p, t, quantum=True) for t in T])
    ax.plot(T, bq[:, 0], color=RC["exact"], label=r"quantum, $K_1$ bond")
    ax.plot(T, T / p.K1, color=RC["classical"], ls="--", label=r"classical $k_BT/K_1$")
    ax.set_xlabel(r"$k_BT$")
    ax.set_ylabel(r"$\langle(u_{2,n}-u_{1,n})^2\rangle$")
    ax.set_title("Bond-length fluctuation")
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "(c)")
    save(fig, FIG / "displacements.png")
    zp = ob.bond_fluctuations(p, 0.0, quantum=True)
    results["displacements"] = dict(
        G_T0_slope_per_ln_r=float(slope_per_ln),
        G_T0_slope_theory=float(p.a / (np.pi * p.total_mass * p.sound_velocity)),
        u2_T0_N256=float(u2_T0[0]), u2_T0_N2048=float(u2_T0[1]),
        u2_classical_over_N_T=float(ob.mean_square_displacement(p, 1024, 1.0, quantum=False)[0] / 1024),
        u2_classical_theory=float((1 / p.K1 + 1 / p.K2) / 12.0),
        bond_zero_point_K1=zp[0], bond_zero_point_K2=zp[1],
    )


# ====================================================================== #
# 8. Langevin dynamics                                                    #
# ====================================================================== #
def fig_langevin(p: ChainParameters, results: dict, quick: bool) -> None:
    n_cells = 64
    T = 1.0
    n_samples = 2 ** 14 if quick else 2 ** 16
    # Run 1: weak friction -> sharp spectrum
    # (N = 256 so that the mode spacing 2 pi v_s/(N a) ~ 0.01 is below gamma)
    spec = lg.simulate(p, n_cells=256, T=T, gamma=0.03, dt=0.05, n_equil=20_000,
                       n_samples=n_samples // 2, stride=10, seed=1, r_max=1)
    # Run 2: moderate friction -> short correlation time, good fluctuation statistics
    flu = lg.simulate(p, n_cells=n_cells, T=T, gamma=0.5, dt=0.05,
                      n_equil=5_000, n_samples=n_samples, stride=10, seed=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    w = spec.spectrum_omega
    mask = w <= 1.15 * p.omega_max
    ax.plot(w[mask], ds.broadened_dos(p, w[mask], width=0.5 * spec.gamma),
            color=RC["exact"], lw=1.6, label=r"exact $g(\omega)$ * Lorentzian ($\gamma/2$)")
    ax.plot(w[mask], spec.spectrum[mask], color=RC["simulation"], lw=1.2, alpha=0.9,
            label=r"Langevin: $\sum_i m_i|\tilde v_i(\omega)|^2$")
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel(r"$g(\omega)$, normalised to 2 modes/cell")
    ax.set_title(f"DOS from the velocity spectrum ($N={spec.n_cells}$, $\\gamma={spec.gamma}$)")
    ax.set_ylim(0, 4.5)
    ax.legend(loc="upper right", fontsize=8)
    panel_label(ax, "(a)")

    ax = axes[1]
    r = flu.relative_disp_r
    ax.plot(r, lg.exact_ring_relative_displacement(p, n_cells, T, r), color=RC["exact"],
            label=r"exact ring: $r(1-r/N)\,k_BT(1/K_1+1/K_2)$")
    ax.plot(r, flu.relative_disp, "o", ms=4, mfc="white", mec=RC["simulation"], mew=1.3,
            label="Langevin (BAOAB)")
    ax.plot(r, r * T * (1 / p.K1 + 1 / p.K2), color=INK_3, ls=":", lw=1.3,
            label=r"infinite chain $r\,k_BT(1/K_1+1/K_2)$")
    ax.set_xlabel("separation $r$ (cells)")
    ax.set_ylabel(r"$\langle(u_{1,n+r}-u_{1,n})^2\rangle$")
    ax.set_title("Relative displacement on a ring (classical)")
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "(b)")
    save(fig, FIG / "langevin.png")

    C, dC = flu.heat_capacity_per_cell()
    b_exact = lg.exact_ring_bond_variance(p, n_cells, T)
    results["langevin"] = dict(
        n_cells=n_cells, T=T, n_samples=n_samples, dt=0.05,
        heat_capacity=C, heat_capacity_err=dC,
        heat_capacity_exact=lg.exact_ring_heat_capacity(n_cells),
        kinetic_T=flu.mean_kinetic_temperature().tolist(),
        bond_sq=flu.bond_sq.mean(axis=0).tolist(), bond_sq_exact=list(b_exact),
        relative_disp_max_rel_dev=float(np.max(np.abs(
            flu.relative_disp / lg.exact_ring_relative_displacement(p, n_cells, T, r) - 1))),
    )


# ====================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true", help="shorter Langevin runs")
    args = parser.parse_args()

    use_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    p = ChainParameters.textbook(mass_ratio=2.0)       # reference chain
    results: dict = {}
    steps = [fig_dispersion, fig_eigenvectors, fig_gap_map, fig_dos, fig_heat_capacity,
             fig_branches, fig_model_errors, fig_debye_temperature, fig_thermo_functions,
             fig_displacements]
    for f in steps:
        print(f"  - {f.__name__}")
        f(p, results)
    print("  - fig_langevin")
    fig_langevin(p, results, args.quick)

    with open(RES / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"Figures in {FIG}\nNumbers in {RES / 'results.json'}")


if __name__ == "__main__":
    main()
