"""
plotting.py
===========

Shared matplotlib style so that every figure of the project looks consistent.

Design choices
--------------
* A fixed categorical colour order (colour-blind-checked palette): the same
  physical object always gets the same colour in every figure
  (acoustic branch -> blue, optical branch -> orange, exact result -> ink,
  Debye -> aqua, Einstein -> violet, hybrid -> magenta, simulation -> green).
* Thin 2 px lines, light recessive grid, no top/right spines, text always in
  neutral ink (never in the series colour).
* One y-axis per panel (never twin axes).

Data types: colours are hex ``str``; ``ROLE_COLORS`` is a plain ``dict`` from
role name to colour so scripts refer to *meaning* ("acoustic") instead of raw
hex values.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "red": "#e34948",
}

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

ROLE_COLORS = {
    "acoustic": PALETTE["blue"],
    "optical": PALETTE["orange"],
    "exact": INK,
    "debye": PALETTE["aqua"],
    "einstein": PALETTE["violet"],
    "hybrid": PALETTE["magenta"],
    "simulation": PALETTE["green"],
    "classical": INK_3,
    "sublattice1": PALETTE["blue"],
    "sublattice2": PALETTE["orange"],
}

# Ordered list used when a figure shows a *family* of curves (e.g. mass ratios).
SERIES = [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["violet"]]


def use_style() -> None:
    """Apply the project style to matplotlib (call once per script)."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK,
    })


def panel_label(ax, text: str) -> None:
    """Small bold panel tag '(a)', '(b)'... in the top-left corner."""
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="bold", color=INK)


def zone_ticks(ax, a: float = 1.0) -> None:
    """Label the k axis of a dispersion plot at -pi/a, 0, pi/a."""
    import numpy as np
    ax.set_xticks([-np.pi / a, -0.5 * np.pi / a, 0.0, 0.5 * np.pi / a, np.pi / a])
    ax.set_xticklabels([r"$-\pi/a$", r"$-\pi/2a$", "0", r"$\pi/2a$", r"$\pi/a$"])
    ax.set_xlim(-np.pi / a, np.pi / a)


def save(fig, path) -> None:
    """Save and close (keeps memory flat when many figures are produced)."""
    fig.savefig(path)
    plt.close(fig)
