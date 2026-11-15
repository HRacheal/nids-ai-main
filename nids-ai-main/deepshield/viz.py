"""Shared matplotlib style for every DeepShield figure.

Colours come from a validated colour-blind-safe reference palette:
categorical hues in a fixed order (never cycled), one sequential hue (blue)
for magnitudes such as confusion matrices, and a blue<->red diverging pair
with a grey midpoint for signed values such as correlations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical order. Series 1 = Normal/benign, series 2 = Attack.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
NORMAL_COLOUR, ATTACK_COLOUR = CATEGORICAL[0], CATEGORICAL[1]
SINGLE = CATEGORICAL[0]

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "ds_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf",
                "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "ds_div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2",
               "#e34948", "#a32020"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "text.color": INK, "legend.frameon": False, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
