"""Shared settings for Physical Review Letters figures.

PRL widths: one column 8.6 cm (3.39 in), two columns 17.8 cm (7.0 in). Text is set in a
Times-like serif (STIX, shipped with matplotlib) at 8 pt, so the figure can go in at 100%
scale. Figures are saved as PDF (vector; heatmaps rasterized at `RASTER_DPI`) plus a PNG
preview next to it.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMN = 3.39  # inches
DOUBLE = 7.0
RASTER_DPI = 600

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "savefig.dpi": RASTER_DPI,
    "pdf.fonttype": 42,  # embed TrueType, editable text
    "ps.fonttype": 42,
}


def use():
    plt.rcParams.update(RC)


def width(text):
    """'col', 'double' or a number of inches."""
    return {"col": COLUMN, "double": DOUBLE}.get(text) or float(text)


def save(fig, path):
    """Save path (.pdf unless another extension is given) and a .png preview."""
    root, ext = os.path.splitext(path)
    ext = ext or ".pdf"
    os.makedirs(os.path.dirname(os.path.abspath(root)), exist_ok=True)
    fig.savefig(root + ext, dpi=RASTER_DPI)
    if ext != ".png":
        fig.savefig(root + ".png", dpi=300)
    print("wrote", root + ext)
