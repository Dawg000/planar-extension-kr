"""Shared plotting helpers: draw a lattice-coordinate field tiled in physical space."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Affine2D

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# Categorical slots 1-8, in fixed order (validated palette; pair with letter labels)
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]

# Diverging blue <-> red with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8e1f1f"]
)


def tile_offsets(A, window, K=12):
    """Physical offsets i*a1 + j*a2 of every cell copy overlapping the window."""
    corners = A @ np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    x0, x1, y0, y1 = window
    out = []
    for i in range(-K, K + 1):
        for j in range(-K, K + 1):
            c = corners + (A @ [i, j])[:, None]
            if (
                c[0].max() > x0
                and c[0].min() < x1
                and c[1].max() > y0
                and c[1].min() < y1
            ):
                out.append(A @ [i, j])
    return out


class TiledField:
    """Draws field[i1, i2] (sampled at lattice nodes xi = (i1, i2)/N) through the
    lattice map x = A xi, repeated over every periodic copy in the window."""

    def __init__(self, ax, window, **imshow_kw):
        self.ax, self.window, self.kw, self.images = ax, window, imshow_kw, []

    def draw(self, field, A):
        for im in self.images:
            im.remove()
        self.images = []
        N = field.shape[0]
        h = 0.5 / N  # pixel centres sit on the lattice nodes
        img = np.swapaxes(field, 0, 1)  # imshow rows = xi2, cols = xi1
        for o in tile_offsets(A, self.window):
            im = self.ax.imshow(
                img,
                origin="lower",
                extent=(-h, 1 - h, -h, 1 - h),
                interpolation="nearest",
                **self.kw,
            )
            T = np.array(
                [[A[0, 0], A[0, 1], o[0]], [A[1, 0], A[1, 1], o[1]], [0, 0, 1]]
            )
            im.set_transform(Affine2D(T) + self.ax.transData)
            self.images.append(im)
        self.ax.set_xlim(self.window[:2])
        self.ax.set_ylim(self.window[2:])


def cell_outline(A):
    """Closed polygon of the lattice cell centred on the origin (the stagnation point)."""
    xi = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]]) - 0.5
    return A @ xi


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def axis_values(values):
    """Positions for an x axis: the values as floats, or, for a categorical parameter such as
    kr_lattice, indices into the sorted labels. Returns (x, labels or None)."""
    try:
        return np.array([float(v) for v in values]), None
    except (TypeError, ValueError):
        labels = sorted({str(v) for v in values})
        return np.array([labels.index(str(v)) for v in values], float), labels


def set_axis_labels(ax, labels):
    """Put categorical tick labels on ax (no-op for a numeric axis)."""
    if labels is not None:
        ax.set_xticks(range(len(labels)), labels)
