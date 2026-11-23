"""Animation of Kraynik-Reinelt periodic boundary conditions in planar extension.

Shows (in fixed physical coordinates, extension along x):
  * lattice points carried by the flow (filled) vs. their initial positions (open);
    after one KR period every moving point lands on an initial one,
  * the periodic cell, which deforms with the flow and is reset at each period,
  * a passive pattern defined in lattice coordinates, tiled over all periodic copies,
    which stays continuous through the reset (the remap only relabels the data).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from viz_common import (
    INK,
    INK2,
    MUTED,
    SERIES,
    SURFACE,
    TiledField,
    cell_outline,
    style_axes,
)

from kr_solver import EPS_P, R

N = 256
FRAMES_PER_PERIOD = 120
HOLD = 45
PERIODS = 2
WINDOW = (-1.6, 1.6, -1.6, 1.6)


def make_pattern():
    xi = np.arange(N) / N
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    img = np.ones((N, N, 3)) * np.array([0.96, 0.96, 0.95])
    for (c1, c2), colour in zip([(0.0, 0.0), (0.3, 0.15), (0.62, 0.55)], SERIES[:3]):
        d1 = (X1 - c1 + 0.5) % 1 - 0.5  # minimum-image separation in xi
        d2 = (X2 - c2 + 0.5) % 1 - 0.5
        r = np.hypot(R[0, 0] * d1 + R[0, 1] * d2, R[1, 0] * d1 + R[1, 1] * d2)
        rgb = np.array([int(colour[k : k + 2], 16) / 255 for k in (1, 3, 5)])
        img[r < 0.11] = rgb
    return img


def remap(img):
    """xi_new = M xi_old (mod 1): exact permutation of lattice nodes."""
    i1, i2 = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    new = np.empty_like(img)
    new[(2 * i1 + i2) % N, (i1 + i2) % N] = img[i1, i2]
    return new


def lattice(strain):
    return np.diag([np.exp(strain), np.exp(-strain)]) @ R


def main(out="kr_boundary_conditions.mp4"):
    fig, ax = plt.subplots(figsize=(6.4, 7.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.12)
    style_axes(ax)
    tiles = TiledField(ax, WINDOW, zorder=1)

    ij = np.array([(i, j) for i in range(-10, 11) for j in range(-10, 11)]).T
    p0 = R @ ij
    ax.plot(p0[0], p0[1], "o", ms=7, mfc="none", mec=MUTED, mew=1.0, zorder=3)
    (dots,) = ax.plot([], [], "o", ms=3.5, color=INK, zorder=4)
    (cell,) = ax.plot([], [], "-", color=INK, lw=1.8, zorder=5)
    (old_cell,) = ax.plot([], [], "--", color=INK2, lw=1.4, zorder=5)

    fig.text(
        0.03,
        0.955,
        "Kraynik–Reinelt periodic boundary conditions",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.03,
        0.915,
        "Planar extension, extension axis horizontal. Open circles: lattice at "
        "ε = 0.\nDots: lattice carried by the flow. Solid outline: current periodic cell.",
        fontsize=8.5,
        color=INK2,
        va="top",
    )
    clock = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        color=INK,
        zorder=6,
        bbox=dict(fc=SURFACE, ec="none", alpha=0.85),
    )
    note = fig.text(0.03, 0.095, "", fontsize=9, color=INK, va="top")

    img = make_pattern()
    frames = []  # (strain within period, period index, is_hold)
    for p in range(PERIODS):
        frames += [
            (EPS_P * k / FRAMES_PER_PERIOD, p, False)
            for k in range(FRAMES_PER_PERIOD + 1)
        ]
        frames += [(EPS_P, p, True)] * HOLD
    frames += [(0.0, PERIODS, False)] * 20

    writer = FFMpegWriter(
        fps=30,
        codec="libx264",
        extra_args=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
    )
    prev_p = 0
    with writer.saving(fig, out, dpi=150):
        for s, p, hold in frames:
            if p != prev_p:  # entering a new period: reset the cell, relabel the data
                img = remap(img)
                prev_p = p
            A = lattice(s)
            tiles.draw(img, A)
            pts = A @ ij
            dots.set_data(pts[0], pts[1])
            cell.set_data(*cell_outline(R if hold else A))
            if hold:
                old_cell.set_data(*cell_outline(A))
                note.set_text(
                    "After Hencky strain ε_p = 2 ln φ ≈ 0.962 every moving lattice point "
                    "lands on an initial one.\nThe sheared cell (dashed) and the square cell "
                    "(solid) tile the same lattice: a₁′ = 2a₁ + a₂,\na₂′ = a₁ + a₂. "
                    "The cell is reset and the fields are relabelled exactly, with no "
                    "interpolation."
                )
            else:
                old_cell.set_data([], [])
                note.set_text("")
            clock.set_text(f"Hencky strain ε = {p * EPS_P + s:4.2f}")
            writer.grab_frame()
    print("wrote", out)


if __name__ == "__main__":
    main()
