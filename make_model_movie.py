"""Startup of planar extension for four constitutive models, each seeded with the
same localized blob of extra conformation at the stagnation point.

Outputs
  model_fields.npz           raw frames (delta tau_xx in lattice coordinates, lattice matrix,
                             strain, cell-averaged C) for re-plotting
  model_comparison.mp4       2x2 tiled fields in physical coordinates + mean N1 vs strain
  startup_N1.png             cell-averaged N1 vs strain for the four models
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from kr_solver import EPS_P, XX, YY, KRSolver, Params, R
from viz_common import (
    DIVERGING,
    GRID,
    INK,
    INK2,
    MUTED,
    SERIES,
    SURFACE,
    TiledField,
    cell_outline,
    style_axes,
)

EPS_DOT = 0.4  # Wi = eps_dot * lam = 0.4 (below the Oldroyd-B coil-stretch limit 0.5)
COMMON = dict(eps_dot=EPS_DOT, lam=1.0, G=1.0, eta_s=0.1, D=1e-4, N=128)
MODELS = [
    ("Oldroyd-B", Params("oldroyd_b", **COMMON), ""),
    ("Giesekus", Params("giesekus", alpha=0.3, **COMMON), "α = 0.3"),
    ("Johnson–Segalman", Params("johnson_segalman", a=0.8, **COMMON), "a = 0.8"),
    (
        "Rolie-Poly",
        Params("rolie_poly", tau_R=0.1, beta=0.0, **COMMON),
        "τ_R/τ_d = 0.1, β = 0",
    ),
]
PERIODS = 4
STEPS_PER_PERIOD = 400
SAVE_EVERY = 8
BLOB_AMP, BLOB_WIDTH = 1.0, 0.08  # for Gaussian blob IC
NOISE_AMP, NOISE_SEED = 0.1, 0
WINDOW = (-0.75, 0.75, -0.75, 0.75)


def seed_blob(s):
    xi = np.arange(s.p.N) / s.p.N
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    d1, d2 = (
        (X1 + 0.5) % 1 - 0.5,
        (X2 + 0.5) % 1 - 0.5,
    )  # min-image distance to the node xi = 0
    r2 = (R[0, 0] * d1 + R[0, 1] * d2) ** 2 + (R[1, 0] * d1 + R[1, 1] * d2) ** 2
    b = BLOB_AMP * np.exp(-r2 / (2 * BLOB_WIDTH**2))
    C = s.C()
    C[XX] += b
    C[YY] += b
    s.set_C(C)


def simulate(path="model_fields.npz"):
    if os.path.exists(path):
        return dict(np.load(path))
    out = {}
    for key, (name, p, _) in enumerate(MODELS):
        s = KRSolver(p, STEPS_PER_PERIOD)
        # seed_blob(s)
        s.perturb(NOISE_AMP, seed=NOISE_SEED, smooth_length=(p.D / p.eps_dot) ** 0.5)
        fields, lattices, strains, means = [], [], [], []
        for k in range(PERIODS * STEPS_PER_PERIOD + 1):
            if k % SAVE_EVERY == 0:
                C = s.C()
                fields.append((p.G * (C[XX] - C[XX].mean())).astype(np.float32))
                lattices.append(s.lattice())
                strains.append(s.t * p.eps_dot)
                means.append(s.mean_C())
            if k < PERIODS * STEPS_PER_PERIOD:
                s.step()
        out[f"field{key}"] = np.array(fields)
        out[f"lattice{key}"] = np.array(lattices)
        out[f"strain{key}"] = np.array(strains)
        out[f"mean{key}"] = np.array(means)
        out[f"loss{key}"] = np.array(s.remap_loss)
        print(f"{name}: remap losses {np.array(s.remap_loss)}")
    np.savez_compressed(path, **out)
    return out


def startup_figure(data, path="startup_N1.png"):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    draw_startup(ax, data)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def draw_startup(ax, data):
    ax.set_facecolor(SURFACE)
    for key, (name, p, _) in enumerate(MODELS):
        m, e = data[f"mean{key}"], data[f"strain{key}"]
        N1 = p.G * (m[:, XX] - m[:, YY])
        ax.plot(e, N1, color=SERIES[key], lw=2, label=name)
        ax.annotate(
            name,
            (e[-1], N1[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK2,
        )
    ax.set_xlabel("Hencky strain ε = ε̇t", color=INK2, fontsize=9)
    ax.set_ylabel("cell-averaged N₁ / G", color=INK2, fontsize=9)
    ax.set_xlim(0, data["strain0"][-1] * 1.32)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)


def movie(data, path="model_comparison.mp4"):
    fig = plt.figure(figsize=(8.0, 10.4), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1, 0.62],
        left=0.09,
        right=0.88,
        top=0.855,
        bottom=0.06,
        hspace=0.18,
        wspace=0.06,
    )
    panels = []
    for key, (name, p, extra) in enumerate(MODELS):
        ax = fig.add_subplot(gs[key // 2, key % 2])
        style_axes(ax)
        ax.set_title(
            name + (f"   ({extra})" if extra else ""),
            fontsize=9.5,
            color=INK,
            loc="left",
        )
        tiles = TiledField(ax, WINDOW, cmap=DIVERGING, vmin=-1, vmax=1, zorder=1)
        (cell,) = ax.plot([], [], "-", color=INK, lw=1.0, zorder=3)
        amp = ax.text(
            0.02,
            0.03,
            "",
            transform=ax.transAxes,
            fontsize=8,
            color=INK,
            zorder=4,
            bbox=dict(fc=SURFACE, ec="none", alpha=0.85),
        )
        panels.append((tiles, cell, amp))
    cax = fig.add_axes([0.9, 0.42, 0.018, 0.4])
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(
        "δτ_xx / max|δτ_xx|   (δτ_xx = τ_xx − ⟨τ_xx⟩)", color=INK2, fontsize=8.5
    )
    cb.ax.tick_params(colors=MUTED, labelsize=7.5)
    cb.outline.set_edgecolor(GRID)

    cax2 = fig.add_subplot(gs[2, :])
    draw_startup(cax2, data)
    marker = cax2.axvline(0, color=MUTED, lw=1, ls="--")

    fig.text(
        0.04,
        0.972,
        "Planar extension with Kraynik–Reinelt boundary conditions",
        fontsize=12.5,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.04,
        0.948,
        f"Startup at Wi = ε̇λ = {EPS_DOT} (λ = τ_d for Rolie-Poly), η_s/Gλ = "
        f"{COMMON['eta_s']}, Re = 0.",
        fontsize=8,
        color=INK2,
        va="top",
    )
    clock = fig.text(0.04, 0.902, "", fontsize=10, color=INK, va="top")

    writer = FFMpegWriter(
        fps=24,
        codec="libx264",
        extra_args=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
    )
    n_frames = len(data["strain0"])
    with writer.saving(fig, path, dpi=120):
        for f in list(range(n_frames)) + [n_frames - 1] * 24:
            for key, (tiles, cell, amp) in enumerate(panels):
                A = data[f"lattice{key}"][f]
                field = data[f"field{key}"][f]
                peak = np.abs(field).max()
                tiles.draw(field / peak, A)
                cell.set_data(*cell_outline(A))
                amp.set_text(f"max|δτ_xx| = {peak:.2g} G")
            e = data["strain0"][f]
            marker.set_xdata([e, e])
            clock.set_text(
                f"Hencky strain ε = {e:4.2f}    (cell reset every ε_p ≈ {EPS_P:.3f})"
            )
            writer.grab_frame()
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    data = simulate()
    startup_figure(data)
    movie(data)
