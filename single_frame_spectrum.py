"""One frame of delta tau_xx: field, 2D power spectrum over physical k, and the 1D spectrum
across the structures (this frame and the average over the window of every folder given).

  python single_frame_spectrum.py kr_match_s0 kr_match_s1 kr_match_s2 --case 7 [--frame -1]
      [--window 0.5:1 | --skip-strain 10] [--out single_frame_spectrum.png]

The first folder supplies the frame shown (--frame indexes the frames in its window); the
cutoff length is pooled over all the folders' windows (structure_kr.gradient_lengths), and
the folders should hold the same case of the same sweep, e.g. one per seed."""

import argparse
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy import ndimage

from fourier_kr import (
    SEQUENTIAL,
    across_direction,
    across_spectrum,
    frame_power,
    load_frames,
    mode_wavevectors,
    parse_window,
    window_mask,
)
from structure_kr import gradient_lengths
from viz_common import DIVERGING, INK, INK2, MUTED, SERIES, SURFACE
from viz_sweep import case_params, load_sweep, style_plot

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("folders", nargs="+")
ap.add_argument("--case", type=int, required=True)
ap.add_argument("--frame", type=int, default=-1)
ap.add_argument(
    "--window", default="0.5:1", help="strain window as fractions of the final strain"
)
ap.add_argument(
    "--skip-strain",
    type=float,
    help="discard frames before this Hencky strain (replaces the window start)",
)
ap.add_argument("--out", default="single_frame_spectrum.png")
args = ap.parse_args()
config, rows, _ = load_sweep(args.folders[0])
p = case_params(config, rows[args.case])
D, RATE = p.D, p.eps_dot
runs = []
for folder in args.folders:
    fr = load_frames(folder, args.case)
    e = fr["strain"]
    sel, _, _ = window_mask(e, parse_window(args.window), args.skip_strain)
    f = fr["dC"][:, 0].astype(float)
    f -= f.mean(axis=(1, 2), keepdims=True)
    runs.append((f[sel], fr["lattice"][sel], e[sel]))
cut = gradient_lengths(
    np.concatenate([r[0] for r in runs]), np.concatenate([r[1] for r in runs])
)["cutoff_across"]
f0, A0, e0 = (runs[0][k][args.frame] for k in range(3))
N = f0.shape[-1]

fig, axes = plt.subplots(
    1, 3, figsize=(13.5, 4.5), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.08, 1.25])
)
fig.patch.set_facecolor(SURFACE)
# field
ax = axes[0]
xs = np.linspace(-0.5, 0.5, 500)
X, Y = np.meshgrid(xs, xs, indexing="ij")
xi = np.linalg.solve(A0, np.stack([X.ravel(), Y.ravel()])) * N
F = ndimage.map_coordinates(f0, xi, order=3, mode="grid-wrap").reshape(X.shape)
lim = np.percentile(np.abs(F), 99.5)
ax.pcolormesh(
    xs,
    xs,
    F.T,
    cmap=DIVERGING,
    norm=TwoSlopeNorm(0, -lim, lim),
    shading="auto",
    rasterized=True,
)
ax.set_aspect("equal")
ax.set_xlabel("x (extension)", color=INK2, fontsize=8.5)
ax.set_ylabel("y", color=INK2, fontsize=8.5)
ax.set_title(
    f"δτ_xx at strain {e0:.2f} (one cell side)", color=INK, fontsize=9.5, loc="left"
)
ax.tick_params(colors=MUTED, labelsize=7.5)
# 2D spectrum: every mode as a point
ax = axes[1]
kx, ky, P = frame_power(f0, A0)
P = P / P.sum()
kmax = 3.2 / cut
o = np.argsort(P)
show = (np.abs(kx[o]) < kmax) & (np.abs(ky[o]) < kmax) & (P[o] > 0)
sc = ax.scatter(
    kx[o][show],
    ky[o][show],
    c=P[o][show],
    s=3,
    cmap=SEQUENTIAL,
    norm=LogNorm(P.max() * 1e-6, P.max()),
    lw=0,
    rasterized=True,
)
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(t) / cut, np.sin(t) / cut, "--", color=INK, lw=1)
ax.text(0.71 / cut, 0.71 / cut, "  |k| = 1/cutoff", color=INK, fontsize=8)
ax.set_xlim(-kmax, kmax)
ax.set_ylim(-kmax, kmax)
ax.set_aspect("equal")
ax.set_xlabel("k_x", color=INK2, fontsize=8.5)
ax.set_ylabel("k_y", color=INK2, fontsize=8.5)
ax.set_title(
    "power of each Fourier mode (log colour)", color=INK, fontsize=9.5, loc="left"
)
ax.tick_params(colors=MUTED, labelsize=7.5)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
cb.ax.tick_params(colors=MUTED, labelsize=7)
cb.set_label("fraction of variance", color=INK2, fontsize=8)
# 1D across spectrum
ax = axes[2]
style_plot(ax)
e_dir = across_direction(
    np.concatenate([r[0] for r in runs]), np.concatenate([r[1] for r in runs])
)
qmax = max(
    np.abs(e_dir[0] * a + e_dir[1] * b).max()
    for _, lats, _ in runs
    for a, b, _ in [mode_wavevectors(A, N) for A in lats[::10]]
)
edges = np.geomspace(np.pi, qmax, 41)
kc = np.sqrt(edges[1:] * edges[:-1])
avg = np.concatenate(
    [across_spectrum(fl, la, e_dir, edges, blocks=5) for fl, la, _ in runs]
).mean(0)
one = across_spectrum(f0[None], A0[None], e_dir, edges, blocks=1)[0]
ok = (avg > 0) & (one > 0)
ax.loglog(
    kc[ok],
    avg[ok],
    color=MUTED,
    lw=2.4,
    label=f"mean of {sum(len(r[0]) for r in runs)} frames",
)
ax.loglog(kc[ok], one[ok], "-o", ms=3, color=SERIES[0], lw=1.5, label="this frame")
ax.axvline(1 / cut, color=INK, ls="--", lw=1)
ax.text(1 / cut, avg[ok].max() * 3, " 1/cutoff", fontsize=8, color=INK)
ax.axvline(2 * np.pi, color=INK, ls=":", lw=1)
ax.text(2 * np.pi, avg[ok].max() * 3, " box 2π/L", fontsize=8, color=INK)
ax.set_ylim(avg[ok].max() * 1e-9, avg[ok].max() * 10)
ax.set_xlabel(
    f"|e·k|, e = across direction ({math.degrees(math.atan2(e_dir[1], e_dir[0])) % 180:.0f}° from x)",
    color=INK2,
    fontsize=8.5,
)
ax.set_ylabel("variance per ln|e·k| (unit total)", color=INK2, fontsize=8.5)
ax.set_title("spectrum across the structures", color=INK, fontsize=9.5, loc="left")
ax.legend(
    frameon=False,
    fontsize=7.5,
    labelcolor=INK2,
    loc="lower left",
    bbox_to_anchor=(0.16, 0.02),
)
fig.text(
    0.01,
    0.01,
    f"D = {D:g}, {args.folders[0]} case {args.case}, ε̇ = {RATE:g}, {p.kr_lattice} cell L = {p.Lbox:g}, "
    f"N = {p.N}. Cutoff length {cut:.4f} (pooled over {len(runs)} folder(s)) = "
    f"{cut / math.sqrt(D / RATE):.2f} √(D/ε̇).",
    fontsize=7,
    color=MUTED,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(args.out, facecolor=SURFACE)
print("ok", cut)
