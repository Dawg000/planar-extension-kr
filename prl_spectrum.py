"""PRL figure: averaged 2D power spectrum in (k_x, k_y), with the (|k|, theta) map as an inset.

  python prl_spectrum.py SWEEP_DIR [SWEEP_DIR ...] --case 7 [--field stretch]
      [--window 0.5:1] [--kmax K] [--n-xy N] [--n-k 48] [--n-theta 36]
      [--decades 4] [--inset-loc "upper right"] [--width col] [--out path.pdf]

Averaging is that of fourier_kr.py compare: every saved frame in the strain window (fractions
of each run's final strain) is normalized to unit variance, its mode powers |F_n|^2 are put
into fixed bins of the physical wavevector k = 2 pi A(t)^-T n, and the binned frames are
averaged. Frames are pooled over all the folders given, which should hold the same case of
the same sweep (e.g. one per seed).

Main panel: variance per unit k-area on a Cartesian (k_x, k_y) grid over |k_x|, |k_y| <= kmax
(default: the smallest square holding every bin within the colour range, i.e. above
10^-decades of the maximum, plus 5%). The field is real, so the map is symmetric under
k -> -k. The k = 0 bin carries no variance (the mean is removed) and is drawn at the bottom
of the scale; any other bin that no mode reached in any frame is too (the count is printed).
The default bin width is the lattice mode spacing 2 pi / L; narrower bins (--n-xy) alternate
between catching a mode and not, and show as a checkerboard.

Inset: variance per unit ln|k| per radian in (|k|, theta) bins, theta = angle of k from the
extension (x) axis folded into [0, 180), exactly as in fourier_kr.py (log-spaced |k| bins no
narrower than 2 pi / L, sparse small-|k| angle bins merged), shown for |k| <= kmax.

Both maps are divided by their own maximum and share one log colour scale spanning --decades
decades, so the one colour bar holds for both (the two densities have different units).
Writes PDF + PNG, by default FIRST_DIR/figures/prl_spectrum_FIELD_case_NNNN.pdf.
"""

import argparse
import math
import os

import numpy as np

import prl_style
from fourier_kr import (
    SEQUENTIAL,
    bin_density,
    bin_polar,
    frame_power,
    k_edges_for,
    load_frames,
    merge_sparse_angles,
    parse_window,
)
from structure_kr import FIELDS, field_frames
from viz_sweep import case_params, load_sweep

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

MINUS = lambda v, _: f"{v:g}".replace("-", "−")


def pooled_frames(folders, case, field, window):
    """(fields, lattices, params) of the frames in the window, pooled over folders."""
    fields, lattices, params = [], [], []
    for folder in folders:
        config, rows, _ = load_sweep(folder)
        p = case_params(config, rows[case])
        fr = load_frames(folder, case)
        e = fr["strain"]
        sel = (e >= window[0] * e[-1] - 1e-9) & (e <= window[1] * e[-1] + 1e-9)
        if not sel.any():
            raise SystemExit(f"{folder} case {case}: no frames in the strain window")
        fields.append(field_frames(fr, p, field)[sel])
        lattices.append(fr["lattice"][sel])
        params.append(p)
    if any(q != params[0] for q in params):
        raise SystemExit(f"case {case} has different parameters in these folders")
    return np.concatenate(fields), np.concatenate(lattices), params[0]


def averaged_spectra(fields, lattices, kmax, n_xy, n_k, n_theta):
    """Cartesian density (per unit k-area) and polar density (per ln|k| per rad), each
    the average over frames of unit-variance frame spectra."""
    N = fields.shape[-1]
    xy_edges = np.linspace(-kmax, kmax, n_xy + 1)
    k_edges = k_edges_for(lattices, N, n_k)
    cart = np.zeros((n_xy, n_xy))
    hits = np.zeros((n_xy, n_xy))
    polar = np.zeros((len(k_edges) - 1, n_theta))
    n = 0
    for f, A in zip(fields, lattices):
        kx, ky, P = frame_power(f, A)
        tot = P.sum()
        if tot <= 0:
            continue
        P = P / tot
        cart += np.histogram2d(kx, ky, [xy_edges, xy_edges], weights=P)[0]
        hits += np.histogram2d(kx, ky, [xy_edges, xy_edges])[0]
        polar += bin_polar(kx, ky, P, k_edges, n_theta)
        n += 1
    L = math.sqrt(abs(np.linalg.det(lattices[0])))
    cart = np.where(hits > 0, cart / n / np.diff(xy_edges)[0] ** 2, np.nan)
    polar = bin_density(merge_sparse_angles(polar / n, k_edges, n_theta, L), k_edges, n_theta)
    return xy_edges, cart, k_edges, polar, n, int((hits == 0).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--field", choices=FIELDS, default="stretch")
    ap.add_argument("--window", default="0.5:1")
    ap.add_argument("--kmax", type=float, help="half-width of the main panel in k")
    ap.add_argument(
        "--n-xy", type=int, help="Cartesian bins per axis (default: width 2 pi / L)"
    )
    ap.add_argument("--n-k", type=int, default=48)
    ap.add_argument("--n-theta", type=int, default=36)
    ap.add_argument("--decades", type=float, default=4, help="colour range in decades")
    ap.add_argument("--inset-loc", default="upper right")
    ap.add_argument("--width", default="col", help="col, double or inches")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    prl_style.use()

    fields, lattices, p = pooled_frames(
        args.folders, args.case, args.field, parse_window(args.window)
    )
    L = math.sqrt(abs(np.linalg.det(lattices[0])))
    n_bins = lambda km: args.n_xy or 2 * int(km / (2 * np.pi / L)) + 1  # odd: k = 0 central
    kmax = args.kmax
    if kmax is None:  # first pass over every resolved mode, then crop to the colour range
        kmax = max(
            np.abs(frame_power(f, A)[i]).max() for f, A in zip(fields[::10], lattices[::10])
            for i in (0, 1)
        )
        e, c = averaged_spectra(fields, lattices, kmax, n_bins(kmax), 4, 4)[:2]
        kc = 0.5 * (e[1:] + e[:-1])
        I, J = np.nonzero(c >= 10 ** -args.decades * np.nanmax(c))
        kmax = 1.05 * max(np.abs(kc[I]).max(), np.abs(kc[J]).max())
    xy_edges, cart, k_edges, polar, n, empty = averaged_spectra(
        fields, lattices, kmax, n_bins(kmax), args.n_k, args.n_theta
    )
    print(
        f"case {args.case}, {args.field}: {n} frames from {len(args.folders)} folder(s); "
        f"kmax {kmax:.4g}; {empty} of {n_bins(kmax) ** 2} "
        "Cartesian bins never reached (blank)"
    )

    norm = LogNorm(10 ** -args.decades, 1)
    W = prl_style.width(args.width)
    cb_w, gap, left, right, bottom, top = 0.08, 0.06, 0.45, 0.62, 0.36, 0.08
    P = W - left - gap - cb_w - right
    H = bottom + P + top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, P / W, P / H])
    cax = fig.add_axes([(left + P + gap) / W, bottom / H, cb_w / W, P / H])

    m = ax.pcolormesh(
        xy_edges, xy_edges, (cart / np.nanmax(cart)).T, cmap=SEQUENTIAL, norm=norm,
        shading="flat", rasterized=True,
    )
    ax.set_facecolor(SEQUENTIAL(0.0))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$", labelpad=1)
    ax.set_ylabel(r"$k_y$", labelpad=1)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(FuncFormatter(MINUS))

    # inset: (|k|, theta)
    iw, ih, pad = 0.32, 0.34, 0.03
    x0 = pad + 0.08 if "left" in args.inset_loc else 1 - pad - iw
    y0 = pad + 0.1 if "lower" in args.inset_loc else 1 - pad - ih
    ins = ax.inset_axes([x0, y0, iw, ih])
    th_edges = np.linspace(0, 180, args.n_theta + 1)
    ins.pcolormesh(
        k_edges, th_edges, np.where(polar > 0, polar / polar.max(), np.nan).T,
        cmap=SEQUENTIAL, norm=norm, shading="flat", rasterized=True,
    )
    ins.set_xscale("log")
    ins.set_xlim(k_edges[0], kmax)
    ins.set_ylim(0, 180)
    ins.yaxis.set_major_locator(MultipleLocator(90))
    ins.yaxis.set_minor_locator(MultipleLocator(45))
    ins.xaxis.set_major_locator(LogLocator(base=10))
    ins.xaxis.set_minor_formatter(NullFormatter())
    ins.tick_params(labelsize=6, pad=1.5, length=2, which="major")
    ins.tick_params(length=1.2, which="minor")
    ins.set_xlabel(r"$|\mathbf{k}|$", fontsize=6.5, labelpad=0)
    ins.set_ylabel(r"$\theta$ (deg)", fontsize=6.5, labelpad=0)
    ins.patch.set_facecolor("white")

    cb = fig.colorbar(m, cax=cax)
    cb.set_label("spectral density / maximum", labelpad=2)
    cb.ax.tick_params(direction="in", length=2, width=0.4, which="both")
    cb.outline.set_linewidth(0.6)

    out = args.out or os.path.join(
        args.folders[0], "figures", f"prl_spectrum_{args.field}_case_{args.case:04d}.pdf"
    )
    prl_style.save(fig, out)


if __name__ == "__main__":
    main()
