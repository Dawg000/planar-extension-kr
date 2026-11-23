"""PRL figure: one frame of the chain-stretch fluctuation with streamlines overlaid.

  python prl_flow_frame.py SWEEP_DIR --case 7 --strain 6.5
      [--streamlines deviation|full|none] [--levels 6] [--window 0.75] [--vmax V]
      [--width col] [--out path.pdf]

Heatmap: delta lambda = lambda - <lambda>, lambda = sqrt(tr C / 3), in physical coordinates,
tiled over the periodic copies, on a symmetric colour scale (+-max|delta lambda| unless
--vmax is given). Lines: streamlines from flowtype_kr (deviation: contours of the
perturbation streamfunction psi' at fixed fractions of max|psi'|, solid psi' > 0
counter-clockwise, dashed clockwise; full: contours of eps_dot x y + psi'). The thin outline
is the current KR cell. Axes are in units of the cell side L = sqrt(det A). The frame shown
is the saved snapshot nearest --strain (printed). Writes PDF + PNG, by default
SWEEP_DIR/figures/prl_flow_case_NNNN_strain_S.pdf.
"""

import argparse
import math
import os

import numpy as np

import prl_style
from fourier_kr import load_frames, stretch_field
from flowtype_kr import frame_fields, streamline_field
from viz_common import DIVERGING, INK, TiledField, cell_outline
from viz_sweep import case_params, load_sweep

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folder")
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--strain", type=float, required=True)
    ap.add_argument(
        "--streamlines", choices=("deviation", "full", "none"), default="deviation"
    )
    ap.add_argument("--levels", type=int, default=6, help="streamline levels per sign")
    ap.add_argument(
        "--window", type=float, default=0.75, help="half-width shown, in cell sides"
    )
    ap.add_argument("--vmax", type=float, help="colour limit for delta lambda")
    ap.add_argument("--width", default="col", help="col, double or inches")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    prl_style.use()

    config, rows, _ = load_sweep(args.folder)
    p = case_params(config, rows[args.case])
    fr = load_frames(args.folder, args.case)
    lam = stretch_field(fr, p)
    f = int(np.argmin(np.abs(fr["strain"] - args.strain)))
    A = fr["lattice"][f]
    L = math.sqrt(abs(np.linalg.det(fr["lattice"][0])))
    need = {"stretch"} | ({"psi"} if args.streamlines != "none" else set())
    F = frame_fields(fr, lam, f, p, need)
    dlam = F["stretch"]
    vmax = args.vmax or float(np.abs(dlam).max())
    print(
        f"case {args.case}: frame {f} at strain {fr['strain'][f]:.3f}, "
        f"<lambda> = {F['mean']:.4g}, max|delta lambda| = {np.abs(dlam).max():.3g}"
    )

    w = args.window * L
    W = prl_style.width(args.width)
    cb_w, gap, left, right, bottom, top = 0.08, 0.06, 0.42, 0.42, 0.36, 0.08
    P = W - left - gap - cb_w - right  # square panel
    H = bottom + P + top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, P / W, P / H])
    cax = fig.add_axes([(left + P + gap) / W, bottom / H, cb_w / W, P / H])

    norm = Normalize(-vmax, vmax)
    tiles = TiledField(
        ax, (-w, w, -w, w), cmap=DIVERGING, norm=norm, zorder=1, rasterized=True
    )
    tiles.draw(dlam, A)
    ax.plot(*cell_outline(A), "-", color=INK, lw=0.5, zorder=3)
    if args.streamlines != "none":
        g = np.linspace(-w, w, 401)
        X, Y = np.meshgrid(g, g, indexing="xy")
        Z, lv, lw, neg = streamline_field(
            F, A, X, Y, p.eps_dot, L, args.streamlines, args.levels
        )
        lw = np.asarray(lw) * 0.7 if np.ndim(lw) else lw * 0.7
        ax.contour(
            X, Y, Z, levels=lv, colors=INK, linewidths=lw,
            negative_linestyles=neg, zorder=2,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-w, w)
    ax.set_ylim(-w, w)
    unit = FuncFormatter(lambda v, _: f"{v / L:g}".replace("-", "\u2212"))
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(MultipleLocator(0.5 * L))
        a.set_major_formatter(unit)
    ax.set_xlabel(r"$x/L$", labelpad=1)
    ax.set_ylabel(r"$y/L$", labelpad=1)
    ax.text(
        0.03, 0.96, rf"$\varepsilon = {fr['strain'][f]:.2f}$", transform=ax.transAxes,
        va="top", ha="left", fontsize=7,
        bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0), zorder=4,
    )
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax)
    cb.set_label(r"$\delta\lambda$", labelpad=2)
    cb.ax.tick_params(direction="in", length=2, width=0.4)
    cb.outline.set_linewidth(0.6)
    cb.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212"))
    )

    out = args.out or os.path.join(
        args.folder, "figures",
        f"prl_flow_case_{args.case:04d}_strain_{fr['strain'][f]:.2f}.pdf",
    )
    prl_style.save(fig, out)


if __name__ == "__main__":
    main()
