"""Chain stretch and flow-type parameter fields for sweep_kr.py runs (needs --save-fields).

  python flowtype_kr.py movie SWEEP_DIR --cases 3 7 12 [--xi-range -1:1] [--fps 12]
  python flowtype_kr.py still SWEEP_DIR --cases 3 7 12 --strain 6.5 [--xi-range -1:1]

Each case (up to 8) gets a pair of panels in physical coordinates, tiled over the periodic
copies: the chain-stretch fluctuation lambda - <lambda>, lambda = sqrt(tr C / 3), scaled by
its current peak (printed with <lambda>), and the flow-type parameter

    xi = (|D| - |Omega|) / (|D| + |Omega|),   +1 pure extension, 0 simple shear, -1 rotation,

with D and Omega the symmetric and antisymmetric parts of the full velocity gradient
kappa = L + grad v. The base flow is pure extension, so xi = 1 wherever the perturbation
velocity is negligible. At Re = 0 the perturbation velocity follows exactly from the saved
stress fluctuation (the same Stokes solve the simulation uses), so xi is exact for every
saved frame. xi is measured in the lab frame (it is not frame-invariant), which is the
natural frame here: the extension axes are fixed.

--xi-range lo:hi sets the colour scale (default -1:1, the full physical range, which keeps
colours comparable between figures). Small perturbations keep xi close to 1; zoom in with
e.g. --xi-range 0.9:1, or use --xi-range auto, which runs from just below the smallest xi in
the frames shown up to 1. A range that spans 0 uses a diverging map centred on simple shear;
a range on one side of 0 uses a sequential map. Each panel prints the current minimum of xi.
A range starting with a minus sign needs the = form: --xi-range=-0.5:1.

--panels chooses the panels drawn for each case, in order (default: stretch xi):
    stretch   chain-stretch fluctuation, as above
    xi        flow-type parameter, as above
    speed     |v| / max|v|, the size of the deviation v = u - eps_dot (x, -y) from ideal
              planar extension (max|v| is printed in units of eps_dot * cell size)

--streamlines overlays instantaneous lab-frame streamlines on every panel (default none):
    deviation contours of the perturbation streamfunction psi', v = (d psi'/dy, -d psi'/dx),
              at fixed fractions +-k/n of max|psi'| (n = --levels), so line spacing shows the
              speed. Solid: psi' > 0, counter-clockwise; dashed: psi' < 0, clockwise. Visible
              at any perturbation amplitude.
    full      contours of the full streamfunction eps_dot x y + psi' (n levels per sign,
              plus the separatrix 0). Indistinguishable from the base hyperbolas until
              |v| is a sizeable fraction of eps_dot * cell size.
psi' is periodic on the lattice (v has zero mean and no divergence) and exact at Re = 0,
like xi. Streamlines are not pathlines: the flow is unsteady and the cell deforms.
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.ndimage import map_coordinates

from fourier_kr import SEQUENTIAL, load_frames, stretch_field
from kr_solver import stokes_velocity_hat
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
from viz_sweep import case_params, load_sweep, param_text, upsample, wrap

LETTERS = "ABCDEFGH"


# --- fields -----------------------------------------------------------------------------
def perturbation_velocity_hat(dS, A, G, eta_s):
    """(kx, ky, vx_hat, vy_hat): wavevectors and the spectral perturbation velocity on the
    lattice grid, from one frame's stress fluctuation dS = tau/G - <tau/G> (xx, xy, yy, ...)."""
    N = dS.shape[-1]
    n = np.fft.fftfreq(N, 1.0 / N)
    n1, n2 = np.meshgrid(n, n, indexing="ij")
    B = 2 * np.pi * np.linalg.inv(A).T  # k = 2 pi A^-T n
    kx, ky = B[0, 0] * n1 + B[0, 1] * n2, B[1, 0] * n1 + B[1, 1] * n2
    vx, vy = stokes_velocity_hat(np.fft.fft2(dS[:3]), kx, ky, G, eta_s)
    return kx, ky, vx, vy


def velocity_gradient(dS, A, G, eta_s, eps_dot):
    """Full velocity gradient (kxx, kxy, kyx, kyy), kappa_ij = du_i/dx_j, on the lattice grid,
    from one frame's stress fluctuation dS = tau/G - <tau/G> (xx, xy, yy, ...)."""
    kx, ky, vx, vy = perturbation_velocity_hat(dS, A, G, eta_s)
    d = lambda k, v: np.fft.ifft2(1j * k * v).real
    return (eps_dot + d(kx, vx), d(ky, vx), d(kx, vy), -eps_dot + d(ky, vy))


def streamfunction(kx, ky, vx, vy):
    """Perturbation streamfunction psi' on the lattice grid, v = (d psi'/dy, -d psi'/dx):
    psi'_hat = i (kx vy_hat - ky vx_hat) / |k|^2, with zero mean."""
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    ph = 1j * (kx * vy - ky * vx) / k2
    ph[0, 0] = 0
    return np.fft.ifft2(ph).real


def sample_periodic(field, A, X, Y):
    """Values at physical points (X, Y) of a lattice-periodic field sampled at the nodes
    xi = (i1, i2)/N, x = A xi (cubic interpolation with periodic wrap)."""
    N = field.shape[0]
    Ai = np.linalg.inv(A)
    i1 = (Ai[0, 0] * X + Ai[0, 1] * Y) * N
    i2 = (Ai[1, 0] * X + Ai[1, 1] * Y) * N
    return map_coordinates(field, [i1, i2], order=3, mode="grid-wrap")


def flow_type(kxx, kxy, kyx, kyy):
    """xi = (|D| - |Omega|) / (|D| + |Omega|) for 2D velocity-gradient components."""
    dxy = 0.5 * (kxy + kyx)
    D = np.sqrt(kxx**2 + kyy**2 + 2 * dxy**2)
    W = np.sqrt(2) * np.abs(0.5 * (kxy - kyx))
    with np.errstate(invalid="ignore", divide="ignore"):
        xi = (D - W) / (D + W)
    return np.where(D + W > 0, xi, 0.0)


def frame_fields(frames, lam, f, p, need):
    """Fields for frame f, upsampled spectrally for display: a dict with any of 'stretch'
    (delta lambda), 'xi', 'speed' (|v|) and 'psi' (psi') named in need, plus 'mean'
    (<lambda>). Components are upsampled before xi and |v| are formed, as neither is
    smooth."""
    dS, A = frames["dC"][f].astype(float), frames["lattice"][f]
    kx, ky, vx, vy = perturbation_velocity_hat(dS, A, p.G, p.eta_s)
    out = {"mean": float(lam[f].mean())}
    if "stretch" in need:
        out["stretch"] = upsample(lam[f] - out["mean"])
    if "xi" in need:
        d = lambda k, v: np.fft.ifft2(1j * k * v).real
        kap = (p.eps_dot + d(kx, vx), d(ky, vx), d(kx, vy), -p.eps_dot + d(ky, vy))
        out["xi"] = flow_type(*(upsample(k) for k in kap))
    if "speed" in need:
        ux, uy = (upsample(np.fft.ifft2(v).real) for v in (vx, vy))
        out["speed"] = np.hypot(ux, uy)
    if "psi" in need:
        out["psi"] = upsample(streamfunction(kx, ky, vx, vy))
    return out


# --- figure -----------------------------------------------------------------------------
def parse_range(text):
    if text == "auto":
        return "auto"
    lo, _, hi = text.partition(":")
    lo, hi = float(lo), float(hi)
    if not -1 <= lo < hi <= 1:
        raise SystemExit("--xi-range must satisfy -1 <= lo < hi <= 1")
    return lo, hi


def build(folder, cases, xi_range, panel_names=("stretch", "xi"), streamlines="none"):
    """Figure, panel handles and per-case data for a set of cases."""
    config, rows, _ = load_sweep(folder)
    names = list(config["sweeps"])
    if not 1 <= len(cases) <= len(LETTERS):
        raise SystemExit(f"choose 1 to {len(LETTERS)} cases")
    data = {}
    for c in cases:
        p = case_params(config, rows[c])
        fr = load_frames(folder, c)
        data[c] = dict(frames=fr, p=p, lam=stretch_field(fr, p))
    vary = [k for k in names if len({repr(rows[c][k]) for c in cases}) > 1]
    common = [k for k in names if k not in vary]

    lo, hi = xi_range
    if lo < 0 < hi:
        xi_cmap, xi_norm = DIVERGING, TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi)
    else:
        xi_cmap, xi_norm = SEQUENTIAL, Normalize(lo, hi)
    # name: panel label, colour map, norm, colour-bar label, colour-bar ticks
    specs = {
        "stretch": (
            "chain stretch  δλ / max|δλ|",
            DIVERGING,
            Normalize(-1, 1),
            "δλ / max|δλ|",
            [-1, 0, 1],
        ),
        "xi": (
            "flow type  ξ",
            xi_cmap,
            xi_norm,
            "ξ   (−1 rotation · 0 shear · +1 extension)",
            None,
        ),
        "speed": (
            "deviation from planar extension  |v| / max|v|",
            SEQUENTIAL,
            Normalize(0, 1),
            "|v| / max|v|,   v = u − ε̇(x, −y)",
            [0, 0.5, 1],
        ),
    }

    n, k = len(cases), len(panel_names)
    groups_per_row = 1 if n == 1 or k >= 3 else 2
    nrows = math.ceil(n / groups_per_row)
    P, gap, group_gap, left = 2.7, 0.12, 0.45, 0.35
    right = 0.2 + 0.85 * k  # one colour bar per panel kind
    group_w = k * P + (k - 1) * gap
    W = left + groups_per_row * group_w + (groups_per_row - 1) * group_gap + right
    line = 7.5 * 1.32 / 72
    chars = group_w * 15.5
    titles = []
    for i, c in enumerate(cases):
        t = wrap(f"{LETTERS[i]}   case {c} · {rows[c]['status']}", chars)
        if vary:
            t += wrap(param_text(rows[c], vary), chars)
        titles.append(t)
    title_h = max(len(t) for t in titles) * line + 0.30  # + the panel labels
    caption = f"{config['model']}" + (
        f" · same for all: {param_text(rows[cases[0]], common)}" if common else ""
    )
    cap = wrap(caption, (W - 0.7) * 13.5)
    if streamlines == "deviation":
        cap += wrap(
            "Lines: streamlines of the deviation v = u − ε̇(x, −y) (contours of ψ′ at "
            "fixed fractions of max|ψ′|; closer lines = faster). Solid: counter-clockwise, "
            "dashed: clockwise.",
            (W - 0.7) * 13.5,
        )
    elif streamlines == "full":
        cap += wrap(
            "Lines: streamlines of the full flow u (contours of ε̇xy + ψ′ at fixed "
            "levels; thick: the separatrix ψ = 0).",
            (W - 0.7) * 13.5,
        )
    header_h = 0.42 + len(cap) * 8 * 1.32 / 72 + 0.15
    H = header_h + nrows * (title_h + P) + (nrows - 1) * 0.2 + 0.25
    fig = plt.figure(figsize=(W, H), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    box = lambda x, y, w, h: [x / W, y / H, w / W, h / H]
    clock = fig.text(left / W, 1 - 0.12 / H, "", fontsize=10, color=INK, va="top")
    fig.text(
        left / W,
        1 - 0.42 / H,
        "\n".join(cap),
        fontsize=8,
        color=INK2,
        va="top",
        linespacing=1.32,
    )

    panels = []
    for i, c in enumerate(cases):
        ri, ci = divmod(i, groups_per_row)
        x = left + ci * (group_w + group_gap)
        y_top = H - header_h - ri * (title_h + P + 0.2)
        fig.patches.append(
            plt.Rectangle(
                (x / W, (y_top - 0.13) / H),
                0.11 / W,
                0.11 / H,
                transform=fig.transFigure,
                color=SERIES[i],
                figure=fig,
            )
        )
        fig.text(
            (x + 0.17) / W,
            y_top / H,
            titles[i][0],
            fontsize=7.5,
            color=INK,
            va="top",
            weight="bold",
        )
        if len(titles[i]) > 1:
            fig.text(
                x / W,
                (y_top - line) / H,
                "\n".join(titles[i][1:]),
                fontsize=7.5,
                color=INK2,
                va="top",
                linespacing=1.32,
            )
        L = math.sqrt(abs(np.linalg.det(data[c]["frames"]["lattice"][0])))
        win = (-0.75 * L, 0.75 * L, -0.75 * L, 0.75 * L)
        if streamlines != "none":  # Cartesian grid for the contours, over the window
            g = np.linspace(-0.75 * L, 0.75 * L, 241)
            data[c]["grid"] = np.meshgrid(g, g, indexing="xy")
        data[c]["L"] = L
        handles = []
        for j, name in enumerate(panel_names):
            label, cmap, norm = specs[name][:3]
            ax = fig.add_axes(box(x + j * (P + gap), y_top - title_h - P, P, P))
            style_axes(ax)
            fig.text(
                (x + j * (P + gap)) / W,
                (y_top - title_h + 0.04) / H,
                label,
                fontsize=7.5,
                color=INK2,
                va="bottom",
            )
            tiles = TiledField(ax, win, cmap=cmap, norm=norm, zorder=1)
            (cell,) = ax.plot([], [], "-", color=INK, lw=0.9, zorder=3)
            note = ax.text(
                0.02,
                0.03,
                "",
                transform=ax.transAxes,
                fontsize=7,
                color=INK,
                zorder=4,
                bbox=dict(fc=SURFACE, ec="none", alpha=0.85),
            )
            handles.append(dict(name=name, ax=ax, tiles=tiles, cell=cell, note=note))
        panels.append((c, handles))

    grid_h = H - header_h - 0.25
    for j, name in enumerate(panel_names):
        _, cmap, norm, label, ticks = specs[name]
        cax = fig.add_axes(
            box(W - right + 0.25 + j * 0.85, 0.25 + 0.15 * grid_h, 0.12, 0.7 * grid_h)
        )
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, ticks=ticks
        )
        cb.set_label(label, color=INK2, fontsize=7.5)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_edgecolor(GRID)
    return fig, clock, panels, data


def streamline_field(F, A, X, Y, eps_dot, L, streamlines, levels):
    """(Z, contour levels, line widths, negative line style) for the overlay on the
    Cartesian grid (X, Y). deviation: psi' / max|psi'| at +-k/levels; full: eps_dot x y + psi'
    at levels fixed by the base flow, so lines do not jump between frames."""
    psi = sample_periodic(F["psi"], A, X, Y)
    if streamlines == "deviation":
        m = float(np.abs(F["psi"]).max())
        Z = psi / m if m > 0 else psi
        lv = [j / levels for j in range(-levels + 1, levels) if j != 0]
        return Z, lv, 0.6, "dashed"
    Z = eps_dot * X * Y + psi
    M = eps_dot * (0.75 * L) ** 2  # base-flow value at the window corners
    lv = list(np.linspace(-M, M, 2 * levels + 1)[1:-1])
    return Z, lv, [1.2 if abs(v) < 1e-12 * M else 0.6 for v in lv], "solid"


def draw(panels, data, e, clock, streamlines="none", levels=8):
    """Show every case at the snapshot nearest strain e (or its last one if it stopped)."""
    for c, handles in panels:
        d = data[c]
        fr, p, L = d["frames"], d["p"], d["L"]
        f = int(np.argmin(np.abs(fr["strain"] - min(e, fr["strain"][-1]))))
        need = {h["name"] for h in handles} | (
            {"psi"} if streamlines != "none" else set()
        )
        F = frame_fields(fr, d["lam"], f, p, need)
        A = fr["lattice"][f]
        stopped = "  (run stopped)" if e > fr["strain"][-1] + 1e-9 else ""
        if streamlines != "none":
            Z, lv, lw, neg = streamline_field(
                F, A, *d["grid"], p.eps_dot, L, streamlines, levels
            )
        for h in handles:
            field = F[h["name"]]
            if h["name"] == "xi":
                h["note"].set_text(f"min ξ = {field.min():.4g}")
            else:
                peak = float(np.abs(field).max())
                if h["name"] == "stretch":
                    text = f"⟨λ⟩ = {F['mean']:.4g}, max|δλ| = {peak:.2g}{stopped}"
                else:
                    text = f"max|v| = {peak / (p.eps_dot * L):.2g} ε̇ℓ{stopped}"
                h["note"].set_text(text)
                field = field / peak if peak > 0 else field
            h["tiles"].draw(field, A)
            h["cell"].set_data(*cell_outline(A))
            if streamlines != "none":
                if h.get("lines") is not None:
                    h["lines"].remove()
                h["lines"] = h["ax"].contour(
                    *d["grid"],
                    Z,
                    levels=lv,
                    colors=INK,
                    linewidths=lw,
                    negative_linestyles=neg,
                    zorder=2,
                )
    clock.set_text(f"Hencky strain ε = {e:5.2f}")


def auto_range(folder, cases, strains):
    """(lo, 1) with lo just below the smallest xi over the frames nearest these strains."""
    config, rows, _ = load_sweep(folder)
    xi_min = 1.0
    for c in cases:
        p = case_params(config, rows[c])
        fr = load_frames(folder, c)
        for f in sorted(
            {
                int(np.argmin(np.abs(fr["strain"] - min(e, fr["strain"][-1]))))
                for e in strains
            }
        ):
            kap = velocity_gradient(
                fr["dC"][f].astype(float), fr["lattice"][f], p.G, p.eta_s, p.eps_dot
            )
            xi_min = min(xi_min, float(flow_type(*(upsample(k) for k in kap)).min()))
    gap = 1 - xi_min
    if gap <= 0:
        return (0.99, 1.0)
    step = 10 ** math.floor(math.log10(gap))  # round the gap up to a 1-2-5 value
    nice = next(m * step for m in (1, 2, 5, 10) if m * step >= gap)
    return (max(-1.0, 1 - nice), 1.0)


def range_tag(xi_range, auto):
    if auto:
        return "_xi-auto"
    return "" if xi_range == (-1.0, 1.0) else f"_xi_{xi_range[0]:g}-{xi_range[1]:g}"


def view_tag(panel_names, streamlines):
    """File-name suffix for non-default panels and overlays (empty for the defaults)."""
    tag = "" if tuple(panel_names) == ("stretch", "xi") else "_" + "-".join(panel_names)
    return tag + ("" if streamlines == "none" else f"_stream-{streamlines}")


def strain_grid(folder, cases):
    strains = [load_frames(folder, c)["strain"] for c in cases]
    step = max(np.median(np.diff(s)) for s in strains)
    e_max = max(s[-1] for s in strains)
    grid = np.arange(0, e_max + 0.5 * step, step)
    return np.append(grid, e_max) if grid[-1] < e_max - 1e-9 else grid


def movie(folder, cases, xi_range, fps, panel_names, streamlines, levels):
    grid = strain_grid(folder, cases)
    auto = xi_range == "auto"
    if auto:
        xi_range = (
            auto_range(folder, cases, grid) if "xi" in panel_names else (-1.0, 1.0)
        )
        print(f"xi colour range {xi_range[0]:g} to 1")
    fig, clock, panels, data = build(folder, cases, xi_range, panel_names, streamlines)
    out = os.path.join(folder, "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(
        out,
        f"flowtype_cases_{'-'.join(map(str, cases))}{range_tag(xi_range, auto)}"
        f"{view_tag(panel_names, streamlines)}.mp4",
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ],
    )
    with writer.saving(fig, path, dpi=110):
        for e in grid:
            draw(panels, data, e, clock, streamlines, levels)
            writer.grab_frame()
    plt.close(fig)
    print("wrote", path)


def still(folder, cases, xi_range, strain, panel_names, streamlines, levels):
    auto = xi_range == "auto"
    if auto:
        xi_range = (
            auto_range(folder, cases, [strain]) if "xi" in panel_names else (-1.0, 1.0)
        )
        print(f"xi colour range {xi_range[0]:g} to 1")
    fig, clock, panels, data = build(folder, cases, xi_range, panel_names, streamlines)
    draw(panels, data, strain, clock, streamlines, levels)
    out = os.path.join(folder, "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(
        out,
        f"flowtype_cases_{'-'.join(map(str, cases))}_strain_{strain:g}"
        f"{range_tag(xi_range, auto)}{view_tag(panel_names, streamlines)}.png",
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("movie", "still"):
        sp = sub.add_parser(name)
        sp.add_argument("folder")
        sp.add_argument("--cases", type=int, nargs="+", required=True)
        sp.add_argument(
            "--xi-range",
            default="-1:1",
            help="colour range for xi: lo:hi, or auto (default -1:1); write a "
            "negative lower end as --xi-range=-0.5:1",
        )
        sp.add_argument(
            "--panels",
            nargs="+",
            choices=("stretch", "xi", "speed"),
            default=["stretch", "xi"],
            help="panels per case, in order (default: stretch xi)",
        )
        sp.add_argument(
            "--streamlines",
            choices=("none", "deviation", "full"),
            default="none",
            help="overlay streamlines of v = u - eps_dot (x, -y) (deviation) or of u "
            "(full) on every panel (default none)",
        )
        sp.add_argument(
            "--levels",
            type=int,
            default=8,
            help="streamline levels per sign (default 8)",
        )
        if name == "movie":
            sp.add_argument("--fps", type=int, default=12)
        else:
            sp.add_argument(
                "--strain",
                type=float,
                required=True,
                help="Hencky strain to show (nearest saved snapshot)",
            )
    args = ap.parse_args(argv)
    rng = parse_range(args.xi_range)
    view = (args.panels, args.streamlines, args.levels)
    if args.cmd == "movie":
        movie(args.folder, args.cases, rng, args.fps, *view)
    else:
        still(args.folder, args.cases, rng, args.strain, *view)


if __name__ == "__main__":
    main()
