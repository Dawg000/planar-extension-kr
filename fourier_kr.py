"""Fourier analysis of the chain stretch for sweep_kr.py runs (needs --save-fields).

  python fourier_kr.py compare SWEEP_DIR --cases 3 7 12 [--window 0.5:1] [--fold]
      Strain-window-averaged polar spectra of the stretch fluctuation, one map per case
      (up to 8), with detected peaks marked, plus the averaged radial spectra overlaid.

  python fourier_kr.py peaks SWEEP_DIR [--cases all|3 7 12] [--x eps_dot] [--hue beta]
      Dominant peaks for many cases (in parallel): fourier_peaks.csv, and a summary of
      peak wavelength and orientation against one swept parameter.

  python fourier_kr.py spectrogram SWEEP_DIR --case 7
      Radial spectrum against strain (coarsening / sharpening), with the peak wavenumber.

  python fourier_kr.py across SWEEP_DIR [SWEEP_DIR ...] --case 4 [--field txx] [--inject A,S]
      1D spectrum across the structures for one case, pooled over sweep folders that differ
      only in the seed, against a fitted smooth background; peaks are runs of bins above
      the background by > 4 sigma and > 20% (see "Across spectrum" below).

  python fourier_kr.py kmap SWEEP_DIR [SWEEP_DIR ...] --case 4 [--field txx] [--fold]
      Mode power on a fixed (kx, ky) grid, averaged over the window (and the folders, e.g.
      seeds): the window average of single_frame_spectrum's k-plane panel.

Window (compare, peaks, across, kmap): --window A:B keeps frames between A and B times the
run's final strain (default 0.5:1). --skip-strain S discards frames before Hencky strain S
instead, e.g. --skip-strain 10 for a run to strain 50 keeps strains 10-50.

Folding (--fold; compare, peaks, kmap): planar extension is symmetric under x -> -x and
y -> -y, which take theta to 180 deg - theta, so mirror-image structures (e.g. bands at
+-30 deg) are one feature. --fold adds theta and 180 - theta (kmap: (kx, ky) -> (|kx|,
|ky|)), giving one peak with the pooled power. The KR cell itself is not mirror-symmetric
(its lattice is tilted), so compare an unfolded map first: folding hides any real
asymmetry. The "overall" mean orientation and anisotropy are always from the unfolded
spectrum.

Field analysed: the chain stretch lambda = sqrt(tr C / 3), as its fluctuation lambda - <lambda>.
tr C comes from the saved tau/G fluctuation plus its cell average (finite extensibility is
inverted: tr(fC) = f tr C). Sweeps saved before the cell average was stored use the
homogeneous steady state instead, which is exact to first order in the perturbation.

Why spectra are binned in physical k: in the KR cell, lattice mode n has the physical
wavevector k = 2 pi A(t)^-T n, which changes continuously as the cell deforms and jumps
to a new mode label at every remap. Each frame's power |F_n|^2 is therefore put into fixed
(|k|, theta) bins, theta = angle of k from the extension (x) axis folded into [0, 180),
and frames are averaged on those bins. A bin holds the power of the modes that fall in it,
so the bins of a frame sum to its stretch variance. Maps show variance per unit ln|k| per
radian, so bins of different widths compare. |k| bins are never narrower than the mode
spacing 2 pi / L, and at small |k|, where a ring holds only a few modes, adjacent angle bins
are averaged until each holds about one mode. theta = 90 deg means variation along y, i.e.
stripes or filaments aligned with the extension axis.

Across spectrum (subcommand across): the frame-averaged <k k^T> picks the direction e of
strongest variation (normal to the structures; structure_kr's 'across' axis). Each frame's
mode powers, normalized to unit variance, are binned by |e.k|: this is the spectrum of the
field's cut along e. Frames are split into contiguous blocks, and block means (from every
folder given) set the error of each bin. The background at each bin is a local quadratic fit of ln E
against ln k to the bins within a factor 2.5 on either side, leaving out the bin and its
neighbours (no assumed global shape), and features are judged against it, not against the
median of all bins: the median sits in the empty high-k tail, so against it the broad band itself
looks like a peak. --inject A,S adds a stripe of rms A x the field rms and spacing S along
e first, to calibrate what amplitude the test can detect.

Peaks: bins above a percentile of the nonzero bins are grouped into connected regions
(8-connected, with theta wrapping from 180 back to 0), and each region gives a power-
weighted wavenumber, orientation, power fraction and contrast (peak bin / median bin).
A percentile threshold always selects something, so regions with contrast under 2 are
dropped as background; judge the rest by contrast and power fraction. For broad,
noise-driven spectra the "overall" numbers (variance-weighted mean |k|, mean orientation
and anisotropy from 0 = isotropic to 1 = one direction) are the more robust summary.
"""

import argparse
import csv
import math
import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter
from scipy import ndimage

from kr_solver import XX, YY, ZZ, stress_conformation
from sweep_kr import steady_state
from viz_common import (
    GRID,
    INK,
    INK2,
    MUTED,
    SERIES,
    SURFACE,
    axis_values,
    set_axis_labels,
)
from viz_sweep import case_params, load_sweep, param_text, style_plot, wrap

# Sequential single-hue ramp (palette blue, light -> dark)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues",
    [
        "#eef5fd",
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
)
LETTERS = "ABCDEFGH"


# --- fields and spectra ---------------------------------------------------------------
def load_frames(folder, c):
    path = os.path.join(folder, "fields", f"case_{c:04d}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: rerun the sweep with --save-fields")
    with np.load(path) as d:
        frames = {k: d[k] for k in d.files}
    if frames["dC"].shape[1] < 4:
        raise SystemExit(
            f"case {c}: the stretch needs the zz component, which sweeps run "
            "before it was saved do not have; rerun with the current sweep_kr.py"
        )
    frames["strain"] = float(frames["eps_dot"]) * frames["t"]
    return frames


def stretch_fluctuation(frames, p):
    """lambda - <lambda> on the lattice grid, one array per frame (float64)."""
    lam = stretch_field(frames, p)
    return lam - lam.mean(axis=(1, 2), keepdims=True)


def stretch_field(frames, p):
    """Chain stretch lambda = sqrt(tr C / 3) on the lattice grid, one array per frame."""
    dC = frames[
        "dC"
    ]  # float32; only the trace is promoted, to keep memory down at large N
    if "mean" in frames:
        mean_tr = frames["mean"][:, XX] + frames["mean"][:, YY] + frames["mean"][:, ZZ]
    else:  # older sweeps: the homogeneous steady state stands in for the cell average
        S0 = stress_conformation(steady_state(p), p)
        mean_tr = np.full(len(dC), S0[XX] + S0[YY] + S0[ZZ])
    tr_S = mean_tr[:, None, None] + (dC[:, XX] + dC[:, YY] + dC[:, ZZ]).astype(float)
    if p.L_max:  # tr(fC) = f tr C, f = a / (1 - trC / 3L^2), a = 1 - 1/L^2
        L2 = p.L_max**2
        tr_C = tr_S / ((1 - 1 / L2) + tr_S / (3 * L2))
    else:
        tr_C = tr_S
    return np.sqrt(tr_C / 3)


def mode_wavevectors(A, N):
    """Physical (kx, ky) of the dealiased modes (k = 0 excluded) and their index mask."""
    n = np.fft.fftfreq(N, 1.0 / N)
    n1, n2 = np.meshgrid(n, n, indexing="ij")
    keep = (np.abs(n1) <= N // 3) & (np.abs(n2) <= N // 3)
    keep[0, 0] = False
    B = 2 * np.pi * np.linalg.inv(A).T
    return (
        B[0, 0] * n1[keep] + B[0, 1] * n2[keep],
        B[1, 0] * n1[keep] + B[1, 1] * n2[keep],
        keep,
    )


def frame_power(field, A):
    """(kx, ky, power) for one real field; the powers sum to the field variance over the
    dealiased modes (Parseval)."""
    N = field.shape[-1]
    kx, ky, keep = mode_wavevectors(A, N)
    F = np.fft.fft2(field) / N**2
    return kx, ky, np.abs(F[keep]) ** 2


def k_edges_for(lattices, N, n_k=48):
    """|k| bin edges covering every resolved mode of every frame: log-spaced, but never
    narrower than the lattice mode spacing 2 pi / L, so small-|k| bins do not alternate
    between catching a mode and being empty."""
    lo, hi = np.inf, 0.0
    for A in lattices:
        kx, ky, _ = mode_wavevectors(A, N)
        k = np.hypot(kx, ky)
        lo, hi = min(lo, k.min()), max(hi, k.max())
    dk = 2 * np.pi / math.sqrt(abs(np.linalg.det(lattices[0])))
    return log_edges_min_width(lo * 0.999, hi * 1.001, dk, n_k)


def log_edges_min_width(lo, hi, dk, n_k):
    """Log-spaced edges from lo to hi (ratio (hi/lo)^(1/n_k)), widened to at least dk."""
    r = (hi / lo) ** (1 / n_k)
    edges = [lo]
    while edges[-1] < hi:
        edges.append(max(edges[-1] * r, edges[-1] + dk))
    edges[-1] = max(edges[-1], hi)
    return np.array(edges)


def merge_sparse_angles(S, k_edges, n_theta, L, fold=False):
    """At small |k| only a few modes lie on each ring, so narrow angle bins alternate between
    catching a mode and being empty. In each |k| row, average groups of g adjacent angle
    bins, with g chosen so a group holds at least one mode on average (theta is cyclic).
    The row's total variance is unchanged. fold: S has been through fold_theta, so each
    bin holds twice the modes (and theta is not cyclic, but grouping does not use that)."""
    cell = (2 * np.pi / L) ** 2  # k-space area per lattice mode
    out = S.copy()
    n_cols = S.shape[1]
    for i in range(S.shape[0]):
        # modes per folded bin: k and -k fall in the same bin, so twice the sector area
        per_bin = (
            2 * (np.pi / n_theta) * (k_edges[i + 1] ** 2 - k_edges[i] ** 2) / 2 / cell
        )
        if fold:
            per_bin *= 2
        g = min(n_cols, int(math.ceil(1 / per_bin))) if per_bin < 1 else 1
        if g > 1:
            groups = np.arange(n_cols) // g
            for gi in np.unique(groups):
                sel = groups == gi
                out[i, sel] = S[i, sel].mean()
    return out


def bin_density(S, k_edges, n_theta):
    """Variance per unit ln|k| per radian, so bins of different widths are comparable."""
    return S / np.log(k_edges[1:] / k_edges[:-1])[:, None] / (np.pi / n_theta)


def overall(S, k_edges, n_theta):
    """Variance-weighted geometric-mean |k|, mean orientation and anisotropy (0 isotropic,
    1 all variance in one direction) of a binned spectrum."""
    kc = np.sqrt(k_edges[:-1] * k_edges[1:])[:, None]
    th = ((np.arange(n_theta) + 0.5) * np.pi / n_theta)[None, :]
    tot = S.sum()
    if tot <= 0:
        return dict(
            k_mean=float("nan"), theta_mean_deg=float("nan"), anisotropy=float("nan")
        )
    c, s = np.sum(S * np.cos(2 * th)) / tot, np.sum(S * np.sin(2 * th)) / tot
    return dict(
        k_mean=float(np.exp(np.sum(S * np.log(kc)) / tot)),
        theta_mean_deg=round(math.degrees(0.5 * math.atan2(s, c)), 9) % 180,
        anisotropy=float(math.hypot(c, s)),
    )


def bin_polar(kx, ky, power, k_edges, n_theta):
    """Sum of power in (|k|, theta) bins, theta in [0, pi)."""
    k = np.hypot(kx, ky)
    th = np.mod(np.arctan2(ky, kx), np.pi)
    H, _, _ = np.histogram2d(
        k, th, bins=[k_edges, np.linspace(0, np.pi, n_theta + 1)], weights=power
    )
    return H


# --- peaks ----------------------------------------------------------------------------
def find_peaks(
    S, k_edges, n_theta, percentile=99.0, max_peaks=3, min_contrast=2.0, fold=False
):
    """Connected regions where the variance density is above a percentile of its nonzero
    bins, with theta wrapping around (not when fold: then 0 and 90 deg are mirror edges). S holds the variance per bin; detection uses the
    density (per unit ln|k| and angle), the reported fractions use S itself. Regions whose
    highest density is under min_contrast x the median are background and are dropped."""
    P = bin_density(S, k_edges, n_theta)
    nz = P[P > 0]
    if nz.size == 0:
        return []
    mask = P >= np.percentile(nz, percentile)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    parent = list(range(n + 1))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nk, n_cols = P.shape
    for i in range(nk if not fold else 0):  # theta = 180 deg is theta = 0: join regions
        for j in (i - 1, i, i + 1):
            if 0 <= j < nk and lab[i, -1] and lab[j, 0]:
                parent[root(lab[i, -1])] = root(lab[j, 0])
    lab = np.vectorize(lambda v: root(v) if v else 0)(lab)

    kc = np.sqrt(k_edges[:-1] * k_edges[1:])[:, None] * np.ones((1, n_cols))
    thc = (np.arange(n_cols) + 0.5) * np.pi / n_theta * np.ones((nk, 1))
    total, median = S.sum(), np.median(nz)
    peaks = []
    for r in np.unique(lab[lab > 0]):
        sel = lab == r
        w = S[sel]
        k_star = float(np.exp(np.sum(w * np.log(kc[sel])) / w.sum()))
        contrast = float(P[sel].max() / median)
        if contrast < min_contrast:
            continue
        th = 0.5 * math.atan2(
            np.sum(w * np.sin(2 * thc[sel])), np.sum(w * np.cos(2 * thc[sel]))
        )
        peaks.append(
            dict(
                k=k_star,
                wavelength=2 * np.pi / k_star,
                theta_deg=round(math.degrees(th), 9) % 180,
                power_fraction=float(w.sum() / total),
                contrast=contrast,
            )
        )
    return sorted(peaks, key=lambda q: -q["power_fraction"])[:max_peaks]


# --- one case -------------------------------------------------------------------------
def parse_window(text):
    a, _, b = text.partition(":")
    return float(a), float(b)


def window_mask(e, window, skip=None):
    """Frames in the strain window, and its (start, end) strain. window = (start, end) as
    fractions of the final strain; skip, if given, is an absolute Hencky strain to discard
    from the start, and replaces the window's start."""
    e0 = window[0] * e[-1] if skip is None else skip
    e1 = window[1] * e[-1]
    return (e >= e0 - 1e-9) & (e <= e1 + 1e-9), e0, e1


def window_text(window, skip=None):
    if skip is None:
        return f"strain {window[0]:g}–{window[1]:g} of each run's final strain"
    end = "the end" if window[1] == 1 else f"{window[1]:g} × the final strain"
    return f"strain {skip:g} to {end}"


def fold_theta(S):
    """Mirror theta -> 180 deg - theta (the reflections x -> -x, y -> -y of planar
    extension) onto [0, 90 deg]: angle bin j is added to bin n_theta - 1 - j. The bin
    width is unchanged, so a folded bin holds twice the modes and the total is kept."""
    n = S.shape[-1]
    if n % 2:
        raise SystemExit("--fold needs an even --n-theta")
    return S[..., : n // 2] + S[..., ::-1][..., : n // 2]


def analyse_case(
    folder,
    config,
    row,
    window=(0.5, 1.0),
    n_k=48,
    n_theta=36,
    normalize=False,
    k_edges=None,
    percentile=99.0,
    skip=None,
    fold=False,
):
    """Spectra of the stretch fluctuation for one case.

    window: (start, end) as fractions of the case's final strain; frames in that strain
            range are averaged. skip: absolute strain to discard instead of the window's
            start (see window_mask). fold: mirror theta onto [0, 90 deg] (fold_theta)
            before peak finding; the "overall" numbers use the unfolded spectrum.
            normalize: divide each frame's spectrum by its variance
            first, so the average weighs the shape of every frame equally (otherwise
            later, larger-amplitude frames dominate)."""
    c = int(row["case"])
    p = case_params(config, row)
    frames = load_frames(folder, c)
    lam = stretch_fluctuation(frames, p)
    e = frames["strain"]
    N = lam.shape[-1]
    if k_edges is None:
        k_edges = k_edges_for(frames["lattice"], N, n_k)
    nk = len(k_edges) - 1
    polar = np.zeros((len(e), nk, n_theta))
    for f in range(len(e)):
        H = bin_polar(*frame_power(lam[f], frames["lattice"][f]), k_edges, n_theta)
        polar[f] = H / H.sum() if normalize and H.sum() > 0 else H
    sel, e0, e1 = window_mask(e, window, skip)
    if not sel.any():
        raise SystemExit(f"case {c}: no frames in strain window {e0:.3g}-{e1:.3g}")
    L = math.sqrt(abs(np.linalg.det(frames["lattice"][0])))
    mean = polar[sel].mean(axis=0)
    avg = merge_sparse_angles(mean, k_edges, n_theta, L)
    shown = (
        merge_sparse_angles(fold_theta(mean), k_edges, n_theta, L, True)
        if fold
        else avg
    )
    radial = polar.sum(axis=2)  # per frame, per |k| bin
    return dict(
        case=c,
        p=p,
        strain=e,
        window=(e0, e1),
        n_frames=int(sel.sum()),
        k_edges=k_edges,
        n_theta=n_theta,
        fold=fold,
        polar=shown,
        radial_avg=radial[sel].mean(axis=0),
        radial_t=radial,
        variance=float(np.mean(np.var(lam[sel], axis=(1, 2)))),
        peaks=find_peaks(shown, k_edges, n_theta, percentile, fold=fold),
        overall=overall(avg, k_edges, n_theta),
    )


# --- plotting -------------------------------------------------------------------------
def draw_polar(ax, res, vmin, vmax):
    k_edges = res["k_edges"]
    th_max = 90 if res["fold"] else 180
    th_edges = np.linspace(0, th_max, res["polar"].shape[1] + 1)
    D = bin_density(res["polar"], k_edges, res["n_theta"])
    P = np.where(D > 0, D, np.nan)
    m = ax.pcolormesh(
        k_edges,
        th_edges,
        P.T,
        cmap=SEQUENTIAL,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="flat",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, th_max)
    ax.set_yticks([0, 45, 90, 135, 180][: 5 if th_max == 180 else 3])
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    for i, pk in enumerate(res["peaks"]):
        ax.plot(pk["k"], pk["theta_deg"], "o", ms=9, mfc="none", mec=INK, mew=1.4)
        ax.annotate(
            str(i + 1),
            (pk["k"], pk["theta_deg"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            weight="bold",
        )
    return m


def peak_text(res):
    o = res["overall"]
    lines = [
        f"overall: mean λ {2 * np.pi / o['k_mean']:.3g} · mean θ "
        f"{o['theta_mean_deg']:.0f}° · anisotropy {o['anisotropy']:.2f}"
    ]
    lines += [
        f"peak {i + 1}: λ {pk['wavelength']:.3g} · θ {pk['theta_deg']:.0f}° · "
        f"{100 * pk['power_fraction']:.0f}% · ×{pk['contrast']:.0f}"
        for i, pk in enumerate(res["peaks"])
    ]
    return lines


def compare(
    folder, cases, window, normalize, n_k, n_theta, percentile, skip=None, fold=False
):
    config, rows, _ = load_sweep(folder)
    names = list(config["sweeps"])
    if not 1 <= len(cases) <= len(LETTERS):
        raise SystemExit(f"choose 1 to {len(LETTERS)} cases")
    # common bins so the maps and radial curves line up
    # (never narrower than the smallest mode spacing, as in k_edges_for)
    lo, hi, dk = np.inf, 0.0, np.inf
    for c in cases:
        fr = load_frames(folder, c)
        ke = k_edges_for(fr["lattice"], fr["dC"].shape[-1], n_k)
        lo, hi = min(lo, ke[0]), max(hi, ke[-1])
        dk = min(dk, 2 * np.pi / math.sqrt(abs(np.linalg.det(fr["lattice"][0]))))
    edges = log_edges_min_width(lo, hi, dk, n_k)
    res = [
        analyse_case(
            folder,
            config,
            rows[c],
            window,
            n_k,
            n_theta,
            True,
            edges,
            percentile,
            skip,
            fold,
        )
        for c in cases
    ]
    vary = [k for k in names if len({repr(rows[c][k]) for c in cases}) > 1]
    common = [k for k in names if k not in vary]

    n = len(cases)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    P, gap, left, right = 3.0, 0.55, 0.7, 1.0
    W = left + ncols * P + (ncols - 1) * gap + right
    line = 7.5 * 1.32 / 72
    titles = []
    for i, (c, r) in enumerate(zip(cases, res)):
        t = wrap(f"{LETTERS[i]}   case {c} · {rows[c]['status']}", P * 15.5)
        if vary:
            t += wrap(param_text(rows[c], vary), P * 15.5)
        for s in peak_text(r):
            t += wrap(s, P * 15.5)
        titles.append(t)
    title_h = max(len(t) for t in titles) * line + 0.22
    span = window_text(window, skip)
    head = (
        f"Chain-stretch spectra, averaged over {span}, each frame normalized to unit "
        f"variance{', θ folded onto 0–90° (θ and 180° − θ added)' if fold else ''}. "
        f"Peaks: wavelength λ = 2π/|k|, angle θ of k, % of variance, ×contrast "
        f"(peak / median density)."
    )
    cap = wrap(
        head
        + (
            f" · {config['model']}, same for all: {param_text(rows[cases[0]], common)}"
            if common
            else f" · {config['model']}"
        ),
        (W - 0.8) * 13.5,
    )
    header_h = 0.2 + len(cap) * 8 * 1.32 / 72 + 0.15
    strip_h = 2.2
    H = header_h + nrows * (title_h + P + 0.45) + strip_h + 0.9
    fig = plt.figure(figsize=(W, H), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    box = lambda x, y, w, h: [x / W, y / H, w / W, h / H]
    fig.text(
        left / W,
        1 - 0.15 / H,
        "\n".join(cap),
        fontsize=8,
        color=INK2,
        va="top",
        linespacing=1.32,
    )

    dens = [bin_density(r["polar"], r["k_edges"], r["n_theta"]) for r in res]
    vals = np.concatenate([d[d > 0] for d in dens])
    vmax = vals.max()
    vmin = max(vals.min(), vmax * 1e-6)
    for i, r in enumerate(res):
        ri, ci = divmod(i, ncols)
        x = left + ci * (P + gap)
        y_top = H - header_h - ri * (title_h + P + 0.45)
        ax = fig.add_axes(box(x, y_top - title_h - P, P, P))
        mesh = draw_polar(ax, r, vmin, vmax)
        ax.set_xlabel("|k|", color=INK2, fontsize=8)
        if ci == 0:
            ax.set_ylabel("θ of k from extension axis (°)", color=INK2, fontsize=8)
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
        fig.text(
            x / W,
            (y_top - line) / H,
            "\n".join(titles[i][1:]),
            fontsize=7.5,
            color=INK2,
            va="top",
            linespacing=1.32,
        )
    grid_bottom = H - header_h - nrows * (title_h + P + 0.45) + 0.45
    cax = fig.add_axes(
        box(W - right + 0.25, grid_bottom + 0.2, 0.14, H - header_h - grid_bottom - 0.4)
    )
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label(
        "fraction of stretch variance per unit ln|k| per rad", color=INK2, fontsize=8
    )
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor(GRID)

    ax = fig.add_axes(box(left, 0.6, ncols * P + (ncols - 1) * gap, strip_h))
    style_plot(ax)
    kc = np.sqrt(edges[:-1] * edges[1:])
    dlog = np.log10(edges[1:] / edges[:-1])
    ends = []
    for i, r in enumerate(res):
        E = r["radial_avg"] / dlog
        ok = E > 0
        ax.loglog(kc[ok], E[ok], color=SERIES[i], lw=1.8)
        i_pk = np.argmax(E[ok])
        ax.plot(kc[ok][i_pk], E[ok][i_pk], "o", ms=4, color=SERIES[i])
        ends.append((kc[ok][-1], E[ok][-1], LETTERS[i]))
    ax.set_xlim(right=kc[-1] * 1.6)
    ax.get_ylim()
    placed = sorted(
        (ax.transData.transform((x_, y_))[1], x_, y_, lab) for x_, y_, lab in ends
    )
    last = -1e9
    for (
        yd,
        x_,
        y_,
        lab,
    ) in placed:  # letter at each curve's end, nudged apart vertically
        yd_new = max(yd, last + 10 * fig.dpi / 72)
        ax.annotate(
            lab,
            (x_, y_),
            xytext=(4, (yd_new - yd) * 72 / fig.dpi),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
            weight="bold",
            va="center",
        )
        last = yd_new
    ax.set_xlabel("|k|", color=INK2, fontsize=8.5)
    ax.set_ylabel("variance per decade of |k|", color=INK2, fontsize=8.5)
    ax.set_title(
        "Radial spectra (same averaging; dot = maximum)",
        color=INK,
        fontsize=8.5,
        loc="left",
    )
    out = os.path.join(folder, "figures")
    os.makedirs(out, exist_ok=True)
    tag = "_folded" if fold else ""
    path = os.path.join(
        out, f"fourier_stretch_cases_{'-'.join(map(str, cases))}{tag}.png"
    )
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)
    for c, r in zip(cases, res):
        print(
            f"case {c}: {r['n_frames']} frames, strain {r['window'][0]:.2f}–{r['window'][1]:.2f}"
        )
        for s in peak_text(r):
            print("   ", s)


def spectrogram(folder, c, n_k, n_theta):
    config, rows, _ = load_sweep(folder)
    r = analyse_case(folder, config, rows[c], (0.0, 1.0), n_k, n_theta, False)
    edges, e = r["k_edges"], r["strain"]
    S = r["radial_t"]
    frac = S / np.where(
        S.sum(axis=1, keepdims=True) > 0, S.sum(axis=1, keepdims=True), 1
    )
    frac_d = frac / np.log(edges[1:] / edges[:-1])[None, :]  # per unit ln|k|
    kc = np.sqrt(edges[:-1] * edges[1:])
    k_peak = np.exp(np.sum(frac * np.log(kc), axis=1))
    de = np.diff(e).mean() if len(e) > 1 else 1.0
    e_edges = np.concatenate([e - de / 2, [e[-1] + de / 2]])
    names = list(config["sweeps"])
    title = wrap(f"case {c} · {rows[c]['status']} · {param_text(rows[c], names)}", 100)
    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(7.5, 6.2 + 0.15 * len(title)),
        dpi=130,
        gridspec_kw=dict(height_ratios=[2.2, 1]),
        sharex=True,
    )
    fig.patch.set_facecolor(SURFACE)
    vals = frac_d[frac_d > 0]
    m = a1.pcolormesh(
        e_edges,
        edges,
        np.where(frac_d > 0, frac_d, np.nan).T,
        cmap=SEQUENTIAL,
        norm=LogNorm(vmin=max(vals.min(), vals.max() * 1e-5), vmax=vals.max()),
        shading="flat",
    )
    a1.plot(e, k_peak, color=SERIES[1], lw=1.8, label="variance-weighted mean |k|")
    a1.set_yscale("log")
    a1.set_ylabel("|k|", color=INK2, fontsize=9)
    a1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    a1.tick_params(colors=MUTED, labelsize=8)
    cb = fig.colorbar(m, ax=a1, pad=0.02)
    cb.set_label("fraction of stretch variance per unit ln|k|", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor(GRID)
    style_plot(a2)
    a2.semilogy(e, S.sum(axis=1), color=INK2, lw=1.6)
    a2.set_ylabel("stretch variance", color=INK2, fontsize=9)
    a2.set_xlabel("Hencky strain ε = ε̇t", color=INK2, fontsize=9)
    fig.text(
        0.02,
        0.985,
        "\n".join(title),
        fontsize=8.5,
        color=INK,
        va="top",
        linespacing=1.3,
    )
    fig.tight_layout(
        rect=(0, 0, 1, 1 - (0.35 + 0.15 * len(title)) / (6.2 + 0.15 * len(title)))
    )
    out = os.path.join(folder, "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"fourier_spectrogram_case_{c:04d}.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def _peaks_job(job):
    folder, config, row, window, normalize, n_k, n_theta, percentile, skip, fold = job
    try:
        r = analyse_case(
            folder,
            config,
            row,
            window,
            n_k,
            n_theta,
            normalize,
            None,
            percentile,
            skip,
            fold,
        )
        return (
            int(row["case"]),
            r["peaks"],
            (r["variance"], r["overall"]),
            r["n_frames"],
            None,
        )
    except SystemExit as e:
        return int(row["case"]), [], (float("nan"), {}), 0, str(e)
    except Exception as e:
        return int(row["case"]), [], (float("nan"), {}), 0, f"{type(e).__name__}: {e}"


def peaks_table(
    folder,
    cases,
    window,
    normalize,
    n_k,
    n_theta,
    percentile,
    workers,
    x_name,
    hue_name,
    skip=None,
    fold=False,
):
    config, rows, _ = load_sweep(folder)
    names = list(config["sweeps"])
    have = [
        c
        for c in rows
        if os.path.exists(os.path.join(folder, "fields", f"case_{c:04d}.npz"))
    ]
    cases = have if cases == ["all"] else [int(c) for c in cases]
    jobs = [
        (
            folder,
            config,
            rows[c],
            window,
            normalize,
            n_k,
            n_theta,
            percentile,
            skip,
            fold,
        )
        for c in cases
    ]
    with Pool(workers) as pool:
        results = sorted(pool.imap_unordered(_peaks_job, jobs))
    tag = "_folded" if fold else ""
    out_csv = os.path.join(folder, f"fourier_peaks{tag}.csv")
    cols = [
        "case",
        *names,
        "status",
        "n_frames",
        "stretch_variance",
        "k_mean",
        "wavelength_mean",
        "theta_mean_deg",
        "anisotropy",
    ]
    for i in (1, 2, 3):
        cols += [
            f"peak{i}_wavelength",
            f"peak{i}_k",
            f"peak{i}_theta_deg",
            f"peak{i}_power_fraction",
            f"peak{i}_contrast",
        ]
    cols.append("note")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for c, peaks, var, nf, err in results:
            var, o = var
            rec = {
                **rows[c],
                **o,
                "case": c,
                "n_frames": nf,
                "stretch_variance": var,
                "note": err or "",
            }
            if o:
                rec["wavelength_mean"] = 2 * np.pi / o["k_mean"]
            for i, pk in enumerate(peaks, 1):
                rec.update({f"peak{i}_{k}": v for k, v in pk.items()})
            w.writerow(rec)
    print(f"wrote {out_csv} ({len(results)} cases)")
    failed = [(c, err) for c, _, _, _, err in results if err]
    if failed:
        print(
            f"{len(failed)} cases could not be analysed, e.g. case {failed[0][0]}: {failed[0][1]}"
        )

    good = [
        (c, peaks[0] if peaks else None, o)
        for c, peaks, (_, o), _, err in results
        if o and not err
    ]
    if not good:
        return
    x_name = x_name or names[0]
    xs, labels = axis_values([rows[c][x_name] for c, _, _ in good])
    pos = {c: xv for (c, _, _), xv in zip(good, xs)}
    hues = sorted({rows[c][hue_name] for c, _, _ in good}) if hue_name else [None]
    if len(hues) > len(SERIES):
        raise SystemExit(
            f"--hue {hue_name} has {len(hues)} values; at most {len(SERIES)}"
        )
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 6.2), dpi=130, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for a in (a1, a2):
        style_plot(a)
    for j, h in enumerate(hues):
        pts = [
            (pos[c], pk, o) for c, pk, o in good if h is None or rows[c][hue_name] == h
        ]
        x = np.array([q[0] for q in pts], float)
        lab = (
            None
            if h is None
            else f"{hue_name} = {h:g}"
            if isinstance(h, float)
            else f"{hue_name} = {h}"
        )
        a1.scatter(
            x,
            [2 * np.pi / q[2]["k_mean"] for q in pts],
            s=40,
            color=SERIES[j],
            edgecolor=SURFACE,
            linewidth=1.2,
            label=lab,
            zorder=3,
        )
        a2.scatter(
            x,
            [q[2]["theta_mean_deg"] for q in pts],
            s=40,
            color=SERIES[j],
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        pk_pts = [(xx, q[1]) for xx, q in zip(x, pts) if q[1]]
        if pk_pts:
            a1.scatter(
                [u for u, _ in pk_pts],
                [v["wavelength"] for _, v in pk_pts],
                s=55,
                facecolor="none",
                edgecolor=SERIES[j],
                linewidth=1.3,
                zorder=2,
            )
            a2.scatter(
                [u for u, _ in pk_pts],
                [v["theta_deg"] for _, v in pk_pts],
                s=55,
                facecolor="none",
                edgecolor=SERIES[j],
                linewidth=1.3,
                zorder=2,
            )
    a1.scatter([], [], s=40, color=MUTED, label="overall (variance-weighted)")
    a1.scatter([], [], s=55, facecolor="none", edgecolor=MUTED, label="strongest peak")
    if labels is None and xs.min() > 0 and xs.max() / xs.min() > 10:
        a1.set_xscale("log")
    set_axis_labels(a2, labels)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.tick_params(which="both", colors=MUTED, labelsize=8)
    a1.set_ylabel("wavelength 2π/|k|", color=INK2, fontsize=9)
    a2.set_ylabel("θ of k from extension axis (°)", color=INK2, fontsize=9)
    a2.set_ylim(0, 90 if fold else 180)
    a2.set_yticks([0, 45, 90] if fold else [0, 45, 90, 135, 180])
    a2.set_xlabel(x_name, color=INK2, fontsize=9)
    a1.set_title(
        f"Chain-stretch spectra per case ({len(good)} cases)",
        color=INK,
        fontsize=9,
        loc="left",
    )
    a1.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    path = os.path.join(folder, "figures", f"fourier_peaks_summary{tag}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


# --- 1D spectrum across the structures, against a fitted background ---------------------
def across_direction(fields, lattices):
    """Unit vector of strongest variation: the eigenvector of the frame-averaged <k k^T>
    with the larger eigenvalue (the 'across' axis of structure_kr.gradient_lengths)."""
    from structure_kr import gradient_moments

    m = gradient_moments(fields, lattices)
    return np.linalg.eigh(np.array([[m[0], m[1]], [m[1], m[2]]]))[1][:, 1]


def inject_stripe(fields, lattices, e, spacing, amp):
    """Add a stripe varying along e with the given physical spacing and rms = amp x the
    frame's rms, using the lattice mode nearest to k = 2 pi e / spacing in each frame.
    For calibrating what the across spectrum can detect."""
    N = fields.shape[-1]
    xi = np.arange(N) / N
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    out = fields.copy()
    for f, A in enumerate(lattices):
        m = np.round(A.T @ (2 * np.pi * np.asarray(e) / spacing) / (2 * np.pi))
        if np.abs(m).max() > N // 3:
            raise SystemExit(f"stripe spacing {spacing} is not resolved on this grid")
        out[f] += (
            amp
            * fields[f].std()
            * np.sqrt(2)
            * np.cos(2 * np.pi * (m[0] * X1 + m[1] * X2))
        )
    return out


def across_spectrum(fields, lattices, e, k_edges, blocks=5):
    """Spectrum of the field's cut along e: each frame's mode powers (normalized to unit
    variance) binned by |e.k|, as variance per unit ln|e.k|. Returns the block means
    (frames split into `blocks` contiguous blocks), shape (blocks, bins)."""
    dens = []
    for fld, A in zip(fields, lattices):
        kx, ky, P = frame_power(fld, A)
        if P.sum() <= 0:
            continue
        q = np.abs(e[0] * kx + e[1] * ky)
        dens.append(np.histogram(q, k_edges, weights=P / P.sum())[0])
    dens = np.array(dens) / np.log(k_edges[1:] / k_edges[:-1])
    return np.array([b.mean(axis=0) for b in np.array_split(dens, blocks)])


def smooth_background(k_edges, E, factor=2.5, gap=1):
    """Background of a spectrum per ln k, without an assumed global shape (the measured
    spectra are roughly flat per ln k and then fall over many decades as a curved,
    power-law-like tail, which a fitted plateau-times-Gaussian form misses by orders of
    magnitude). For each bin, ln E is fitted by a quadratic in ln k (a line where only two
    points are available) on the bins within a factor `factor` in k on either side,
    leaving out the bin itself and `gap` bins next to it, and evaluated at the bin. A
    feature up to 2 gap + 1 bins wide therefore does not raise its own background, while
    slopes and curvature of the band are followed. NaN unless there are bins on both sides
    (at the ends the fit would be an extrapolation). A very strong feature can lower the
    background of bins up to `factor` away, so judge weak detections next to a strong one
    with care."""
    u = np.log(np.sqrt(k_edges[:-1] * k_edges[1:]))
    h = max(gap + 2, int(round(np.log(factor) / np.log(k_edges[1] / k_edges[0]))))
    B = np.full(len(E), np.nan)
    for i in range(len(E)):
        j = np.array(
            [j for j in range(i - h, i + h + 1) if abs(j - i) > gap and 0 <= j < len(E)]
        )
        j = j[E[j] > 0] if len(j) else j
        if np.any(j < i) and np.any(j > i):  # interpolate, never extrapolate
            c = np.polyfit(u[j] - u[i], np.log(E[j]), 2 if len(j) >= 4 else 1)
            B[i] = np.exp(c[-1])
    return B


def background_peaks(k_edges, E, sig, B, z_min=4.0, min_excess=0.2, floor=1e-3):
    """Runs of adjacent bins standing above the background by more than z_min sigma and by
    more than a fraction min_excess, among bins holding at least `floor` x the largest
    density (the far diffusive tail carries no variance, and the smooth background is not
    meant to follow it to the precision its tiny errors would demand). z_min = 4 because the
    lowest bins hold few box modes, whose scatter is large and far from Gaussian: on
    synthetic bands without any feature, z_min = 3 gave false box-scale peaks. Each gives its wavelength 2 pi / k (k: excess-weighted
    geometric mean), the largest E / B, the largest z, and the share of the variance in
    the excess over the background."""
    kc = np.sqrt(k_edges[:-1] * k_edges[1:])
    dl = np.log(k_edges[1:] / k_edges[:-1])
    z = (E - B) / np.where(sig > 0, sig, np.inf)
    hit = (z > z_min) & (E > (1 + min_excess) * B) & (E >= floor * E.max())
    hit &= np.isfinite(B)
    lab, n = ndimage.label(hit)
    peaks = []
    for r in range(1, n + 1):
        sel = lab == r
        x = (E - B)[sel] * dl[sel]
        k_star = float(np.exp(np.sum(x * np.log(kc[sel])) / x.sum()))
        peaks.append(
            dict(
                k=k_star,
                wavelength=2 * np.pi / k_star,
                ratio=float((E / B)[sel].max()),
                z=float(z[sel].max()),
                excess_fraction=float(x.sum() / np.sum(E * dl)),
            )
        )
    return sorted(peaks, key=lambda q: -q["excess_fraction"])


def across_analysis(runs, blocks=5, n_k=40, inject=None):
    """Across spectrum pooled over independent runs (a list of (fields, lattices) pairs,
    e.g. one per seed; each gives `blocks` block means), its fitted background and peaks.
    inject = (amp, spacing) adds a calibration stripe to every run first."""
    e = across_direction(
        np.concatenate([f for f, _ in runs]), np.concatenate([A for _, A in runs])
    )
    if inject:
        runs = [(inject_stripe(f, A, e, inject[1], inject[0]), A) for f, A in runs]
    N = runs[0][0].shape[-1]
    q_max = max(
        np.abs(e[0] * kx + e[1] * ky).max()
        for _, lats in runs
        for kx, ky, _ in (mode_wavevectors(A, N) for A in lats)
    )
    L = math.sqrt(abs(np.linalg.det(runs[0][1][0])))
    k_edges = np.geomspace(np.pi / L, q_max, n_k + 1)
    blk = np.concatenate([across_spectrum(f, A, e, k_edges, blocks) for f, A in runs])
    E = blk.mean(axis=0)
    sig = blk.std(axis=0, ddof=1) / np.sqrt(len(blk))
    B = smooth_background(k_edges, E)
    return dict(
        e=e,
        theta_across_deg=round(math.degrees(math.atan2(e[1], e[0])), 9) % 180,
        k_edges=k_edges,
        E=E,
        sig=sig,
        B=B,
        n_blocks=len(blk),
        inject=inject,
        peaks=background_peaks(k_edges, E, sig, B),
    )


def across_case(
    folders, c, field="txx", window=(0.5, 1.0), blocks=5, n_k=40, inject=None, skip=None
):
    """across_analysis for one case of several sweep folders that differ only in the seed."""
    from structure_kr import field_frames

    runs, params = [], []
    for folder in folders:
        config, rows, _ = load_sweep(folder)
        p = case_params(config, rows[c])
        fr = load_frames(folder, c)
        sel, e0, e1 = window_mask(fr["strain"], window, skip)
        if sel.sum() < blocks:
            raise SystemExit(
                f"{folder} case {c}: fewer than {blocks} frames in {e0:g}-{e1:g}"
            )
        runs.append((field_frames(fr, p, field)[sel], fr["lattice"][sel]))
        params.append(p)
    if any(p != params[0] for p in params):
        raise SystemExit(f"case {c} has different parameters in these folders")
    r = across_analysis(runs, blocks, n_k, inject)
    r.update(case=c, p=params[0], field=field, folders=list(folders))
    return r


def across_figure(r):
    kc = np.sqrt(r["k_edges"][:-1] * r["k_edges"][1:])
    E, sig, B = r["E"], r["sig"], r["B"]
    ok = E > 0
    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.4),
        dpi=130,
        sharex=True,
        gridspec_kw=dict(height_ratios=[1.4, 1]),
    )
    fig.patch.set_facecolor(SURFACE)
    for a in (a1, a2):
        style_plot(a)
    a1.fill_between(
        kc[ok],
        (E - 2 * sig)[ok].clip(E[ok].min() * 1e-3),
        (E + 2 * sig)[ok],
        color=SERIES[0],
        alpha=0.2,
        lw=0,
    )
    a1.loglog(
        kc[ok], E[ok], "-o", ms=3, color=SERIES[0], lw=1.6, label="spectrum (band: ±2σ)"
    )
    a1.loglog(kc[ok], B[ok], "--", color=INK2, lw=1.3, label="background (local fit)")
    a2.fill_between(
        kc[ok],
        ((E - 2 * sig) / B)[ok],
        ((E + 2 * sig) / B)[ok],
        color=SERIES[0],
        alpha=0.2,
        lw=0,
    )
    a2.semilogx(kc[ok], (E / B)[ok], "-o", ms=3, color=SERIES[0], lw=1.6)
    a2.axhline(1, color=INK2, lw=0.8)
    for i, pk in enumerate(r["peaks"]):
        for a in (a1, a2):
            a.axvline(pk["k"], color=SERIES[3], lw=1.2, ls=":")
        a2.annotate(
            f"λ {pk['wavelength']:.3g}, ×{pk['ratio']:.2f}, z {pk['z']:.0f}",
            (pk["k"], (E / B)[ok].max()),
            xytext=(4, -10 - 12 * i),
            textcoords="offset points",
            fontsize=7.5,
            color=INK,
        )
    lD = math.sqrt(r["p"].D / r["p"].eps_dot)
    for a in (a1, a2):
        a.axvline(1 / lD, color=MUTED, lw=0.8, ls="-.")
    a1.set_ylabel("variance per ln |e·k|", color=INK2, fontsize=8.5)
    a2.set_ylabel("spectrum / background", color=INK2, fontsize=8.5)
    a2.set_xlabel(
        f"|e·k|, e at {r['theta_across_deg']:.0f}° from x (dash-dot: 1/sqrt(D/eps_dot))",
        color=INK2,
        fontsize=8.5,
    )
    a1.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    inj = (
        f" · injected stripe rms ×{r['inject'][0]:g}, spacing {r['inject'][1]:g}"
        if r["inject"]
        else ""
    )
    a1.set_title(
        f"case {r['case']} · {r['field']} · {len(r['folders'])} folder(s), {r['n_blocks']} blocks{inj}",
        color=INK,
        fontsize=8.5,
        loc="left",
    )
    fig.tight_layout()
    out = os.path.join(r["folders"][0], "figures")
    os.makedirs(out, exist_ok=True)
    tag = f"_inject{r['inject'][0]:g}_{r['inject'][1]:g}" if r["inject"] else ""
    path = os.path.join(
        out, f"fourier_across_{r['field']}_case_{r['case']:04d}{tag}.png"
    )
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- (kx, ky) map averaged over the window ------------------------------------------------
def kmap_case(
    folders,
    c,
    field="txx",
    window=(0.5, 1.0),
    skip=None,
    fold=False,
    kmax=None,
    bin_=1.0,
):
    """Mode power on a fixed Cartesian (kx, ky) grid, averaged over the window of every
    folder given (same case, other seeds), each frame normalized to unit variance. Bins
    are bin_ x 2 pi / L wide, i.e. about bin_^2 modes each (the k-space area per mode);
    the lattice modes move through the fixed bins as the cell deforms, so the average
    fills in what a single frame leaves as scattered points. fold: (kx, ky) -> (|kx|, |ky|),
    adding the mirror images; without it the map is already symmetric under k -> -k."""
    from structure_kr import field_frames, gradient_lengths

    fields, lattices, n_per, params = [], [], [], []
    for folder in folders:
        config, rows, _ = load_sweep(folder)
        p = case_params(config, rows[c])
        fr = load_frames(folder, c)
        sel, e0, e1 = window_mask(fr["strain"], window, skip)
        if not sel.any():
            raise SystemExit(f"{folder} case {c}: no frames in strain {e0:g}-{e1:g}")
        fields.append(field_frames(fr, p, field)[sel])
        lattices.append(fr["lattice"][sel])
        n_per.append(int(sel.sum()))
        params.append(p)
    if any(q != params[0] for q in params):
        raise SystemExit(f"case {c} has different parameters in these folders")
    fields, lattices = np.concatenate(fields), np.concatenate(lattices)
    cut = gradient_lengths(fields, lattices)["cutoff_across"]
    kmax = kmax or 3.2 / cut
    dk = bin_ * 2 * np.pi / math.sqrt(abs(np.linalg.det(lattices[0])))
    n = int(math.ceil(kmax / dk))
    edges = dk * np.arange(0, n + 1) if fold else dk * (np.arange(-n, n + 2) - 0.5)
    H = np.zeros((len(edges) - 1,) * 2)
    for fld, A in zip(fields, lattices):
        kx, ky, P = frame_power(fld, A)
        if P.sum() <= 0:
            continue
        if fold:
            kx, ky = np.abs(kx), np.abs(ky)
        H += np.histogram2d(kx, ky, [edges, edges], weights=P / P.sum())[0]
    return dict(
        case=c,
        p=params[0],
        field=field,
        folders=list(folders),
        n_frames=n_per,
        window=window,
        skip=skip,
        fold=fold,
        cutoff=cut,
        kmax=kmax,
        edges=edges,
        density=H / len(fields) / dk**2,  # fraction of variance per unit k-area
        e_across=across_direction(fields, lattices),
    )


def kmap_figure(r):
    edges, D, kmax, cut = r["edges"], r["density"], r["kmax"], r["cutoff"]
    fig, ax = plt.subplots(figsize=(5.6 if r["fold"] else 6.2, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    vals = D[D > 0]
    m = ax.pcolormesh(
        edges,
        edges,
        np.where(D > 0, D, np.nan).T,
        cmap=SEQUENTIAL,
        norm=LogNorm(vals.max() * 1e-6, vals.max()),
        shading="flat",
        rasterized=True,
    )
    t = np.linspace(0, np.pi / 2 if r["fold"] else 2 * np.pi, 200)
    ax.plot(np.cos(t) / cut, np.sin(t) / cut, "--", color=INK, lw=1)
    ax.text(0.71 / cut, 0.71 / cut, "  |k| = 1/cutoff", color=INK, fontsize=8)
    e = np.abs(r["e_across"]) if r["fold"] else r["e_across"]
    a = 0 if r["fold"] else -1
    ax.plot(
        [a * e[0] * kmax, e[0] * kmax],
        [a * e[1] * kmax, e[1] * kmax],
        ":",
        color=INK2,
        lw=1,
    )
    lo = 0 if r["fold"] else -kmax
    ax.set_xlim(lo, kmax)
    ax.set_ylim(lo, kmax)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("|k_x|" if r["fold"] else "k_x", color=INK2, fontsize=8.5)
    ax.set_ylabel("|k_y|" if r["fold"] else "k_y", color=INK2, fontsize=8.5)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor(GRID)
    cb.set_label("fraction of variance per unit k-area", color=INK2, fontsize=8)
    span = window_text(r["window"], r["skip"])
    ax.set_title(
        "\n".join(
            wrap(
                f"case {r['case']} · {r['field']} · mean of {sum(r['n_frames'])} frames "
                f"({len(r['folders'])} folder(s)), {span}"
                + (" · folded (|k_x|, |k_y|)" if r["fold"] else "")
                + " · dotted: across direction",
                70,
            )
        ),
        color=INK,
        fontsize=8.5,
        loc="left",
    )
    fig.tight_layout()
    out = os.path.join(r["folders"][0], "figures")
    os.makedirs(out, exist_ok=True)
    tag = "_folded" if r["fold"] else ""
    path = os.path.join(out, f"fourier_kmap_{r['field']}_case_{r['case']:04d}{tag}.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("folder")
        sp.add_argument("--n-k", type=int, default=48, help="log-spaced |k| bins")
        sp.add_argument(
            "--n-theta", type=int, default=36, help="angle bins over 0-180 deg"
        )
        sp.add_argument(
            "--percentile",
            type=float,
            default=99.0,
            help="peak threshold, percentile of the nonzero bins",
        )

    def windowed(sp, fold=True):
        sp.add_argument(
            "--window",
            default="0.5:1",
            help="strain window as fractions of each run's final strain (default 0.5:1)",
        )
        sp.add_argument(
            "--skip-strain",
            type=float,
            help="discard frames before this Hencky strain (replaces the window start)",
        )
        if fold:
            sp.add_argument(
                "--fold",
                action="store_true",
                help="mirror theta onto 0-90 deg (theta and 180 - theta added)",
            )

    c = sub.add_parser("compare", help="averaged polar spectra for up to 8 cases")
    common(c)
    c.add_argument("--cases", type=int, nargs="+", required=True)
    windowed(c)
    p = sub.add_parser("peaks", help="peak table and summary across many cases")
    common(p)
    p.add_argument("--cases", nargs="+", default=["all"])
    windowed(p)
    p.add_argument(
        "--absolute",
        action="store_true",
        help="average absolute spectra (larger-amplitude frames weigh more); "
        "default: each frame normalized to unit variance, as in compare",
    )
    p.add_argument(
        "--x", dest="x_name", help="swept parameter on the x axis (default: first)"
    )
    p.add_argument(
        "--hue", dest="hue_name", help="swept parameter for colour (<= 8 values)"
    )
    p.add_argument("--workers", type=int, default=4)
    s = sub.add_parser(
        "spectrogram", help="radial spectrum against strain for one case"
    )
    common(s)
    s.add_argument("--case", type=int, required=True)
    x = sub.add_parser(
        "across", help="1D spectrum across the structures against a fitted background"
    )
    x.add_argument("folders", nargs="+", help="sweep folders (same sweep, other seeds)")
    x.add_argument("--case", type=int, required=True)
    x.add_argument(
        "--field", choices=("txx", "tyy", "txy", "N1", "stretch"), default="stretch"
    )
    windowed(x, fold=False)
    x.add_argument("--blocks", type=int, default=5, help="frame blocks per folder")
    x.add_argument("--n-k", type=int, default=40, help="log-spaced |e.k| bins")
    x.add_argument(
        "--inject",
        help="A,S: add a stripe of rms A x field rms and spacing S along e first",
    )
    km = sub.add_parser("kmap", help="(kx, ky) power map averaged over the window")
    km.add_argument(
        "folders", nargs="+", help="sweep folders (same sweep, other seeds)"
    )
    km.add_argument("--case", type=int, required=True)
    km.add_argument(
        "--field", choices=("txx", "tyy", "txy", "N1", "stretch"), default="stretch"
    )
    windowed(km)
    km.add_argument(
        "--kmax", type=float, help="plot range (default 3.2 / cutoff length)"
    )
    km.add_argument(
        "--bin",
        type=float,
        default=1.0,
        help="bin width in units of 2 pi / L (default 1)",
    )
    args = ap.parse_args(argv)
    if args.cmd == "kmap":
        r = kmap_case(
            args.folders,
            args.case,
            args.field,
            parse_window(args.window),
            args.skip_strain,
            args.fold,
            args.kmax,
            args.bin,
        )
        print(
            f"case {r['case']}, {r['field']}: {sum(r['n_frames'])} frames, cutoff {r['cutoff']:.4g}"
        )
        print("wrote", kmap_figure(r))
        return
    if args.cmd == "across":
        inj = tuple(float(v) for v in args.inject.split(",")) if args.inject else None
        r = across_case(
            args.folders,
            args.case,
            args.field,
            parse_window(args.window),
            args.blocks,
            args.n_k,
            inj,
            args.skip_strain,
        )
        print(
            f"case {r['case']}, {r['field']}: across direction {r['theta_across_deg']:.1f} deg "
            f"from x; {r['n_blocks']} blocks"
        )
        for pk in r["peaks"] or [None]:
            print(
                "  no bins above the background by > 4 sigma and > 20%"
                if pk is None
                else f"  peak: wavelength {pk['wavelength']:.4g}, x{pk['ratio']:.2f} the "
                f"background, z {pk['z']:.1f}, {100 * pk['excess_fraction']:.1f}% of the variance"
            )
        print("wrote", across_figure(r))
        return
    if args.cmd == "compare":
        compare(
            args.folder,
            args.cases,
            parse_window(args.window),
            True,
            args.n_k,
            args.n_theta,
            args.percentile,
            args.skip_strain,
            args.fold,
        )
    elif args.cmd == "peaks":
        peaks_table(
            args.folder,
            args.cases,
            parse_window(args.window),
            not args.absolute,
            args.n_k,
            args.n_theta,
            args.percentile,
            args.workers,
            args.x_name,
            args.hue_name,
            args.skip_strain,
            args.fold,
        )
    else:
        spectrogram(args.folder, args.case, args.n_k, args.n_theta)


if __name__ == "__main__":
    main()
