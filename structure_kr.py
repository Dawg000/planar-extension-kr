"""Length scales of sustained structures, and a test for locking to the KR cell cycle.

  python structure_kr.py case SWEEP_DIR --case 12 [--field txx] [--window 0.5:1]
      One case: printed measures, plus a figure with the time-averaged autocorrelation map,
      its cuts along and across the extension axis, and the amplitude folded on the cell cycle.

  python structure_kr.py sweep SWEEP_DIR [--cases all] [--field txx] [--x D] [--workers 4]
      Many cases in parallel: structure.csv and a summary figure against one swept parameter.

  python structure_kr.py phases SWEEP_DIR [SWEEP_DIR ...] --case 5 [--field txx] [--bins 5]
      Strongest repeat of the autocorrelation separately for each phase of the KR period, per
      folder (e.g. per seed) and pooled: a repeat set by the cell geometry appears only at
      some phases, with its angle's sign flipping half a period apart.

Fields (--field): txx, tyy, txy (stress fluctuation delta tau / G), N1 (delta(tau_xx - tau_yy)/G),
stretch (delta lambda). In the runs examined so far the structure is in the orientation
(stress) field, while the chain stretch follows it with a much smaller amplitude.

Length scales: for each saved frame in the strain window, the autocorrelation C(delta xi) of
the field is computed exactly on the lattice (inverse FFT of |F|^2, C(0) = 1) and mapped to
physical separations r = A(t) delta xi (with the neighbouring periodic images), then averaged
on a fixed physical grid. Along the extension axis (x) and across it (y) the cuts give:
  half     separation where C first drops to 0.5 (correlation half-width)
  zero     first zero crossing
  min      first minimum (value and separation): the anticorrelated neighbour
  repeat   the next maximum after it (value and separation): the spacing of repeats. A
           repeat value near 0 means no real periodicity.
Searches stop at half the cell side, beyond which periodic images of the cell contaminate C.
The axis cuts can miss a repeat that lies along a tilted direction, so the strongest local
maximum of C anywhere outside its central lobe is also reported (peak_r, peak_theta_deg,
peak_C), with peak_image_distance, its distance to the nearest periodic image of the cell
in any frame: a repeat on a lattice point may be the cell's own periodicity.

Gradient lengths (no binning): the half-width, zero and repeat above weigh the whole
spectrum, and for a spectrum flat per ln k between the box (k_a = 2 pi / L) and the
diffusion cutoff (k_b) the half-width sits near 1 / sqrt(k_a k_b): it moves with the box.
Moments of each frame's power spectrum over the physical wavevectors are exact and depend
on the high-k end instead. With M2 = <k k^T> (power-weighted, each frame normalized, then
averaged over frames), the lobe of C near r = 0 is C = 1 - r.M2.r / 2, so the eigenvectors
of M2 are the lobe's principal axes:
  lobe_theta_deg   angle of the lobe's long axis from the extension axis, in (-90, 90];
                   its sign is the chirality
  taylor_*         1 / sqrt(<(e.k)^2>): curvature width of the lobe along unit vector e
                   (still grows like sqrt(ln(k_b / k_a)) with the box)
  cutoff_*         sqrt(<(e.k)^2> / <(e.k)^4>): set by the high-k cutoff only. For a
                   Gaussian C of width l along e it is l / sqrt(3); for a band flat per ln k
                   ending in exp(-k^2 l^2) it is 2 l / sqrt(3), whatever the box size.
'across' is the direction of strongest variation (normal to the structures), 'along' the
lobe's long axis. The tilt only means something when the lobe is elongated (taylor_along
clearly above taylor_across); for a nearly isotropic lobe its value is arbitrary.

Cell-cycle test: for the golden and silver lattices the cell at half a period is an exact
mirror image (y -> -y) of the starting cell, so anything set by the cell geometry (grid
resolution, available modes) repeats every eps_p / 2 (0.481 strain for golden, 0.881 for
silver; see kr_solver.kr_lattice, field 'cycle'). The rms amplitude of the field over the window is
folded on that phase. 'cell_locked_fraction' is the fraction of the amplitude's variance
explained by the phase-averaged curve; for an amplitude unrelated to the cycle it is about
(phase bins - 1) / (samples - 1), reported as 'cell_locked_chance'. 'cell_modulation' is the
peak-to-peak of the phase-averaged curve relative to the mean amplitude. A fraction well
above chance means the cell cycle is shaping the dynamics: check whether it shrinks with N
(numerical) or with a larger cell (finite-size), see the module notes in the reply.
Pattern decorrelation: the field is evaluated at fixed physical points in every frame
(Fourier series, exact) and correlated with itself at later strains; 'decorrelation_strain'
is the lag where that falls below 1/e.
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
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage

from fourier_kr import (
    frame_power,
    load_frames,
    mode_wavevectors,
    parse_window,
    stretch_field,
)
from kr_solver import EPS_P, kr_lattice
from viz_common import (
    DIVERGING,
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

FIELDS = ("txx", "tyy", "txy", "N1", "stretch")
HALF = EPS_P / 2


# --- fields -------------------------------------------------------------------------------
def field_frames(frames, p, name):
    """The chosen fluctuation field, one lattice-grid array per frame (zero mean)."""
    dC = frames["dC"]
    if name == "stretch":
        lam = stretch_field(frames, p)
        f = lam - lam.mean(axis=(1, 2), keepdims=True)
    elif name == "N1":
        f = (dC[:, 0] - dC[:, 2]).astype(float)
    else:
        f = dC[:, {"txx": 0, "txy": 1, "tyy": 2}[name]].astype(float)
    return f - f.mean(axis=(1, 2), keepdims=True)


# --- autocorrelation ----------------------------------------------------------------------
def physical_acf(fields, lattices, half_width=0.6, h=0.005):
    """Autocorrelation averaged over frames on a fixed physical grid of separations."""
    N = fields.shape[-1]
    edges = np.arange(-half_width, half_width + h / 2, h)
    acc = np.zeros((len(edges) - 1,) * 2)
    cnt = np.zeros_like(acc)
    idx = np.fft.fftfreq(N, 1.0 / N) / N
    X1, X2 = np.meshgrid(idx, idx, indexing="ij")
    for fld, A in zip(fields, lattices):
        var = fld.var()
        if var <= 0:
            continue
        C = np.fft.ifft2(np.abs(np.fft.fft2(fld)) ** 2).real / (N**2 * var)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                rx = (A[0, 0] * (X1 + a) + A[0, 1] * (X2 + b)).ravel()
                ry = (A[1, 0] * (X1 + a) + A[1, 1] * (X2 + b)).ravel()
                acc += np.histogram2d(rx, ry, [edges, edges], weights=C.ravel())[0]
                cnt += np.histogram2d(rx, ry, [edges, edges])[0]
    return 0.5 * (edges[1:] + edges[:-1]), np.where(
        cnt > 0, acc / np.maximum(cnt, 1), np.nan
    )


def axis_cut(centres, C, axis):
    """C along r >= 0 on one axis (0: x, 1: y), averaging the two bins either side of 0."""
    i0 = np.searchsorted(centres, 0)
    cut = (
        C[:, i0 - 1 : i0 + 1].mean(axis=1)
        if axis == 0
        else C[i0 - 1 : i0 + 1, :].mean(axis=0)
    )
    r, v = centres[i0:], cut[i0:]
    ok = np.isfinite(v)
    return r[ok], v[ok]


def cut_measures(r, v, r_max, smooth=5, prominence=0.05):
    """half / zero / min / repeat measures of one cut, searched within r_max. The cut is
    smoothed with a short running mean first; the minimum is searched after the first zero
    crossing (the anticorrelated neighbour), and a repeat must be an interior maximum at
    least `prominence` above that minimum."""
    sel = r <= r_max
    r, v = r[sel], v[sel]
    nan = float("nan")
    out = dict(half=nan, zero=nan, min_r=nan, min_C=nan, repeat_r=nan, repeat_C=nan)
    if len(v) < smooth + 3:
        return out
    k = np.ones(smooth) / smooth
    vs = np.convolve(
        np.pad(v, (smooth // 2, smooth // 2), mode="edge"), k, mode="valid"
    )
    if np.any(v < 0.5):
        out["half"] = float(r[np.argmax(v < 0.5)])
    if not np.any(vs < 0):
        return out
    iz = int(np.argmax(vs < 0))
    out["zero"] = float(r[iz])
    i_min = iz + int(np.argmin(vs[iz:]))
    out["min_r"], out["min_C"] = float(r[i_min]), float(vs[i_min])
    if i_min < len(vs) - 2:
        i_rep = i_min + int(np.argmax(vs[i_min:]))
        if i_min < i_rep < len(vs) - 1 and vs[i_rep] - vs[i_min] >= prominence:
            out["repeat_r"], out["repeat_C"] = float(r[i_rep]), float(vs[i_rep])
    return out


def secondary_peak(centres, C, lattices, r_max, smooth=5, core=0.5):
    """Strongest local maximum of the (smoothed) autocorrelation outside its central lobe,
    within |r| <= r_max: the separation vector to the nearest repeated feature, in any
    direction. Also returns the distance from that vector to the nearest periodic image of
    the cell (any frame of the window): a repeat sitting on a lattice point may be the
    cell's own periodicity rather than a physical spacing."""
    nan = float("nan")
    out = dict(peak_r=nan, peak_theta_deg=nan, peak_C=nan, peak_image_distance=nan)
    w = np.isfinite(C).astype(float)  # normalized smoothing: empty bins carry no weight
    num = ndimage.uniform_filter(
        np.nan_to_num(C, nan=0.0), size=smooth, mode="constant"
    )
    den = ndimage.uniform_filter(w, size=smooth, mode="constant")
    Cs = np.where(den > 0.2, num / np.maximum(den, 1e-12), -np.inf)
    RX, RY = np.meshgrid(centres, centres, indexing="ij")
    R = np.hypot(RX, RY)
    lab, _ = ndimage.label(np.isfinite(Cs) & (Cs > core))
    i0 = np.unravel_index(np.argmin(R), R.shape)
    central = (lab == lab[i0]) if lab[i0] else np.zeros_like(Cs, bool)
    is_max = Cs == ndimage.maximum_filter(Cs, size=2 * smooth + 1, mode="constant")
    cand = is_max & ~central & (R <= r_max) & (R > 0) & np.isfinite(Cs) & (Cs > 0)
    if not cand.any():
        return out
    j = np.argmax(np.where(cand, Cs, -np.inf))
    rx, ry = RX.flat[j], RY.flat[j]
    n = np.array(
        [(a, b) for a in range(-3, 4) for b in range(-3, 4) if (a, b) != (0, 0)]
    ).T
    d = min(np.hypot(*(A @ n - np.array([[rx], [ry]]))).min() for A in lattices)
    out.update(
        peak_r=float(np.hypot(rx, ry)),
        peak_theta_deg=round(math.degrees(math.atan2(ry, rx)), 9) % 180,
        peak_C=float(Cs.flat[j]),
        peak_image_distance=float(d),
    )
    return out


# --- gradient lengths ---------------------------------------------------------------------
def gradient_moments(fields, lattices):
    """Power-weighted moments of the physical wavevector, each frame normalized to unit
    variance and then averaged over frames: second order (xx, xy, yy) and fourth order
    (xxxx, xxxy, xxyy, xyyy, yyyy). Exact sums over the dealiased modes, no binning."""
    acc, n = np.zeros(8), 0
    for fld, A in zip(fields, lattices):
        kx, ky, P = frame_power(fld, A)
        tot = P.sum()
        if tot <= 0:
            continue
        w = P / tot
        acc += [
            w @ (kx * kx),
            w @ (kx * ky),
            w @ (ky * ky),
            w @ kx**4,
            w @ (kx**3 * ky),
            w @ (kx**2 * ky**2),
            w @ (kx * ky**3),
            w @ ky**4,
        ]
        n += 1
    return acc / max(n, 1)


def gradient_lengths(fields, lattices):
    """Lobe tilt and the Taylor and cutoff lengths along the lobe's principal axes (see the
    module docstring)."""
    m = gradient_moments(fields, lattices)
    M2 = np.array([[m[0], m[1]], [m[1], m[2]]])
    _, V = np.linalg.eigh(M2)  # ascending: V[:, 0] long axis, V[:, 1] across

    def lengths(e):
        c, s = e
        m2 = e @ M2 @ e
        m4 = (
            c**4 * m[3]
            + 4 * c**3 * s * m[4]
            + 6 * c**2 * s**2 * m[5]
            + 4 * c * s**3 * m[6]
            + s**4 * m[7]
        )
        return 1 / math.sqrt(m2), math.sqrt(m2 / m4)

    ta, ca = lengths(V[:, 1])
    tl, cl = lengths(V[:, 0])
    th = math.degrees(math.atan2(V[1, 0], V[0, 0]))
    th = (th + 90) % 180 - 90  # long axis is a line: fold into [-90, 90)
    return dict(
        lobe_theta_deg=90.0 if th == -90 else th,
        taylor_across=ta,
        taylor_along=tl,
        cutoff_across=ca,
        cutoff_along=cl,
    )


# --- time behaviour -----------------------------------------------------------------------
def cell_cycle_test(strain, amp, bins=None, cycle=HALF):
    """Fold the amplitude on the cell cycle (the strain after which the cell repeats up to a
    mirror; half a KR period for the golden lattice). Returns locked fraction, its chance
    level, peak-to-peak modulation (relative to the mean), and the phase curve."""
    phase = np.mod(strain / cycle + 1e-6, 1.0)
    if bins is None:  # one bin per sample slot within the cycle
        bins = max(3, int(round(cycle / np.median(np.diff(strain)))))
    b = np.minimum((phase * bins).astype(int), bins - 1)
    mean_b = np.array(
        [amp[b == i].mean() if np.any(b == i) else np.nan for i in range(bins)]
    )
    fitted = mean_b[b]
    tot = np.sum((amp - amp.mean()) ** 2)
    frac = float(1 - np.sum((amp - fitted) ** 2) / tot) if tot > 0 else float("nan")
    return dict(
        cell_locked_fraction=frac,
        cell_locked_chance=(bins - 1) / max(1, len(amp) - 1),
        cell_modulation=float((np.nanmax(mean_b) - np.nanmin(mean_b)) / amp.mean()),
        phase_curve=mean_b,
        bins=bins,
    )


def pattern_decorrelation(fields, lattices, strain, n_pts=32, eps_p=EPS_P):
    """Lag (in strain) at which the pattern at fixed physical points decorrelates to 1/e,
    plus the mean correlation at a lag of one KR period."""
    N = fields.shape[-1]
    g = np.linspace(-0.5, 0.5, n_pts, endpoint=False)
    PX, PY = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([PX.ravel(), PY.ravel()])
    Z = []
    for fld, A in zip(fields, lattices):
        kx, ky, keep = mode_wavevectors(A, N)
        F = (np.fft.fft2(fld) / N**2)[keep]
        v = np.zeros(pts.shape[1])
        for s in range(0, len(F), 2000):
            v += (
                (
                    F[s : s + 2000, None]
                    * np.exp(
                        1j
                        * (
                            np.outer(kx[s : s + 2000], pts[0])
                            + np.outer(ky[s : s + 2000], pts[1])
                        )
                    )
                )
                .sum(0)
                .real
            )
        sd = v.std()
        Z.append((v - v.mean()) / sd if sd > 0 else v * 0)
    Z = np.array(Z)
    R = Z @ Z.T / Z.shape[1]
    de = np.median(np.diff(strain))
    lags, corr = [], []
    for lag in range(1, len(strain)):
        vals = np.diag(R, lag)
        if len(vals) < 3:
            break
        lags.append(lag * de)
        corr.append(float(vals.mean()))
    lags, corr = np.array(lags), np.array(corr)
    below = np.where(corr < 1 / math.e)[0]
    dec = float(lags[below[0]]) if len(below) else float("nan")
    at_period = (
        float(np.interp(eps_p, lags, corr))
        if len(lags) and lags[-1] >= eps_p
        else float("nan")
    )
    return dec, at_period, lags, corr


# --- one case -----------------------------------------------------------------------------
def analyse(folder, config, row, field="txx", window=(0.5, 1.0), with_pattern=True):
    c = int(row["case"])
    p = case_params(config, row)
    fr = load_frames(folder, c)
    e = fr["strain"]
    fld = field_frames(fr, p, field)
    sel = (e >= window[0] * e[-1] - 1e-9) & (e <= window[1] * e[-1] + 1e-9)
    if sel.sum() < 3:
        raise SystemExit(f"case {c}: fewer than 3 frames in the strain window")
    L = math.sqrt(abs(np.linalg.det(fr["lattice"][0])))
    N = fld.shape[-1]  # bins no finer than the grid spacing, so few are empty
    centres, C = physical_acf(
        fld[sel], fr["lattice"][sel], half_width=0.6 * L, h=max(0.005, 0.8 / N) * L
    )
    lat = kr_lattice(p.kr_lattice)
    res = dict(
        case=c,
        p=p,
        field=field,
        centres=centres,
        C=C,
        L=L,
        strain=e,
        sel=sel,
        cycle=lat.cycle,
    )
    for axis, tag in ((0, "x"), (1, "y")):
        r, v = axis_cut(centres, C, axis)
        res[f"cut_{tag}"] = (r, v)
        res.update(
            {f"{tag}_{k}": val for k, val in cut_measures(r, v, 0.5 * L).items()}
        )
    res.update(secondary_peak(centres, C, fr["lattice"][sel], 0.5 * L))
    res.update(gradient_lengths(fld[sel], fr["lattice"][sel]))
    amp = fld.std(axis=(1, 2))
    res["amp"] = amp
    res["amp_mean"] = float(amp[sel].mean())
    cc = cell_cycle_test(e[sel], amp[sel], cycle=lat.cycle)
    res.update(cc)
    if with_pattern:
        dec, at_p, lags, corr = pattern_decorrelation(
            fld[sel], fr["lattice"][sel], e[sel], eps_p=lat.eps_p
        )
        res.update(
            decorrelation_strain=dec, pattern_corr_one_period=at_p, lags=lags, corr=corr
        )
    lengths = dict(
        sqrt_D_tau=math.sqrt(p.D * p.lam), sqrt_D_over_rate=math.sqrt(p.D / p.eps_dot)
    )
    res.update(lengths)
    return res


def summary_lines(r):
    f = lambda v, d=3: "–" if not np.isfinite(v) else f"{v:.{d}g}"
    lD = r["sqrt_D_over_rate"]
    rel = lambda v: f" ({f(v / lD)} sqrt(D/eps_dot))" if lD > 0 else ""
    lines = [
        f"across the extension axis (y): half-width {f(r['y_half'])}, first zero {f(r['y_zero'])}, "
        f"minimum {f(r['y_min_C'], 2)} at {f(r['y_min_r'])}, repeat {f(r['y_repeat_C'], 2)} at "
        f"{f(r['y_repeat_r'])}",
        f"along the extension axis (x): half-width {f(r['x_half'])}, first zero {f(r['x_zero'])}, "
        f"minimum {f(r['x_min_C'], 2)} at {f(r['x_min_r'])}, repeat {f(r['x_repeat_C'], 2)} at "
        f"{f(r['x_repeat_r'])}",
        f"strongest repeat in any direction: C = {f(r['peak_C'], 2)} at |r| = {f(r['peak_r'])}, "
        f"{f(r['peak_theta_deg'], 3)}° from the extension axis; nearest periodic image of the "
        f"cell {f(r['peak_image_distance'])} away",
        f"gradient lengths: lobe long axis at {f(r['lobe_theta_deg'], 3)}°; across it cutoff "
        f"{f(r['cutoff_across'])}{rel(r['cutoff_across'])}, Taylor {f(r['taylor_across'])}; "
        f"along it cutoff {f(r['cutoff_along'])}, Taylor {f(r['taylor_along'])}",
        f"diffusion lengths: sqrt(D tau_d) = {f(r['sqrt_D_tau'])}, sqrt(D/eps_dot) = "
        f"{f(r['sqrt_D_over_rate'])}; cell side {f(r['L'])}",
        f"cell cycle: {100 * r['cell_locked_fraction']:.0f}% of the amplitude's variance follows "
        f"the {r['cycle']:.3f}-strain cell cycle (chance ~{100 * r['cell_locked_chance']:.0f}%), peak-to-peak "
        f"{100 * r['cell_modulation']:.0f}% of the mean amplitude",
    ]
    if "decorrelation_strain" in r:
        lines.append(
            f"pattern decorrelates (1/e) after strain {f(r['decorrelation_strain'])}; "
            f"correlation one KR period later {f(r['pattern_corr_one_period'], 2)}"
        )
    return lines


def case_figure(folder, config, rows, r):
    c = r["case"]
    names = list(config["sweeps"])
    fig = plt.figure(figsize=(11.0, 7.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    head = wrap(f"case {c} · {rows[c]['status']} · {param_text(rows[c], names)}", 150)
    fig.text(
        0.03, 0.985, "\n".join(head), fontsize=8.5, color=INK, va="top", linespacing=1.3
    )
    fig.text(
        0.03,
        0.985 - 0.03 * len(head),
        f"field: {r['field']} · strain window "
        f"{r['strain'][r['sel']][0]:.2f}–{r['strain'][-1]:.2f} ({int(r['sel'].sum())} frames)",
        fontsize=8,
        color=INK2,
        va="top",
    )
    top = 0.86 - 0.03 * (len(head) - 1)
    ax = fig.add_axes([0.05, 0.36, 0.36, top - 0.36])
    lim = 0.5 * r["L"]
    cen, C = r["centres"], r["C"]
    m = ax.pcolormesh(
        cen, cen, C.T, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1), shading="nearest"
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("separation along x (extension)", color=INK2, fontsize=8.5)
    ax.set_ylabel("separation along y (compression)", color=INK2, fontsize=8.5)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    if np.isfinite(r["peak_r"]):
        th = math.radians(r["peak_theta_deg"])
        for s in (1, -1):
            ax.plot(
                s * r["peak_r"] * math.cos(th),
                s * r["peak_r"] * math.sin(th),
                "o",
                ms=9,
                mfc="none",
                mec=INK,
                mew=1.4,
            )
    ax.set_title(
        "time-averaged autocorrelation (○ strongest repeat)",
        color=INK,
        fontsize=9,
        loc="left",
    )
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor(GRID)

    a2 = fig.add_axes([0.52, 0.36 + (top - 0.36) * 0.52, 0.45, (top - 0.36) * 0.48])
    style_plot(a2)
    for tag, col, lab in (
        ("y", SERIES[0], "across the axis (y)"),
        ("x", SERIES[1], "along the axis (x)"),
    ):
        rr, vv = r[f"cut_{tag}"]
        keep = rr <= lim
        a2.plot(rr[keep], vv[keep], color=col, lw=1.8, label=lab)
        if np.isfinite(r[f"{tag}_repeat_r"]):
            a2.plot(r[f"{tag}_repeat_r"], r[f"{tag}_repeat_C"], "o", color=col, ms=6)
        if np.isfinite(r[f"{tag}_min_r"]):
            a2.plot(r[f"{tag}_min_r"], r[f"{tag}_min_C"], "v", color=col, ms=6)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_ylabel("C", color=INK2, fontsize=8.5)
    a2.set_xlabel("separation r (▼ first minimum, ● repeat)", color=INK2, fontsize=8.5)
    a2.legend(frameon=False, fontsize=8, labelcolor=INK2)
    a2.set_title("cuts", color=INK, fontsize=9, loc="left")

    a3 = fig.add_axes([0.52, 0.36, 0.45, (top - 0.36) * 0.36])
    style_plot(a3)
    sel = r["sel"]
    ph = (np.arange(r["bins"]) + 0.5) / r["bins"]
    e = r["strain"][sel]
    a3.plot(
        np.mod(e / r["cycle"] + 1e-6, 1.0),
        r["amp"][sel] / r["amp"][sel].mean(),
        "o",
        ms=3.5,
        color=MUTED,
        alpha=0.7,
        label="frames",
    )
    a3.plot(
        ph,
        r["phase_curve"] / r["amp"][sel].mean(),
        "-",
        color=SERIES[3],
        lw=2,
        label="phase average",
    )
    a3.set_xlabel(
        f"phase within the cell cycle ({r['cycle']:.3f} strain)",
        color=INK2,
        fontsize=8.5,
    )
    a3.set_ylabel("amplitude / mean", color=INK2, fontsize=8.5)
    a3.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper right")

    a4 = fig.add_axes([0.05, 0.07, 0.92, 0.2])
    style_plot(a4)
    a4.semilogy(r["strain"], r["amp"], color=INK2, lw=1.4)
    a4.axvspan(e[0], e[-1], color=GRID, alpha=0.5, lw=0)
    for k in np.arange(0, r["strain"][-1] + 1e-9, r["cycle"]):
        a4.axvline(k, color=MUTED, lw=0.5, ls=":")
    a4.set_xlabel(
        "Hencky strain (dotted: cell cycle; shaded: analysis window)",
        color=INK2,
        fontsize=8.5,
    )
    a4.set_ylabel(f"rms {r['field']}", color=INK2, fontsize=8.5)
    out = os.path.join(folder, "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"structure_{r['field']}_case_{c:04d}.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- repeat resolved by phase of the KR period -------------------------------------------
def ridge_and_lattice(cen, C, lattices, L, ring=(0.3, 0.4)):
    """Direction of the large-scale correlation ridge: the angle (signed, in (-90, 90])
    where C, averaged on a ring of radii ring x L, is largest. Elongated structures give
    an autocorrelation of ridges, not spots, so this is better defined than a 'repeat'
    point on a ridge. Alongside, the angles of the cell's two lattice vectors a1 = A e1
    and a2 = A e2, averaged over the frames: structures as long as the cell join their
    periodic images along a lattice vector, so a ridge that follows a1 or a2 through the
    period is set by the cell, not by the flow."""
    RX, RY = np.meshgrid(cen, cen, indexing="ij")
    th = (np.degrees(np.arctan2(RY, RX)) + 90) % 180 - 90
    R = np.hypot(RX, RY)
    on = (R >= ring[0] * L) & (R <= ring[1] * L) & np.isfinite(C)
    edges = np.arange(-90, 91, 3.0)
    idx = np.digitize(th[on], edges) - 1
    prof = np.array(
        [C[on][idx == i].mean() if np.any(idx == i) else -np.inf for i in range(60)]
    )
    i = int(np.argmax(prof))
    ang = lambda v: (np.degrees(np.arctan2(v[1], v[0])) + 90) % 180 - 90
    return dict(
        ridge_theta_deg=float(0.5 * (edges[i] + edges[i + 1])),
        ridge_C=float(prof[i]),
        a1_theta_deg=float(np.mean([ang(A[:, 0]) for A in lattices])),
        a2_theta_deg=float(np.mean([ang(A[:, 1]) for A in lattices])),
    )


def phase_resolved_repeat(fields, lattices, strain, eps_p, bins=5):
    """Strongest repeat of the autocorrelation averaged separately over the frames in each
    phase bin of the KR period (phase = strain / eps_p mod 1; 0 = undeformed cell). A
    physical arrangement of the structures should repeat at every phase; one set by the
    cell geometry appears only at some, and for golden and silver cells the sign of its
    angle flips between phases half a period apart (the cell is then the mirror image).
    Angles are signed, in (-90, 90] from the extension axis."""
    L = math.sqrt(abs(np.linalg.det(lattices[0])))
    N = fields.shape[-1]
    phase = np.mod(np.asarray(strain) / eps_p + 1e-6, 1.0)
    b = np.minimum((phase * bins).astype(int), bins - 1)
    rows, maps = [], []
    for i in range(bins):
        sel = b == i
        row = dict(phase_lo=i / bins, phase_hi=(i + 1) / bins, n_frames=int(sel.sum()))
        if not sel.any():
            nan = float("nan")
            rows.append(
                {
                    **row,
                    **dict.fromkeys(
                        ("peak_r", "peak_theta_deg", "peak_C", "peak_image_distance")
                        + (
                            "ridge_theta_deg",
                            "ridge_C",
                            "a1_theta_deg",
                            "a2_theta_deg",
                        ),
                        nan,
                    ),
                }
            )
            maps.append(None)
            continue
        cen, C = physical_acf(
            fields[sel], lattices[sel], half_width=0.6 * L, h=max(0.005, 0.8 / N) * L
        )
        pk = secondary_peak(cen, C, lattices[sel], 0.5 * L)
        if np.isfinite(pk["peak_theta_deg"]):
            pk["peak_theta_deg"] = (pk["peak_theta_deg"] + 90) % 180 - 90
        rows.append({**row, **pk, **ridge_and_lattice(cen, C, lattices[sel], L)})
        maps.append((cen, C))
    return rows, maps


def phases(folders, c, field="txx", window=(0.5, 1.0), bins=5):
    """phase_resolved_repeat for one case in each folder (e.g. one per seed) and pooled
    over them: a printed table and a figure of the maps (rows: folders, then pooled)."""
    runs = []
    for folder in folders:
        config, rows, _ = load_sweep(folder)
        p = case_params(config, rows[c])
        fr = load_frames(folder, c)
        e = fr["strain"]
        sel = (e >= window[0] * e[-1] - 1e-9) & (e <= window[1] * e[-1] + 1e-9)
        runs.append(
            (folder, field_frames(fr, p, field)[sel], fr["lattice"][sel], e[sel])
        )
    lat = kr_lattice(p.kr_lattice)
    if len(runs) > 1:
        runs.append(
            ("pooled",) + tuple(np.concatenate([r[k] for r in runs]) for k in (1, 2, 3))
        )
    f = lambda v, d=3: "–" if not np.isfinite(v) else f"{v:.{d}g}"
    print(
        f"case {c}, {field}, {lat.name} lattice (period {lat.eps_p:.3f} strain): strongest "
        "repeat per phase bin, angle signed from the extension axis"
    )
    results = []
    for name, fl, A, e in runs:
        rows, maps = phase_resolved_repeat(fl, A, e, lat.eps_p, bins)
        results.append((name, rows, maps))
        print(f"  {name} (strain {e[0]:.2f}–{e[-1]:.2f}):")
        for r in rows:
            print(
                f"    phase {r['phase_lo']:.1f}–{r['phase_hi']:.1f}: {r['n_frames']:3d} frames, "
                f"angle {f(r['peak_theta_deg'])}°, |r| {f(r['peak_r'])}, C {f(r['peak_C'], 2)}, "
                f"nearest cell image {f(r['peak_image_distance'])} away; large-scale ridge "
                f"{f(r['ridge_theta_deg'])}° (C {f(r['ridge_C'], 2)}); lattice vectors a1 "
                f"{f(r['a1_theta_deg'])}°, a2 {f(r['a2_theta_deg'])}°"
            )
    fig, axes = plt.subplots(
        len(results),
        bins,
        figsize=(2.3 * bins, 2.35 * len(results)),
        dpi=120,
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    lim = 0.5 * math.sqrt(abs(np.linalg.det(runs[0][2][0])))
    for i, (name, rows, maps) in enumerate(results):
        for j, (r, m) in enumerate(zip(rows, maps)):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if m is not None:
                cen, C = m
                ax.pcolormesh(
                    cen,
                    cen,
                    C.T,
                    cmap=DIVERGING,
                    norm=TwoSlopeNorm(0, -1, 1),
                    shading="nearest",
                )
                if np.isfinite(r["peak_r"]):
                    th = math.radians(r["peak_theta_deg"])
                    for s in (1, -1):
                        ax.plot(
                            s * r["peak_r"] * math.cos(th),
                            s * r["peak_r"] * math.sin(th),
                            "o",
                            ms=7,
                            mfc="none",
                            mec=INK,
                            mew=1.2,
                        )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_title(
                f"phase {r['phase_lo']:.1f}–{r['phase_hi']:.1f}: {f(r['peak_theta_deg'], 2)}°",
                fontsize=7.5,
                color=INK,
                loc="left",
            )
            if j == 0:
                ax.set_ylabel(
                    os.path.basename(os.path.normpath(name)), fontsize=7.5, color=INK2
                )
    fig.tight_layout()
    out = os.path.join(folders[0], "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"structure_phases_{field}_case_{c:04d}.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)
    return results


# --- many cases ---------------------------------------------------------------------------
COLUMNS = [
    "peak_r",
    "peak_theta_deg",
    "peak_C",
    "peak_image_distance",
    "y_half",
    "y_zero",
    "y_min_r",
    "y_min_C",
    "y_repeat_r",
    "y_repeat_C",
    "x_half",
    "x_zero",
    "x_min_r",
    "x_min_C",
    "x_repeat_r",
    "x_repeat_C",
    "lobe_theta_deg",
    "taylor_across",
    "taylor_along",
    "cutoff_across",
    "cutoff_along",
    "amp_mean",
    "cell_locked_fraction",
    "cell_locked_chance",
    "cell_modulation",
    "decorrelation_strain",
    "pattern_corr_one_period",
    "sqrt_D_tau",
    "sqrt_D_over_rate",
]


def _job(job):
    folder, config, row, field, window = job
    try:
        r = analyse(folder, config, row, field, window)
        return int(row["case"]), {k: r[k] for k in COLUMNS}, None
    except SystemExit as e:
        return int(row["case"]), {}, str(e)
    except Exception as e:
        return int(row["case"]), {}, f"{type(e).__name__}: {e}"


def sweep(folder, cases, field, window, x_name, workers):
    config, rows, _ = load_sweep(folder)
    names = list(config["sweeps"])
    have = [
        c
        for c in rows
        if os.path.exists(os.path.join(folder, "fields", f"case_{c:04d}.npz"))
    ]
    cases = have if cases == ["all"] else [int(c) for c in cases]
    if not cases:
        raise SystemExit("no field files found (run the sweep with --save-fields)")
    with Pool(workers) as pool:
        results = sorted(
            pool.imap_unordered(
                _job, [(folder, config, rows[c], field, window) for c in cases]
            )
        )
    path = os.path.join(folder, f"structure_{field}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f, ["case", *names, "status", *COLUMNS, "note"], extrasaction="ignore"
        )
        w.writeheader()
        for c, vals, err in results:
            w.writerow({**rows[c], **vals, "case": c, "note": err or ""})
    print(f"wrote {path} ({len(results)} cases)")
    bad = [(c, err) for c, _, err in results if err]
    if bad:
        print(f"{len(bad)} cases failed, e.g. case {bad[0][0]}: {bad[0][1]}")
    good = [(c, v) for c, v, err in results if not err]
    if len(good) < 2:
        return
    if not x_name:  # first swept parameter that actually takes more than one value
        varied = [n for n in names if len({rows[c][n] for c, _ in good}) > 1]
        x_name = varied[0] if varied else names[0]
    x, labels = axis_values([rows[c][x_name] for c, _ in good])
    order = np.argsort(x, kind="stable")
    x = x[order]
    get = lambda k: np.array([v[k] for _, v in good])[order]
    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        dpi=130,
        sharex=True,
        gridspec_kw=dict(height_ratios=[1.6, 1]),
    )
    fig.patch.set_facecolor(SURFACE)
    for a in (a1, a2):
        style_plot(a)
    series = [
        ("peak_r", "strongest repeat, any direction", SERIES[2], "-o"),
        ("y_repeat_r", "repeat spacing across the axis (y)", SERIES[0], "-o"),
        ("y_half", "half-width across the axis (y)", SERIES[0], "--s"),
        ("x_half", "half-width along the axis (x)", SERIES[1], "--s"),
        ("cutoff_across", "cutoff length across the lobe", SERIES[3], "-.^"),
    ]
    for k, lab, col, sty in series:
        y = get(k)
        ok = np.isfinite(y)
        a1.plot(
            x[ok],
            y[ok],
            sty,
            color=col,
            lw=1.5,
            ms=4.5,
            label=lab,
            mfc=col if "o" in sty else "none",
        )
    if x_name == "D" and np.all(x > 0):  # reference slope 1/2 (diffusive scaling)
        y0 = get("peak_r")
        ok = np.isfinite(y0)
        if ok.any():
            xm = np.exp(np.mean(np.log(x[ok])))
            ym = np.exp(np.mean(np.log(y0[ok])))
            a1.plot(
                x,
                ym * np.sqrt(x / xm),
                ":",
                color=MUTED,
                lw=1.2,
                label="∝ D^½ (reference)",
            )
    if labels is None and x.min() > 0 and x.max() / x.min() > 10:
        a1.set_xscale("log")
        a1.set_yscale("log")
    a1.set_ylabel("length (cell side = 1)", color=INK2, fontsize=9)
    a1.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    a1.set_title(
        f"Structure of {field} ({len(good)} cases)", color=INK, fontsize=9.5, loc="left"
    )
    a2.plot(
        x,
        100 * get("cell_locked_fraction"),
        "-o",
        color=SERIES[3],
        ms=4.5,
        label="amplitude variance following the cell cycle",
    )
    a2.plot(x, 100 * get("cell_locked_chance"), ":", color=MUTED, label="chance level")
    a2.set_ylabel("%", color=INK2, fontsize=9)
    a2.set_xlabel(x_name, color=INK2, fontsize=9)
    set_axis_labels(a2, labels)
    a2.set_ylim(0, 100)
    a2.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.tight_layout()
    out = os.path.join(folder, "figures", f"structure_{field}_summary.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("case", help="measures and figure for one case")
    a.add_argument("folder")
    a.add_argument("--case", type=int, required=True)
    s = sub.add_parser("sweep", help="measures for many cases, CSV and summary")
    s.add_argument("folder")
    s.add_argument("--cases", nargs="+", default=["all"])
    s.add_argument("--x", dest="x_name", help="swept parameter for the summary x axis")
    s.add_argument("--workers", type=int, default=4)
    ph = sub.add_parser("phases", help="strongest repeat per phase of the KR period")
    ph.add_argument(
        "folders", nargs="+", help="sweep folders (same sweep, other seeds)"
    )
    ph.add_argument("--case", type=int, required=True)
    ph.add_argument("--bins", type=int, default=5, help="phase bins per KR period")
    for sp in (a, s, ph):
        sp.add_argument("--field", choices=FIELDS, default="txx")
        sp.add_argument(
            "--window",
            default="0.5:1",
            help="strain window as fractions of the final strain (default 0.5:1)",
        )
    args = ap.parse_args(argv)
    if args.cmd == "phases":
        phases(
            args.folders, args.case, args.field, parse_window(args.window), args.bins
        )
    elif args.cmd == "case":
        config, rows, _ = load_sweep(args.folder)
        r = analyse(
            args.folder, config, rows[args.case], args.field, parse_window(args.window)
        )
        print(f"case {args.case}, field {args.field}:")
        for line in summary_lines(r):
            print("  " + line)
        print("wrote", case_figure(args.folder, config, rows, r))
    else:
        sweep(
            args.folder,
            args.cases,
            args.field,
            parse_window(args.window),
            args.x_name,
            args.workers,
        )


if __name__ == "__main__":
    main()
