"""Plots and movies for a sweep_kr.py output folder.

  python viz_sweep.py growth SWEEP_DIR [--measure stress|velocity|stretch]
      One figure per case: the perturbation size vs strain (log scale), the fitted growth
      rate over the fit window, and a GROWING / DECAYING label. Also writes an all-case
      overview and, for one or two swept parameters, a sigma*lam map.
        stress   (default) rms of C - <C>              growth_case_NNNN.png, growth_all.png,
                                                       sigma_map.png
        velocity rms of u - eps_dot (x, -y), the       growth_velocity_case_NNNN.png, ...
                 deviation from ideal planar extension,
                 in units of eps_dot * cell size. Uses the sweep's velocity record, or
                 reconstructs it from fields/ for sweeps run before it was recorded.
        stretch  rms of lambda - <lambda>, the chain   growth_stretch_case_NNNN.png, ...
                 stretch lambda = sqrt(tr C / 3) (finite extensibility inverted, as in
                 fourier_kr.py). Reconstructed from fields/, so it needs --save-fields
                 and has the snapshot spacing (fields per period), not the series'.

  python viz_sweep.py movie SWEEP_DIR --cases 0 3 7 [--quantity txx]
                                      [--history stress|velocity|stretch|both|all]
      Side-by-side movie (up to 8 cases) of a stress component (txx, tyy, txy, N1, trace)
      or the chain-stretch fluctuation (stretch) in physical coordinates,
      tiled over the periodic copies, with the growth/decay history below (stress, velocity
      deviation from planar extension, chain stretch; both = stress and velocity, all =
      all three, one strip each). Needs a sweep run with
      --save-fields. Cases are synchronized by Hencky strain.

  python viz_sweep.py rank SWEEP_DIR [--target 1.0] [--top 8]
      Classifies every case from its max|δτ_xx| / G history and prints shortlists, each with
      a ready-to-run movie command:
        plateau         grows by >= 10x and levels out (late slope |d ln/dε| <= 0.05)
        noise_floor     levels out at a level that follows forcing_amp (exponent > 0.5)
        growing_at_end  still growing over the last 30% of the strain
        transient       amplified >= 10x, then fell >= 3x from the peak
        quiet           none of the above
      With several forcing amplitudes, the forcing exponent d ln(final) / d ln(forcing_amp)
      separates noise-driven states (~1) from flow-driven ones (~0). Writes triage.csv.

For sweeps run with --schedule, strain follows each stage's eps_dot, a dotted line marks
each stage switch (labelled with the stage's scheduled values), the dashed fits and sigma*lam
are per stage (second half of each stage, as in sweep_kr), the growth command also writes
one sigma map per stage (sigma_map_stageN.png), and the movie clock shows the current stage.

Figures and movies go to SWEEP_DIR/figures/.
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
)

from kr_solver import Params, kr_lattice
from sweep_kr import CASE_OPTIONS, fit_window, velocity_rms
from viz_common import (
    DIVERGING,
    GRID,
    INK,
    INK2,
    MUTED,
    SERIES,
    SURFACE,
    TiledField,
    axis_values,
    cell_outline,
    set_axis_labels,
    style_axes,
)

GROW, DECAY = "#e34948", "#2a78d6"  # diverging poles: red = growing, blue = decaying
QUANTITIES = {  # label, function of (G, dC) with dC = (xx, xy, yy) fluctuation
    "txx": ("δτ_xx", lambda G, d: G * d[0]),
    "tyy": ("δτ_yy", lambda G, d: G * d[2]),
    "txy": ("δτ_xy", lambda G, d: G * d[1]),
    "N1": ("δN₁", lambda G, d: G * (d[0] - d[2])),
    # trace over xx + yy + zz (zz is the neutral direction, not equal to yy in planar flow);
    # equals G * delta tr C without finite extensibility. Needs snapshots with zz.
    "trace": ("δ tr τ", lambda G, d: G * (d[0] + d[2] + d[3])),
    # chain stretch lambda - <lambda>, lambda = sqrt(tr C / 3) with finite extensibility
    # inverted (dimensionless, not in units of G). Needs snapshots with zz.
    "stretch": ("δλ", None),
}


# --- loading --------------------------------------------------------------------------
def load_sweep(folder):
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    rows = {}
    with open(os.path.join(folder, "results.csv")) as f:
        for row in csv.DictReader(
            f
        ):  # with --resume a case can appear twice: last row wins
            for k, v in row.items():
                try:
                    row[k] = float(v) if v not in ("", "True", "False") else v
                except ValueError:
                    pass
            rows[int(row["case"])] = row
    series = {}
    old = os.path.join(folder, "series.npz")  # sweeps from before per-case series files
    if os.path.exists(old):
        series.update(np.load(old))
    sdir = os.path.join(folder, "series")
    if os.path.isdir(sdir):
        for name in sorted(os.listdir(sdir)):
            if name.startswith("case_") and name.endswith(".npz"):
                c = int(name[5:-4])
                with np.load(os.path.join(sdir, name)) as d:
                    series.update({f"{k}_{c}": d[k] for k in d.files})
    return config, dict(sorted(rows.items())), series


def upsample(field, n_min=192):
    """Exact Fourier interpolation of a periodic, band-limited field to >= n_min points."""
    N = field.shape[0]
    f = max(1, math.ceil(n_min / N))
    if f == 1:
        return field
    F = np.fft.fftshift(np.fft.fft2(field))
    pad = (N * f - N) // 2
    F = np.pad(F, ((pad, N * f - N - pad), (pad, N * f - N - pad)))
    return np.fft.ifft2(np.fft.ifftshift(F)).real * f * f


def growing(row):
    return str(row["status"]).startswith(
        ("unstable", "blew_up", "exceeded_L_max", "cfl_exceeded")
    )


def param_text(row, names):
    return ", ".join(
        f"{n}={row[n]:.3g}" if isinstance(row[n], float) else f"{n}={row[n]}"
        for n in names
    )


def wrap(text, width):
    return textwrap.wrap(text, width=max(10, int(width)), break_on_hyphens=False) or [
        ""
    ]


def case_label(row, names):
    parts = [
        f"{n} = {row[n]:.3g}" if isinstance(row[n], float) else f"{n} = {row[n]}"
        for n in names
    ]
    return f"case {int(row['case'])}" + (": " + ", ".join(parts) if parts else "")


def style_plot(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")


# --- growth figures -------------------------------------------------------------------
MEASURES = {
    "stress": dict(ylabel="rms(C − ⟨C⟩)", legend="rms fluctuation of C", suffix=""),
    "velocity": dict(
        ylabel="rms|v| / (ε̇ℓ)",
        legend="rms deviation from planar extension, |u − ε̇(x, −y)|",
        suffix="_velocity",
    ),
    "stretch": dict(
        ylabel="rms(λ − ⟨λ⟩)",
        legend="rms fluctuation of the chain stretch λ = √(tr C / 3)",
        suffix="_stretch",
    ),
}
TITLES = {  # growth-rate map titles and movie history strips
    "stress": ("stress fluctuation", "Stress fluctuation, rms(C − ⟨C⟩)"),
    "velocity": (
        "velocity deviation",
        "Velocity deviation from planar extension, rms|u − ε̇(x, −y)|",
    ),
    "stretch": ("chain-stretch fluctuation", "Chain stretch fluctuation, rms(λ − ⟨λ⟩)"),
}
HISTORIES = {"both": ["stress", "velocity"], "all": list(MEASURES)}


def case_params(config, row):
    swept = {n: row[n] for n in config["sweeps"]}
    values = {**config["fixed"], **swept}
    p = Params(
        model=config["model"],
        **{k: v for k, v in values.items() if k not in CASE_OPTIONS},
    )
    return dataclasses.replace(p, N=int(p.N))


# --- schedules ------------------------------------------------------------------------
def stage_table(config, row, series):
    """[(t_start, strain_start, Params, scheduled values)] for each stage a case reached.
    Without a schedule: one stage from t = 0 with the case's parameters. Every stage lasts
    --remaps-per-stage KR periods, so stage i starts at strain i K eps_p even when
    eps_dot is scheduled."""
    p0 = case_params(config, row)
    sched, c = config.get("schedule") or {}, int(row["case"])
    if not sched or f"t_switch_{c}" not in series:
        return [(0.0, 0.0, p0, {})]
    K = config["remaps_per_stage"] * kr_lattice(p0.kr_lattice).eps_p
    stages = []
    for i, t0 in enumerate(series[f"t_switch_{c}"]):
        values = {k: v[i] for k, v in sched.items()}
        stages.append((float(t0), i * K, dataclasses.replace(p0, **values), values))
    return stages


def stage_index(t, stages):
    """Stage of each time; a sample at a switch ends the earlier stage (as in sweep_kr)."""
    t0 = np.array([st[0] for st in stages])
    return np.clip(np.searchsorted(t0, t, side="left") - 1, 0, None)


def to_strain(t, stages):
    """Hencky strain at times t, stage by stage (eps_dot may be scheduled)."""
    i = stage_index(t, stages)
    t0, e0, ed = (
        np.array(a) for a in zip(*[(a, b, p.eps_dot) for a, b, p, _ in stages])
    )
    return e0[i] + ed[i] * (np.asarray(t) - t0[i])


def stage_fits(t, y, stages):
    """Per stage: (fitted samples, polyfit coefficients in t, sigma*lam of the stage) or
    None, fitted like sweep_kr: over the second half of the stage."""
    idx, fits = stage_index(t, stages), []
    for i, (t0, _, p, _) in enumerate(stages):
        own = np.flatnonzero(idx == i)
        sel = own[fit_window(t[own] - t0, y[own])] if len(own) >= 2 else own
        if len(sel) < 2:
            fits.append(None)
            continue
        coef = np.polyfit(t[sel], np.log(y[sel]), 1)
        fits.append((sel, coef, coef[0] * p.lam))
    return fits


def stage_text(values):
    return ", ".join(
        f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in values.items()
    )


def schedule_text(config):
    """One-line description of the sweep's schedule, or "" without one."""
    sched = config.get("schedule")
    if not sched:
        return ""
    steps = "; ".join(
        f"{k}: "
        + " → ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in vals)
        for k, vals in sched.items()
    )
    K = config["remaps_per_stage"]
    return f"schedule {steps}, every {K} KR period{'s' if K != 1 else ''}"


def mark_stages(ax, stages, fontsize=7):
    """Dotted line at each stage switch, labelled with the stage's scheduled values."""
    for i, (_, e0, _, values) in enumerate(stages):
        if i:
            ax.axvline(e0, color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
        if values:
            ax.text(
                e0,
                0.98,
                f" {i}: {stage_text(values)}",
                transform=ax.get_xaxis_transform(),
                fontsize=fontsize,
                color=INK2,
                va="top",
                ha="left",
                zorder=4,
                bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=0.5),
            )


def case_curve(folder, config, row, series, measure):
    """(t, y, sigma*lam, growing) for one case, or None if the data are not available.

    Velocity comes from the series recorded by the sweep, or, for sweeps run before it
    was recorded, is reconstructed from the saved field snapshots (exact: the Stokes
    velocity is linear in the stress fluctuation). The chain stretch is always
    reconstructed from the snapshots (the sweep does not record it)."""
    c = int(row["case"])
    if f"t_{c}" not in series:
        return None
    if measure == "stress":
        return series[f"t_{c}"], series[f"rms_{c}"], row["sigma_lam"], growing(row)
    p = case_params(config, row)
    if measure == "stretch":
        from fourier_kr import stretch_field  # here: fourier_kr imports this module

        path = os.path.join(folder, "fields", f"case_{c:04d}.npz")
        if not os.path.exists(path):
            return None
        d = dict(np.load(path))
        if d["dC"].shape[1] < 4:  # snapshots saved before zz was
            return None
        t, stages = d["t"], stage_table(config, row, series)
        idx = stage_index(t, stages)
        y = np.empty(len(t))
        for i, (_, _, p_i, _) in enumerate(stages):  # L_max may be scheduled
            sel = idx == i
            if sel.any():
                lam = stretch_field(dict(dC=d["dC"][sel], mean=d["mean"][sel]), p_i)
                y[sel] = (lam - lam.mean(axis=(1, 2), keepdims=True)).std(axis=(1, 2))
    elif f"vrms_{c}" in series:
        t, y = series[f"t_{c}"], series[f"vrms_{c}"]
    else:
        path = os.path.join(folder, "fields", f"case_{c:04d}.npz")
        if not os.path.exists(path):
            return None
        d = np.load(path)
        t = d["t"]
        stages = stage_table(config, row, series)
        ps = [stages[i][2] for i in stage_index(t, stages)]
        y = np.array(
            [
                velocity_rms(dC, A, q.G, q.eta_s, q.eps_dot)
                for dC, A, q in zip(d["dC"].astype(float), d["lattice"], ps)
            ]
        )
    fits = [f for f in stage_fits(t, y, stage_table(config, row, series)) if f]
    sigma_lam = fits[-1][2] if fits else float("nan")  # last stage reached
    blew = str(row["status"]).startswith(
        ("unstable_saturated", "blew_up", "exceeded_L_max")
    )
    return t, y, sigma_lam, blew or sigma_lam > config["sigma_tol"]


def growth_figures(folder, measure):
    config, rows, series = load_sweep(folder)
    names = list(config["sweeps"])
    m = MEASURES[measure]
    out = os.path.join(folder, "figures")
    os.makedirs(out, exist_ok=True)

    curves, strains, stage_sets = {}, {}, {}
    for c, row in rows.items():
        curve = case_curve(folder, config, row, series, measure)
        if curve is None:
            why = (
                "no time series"
                if f"t_{c}" not in series
                else "no velocity record or saved fields (rerun the sweep)"
                if measure == "velocity"
                else "no saved fields with the zz component (rerun with --save-fields)"
            )
            print(f"case {c}: {row['status']}, {why}")
            continue
        curves[c] = curve
        t, y, sigma_lam, grows = curve
        stages = stage_table(config, row, series)
        e = to_strain(t, stages)
        strains[c], stage_sets[c] = e, stages
        colour = GROW if grows else DECAY

        sched = schedule_text(config)
        label = wrap(
            f"case {c}: {param_text(row, names)}   ·   status: {row['status']}"
            + (f"   ·   {sched} (status: last stage reached)" if sched else ""),
            92,
        )
        extra = 0.15 * (len(label) - 1)
        fig, ax = plt.subplots(figsize=(6.0, 3.8 + extra), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        style_plot(ax)
        ax.semilogy(e, y, color=INK2, lw=1.5, label=m["legend"])
        fits = stage_fits(t, y, stages)
        if len(stages) == 1:
            sigma_lam_i = [sigma_lam]
        else:  # stress: the sweep's own per-stage rates; others: refitted per stage
            sigma_lam_i = [
                (
                    row.get(f"sigma_lam_s{i}", f[2] if f else np.nan)
                    if measure == "stress"
                    else (f[2] if f else np.nan)
                )
                for i, f in enumerate(fits)
            ]
        for i, (f, s_i) in enumerate(zip(fits, sigma_lam_i)):
            if f is None:
                continue
            sel, coef, _ = f
            ax.semilogy(
                e[sel],
                np.exp(np.polyval(coef, t[sel])),
                color=(
                    (GROW if s_i > config["sigma_tol"] else DECAY)
                    if len(stages) > 1
                    else colour
                ),
                lw=2.5,
                ls="--",
                label=(
                    f"fit: σλ = {s_i:+.3f}"
                    if len(stages) == 1
                    else ("fit per stage (second half)" if i == 0 else None)
                ),
            )
            if len(stages) > 1:
                ax.annotate(
                    f"σλ {s_i:+.2f}",
                    (e[sel][0], np.exp(np.polyval(coef, t[sel][0]))),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="left",
                    fontsize=7,
                    color=INK2,
                )
        mark_stages(ax, stages)
        ax.axhline(y[0], color=MUTED, lw=0.8, ls=":", label="initial value")
        ax.set_xlabel("Hencky strain ε = ε̇t", color=INK2, fontsize=9)
        ax.set_ylabel(m["ylabel"], color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
        forced = case_params(config, row).forcing_amp > 0
        verdict = (
            "GROWING"
            if grows
            else "STABLE · settles at the forcing noise floor"
            if forced
            else "DECAYING"
        )
        H = 3.8 + extra
        fig.text(
            0.02,
            1 - 0.11 / H,
            verdict,
            color=colour,
            fontsize=12,
            weight="bold",
            va="top",
        )
        fig.text(
            0.02,
            1 - 0.36 / H,
            "\n".join(label),
            color=INK2,
            fontsize=8.5,
            va="top",
            linespacing=1.3,
        )
        fig.tight_layout(rect=(0, 0, 1, 1 - (0.55 + extra) / H))
        fig.savefig(
            os.path.join(out, f"growth{m['suffix']}_case_{c:04d}.png"),
            facecolor=SURFACE,
        )
        plt.close(fig)

    overview(curves, strains, stage_sets, len(rows), out, m)
    if len(names) in (1, 2):
        last = " (last stage reached)" if config.get("schedule") else ""
        sigma_map(rows, curves, names, out, m, last)
        if config.get("schedule"):  # one map per stage: the rate each stage gave
            for i in range(len(config["schedule"][next(iter(config["schedule"]))])):
                per = {}
                for c, (t, y, _, _) in curves.items():
                    st = stage_sets[c]
                    if i >= len(st):
                        continue
                    if measure == "stress":
                        s_i = rows[c].get(f"sigma_lam_s{i}", np.nan)
                    else:
                        f = stage_fits(t, y, st)[i]
                        s_i = f[2] if f else np.nan
                    if isinstance(s_i, float) and np.isfinite(s_i):
                        per[c] = (t, y, s_i, s_i > config["sigma_tol"])
                values = {k: v[i] for k, v in config["schedule"].items()}
                sigma_map(
                    rows,
                    per,
                    names,
                    out,
                    m,
                    f", stage {i}: {stage_text(values)}",
                    f"_stage{i}",
                )
    else:
        print("sigma map needs exactly one or two swept parameters; skipped")
    print(f"figures written to {out}")


def overview(curves, strains, stage_sets, n_cases, out, m):
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_plot(ax)
    if stage_sets:  # stages switch at the same strain in every case (same lattice)
        mark_stages(ax, max(stage_sets.values(), key=len))
    for c, (t, y, _, grows) in curves.items():
        ax.semilogy(
            strains[c],
            y / y[0],
            lw=1.2,
            alpha=0.8,
            color=GROW if grows else DECAY,
        )
    ax.plot([], [], color=GROW, label="growing")
    ax.plot([], [], color=DECAY, label="decaying")
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Hencky strain ε = ε̇t", color=INK2, fontsize=9)
    ax.set_ylabel(f"{m['ylabel']}, relative to initial", color=INK2, fontsize=9)
    ax.set_title(
        f"All cases ({len(curves)} of {n_cases} with data)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"growth{m['suffix']}_all.png"), facecolor=SURFACE)
    plt.close(fig)


def sigma_map(rows, curves, names, out, m, title_extra="", file_tag=""):
    if not curves:
        return
    have = [rows[c] for c in curves]
    s_all = np.array([curves[c][2] for c in curves])
    grows = [curves[c][3] for c in curves]
    lim = max(np.abs(s_all).max(), 1e-12)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_plot(ax)

    if len(names) == 1:
        n = names[0]
        x, labels = axis_values([r[n] for r in have])
        order = np.argsort(x, kind="stable")
        ax.plot(x[order], s_all[order], color=MUTED, lw=1)
        ax.scatter(
            x,
            s_all,
            s=40,
            zorder=3,
            edgecolor=SURFACE,
            linewidth=1.5,
            c=[GROW if g else DECAY for g in grows],
        )
        ax.axhline(0, color=INK2, lw=0.8)
        if labels is None and x.min() > 0 and x.max() / x.min() > 10:
            ax.set_xscale("log")
        set_axis_labels(ax, labels)
        ax.set_xlabel(n, color=INK2, fontsize=9)
        ax.set_ylabel("σλ", color=INK2, fontsize=9)
    else:
        n1, n2 = names
        v1 = sorted({r[n1] for r in rows.values()})
        v2 = sorted({r[n2] for r in rows.values()})
        grid = np.full((len(v2), len(v1)), np.nan)
        for r, s in zip(have, s_all):
            grid[v2.index(r[n2]), v1.index(r[n1])] = s
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=DIVERGING,
            norm=TwoSlopeNorm(0, -lim, lim),
        )
        for i in range(len(v2)):
            for j in range(len(v1)):
                txt = "n/a" if np.isnan(grid[i, j]) else f"{grid[i, j]:+.2f}"
                dark = not np.isnan(grid[i, j]) and abs(grid[i, j]) > 0.55 * lim
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=SURFACE if dark else INK,
                )
        fmt = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
        ax.set_xticks(range(len(v1)), [fmt(v) for v in v1])
        ax.set_yticks(range(len(v2)), [fmt(v) for v in v2])
        ax.set_xlabel(n1, color=INK2, fontsize=9)
        ax.set_ylabel(n2, color=INK2, fontsize=9)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("σλ  (red: growing, blue: decaying)", color=INK2, fontsize=8.5)
        cb.ax.tick_params(colors=MUTED, labelsize=7.5)
        cb.outline.set_edgecolor(GRID)
    what = next(TITLES[k][0] for k, v in MEASURES.items() if v is m)
    ax.set_title(
        f"Growth rate of the {what}{title_extra}"
        + ("\n" if title_extra else " ")
        + "(n/a: no data or no steady base state)",
        color=INK,
        fontsize=9.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, f"sigma{m['suffix']}_map{file_tag}.png"), facecolor=SURFACE
    )
    plt.close(fig)


# --- movie ----------------------------------------------------------------------------
def movie(folder, cases, quantity, fps, history="stress"):
    config, rows, series = load_sweep(folder)
    names = list(config["sweeps"])
    if not 1 <= len(cases) <= len(SERIES):
        raise SystemExit(
            f"choose 1 to {len(SERIES)} cases (one colour and letter each)"
        )
    data = {}
    for c in cases:
        path = os.path.join(folder, "fields", f"case_{c:04d}.npz")
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} not found: rerun the sweep with --save-fields "
                f"(case {c} status: {rows.get(c, {}).get('status', 'unknown')})"
            )
        data[c] = dict(np.load(path))
        data[c]["stages"] = stage_table(config, rows[c], series)
        data[c]["strain"] = to_strain(data[c]["t"], data[c]["stages"])
    qlabel, qfun = QUANTITIES[quantity]
    measures = HISTORIES.get(history, [history])
    curves = {}
    for m in measures:
        for c in cases:
            curves[m, c] = case_curve(folder, config, rows[c], series, m)
            if curves[m, c] is None:
                raise SystemExit(f"case {c}: no {m} history available")

    # Common strain grid at the coarsest typical snapshot spacing, so no case repeats frames
    # (the median ignores the short final gap older sweeps saved). Each case shows its
    # nearest snapshot, and holds its last one once it has stopped.
    d_strain = max(np.median(np.diff(d["strain"])) for d in data.values())
    e_max = max(d["strain"][-1] for d in data.values())
    strains = np.arange(0, e_max + 0.5 * d_strain, d_strain)
    if strains[-1] < e_max - 1e-9:
        strains = np.append(strains, e_max)

    n = len(cases)
    if quantity in ("trace", "stretch") and any(
        d["dC"].shape[1] < 4 for d in data.values()
    ):
        raise SystemExit(
            f"the {quantity} needs the zz component, which sweeps run before it was "
            "saved do not have; rerun these cases with the current sweep_kr.py"
        )
    if quantity == "stretch":
        from fourier_kr import (
            stretch_fluctuation,  # here: fourier_kr imports this module
        )

        for c in cases:
            data[c]["dlam"] = stretch_fluctuation(data[c], case_params(config, rows[c]))
    unit = "" if qfun is None else " G"
    letters = "ABCDEFGH"
    vary = [k for k in names if len({repr(rows[c][k]) for c in cases}) > 1]
    common = [k for k in names if k not in vary]

    # Layout in inches, so wrapped text never overlaps the panels.
    ncols = {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}[n]
    nrows = math.ceil(n / ncols)
    P = 4.0 if ncols <= 2 else 3.1  # panel size
    gap_x, left, right = 0.25, 0.35, 1.15  # right margin holds the colour bar
    W = left + ncols * P + (ncols - 1) * gap_x + right
    line = 7.5 * 1.32 / 72  # height of one 7.5 pt text line
    chars_panel = P * 15.5  # characters per line at 7.5 pt
    titles = []
    for i, c in enumerate(cases):
        r = rows[c]
        lines = wrap(f"{letters[i]}   case {c} · {r['status']}", chars_panel)
        if vary:
            lines += wrap(param_text(r, vary), chars_panel)
        sig = " · ".join(f"σλ {m} {curves[m, c][2]:+.3g}" for m in measures)
        lines += wrap(sig, chars_panel)
        titles.append(lines)
    title_h = max(len(t) for t in titles) * line + 0.12
    caption = f"{config['model']}" + (
        f" · same for all panels: {param_text(rows[cases[0]], common)}"
        if common
        else ""
    )
    cap_lines = wrap(caption, (W - 0.7) * 13.5)
    header_h = 0.42 + len(cap_lines) * 8 * 1.32 / 72 + 0.15
    strip_h, strip_gap, bottom = 1.9, 0.55, 0.55
    H = (
        header_h
        + nrows * (title_h + P)
        + (nrows - 1) * 0.15
        + len(measures) * (strip_h + strip_gap)
        + bottom
    )
    fig = plt.figure(figsize=(W, H), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    box = lambda x, y, w, h: [x / W, y / H, w / W, h / H]  # inches from bottom-left

    clock = fig.text(left / W, 1 - 0.12 / H, "", fontsize=10, color=INK, va="top")
    fig.text(
        left / W,
        1 - 0.42 / H,
        "\n".join(cap_lines),
        fontsize=8,
        color=INK2,
        va="top",
        linespacing=1.32,
    )

    panels = []
    for i, c in enumerate(cases):
        row_i, col_i = divmod(i, ncols)
        x = left + col_i * (P + gap_x)
        y_top = H - header_h - row_i * (title_h + P + 0.15)
        ax = fig.add_axes(box(x, y_top - title_h - P, P, P))
        style_axes(ax)
        fig.patches.append(
            plt.Rectangle(
                ((x) / W, (y_top - 0.13) / H),
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
        L = math.sqrt(abs(np.linalg.det(data[c]["lattice"][0])))
        tiles = TiledField(
            ax,
            (-0.75 * L, 0.75 * L, -0.75 * L, 0.75 * L),
            cmap=DIVERGING,
            vmin=-1,
            vmax=1,
            zorder=1,
        )
        (cell,) = ax.plot([], [], "-", color=INK, lw=1.0, zorder=3)
        amp = ax.text(
            0.02,
            0.03,
            "",
            transform=ax.transAxes,
            fontsize=7,
            color=INK,
            zorder=4,
            bbox=dict(fc=SURFACE, ec="none", alpha=0.85),
        )
        panels.append((c, tiles, cell, amp))
    panels_bottom = H - header_h - nrows * (title_h + P) - (nrows - 1) * 0.15
    cax = fig.add_axes(
        box(
            W - right + 0.2,
            panels_bottom + 0.15 * (H - header_h - panels_bottom),
            0.14,
            0.6 * (H - header_h - panels_bottom),
        )
    )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1)), cax=cax
    )
    cb.set_label(f"{qlabel} / max|{qlabel}|", color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=MUTED, labelsize=7.5)
    cb.outline.set_edgecolor(GRID)

    markers = []
    strip_w = ncols * P + (ncols - 1) * gap_x - 0.55
    for k, m in enumerate(measures):
        y0 = panels_bottom - (k + 1) * (strip_h + strip_gap)
        hist = fig.add_axes(box(left + 0.55, y0, strip_w, strip_h))
        style_plot(hist)
        ends = []
        for i, c in enumerate(cases):
            t, yv, sigma_lam, grows = curves[m, c]
            xs, ys = to_strain(t, data[c]["stages"]), yv / yv[0]
            hist.semilogy(xs, ys, color=SERIES[i], lw=1.6)
            ends.append((xs[-1], ys[-1], letters[i]))
        hist.axhline(1, color=MUTED, lw=0.8, ls=":")
        mark_stages(hist, max((data[c]["stages"] for c in cases), key=len), 6.5)
        markers.append(hist.axvline(0, color=MUTED, lw=1, ls="--"))
        hist.set_xlim(0, e_max * 1.06)
        # compact plain labels at 1, 2, 5 x 10^k (after plotting: semilogy resets these)
        lo, hi = hist.get_ylim()
        if hi / lo >= 1e4:  # many decades (e.g. decay to roundoff): powers of 10 only
            hist.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            hist.yaxis.set_minor_locator(NullLocator())
        elif hi / lo >= 10:
            hist.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            hist.yaxis.set_minor_locator(
                LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0))
            )
        else:  # under a decade: evenly spaced plain ticks, so several are labelled
            hist.yaxis.set_major_locator(MaxNLocator(nbins=5))
            hist.yaxis.set_minor_locator(NullLocator())
        hist.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        hist.yaxis.set_minor_formatter(NullFormatter())
        # direct letter labels at the line ends, nudged apart vertically
        hist.get_ylim()
        to_disp = hist.transData.transform
        placed = sorted(((to_disp((xe, ye))[1], xe, ye, lab) for xe, ye, lab in ends))
        last = -1e9
        for yd, xe, ye, lab in placed:
            yd_new = max(yd, last + 9 * fig.dpi / 72)
            hist.annotate(
                lab,
                (xe, ye),
                xytext=(3, (yd_new - yd) * 72 / fig.dpi),
                textcoords="offset points",
                fontsize=7.5,
                color=INK2,
                va="center",
                weight="bold",
            )
            last = yd_new
        hist.set_title(
            TITLES[m][1],
            color=INK,
            fontsize=8.5,
            loc="left",
        )
        hist.set_ylabel("relative to initial", color=INK2, fontsize=8.5)
        if k == len(measures) - 1:
            hist.set_xlabel("Hencky strain ε = ε̇t", color=INK2, fontsize=9)

    out = os.path.join(folder, "figures")
    os.makedirs(out, exist_ok=True)
    tag = "" if history == "stress" else f"_{history}"
    path = os.path.join(
        out, f"movie_{quantity}{tag}_cases_{'-'.join(map(str, cases))}.mp4"
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
            # H.264 needs even frame sizes; pad by at most 1 px
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ],
    )
    longest = max((data[c]["stages"] for c in cases), key=len)

    def stage_now(e):
        if len(longest) == 1:
            return ""
        i = int(np.searchsorted([st[1] for st in longest], e, side="right")) - 1
        return f"   ·   stage {i}: {stage_text(longest[i][3])}"

    with writer.saving(fig, path, dpi=110):
        for e in strains:
            for c, tiles, cell, amp in panels:
                d = data[c]
                f = int(np.argmin(np.abs(d["strain"] - min(e, d["strain"][-1]))))
                raw = d["dlam"][f] if qfun is None else qfun(float(d["G"]), d["dC"][f])
                field = upsample(raw)
                peak = float(np.abs(field).max())
                tiles.draw(field / peak if peak > 0 else field, d["lattice"][f])
                cell.set_data(*cell_outline(d["lattice"][f]))
                stopped = "   (run stopped)" if e > d["strain"][-1] + 1e-9 else ""
                amp.set_text(f"max|{qlabel}| = {peak:.2g}{unit}{stopped}")
            for marker in markers:
                marker.set_xdata([e, e])
            clock.set_text(f"Hencky strain ε = {e:5.2f}" + stage_now(e))
            writer.grab_frame()
    plt.close(fig)
    print("wrote", path)


# --- ranking ------------------------------------------------------------------------
PLATEAU_TOL = 0.05  # |d ln(max|dtau_xx|) / d strain| below this over the late window
BIG = 10.0  # "large" growth factor


def late_slope(e, y, late_frac=0.3):
    """d ln y / d strain fitted over the last late_frac of the strain range."""
    sel = (e >= e[-1] - late_frac * (e[-1] - e[0])) & (y > 0)
    if sel.sum() < 3:
        return float("nan")
    return float(np.polyfit(e[sel], np.log(y[sel]), 1)[0])


def behaviour(m):
    """One label per case from its max|dtau_xx| history (see rank_cases)."""
    if not np.isfinite(m["late_slope"]):
        return "too_short"
    if m["late_slope"] > PLATEAU_TOL:
        return "growing_at_end"
    if m["growth"] >= BIG and abs(m["late_slope"]) <= PLATEAU_TOL:
        return "plateau"
    if m["peak_over_initial"] >= BIG and m["peak_over_final"] >= 3:
        return "transient"
    return "quiet"


def forcing_exponents(rows, metrics, names):
    """For cases that differ only in forcing_amp: slope of log(final max|dtau_xx|) vs
    log(forcing_amp). About 1: amplitude set by the noise (linear response). About 0:
    amplitude set by the flow (self-sustained / saturated)."""
    if "forcing_amp" not in names:
        return {}
    others = [n for n in names if n != "forcing_amp"]
    groups = {}
    for c, r in rows.items():
        if (
            c in metrics
            and isinstance(r.get("forcing_amp"), float)
            and r["forcing_amp"] > 0
        ):
            groups.setdefault(tuple(r[n] for n in others), []).append(c)
    out = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        x = np.log([rows[c]["forcing_amp"] for c in members])
        y = np.log([metrics[c]["dtxx_final"] for c in members])
        if np.ptp(x) > 0 and np.all(np.isfinite(y)):
            k = float(np.polyfit(x, y, 1)[0])
            out.update({c: k for c in members})
    return out


def rank_cases(folder, target=1.0, top=8):
    """Classify every case and print shortlists with ready-made movie commands."""
    config, rows, series = load_sweep(folder)
    names = list(config["sweeps"])
    metrics = {}
    for c, r in rows.items():
        if f"dtxx_{c}" not in series:
            continue
        t, y = series[f"t_{c}"], series[f"dtxx_{c}"]
        e = to_strain(t, stage_table(config, r, series))
        metrics[c] = dict(
            dtxx_initial=y[0],
            dtxx_final=y[-1],
            dtxx_peak=y.max(),
            growth=y[-1] / y[0],
            peak_over_initial=y.max() / y[0],
            peak_over_final=y.max() / y[-1],
            late_slope=late_slope(e, y),
            strain_end=e[-1],
        )
    if not metrics:
        raise SystemExit(
            "no max|dtau_xx| records found: rank needs a sweep run with this "
            "version of sweep_kr.py (series/case_NNNN.npz with 'dtxx')"
        )
    kexp = forcing_exponents(rows, metrics, names)
    for c, m in metrics.items():
        m["behaviour"] = behaviour(m)
        m["forcing_exponent"] = kexp.get(c, float("nan"))
        if (
            m["behaviour"] in ("plateau", "quiet")
            and m["forcing_exponent"] > 0.5
            and abs(m["late_slope"]) <= PLATEAU_TOL
        ):
            m["behaviour"] = "noise_floor"  # levels out at a level set by the forcing
        m["dist_to_target"] = (  # 0 at the roundoff floor (scheduled runs reach it)
            abs(math.log10(m["dtxx_final"] / target))
            if m["dtxx_final"] > 0
            else math.inf
        )

    # full table
    cols = [
        "case",
        *names,
        "status",
        "behaviour",
        "dtxx_final",
        "dtxx_peak",
        "growth",
        "late_slope",
        "forcing_exponent",
        "strain_end",
        "diff_len_dx",
        "max_cfl",
    ]
    out = os.path.join(folder, "triage.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for c in sorted(metrics):
            w.writerow({**rows[c], **metrics[c], "case": c})

    # status overview
    statuses = {}
    for r in rows.values():
        key = str(r["status"]).split(":")[0]
        statuses[key] = statuses.get(key, 0) + 1
    kinds = {}
    for m in metrics.values():
        kinds[m["behaviour"]] = kinds.get(m["behaviour"], 0) + 1
    print(
        f"{len(rows)} cases. Status: "
        + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items()))
    )
    print(
        "Behaviour of max|δτ_xx|: "
        + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    )
    if kexp:
        print(
            f"Forcing exponent available for {len(kexp)} cases "
            "(~1: amplitude follows the noise, ~0: amplitude set by the flow)"
        )

    def fmt(c):
        r, m = rows[c], metrics[c]
        p = ", ".join(
            f"{n}={r[n]:.3g}" if isinstance(r[n], float) else f"{n}={r[n]}"
            for n in names
        )
        k = (
            f"  k={m['forcing_exponent']:.2f}"
            if np.isfinite(m["forcing_exponent"])
            else ""
        )
        flag = "  [under-resolved]" if "underresolved" in str(r["status"]) else ""
        return (
            f"  case {c:>4}: final {m['dtxx_final']:.3g} G, peak {m['dtxx_peak']:.3g} G, "
            f"late slope {m['late_slope']:+.3f}{k}{flag}\n           {p}"
        )

    def section(title, cases, why):
        print(f"\n== {title} ({len(cases)}) ==  {why}")
        for c in cases[:top]:
            print(fmt(c))
        if cases:
            print(
                f"  movie: python viz_sweep.py movie {folder} --cases "
                f"{' '.join(map(str, cases[:4]))} --history both"
            )

    pick = lambda kind: [c for c, m in metrics.items() if m["behaviour"] == kind]
    flow_set = [c for c in pick("plateau")]
    section(
        f"Levels out near {target:g} G",
        sorted(flow_set, key=lambda c: metrics[c]["dist_to_target"]),
        "plateau not explained by the forcing, closest to the target first",
    )
    section(
        "Still growing at the end",
        sorted(pick("growing_at_end"), key=lambda c: -metrics[c]["late_slope"]),
        "may level out with more strain (rerun with a larger --strain)",
    )
    section(
        "Transient growth then decay",
        sorted(pick("transient"), key=lambda c: -metrics[c]["peak_over_initial"]),
        "largest amplification first",
    )
    section(
        "Levels out at the noise floor",
        sorted(pick("noise_floor"), key=lambda c: metrics[c]["dist_to_target"]),
        "amplitude follows forcing_amp: stable, noise-driven",
    )
    bad = [
        c
        for c, r in rows.items()
        if str(r["status"]).startswith(
            ("error", "cfl_exceeded", "blew_up", "exceeded_L_max")
        )
    ]
    if bad:
        print(
            f"\n== Need attention ({len(bad)}) ==  "
            + ", ".join(f"{c}: {str(rows[c]['status'])[:40]}" for c in bad[:top])
        )
    print(f"\nFull table: {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("growth", help="growth/decay figures for every case")
    g.add_argument("folder")
    g.add_argument(
        "--measure",
        choices=MEASURES,
        default="stress",
        help="stress: rms of C - <C>; velocity: rms deviation from ideal planar "
        "extension, in units of eps_dot * cell size; stretch: rms of the chain-stretch "
        "fluctuation (from saved fields)",
    )
    m = sub.add_parser("movie", help="side-by-side field movie for selected cases")
    m.add_argument("folder")
    m.add_argument("--cases", type=int, nargs="+", required=True)
    m.add_argument("--quantity", choices=QUANTITIES, default="txx")
    m.add_argument(
        "--history",
        choices=(*MEASURES, *HISTORIES),
        default="stress",
        help="growth/decay strip(s) below the fields (both: stress and velocity; "
        "all: stress, velocity and stretch)",
    )
    m.add_argument("--fps", type=int, default=12)
    r = sub.add_parser("rank", help="classify cases and shortlist interesting ones")
    r.add_argument("folder")
    r.add_argument(
        "--target",
        type=float,
        default=1.0,
        help="max|dtau_xx| / G level of interest (default 1)",
    )
    r.add_argument("--top", type=int, default=8, help="cases listed per category")
    args = ap.parse_args()
    if args.cmd == "growth":
        growth_figures(args.folder, args.measure)
    elif args.cmd == "rank":
        rank_cases(args.folder, args.target, args.top)
    else:
        movie(args.folder, args.cases, args.quantity, args.fps, args.history)


if __name__ == "__main__":
    main()
