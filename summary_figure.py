"""Cutoff length and orientation of the structures against D, from summary_data.py output.

  python summary_figure.py [--data summary_data.pkl] [--out summary_cutoff_orientation.png]
      [--N 256] [--min-dx 3] [--max-growth 1.5] [--min-frames 10]

Runs are grouped by grid size N and KR lattice (one colour each) and pooled over seeds at
each D: large markers are the mean over runs (bars: sd over runs, when there are several),
small faint markers the individual runs. Runs whose sqrt(D/eps_dot) spans fewer than
--min-dx grid spacings are drawn hollow and left out of the fits (their cutoff locks onto
the grid). Runs that had not saturated in their window (rms growth above --max-growth) or
have fewer than --min-frames frames there are left out entirely and listed.
"""

import argparse
import math
import pickle
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from kr_solver import kr_lattice
from viz_common import INK, INK2, MUTED, SERIES, SURFACE
from viz_sweep import style_plot

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--data", default="summary_data.pkl")
ap.add_argument("--out", default="summary_cutoff_orientation.png")
ap.add_argument("--N", type=int, nargs="+", help="only these grid sizes")
ap.add_argument("--min-dx", type=float, default=3.0)
ap.add_argument("--max-growth", type=float, default=1.5)
ap.add_argument("--min-frames", type=int, default=10)
args = ap.parse_args()

runs = list(pickle.load(open(args.data, "rb")).values())
if args.N:
    runs = [r for r in runs if r["N"] in args.N]
bad = [
    r for r in runs if r["growth"] > args.max_growth or r["n_frames"] < args.min_frames
]
for r in sorted(bad, key=lambda r: (r["folder"], r["case"])):
    print(
        f"left out: {r['folder']} case {r['case']} (D={r['D']:.3g}, N={r['N']}): strain to "
        f"{r['final_strain']:.2f}, {r['n_frames']} frames, growth {r['growth']:.1f}, {r['status']}"
    )
runs = [r for r in runs if r not in bad]
if not runs:
    raise SystemExit("no saturated runs to plot")
for r in runs:
    r["lD"] = math.sqrt(r["D"] / r["eps_dot"])
    r["resolved"] = r["lD"] * r["N"] / r["Lbox"] >= args.min_dx
groups = defaultdict(list)
for r in runs:
    groups[(r["N"], r["lattice"])].append(r)
keys = sorted(groups)
if len(keys) > len(SERIES):
    raise SystemExit(f"{len(keys)} groups; at most {len(SERIES)} (use --N)")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.7), dpi=150)
fig.patch.set_facecolor(SURFACE)
plain = FuncFormatter(lambda v, _: f"{v:g}")
for a in (a1, a2):
    style_plot(a)
    a.set_xscale("log")
    a.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    a.xaxis.set_minor_locator(NullLocator())
    a.xaxis.set_major_formatter(plain)
    a.set_xlabel("stress diffusivity D", color=INK2, fontsize=9)


def by_D(rs):
    out = defaultdict(list)
    for r in rs:
        out[r["D"]].append(r)
    return sorted(out.items())


# (a) cutoff length
fit_text, pooled_ratio = [], []
for g, key in enumerate(keys):
    col, rs = SERIES[g], groups[key]
    for r in rs:
        a1.plot(
            r["D"],
            r["pooled"]["cutoff_across"],
            "o",
            ms=3.5,
            color=col,
            alpha=0.45,
            mfc=col if r["resolved"] else "none",
        )
    Ds, m, sd, res = [], [], [], []
    for D, rr in by_D(rs):
        v = np.array([r["pooled"]["cutoff_across"] for r in rr])
        Ds.append(D)
        m.append(v.mean())
        sd.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        res.append(all(r["resolved"] for r in rr))
    Ds, m, sd, res = map(np.array, (Ds, m, sd, res))
    n_seeds = len({r["folder"] for r in rs})
    lab = f"N = {key[0]}, {key[1]} cell ({n_seeds} folder{'s' if n_seeds > 1 else ''})"
    a1.plot(Ds, m, "-", color=col, lw=1.4)
    a1.errorbar(
        Ds[res], m[res], yerr=sd[res], fmt="o", ms=6, color=col, capsize=3, label=lab
    )
    a1.errorbar(
        Ds[~res],
        m[~res],
        yerr=sd[~res],
        fmt="o",
        ms=6,
        color=col,
        mfc="none",
        capsize=3,
    )
    ok = [r for r in rs if r["resolved"]]
    if len({r["D"] for r in ok}) >= 3:
        x = np.log([r["D"] for r in ok])
        y = np.log([r["pooled"]["cutoff_across"] for r in ok])
        (b, _), cov = np.polyfit(x, y, 1, cov=True)
        fit_text.append(f"N = {key[0]}: exponent {b:.2f} ± {math.sqrt(cov[0, 0]):.2f}")
    pooled_ratio += [r["pooled"]["cutoff_across"] / r["lD"] for r in ok]
if pooled_ratio:
    c = float(np.exp(np.mean(np.log(pooled_ratio))))
    rate = np.median([r["eps_dot"] for r in runs])
    Dl = np.geomspace(
        min(r["D"] for r in runs) / 1.3, max(r["D"] for r in runs) * 1.3, 50
    )
    a1.plot(
        Dl,
        c * np.sqrt(Dl / rate),
        ":",
        color=MUTED,
        lw=1.3,
        label=f"{c:.2f} √(D/ε̇) (slope ½)",
    )
a1.set_yscale("log")
a1.yaxis.set_major_locator(LogLocator(subs=(1, 1.5, 2, 3, 5, 7)))
a1.yaxis.set_minor_locator(NullLocator())
a1.yaxis.set_major_formatter(plain)
a1.set_ylabel("cutoff length across the structures", color=INK2, fontsize=9)
a1.set_title("Cutoff length (sharpest-edge scale)", color=INK, fontsize=10, loc="left")
a1.text(
    0.97,
    0.04,
    "\n".join(fit_text),
    transform=a1.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    color=INK2,
)
a1.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")

# (b) orientation
for g, key in enumerate(keys):
    col = SERIES[g]
    for qty, mk, ls in (("ridge_theta_deg", "o", "-"), ("lobe_theta_deg", "s", "--")):
        Ds, m, sd = [], [], []
        for D, rr in by_D(groups[key]):
            v = np.abs([fr[qty] for r in rr for fr in r["frames"]])
            Ds.append(D)
            m.append(v.mean())
            sd.append(v.std())
            for r in rr:
                a2.plot(
                    D,
                    np.mean(np.abs([fr[qty] for fr in r["frames"]])),
                    mk,
                    ms=3,
                    color=col,
                    alpha=0.4,
                )
        a2.errorbar(Ds, m, yerr=sd, fmt=mk + ls, ms=5.5, color=col, lw=1.4, capsize=3)
for name in sorted({r["lattice"] for r in runs}):
    th = abs(math.degrees(kr_lattice(name).theta))
    a2.axhline(th, color=MUTED, ls=":", lw=1.2)
    a2.text(
        0.99,
        th,
        f"{name} cell angle {th:.1f}°",
        transform=a2.get_yaxis_transform(),
        fontsize=7.5,
        color=INK2,
        va="bottom",
        ha="right",
    )
a2.set_ylim(0, 62)
a2.set_ylabel("|angle| from the extension axis (°)", color=INK2, fontsize=9)
a2.set_title("Orientation (bars: sd over frames)", color=INK, fontsize=10, loc="left")
handles = [
    Line2D([], [], color=SERIES[g], lw=1.4, label=f"N = {k[0]}, {k[1]}")
    for g, k in enumerate(keys)
]
handles += [
    Line2D(
        [],
        [],
        color=INK2,
        marker="o",
        ls="-",
        label="large-scale tilt |θ| (ridge, r = 0.3–0.4 L)",
    ),
    Line2D(
        [],
        [],
        color=INK2,
        marker="s",
        ls="--",
        label="structure axis |θ| (central lobe)",
    ),
]
a2.legend(
    handles=handles, frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper right"
)

nf = [r["n_frames"] for r in runs]
models = ", ".join(sorted({r["model"] for r in runs}))
rates = ", ".join(f"{v:g}" for v in sorted({r["eps_dot"] for r in runs}))
fig.text(
    0.01,
    0.005,
    f"{models}, ε̇ = {rates}; {runs[0]['field']}, second half of each run ({min(nf)}–{max(nf)} frames per run); "
    f"{len(runs)} runs, {len(bad)} unsaturated run{'' if len(bad) == 1 else 's'} left out. Small markers: individual runs; hollow: "
    f"√(D/ε̇) < {args.min_dx:g} grid spacings (not fitted). Angle signs differ between runs and flip within "
    f"each KR period, so magnitudes are shown.",
    fontsize=6.5,
    color=MUTED,
    wrap=True,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(args.out, facecolor=SURFACE)
print("wrote", args.out)
for t in fit_text:
    print(" ", t)
